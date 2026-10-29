import contextlib
import copy
import os
import numpy as np
import torch
import torch.nn as nn

from networks import RecurrentModel, LatentNet, Encoder, Decoder, TwoHotHead, ContinueModel, Actor
from distributions import categoricalKL, symlog
from optim import LaProp
from utils import Moments, lambdaReturns, replayLambdaReturns


class Dreamer:
    def __init__(self, observationShapes, actionSize, device, config):
        self.observationShapes  = observationShapes
        self.actionSize         = actionSize
        self.config             = config
        self.device             = device
        model                   = config.model

        self.recurrentSize  = model.recurrentSize
        self.latentSize     = model.latentLength*model.latentClasses
        self.fullStateSize  = model.recurrentSize + self.latentSize

        self.encoder            = Encoder(observationShapes,                                            model).to(device)
        self.decoder            = Decoder(self.fullStateSize, observationShapes,                        model).to(device)
        self.recurrentModel     = RecurrentModel(self.recurrentSize, self.latentSize, actionSize,       model).to(device)
        self.priorNet           = LatentNet(self.recurrentSize, model.priorLayers,                      model).to(device)
        self.posteriorNet       = LatentNet(self.recurrentSize + self.encoder.outputSize, model.posteriorLayers, model).to(device)
        self.rewardPredictor    = TwoHotHead(self.fullStateSize, model.rewardLayers,                    model).to(device)
        self.continuePredictor  = ContinueModel(self.fullStateSize,                                     model).to(device)
        self.actor              = Actor(self.fullStateSize, actionSize,                                 model).to(device)
        self.critic             = TwoHotHead(self.fullStateSize, model.criticLayers,                    model).to(device)
        self.slowCritic         = copy.deepcopy(self.critic).requires_grad_(False)

        self.worldModelParameters = [parameter for network in (self.encoder, self.decoder, self.recurrentModel, self.priorNet, self.posteriorNet,
                                                               self.rewardPredictor, self.continuePredictor) for parameter in network.parameters()]
        optimizer = lambda parameters: LaProp(parameters, lr=config.learningRate, eps=config.optimizerEpsilon, agc=config.adaptiveGradientClip)
        self.worldModelOptimizer    = optimizer(self.worldModelParameters)
        self.actorOptimizer         = optimizer(self.actor.parameters())
        self.criticOptimizer        = optimizer(self.critic.parameters())
        self.returnMoments          = Moments(device, config.returnNormDecay, config.returnNormLimit)
        self.mixedPrecision         = config.mixedPrecision and (device.type != "cuda" or torch.cuda.is_bf16_supported())

        self.totalEpisodes      = 0
        self.totalEnvSteps      = 0
        self.totalGradientSteps = 0


    def precision(self):
        """bfloat16 compute for the networks (as in the reference DreamerV3); distributions and losses stay float32."""
        return torch.autocast(self.device.type, dtype=torch.bfloat16) if self.mixedPrecision else contextlib.nullcontext()


    def initialState(self, batchSize):
        return (torch.zeros(batchSize, self.recurrentSize,  device=self.device),
                torch.zeros(batchSize, self.latentSize,     device=self.device),
                torch.zeros(batchSize, self.actionSize,     device=self.device))


    def observe(self, encodedObservations, actions, isFirst, state=None):
        """Posterior rollout over (batch, time). The state (and previous action) is reset to zero wherever an episode starts."""
        recurrentState, latentState, previousAction = state or self.initialState(actions.shape[0])
        recurrentStates, posteriors, priorsLogits, posteriorsLogits = [], [], [], []
        for t in range(actions.shape[1]):
            keep = 1.0 - isFirst[:, t, None].float()
            recurrentState, latentState, previousAction = recurrentState*keep, latentState*keep, previousAction*keep

            recurrentState              = self.recurrentModel(recurrentState, latentState, previousAction)
            _, priorLogits              = self.priorNet(recurrentState)
            latentState, posteriorLogits = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))
            previousAction              = actions[:, t]

            recurrentStates.append(recurrentState)
            posteriors.append(latentState)
            priorsLogits.append(priorLogits)
            posteriorsLogits.append(posteriorLogits)
        return (torch.stack(recurrentStates, 1), torch.stack(posteriors, 1), torch.stack(priorsLogits, 1), torch.stack(posteriorsLogits, 1),
                (recurrentState, latentState, previousAction))


    def worldModelTraining(self, data):
        with self.precision():
            observations = {key: data[key] for key in self.observationShapes}
            encodedObservations = self.encoder(observations)
            recurrentStates, posteriors, priorsLogits, posteriorsLogits, _ = self.observe(encodedObservations, data["action"], data["isFirst"])
            fullStates = torch.cat((recurrentStates, posteriors), -1)                                      # (batchSize, batchLength, fullStateSize)

            reconstructions = self.decoder(fullStates)
            losses = {}
            if "image" in reconstructions:
                losses["image"] = (reconstructions["image"].float() - data["image"].float()/255.0).pow(2).sum((-3, -2, -1))
            if "state" in reconstructions:
                losses["state"] = (reconstructions["state"].float() - symlog(data["state"])).pow(2).sum(-1)
            losses["reward"]    = -self.rewardPredictor(fullStates).log_prob(data["reward"])
            losses["continue"]  = -self.continuePredictor(fullStates).log_prob(1.0 - data["isTerminal"].float())

        # KL balancing with free bits: the dynamics loss trains the prior towards the (fixed) posterior,
        # the smaller representation loss makes the posterior more predictable. Both are clipped below freeNats.
        dynamicsKL          = categoricalKL(posteriorsLogits.detach(), priorsLogits)
        representationKL    = categoricalKL(posteriorsLogits, priorsLogits.detach())
        losses["dynamics"]          = self.config.betaDynamics*torch.clamp(dynamicsKL, min=self.config.freeNats)
        losses["representation"]    = self.config.betaRepresentation*torch.clamp(representationKL, min=self.config.freeNats)

        worldModelLoss = sum(loss.mean() for loss in losses.values())
        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        self.worldModelOptimizer.step()

        metrics = {f"{key}Loss": loss.mean().item() for key, loss in losses.items()}
        metrics |= {"worldModelLoss": worldModelLoss.item(), "klDivergence": dynamicsKL.mean().item(),
                    "posteriorEntropy": -(posteriorsLogits.exp()*posteriorsLogits).sum(-1).mean().item()}
        return fullStates.detach().float(), metrics


    def behaviorTraining(self, replayFullStates, data):
        config = self.config
        batchSize, batchLength = replayFullStates.shape[:2]

        # Imagine imaginationHorizon steps from every posterior state of the batch: states s_0..s_H, actions a_0..a_{H-1}.
        with torch.no_grad(), self.precision():
            fullState = replayFullStates.reshape(-1, self.fullStateSize)
            recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
            fullStates, actions = [fullState], []
            for _ in range(config.imaginationHorizon):
                action          = self.actor(fullState).sample().clamp(-1, 1)
                recurrentState  = self.recurrentModel(recurrentState, latentState, action)
                latentState, _  = self.priorNet(recurrentState)
                fullState       = torch.cat((recurrentState, latentState), -1).float()
                fullStates.append(fullState)
                actions.append(action)
            fullStates  = torch.stack(fullStates, 1)                                                        # (N, H+1, fullStateSize)
            actions     = torch.stack(actions, 1)                                                           # (N, H, actionSize)

            rewards     = self.rewardPredictor(fullStates).mean                                             # r_t predicted from s_t
            continues   = self.continuePredictor(fullStates).mean
            continues[:, 0] = 1.0 - data["isTerminal"].reshape(-1).float()                                  # the start state is real
            values      = self.critic(fullStates).mean
            returns     = lambdaReturns(rewards, values, continues, config.discount, config.lambda_)        # (N, H), R_t for s_0..s_{H-1}
            weights     = torch.cumprod(config.discount*continues, 1)[:, :-1]/config.discount               # probability of still being alive
            _, scale    = self.returnMoments(returns)
            advantages  = (returns - values[:, :-1])/scale

        with self.precision():
            policy  = self.actor(fullStates[:, :-1])
        logprobs    = policy.log_prob(actions)
        entropies   = policy.entropy()
        actorLoss   = -torch.mean(weights*(logprobs*advantages + config.entropyScale*entropies))

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        self.actorOptimizer.step()

        # Critic: imagined lambda returns, plus lambda returns over the real replay rewards bootstrapped with the imagined
        # returns of each start state. Both are regularized towards the slow (EMA) critic's predictions.
        with self.precision():
            valueDistributions  = self.critic(fullStates[:, :-1])
            replayDistributions = self.critic(replayFullStates[:, :-1])
            with torch.no_grad():
                slowValues          = self.slowCritic(fullStates[:, :-1]).mean
                slowReplayValues    = self.slowCritic(replayFullStates[:, :-1]).mean
        imaginationCriticLoss   = -weights*(valueDistributions.log_prob(returns) + config.criticEMARegularizer*valueDistributions.log_prob(slowValues))

        replayReturns, replayMask   = replayLambdaReturns(data["reward"], data["isTerminal"].float(), data["isFirst"].float(),
                                                          returns[:, 0].view(batchSize, batchLength), config.discount, config.lambda_)
        replayCriticLoss            = -replayMask*(replayDistributions.log_prob(replayReturns) + config.criticEMARegularizer*replayDistributions.log_prob(slowReplayValues))

        criticLoss = imaginationCriticLoss.mean() + config.criticReplayScale*replayCriticLoss.mean()
        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        with torch.no_grad():
            for slowParameter, parameter in zip(self.slowCritic.parameters(), self.critic.parameters()):
                slowParameter.lerp_(parameter, 1.0 - config.criticEMADecay)

        metrics = {
            "actorLoss"         : actorLoss.item(),
            "criticLoss"        : criticLoss.item(),
            "entropy"           : entropies.mean().item(),
            "actorStd"          : policy.base_dist.scale.mean().item(),
            "advantages"        : advantages.mean().item(),
            "imaginedReturns"   : returns.mean().item(),
            "imaginedRewards"   : rewards[:, 1:].mean().item(),
            "criticValues"      : values.mean().item(),
            "returnScale"       : scale.item()}
        return metrics


    @torch.no_grad()
    def act(self, observations, state, evaluation=False):
        """One policy step for a batch of environments. observations: dict of numpy arrays with a leading env axis."""
        recurrentState, latentState, previousAction = state
        keep = 1.0 - torch.as_tensor(observations["isFirst"], device=self.device).float()[:, None]
        recurrentState, latentState, previousAction = recurrentState*keep, latentState*keep, previousAction*keep

        with self.precision():
            encodedObservation = self.encoder({key: torch.as_tensor(observations[key], device=self.device) for key in self.observationShapes})
            recurrentState  = self.recurrentModel(recurrentState, latentState, previousAction)
            latentState, _  = self.posteriorNet(torch.cat((recurrentState, encodedObservation), -1))
            policy          = self.actor(torch.cat((recurrentState, latentState), -1))
        action          = (policy.mean if evaluation else policy.sample()).clamp(-1, 1)
        return action.cpu().numpy(), (recurrentState, latentState, action)


    @torch.no_grad()
    def openLoopPrediction(self, data, contextLength=5, numSequences=4):
        """Video frames (T, 3*H, numSequences*W, 3): ground truth, prediction, error. The model observes the first
        contextLength frames, then imagines the rest from the recorded actions using only its prior."""
        with self.precision():
            return self._openLoopPrediction(data, contextLength, numSequences)

    def _openLoopPrediction(self, data, contextLength, numSequences):
        data = {key: value[:numSequences] for key, value in data.items()}
        encoded = self.encoder({key: data[key] for key in self.observationShapes})
        recurrentStates, posteriors, _, _, state = self.observe(encoded[:, :contextLength], data["action"][:, :contextLength], data["isFirst"][:, :contextLength])
        recurrentState, latentState, previousAction = state
        fullStates = [torch.cat((recurrentStates, posteriors), -1)]
        for t in range(contextLength, data["action"].shape[1]):
            recurrentState  = self.recurrentModel(recurrentState, latentState, previousAction)
            latentState, _  = self.priorNet(recurrentState)
            previousAction  = data["action"][:, t]
            fullStates.append(torch.cat((recurrentState, latentState), -1)[:, None])
        prediction  = self.decoder(torch.cat(fullStates, 1))["image"].float()
        truth       = data["image"].float()/255.0
        error       = (prediction - truth).abs()
        frames      = torch.cat((truth, prediction, error), 2)                                              # (numSequences, T, 3H, W, 3)
        frames      = torch.cat(list(frames), 2)                                                            # (T, 3H, numSequences*W, 3)
        return (frames.clamp(0, 1)*255).byte().cpu().numpy()


    def networks(self):
        return {"encoder": self.encoder, "decoder": self.decoder, "recurrentModel": self.recurrentModel, "priorNet": self.priorNet,
                "posteriorNet": self.posteriorNet, "rewardPredictor": self.rewardPredictor, "continuePredictor": self.continuePredictor,
                "actor": self.actor, "critic": self.critic, "slowCritic": self.slowCritic, "returnMoments": self.returnMoments,
                "worldModelOptimizer": self.worldModelOptimizer, "actorOptimizer": self.actorOptimizer, "criticOptimizer": self.criticOptimizer}


    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        checkpoint = {name: network.state_dict() for name, network in self.networks().items()}
        checkpoint |= {'totalEpisodes': self.totalEpisodes, 'totalEnvSteps': self.totalEnvSteps, 'totalGradientSteps': self.totalGradientSteps}
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        for name, network in self.networks().items():
            network.load_state_dict(checkpoint[name])
        self.totalEpisodes      = checkpoint['totalEpisodes']
        self.totalEnvSteps      = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
