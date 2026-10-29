import argparse
import os
import time
from collections import defaultdict

import imageio
import numpy as np
import torch

from buffer     import ReplayBuffer
from dreamer    import Dreamer
from envs       import makeEnvironment, environmentFactories
from envs.parallel import ParallelEnvs
from utils      import loadConfig, applyOverrides, seedEverything, plotMetrics, saveLossesToCSV, ensureParentFolders


def main(configFile, overrides=()):
    config = applyOverrides(loadConfig(configFile), overrides)
    seedEverything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    runName                 = f"{config.environment.name}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)

    envs            = ParallelEnvs(environmentFactories(config.environment, config.numEnvs, config.seed), parallel=config.numEnvs > 1)
    envEvaluation   = makeEnvironment(config.environment)
    print(f"Observations {envs.observationShapes}, action size {envs.actionSize}, {config.numEnvs} envs, device {device}")

    dreamer = Dreamer(envs.observationShapes, envs.actionSize, device, config.dreamer)
    buffer  = ReplayBuffer(envs.observationShapes, envs.actionSize, config.numEnvs, config.dreamer.bufferCapacity, device)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)
    print(f"Parameters: world model {sum(p.numel() for p in dreamer.worldModelParameters)/1e6:.1f}M, "
          f"actor {sum(p.numel() for p in dreamer.actor.parameters())/1e6:.1f}M, critic {sum(p.numel() for p in dreamer.critic.parameters())/1e6:.1f}M")

    batchSize, batchLength      = config.dreamer.batchSize, config.dreamer.batchLength
    gradientStepsPerEnvStep     = config.replayRatio/(batchSize*batchLength)    # replay ratio = trained steps per collected step
    trainingDebt                = 0.0
    nextLog, nextEvaluation     = dreamer.totalEnvSteps + config.logInterval, dreamer.totalEnvSteps + config.evaluationInterval

    observations, _     = envs.reset()
    agentState          = dreamer.initialState(config.numEnvs)
    episodeReturns      = np.zeros(config.numEnvs)
    episodeLengths      = np.zeros(config.numEnvs, dtype=int)
    episodeStatistics   = defaultdict(list)
    trainingMetrics     = defaultdict(list)
    lastLogTime, lastLogEnvSteps, lastLogGradientSteps = time.time(), dreamer.totalEnvSteps, dreamer.totalGradientSteps

    while dreamer.totalEnvSteps < config.envSteps:
        actions, agentState = dreamer.act(observations, agentState)
        buffer.add(observations, actions)
        observations, infos = envs.step(actions)
        dreamer.totalEnvSteps += config.numEnvs

        episodeReturns = np.where(observations["isFirst"], 0.0, episodeReturns + observations["reward"])
        episodeLengths = np.where(observations["isFirst"], 0, episodeLengths + 1)
        for i in np.nonzero(observations["isLast"])[0]:
            for key, value in {"episodeReturn": episodeReturns[i], "episodeLength": episodeLengths[i], **infos[i]}.items():
                episodeStatistics[key].append(value)
            dreamer.totalEpisodes += 1

        if buffer.size >= batchLength and len(buffer) >= config.prefillSteps:
            trainingDebt += config.numEnvs*gradientStepsPerEnvStep
            while trainingDebt >= 1:
                data                            = buffer.sample(batchSize, batchLength)
                fullStates, worldModelMetrics   = dreamer.worldModelTraining(data)
                behaviorMetrics                 = dreamer.behaviorTraining(fullStates, data)
                dreamer.totalGradientSteps     += 1
                trainingDebt                   -= 1
                for key, value in (worldModelMetrics | behaviorMetrics).items():
                    trainingMetrics[key].append(value)

        if dreamer.totalEnvSteps >= nextLog:
            elapsed = time.time() - lastLogTime
            rates   = {"envStepsPerSecond": (dreamer.totalEnvSteps - lastLogEnvSteps)/elapsed,
                       "gradientStepsPerSecond": (dreamer.totalGradientSteps - lastLogGradientSteps)/elapsed}
            row     = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "episodes": dreamer.totalEpisodes}
            row    |= {key: float(np.mean(values)) for key, values in (episodeStatistics | trainingMetrics).items()} | rates
            print(f"{dreamer.totalEnvSteps:>9} env steps | {dreamer.totalGradientSteps:>7} grad steps | return {row.get('episodeReturn', float('nan')):7.3f}"
                  f" | success {row.get('success', float('nan')):5.2f} | {rates['envStepsPerSecond']:6.1f} env/s, {rates['gradientStepsPerSecond']:5.2f} grad/s")
            if config.saveMetrics:
                saveLossesToCSV(metricsFilename, row)
                plotMetrics(metricsFilename, savePath=plotFilename, title=runName)
            episodeStatistics.clear()
            trainingMetrics.clear()
            nextLog += config.logInterval
            lastLogTime, lastLogEnvSteps, lastLogGradientSteps = time.time(), dreamer.totalEnvSteps, dreamer.totalGradientSteps

        if dreamer.totalEnvSteps >= nextEvaluation:
            suffix = f"{dreamer.totalEnvSteps/1000:.0f}k"
            evaluation = evaluate(dreamer, envEvaluation, config.numEvaluationEpisodes, f"{videoFilenameBase}_{suffix}")
            if buffer.size >= batchLength and "image" in envs.observationShapes:
                writeVideo(f"{videoFilenameBase}_{suffix}_openLoop.mp4", dreamer.openLoopPrediction(buffer.sample(4, batchLength)), fps=10)
            if config.saveCheckpoints:
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
            if config.saveMetrics:
                saveLossesToCSV(f"{metricsFilename}_evaluation", {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps} | evaluation)
                plotMetrics(f"{metricsFilename}_evaluation", savePath=f"{plotFilename}_evaluation", title=f"{runName} evaluation")
            print(f"Evaluation at {suffix:>6} env steps: " + ", ".join(f"{key} {value:.3f}" for key, value in evaluation.items()))
            nextEvaluation += config.evaluationInterval
    envs.close()


def evaluate(dreamer, env, numEpisodes, videoFilename):
    """Runs the actor's mean action. Episodes use fixed seeds, so evaluations are comparable over training."""
    statistics, frames = defaultdict(list), []
    for episode in range(numEpisodes):
        observation, _  = env.reset(seed=10_000 + episode)
        state           = dreamer.initialState(1)
        episodeReturn   = 0.0
        while True:
            action, state = dreamer.act({key: np.asarray(value)[None] for key, value in observation.items()}, state, evaluation=True)
            if episode == 0:
                frames.append(videoFrame(env, observation))
            observation, info = env.step(action[0])
            episodeReturn += observation["reward"]
            if observation["isLast"]:
                break
        for key, value in {"episodeReturn": episodeReturn, **info}.items():
            statistics[key].append(value)
    statistics = {key: float(np.mean(values)) for key, values in statistics.items()}
    writeVideo(f"{videoFilename}_return_{statistics['episodeReturn']:.2f}.mp4", frames)
    return statistics


def videoFrame(env, observation):
    """Third-person render next to the (upscaled) image the policy sees."""
    frame = env.render()
    if "image" in observation:
        scale   = max(1, frame.shape[0]//observation["image"].shape[0])
        policy  = np.repeat(np.repeat(observation["image"], scale, 0), scale, 1)
        policy  = np.pad(policy, ((0, frame.shape[0] - policy.shape[0]), (0, 0), (0, 0)))
        frame   = np.concatenate([frame, policy], 1)
    return frame


def writeVideo(filename, frames, fps=30, macroBlockSize=16):
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            targetHeight    = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize # getting rid of imagio warning
            targetWidth     = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
            video.append_data(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="catching.yml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. dreamer.batchSize=8")
    arguments = parser.parse_args()
    main(arguments.config, arguments.set)
