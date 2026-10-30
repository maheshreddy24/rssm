import argparse
import os
import time
from collections import defaultdict

import imageio
import numpy as np
import torch
from tqdm import tqdm

from buffer     import ReplayBuffer
from dreamer    import Dreamer
from envs       import makeEnvironment, environmentFactories
from envs.parallel import ParallelEnvs
from utils      import loadConfig, applyOverrides, seedEverything, plotMetrics, saveLossesToCSV, ensureParentFolders, setupLogger


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
    logFilename             = f"{checkpointFilenameBase}_logger.txt"
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)

    logger = setupLogger(logFilename)
    logger.info(f"Run {runName}, config {configFile}" + (f", overrides {' '.join(overrides)}" if overrides else ""))
    wandbRun = None
    if config.wandb.mode != "disabled":
        import wandb
        wandbRun = wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=runName, mode=config.wandb.mode, config=config)
        logger.info(f"wandb: {wandbRun.url or 'offline run in ' + wandbRun.dir}")

    envs            = ParallelEnvs(environmentFactories(config.environment, config.numEnvs, config.seed), parallel=config.numEnvs > 1)
    envEvaluation   = makeEnvironment(config.environment)
    logger.info(f"Observations {envs.observationShapes}, action size {envs.actionSize}, {config.numEnvs} envs, device {device}")

    dreamer = Dreamer(envs.observationShapes, envs.actionSize, device, config.dreamer)
    buffer  = ReplayBuffer(envs.observationShapes, envs.actionSize, config.numEnvs, config.dreamer.bufferCapacity, device)
    bufferGB, machineGB = buffer.nbytes()/1e9, os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1e9
    logger.info(f"Replay buffer fills up to {bufferGB:.1f} GB of RAM (machine has {machineGB:.0f} GB); env workers need ~0.35 GB each")
    if bufferGB + 0.35*config.numEnvs + 3 > machineGB:
        logger.warning("WARNING: that will not fit in RAM and the run will crash once the buffer fills; lower dreamer.bufferCapacity")
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)
        logger.info(f"Resumed from {checkpointToLoad} at {dreamer.totalEnvSteps} env steps")
    logger.info(f"Parameters: world model {sum(p.numel() for p in dreamer.worldModelParameters)/1e6:.1f}M, "
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
    progressBar         = tqdm(total=config.envSteps, initial=dreamer.totalEnvSteps, desc="prefill", unit="step", unit_scale=True, dynamic_ncols=True, smoothing=0.05)

    try:
        while dreamer.totalEnvSteps < config.envSteps:
            actions, agentState = dreamer.act(observations, agentState)
            buffer.add(observations, actions)
            observations, infos = envs.step(actions)
            dreamer.totalEnvSteps += config.numEnvs
            progressBar.update(config.numEnvs)

            episodeReturns = np.where(observations["isFirst"], 0.0, episodeReturns + observations["reward"])
            episodeLengths = np.where(observations["isFirst"], 0, episodeLengths + 1)
            for i in np.nonzero(observations["isLast"])[0]:
                for key, value in {"episodeReturn": episodeReturns[i], "episodeLength": episodeLengths[i], **infos[i]}.items():
                    episodeStatistics[key].append(value)
                dreamer.totalEpisodes += 1

            if buffer.size >= batchLength and len(buffer) >= config.prefillSteps:
                progressBar.set_description("train", refresh=False)
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
                elapsed     = time.time() - lastLogTime
                rates       = {"envStepsPerSecond": (dreamer.totalEnvSteps - lastLogEnvSteps)/elapsed,
                               "gradientStepsPerSecond": (dreamer.totalGradientSteps - lastLogGradientSteps)/elapsed}
                episodes    = {key: float(np.mean(values)) for key, values in episodeStatistics.items()}
                training    = {key: float(np.mean(values)) for key, values in trainingMetrics.items()}
                row         = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "episodes": dreamer.totalEpisodes}
                row        |= episodes | training | rates
                logger.info(logLine(row))
                progressBar.set_postfix_str(f"return {row.get('episodeReturn', float('nan')):.3f}, success {row.get('success', float('nan')):.2f}, "
                                            f"{dreamer.totalGradientSteps} grad steps", refresh=False)
                if config.saveMetrics:
                    saveLossesToCSV(metricsFilename, row)
                    plotMetrics(metricsFilename, savePath=plotFilename, title=runName)
                if wandbRun:
                    wandbRun.log({f"episode/{key}": value for key, value in episodes.items()} | {f"train/{key}": value for key, value in training.items()}
                                 | {f"perf/{key}": value for key, value in rates.items()}
                                 | {"progress/gradientSteps": dreamer.totalGradientSteps, "progress/episodes": dreamer.totalEpisodes}, step=dreamer.totalEnvSteps)
                episodeStatistics.clear()
                trainingMetrics.clear()
                nextLog += config.logInterval
                lastLogTime, lastLogEnvSteps, lastLogGradientSteps = time.time(), dreamer.totalEnvSteps, dreamer.totalGradientSteps

            if dreamer.totalEnvSteps >= nextEvaluation:
                description = progressBar.desc
                progressBar.set_description("eval")
                suffix = f"{dreamer.totalEnvSteps/1000:.0f}k"
                evaluation, videoFilename = evaluate(dreamer, envEvaluation, config.numEvaluationEpisodes, f"{videoFilenameBase}_{suffix}")
                openLoopFilename = None
                if buffer.size >= batchLength and "image" in envs.observationShapes:
                    openLoopFilename = f"{videoFilenameBase}_{suffix}_openLoop.mp4"
                    writeVideo(openLoopFilename, dreamer.openLoopPrediction(buffer.sample(4, batchLength)), fps=10)
                if config.saveCheckpoints:
                    dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                if config.saveMetrics:
                    saveLossesToCSV(f"{metricsFilename}_evaluation", {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps} | evaluation)
                    plotMetrics(f"{metricsFilename}_evaluation", savePath=f"{plotFilename}_evaluation", title=f"{runName} evaluation")
                if wandbRun:
                    videos = {"eval/video": wandb.Video(videoFilename, format="mp4")}
                    if openLoopFilename:
                        videos["eval/openLoop"] = wandb.Video(openLoopFilename, format="mp4")
                    wandbRun.log({f"eval/{key}": value for key, value in evaluation.items()} | videos, step=dreamer.totalEnvSteps)
                logger.info(f"Evaluation at {suffix} env steps: " + ", ".join(f"{key} {value:.3f}" for key, value in evaluation.items()))
                progressBar.set_description(description)
                nextEvaluation += config.evaluationInterval
        logger.info(f"Finished {dreamer.totalEnvSteps} env steps, {dreamer.totalGradientSteps} gradient steps, {dreamer.totalEpisodes} episodes")
    finally:
        progressBar.close()
        envs.close()
        if wandbRun:
            wandbRun.finish()


def logLine(row):
    """One line per log interval: progress, episode results, the main losses and throughput."""
    get = lambda key: row.get(key, float("nan"))
    return (f"{row['envSteps']:>9} env | {row['gradientSteps']:>7} grad | return {get('episodeReturn'):7.3f} | success {get('success'):4.2f}"
            f" | wm {get('worldModelLoss'):8.2f} | actor {get('actorLoss'):7.3f} | critic {get('criticLoss'):7.3f}"
            f" | {get('envStepsPerSecond'):6.1f} env/s, {get('gradientStepsPerSecond'):5.2f} grad/s")


def evaluate(dreamer, env, numEpisodes, videoFilename):
    """Runs the actor's mean action. Episodes use fixed seeds, so evaluations are comparable over training.
    Returns the mean statistics and the path of the video of the first episode."""
    statistics, frames = defaultdict(list), []
    for episode in tqdm(range(numEpisodes), desc="eval episodes", leave=False, dynamic_ncols=True):
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
    videoFilename = f"{videoFilename}_return_{statistics['episodeReturn']:.2f}.mp4"
    writeVideo(videoFilename, frames)
    return statistics, videoFilename


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
