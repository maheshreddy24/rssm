import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import os
import attridict
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    if not os.path.exists(config_path):
        config_path = findFile(os.path.basename(config_path))
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def applyOverrides(config, overrides):
    """Overrides like "dreamer.batchSize=8" or "environment.reward.action=0.001"; values are parsed as YAML."""
    for override in overrides:
        path, value = override.split("=", 1)
        *parents, key = path.split(".")
        node = config
        for parent in parents:
            node = node[parent]
        if key not in node:
            raise KeyError(f"Unknown config key '{path}'")
        node[key] = yaml.safe_load(value)
    return config


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10, xAxis="envSteps"):
    if not filename.endswith(".csv"):
        filename += ".csv"

    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue

        fig.add_trace(pgo.Scatter(
            x=data[xAxis], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))

        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data[xAxis], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2),
            visible=True if column in ("success", "episodeReturn") else 'legendonly'))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Environment Steps" if xAxis == "envSteps" else "Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def lambdaReturns(rewards, values, continues, discount, lambda_):
    """Imagined lambda returns. rewards, values, continues: (N, H+1), index t predicted from state s_t.
    R_t = r_{t+1} + discount*c_{t+1}*((1 - lambda)*v_{t+1} + lambda*R_{t+1}), R_H = v_H. Returns R_0..R_{H-1}: (N, H)."""
    returns = [values[:, -1]]
    for t in reversed(range(rewards.shape[1] - 1)):
        returns.append(rewards[:, t + 1] + discount*continues[:, t + 1]*((1 - lambda_)*values[:, t + 1] + lambda_*returns[-1]))
    return torch.stack(returns[::-1][:-1], 1)


def replayLambdaReturns(rewards, isTerminal, isFirst, bootstrap, discount, lambda_):
    """Lambda returns over replayed sequences (batch, time), using the real rewards and terminals and the imagined
    returns of each step as bootstrap values. Returns (batch, time-1) targets and a mask that drops steps whose
    next step already belongs to a new episode."""
    returns, masks = [bootstrap[:, -1]], []
    for t in reversed(range(rewards.shape[1] - 1)):
        sameEpisode = 1.0 - isFirst[:, t + 1]
        target = rewards[:, t + 1] + discount*(1.0 - isTerminal[:, t + 1])*((1 - lambda_)*bootstrap[:, t + 1] + lambda_*returns[-1])
        returns.append(sameEpisode*target + (1.0 - sameEpisode)*bootstrap[:, t])
        masks.append(sameEpisode)
    return torch.stack(returns[::-1][:-1], 1), torch.stack(masks[::-1], 1)


def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)


class Moments(nn.Module):
    """Return normalization: EMA of the 5th and 95th percentile of the returns, scale = max(limit, high - low)."""
    def __init__(self, device, decay=0.99, limit=1.0, percentileLow=0.05, percentileHigh=0.95):
        super().__init__()
        self.decay, self.limit = decay, limit
        self.register_buffer("percentiles", torch.tensor([percentileLow, percentileHigh], device=device))
        self.register_buffer("low", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32, device=device))

    def forward(self, x):
        low, high = torch.quantile(x.detach().flatten().float(), self.percentiles)
        self.low.lerp_(low, 1 - self.decay)
        self.high.lerp_(high, 1 - self.decay)
        return self.low.clone(), torch.clamp(self.high - self.low, min=self.limit)
