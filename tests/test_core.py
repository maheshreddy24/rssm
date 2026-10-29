"""Checks for the DreamerV3 building blocks. Run: python -m pytest tests (or python tests/test_core.py)."""
import os
import sys

import attridict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from buffer import ReplayBuffer                                                     # noqa: E402
from distributions import TwoHot, twoHotBins, twoHotEncode, symlog, symexp         # noqa: E402
from dreamer import Dreamer                                                         # noqa: E402
from optim import LaProp                                                            # noqa: E402
from utils import lambdaReturns, replayLambdaReturns, loadConfig                    # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_symlog_inverse():
    x = torch.tensor([-1e6, -3.0, -1e-3, 0.0, 1e-3, 3.0, 1e6])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5)


def test_twohot_roundtrip_and_zero_mean():
    bins = twoHotBins(255)
    assert torch.equal(bins, -bins.flip(0)) and bins[127] == 0
    targets = torch.tensor([-5000.0, -1.0, -0.013, 0.0, 0.02, 0.5, 1.0, 37.0, 1e4])
    assert torch.allclose((twoHotEncode(targets, bins)*bins).sum(-1), targets, rtol=1e-4, atol=1e-6)
    assert TwoHot(torch.zeros(3, 255), bins).mean.abs().max() == 0   # zero-initialized heads predict exactly 0


def test_lambda_returns():
    # lambda = 1, continues = 1: R_0 = r_1 + g r_2 + g^2 v_2
    rewards, values, continues = torch.tensor([[0.0, 1.0, 2.0]]), torch.tensor([[0.0, 0.0, 10.0]]), torch.ones(1, 3)
    returns = lambdaReturns(rewards, values, continues, 0.9, 1.0)
    assert torch.allclose(returns, torch.tensor([[1 + 0.9*2 + 0.81*10, 2 + 0.9*10]]))
    # terminal at s_1: nothing after r_1 counts
    returns = lambdaReturns(rewards, values, torch.tensor([[1.0, 0.0, 1.0]]), 0.9, 0.95)
    assert torch.allclose(returns[0, 0], torch.tensor(1.0))


def test_replay_returns_respect_episode_boundaries():
    rewards     = torch.tensor([[0.0, 1.0, 0.0, 5.0]])
    isTerminal  = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    isFirst     = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    bootstrap   = torch.tensor([[7.0, 7.0, 7.0, 7.0]])
    returns, mask = replayLambdaReturns(rewards, isTerminal, isFirst, bootstrap, 0.9, 0.95)
    assert torch.allclose(returns[0, 0], torch.tensor(1.0))          # terminal next step: just its reward
    assert mask.tolist() == [[1.0, 0.0, 1.0]]                         # step 1 -> 2 crosses into a new episode


def test_buffer_wraps_and_stays_contiguous():
    buffer = ReplayBuffer({"state": (1,)}, 1, numEnvs=2, capacity=20, device="cpu")
    for step in range(25):
        buffer.add({"state": np.full((2, 1), step, np.float32), "reward": np.zeros(2), "isFirst": np.zeros(2, bool),
                    "isTerminal": np.zeros(2, bool)}, np.zeros((2, 1)))
    sample = buffer.sample(64, 5)["state"][..., 0]
    assert torch.all(sample[:, 1:] - sample[:, :-1] == 1) and sample.min() >= 15   # only the newest 10 steps, in order


def test_laprop_minimizes():
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    optimizer = LaProp([x], lr=0.05, agc=0.0)
    for _ in range(300):
        optimizer.zero_grad()
        (x**2).sum().backward()
        optimizer.step()
    assert x.abs().max() < 0.1


def tinyDreamer(observationShapes):
    config = loadConfig(os.path.join(ROOT, "configs", "catching.yml")).dreamer
    config.batchSize, config.batchLength, config.imaginationHorizon = 3, 8, 4
    config.model |= {"hiddenSize": 32, "recurrentSize": 64, "latentLength": 4, "latentClasses": 4, "cnnDepth": 4}
    return Dreamer(observationShapes, 3, torch.device("cpu"), attridict(config))


def fakeBatch(batchSize, batchLength, imageSize):
    isFirst = torch.zeros(batchSize, batchLength, dtype=torch.bool)
    isFirst[:, 0] = isFirst[:, 5] = True
    isTerminal = torch.zeros(batchSize, batchLength, dtype=torch.bool)
    isTerminal[:, 4] = True
    return {"image": torch.randint(0, 255, (batchSize, batchLength, imageSize, imageSize, 3), dtype=torch.uint8),
            "state": torch.randn(batchSize, batchLength, 22), "action": torch.rand(batchSize, batchLength, 3)*2 - 1,
            "reward": torch.rand(batchSize, batchLength), "isFirst": isFirst, "isTerminal": isTerminal}


def test_observe_resets_at_episode_start():
    dreamer = tinyDreamer({"image": (64, 64, 3), "state": (22,)})
    data    = fakeBatch(2, 8, 64)
    data["image"][1, :5], data["state"][1, :5], data["action"][1, :5] = 0, 0, 0   # different history before the reset at t=5
    with torch.no_grad():
        recurrentStates, *_ = dreamer.observe(dreamer.encoder({k: data[k] for k in ("image", "state")}), data["action"], data["isFirst"])
    assert torch.allclose(recurrentStates[0, 5], recurrentStates[1, 5], atol=1e-6)
    assert not torch.allclose(recurrentStates[0, 4], recurrentStates[1, 4])


def test_training_step_runs():
    for imageSize in (64, 96):
        dreamer = tinyDreamer({"image": (imageSize, imageSize, 3), "state": (22,)})
        data    = fakeBatch(3, 8, imageSize)
        fullStates, worldModelMetrics = dreamer.worldModelTraining(data)
        behaviorMetrics = dreamer.behaviorTraining(fullStates, data)
        assert all(np.isfinite(value) for value in (worldModelMetrics | behaviorMetrics).values())
        frames = dreamer.openLoopPrediction(data, contextLength=3, numSequences=2)
        assert frames.shape == (8, 3*imageSize, 2*imageSize, 3)
        actions, _ = dreamer.act({"image": data["image"][:, 0].numpy(), "state": data["state"][:, 0].numpy(),
                                  "isFirst": np.ones(3, bool)}, dreamer.initialState(3))
        assert actions.shape == (3, 3) and np.abs(actions).max() <= 1


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"ok  {name}")
