import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough


def symlog(x):
    return torch.sign(x)*torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x)*torch.expm1(torch.abs(x))


def twoHotBins(numBins=255, limit=20.0):
    """Exponentially spaced bins, exactly symmetric around 0 (symexp of a linear grid in [-limit, limit])."""
    half = symexp(torch.linspace(-limit, 0.0, (numBins - 1)//2 + 1))
    return torch.cat([half, -half[:-1].flip(0)])


class TwoHot:
    """Softmax over exponentially spaced bins, trained towards twohot encoded targets (symexp twohot loss)."""
    def __init__(self, logits, bins):
        self.logits = logits.float()
        self.bins   = bins

    @property
    def mean(self):
        # Sum negative and positive bins pairwise from small to large magnitude, so that uniform
        # probabilities give exactly 0 (the paper notes the naive sum drifts with bins this large).
        probs   = self.logits.softmax(-1)
        middle  = (len(self.bins) - 1)//2
        negative, positive = (probs[..., :middle]*self.bins[:middle]).flip(-1), probs[..., middle + 1:]*self.bins[middle + 1:]
        return (negative + positive).sum(-1) + probs[..., middle]*self.bins[middle]

    def log_prob(self, target):
        return (twoHotEncode(target.detach().float(), self.bins)*self.logits.log_softmax(-1)).sum(-1)


def twoHotEncode(target, bins):
    """Weights on the two bins around each target, linear in the distance, so that sum(weights*bins) == target
    (targets outside the bin range go fully onto the outermost bin)."""
    below   = ((bins <= target[..., None]).sum(-1) - 1).clamp(0, len(bins) - 1)
    above   = (len(bins) - (bins > target[..., None]).sum(-1)).clamp(0, len(bins) - 1)
    equal   = below == above
    distanceBelow   = torch.where(equal, 1, torch.abs(bins[below] - target))
    distanceAbove   = torch.where(equal, 1, torch.abs(bins[above] - target))
    total           = distanceBelow + distanceAbove
    return (F.one_hot(below, len(bins))*(distanceAbove/total)[..., None] +
            F.one_hot(above, len(bins))*(distanceBelow/total)[..., None])


def uniformMixLogits(rawLogits, uniformMix):
    """Categorical logits that are (1 - uniformMix) network output and uniformMix uniform."""
    probabilities = rawLogits.softmax(-1)
    probabilities = (1 - uniformMix)*probabilities + uniformMix/rawLogits.shape[-1]
    return torch.log(probabilities)


def sampleLatent(logits):
    """Straight-through sample of a vector of categoricals, logits (..., latentLength, latentClasses)."""
    return OneHotCategoricalStraightThrough(logits=logits).rsample()


def categoricalKL(logitsP, logitsQ):
    """KL(P || Q) summed over the latents; logits are normalized log-probabilities."""
    return (logitsP.exp()*(logitsP - logitsQ)).sum(-1).sum(-1)


def boundedNormal(rawOutput, minStd, maxStd):
    """DreamerV3's continuous action distribution: mean squashed by tanh, std in [minStd, maxStd]."""
    mean, std = rawOutput.chunk(2, dim=-1)
    std = (maxStd - minStd)*torch.sigmoid(std + 2.0) + minStd
    return Independent(Normal(torch.tanh(mean), std), 1)
