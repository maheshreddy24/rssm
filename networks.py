import math
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from distributions import TwoHot, twoHotBins, uniformMixLogits, sampleLatent, boundedNormal, symlog


def initLinear(layer, scale=1.0):
    """Truncated normal with fan-in scaling; scale 0 gives a zero-initialized output layer."""
    fanIn = layer.weight[0].numel()
    std   = scale*math.sqrt(1.0/fanIn)/0.87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std) if std > 0 else nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """numLayers x (Linear -> RMSNorm -> SiLU), then an optional output Linear."""
    def __init__(self, inputSize, outputSize, hiddenSize, numLayers, outScale=1.0):
        super().__init__()
        layers = []
        for i in range(numLayers):
            layers += [initLinear(nn.Linear(inputSize if i == 0 else hiddenSize, hiddenSize)), nn.RMSNorm(hiddenSize, eps=1e-4), nn.SiLU()]
        if outputSize is not None:
            layers.append(initLinear(nn.Linear(hiddenSize if numLayers else inputSize, outputSize), outScale))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BlockLinear(nn.Module):
    """Linear layer with block-diagonal weights: input (..., blocks, inSize) -> output (..., blocks*outSize)."""
    def __init__(self, blocks, inSize, outSize):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(blocks, inSize, outSize))
        self.bias   = nn.Parameter(torch.zeros(blocks*outSize))
        std = math.sqrt(1.0/inSize)/0.87962566103423978
        nn.init.trunc_normal_(self.weight, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return torch.einsum("...gi,gio->...go", x, self.weight).flatten(-2) + self.bias


class RecurrentModel(nn.Module):
    """Block GRU sequence model h_t = f(h_{t-1}, z_{t-1}, a_{t-1}): the recurrent state is split into blocks
    with block-diagonal recurrent weights; every block also sees embeddings of h, z and a to mix information."""
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.blocks     = config.blocks
        self.blockSize  = recurrentSize//config.blocks
        hiddenSize      = config.hiddenSize
        embedding       = lambda inputSize: nn.Sequential(initLinear(nn.Linear(inputSize, hiddenSize)), nn.RMSNorm(hiddenSize, eps=1e-4), nn.SiLU())
        self.recurrentIn, self.latentIn, self.actionIn = embedding(recurrentSize), embedding(latentSize), embedding(actionSize)
        self.hidden     = BlockLinear(self.blocks, self.blockSize + 3*hiddenSize, self.blockSize)
        self.hiddenNorm = nn.Sequential(nn.RMSNorm(recurrentSize, eps=1e-4), nn.SiLU())
        self.gates      = BlockLinear(self.blocks, self.blockSize, 3*self.blockSize)

    def forward(self, recurrentState, latentState, action):
        action  = action/action.abs().clamp(min=1).detach()
        inputs  = torch.cat((self.recurrentIn(recurrentState), self.latentIn(latentState), self.actionIn(action)), -1)
        inputs  = inputs.unsqueeze(-2).expand(*inputs.shape[:-1], self.blocks, inputs.shape[-1])
        x       = torch.cat((recurrentState.unflatten(-1, (self.blocks, self.blockSize)), inputs), -1)
        x       = self.hiddenNorm(self.hidden(x))
        gates   = self.gates(x.unflatten(-1, (self.blocks, self.blockSize))).unflatten(-1, (self.blocks, 3*self.blockSize))
        reset, candidate, update = [gate.flatten(-2) for gate in gates.chunk(3, dim=-1)]
        candidate   = torch.tanh(torch.sigmoid(reset)*candidate)
        update      = torch.sigmoid(update - 1)
        return update*candidate + (1 - update)*recurrentState


class LatentNet(nn.Module):
    """Prior (input h) or posterior (input h + encoded observation): a vector of categoricals with 1% unimix."""
    def __init__(self, inputSize, numLayers, config):
        super().__init__()
        self.latentLength, self.latentClasses, self.uniformMix = config.latentLength, config.latentClasses, config.uniformMix
        self.network = MLP(inputSize, self.latentLength*self.latentClasses, config.hiddenSize, numLayers)

    def forward(self, x):
        logits = uniformMixLogits(self.network(x).float().unflatten(-1, (self.latentLength, self.latentClasses)), self.uniformMix)
        return sampleLatent(logits).flatten(-2), logits


class ConvStack(nn.Module):
    """Stride-2 convolutions (or transposed convolutions), each followed by channel RMSNorm and SiLU."""
    def __init__(self, channels, transposed=False, finalActivation=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (inChannels, outChannels) in enumerate(zip(channels[:-1], channels[1:])):
            conv = nn.ConvTranspose2d(inChannels, outChannels, 4, 2, 1) if transposed else nn.Conv2d(inChannels, outChannels, 5, 2, 2)
            last = i == len(channels) - 2
            self.layers.append(nn.ModuleList([conv, nn.RMSNorm(outChannels, eps=1e-4) if not last or finalActivation else nn.Identity()]))
        self.finalActivation = finalActivation

    def forward(self, x):
        for i, (conv, norm) in enumerate(self.layers):
            x = conv(x)
            if i < len(self.layers) - 1 or self.finalActivation:
                x = nn.functional.silu(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


def convChannels(imageSize, depth):
    """Halve the resolution until it is at most 6 (96 -> 6x6, 64 -> 4x4); channels depth * (1, 2, 3, 4, 4, ...)."""
    numLayers, size = 0, imageSize
    while size > 6:
        size, numLayers = size//2, numLayers + 1
    return [depth*multiplier for multiplier in (1, 2, 3, 4, 4, 4, 4)[:numLayers]], size


class Encoder(nn.Module):
    """CNN for "image" (uint8, H x W x 3) and a symlog MLP for "state"; outputs their concatenation."""
    def __init__(self, observationShapes, config):
        super().__init__()
        self.hasImage, self.hasState = "image" in observationShapes, "state" in observationShapes
        self.outputSize = 0
        if self.hasImage:
            channels, finalSize = convChannels(observationShapes["image"][0], config.cnnDepth)
            self.convolutional  = ConvStack([3] + channels)
            self.outputSize    += channels[-1]*finalSize*finalSize
        if self.hasState:
            self.stateNetwork   = MLP(observationShapes["state"][0], None, config.hiddenSize, config.encoderLayers)
            self.outputSize    += config.hiddenSize

    def forward(self, observations):
        features = []
        if self.hasImage:
            image = observations["image"]
            leading = image.shape[:-3]
            image = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2).float()/255.0 - 0.5
            features.append(self.convolutional(image).flatten(1).view(*leading, -1))
        if self.hasState:
            features.append(self.stateNetwork(symlog(observations["state"])))
        return torch.cat(features, -1)


class Decoder(nn.Module):
    """Transposed CNN with sigmoid output for "image", MLP predicting symlog("state")."""
    def __init__(self, inputSize, observationShapes, config):
        super().__init__()
        self.hasImage, self.hasState = "image" in observationShapes, "state" in observationShapes
        if self.hasImage:
            channels, self.finalSize = convChannels(observationShapes["image"][0], config.cnnDepth)
            self.channels   = channels[-1]
            self.project    = nn.Sequential(initLinear(nn.Linear(inputSize, self.channels*self.finalSize**2)), nn.RMSNorm(self.channels*self.finalSize**2, eps=1e-4), nn.SiLU())
            self.deconvolutional = ConvStack(channels[::-1] + [3], transposed=True, finalActivation=False)
        if self.hasState:
            self.stateNetwork = MLP(inputSize, observationShapes["state"][0], config.hiddenSize, config.decoderLayers)

    def forward(self, fullState):
        outputs = {}
        if self.hasImage:
            leading = fullState.shape[:-1]
            x = self.project(fullState.reshape(-1, fullState.shape[-1])).view(-1, self.channels, self.finalSize, self.finalSize)
            image = torch.sigmoid(self.deconvolutional(x)).permute(0, 2, 3, 1)
            outputs["image"] = image.reshape(*leading, *image.shape[1:])
        if self.hasState:
            outputs["state"] = self.stateNetwork(fullState)
        return outputs


class TwoHotHead(nn.Module):
    """Reward predictor or critic: MLP with a zero-initialized output over exponentially spaced bins."""
    def __init__(self, inputSize, numLayers, config):
        super().__init__()
        self.register_buffer("bins", twoHotBins(config.twoHotBins))
        self.network = MLP(inputSize, config.twoHotBins, config.hiddenSize, numLayers, outScale=0.0)

    def forward(self, x):
        return TwoHot(self.network(x), self.bins)


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.network = MLP(inputSize, 1, config.hiddenSize, config.continueLayers)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1).float())


class Actor(nn.Module):
    """Actions in [-1, 1]^actionSize (the env maps them to its own range)."""
    def __init__(self, inputSize, actionSize, config):
        super().__init__()
        self.minStd, self.maxStd = config.actorStd
        self.network = MLP(inputSize, 2*actionSize, config.hiddenSize, config.actorLayers, outScale=0.01)

    def forward(self, x):
        return boundedNormal(self.network(x).float(), self.minStd, self.maxStd)
