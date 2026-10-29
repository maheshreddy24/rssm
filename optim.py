import torch


class LaProp(torch.optim.Optimizer):
    """LaProp (RMSProp normalisation followed by momentum) with Adaptive Gradient Clipping, as in DreamerV3.

    AGC rescales each tensor's gradient so its norm is at most `agc` times the norm of the tensor itself
    (floored at `agcEpsilon`), which makes the clipping threshold independent of the loss scale."""
    def __init__(self, params, lr=4e-5, betas=(0.9, 0.99), eps=1e-20, agc=0.3, agcEpsilon=1e-3):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, agc=agc, agcEpsilon=agcEpsilon))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                gradient = param.grad
                if group["agc"]:
                    maxNorm  = group["agc"]*param.norm().clamp(min=group["agcEpsilon"])
                    gradient = gradient*torch.clamp(maxNorm/(gradient.norm() + 1e-6), max=1.0)

                state = self.state[param]
                if not state:
                    state["step"]           = 0
                    state["squaredAverage"] = torch.zeros_like(param)
                    state["momentum"]       = torch.zeros_like(param)
                state["step"] += 1
                state["squaredAverage"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                normalizer = (state["squaredAverage"]/(1 - beta2**state["step"])).sqrt_().add_(group["eps"])
                state["momentum"].mul_(beta1).add_(gradient/normalizer, alpha=1 - beta1)
                param.add_(state["momentum"], alpha=-group["lr"]/(1 - beta1**state["step"]))
