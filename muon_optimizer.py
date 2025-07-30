import math
import torch
from torch.optim.optimizer import Optimizer

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    This optimizer combines the Muon update rule for 2D weight matrices with a standard
    AdamW update for all other parameters (e.g., embeddings, biases, LayerNorm).

    The separation of parameters is handled automatically. Simply pass `model.named_parameters()`
    during initialization.

    Arguments:
        named_params: An iterator of (name, param) tuples, as returned by `model.named_parameters()`.
                      This is REQUIRED for automatically separating parameters.
        lr: The learning rate (default: 1e-3).
        weight_decay: The weight decay (L2 penalty) (default: 0.1).
        momentum: The momentum factor for the Muon updates (default: 0.95).
        nesterov: Enables Nesterov momentum for the Muon updates (default: True).
        ns_steps: Number of Newton-Schulz iterations for orthogonalization (default: 5).
        betas: Coefficients used for computing running averages of gradient and its square in AdamW (default: (0.9, 0.95)).
        eps: Term added to the denominator to improve numerical stability in AdamW (default: 1e-8).
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        betas=(0.9, 0.95),
        eps=1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        # --- Parameter Segregation ---
        # Parameters are automatically segregated into two groups:
        # 1. Muon Group: 2D matrices that are not embeddings or the final LM head.
        # 2. AdamW Group: All other parameters (embeddings, biases, LayerNorms, etc.).

        muon_params = []
        adamw_params = []
        
        param_list = list(named_params)
        if not param_list:
            raise ValueError("Optimizer received an empty list of parameters.")
        if not isinstance(param_list[0], tuple):
            raise ValueError("Muon optimizer requires `model.named_parameters()` for initialization, "
                             "which yields (name, param) tuples. Please pass `model.named_parameters()` instead of `model.parameters()`.")

        for name, p in param_list:
            if p.requires_grad:
                # The core logic: apply Muon to 2D weights, excluding embeddings and output layers.
                if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

        param_groups = [
            # Group for Muon parameters
            dict(
                params=muon_params,
                is_muon=True,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
            ),
            # Group for AdamW parameters
            dict(
                params=adamw_params,
                is_muon=False,
                betas=betas,
                eps=eps,
            ),
        ]
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    def adjust_lr_for_muon(self, lr, param_shape):
        """Adjusts LR for Muon based on matrix dimensions."""
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not in None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            # ============================
            # --- Handle Muon Group ---
            # ============================
            if group.get("is_muon", False):
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    if grad.ndim > 2:
                        grad = grad.view(grad.size(0), -1)

                    # Calculate momentum-driven gradient
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                    
                    # Orthogonalize the update matrix
                    update_matrix = zeropower_via_newtonschulz5(grad, steps=ns_steps)

                    # Apply weight decay to the parameter
                    p.data.mul_(1 - lr * weight_decay)

                    # Apply the orthogonalized update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    p.data.add_(update_matrix, alpha=-adjusted_lr)

            # ============================
            # --- Handle AdamW Group ---
            # ============================
            else: # is_muon is False
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1
                    
                    # Weight decay
                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)

                    # AdamW update logic
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    
                    step = state["step"]
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    step_size = lr / bias_correction1
                    
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss