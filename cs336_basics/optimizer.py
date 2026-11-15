
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid betas")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # 更新一阶矩、二阶矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )

                # 偏差修正
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 解耦权重衰减
                p.mul_(1 - lr * wd)

                # 参数更新
                p.addcdiv_(
                    m_hat,
                    v_hat.sqrt().add(eps),
                    value=-lr,
                )

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    # 1. 线性 Warmup
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 2. 衰减结束后固定最小学习率
    if it >= cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦衰减
    progress = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    return min_learning_rate + 0.5 * (
        max_learning_rate - min_learning_rate
    ) * (1 + math.cos(math.pi * progress))

def gradient_clipping(parameters, max_l2_norm):
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    params = [p for p in parameters if p.grad is not None]

    if not params:
        return

    # 全部梯度的整体 L2 范数
    total_norm = torch.linalg.vector_norm(
        torch.stack([
            torch.linalg.vector_norm(p.grad.detach().float())
            for p in params
        ])
    )

    # 避免除零
    scale = torch.clamp(
        max_l2_norm / (total_norm + 1e-6),
        max=1.0,
    )

    for p in params:
        p.grad.mul_(scale)
