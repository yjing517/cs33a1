
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


# 1. CPU 小模型配置
torch.manual_seed(42)

VOCAB_SIZE = 256
CONTEXT_LENGTH = 16
BATCH_SIZE = 4

MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=32,
    num_layers=2,
    num_heads=4,
    d_ff=64,
    rope_theta=10000.0,
    device="cpu",
)

MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 0
COSINE_CYCLE_ITERS = 100

CHECKPOINT_PATH = Path("outputs/cpu_test_checkpoint.pt")


def create_model_optimizer():
    model = TransformerLM(**MODEL_CONFIG)
    optimizer = AdamW(model.parameters(), lr=MAX_LR)
    return model, optimizer


def train_step(model, optimizer, x, y, iteration):
    # 每次更新前设置实际学习率
    lr = get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=MAX_LR,
        min_learning_rate=MIN_LR,
        warmup_iters=WARMUP_ITERS,
        cosine_cycle_iters=COSINE_CYCLE_ITERS,
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad()

    # (B, T) -> (B, T, V)
    logits = model(x)

    # (B, T, V) -> (B*T, V)
    loss = cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        y.reshape(-1),
    )

    assert torch.isfinite(loss).item(), "Loss is not finite"

    loss.backward()

    # 检查是否有有效的非零梯度
    grads = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]
    assert grads, "No gradients"
    assert all(torch.isfinite(g).all().item() for g in grads)
    assert any(torch.any(g != 0).item() for g in grads)

    optimizer.step()

    return loss.item(), lr


def main():
    # 使用简单的重复 token 序列
    train_data = np.load("outputs/lm_data/train.npy", mmap_mode="r")
    val_data = np.load("outputs/lm_data/val.npy", mmap_mode="r")

    # 固定批次，只采样一次
    x, y = get_batch(
        dataset,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        device="cpu",
    )

    model, optimizer = create_model_optimizer()

    # 实验 1：单步更新
    print("=== Experiment 1: Single Step ===")

    param = next(model.parameters())
    before = param.detach().clone()

    loss, lr = train_step(model, optimizer, x, y, 0)

    delta = (param.detach() - before).abs().max().item()

    print(f"loss={loss:.6f}, lr={lr:.6f}")
    print(f"max parameter change={delta:.8f}")

    assert delta > 0, "Parameters did not change"

    # 实验 2：固定批次过拟合
    print("\n=== Experiment 2: Overfit One Batch ===")

    first_loss = loss

    for step in range(1, 100):
        loss, lr = train_step(
            model, optimizer, x, y, step
        )

        if (step + 1) % 10 == 0:
            print(f"step={step+1}, loss={loss:.6f}")

    print(f"first_loss={first_loss:.6f}")
    print(f"final_loss={loss:.6f}")

    assert loss < first_loss, "Loss did not decrease"

    # 实验 3：断点续训
    print("\n=== Experiment 3: Checkpoint Resume ===")

    CHECKPOINT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    # iteration 表示已经完成的更新次数
    save_checkpoint(
        model, optimizer, 100, CHECKPOINT_PATH
    )

    restored_model, restored_optimizer = (
        create_model_optimizer()
    )

    iteration = load_checkpoint(
        CHECKPOINT_PATH,
        restored_model,
        restored_optimizer,
    )

    assert iteration == 100

    # 对比恢复前后的模型参数
    for p1, p2 in zip(
        model.parameters(),
        restored_model.parameters(),
    ):
        assert torch.equal(p1, p2)

    # 对比恢复前后的优化器状态
    for p1, p2 in zip(
        model.parameters(),
        restored_model.parameters(),
    ):
        s1 = optimizer.state[p1]
        s2 = restored_optimizer.state[p2]
        assert s1.keys() == s2.keys()
        for key in s1:
            v1, v2 = s1[key], s2[key]
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2

    # 连续训练路径
    continuous_loss, continuous_lr = train_step(
        model, optimizer, x, y, iteration
    )

    # 恢复训练路径
    resumed_loss, resumed_lr = train_step(
        restored_model,
        restored_optimizer,
        x, y, iteration
    )

    assert continuous_lr == resumed_lr

    for p1, p2 in zip(
        model.parameters(),
        restored_model.parameters(),
    ):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    print(f"resumed iteration={iteration}")
    print(f"continuous lr={continuous_lr:.6f}")
    print(f"resumed lr={resumed_lr:.6f}")
    print(f"continuous loss={continuous_loss:.6f}")
    print(f"resumed loss={resumed_loss:.6f}")
    print("All CPU experiments passed!")


if __name__ == "__main__":
    main()
