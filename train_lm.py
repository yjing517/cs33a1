
import argparse
import csv
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    selected, remaining = config_parser.parse_known_args()
    p = argparse.ArgumentParser(parents=[config_parser])

    p.add_argument("--train-data", default="outputs/lm_data/train.npy")
    p.add_argument("--val-data", default="outputs/lm_data/val.npy")
    p.add_argument("--output-dir", default="outputs/lm_runs/smoke")
    p.add_argument("--resume", default=None)

    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--cosine-cycle-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=10)

    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)

    if selected.config:
        with selected.config.open(encoding="utf-8") as f:
            config = json.load(f)
        unknown = config.keys() - vars(p.parse_args([])).keys()
        if unknown:
            p.error(f"Unknown config keys: {sorted(unknown)}")
        p.set_defaults(**config)
    return p.parse_args(remaining)


@torch.no_grad()
def evaluate(model, data, windows, args):
    model.eval()
    losses = []

    offsets = np.arange(args.context_length)
    for starts in windows:
        positions = starts[:, None] + offsets
        x = torch.as_tensor(data[positions], dtype=torch.long, device=args.device)
        y = torch.as_tensor(data[positions + 1], dtype=torch.long, device=args.device)

        logits = model(x)

        loss = cross_entropy(
            logits.reshape(-1, args.vocab_size),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_training_state(model, optimizer, step, path, best_loss, schedule, device):
    buffer = BytesIO()
    save_checkpoint(model, optimizer, step, buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    state.update(
        best_loss=best_loss,
        schedule=schedule,
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def create_model(args):
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
        device=args.device,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    assert args.d_model % args.num_heads == 0
    device = torch.device(args.device)
    args.device = str(device)
    assert args.eval_every > 0
    assert args.save_every > 0
    assert args.eval_batches > 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载真实数据
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    for name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        assert data.ndim == 1
        assert len(data) > args.context_length
        assert int(data.min()) >= 0
        assert int(data.max()) < args.vocab_size, (
            f"{name} token ID exceeds vocab size"
        )

    # Keep validation independent of both the training seed and batch size.
    windows = np.random.default_rng(0).integers(
        len(val_data) - args.context_length, size=(args.eval_batches, 8)
    )

    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens: {len(val_data)}")

    # 2. 创建 Transformer
    model = create_model(args)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # 3. 恢复 checkpoint
    start_step = 0

    best_loss = float("inf")
    schedule = dict(
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "schedule" in state:
            schedule = state["schedule"]
            best_loss = state["best_loss"]
        else:
            print("Legacy checkpoint: using supplied schedule and a new RNG history.")
        start_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
        )
        assert 0 <= start_step <= args.max_steps
        if "cpu_rng" in state:
            torch.set_rng_state(state["cpu_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], device)

    assert 0 <= schedule["warmup_iters"] < schedule["cosine_cycle_iters"]

    # 保存配置，便于后续对比实验
    config_path = output_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "schedule": schedule}, f, indent=2)

    log_path = output_dir / "metrics.csv"
    history = []
    if args.resume and log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as f:
            history = [
                row for row in list(csv.reader(f))[1:] if int(row[0]) <= start_step
            ]

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "lr"])
        writer.writerows(history)

        model.train()

        # 5. 正式训练循环
        for step in range(start_step, args.max_steps):

            lr = get_lr_cosine_schedule(it=step, **schedule)

            for group in optimizer.param_groups:
                group["lr"] = lr

            # 每一步重新随机采样
            x, y = get_batch(
                train_data,
                args.batch_size,
                args.context_length,
                args.device,
            )

            optimizer.zero_grad()

            logits = model(x)

            loss = cross_entropy(
                logits.reshape(-1, args.vocab_size),
                y.reshape(-1),
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at step {step}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            optimizer.step()

            completed = step + 1

            val_loss = None
            if completed == 1 or completed % args.eval_every == 0:
                val_loss = evaluate(model, val_data, windows, args)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_training_state(
                        model, optimizer, completed, output_dir / "best.pt",
                        best_loss, schedule, device,
                    )

            writer.writerow([completed, loss.item(), val_loss, lr])
            f.flush()
            if val_loss is not None:
                print(
                    f"step={completed} train={loss.item():.4f} "
                    f"val={val_loss:.4f} lr={lr:.6f}",
                    flush=True,
                )

            if completed % args.save_every == 0 or completed == args.max_steps:
                save_training_state(
                    model, optimizer, completed, output_dir / "last.pt",
                    best_loss, schedule, device,
                )

    print("Training finished:", output_dir)


if __name__ == "__main__":
    main()
