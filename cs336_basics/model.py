import torch
from torch import nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE 要求 d_k 为偶数"

        i = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        # 每个维度对的逆频率: theta^(-2i/d_k)
        inv_freq = 1.0 / (theta ** (i / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cache", torch.cos(angles).to(dtype=dtype), persistent=False
        )
        self.register_buffer(
            "sin_cache", torch.sin(angles).to(dtype=dtype), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
            token_positions = token_positions.expand(x.shape[:-1])

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 相邻两个分量配对
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        # 合并成原来的形状
        return torch.stack(
            (y_even, y_odd),
            dim=-1,
        ).flatten(-2)

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features, device=device, dtype=dtype))
        std=(2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std,a=-3*std,b=3*std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T



class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.float()
        mean_squared = x.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(mean_squared + self.eps)

        return x.to(in_dtype) * self.weight



class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 分支 1：后面接 SiLU
        self.w1 = Linear(
            d_model, d_ff,
            device=device, dtype=dtype,
        )

        # 输出投影：d_ff -> d_model
        self.w2 = Linear(
            d_ff, d_model,
            device=device, dtype=dtype,
        )

        # 分支 2：不经过激活
        self.w3 = Linear(
            d_model, d_ff,
            device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        value = self.w3(x)

        hidden = gate * value

        return self.w2(hidden)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = (Q @ K.transpose(-2, -1)) / Q.shape[-1] ** 0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        # Give fully masked rows a finite input to softmax.
        scores = scores.masked_fill(~mask.any(dim=-1, keepdim=True), 0)

    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0)

    return weights @ V
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        # 四个线性投影
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        B, T, D = x.shape

        # 1. 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分多个注意力头
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # 3. 调整维度 (B, H, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. 可选 RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)

            # 兼容 (T,) 和 (B,T) 的位置张量
            if token_positions.ndim == 2:
                token_positions = token_positions.unsqueeze(1)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 5. 因果 Mask
        mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=x.device)
        )

        # 6. 多头注意力
        out = scaled_dot_product_attention(q, k, v, mask)

        # 7. 合并注意力头
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, D)

        # 8. 输出投影
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()


        self.ln1=RMSNorm(d_model,device=device,dtype=dtype)

        rope = RotaryPositionalEmbedding(
            d_k=d_model // num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        self.ln2=RMSNorm(d_model,device=device,dtype=dtype)

        self.ffn=SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        h=x+self.attn(self.ln1(x))
        return h+self.ffn(self.ln2(h))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 1. Token Embedding
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. 多层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3. 最终 RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        # 4. 输出投影
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        # (B, T) -> (B, T, d_model)
        x = self.token_embeddings(token_ids)

        # 逐层经过 Transformer Block
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.ln_final(x)

        # (B, T, d_model) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 减去最大值，防止指数溢出
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    # inputs: (batch_size, vocab_size)
    # targets: (batch_size,)

    # 1. 每行最大值
    max_logits = inputs.max(dim=-1, keepdim=True).values

    # 2. 平移 logits
    shifted = inputs - max_logits

    # 3. 稳定计算 log(sum(exp(logits)))
    log_sum_exp = (
        torch.log(torch.exp(shifted).sum(dim=-1))
        + max_logits.squeeze(-1)
    )

    # 4. 取正确类别对应的 logit
    target_logits = inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 5. 平均交叉熵
    return (log_sum_exp - target_logits).mean()

if __name__ == "__main__":
    layer = Linear(4, 3)

    x = torch.randn(2, 5, 4)
    y = layer(x)

    print("输入:", x.shape)
    print("权重:", layer.weight.shape)
    print("输出:", y.shape)

    embedding = Embedding(3, 2)

    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor([
                [10., 11.],
                [20., 21.],
                [30., 31.],
            ])
        )

    ids = torch.tensor([
        [2, 0],
        [1, 2],
    ])

    output = embedding(ids)

    print(output)
    print(output.shape)
