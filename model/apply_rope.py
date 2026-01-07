def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: (..., seq_len, head_dim)
    head_dim = x.shape[-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Reshape cos, sin để broadcast đúng: (1, 1, seq_len, head_dim/2)
    # Lưu ý: cos, sin từ hàm RoPE của bạn đang có shape (seq_len, head_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)