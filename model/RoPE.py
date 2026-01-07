class RoPE(nn.Module):
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        needs_refresh = (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if needs_refresh:
            self._seq_len_cached = seq_len
            position = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(position, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = seq_len or x.size(-2)
        self._update(seq_len, x.device, x.dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]