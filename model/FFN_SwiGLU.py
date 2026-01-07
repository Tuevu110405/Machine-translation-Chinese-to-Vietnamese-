class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_ff = d_ff  # <--- Thêm dòng này
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear1(x)
        g, v = h[..., :self.d_ff], h[..., self.d_ff:]
        # Công thức đúng: (SiLU(g) * v)
        # F.silu(g) tương đương g * sigmoid(g)
        s = F.silu(g) * v  # <--- Sửa logic nhân với v
        out = self.linear2(s)
        return self.dropout(out)