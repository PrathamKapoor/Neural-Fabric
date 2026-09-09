"""CNN primitive — 2D convolutional filter over a small grid."""
import math, random

class CNN:
    def __init__(self, in_ch=1, out_ch=8, k=3, seed=0):
        random.seed(seed)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.filters = [
            [
                [
                    [random.gauss(0, 0.05) for _ in range(k)]
                    for _ in range(k)
                ]
                for _ in range(in_ch)
            ]
            for _ in range(out_ch)
        ]
        self.name = "cnn"

    def forward(self, x, state=None):
        # Deterministic 2D convolution over HxW grid (default 8x8 = 64 floats)
        H, W = 8, 8
        expected_len = self.in_ch * H * W
        if not isinstance(x, list):
            x = list(x)
        if len(x) == 0 or isinstance(x[0], (int, float)):
            flat = (x + [0.0] * max(0, expected_len - len(x)))[:expected_len]
            img = []
            idx = 0
            for _ in range(self.in_ch):
                ch = []
                for _ in range(H):
                    ch.append(flat[idx:idx + W])
                    idx += W
                img.append(ch)
        else:
            if isinstance(x[0], list) and (len(x[0]) == 0 or isinstance(x[0][0], (int, float))):
                img = [x]
            else:
                img = x

        pad = self.k // 2
        out_features = []
        for oc in range(self.out_ch):
            conv_map = []
            for r in range(H):
                row = []
                for c in range(W):
                    val = 0.0
                    for ic in range(min(len(img), self.in_ch)):
                        for kr in range(self.k):
                            ir = r + kr - pad
                            if 0 <= ir < len(img[ic]):
                                in_row = img[ic][ir]
                                for kc in range(self.k):
                                    ic_col = c + kc - pad
                                    if 0 <= ic_col < len(in_row):
                                        val += in_row[ic_col] * self.filters[oc][ic][kr][kc]
                    row.append(max(0.0, val))
                conv_map.append(row)

            # 2x2 spatial average pooling over 8x8 -> 4 pooled values per output channel
            for qr in range(2):
                r_start, r_end = qr * 4, (qr + 1) * 4
                for qc in range(2):
                    c_start, c_end = qc * 4, (qc + 1) * 4
                    pool_sum = sum(conv_map[r][c] for r in range(r_start, r_end) for c in range(c_start, c_end))
                    out_features.append(pool_sum / 16.0)

        return out_features

    def params(self):
        return self.in_ch * self.out_ch * self.k * self.k

    def flops(self):
        # Boundary-aware count (see cost_model.py): the forward loop visits
        # all 64 positions with same-padding edge checks, plus quadrant
        # average-pooling arithmetic. v1 used in*out*k*k*16 (=1152), which
        # wrongly assumed 16 positions; corrected value for defaults is 8256.
        from .cost_model import cnn_flops
        return cnn_flops(self.in_ch, self.out_ch, self.k)
