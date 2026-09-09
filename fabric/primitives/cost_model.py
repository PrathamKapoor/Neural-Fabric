"""Neural Fabric cost model — single source of truth for primitive compute costs.

COST_MODEL_VERSION = "v2-corrected" (supersedes the undocumented v1 model that
shipped with commit 0a4cbea and earlier).

Counting convention (applies to every number in this module):
- 1 FLOP = one floating-point multiply OR one floating-point add
  (i.e. one multiply-accumulate, MAC, = 2 FLOPs).
- Bias additions ARE counted (one add per output element).
- The CNN's 2x2-quadrant average-pooling arithmetic (adds + divides) IS
  counted because it executes on every forward pass.
- Elementwise ReLU comparisons (max(0, .)), softmax exp/log/normalization in
  the *router*, and Python-level index arithmetic are NOT counted. They are
  real but second-order next to the dense linear algebra, and excluding them
  keeps the model implementation-independent. This exclusion is documented
  here instead of being silently mixed in.
- Router cost (ROUTER_FLOPS) counts only the 64x3 linear encode
  (2*64*3 = 384 FLOPs); the router softmax is excluded per the rule above.
- Never mix MACs and FLOPs: all values below are FLOPs under this convention.

Validated values (default constructors):
- MLP   (in=64, hidden=128, out=3): 2*64*128 + 2*128*3 = 17152.
  The v1 value 17152 was already correct under this convention.
- CNN   (in_ch=1, out_ch=8, k=3, 8x8 input): 8256, boundary-aware.
  The v1 value 1152 (= in*out*k*k*16) assumed 16 output positions; the real
  forward loop visits all 64 positions with same-padding boundary checks, so
  only interior positions execute the full k*k kernel. Derivation: per output
  position the executed kernel taps equal (#valid rows)x(#valid cols); over
  the 8x8 grid with pad=1, k=3 this sums to 484 taps per (in_ch, out_ch)
  pair -> 2*1*8*484 = 7744 conv FLOPs, plus 8ch x 4 quadrants x 16
  (15 adds + 1 divide) = 512 pooling FLOPs -> 8256 total.
- Attention-legacy (d=32): 2048 = 2*32*32, ONE 32x32 matrix-vector product.
  The v1 value 12288 (= 3*d*d*4) charged for Q, K, V plus score math that the
  pure-Python forward() never executes: only Wq is used; Wk/Wv are dead
  weights and there is no QK^T or softmax. See attention.py (legacy).

Corrected cost ordering: attention-legacy (2048) < cnn (8256) < mlp (17152).
"""

COST_MODEL_VERSION = "v2-corrected"
COST_MODEL_SUPERSEDES = "v1-undocumented (mlp=17152, cnn=1152, attention=12288)"

CONVENTION = (
    "1 FLOP = 1 mul or 1 add (1 MAC = 2 FLOPs); "
    "bias adds counted; CNN pooling arithmetic counted; "
    "ReLU comparisons and router softmax excluded (documented, second-order)."
)


def mlp_flops(in_dim=64, hidden=128, out_dim=3):
    """Exact mul+add count of MLP.forward (both matvecs incl. bias)."""
    return 2 * in_dim * hidden + 2 * hidden * out_dim


def cnn_flops(in_ch=1, out_ch=8, k=3, h=8, w=8):
    """Boundary-aware mul+add count of CNN.forward incl. quadrant pooling.

    Mirrors the forward loop bounds arithmetically: for each output position
    only in-bounds kernel taps execute (same-padding edge behavior), then
    adds the 2x2-quadrant average-pooling arithmetic (16 ops per quadrant:
    15 adds + 1 divide).
    """
    pad = k // 2
    taps = 0
    for r in range(h):
        n_valid_rows = sum(1 for kr in range(k) if 0 <= r + kr - pad < h)
        for c in range(w):
            n_valid_cols = sum(1 for kc in range(k) if 0 <= c + kc - pad < w)
            taps += n_valid_rows * n_valid_cols
    conv_flops = 2 * in_ch * out_ch * taps
    pooling_flops = out_ch * 4 * 16
    return conv_flops + pooling_flops


def legacy_attention_flops(d=32):
    """Executed mul+add of the legacy Attention.forward: one d×d matvec."""
    return 2 * d * d


PRIM_FLOPS = {
    "mlp": mlp_flops(),
    "cnn": cnn_flops(),
    "attention": legacy_attention_flops(),
}
TOTAL_FLOPS = float(sum(PRIM_FLOPS.values()))
PRIM_NAMES = ["mlp", "cnn", "attention"]

# Router linear-encode cost only (softmax excluded, see convention).
ROUTER_FLOPS = 2 * 64 * 3
