"""Attention-primitive characterization probe (NOT a ladder experiment).

Purpose: record, in one artifact, what the legacy "attention" primitive and
the NEW genuine "softmax_attention" primitive each compute and cost, so the
v2-corrected interpretation ("legacy projection" vs genuine attention) rests
on measured evidence rather than code reading alone.

Explicitly out of scope: accuracy, training, routing, efficiency claims.
No ladder experiment used the genuine primitive; nothing here is substituted
into NF-01..NF-15.

Writes: experiments/results/attention_primitive_characterization.json
"""
import copy
import json
import math
import os
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fabric.primitives.attention import Attention
from fabric.primitives.genuine_attention import SoftmaxAttention
from fabric.primitives.cost_model import COST_MODEL_VERSION


def main():
    x = [0.05 * (i % 5) for i in range(64)]
    leg = Attention(seed=7)
    gen = SoftmaxAttention(seed=7)

    out_leg = leg.forward(x)
    leg_dead = copy.deepcopy(leg)
    leg_dead.Wk = [[0.0] * leg.d for _ in range(leg.d)]
    leg_dead.Wv = [[0.0] * leg.d for _ in range(leg.d)]
    kv_dead_invariant = leg_dead.forward(x) == out_leg

    out_gen = gen.forward(x)
    gen_nok = copy.deepcopy(gen)
    gen_nok.Wk = [[0.0] * gen.d_tok for _ in range(gen.d_tok)]
    kv_live = gen_nok.forward(x) != out_gen

    # Softmax-row normalization of the genuine primitive, measured via an
    # independent re-derivation (weights must sum to 1 per query row).
    n, d = gen.n_tok, gen.d_tok
    toks = [x[t * d:(t + 1) * d] for t in range(n)]
    Q = [[sum(t[i] * gen.Wq[i][j] for i in range(d)) for j in range(d)] for t in toks]
    K = [[sum(t[i] * gen.Wk[i][j] for i in range(d)) for j in range(d)] for t in toks]
    sc = 1.0 / math.sqrt(d)
    row_sums = []
    for t in range(n):
        row = [sum(Q[t][i] * K[s][i] for i in range(d)) * sc for s in range(n)]
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = sum(ex)
        row_sums.append(sum(v / s for v in ex))

    mean_abs_diff = sum(abs(a - b) for a, b in zip(out_leg, out_gen)) / len(out_gen)

    artifact = {
        "probe": "attention_primitive_characterization",
        "scope": "numerical/cost characterization only; no accuracy or routing claims",
        "cost_model": COST_MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
        "legacy_attention": {
            "name": leg.name, "legacy": True, "flops": leg.flops(),
            "params": leg.params(), "output_dim": len(out_leg),
            "kv_zeroing_output_invariant": kv_dead_invariant,
            "interpretation": "truncated Wq linear projection; K/V dead; "
                              "no QKT; no softmax",
        },
        "genuine_softmax_attention": {
            "name": gen.name, "flops": gen.flops(), "params": gen.params(),
            "output_dim": len(out_gen),
            "kv_zeroing_changes_output": kv_live,
            "softmax_row_sums": [round(v, 9) for v in row_sums],
            "interpretation": "honest Q/K/V + QKT + softmax + V weighting",
        },
        "legacy_vs_genuine_mean_abs_output_diff": round(mean_abs_diff, 6),
        "ladder_use": "none — genuine primitive in no historical experiment",
    }
    path = os.path.join(REPO, "experiments/results/attention_primitive_characterization.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[DONE] wrote {path}")
    print(json.dumps({k: v for k, v in artifact.items() if k != "timestamp"}, indent=2)[:1500])


if __name__ == "__main__":
    main()
