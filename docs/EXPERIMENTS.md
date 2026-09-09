# Experiments

Every experiment writes a machine-readable artifact (see
`lab/experiment.py::build_envelope`): ID, hypothesis, config, git commit,
environment, dataset, split, seeds, model, budgets, metrics, compute,
artifacts, timestamp, status. Seeds are never averaged away: per-seed rows
persist alongside mean/std/CI.

## Neural Fabric battery (all CPU-only, deterministic by seed)

| ID | Question | Seeds | Artifact |
|---|---|---|---|
| NF-01 | fixed baselines | 8 | `experiments/results/nf01_fixed_baselines.json` |
| NF-02 | soft routing | 5 | `.../nf02_soft_training.json` |
| NF-03 | hard baseline (untrained) | 3 | `.../nf03_hard_baseline_*.json` |
| NF-04 | top-k compute ablation | 10 samples | `.../nf04_topk_ablation.json` |
| NF-07 | compute-aware λ sweep | 3×9 cells | `.../nf07_sweep_matrix.json` |
| NF-09 | noise curve (5 levels) | 5 | `.../nf09_robustness.json` |
| NF-10 | distribution shift | — | `.../nf10_distribution_shift.json` |
| NF-11 | true-FLOP scaling | 100 samples | `.../nf11_scalable_100.json` |
| NF-12 | routing modes + paired tests | 5 | `.../nf12_routing_modes.json` |
| NF-13 | specialization (MI) | 5 | `.../nf13_specialization.json` |
| NF-14 | hierarchical transfer | 5 | `.../nf14_hierarchical.json` |
| NF-15 | fresh-seed extension (6–10) | 5 | `.../nf15_seed_extension.json` |

Rerun any script directly (e.g. `.venv\Scripts\python.exe
experiments\nf02_soft_training.py`); artifacts regenerate byte-identically
modulo timestamps and wall-clock fields.

## Cost-model correction pass (v2-corrected; historical artifacts preserved)

Historical artifacts above are the v1 cost-model record and are NOT
overwritten. Corrected runs (same code protocol, corrected
`fabric/primitives/cost_model.py`: mlp=17152, cnn=8256, attention-legacy=2048)
were executed with `--corrected` and saved alongside:

| ID | Corrected artifact | Result vs historical |
|---|---|---|
| NF-07 | `nf07_sweep_matrix_corrected.json` | tradeoff shape survives: λ_comp=2.0 cuts top-1 FLOPs ~11% (8836→7894) for −6pp acc (0.94→0.88); no collapse either run |
| NF-02 | `nf02_soft_training_corrected.json` (+ `runs/nf02_soft_corrected_seed*.json`) | acc/entropy bit-identical (0.94±0.051); only compute fields change (executed 30976→27840) |
| NF-12 | `nf12_routing_modes_corrected.json` | acc + paired p-values bit-identical; top-1 FLOPs 10004.8→8835.2; null conclusions unchanged |
| NF-14 | `nf14_hierarchical_corrected.json` | acc + paired p bit-identical; h_hier_supported=False both; only exec-FLOPs change |
| NF-04 | `nf04_topk_ablation_corrected.json` | k1<k2<k3 ordering preserved (unseeded 10-sample demo; values not cross-run comparable) |
| NF-09 | `nf09_robustness_corrected.json` | path-change curve bit-identical; only compute fields change |
| NF-10 | `nf10_distribution_shift_corrected.json` | match rates identical (0.45→0.0); only mean_compute changes |
| NF-11 | `nf11_scalable_100_corrected.json` | per-sample FLOPs now from corrected primitives |
| NF-01 | `nf01_fixed_baselines_corrected_compute.json` | REPORTING-ONLY derivation (training is cost-independent): pareto becomes {CNN, Attention, MLP}; efficiency winner flips CNN→Attention (0.44 acc/kFLOP); best-fixed identity flips on tied seeds 456, 5 (accuracies unchanged) |
| NF-13 | no corrected file (verified UNCHANGED) | rerun bit-identical modulo wall-clock; MI cost-free; interpretation note: "attention" selections = legacy projection |
| NF-15 | NOT AFFECTED (accuracy-only, no FLOPs fields) | — |

Attention probe (non-ladder): `attention_primitive_characterization.json`
from `experiments/probe_attention_primitives.py` — numerical/cost evidence
only, no accuracy claims.

## Robustness pass (10 seeds; historical + n=3 artifacts preserved)

| ID | Artifact | Finding |
|---|---|---|
| NF-07 10-seed | `nf07_sweep_10seed_corrected.json` (+ `nf07_10seed_statistics.json` via `scripts/analyze_nf07_10seed.py`) | λ_comp=2.0 cuts compute on 10/10 seeds (mean −752 FLOPs, p≈0.0006); accuracy cost −4.8pp (p≈0.006); weak λ_comp=0.5 null; 0 failures |
| NF-01 10-seed | `nf01_fixed_10seed_corrected.json` (via `experiments/nf01_10seed_robustness.py`, detached job) | efficiency ordering stable: legacy-projection > CNN > MLP acc/kFLOP on all seeds; mean frontier {attention, cnn, mlp} |
| NX-ATTN-01 (new, non-ladder) | `nx_attn_genuine_vs_legacy.json` (via `experiments/nx_attn_genuine.py`, detached job; gradients finite-diff verified ~2e-10) | genuine softmax attention LOSES to legacy projection 10/10 seeds (mean −27pp) at identical budget; costs 5,104 vs 2,096 FLOPs |

All three long runs executed detached through `lab/jobs.py`
(queue→detached→poll→completion); job records live in git-ignored
`.agent_loop/jobs/`.
