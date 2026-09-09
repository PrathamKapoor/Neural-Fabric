python -c "
import os, sys, json, time, math
from collections import Counter
sys.path.insert(0, '.')

from experiments.nf02_soft_training import train_soft_single_seed
from fabric.engine import FabricEngine
from fabric.router.router import Router
from fabric.primitives.mlp import MLP
from evaluation.synthetic_dataset import make_sample

seeds = [42, 123, 456]
os.makedirs('experiments/results', exist_ok=True)

# 1. NF-02 Soft Routing Training (multi-seed)
print('=== Running NF-02 Soft Training for seeds: %s ===' % seeds)
for s in seeds:
    t0 = time.time()
    res = train_soft_single_seed(seed=s, n_epochs=30, n_train=100, n_test=50, lr=0.05)
    train_time = round(time.time() - t0, 4)
    nf02_payload = {
        'experiment': 'nf02_soft_training',
        'seed': s,
        'final_train_acc': res['train_accuracy_final'],
        'final_val_acc': None,
        'final_test_acc': res['test_accuracy'],
        'routing_entropy': res['test_entropy'],
        'compute_flops': res['compute']['executed_flops'],
        'weighted_expected_flops': res['compute']['weighted_expected_flops'],
        'training_time': train_time,
        'training_curves': {
            'loss': res['loss_trend'],
            'entropy': res['entropy_trend'],
            'train_accuracy': res['train_accuracy_trend']
        },
        'routing_statistics': res['collapse_metrics'],
        'collapse_detected': res['collapse_metrics']['collapse_detected']
    }
    out_path = 'experiments/results/nf02_soft_training_%d.json' % s
    with open(out_path, 'w') as f:
        json.dump(nf02_payload, f, indent=2)
    print('Saved %s | Train Acc: %s | Test Acc: %s | Entropy: %s | FLOPs: %s' % 
          (out_path, nf02_payload['final_train_acc'], nf02_payload['final_test_acc'], nf02_payload['routing_entropy'], nf02_payload['compute_flops']))

# 2. NF-03 Hard Routing Baseline (multi-seed)
print('\n=== Running NF-03 Hard Baseline for seeds: %s ===' % seeds)
expected_bias = {'spatial': 'cnn', 'relational': 'attention', 'dense': 'mlp'}
for s in seeds:
    t0 = time.time()
    engine = FabricEngine(mode='hard')
    engine.router = Router(in_dim=64, n_prims=3, mode='hard', seed=s)
    base = MLP()
    samples = [make_sample(m, seed=s * 1000 + i) for i, m in enumerate(['spatial', 'relational', 'dense'] * 10)]
    results = [engine.predict(samp['x'], graph_mode=True) for samp in samples]
    eval_time = round(time.time() - t0, 4)
    
    flops_list = [r['compute'] for r in results]
    selected = [r['graph']['nodes'][0]['primitive'] for r in results]
    correct = sum(1 for samp, sel in zip(samples, selected) if expected_bias[samp['label']] == sel)
    test_acc = round(correct / len(samples), 4)
    
    sel_counts = dict(Counter(selected))
    for p in ['mlp', 'cnn', 'attention']:
        sel_counts.setdefault(p, 0)
    sel_freqs = {p: round(sel_counts[p] / len(samples), 4) for p in ['mlp', 'cnn', 'attention']}
    
    probs = [sel_freqs[p] for p in ['mlp', 'cnn', 'attention']]
    ent = -sum(p * math.log(max(p, 1e-15)) for p in probs if p > 0)
    max_ratio = max(sel_freqs.values())
    min_ratio = min(sel_freqs.values())
    collapse = (max_ratio > 0.85) or (ent < 0.50)
    
    mean_flops = round(sum(flops_list) / len(flops_list), 2)
    base_flops = base.flops()
    savings_pct = round((1.0 - mean_flops / base_flops) * 100.0, 2)
    
    nf03_payload = {
        'experiment': 'nf03_hard_baseline',
        'seed': s,
        'final_train_acc': None,
        'final_val_acc': None,
        'final_test_acc': test_acc,
        'routing_entropy': round(ent, 4),
        'compute_flops': mean_flops,
        'baseline_compute_flops': base_flops,
        'compute_savings_pct': savings_pct,
        'training_time': eval_time,
        'training_curves': {
            'loss': [],
            'entropy': [],
            'train_accuracy': []
        },
        'routing_statistics': {
            'selection_counts': sel_counts,
            'selection_frequencies': sel_freqs,
            'max_primitive_ratio': max_ratio,
            'min_primitive_ratio': min_ratio,
            'global_routing_entropy': round(ent, 4),
            'collapse_detected': collapse
        },
        'samples_evaluated': len(samples),
        'note': 'Untrained hard routing baseline vs fixed MLP baseline per nf03 specification.'
    }
    out_path = 'experiments/results/nf03_hard_baseline_%d.json' % s
    with open(out_path, 'w') as f:
        json.dump(nf03_payload, f, indent=2)
    print('Saved %s | Test Acc: %s | Entropy: %s | FLOPs: %s | Savings: %s%% | Collapse: %s' % 
          (out_path, nf03_payload['final_test_acc'], nf03_payload['routing_entropy'], nf03_payload['compute_flops'], savings_pct, collapse))
"