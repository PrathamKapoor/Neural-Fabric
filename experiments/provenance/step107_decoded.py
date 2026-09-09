"python -c \"
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
print('\
=== Running NF-03 Hard Baseline for seeds: %s ===' % seeds)
expected_bias = {'spatial': 'cnn', 'relational': 'a
<truncated 2784 bytes>