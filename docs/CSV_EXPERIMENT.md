# Numeric CSV classification

The CSV workflow is limited to finite numerical features and categorical text
labels. It rejects missing cells, duplicate headers, absent columns and classes
with fewer than five examples. The split is stratified by class, approximately
60/20/20. Means and population standard deviations are fitted on training rows
only and saved with the checkpoint. Evaluation checks the raw CSV SHA-256
against the training input and uses its held-out indices. Prediction accepts
new rows using the saved feature order and transformations.

## Reproducing the Iris example

[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), Fisher (1988), DOI
10.24432/C56C76, is licensed by UCI under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We used the
original `iris.data` (150 examples, four numeric features), downloaded from
`https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`.
The raw SHA-256 was
`6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0`.
The example adds a header and removes the trailing blank line:

```python
from pathlib import Path
raw = Path("iris.data").read_text().strip()
Path("iris.csv").write_text("sepal_length,sepal_width,petal_length,petal_width,species\n" + raw + "\n")
```

For each seed 1, 2 and 3, train with `--epochs 40 --seed SEED`, then run
`evaluate-csv --csv iris.csv --model MODEL --mode top1`. Each split contains
90 training, 30 validation and 30 test examples. The routed model and fixed
linear, 16-hidden and 32-hidden MLP baselines use the same splits and
preprocessing, Adam learning rate 0.01, and validation-loss checkpoint
selection; initializations and parameter counts differ.

| Seed | Model | Accuracy | Cross-entropy loss | Estimated FLOPs/sample | Median ms |
|---|---|---:|---:|---:|---:|
| 1 | routed top-1 | 0.933 | 0.129 | 397.3 | 0.354 |
| 1 | linear | 0.833 | 0.399 | 24 | 0.034 |
| 1 | small MLP | 0.933 | 0.151 | 224 | 0.081 |
| 1 | wide MLP | 0.933 | 0.107 | 448 | 0.072 |
| 2 | routed top-1 | 0.833 | 0.393 | 397.3 | 0.224 |
| 2 | linear | 0.733 | 0.579 | 24 | 0.017 |
| 2 | small MLP | 0.900 | 0.227 | 224 | 0.040 |
| 2 | wide MLP | 0.900 | 0.216 | 448 | 0.041 |
| 3 | routed top-1 | 0.933 | 0.177 | 322.7 | 0.215 |
| 3 | linear | 0.800 | 0.408 | 24 | 0.017 |
| 3 | small MLP | 0.933 | 0.170 | 224 | 0.041 |
| 3 | wide MLP | 0.933 | 0.146 | 448 | 0.040 |

These are exploratory, small-sample results, not evidence of generalization.
Top-1 used more estimated compute than the small MLP in all three runs and
was slower in these measurements. FLOPs count two operations per multiply-add,
excluding bias, activations, softmax and dispatch. Latency is median of 30
batch-one CPU calls after five warmups, excluding HTTP, on Python 3.13 and
PyTorch 2.13 CPU; timing varies across environments.

## Container

`Dockerfile.trained` is a separate image for the trained demo. Build with
`docker build -f Dockerfile.trained -t neural-fabric-trained .`, train a demo
checkpoint locally, and mount its directory at `/models` when running:
`docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD/artifacts:/models:ro" neural-fabric-trained`.
The container expects `/models/demo-model.pt`. The Docker daemon was
unavailable during this update, so build and runtime remain unverified.
