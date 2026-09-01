# ENTRAP — Installation & Usage

ENTRAP (ENtropy-based Topological Rescue of Ambiguous Points) is a Cython-based Python library that refines HDBSCAN clustering results.

---

## 1. Prerequisite: Install Git

`pip install git+...` requires **Git** to clone the repository before building it. If you use Anaconda/Miniconda and don't have Git yet:

```bash
conda install -c conda-forge git
```

Verify:

```bash
git --version
```

> **Not using conda?**
> - **Windows**: [git-scm.com](https://git-scm.com/download/win)
> - **macOS**: `brew install git`, or `xcode-select --install`
> - **Ubuntu/Debian**: `sudo apt install git`

### Also required: a C compiler

ENTRAP is built with Cython — `pip install` will **compile C code from source**, not download a prebuilt binary.

| OS | Compiler needed |
|---|---|
| Linux | `gcc` (if missing: `sudo apt install build-essential`) |
| macOS | Xcode Command Line Tools (`xcode-select --install`) |
| Windows | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

---

## 2. Install ENTRAP

```bash
pip install git+https://github.com/AkmalHusain2003/ENTRAP_Cython.git
```

To pin a specific commit (recommended for reproducibility):

```bash
pip install git+https://github.com/AkmalHusain2003/ENTRAP_Cython.git@1bf3d01b24afece25ad584b31e0c59e9089c1ee5
```

Requires **Python ≥ 3.8**. Dependencies (numpy, scikit-learn, scipy, hdbscan, ripser, kneed, joblib, matplotlib) install automatically.

**Verify:**

```python
import entrap
print(entrap.__version__)   # -> "1.0.0"
```

---

## 3. Usage

### Basic

```python
import numpy as np
from entrap import ENTRAP

X = np.random.randn(500, 2)   # n_samples x n_features

model = ENTRAP(min_cluster_size=30, metric='euclidean')
model.fit(X)

labels = model.labels_        # -1 = noise, same convention as HDBSCAN
print(model.get_summary())
```

Or directly:

```python
labels = model.fit_predict(X)
```

### `ENTRAP` parameters

| Parameter | Default | Description |
|---|---|---|
| `min_cluster_size` | `30` | Passed to HDBSCAN. |
| `min_samples` | `None` | Passed to HDBSCAN; defaults to `min_cluster_size` if `None`. |
| `ridge_epsilon` | `1e-6` | Internal covariance regularization. |
| `metric` | `'euclidean'` | See supported metrics below. |
| `metric_params` | `None` | Extra dict params for the chosen metric (e.g. `{'p': 3}`). |
| `use_memmap` | `True` | Stored as an attribute; has no effect on `fit()` behavior in this version. |
| `enable_tracking` | `False` | If `True`, enables diagnostic tracking (see below). |
| `n_jobs` | `-1` | Parallel workers. `-1` = all CPU cores. |

### Supported metrics

```
euclidean, manhattan, cityblock, minkowski, chebyshev,
cosine, correlation, hamming, jaccard, canberra,
braycurtis, mahalanobis, seuclidean, sqeuclidean
```

A custom Python callable is also accepted.

### Accessing full results

```python
result = model.result_

result.labels             # same as model.labels_
result.probabilities      # HDBSCAN cluster membership probabilities
result.noise_rescued      # number of rescued noise points
result.execution_time     # seconds
result.n_clusters         # final cluster count
result.cluster_stats      # per-cluster stats
result.hyperparameters    # hyperparameters used
result.tracker            # tracker object, only if enable_tracking=True
```

### Diagnostics (optional)

```python
model = ENTRAP(min_cluster_size=30, enable_tracking=True)
model.fit(X)

model.list_tracked_clusters()
model.plot_entropy_curve(cluster_id=0)
model.plot_rejected_analysis(cluster_id=0)
model.plot_comparison(cluster_ids=[0, 1, 2])

accepted = model.get_accepted_candidates(cluster_id=0)
rejected = model.get_rejected_candidates(cluster_id=0)
summary = model.export_entropy_summary()
```

All `plot_*` methods accept an optional `save_path` to save the figure.

> Calling these without `enable_tracking=True` at `fit()` time raises a `ValueError`.

---

## 4. Install Troubleshooting

**`error: Microsoft Visual C++ 14.0 or greater is required` (Windows)**
→ Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), select "Desktop development with C++".

**`fatal error: Python.h: No such file or directory` (Linux)**
→ `sudo apt install python3-dev`

**`git: command not found`**
→ See [§1](#1-prerequisite-install-git).

**Install feels slow**
→ Normal — 10 `.pyx` modules are compiled to C then to native binaries on first install.
