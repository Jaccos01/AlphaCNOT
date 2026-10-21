# AlphaCNOT: Learning CNOT Minimization with Model-Based Planning

A basic implementation of AlphaZero-style learning with the MCTX framework for
Monte Carlo Tree Search in JAX. The training code assumes that the agent has
access to the exact environment dynamics. It learns policy and value functions
from tree-search results rather than learning an environment model.

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are JAX, Haiku, MCTX, Chex, NumPy, SciPy, and tqdm.

## Pretrained models

The packaged models are full `.pkl` checkpoints. Each checkpoint contains its
network parameters and inference configuration; the loader extracts
`checkpoint["state"]["params"]` automatically.

Models are grouped first by cardinality and then by model family:

```text
trained_models/
├── Card_4/                 # 8 checkpoints
├── Card_5/                 # 8 checkpoints
├── Card_6/                 # 10 checkpoints
├── Card_7/                 # 10 checkpoints
└── Card_8/                 # 12 checkpoints
    ├── identity-inf/       # ALL, exact identity, informed reward
    ├── identity-mix/       # ALL, exact identity, mixed reward
    ├── perm-inf/           # ALL, permutation-relaxed, informed reward
    ├── perm-mix/           # ALL, permutation-relaxed, mixed reward
    ├── top-id/             # topology-specific, exact identity
    └── top-perm/           # topology-specific, permutation-relaxed
```

Checkpoint names follow
`alphacnot_<cardinality><topology>_<id|perm>.pkl`. In addition to `ALL`, the
packaged topology-specific checkpoints are `L` and `Y` for cardinality 4; `L`
and `T` for cardinality 5; `L`, `T`, and `Y` for cardinalities 6 and 7; and
`F`, `H`, `T1`, and `T2` for cardinality 8. The `8L` topology is not part of
this work. For cardinality 8, `T` is accepted as an alias for `T1`.

Files ending in `_perm.pkl` are **permutation-relaxed** models. Their goal is
any permutation matrix rather than the identity matrix, so a successful result
is equivalent up to an output-wire permutation. Files ending in `_id.pkl`
target the identity exactly.

## Loading a pretrained model

Use `load_alphacnot_model` to select a checkpoint by cardinality, topology,
reward, and terminal mode:

```python
from scripts.utils import load_alphacnot_model

model = load_alphacnot_model(
    cardinality=8,
    topology="ALL",
    source="trained_models",
    reward_type="mix",                    # "mix" or "inf" for ALL
    terminal_mode="permutation-relaxed",  # or "identity"
)
```

For topology-specific models, the loader selects `top-id` or `top-perm`; the
`reward_type` selector is only used by fully connected (`ALL`) models. Paths
are resolved relative to the repository, so loading does not depend on the
current working directory.

Locally trained legacy models stored as `.params` remain supported with
`source="user"`.

## Training a model

Start a new training run with:

```bash
python main_train.py \
  --n_qubits 5 \
  --topology ALL \
  --reward mix \
  --epochs 1000
```

Training output is written below `models/Card_<cardinality>/`.

## Gold Matrices

`Gold Matrices/` contains the frozen benchmark used to evaluate AlphaCNOT. It
is independent of the online random stream used during training, so different
models and synthesis modes can be compared on exactly the same inputs.

For each supported cardinality from 4 to 8, the repository provides 100 unique
and invertible binary parity matrices:

```text
Gold Matrices/
├── Card_4/matrix_000.npy ... matrix_099.npy
├── Card_5/matrix_000.npy ... matrix_099.npy
├── Card_6/matrix_000.npy ... matrix_099.npy
├── Card_7/matrix_000.npy ... matrix_099.npy
└── Card_8/matrix_000.npy ... matrix_099.npy
```

Each `.npy` file stores an `N x N` `float32` array whose entries are 0 or 1,
interpreted over GF(2). Such a parity matrix represents the reversible linear
transformation implemented by a CNOT circuit. "Gold" means that the test set is
fixed and shared: these files are evaluation instances, not model parameters,
training examples, or labels containing an optimal synthesis.

The same matrices can be used to compare exact-identity and
permutation-relaxed models; only their accepted terminal condition changes.
Keeping the inputs fixed makes results reproducible and enables fair, paired
comparisons between model families and topologies.

Load one matrix with NumPy as follows:

```python
import numpy as np

matrix = np.load("Gold Matrices/Card_6/matrix_000.npy")
```

## Validating a model

Validate a packaged permutation-relaxed checkpoint with:

```bash
python main_validation.py \
  --n_qubits 8 \
  --topology ALL \
  --source trained_models \
  --reward mix \
  --terminal_mode permutation-relaxed \
  --test 100 \
  --rep 10
```

Use `--terminal_mode identity` for an exact-identity model. When `--rep` is
greater than one, AlphaCNOT tries multiple syntheses for each matrix and keeps
the shortest successful result.

To validate a locally trained legacy model, use `--source user` together with
the corresponding cardinality, topology, and reward.

## Synthesizing a custom circuit

Open `synthesize_circuit.ipynb` and:

1. Select and load an AlphaCNOT model.
2. Define a CNOT circuit.
3. Convert it to its parity matrix.
4. Run AlphaCNOT synthesis.

## Citation

```bibtex
@article{cossio2026alphacnot,
  title={AlphaCNOT: Learning CNOT Minimization with Model-Based Planning},
  author={Cossio, Jacopo and Lizzio Bosco, Daniele and Romanello, Riccardo and Serra, Giuseppe and Piazza, Carla},
  journal={arXiv preprint arXiv:2604.13812},
  year={2026}
}
```
