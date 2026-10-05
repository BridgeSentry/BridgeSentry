# BridgeSentry

This repository contains the code related to the paper **"BridgeSentry: Meta-Path-Guided Graph Learning for
Cross-chain Bridge Attack Detection"**.

BridgeSentry detects anomalous (attack-related) activity in cross-chain bridges
by classifying the transactions that pass through them as **graphs**.

Each blockchain transaction (or each full cross-chain transaction, spanning the
source chain, off-chain processing and the destination chain) is turned into a
**heterogeneous graph**. Nodes represent users, routers, tokens, other accounts,
log events and validators. Edges represent transactions, token transfers, token
approvals, function calls, log relations and cross-chain relations. A
heterogeneous graph neural network then classifies each graph as `normal` or
`anomaly`.

The graph data comes from
[XChainDataGen](https://github.com/XChainDataGen/XChainDataGen), which extracts
bridge activity from the blockchains and stores the graph nodes and edges in a
PostgreSQL database. BridgeSentry reads that database, builds a
[PyTorch Geometric](https://pyg.org/) dataset, and trains and evaluates the
detector on it.

## Branches: BridgeSentry-C vs. BridgeSentry-SAD

There are two variants of the detector, and each one lives on its own branch:

| Variant              | Branch                                                                                             | Approach                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **BridgeSentry-C**   | [`bridgesentry-c`](https://github.com/BridgeSentry/BridgeSentry/tree/bridgesentry-c) (this branch) | Supervised **classifier**: a SeHGNN-based encoder followed by an MLP head trained with class-weighted cross-entropy. |
| **BridgeSentry-SAD** | [`bridgesentry-sad`](https://github.com/BridgeSentry/BridgeSentry/tree/bridgesentry-sad)           | Semi-supervised anomaly detection: the same graph encoder, trained with a **Deep SAD** objective.                    |

> [!WARNING]
> **This branch only contains the BridgeSentry-C logic.** To use
> BridgeSentry-SAD, switch to the
> [`bridgesentry-sad`](https://github.com/BridgeSentry/BridgeSentry/tree/bridgesentry-sad)
> branch:
>
> ```bash
> git checkout bridgesentry-sad
> ```

### How BridgeSentry-C works

1. **Graph dataset:** transaction graphs are loaded from the XChainDataGen
   database and encoded as PyG `HeteroData` objects with per-node-type
   features.
2. **Data split:** real graphs are split into a stratified held-out test set
   (`--test-split`, 15% by default) and a train/validation pool that is used
   for stratified K-fold cross-validation (`--kfolds`).
3. **Differential Meta-path Extraction (DME):** meta-paths of up to 5 hops are
   counted, and those whose occurrence differs most between normal and anomalous
   graphs (above `--dme-threshold`) are kept. Their aggregated features are
   pre-computed once per graph.
4. **Model:** for each node type, a SeHGNN encoder projects the meta-path
   features, fuses them with a semantic fusion transformer (this can be removed
   for ablation with `--rm-semantic-fusion`), and pools them into a type-level
   embedding (`--pooling`). The type-level embeddings are concatenated and
   passed to an MLP classifier.
5. **Training:** the loss is class-weighted cross-entropy, which counters the
   strong normal/anomaly imbalance. Anomaly graphs can optionally be augmented
   (`--augment-factor`). Augmented copies only ever go into the training
   partition of the fold that holds their source graph.

### Dataset types

| `--dataset` | Each graph is...                                                                                                                                                      |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `single`    | a single on-chain transaction (labels: `normal` / `anomaly`)                                                                                                          |
| `cctx`      | a full cross-chain transaction (labels: `normal` / `anomaly_source` / `anomaly_offchain` / `anomaly_destination`, collapsed to normal vs. anomaly for classification) |

## Setup

### Prerequisites

- **Python 3.14** (recommended; you can use `pyenv` to manage versions).
- **PostgreSQL** with a database populated by **XChainDataGen**. BridgeSentry
  does not extract blockchain data itself. It reads the `graph_mapping_blockchain`,
  `graph_mapping_cross_chain`, `graph_nodes` and `graph_edges` tables that
  XChainDataGen creates. Follow the XChainDataGen instructions to install it,
  then, for each bridge you want to analyse, run:
  1. `extract`: extract the bridge's on-chain data;
  2. `generate`: match the cross-chain transactions;
  3. `generate_graph_data`: build the heterogeneous graph nodes and edges.

### Installation

1. Clone the repository and check out this branch:

   ```bash
   git clone https://github.com/BridgeSentry/BridgeSentry.git
   cd BridgeSentry
   git checkout bridgesentry-c
   ```

2. Create and activate a virtual environment with Python 3.14:

   ```bash
   python3.14 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the environment variables. Copy the template and fill it in:

   ```bash
   cp template.env .env
   ```

   Make sure to set `DATABASE_URL` to the SQLAlchemy URL of the **same** PostgreSQL database XChainDataGen writes to, e.g. `postgresql://user:password@localhost:5432/db_app`. Do not use `db` as the host of the database, as it is only valid inside the Docker network. Use `localhost` or the actual hostname instead.

## Usage

All functionality is exposed through `python main.py`:

```
usage: main.py [-h] {graph-dataset,train,eval,time-infer} ...

Bridge Defender CLI

options:
  -h, --help            show this help message and exit

Actions:
  Available actions

  {graph-dataset,train,eval,time-infer}
    graph-dataset       Generate graph dataset for machine learning
    train               Train the BridgeDefender model on a graph dataset
    eval                Evaluate saved fold checkpoints (from a training run) against a dataset
    time-infer          Measure per-transaction inference time (preprocessing + forward pass) of a
                        saved fold checkpoint
```

A typical workflow:

```bash
# 1. Build a dataset from the database and train on it, in a single step
python main.py train --dataset single --new-dataset my,experiment --num-epochs 100 --early-stopping 15 --learning-rate 0.005 --augment-factor 0 --rm-semantic-fusion
# If the dataset already exists, you can reuse it with:
python main.py train --dataset single --load-data outputs/<YYYYMMDD>-single-my-experiment/_dataset --num-epochs 100 --early-stopping 15 --learning-rate 0.005 --augment-factor 0 --rm-semantic-fusion

# 2. Evaluate the trained fold checkpoints on a (possibly different) dataset
python main.py eval \
    --load-data outputs/<YYYYMMDD>-single-my-experiment/_dataset \
    --checkpoint-data outputs/<YYYYMMDD>-single-my-experiment/<train_run>/models

# 3. Measure inference latency of one of the checkpoints
python main.py time-infer \
    --load-data outputs/<YYYYMMDD>-single-my-experiment/_dataset \
    --checkpoint-data outputs/<YYYYMMDD>-single-my-experiment/<train_run>/models \
    --batch-sizes 1,32
```

### `train`: train BridgeSentry-C

Trains one model per fold using stratified K-fold cross-validation. If the
test split is non-zero, every fold's model is also evaluated on the held-out
test set.

You must pass exactly one of these two options:

- `--load-data <dir>`: train on an existing dataset directory. The training
  report is written to the **parent** of that directory.
- `--new-dataset [tags]`: build a new dataset from the database in
  `outputs/<YYYYMMDD>-<dataset>-<tags>/_dataset/` and train on it. If that
  dataset already exists, it is reused unless you pass `-f/--force-reload`.

```
usage: main.py train [-h] [--tags TAGS] --dataset {mixed,single,cctx} (--load-data LOAD_DATA |
                     --new-dataset [NEW_DATASET]) [-f] [--num-workers NUM_WORKERS]
                     [--kfolds KFOLDS] [--num-epochs NUM_EPOCHS] [--early-stopping EARLY_STOPPING]
                     [--dme-threshold DME_THRESHOLD] [--test-split TEST_SPLIT]
                     [--learning-rate LEARNING_RATE] [--device {cpu,cuda}]
                     [--first-layer-channels FIRST_LAYER_CHANNELS]
                     [--hidden-channels HIDDEN_CHANNELS] [--dropout DROPOUT]
                     [--input-drop INPUT_DROP] [--att-drop ATT_DROP] [--n-fp-layers N_FP_LAYERS]
                     [--n-mlp-layers N_MLP_LAYERS] [--activation {relu,leaky_relu,sigmoid,none}]
                     [--residual] [--pooling {mean,max,sum,mean_max}]
                     [--rm-semantic-fusion | --no-rm-semantic-fusion]
                     [--augment-factor AUGMENT_FACTOR] [--random-seed RANDOM_SEED]

options:
  -h, --help            show this help message and exit
  --tags TAGS           Optional comma-separated tags to append to the training run folder name
                        for identification
  --dataset {mixed,single,cctx}
                        The type of graph dataset to train on
  --load-data LOAD_DATA
                        The directory where the graph dataset is located.
  --new-dataset [NEW_DATASET]
                        Create a new dataset. It will be stored in a new directory inside the
                        outputs/ folder. Optionally pass comma-separated tags for folder naming.
  -f, --force-reload    Force reload the graph dataset even if it already exists (default: False)
  --num-workers NUM_WORKERS
                        Number of worker processes for data loading (default: 0, i.e., no
                        multiprocessing)
  --kfolds KFOLDS       Number of folds for K-Fold cross-validation (default: 5)
  --num-epochs NUM_EPOCHS
                        Number of training epochs per fold (default: 100)
  --early-stopping EARLY_STOPPING
                        Whether to use early stopping based on validation loss, and the patience
                        value (default: 15, disable with 'none' or 0)
  --dme-threshold DME_THRESHOLD
                        Threshold for the Differential Meta-Path Extraction (default: 0.5)
  --test-split TEST_SPLIT
                        Fraction of the real-graph dataset held out as a final test set (default:
                        0.15). Set to 0 to use the entire dataset for training/K-fold validation
                        and skip test-set evaluation entirely.
  --learning-rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.005)
  --device {cpu,cuda}   Device to use for training (default: 'cuda' if available, otherwise 'cpu')
  --first-layer-channels FIRST_LAYER_CHANNELS
                        Number of channels in the first layer of SeHGNNConv (default: 128)
  --hidden-channels HIDDEN_CHANNELS
                        Number of hidden channels in SeHGNNConv and MLP layers (default: 64)
  --dropout DROPOUT     Dropout rate for SeHGNNConv and MLP layers (default: 0.5)
  --input-drop INPUT_DROP
                        Dropout rate for input features (default: 0.0)
  --att-drop ATT_DROP   Dropout rate for attention weights in the semantic transformer (default:
                        0.0)
  --n-fp-layers N_FP_LAYERS
                        Number of feature propagation layers in SeHGNNConv (default: 2)
  --n-mlp-layers N_MLP_LAYERS
                        Number of layers in the final MLP classifier (default: 2)
  --activation {relu,leaky_relu,sigmoid,none}
                        Activation function to use in SeHGNNConv and MLP layers (default: relu)
  --residual            If used, enables residual connections in SeHGNNConv layers
  --pooling {mean,max,sum,mean_max}
                        Pooling method to use for aggregating node embeddings into type-level
                        embeddings (default: mean)
  --rm-semantic-fusion, --no-rm-semantic-fusion
                        Ablation: if used, removes the Semantic Fusion Transformer from SeHGNN and
                        passes the feature projection output directly to the pooling step
                        (default: enabled; use --no-rm-semantic-fusion to keep the transformer)
  --augment-factor AUGMENT_FACTOR
                        Number of augmented copies per anomaly graph generated at training time (0
                        = disabled, default: 0)
  --random-seed RANDOM_SEED
                        Random seed for reproducibility (default: 42)
```

Each training run creates a folder of the format
`train_epochs<N>_augment<N>[_<non-default params>][_<tags>]_<timestamp>/`. The folder contains the following files and subfolders:

| File / folder                                 | Contents                                                                           |
| --------------------------------------------- | ---------------------------------------------------------------------------------- |
| `params.yaml`                                 | All hyperparameters, dataset information and statistics for the run                |
| `summary.txt`                                 | Human-readable summary of the results                                              |
| `dme_report.csv`                              | Meta-paths selected by DME and their differential values                           |
| `val_<metric>.csv`, `charts/val_<metric>.svg` | Per-epoch validation metrics for each fold                                         |
| `val_best_metrics.csv`                        | Validation metrics at each fold's best epoch (use these for hyperparameter tuning) |
| `test_metrics.csv`                            | Held-out test-set metrics for each fold (only when `--test-split` > 0)             |
| `fold_durations.csv`                          | Training time per fold                                                             |
| `confusion_matrices/`                         | Confusion matrices for each fold, overall and per bridge                           |
| `models/fold_<k>.pt`                          | Fold checkpoints, used by `eval` and `time-infer`                                  |

>[!WARNING] IMPORTANT!
>**Default argument deviations used in the paper's experiments:**
>
>For BridgeSentry-C:
>
>- *Single-chain scope:* Use the `single` dataset type, specifying the new or current set with `--new-dataset` or `--load-data`. Unless stated in other experiments, the other default arguments are used.
>- *Cross-chain scope:* Use the `cctx` dataset type, specifying the new or current set with `--new-dataset` or `--load-data`. **Learning rate should be set to 0.002 (`--learning-rate 0.002`).** Unless stated in other experiments, the other default arguments are used.


### `eval`: evaluate trained checkpoints

Loads every `fold_*.pt` checkpoint in `--checkpoint-data` and evaluates it on
the dataset in `--load-data`. The checkpoints contain the normalisation
statistics and DME meta-paths from training, so you can evaluate on a
different dataset (e.g. new bridges or attacks) without leaking information.
Results go to `inference_[<tags>_]<timestamp>/`, next to the `models/` folder:
`eval_stats.csv` (metrics per fold), `infer_report.csv` (predictions per
graph), and confusion matrices.

```
usage: main.py eval [-h] --load-data LOAD_DATA --checkpoint-data CHECKPOINT_DATA
                    [--device {cpu,cuda}] [--tags TAGS] [--batch-size BATCH_SIZE]
                    [--num-workers NUM_WORKERS] [-f]

options:
  -h, --help            show this help message and exit
  --load-data LOAD_DATA
                        The directory where the graph dataset to evaluate is located.
  --checkpoint-data CHECKPOINT_DATA
                        The directory containing the fold checkpoints to evaluate (a training
                        run's 'models/' folder).
  --device {cpu,cuda}   Device to use for evaluation (default: 'cuda' if available, otherwise
                        'cpu')
  --tags TAGS           Comma-separated tags to include in the evaluation report folder name.
  --batch-size BATCH_SIZE
                        Batch size to use for evaluation (default: 32)
  --num-workers NUM_WORKERS
                        Number of worker processes for data loading (default: 0, i.e., no
                        multiprocessing)
  -f, --force-reload    Force reload the graph dataset even if it already exists (default: False)
```

### `time-infer`: measure inference latency

Measures the time BridgeSentry-C takes per transaction, using a coding process that is similar to `eval`. Results go to
`timing_[<tags>_]<timestamp>/timing_report.csv`, next to the `models/` folder.

```
usage: main.py time-infer [-h] --load-data LOAD_DATA --checkpoint-data CHECKPOINT_DATA
                          [--fold FOLD] [--device {cpu,cuda}] [--tags TAGS]
                          [--batch-sizes BATCH_SIZES] [--warmup-batches WARMUP_BATCHES]
                          [--repeats REPEATS] [--num-workers NUM_WORKERS] [-f]

options:
  -h, --help            show this help message and exit
  --load-data LOAD_DATA
                        The directory where the graph dataset to time is located.
  --checkpoint-data CHECKPOINT_DATA
                        The directory containing the fold checkpoints to time (a training run's
                        'models/' folder).
  --fold FOLD           Which fold checkpoint (1-indexed) to load and time; all folds share the
                        same architecture so timing one is representative (default: 1)
  --device {cpu,cuda}   Device to use for timing (default: 'cuda' if available, otherwise 'cpu')
  --tags TAGS           Comma-separated tags to include in the timing report folder name.
  --batch-sizes BATCH_SIZES
                        Comma-separated batch sizes to time the forward pass at (default: '1,32')
  --warmup-batches WARMUP_BATCHES
                        Number of batches to run un-timed before recording, per batch size
                        (default: 3)
  --repeats REPEATS     Number of times to loop the dataset before timing, so small test sets
                        still yield enough post-warmup batches at larger batch sizes (default: 3)
  --num-workers NUM_WORKERS
                        Number of worker processes for data loading (default: 0, i.e., no
                        multiprocessing)
  -f, --force-reload    Force reload the graph dataset even if it already exists (default: False)
```
