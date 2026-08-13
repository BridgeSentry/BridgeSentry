import argparse

import torch

from config.constants import Bridge
from dataset_generator.generator import GraphDatasetGenerator
from dataset_generator.dataset_output import resolve_dataset_and_reports_root
from dataset_generator.types import DATASET_TYPE
from utils.utils import get_enum_instance

from dataset_generator.model_training import train
from dataset_generator.model_evaluation import evaluate
from dataset_generator.inference_timing import measure_inference_time

class Cli:
    CLASS_NAME = "Cli"

    def generate_graph_dataset(args):
        bridges = [get_enum_instance(Bridge, bridge) for bridge in args.bridge] if args.bridge else list(Bridge)
        dataset_type = args.dataset
        output_folder = args.data_dir
        force_reload = args.force_reload

        # Create generator file
        generator = GraphDatasetGenerator(bridges, dataset_type, output_folder, force_reload)
        generator.generate_graph_dataset()

    def train_model(args):
        dataset_type = args.dataset
        dataset_path, reports_root, force_reload = resolve_dataset_and_reports_root(
            dataset_type=dataset_type,
            load_data=args.load_data,
            new_dataset_tags=args.new_dataset,
            force_reload=args.force_reload,
        )
        model_args = {
            "first_layer_channels": args.first_layer_channels,
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "input_drop": args.input_drop,
            "att_drop": args.att_drop,
            "n_fp_layers": args.n_fp_layers,
            "n_mlp_layers": args.n_mlp_layers,
            "act": args.activation,
            "residual": args.residual,
            "pooling": args.pooling,
            "rm_semantic_fusion": args.rm_semantic_fusion,
            "learning_rate": args.learning_rate,
            "dme_threshold": args.dme_threshold,
        }

        kwargs = {
            "gpu": args.device,
            "k_folds": args.kfolds,
            "num_epochs": args.num_epochs,
            "early_stopping": args.early_stopping,
            "num_workers": args.num_workers,
            "augment_factor": args.augment_factor,
            "run_name_prefix": Cli._build_run_name_prefix(args),
            "random_seed": args.random_seed,
            "test_split": args.test_split,
        }

        return train(
            tags=args.tags,
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            force_reload=force_reload,
            model_args=model_args,
            reports_root=reports_root,
            **kwargs
        )

    def evaluate_model(args):
        evaluate(
            load_data=args.load_data,
            checkpoint_dir=args.checkpoint_data,
            tags=args.tags,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_reload=args.force_reload,
        )

    def time_inference(args):
        measure_inference_time(
            load_data=args.load_data,
            checkpoint_dir=args.checkpoint_data,
            fold=args.fold,
            tags=args.tags,
            device=args.device,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
            batch_sizes=[int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()],
            num_workers=args.num_workers,
            force_reload=args.force_reload,
        )

    def _build_run_name_prefix(args) -> str:
        """Build a short folder-name prefix from CLI args.

        Always includes epochs and augment_factor. Any other param that differs
        from its argparse default is appended as _{shortname}{value}.
        """
        # Params always included in the name
        parts = [
            f"epochs{args.num_epochs}",
            f"augment{args.augment_factor}",
        ]

        # Params included only when non-default
        non_default_params = [
            ("kfolds",              5,     "k"),
            ("learning_rate",       0.01,  "lr"),
            ("dme_threshold",       0.5,   "dme"),
            ("first_layer_channels",128,   "fl"),
            ("hidden_channels",     64,    "hc"),
            ("dropout",             0.5,   "do"),
            ("input_drop",          0.0,   "id"),
            ("att_drop",            0.0,   "ad"),
            ("n_fp_layers",         2,     "fp"),
            ("n_mlp_layers",        2,     "mlp"),
            ("activation",          "relu","act"),
            ("pooling",             "mean","pool"),
            ("residual",            False, "res"),
            ("rm_semantic_fusion",  False, "nosf"),
            ("early_stopping",      None,  "es"),
            ("random_seed",         42,    "seed"),
            ("test_split",          0.15,  "ts"),
        ]
        for attr, default, short in non_default_params:
            value = getattr(args, attr)
            if value != default:
                if isinstance(value, bool):
                    parts.append(short)
                else:
                    parts.append(f"{short}{value}")

        return "train_" + "_".join(parts)

    def cli():
        parser = argparse.ArgumentParser(description="Bridge Defender CLI")
        subparsers = parser.add_subparsers(
            title="Actions", description="Available actions", dest="action"
        )

        # Generate torch-geometric graph dataset
        graph_dataset_parser = subparsers.add_parser(
            "graph-dataset",
            help="Generate graph dataset for machine learning",
        )
        graph_dataset_parser.add_argument(
            "--dataset",
            type=str,
            default=DATASET_TYPE.SINGLE,
            choices=[
                DATASET_TYPE.MIXED, 
                DATASET_TYPE.SINGLE, 
                DATASET_TYPE.CCTX
            ],
            required=True,
            help="The type of graph dataset to generate (default: mixed)",
        )
        graph_dataset_parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="The output directory where the generated graph dataset will be saved.",
        )
        graph_dataset_parser.add_argument(
            "--bridge",
            choices=[bridge.value for bridge in Bridge],
            nargs="+",
            required=False,
            help="The bridge(s) to generate the graph dataset for. If not specified, datasets for all bridges will be generated.",
        )
        graph_dataset_parser.add_argument(
            "--force-reload",
            action="store_true",
            help="Force reload the graph dataset even if it already exists (default: False)",
        )
        graph_dataset_parser.set_defaults(func=Cli.generate_graph_dataset)

        training_parser = subparsers.add_parser(
            "train",
            help="Train the BridgeDefender model on a graph dataset",
        )
        training_parser.add_argument(
            "--tags",
            type=str,
            default=None,
            help="Optional comma-separated tags to append to the training run folder name for identification",
        )
        training_parser.add_argument(
            "--dataset",
            type=str,
            default="mixed",
            choices=[
                DATASET_TYPE.MIXED, 
                DATASET_TYPE.SINGLE, 
                DATASET_TYPE.CCTX
            ],
            required=True,
            help="The type of graph dataset to train on",
        )
        dataset_group = training_parser.add_mutually_exclusive_group(required=True)
        dataset_group.add_argument(
            "--load-data",
            type=str,
            help="The directory where the graph dataset is located.",
        )
        dataset_group.add_argument(
            "--new-dataset",
            nargs='?',
            const='',
            default=None,
            help="Create a new dataset. It will be stored in a new directory inside the outputs/ folder. Optionally pass comma-separated tags for folder naming."
        )
        training_parser.add_argument(
            "-f", "--force-reload",
            action="store_true",
            help="Force reload the graph dataset even if it already exists (default: False)",
        )
        training_parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="Number of worker processes for data loading (default: 0, i.e., no multiprocessing)",
        )
        training_parser.add_argument(
            "--kfolds",
            type=int,
            default=5,
            help="Number of folds for K-Fold cross-validation (default: 5)",
        )
        training_parser.add_argument(
            "--num-epochs",
            type=int,
            default=100,
            help="Number of training epochs per fold (default: 100)",
        )
        training_parser.add_argument(
            "--early-stopping",
            type=int,
            default=None,
            help="Whether to use early stopping based on validation loss, and the patience value (default: None)",
        )
        training_parser.add_argument(
            "--dme-threshold",
            type=float,
            default=0.5,
            help="Threshold for the Differential Meta-Path Extraction (default: 0.5)",
        )
        training_parser.add_argument(
            "--test-split",
            type=float,
            default=0.15,
            help="Fraction of the real-graph dataset held out as a final test set "
                 "(default: 0.15). Set to 0 to use the entire dataset for "
                 "training/K-fold validation and skip test-set evaluation entirely.",
        )
        training_parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.01,
            help="Learning rate for the optimizer (default: 0.01)",
        )
        training_parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda"],
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use for training (default: 'cuda' if available, otherwise 'cpu')",
        )
        training_parser.add_argument(
            "--first-layer-channels",
            type=int,
            default=128,
            help="Number of channels in the first layer of SeHGNNConv (default: 128)",
        )
        training_parser.add_argument(
            "--hidden-channels",
            type=int,
            default=64,
            help="Number of hidden channels in SeHGNNConv and MLP layers (default: 64)",
        )
        training_parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout rate for SeHGNNConv and MLP layers (default: 0.5)",
        )
        training_parser.add_argument(
            "--input-drop",
            type=float,
            default=0.0,
            help="Dropout rate for input features (default: 0.0)",
        )
        training_parser.add_argument(
            "--att-drop",
            type=float,
            default=0.0,
            help="Dropout rate for attention weights in the semantic transformer (default: 0.0)",
        )
        training_parser.add_argument(
            "--n-fp-layers",
            type=int,
            default=2,
            help="Number of feature propagation layers in SeHGNNConv (default: 2)",
        )
        training_parser.add_argument(
            "--n-mlp-layers",
            type=int,
            default=2,
            help="Number of layers in the final MLP classifier (default: 2)",
        )
        training_parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            choices=["relu", "leaky_relu", "sigmoid", "none"],
            help="Activation function to use in SeHGNNConv and MLP layers (default: relu)",
        )
        training_parser.add_argument(
            "--residual",
            action="store_true",
            help="If used, enables residual connections in SeHGNNConv layers",
        )
        training_parser.add_argument(
            "--pooling",
            type=str,
            default="mean",
            choices=["mean", "max", "sum", "mean_max"],
            help="Pooling method to use for aggregating node embeddings into type-level embeddings (default: mean)",
        )
        training_parser.add_argument(
            "--rm-semantic-fusion",
            action="store_true",
            help="Ablation: if used, removes the Semantic Fusion Transformer from SeHGNN and "
                 "passes the feature projection output directly to the pooling step",
        )
        training_parser.add_argument(
            "--augment-factor",
            type=int,
            default=0,
            help="Number of augmented copies per anomaly graph generated at training time (0 = disabled, default: 0)",
        )
        training_parser.add_argument(
            "--random-seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42)",
        )
        training_parser.set_defaults(func=Cli.train_model)

        eval_parser = subparsers.add_parser(
            "eval",
            help="Evaluate saved fold checkpoints (from a training run) against a dataset",
        )
        eval_parser.add_argument(
            "--load-data",
            type=str,
            required=True,
            help="The directory where the graph dataset to evaluate is located.",
        )
        eval_parser.add_argument(
            "--checkpoint-data",
            type=str,
            required=True,
            help="The directory containing the fold checkpoints to evaluate (a training run's 'models/' folder).",
        )
        eval_parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda"],
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use for evaluation (default: 'cuda' if available, otherwise 'cpu')",
        )
        eval_parser.add_argument(
            "--tags",
            type=str,
            default=None,
            help="Comma-separated tags to include in the evaluation report folder name.",
        )
        eval_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size to use for evaluation (default: 32)",
        )
        eval_parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="Number of worker processes for data loading (default: 0, i.e., no multiprocessing)",
        )
        eval_parser.add_argument(
            "-f", "--force-reload",
            action="store_true",
            help="Force reload the graph dataset even if it already exists (default: False)",
        )
        eval_parser.set_defaults(func=Cli.evaluate_model)

        timing_parser = subparsers.add_parser(
            "time-infer",
            help="Measure per-transaction inference time (preprocessing + forward pass) of a saved fold checkpoint",
        )
        timing_parser.add_argument(
            "--load-data",
            type=str,
            required=True,
            help="The directory where the graph dataset to time is located.",
        )
        timing_parser.add_argument(
            "--checkpoint-data",
            type=str,
            required=True,
            help="The directory containing the fold checkpoints to time (a training run's 'models/' folder).",
        )
        timing_parser.add_argument(
            "--fold",
            type=int,
            default=1,
            help="Which fold checkpoint (1-indexed) to load and time; all folds share the same "
                 "architecture so timing one is representative (default: 1)",
        )
        timing_parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda"],
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device to use for timing (default: 'cuda' if available, otherwise 'cpu')",
        )
        timing_parser.add_argument(
            "--tags",
            type=str,
            default=None,
            help="Comma-separated tags to include in the timing report folder name.",
        )
        timing_parser.add_argument(
            "--batch-sizes",
            type=str,
            default="1,32",
            help="Comma-separated batch sizes to time the forward pass at (default: '1,32')",
        )
        timing_parser.add_argument(
            "--warmup-batches",
            type=int,
            default=3,
            help="Number of batches to run un-timed before recording, per batch size (default: 3)",
        )
        timing_parser.add_argument(
            "--repeats",
            type=int,
            default=3,
            help="Number of times to loop the dataset before timing, so small test sets still yield "
                 "enough post-warmup batches at larger batch sizes (default: 3)",
        )
        timing_parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="Number of worker processes for data loading (default: 0, i.e., no multiprocessing)",
        )
        timing_parser.add_argument(
            "-f", "--force-reload",
            action="store_true",
            help="Force reload the graph dataset even if it already exists (default: False)",
        )
        timing_parser.set_defaults(func=Cli.time_inference)

        args = parser.parse_args()
        if args.action:
            args.func(args)
        else:
            parser.print_help()