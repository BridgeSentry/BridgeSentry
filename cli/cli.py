import argparse

import torch

from config.constants import Bridge
from dataset_generator.generator import GraphDatasetGenerator
from dataset_generator.types import DATASET_TYPE
from utils.utils import get_enum_instance

from dataset_generator.model_training import train

class Cli:
    CLASS_NAME = "Cli"

    def generate_graph_dataset(args):
        bridges = [get_enum_instance(Bridge, bridge) for bridge in args.bridge] if args.bridge else list(Bridge)
        dataset_type = args.dataset
        output_folder = args.data_dir
        force_reload = args.force_reload

        # Create generator file
        augment_factor = args.augment_factor
        generator = GraphDatasetGenerator(bridges, dataset_type, output_folder, force_reload, augment_factor=augment_factor)
        generator.generate_graph_dataset()

    def train_model(args):
        dataset_type = args.dataset
        dataset_path = args.data_dir
        force_reload = args.force_reload
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
            "learning_rate": args.learning_rate,
        }

        kwargs = {
            "gpu": args.device,
            "kfolds": args.kfolds,
            "num_epochs": args.num_epochs,
            "early_stopping": args.early_stopping,
            "num_workers": args.num_workers,
            "augment_factor": args.augment_factor,
        }

        train(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            force_reload=force_reload,
            model_args=model_args,
            **kwargs
        )

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
        graph_dataset_parser.add_argument(
            "--augment-factor",
            type=int,
            default=0,
            help="Number of augmented copies per anomaly graph saved in the dataset (0 = disabled, default: 0)",
        )
        graph_dataset_parser.set_defaults(func=Cli.generate_graph_dataset)

        training_parser = subparsers.add_parser(
            "train",
            help="Train the BridgeDefender model on a graph dataset",
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
        training_parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="The directory where the graph dataset is located.",
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
            action="store_true",
            help="Whether to use early stopping based on validation loss (default: True)",
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
            help="Number of channels in the first layer of SeHGNNConv",
        )
        training_parser.add_argument(
            "--hidden-channels",
            type=int,
            default=64,
            help="Number of hidden channels in SeHGNNConv and MLP layers",
        )
        training_parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout rate for SeHGNNConv and MLP layers",
        )
        training_parser.add_argument(
            "--input-drop",
            type=float,
            default=0.0,
            help="Dropout rate for input features",
        )
        training_parser.add_argument(
            "--att-drop",
            type=float,
            default=0.0,
            help="Dropout rate for attention weights in the semantic transformer",
        )
        training_parser.add_argument(
            "--n-fp-layers",
            type=int,
            default=2,
            help="Number of feature propagation layers in SeHGNNConv",
        )
        training_parser.add_argument(
            "--n-mlp-layers",
            type=int,
            default=2,
            help="Number of layers in the final MLP classifier",
        )
        training_parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            choices=["relu", "leaky_relu", "sigmoid", "none"],
            help="Activation function to use in SeHGNNConv and MLP layers",
        )
        training_parser.add_argument(
            "--residual",
            action="store_true",
            help="Whether to use residual connections in SeHGNNConv layers",
        )
        training_parser.add_argument(
            "--pooling",
            type=str,
            default="mean",
            choices=["mean", "max", "sum"],
            help="Pooling method to use for aggregating node embeddings into type-level embeddings",
        )
        training_parser.add_argument(
            "--augment-factor",
            type=int,
            default=0,
            help="Additional runtime augmented copies per anomaly graph at training time (0 = use only pre-generated, default: 0)",
        )
        training_parser.set_defaults(func=Cli.train_model)

        args = parser.parse_args()
        if args.action:
            args.func(args)
        else:
            parser.print_help()