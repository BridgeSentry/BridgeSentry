import argparse

from config.constants import Bridge
from dataset_generator.generator import GraphDatasetGenerator
from utils.utils import get_enum_instance

from dataset_generator.model_training import train

class Cli:
    CLASS_NAME = "Cli"

    def generate_graph_dataset(args):
        bridges = [get_enum_instance(Bridge, bridge) for bridge in args.bridge] if args.bridge else list(Bridge)
        output_folder = args.out

        # Create generator file
        generator = GraphDatasetGenerator(bridges, output_folder)
        generator.generate_graph_dataset()

    def train_model(args):
        dataset_path = args.dataset
        
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
        }

        kwargs = {
            "gpu": not args.no_gpu,
            "kfolds": args.kfolds,
            "num_workers": args.num_workers
        }

        train(
            dataset_path=dataset_path,
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
            "--bridge",
            choices=[bridge.value for bridge in Bridge],
            nargs="+",
            required=False,
            help="The bridge(s) to generate the graph dataset for. If not specified, datasets for all bridges will be generated.",
        )
        graph_dataset_parser.add_argument(
            "--out",
            type=str,
            required=True,
            help="The output directory where the generated graph dataset will be saved.",
        )
        graph_dataset_parser.set_defaults(func=Cli.generate_graph_dataset)

        training_parser = subparsers.add_parser(
            "train",
            help="Train the BridgeDefender model on a graph dataset",
        )
        training_parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="Path to the graph dataset to train on (should be the output folder of the graph-dataset action)",
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
            "--no-gpu",
            action="store_true",
            help="Disable GPU usage for training (default: False)",
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
        training_parser.set_defaults(func=Cli.train_model)

        args = parser.parse_args()
        if args.action:
            args.func(args)
        else:
            parser.print_help()