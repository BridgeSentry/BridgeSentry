import argparse

from config.constants import Bridge
from dataset_generator.generator import GraphDatasetGenerator
from utils.utils import get_enum_instance


class Cli:
    CLASS_NAME = "Cli"

    def generate_graph_dataset(args):
        bridges = [get_enum_instance(Bridge, bridge) for bridge in args.bridge] if args.bridge else list(Bridge)
        output_folder = args.out

        # Create generator file
        generator = GraphDatasetGenerator(bridges, output_folder)
        generator.generate_graph_dataset()

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

        args = parser.parse_args()
        if args.action:
            args.func(args)
        else:
            parser.print_help()