from config.constants import Bridge
from dataset_generator.cctx_dataset import CrossChainTransactionsDataset
from dataset_generator.types import DATASET_TYPE
from dataset_generator.types import DATASET_CLASS
from utils.utils import CustomException, load_module
import os


class GraphDatasetGenerator:
    CLASS_NAME = "GraphDatasetGenerator"
    
    def __init__(self, bridges: list[Bridge], dataset_type: DATASET_TYPE, output_folder: str, force_reload: bool = False):
        self.bridges = bridges
        self.dataset_class = DATASET_CLASS[dataset_type]
        self.output_folder = output_folder
        self.force_reload = force_reload
        self.load_db_modules()

    def load_db_modules(self):
        function_name = "load_db_models"

        try:
            load_module("repository.db")

            # It's expected that the database tables have been created
            # by XChainDataGen, so we don't need to call create_tables() here.
        except Exception as e:
            raise CustomException(GraphDatasetGenerator.CLASS_NAME, function_name, "Failed to load database module") from e

    def generate_graph_dataset(self):
        storage_folder = os.path.abspath(self.output_folder)
        os.makedirs(storage_folder, exist_ok=True)

        # Delegate graph generation to the dataset class so the processing
        # logic stays centralized in one place.
        dataset = self.dataset_class(root=storage_folder, force_reload=self.force_reload)
        
        print(f"Generated dataset with {len(dataset)} graphs")
        print("dataset[0]:", dataset[0])  # Print the first graph data object for inspection

        # From the list of graphs, count the ones that are fraudulent vs non-fraudulent and print the statistics
        # Also count the ones that are cross-chain vs single-chain and print the statistics
        fraud_count = 0
        non_fraud_count = 0
        bridge_counts = {bridge.value: 0 for bridge in self.bridges}
        for graph in dataset:
            if graph.y.item() == 1:
                fraud_count += 1
            else:
                non_fraud_count += 1

            if hasattr(graph, 'bridge') and graph.bridge in bridge_counts:
                bridge_counts[graph.bridge] += 1

        print("========= STATISTICS =========")
        print(f"Normal-labeled graphs: {non_fraud_count}")
        print(f"Anomaly-labeled graphs: {fraud_count}")
        print(f"Anomaly ratio: {((fraud_count / len(dataset)) * 100):.3f}%")
        print("------------------------------")
        print("Bridge distribution:")
        for bridge, count in bridge_counts.items():
            print(f"  {bridge}: {count} graphs ({((count / len(dataset)) * 100):.3f}%)")