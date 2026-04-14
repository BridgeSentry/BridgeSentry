from dataset_generator.dataset import CrossChainTransactionsDataset
from utils.utils import CustomException, load_module
import os


class GraphDatasetGenerator:
    CLASS_NAME = "GraphDatasetGenerator"
    
    def __init__(self, bridges, output_folder):
        self.bridges = bridges
        self.output_folder = output_folder
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
        dataset = CrossChainTransactionsDataset(root=storage_folder)
        
        print(f"Dataset length: {len(dataset)}")