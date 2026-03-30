import os

import torch
from torch_geometric.data import Dataset, HeteroData

class CrossChainDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        root_dir = os.path.join(os.path.dirname(__file__), "data")
        super().__init__(root_dir, transform, pre_transform, pre_filter)
    
    def saveGraph(self, graph_data: HeteroData, cctx_graph_id):
        torch.save(graph_data, os.path.join(self.raw_dir, f'{cctx_graph_id}.pt'))

    def process(self, graph_data: HeteroData, cctx_graph_id):
        if self.pre_filter is not None and not self.pre_filter(graph_data):
            return
        
        if self.pre_transform is not None:
            graph_data = self.pre_transform(graph_data)

        torch.save(graph_data, os.path.join(self.processed_dir, f'{cctx_graph_id}.pt'))
    
    def len(self):
        return len(self.processed_file_names)
    
    def getGraph(self, cctx_graph_id):
        graph_path = os.path.join(self.processed_dir, f'{cctx_graph_id}.pt')
        return torch.load(graph_path)