from typing import TypeAlias

from dataset_generator.cctx_dataset import CrossChainTransactionsDataset
from dataset_generator.mixed_transactions_dataset import MixedTransactionsDataset
from dataset_generator.single_chain_dataset import BlockchainTransactionsDataset


CanonicalEdgeType: TypeAlias = tuple[str, str, str]
EdgeMetapath: TypeAlias = tuple[CanonicalEdgeType, ...]


class DATASET_TYPE:
    MIXED = "mixed"
    SINGLE = "single"
    CCTX = "cctx"


DATASET_CLASS = {
    DATASET_TYPE.MIXED: MixedTransactionsDataset,
    DATASET_TYPE.SINGLE: BlockchainTransactionsDataset,
    DATASET_TYPE.CCTX: CrossChainTransactionsDataset
}