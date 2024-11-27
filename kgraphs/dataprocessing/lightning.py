from typing import Optional, Tuple, List

import lightning.pytorch as pl
from kgraphs.dataprocessing.datasets import GutenbergDatasetStreamed, DerivativeNonStreamedDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DocumentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_stream_name: str,
        buffer_size: int,
        batch_size: int,
        window_size: int,
        tokenizer_name: str,
        split: Tuple[float, float, float]
    ):

        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # Units are windows
        self.window_size = window_size
        self.dataset_stream_name = dataset_stream_name
        self.tokenizer_name = tokenizer_name
        self.gutenberg_dataset = GutenbergDatasetStreamed(
                self.dataset_stream_name,
                self.buffer_size,
                self.window_size,
                self.tokenizer_name,
                self.batch_size,
                split
            )

        self.dataloader = DataLoader(
            self.gutenberg_dataset,
            batch_size=self.batch_size,
        )

    def prepare_data(self):
        print("Preparing the data...Not really")

    def setup(self, stage: Optional[str] = None):
        print(f"Setting up the data for stage {stage}")

    def train_dataloader(self):
        return DataLoader(DerivativeNonStreamedDataset(self.gutenberg_dataset.get_validation_windows()), batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(DerivativeNonStreamedDataset(self.gutenberg_dataset.get_test_windows()), batch_size=self.batch_size)
    def predict_dataloader(self):
        return None
