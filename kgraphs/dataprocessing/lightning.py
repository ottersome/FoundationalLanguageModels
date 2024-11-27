import pytorch_lightning as pl
from kgraphs.dataprocessing.datasets import  GutenbergDatasetStreamed

class DocumentDataModule(pl.LightningDataModule):
    def __init__(self, dataset_stream_name: str,  buffer_size: int, batch_size: int, window_size: int, tokenizer_name: str):
        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size # Units are windows
        self.window_size = window_size
        self.dataset_stream_name = dataset_stream_name
        self.tokenizer_name = tokenizer_name

    def train_dataloader(self):
        dataset = GutenbergDatasetStreamed(self.dataset_stream_name,  self.buffer_size,self.window_size, self.batch_size)
