from typing import Any, List, Tuple
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import random
from torch.utils.data import IterableDataset
import re
from itertools import islice
from logging import INFO

from kgraphs.utils.logging import create_logger
from kgraphs.dataprocessing.gutenberg_data import strip_headers, textblock_to_window_iterator


class TextDataSet(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].tolist()).to(torch.long)



class GutenbergDataset(Dataset):
    def __init__(self, split="train", max_length=512, tokenizer_name="gpt2", cache_dir=None):
        # Load the Gutenberg dataset using Hugging Face's datasets library
        self.dataset = load_dataset("gutenberg", split=split, cache_dir=cache_dir)
        
        # Initialize tokenizer, using GPT-2 as an example
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve text from the dataset
        text = self.dataset[idx]["text"]
        
        # Tokenize the text
        encode = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare tensors
        input_ids = encode["input_ids"].squeeze(0)
        attention_mask = encode["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Return a dictionary of inputs and labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class DerivativeNonStreamedDataset(Dataset):
    def __init__(self, split_list: List[List[int]]):
        self.split_list = split_list

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        return torch.LongTensor(self.split_list[idx])


class GutenbergDatasetStreamed(IterableDataset):
    """
    Buffered-Streaming approach. 
    Will load a fixed number of documents into memory and shuffle them before serving.
    """
    LANG = "en"
    REMOVE_REGEXES = [
        r"(?i)^\s*(Chapter|Section|Part)\s+\w+",  # Headings
        r"^\s*\d+(\.\d+)*\s+.*$",  # Numerical Patterns
        r"^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\s*$",  # Roman Numerals
        r"\[\d+\]",  # References
    ]

    def __init__(
        self,
        dataset_stream_name: str,
        buffer_size: int,
        window_size: int,
        tokenizer_name: str,
        number_of_windows_in_batch: int,
        split : Tuple[float, float, float],
    ):
        self.buffer_size = buffer_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.logger = create_logger(__class__.__name__, INFO)

        # Load the Gutenberg dataset from Hugging Face
        self.dataset_stream = load_dataset(
            dataset_stream_name, split=self.LANG, streaming=True
        )

        self.stream_pointer = 0 # Tell us where we are in stream
        self.length = self.dataset_stream.info.splits[self.LANG].num_examples  # type: ignore

        # Set padding token to EOS token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.global_doc_idx = 0
        self.window_size = window_size
        self.windows_in_buffer = []
        self.windows_per_batch = number_of_windows_in_batch
        self.current_window_in_buffer_offset = 0
        self.windows_corresponding_to_buffer = []
        self.regexes = [re.compile(r) for r in self.REMOVE_REGEXES]
        self.split = split

        # For Validation and Testing
        # These should be done on a per batch basis
        self.validation_windows = []
        self.test_windows = []

        self.initialize_buffer()

    def get_validation_windows(self):
        assert len(self.test_windows) != 0, "validation windows are empty"
        return self.validation_windows  

    def get_test_windows(self):
        assert len(self.test_windows) != 0, "test windows are empty"
        return self.test_windows

    def initialize_buffer(self):
        self._load_more_buffer()

    def _load_more_buffer(self):
        # NOTE: The `islice` will move the `self.dataset_stream` iterator forward
        self.logger.info(f"Checking for more buffer with {self.buffer_size} items")
        self.buffer  = [id_text_dict["text"] for id_text_dict in islice(self.dataset_stream, self.buffer_size)]
        random.shuffle(self.buffer)

        self.windows_corresponding_to_buffer = self.buffer_to_windows(self.buffer)
        self.current_window_in_buffer_offset = 0
        self.logger.info(f"Done downloading more buffer")

        # Likewise we reserve some of those windows for validation and testing
        self._reserve_for_validation(self.windows_corresponding_to_buffer)


    def buffer_to_windows(self, buffer: List[str], shuffle_windows: bool = False) -> List[Any]:
        """
        buffer: List[str]: The documents I have to chop into windows for the model to learn from.
        shuffle_windows: bool: Whether or not to shuffle the windows before returning them.
            I do'nt think this is is necessary
        """
        buffer_windows = []
        for doc in tqdm(buffer, desc="Converting Buffer to Windows"):
            buffer_windows += self._doc_to_windows(doc)

        if shuffle_windows:
            random.shuffle(buffer_windows)

        return buffer_windows

    def _reserve_for_validation(self, buffer_windows: List[Any]):
        num_limit_train = int(self.split[0] * len(buffer_windows))
        num_limit_val = int(self.split[1] * len(buffer_windows))

        self.validation_windows = buffer_windows[num_limit_train : num_limit_train + num_limit_val]
        self.test_windows = buffer_windows[num_limit_train + num_limit_val :]

    def there_still_batches(self) -> bool:
        """
        Look at the current document index and see if there are more documents.
        If no more documents than if there are more batches within last document
        """
        if self.global_doc_idx < (self.dataset_stream.info.splits[self.LANG].num_examples - 1): # type: ignore
            return True
        else:
            if len(self.windows_corresponding_to_buffer) - self.current_window_in_buffer_offset >= self.windows_per_batch:
                return True

        return False


    def __iter__(self):

        while self.there_still_batches():
            nextbatch_window_ptr = self.current_window_in_buffer_offset + self.windows_per_batch
            windows_to_return = []
            current_num_windows = len(self.windows_corresponding_to_buffer)

            next_idx = min(nextbatch_window_ptr, current_num_windows-1)
            windows_to_return = self.windows_corresponding_to_buffer[self.current_window_in_buffer_offset : next_idx]
            still_needed_windows = self.windows_per_batch - (len(windows_to_return))
            # In case we need to replenish the buffer
            if still_needed_windows > 0:
                # widows_to_return = self.windows_corresponding_to_buffer[self.current_window_in_buffer_offset:]
                self.logger.info(f"Buffer is empty, filling it")
                self._load_more_buffer()
                self.logger.info(f"New Number of windows in buffer is {len(self.windows_corresponding_to_buffer)}")

                if still_needed_windows > 0:
                    windows_to_return += self.windows_corresponding_to_buffer[ 0 : 0 + still_needed_windows ]
                    self.current_window_in_buffer_offset += still_needed_windows

            # Now just draw from these windows the next windows
            # DEBUG: Remove when this assertion feel safe
            assert (
                len(windows_to_return) == self.windows_per_batch
            ), "You are not returning batch_size windows to your model"

            yield torch.LongTensor(windows_to_return)
        

    def _doc_to_windows(self, doc: str) -> List[List[int]]: 
        """
        Extract the document into windows
        """
        text_block = strip_headers(doc)
        tkn_win_iterator = textblock_to_window_iterator(text_block, self.regexes, self.tokenizer, self.window_size)
        return list(tkn_win_iterator)
        
