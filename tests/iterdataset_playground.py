import random

import debugpy
import torch
from transformers import AutoTokenizer

from kgraphs.dataprocessing.datasets import GutenbergDatasetStreamed


def main():

    # print("Initializing the debugger")
    # debugpy.listen(42020)
    # print("Waiting for debugger to connect...")
    # debugpy.wait_for_client()
    # print("Client Connected.")
    
    print("Loading the dataset")
    write_test_to = "./test.txt"
    file = open(write_test_to, "w")
    dataset = GutenbergDatasetStreamed(
        dataset_stream_name = "manu/project_gutenberg",
        buffer_size=10,
        window_size=1024,
        tokenizer_name="facebook/bart-base",
        number_of_windows_in_batch=4,
    )

    majical_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # Iterating over the dataset
    # for window_batch in enumerate(dataset):
    dataset_iterator = iter(dataset)
    for idx in range(3):
        window_batch = next(dataset_iterator)
        # Pick three windows at random:
        random_window_idxs = random.sample(range(len(window_batch)), 3)
        # Decode the tokens in the windows to BPE tokens
        bpe_tokens = [majical_tokenizer.decode(window_batch[idx]) for idx in random_window_idxs]
        for window in bpe_tokens:
            file.write(window)
            file.write("\n")
        print(f"Window {idx} is written sj")

        
    print("Dataset loaded")
    file.close()

if __name__ == "__main__":
    main()

