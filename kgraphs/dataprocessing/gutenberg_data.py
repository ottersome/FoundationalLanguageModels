import copy
import itertools
import json
import os
import re
from collections import defaultdict
from logging import DEBUG, INFO
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from kgraphs.utils.logging import create_logger

from .constants import (
    LEGALESE_END_MARKERS,
    LEGALESE_START_MARKERS,
    TEXT_END_MARKERS,
    TEXT_START_MARKERS,
)

"""
For now we will load project gutenberg and shard it into managable windows:
"""

_SPLIT_TO_POS = ["TRAIN", "VAL", "TEST"]

MAX_INTRASPACE_THRESHOLD = 5


class BasicDataset(Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def strip_headers(text):
    """
    Courtesy of `Gutenberg`. Cant find the github right now. If you can PR it please
    This on itself is a port from
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp

    Args:
        text (unicode): The body of the text to clean up.

    Returns:
        unicode: The text with any non-text content removed.

    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)


# Lets create an iterable class that will take in :
# either a file diretory list or a return from `load_dataset`
# and *yield* text within them
class TextStream(Iterable):
    def __init__(self, dataset_iter: Iterable[Any]):
        self.logger = create_logger(__class__.__name__, INFO)
        self.dataset_iter = dataset_iter
        # HACK: but meh:
        first_item = next(iter(self.dataset_iter))
        # place back first item into self.dataset_iter

        self.local = False
        try:
            "text" in first_item  # type:ignore
        except:
            self.local = True
        # Check how many items dataset_iter has
        self.dataset_iter = itertools.chain([first_item], self.dataset_iter)

    def __iter__(self):
        # Check if it is an iterable of str or the one provided by load_dataset
        if self.local:
            for filepath in self.dataset_iter:
                with open(filepath, "r") as f:
                    self.logger.debug(f"We are now looking into using file  {filepath}")
                    # yield content and name of file
                    content = ""
                    try:
                        content = f.read()
                    except UnicodeDecodeError:
                        self.logger.warn(
                            f"Cannot process, and thus skipping {filepath} (encoding not UTF-8)"
                        )
                        continue

                    self.logger.debug(f"Adding content of size: {len(content)}")
                    yield content, filepath
        else:
            for doc in self.dataset_iter:
                self.logger.debug("We are  actually using webstreams")
                # TODO: Ensure it works as above case
                yield doc["text"]


# DatasetFactory
class DatasetFactory:
    CACHE_RELPATH = ".cache/datasets/"
    SUPP_CTX_LEN = 128  # Supplementary Context Length
    CACHE_FORMAT = "{split}_{split_percent}_tknCap{token_cap}.csv"
    LANGUAGE = "english"
    CACHE_FORMAT_NFILES = "{split}_{split_percent}_tknCap{token_cap}_{nth_file}.csv"

    SPACE_PERC_THRESH = 0.1
    NUM_FILES_TO_CACHE_INTO = 100  # CHECK: if you rly want to use
    MIN_NUMBER_WORDS = 15
    WINDOW_OVERLAP = 128  # CHECK: implement if necessary
    GARBAGE_COLLECTION_THRESHOLD = 900
    REMOVE_REGEXES = [
        r"(?i)^\s*(Chapter|Section|Part)\s+\w+",  # Headings
        r"^\s*\d+(\.\d+)*\s+.*$",  # Numerical Patterns
        r"^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\s*$",  # Roman Numerals
        r"\[\d+\]",  # References
    ]

    def __init__(
        self,
        dataset_name: str,
        ptrnd_tknzr: PreTrainedTokenizer,
        window_size: int,
        amnt_tkns_for_training: int = -1,
        split: Tuple = (0.85, 0.15, 0),
        ds_location: str = "",
        stream: bool = False,
    ):
        """
        dataset_name: usually just the same one
        ptrnd_tknzr: PreTrainedTokenizer,
        ds_size:
        window_size: int: How big of a window we will have when training language model
        tokenizer_cap: int = How many tokenizers for dataset. Mostly for keeping test trains fast
        """

        assert sum(split) == 1, "Split percentages must some to 1"
        self.logger = create_logger(__class__.__name__, INFO)
        self.ds_location = ds_location
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.tknzr = ptrnd_tknzr
        self.split = split
        self.stream = stream

        if ds_location != "" and self.stream == True:
            self.logger.warn(
                f"You have provided ds_location {ds_location}, yet are using `self.stream"
                f"Local dataset location {ds_location} will be ignored"
            )

        # Token Cap is semantically for training so we pull the rest as:
        self.amnt_tkns_for_trning = amnt_tkns_for_training
        self.tot_tkns = amnt_tkns_for_training / split[0]

        # TOREM: Trashy state use
        self.cur_amount_of_tokens = 0

        # Read "data/gutenberg-metadata.json"
        self.metadata = json.load(open("data/gutenberg-metadata.json"))

        # os.makedirs(CACHE_NAME, exists_ok=True)
        self.cached_files = self._check_cache()  # If empty then we have no cache

    def _check_cache(self):
        # Ensure Cache works
        # Get cwd
        file_path = os.getcwd()
        self.cache_path = os.path.join(file_path, self.CACHE_RELPATH)
        self.logger.info(f"Using file_path {file_path}")

        files = []
        for i, s in enumerate(self.split):
            # if s == 0.0:
            #     continue
            filename = self.CACHE_FORMAT.format(
                split=_SPLIT_TO_POS[i],
                split_percent=s,
                token_cap=self.amnt_tkns_for_trning,
            )
            filepath = os.path.join(self.cache_path, filename)
            if os.path.exists(filepath):
                self.logger.info(f"Found file {filename}, loading it now")
                files.append(filepath)
            else:
                files.clear()
                break

        self.logger.info(
            f"We found  {len(files)} files in cache directory {self.cache_path}"
        )
        return files

    def _load_cached_files(self, files_to_load: List[str]) -> Tuple[pd.DataFrame, ...]:
        dss: List[pd.DataFrame] = []
        for f in files_to_load:
            self.logger.info(f"Loading partition {f}")
            try:
                csv: pd.DataFrame = pd.read_csv(f)
            # Catch EmptyDataError
            except pd.errors.EmptyDataError:
                csv = pd.DataFrame([])  # Just give it something empty
            dss.append(csv)

        assert (
            len(dss) == 3
        ), f"Not all partitions were loaded, instead we get {len(dss)}"
        return tuple(dss)

    def load_split(self) -> Tuple[pd.DataFrame, ...]:
        self.logger.info(f"Number of cached files is {self.cached_files}")
        if len(self.cached_files) == 0:
            self._compute_ds()

        return tuple(self._load_cached_files(self.cached_files))

    def _initialize_regex(self) -> List[re.Pattern]:
        return [re.compile(r) for r in self.REMOVE_REGEXES]

    def get_dataset_iter(self) -> Iterable[Any]:
        if self.stream:
            self.logger.info("Using Internet Stream to load it")
            return TextStream(
                load_dataset(self.dataset_name, split="en", streaming=True)
            )
        else:
            # Check size of file
            self.logger.info(f"Using local dataset at {self.ds_location}")
            self.logger.info(
                f"Berofere we send this ds location we can see it has {len(list(Path(self.ds_location).iterdir()))}"
            )
            return TextStream(Path(self.ds_location).iterdir())

    def _get_split_idx(
        self, current_file_idx, total_files: int, split: Tuple[float, float, float]
    ):
        train_cap = split[0] * total_files
        val_cap = (split[0] + split[1]) * total_files
        if current_file_idx < train_cap:
            self.logger.debug(
                f"Returning split idx {0} (TRAIN) deemed below {split[0]*total_files}"
            )
            return 0
        elif train_cap < current_file_idx and current_file_idx < val_cap:
            self.logger.debug(f"Returning split idx {1} (VAL)")
            return 1
        else:
            self.logger.debug(f"Returning split idx {2} (VAL)")
            return 2

    def _garbage_collection_split(self, ds: List[List[int]]):
        self.logger.debug(f"Performing garbage collection with list of size {len(ds)}")
        num_samples = len(ds)
        cache_path = Path(self.CACHE_RELPATH)
        cache_path.mkdir(parents=True, exist_ok=True)
        assign_file = np.random.choice(
            a=_SPLIT_TO_POS, p=self.split, size=(num_samples,)
        )
        file_to_sampleidx = {l: [] for l in _SPLIT_TO_POS}
        for i, a in enumerate(assign_file):
            file_to_sampleidx[a].append(i)

        for split_idx, (split_name, lista) in enumerate(file_to_sampleidx.items()):
            cachefile_path = cache_path / self.CACHE_FORMAT.format(
                split=split_name,
                split_percent=self.split[split_idx],
                token_cap=self.amnt_tkns_for_trning,
            )  # TODO: add test

            self.logger.debug(f"Saving to {cachefile_path} file")

            specific_ds = [ds[idx] for idx in lista]

            # Save as csv for now in append mode
            df = pd.DataFrame(specific_ds)
            df.to_csv(cachefile_path, mode="a", header=False, index=False)

            # Save as parquet file
            # with open(filecache_path, "a") as f:
            #     for idx in lista:
            #         f.write(json.dumps(ds[idx]) + "\n")
        self.logger.debug("Carbage Collection Done")

    # Probably used till later
    def _garbage_collection_nfiles(self, ds: List[List[int]]):
        self.logger.debug(f"Performing garbage collection with list of size {len(ds)}")
        num_samples = len(ds)
        cache_path = Path(self.CACHE_RELPATH)
        cache_path.mkdir(parents=True, exist_ok=True)
        assign_file = np.random.randint(
            low=0, high=self.NUM_FILES_TO_CACHE_INTO, size=(num_samples,)
        ).tolist()
        file_to_sampleidx = defaultdict(list)
        for i, a in enumerate(assign_file):
            file_to_sampleidx[a].append(i)

        for file_idx, lista in file_to_sampleidx.items():
            self.logger.debug(f"Saving at file with idx {file_idx}")
            split_idx = self._get_split_idx(
                file_idx, self.NUM_FILES_TO_CACHE_INTO, self.split[file_idx]
            )

            filecache_path = cache_path / self.CACHE_FORMAT_NFILES.format(
                split=_SPLIT_TO_POS[split_idx],
                split_percent=self.split[split_idx],
                token_cap=self.amnt_tkns_for_trning,
                nth_file=file_idx,
            )  # TODO: add test

            with open(filecache_path, "a") as f:
                for idx in lista:
                    f.write(json.dumps(ds[idx]) + "\n")
        self.logger.debug("Carbage Collection Done")

    def _compute_ds(self):
        """
        💫
        Assume we always load in the same order.
        """
        dataset_iter = self.get_dataset_iter()
        train: List[List[int]] = []
        cap = 2
        b = tqdm(total=cap + 1)
        self.cur_amount_of_tokens = 0

        clean_regexes = self._initialize_regex()
        windows_added = 0

        self.logger.debug("We are about to enter the dataset_iteration")
        for i, (doc_content, doc_name) in enumerate(dataset_iter):
            if self.cur_amount_of_tokens > self.amnt_tkns_for_trning:
                break

            # Check for language before processing
            PG_id = (str(doc_name).split("/")[-1].split("_")[0]).replace("PG", "")
            self.logger.debug(f"Checking PG_id {PG_id} for language")
            lang_id = self.metadata[PG_id]["language"][0]

            if lang_id == "en":
                new_list = self._doc(doc_content, b, clean_regexes)  # type:ignore
                self.logger.debug(f"Have added {len(new_list)} windows to our dataset")
                train += new_list  # type:ignore
                windows_added += len(new_list)
                b.set_description(f"Added {windows_added} samples")
            else:
                self.logger.debug(
                    f"Skipping file {doc_name} becuase language {lang_id} != {self.LANGUAGE}"
                )
            b.update(1)

            # Spread into the self.FILES_TO_CACHE_INTO
            if len(train) >= self.GARBAGE_COLLECTION_THRESHOLD:
                b.set_description(f"Garbage Collectin with len(train) = {len(train)}")
                self._garbage_collection_split(train)
                train.clear()

        # At the very end do one last round of garbage collection
        self._garbage_collection_split(train)

        # List to pretty, indent json
        # self.logger.info(f"Final Train boi is \n{train}")
        # pretty_list = json.dumps(train, indent=4) # DEBUG: remove
        # f.write(pretty_list)
        # f.close()
        # Once that is all good we cache it in some parquet file

    def _cache_tokenized(self, samples_list):
        df = pd.DataFrame(samples_list)
        # TODO: finish
        # df.to_parquet(

    def _doc_to_window_iterator(
        self, doc: str, removal_rgxs: List[re.Pattern]
    ) -> Iterator[List[int]]:
        """Im so sorry"""
        copy_doc = copy.deepcopy(doc)
        while len(copy_doc) > 1:
            return_tokens = []
            while len(return_tokens) < self.window_size:
                next_newline = copy_doc.find("\n")
                clean_seg = clean_segment(copy_doc[:next_newline], removal_rgxs)

                left_overs = []
                if len(clean_seg) != 0:
                    word_split = clean_seg.split(" ")
                    words_used, left_overs = self._process_words(
                        word_split, return_tokens
                    )

                if next_newline == -1 and len(left_overs) == 0:
                    return ""
                copy_doc = str(" ".join(left_overs)) + copy_doc[next_newline + 1 :]

            yield return_tokens

    def _process_words(
        self, word_split: List[str], return_tokens: List[int]
    ) -> Tuple[int, List[str]]:
        words_used = 0
        left_overs = []
        for word in word_split:
            words_used += 1
            enc_word = self.tknzr.encode(word, add_special_tokens=False)
            space_avail = self.window_size - len(return_tokens)
            return_tokens += enc_word[:space_avail]

            if len(return_tokens) >= self.window_size:
                left_overs = word_split[words_used:]
                break
        return words_used, left_overs

    def _doc(
        self, doc: str, bar: tqdm, clean_regexs: List[re.Pattern]
    ) -> List[List[int]]:
        # TODO: Lol clean this up into a single line or two
        doc = strip_headers(doc)
        tkn_win_iterator = self._doc_to_window_iterator(doc, clean_regexs)
        token_windows = []

        # While we can iterate over doc_tokenizer
        for tkn_win in tkn_win_iterator:
            ## Cleaning
            token_windows.append(tkn_win)
        return token_windows


class DatasetStream:
    SUPP_CTX_LEN = 128  # Supplementary Context Length
    LANGUAGE = "english"

    SPACE_PERC_THRESH = 0.1
    NUM_FILES_TO_CACHE_INTO = 100  # CHECK: if you rly want to use
    MIN_NUMBER_WORDS = 15
    WINDOW_OVERLAP = 128  # CHECK: implement if necessary
    GARBAGE_COLLECTION_THRESHOLD = 900
    REMOVE_REGEXES = [
        r"(?i)^\s*(Chapter|Section|Part)\s+\w+",  # Headings
        r"^\s*\d+(\.\d+)*\s+.*$",  # Numerical Patterns
        r"^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\s*$",  # Roman Numerals
        r"\[\d+\]",  # References
    ]

    def __init__(
        self,
        dataset_name: str,
        ptrnd_tknzr: PreTrainedTokenizer,
        window_size: int,
        amnt_tkns_for_training: int = -1,
        split: Tuple = (0.85, 0.15, 0),
        ds_location: str = "",
        stream: bool = False,
    ):
        """
        dataset_name: usually just the same one
        ptrnd_tknzr: PreTrainedTokenizer,
        ds_size:
        window_size: int: How big of a window we will have when training language model
        tokenizer_cap: int = How many tokenizers for dataset. Mostly for keeping test trains fast
        """

        assert sum(split) == 1, "Split percentages must some to 1"
        self.logger = create_logger(__class__.__name__, INFO)
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.tknzr = ptrnd_tknzr
        self.split = split

        # Samples Processed will be used to resume training from checkpoint
        self.samples_processed = 0

        # Read "data/gutenberg-metadata.json"
        self.metadata = json.load(open("data/gutenberg-metadata.json"))

    def _initialize_regex(self) -> List[re.Pattern]:
        return [re.compile(r) for r in self.REMOVE_REGEXES]

    def get_dataset_iter(self) -> Iterable[Any]:
        return TextStream(load_dataset(self.dataset_name, split="en", streaming=True))

    # Probably used till later

    def _compute_ds(self):
        """ """
        dataset_iter = self.get_dataset_iter()
        train: List[List[int]] = (
            []
        )  # LIst of tokens? List[int] is a document so We have List[Document] ?
        cap = 2
        b = tqdm(total=cap + 1)
        self.cur_amount_of_tokens = 0

        clean_regexes = self._initialize_regex()
        windows_added = 0

        self.logger.debug("We are about to enter the dataset_iteration")
        for i, (doc_content, doc_name) in enumerate(dataset_iter):
            if self.cur_amount_of_tokens > self.amnt_tkns_for_trning:
                break

            # Check for language before processing
            PG_id = (str(doc_name).split("/")[-1].split("_")[0]).replace("PG", "")
            self.logger.debug(f"Checking PG_id {PG_id} for language")
            lang_id = self.metadata[PG_id]["language"][0]

            if lang_id == "en":
                new_list_of_tokens = self._doc(
                    doc_content, b, clean_regexes
                )  # type:ignore
                # Here is where we add windows.
                self.logger.debug(f"Have added {len(new_list)} windows to our dataset")
                train += new_list  # type:ignore
                # Train contains a list of windows
                windows_added += len(new_list)
                b.set_description(f"Added {windows_added} samples")
            else:
                self.logger.debug(
                    f"Skipping file {doc_name} becuase language {lang_id} != {self.LANGUAGE}"
                )
            b.update(1)

            # Spread into the self.FILES_TO_CACHE_INTO
            # We only store the windows once we have enough
            if len(train) >= self.GARBAGE_COLLECTION_THRESHOLD:
                b.set_description(f"Garbage Collectin with len(train) = {len(train)}")
                self._garbage_collection_split(train)
                train.clear()

        # At the very end do one last round of garbage collection
        self._garbage_collection_split(train)

        # List to pretty, indent json
        # self.logger.info(f"Final Train boi is \n{train}")
        # pretty_list = json.dumps(train, indent=4) # DEBUG: remove
        # f.write(pretty_list)
        # f.close()
        # Once that is all good we cache it in some parquet file

    def _cache_tokenized(self, samples_list):
        df = pd.DataFrame(samples_list)
        # TODO: finish
        # df.to_parquet(

    def _doc_to_window_iterator(
        self, doc: str, removal_rgxs: List[re.Pattern]
    ) -> Iterator[List[int]]:
        """Im so sorry"""
        copy_doc = copy.deepcopy(doc)
        while len(copy_doc) > 1:
            return_tokens = []
            while len(return_tokens) < self.window_size:
                next_newline = copy_doc.find("\n")
                clean_seg = clean_segment(copy_doc[:next_newline], removal_rgxs)

                left_overs = []
                if len(clean_seg) != 0:
                    word_split = clean_seg.split(" ")
                    words_used, left_overs = self._process_words(
                        word_split, return_tokens
                    )

                if next_newline == -1 and len(left_overs) == 0:
                    return ""
                copy_doc = str(" ".join(left_overs)) + copy_doc[next_newline + 1 :]

            yield return_tokens

    def _process_words(
        self, word_split: List[str], return_tokens: List[int]
    ) -> Tuple[int, List[str]]:
        words_used = 0
        left_overs = []
        for word in word_split:
            words_used += 1
            enc_word = self.tknzr.encode(word, add_special_tokens=False)
            space_avail = self.window_size - len(return_tokens)
            return_tokens += enc_word[:space_avail]

            if len(return_tokens) >= self.window_size:
                left_overs = word_split[words_used:]
                break
        return words_used, left_overs

    def _doc(
        self, doc: str, bar: tqdm, clean_regexs: List[re.Pattern]
    ) -> List[List[int]]:
        doc = strip_headers(doc)
        tkn_win_iterator = self._doc_to_window_iterator(doc, clean_regexs)
        token_windows = []

        # While we can iterate over doc_tokenizer
        for tkn_win in tkn_win_iterator:
            ## Cleaning
            token_windows.append(tkn_win)
        return token_windows


def clean_segment(segment: str, removal_rgxs: List[re.Pattern]) -> str:
    ### Miscelannea Round
    tot_len = len(segment)
    if tot_len == 0:
        return ""

    max_space = 0
    for word in segment.split(" "):
        if word == "":
            max_space += 1
        if max_space > MAX_INTRASPACE_THRESHOLD:
            return ""

    ###  Regex round
    for r in removal_rgxs:
        segment = r.sub("", segment)

    ########################################
    # Looks clean, final touches
    ########################################

    final_segment = segment.strip()
    return final_segment


def textblock_to_window_iterator(
    doc: str,
    removal_rgxs: List[re.Pattern],
    tokenizer: PreTrainedTokenizer,
    window_size: int,
) -> Iterator[List[int]]:
    copy_doc = copy.deepcopy(doc)

    while len(copy_doc) > 1:
        return_tokens = []

        while len(return_tokens) < window_size:
            next_newline = copy_doc.find("\n")
            clean_seg = clean_segment(copy_doc[:next_newline], removal_rgxs)

            left_overs = ""
            if len(clean_seg) != 0:
                # word_split = clean_seg.split(" ")
                num_tokens_so_far = len(return_tokens)
                tokens_collected, left_overs = process_words(
                    num_tokens_so_far, tokenizer, window_size, clean_seg
                )
                return_tokens += tokens_collected

            if next_newline == -1 and len(left_overs) == 0:
                return ""
            copy_doc = left_overs + copy_doc[next_newline + 1 :]

        yield return_tokens


def process_words(
    preexisting_num_tokens: int,
    tokenizer: PreTrainedTokenizer,
    window_size: int,
    clean_seg: str,
) -> Tuple[List[int], str]:
    """
    word_split: List[str]: The words to be processed. Generally separated by spaces.
    returns:
        - List[int]: The tokens collected
        - List[str]: The leftovers
    """
    tokens_collected = []
    num_tokens_so_far = preexisting_num_tokens
    space_avail = window_size - num_tokens_so_far

    encoded_seg = tokenizer.encode(clean_seg, add_special_tokens=False)
    tokens_collected = encoded_seg[:space_avail]
    left_overs = tokenizer.decode(encoded_seg[space_avail:])

    return tokens_collected, left_overs


# NOTE: Should replaace the ones inside the class
def process_words_without_space(
    preexisting_num_tokens: int,
    tokenizer: PreTrainedTokenizer,
    window_size: int,
    word_split: List[str],
) -> Tuple[List[int], List[str]]:
    """
    word_split: List[str]: The words to be processed. Generally separated by spaces.
    returns:
        - List[int]: The tokens collected
        - List[str]: The leftovers
    """
    words_used = 0
    left_overs = []
    tokens_collected = []
    num_tokens_so_far = preexisting_num_tokens
    for word in word_split:
        words_used += 1
        enc_word = tokenizer.encode(word, add_special_tokens=False)
        space_avail = window_size - num_tokens_so_far
        added_tokens = enc_word[:space_avail]
        tokens_collected += added_tokens
        num_tokens_so_far += len(added_tokens)

        # TODO: I think it should be here that we want to introduce overlap. Though I am not entirely sure.
        # Not pressing. Think about it if it ever becomes reasonable.
        if num_tokens_so_far >= window_size:
            left_overs = word_split[words_used:]
            break
    return tokens_collected, left_overs

