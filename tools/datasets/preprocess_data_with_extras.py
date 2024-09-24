# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys

import lm_dataformat as lmd
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
import time
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text)


def get_args(input_args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Default: text",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group.add_argument(
        "--max-lines-per-file",
        default=0,
        help="Limit data on line-level (before filters).",
        type=int,
    )
    group.add_argument(
        "--skip-lines-per-file",
        default=0,
        help="Limit data on line-level (before filters).",
        type=int,
    )
    group.add_argument(
        "--max-docs-per-file",
        default=0,
        help="Limit data on document-level (after filters).",
        type=int,
    )
    group.add_argument(
        "--skip-docs-per-file",
        default=0,
        help="Limit data on document-level (after filters).",
        type=int,
    )
    group.add_argument(
        "--apply-domain-filter",
        action="store_true",
        help="Limit data.",
    )
    group.add_argument(
        "--exclude-domains",
        action="store_true",
        help="Limit data.",
    )
    group.add_argument(
        "--domains-path",
        default=None,
        help="Limit data.",
        type=str,
    )
    group.add_argument(
        "--save-filtered-input",
        action="store_true",
        help="Save filtered input into JSONL in `output_prefix`",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "SPMTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args(input_args)
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(
    fnames: list,
    semaphore,
    glob_pattern="*.jsonl.gz",
    text_key="text",
    domain_key="domain",
    max_lines_per_file=0,
    skip_lines_per_file=0,
    max_docs_per_file=0,
    skip_docs_per_file=0,
    apply_domain_filter=False,
    exclude_domains=False,
    domains_path=None,
    save_filtered_input=False,
):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    # lmd.Reader: does not support ".jsonl.gz" --> it will return decoded JSON strings
    # def yielder(fname, semaphore):
    #     for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
    #         semaphore.acquire()
    #         yield f

    print("Input settings:")
    print("apply_domain_filter", apply_domain_filter)
    print("exclude_domains", exclude_domains)
    print("max_lines_per_file", max_lines_per_file)
    print("skip_lines_per_file", skip_lines_per_file)
    print("max_docs_per_file", max_docs_per_file)
    print("skip_docs_per_file", skip_docs_per_file)
    print("---")

    if save_filtered_input:
        # wrappper around fin but also write text into
        raise NotImplementedError

    if apply_domain_filter:
        print(f"Reading domains from {domains_path}")
        include_domains_df = pd.read_csv(domains_path)
        include_domains_set = set(include_domains_df.domain)

        print(f"Loaded {len(include_domains_set):,} domains")
    else:
        include_domains_set = None

    def yielder(fname, semaphore):
        if not fname.endswith(".jsonl.gz"):
            raise ValueError("Only .jsonl.gz files are supported!")

        yielded_docs = 0
        skipped_docs = 0

        with gzip.open(fname) as f:
            for i, line in enumerate(f):
                if skip_lines_per_file > 0 and i < skip_lines_per_file:
                    # print("Skip line", i)
                    continue

                doc = json.loads(line)

                if apply_domain_filter:
                    domain = doc[domain_key]

                    if exclude_domains and domain in include_domains_set:
                        # domain is part of set -> skip because exclude_domains is enabled
                        continue
                    elif not exclude_domains and domain not in include_domains_set:
                        # domain is not part of set -> skip because exclude_domains is disabled
                        continue

                text = doc[text_key]

                # print(">>> text = ", text[:10])

                if skip_docs_per_file > 0 and skipped_docs < skip_docs_per_file:
                    # print("Skip doc", i)
                    skipped_docs += 1
                    continue

                semaphore.acquire()

                yield text
                yielded_docs += 1

                if max_lines_per_file > 0 and i >= max_lines_per_file:
                    print("Max lines per file reached: ", i)
                    break

                if max_docs_per_file > 0 and yielded_docs >= max_docs_per_file:
                    print("Max docs per file reached: ", yielded_docs)
                    break

    for fname in fnames:
        semaphore.acquire()

        if os.path.isdir(fname):
            files_in_dir = list(sorted(Path(fname).rglob(glob_pattern)))
            print(
                "Found %i files in %s with pattern %s"
                % (len(files_in_dir), fname, glob_pattern)
            )
            for i, fname_in_dir in enumerate(files_in_dir, 1):
                yield from yielder(str(fname_in_dir), semaphore)
                print(
                    "File in dir completed %i/%i from %s"
                    % (i, len(files_in_dir), fname)
                )
        else:
            yield from yielder(fname, semaphore)


def main(input_args=None):
    args = get_args(input_args)
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(
        args.input.split(","),
        semaphore,
        max_lines_per_file=args.max_lines_per_file,
        skip_lines_per_file=args.skip_lines_per_file,
        max_docs_per_file=args.max_docs_per_file,
        skip_docs_per_file=args.skip_docs_per_file,
        apply_domain_filter=args.apply_domain_filter,
        exclude_domains=args.exclude_domains,
        domains_path=args.domains_path,
        save_filtered_input=args.save_filtered_input,
    )

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed :.2f} docs/s, {mbs:.2f} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
