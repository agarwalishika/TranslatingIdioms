# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs

def process(file, language, train_set):
    data_source = f"grpo-{language}-idioms"
    df = pd.read_csv(file)
    train_dataset = datasets.Dataset.from_pandas(df[:train_set])
    test_dataset = datasets.Dataset.from_pandas(df[train_set:])

    instruction_following = lambda language, idiom: f"""Given the following idiom in {language}, translate it semantically into English.\nIdiom: " + {idiom} + "\n### Semantic Translation:"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = instruction_following(language, example.get('src'))
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "data_synthesis",
                "reward_model": {"ground_truth": example.get('true_meaning')},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "src": example.get('src'),
                    "true_meaning": example.get('true_meaning'),
                    "literal_translation": example.get('literal_translation') 
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, f"{data_source}_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, f"{data_source}_test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="Dataset/", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    process("/home/ishikaa2/translating_idioms/Dataset/hindi-english_idioms.csv", "Hindi", 800)
    process("/home/ishikaa2/translating_idioms/Dataset/petci_chinese_english_improved.csv", "Chinese", 1000)