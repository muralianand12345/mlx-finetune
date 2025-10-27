import os
import json
import argparse
from typing import Callable
from datasets import load_dataset, Dataset, DatasetDict

class LoadData:
    """
    Utility for loading, transforming and saving conversational datasets for LLM fine-tuning.

    The `LoadData` class wraps a HF/datasets-style dataset (loaded via
    `datasets.load_dataset`) and provides helpers to load and optionally
    subsample a split, split a dataset into train/test/valid, transform
    conversation items (expected to be a `messages` list) into a simple
    JSONL record with a single `text` field, and stream or write the
    transformed splits to disk.

    See module-level doc for full details. The implementation expects dataset
    entries to include a `messages` list where the first three entries
    correspond to system, user and assistant messages respectively. Malformed
    entries (missing or too-short `messages`) are skipped.
    """
    def __init__(self, folder: str, dataset_name: str):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.dataset_name = dataset_name
    
    @staticmethod
    def _format_text(messages: list) -> str:
        """
        Helper to format the text from messages list.
        Expects messages to be a list of dicts with 'role' and 'content' keys.

        Args:
            messages (list): List of message dicts.
        Returns:
            str: Formatted text combining system, user and assistant messages.
        Raises:
            ValueError: If the formatted text is too short, indicating malformed messages.
        """
        text = ""
        for message in messages:
            role = message.get('role', '').strip()
            content = message.get('content', '').strip()
            if role and content:
                text += f"{role.capitalize()}: {content}\n"
        if len(text) < 5:
            raise ValueError("Formatted text is too short. Ensure messages contain valid 'role' and 'content'.")
        return text.strip()

    def _get_dataset(self, function: Callable, split: str = 'train', n: int | None = None) -> Dataset:
        """
        Helper to load and optionally subsample a dataset split, applying
        a transformation function to each item.

        Args:
            function (Callable): Function to apply to each dataset item.
            split (str): Dataset split to load (default: 'train').
            n (int | None): Optional number of items to select from the split. If None, loads the full split (default: None).
        Returns:
            Dataset: Transformed dataset.
        """
        dataset = load_dataset(self.dataset_name, split=split).shuffle().select(range(n)) if n is not None else load_dataset(self.dataset_name, split=split).shuffle()
        return dataset.map(function, remove_columns=dataset.features, batched=False)

    def _split_dataset(self, dataset: Dataset, test_split_ratio: float, valid_split_ratio: float) -> DatasetDict:
        """
        Helper to split a dataset into train/test/valid splits.

        Args:
            dataset (Dataset): The dataset to split.
            test_split_ratio (float): Proportion of data to allocate to the test split.
            valid_split_ratio (float): Proportion of the remaining data (after test split) to allocate to the validation split.
        Returns:
            DatasetDict: A dictionary with 'train', 'test', and 'valid' splits.
        """
        dataset_train_test = dataset.train_test_split(test_split_ratio)
        dataset_test_valid = dataset_train_test['train'].train_test_split(valid_split_ratio)
        return DatasetDict({
            'train': dataset_test_valid['train'],
            'test': dataset_train_test['test'],
            'valid': dataset_test_valid['test']
        })
    
    def _transform_data(self, dataset: DatasetDict) -> dict:
        """
        Helper to transform a dataset by formatting the `messages` field
        into a single `text` field.

        Args:
            dataset (DatasetDict): The dataset to transform.
        Returns:
            dict: A dictionary with transformed splits.
        """
        out = {k: [] for k in dataset.keys()}
        for split, ds in dataset.items():
            for item in ds:
                messages = item.get('messages')
                if not messages or len(messages) < 3:
                    # skip malformed entries
                    continue
                text = self._format_text(messages)
                out[split].append({"text": text})
        return out

    def stream_transformed(self, dataset: DatasetDict):
        """
        Helper to stream transformed dataset entries.

        Args:
            dataset (DatasetDict): The dataset to transform and stream.
        Yields:
            tuple: A tuple containing the split name and the transformed entry.
        """
        for split, ds in dataset.items():
            for item in ds:
                messages = item.get('messages')
                if not messages or len(messages) < 3:
                    continue
                yield split, {"text": self._format_text(messages)}

    def save(self, function: Callable, n: int | None = None, test_split_ratio: float = 0.2, valid_split_ratio: float = 0.2, write_files: bool = True) -> dict:
        """
        Helper to save the processed dataset.

        Args:
            function (Callable): Function to apply to each dataset item.
            n (int | None): Optional number of items to select from the split. If None, loads the full split (default: None).
            test_split_ratio (float): Proportion of data to allocate to the test split.
            valid_split_ratio (float): Proportion of the remaining data (after test split) to allocate to the validation split.
            write_files (bool): Whether to write the splits to disk (default: True). If False, returns a generator of transformed entries.
        Returns:
            dict: If write_files is True, returns a dict with paths to saved files. If False, returns a generator yielding transformed entries.
        """
        dataset = self._get_dataset(function=function, n=n)
        dataset_train_test_valid = self._split_dataset(dataset=dataset, test_split_ratio=test_split_ratio, valid_split_ratio=valid_split_ratio)

        if not write_files:
            return self.stream_transformed(dataset_train_test_valid)

        paths = {}
        writers = {}
        try:
            for split in ['train', 'test', 'valid']:
                path = os.path.join(self.folder, f"{split}.jsonl")
                paths[split] = path
                writers[split] = open(path, 'w', encoding='utf-8')

            for split, entry in self.stream_transformed(dataset_train_test_valid):
                json.dump(entry, writers[split], ensure_ascii=False)
                writers[split].write('\n')

        finally:
            for f in writers.values():
                try:
                    f.close()
                except Exception:
                    pass

        return paths

def build_parser():
    """Command-line interface to load, process and save a dataset."""
    parser = argparse.ArgumentParser(description="Load and preprocess dataset for LLM fine-tuning using MLX's LoRA")
    parser.add_argument("--folder", "-f", type=str, help="Folder to save the processed data", default="data")
    parser.add_argument("--dataset_name", "-d", type=str, help="Name of the dataset to load", default="b-mc2/sql-create-context")
    parser.add_argument("--test_split_ratio", "-t", type=float, help="Test split ratio", default=0.2)
    parser.add_argument("--valid_split_ratio", "-v", type=float, help="Validation split ratio", default=0.2)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    loader = LoadData(folder=args.folder, dataset_name=args.dataset_name)
    loader.save(test_split_ratio=args.test_split_ratio, valid_split_ratio=args.valid_split_ratio)
    
    print("=" * 20)
    print(f"Data saved in folder: {args.folder}")