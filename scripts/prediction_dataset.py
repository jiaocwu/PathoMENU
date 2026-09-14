#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm


class PredictionPairDataset(Dataset):

    def __init__(
        self,
        tags: Iterable[str],
        processed_dir: Path,
        strict: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.strict = strict
        self.available_tags: list[str] = []
        self.missing_tags: list[tuple[str, str]] = []
        self.error_tags: list[str] = []

        tag_list = [str(tag).strip() for tag in tags if str(tag).strip()]
        print(f"Verifying data integrity for {len(tag_list)} items...")
        for tag in tqdm(tag_list, desc="Checking files"):
            try:
                path = self._get_file_path(tag)
                if path.exists():
                    self.available_tags.append(tag)
                else:
                    self.missing_tags.append((tag, str(path)))
            except Exception as exc:
                print(f"\n[Init Error] Failed to generate path for tag: {tag}")
                print(f"Error details: {exc}")
                self.error_tags.append(tag)

        if strict and self.missing_tags:
            raise FileNotFoundError(f"{len(self.missing_tags)} feature files are missing.")
        if not self.available_tags:
            raise RuntimeError(f"No usable .pt files found under {self.processed_dir}")

        original_count = len(tag_list)
        available_count = len(self.available_tags)
        if original_count > available_count:
            print(f"\nWarning: Found {available_count}/{original_count} available data files.")
            print(f"  {original_count - available_count} items were skipped.")
            if self.missing_tags:
                print("  Example missing files (first 5):")
                for tag, path in self.missing_tags[:5]:
                    print(f"    Tag: {tag} -> Not found at: {path}")
            if self.error_tags:
                print(f"  Tags causing format errors: {len(self.error_tags)}")
        else:
            print(f"Success: All {available_count} data files found. Dataset is ready.")

    def _get_file_path(self, tag: str) -> Path:
        return self.processed_dir / f"{tag}.pt"

    def __len__(self) -> int:
        return len(self.available_tags)

    def __getitem__(self, index: int):
        tag = self.available_tags[index]
        try:
            path = self._get_file_path(tag)
            with path.open("rb") as f:
                try:
                    wt_data, mut_data = torch.load(f, weights_only=False)
                except TypeError as exc:
                    if "weights_only" not in str(exc):
                        raise
                    f.seek(0)
                    wt_data, mut_data = torch.load(f)

            wt_data = wt_data.clone()
            mut_data = mut_data.clone()
            wt_data.tag = tag
            mut_data.tag = tag
            wt_data.y = torch.tensor(0.0, dtype=torch.float32)
            return wt_data, mut_data, tag
        except Exception as exc:
            print(f"\n[Data Corrupted] Skipping corrupted file for tag {tag}: {exc}")
            return None


def graph_pair_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    original_list, mut_list, tags = zip(*batch)
    original_batch = Batch.from_data_list(original_list)
    mut_batch = Batch.from_data_list(mut_list)

    attr = "res_esm1v_fea"
    if hasattr(original_list[0], attr):
        setattr(original_batch, attr, torch.stack([getattr(data, attr) for data in original_list]))
        setattr(mut_batch, attr, torch.stack([getattr(data, attr) for data in mut_list]))
    return original_batch, mut_batch, list(tags)
