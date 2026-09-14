#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def load_tags(path: Path, column: str | None) -> list[str]:
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as handle:
            values = pickle.load(handle)
        return [str(value).strip() for value in values if str(value).strip()]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        selected = column or "Tag"
        if selected not in frame.columns:
            raise ValueError(f"Column {selected!r} is absent from {path}; columns={list(frame.columns)}")
        return frame[selected].dropna().astype(str).str.strip().tolist()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tensor(directory: Path, tag: str, field: str) -> torch.Tensor:
    path = directory / f"{tag}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field}: {path}")
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor


def build_edges(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    positions = positions.float()
    distance = torch.cdist(positions, positions)
    adjacent = distance < float(cutoff)
    adjacent.fill_diagonal_(False)
    return adjacent.nonzero(as_tuple=False).t().contiguous().long()


def assemble_one(tag: str, args: argparse.Namespace) -> Path:
    wild_structure = load_tensor(args.wild_structure_dir, tag, "wild-type structure feature")
    mutant_structure = load_tensor(args.mutant_structure_dir, tag, "mutant structure feature")
    wild_positions = load_tensor(args.wild_position_dir, tag, "wild-type coordinates").float()
    mutant_positions = load_tensor(args.mutant_position_dir, tag, "mutant coordinates").float()
    wild_sequence = load_tensor(args.wild_sequence_dir, tag, "wild-type sequence feature").float()
    mutant_sequence = load_tensor(args.mutant_sequence_dir, tag, "mutant sequence feature").float()
    wild_mutation_index = load_tensor(args.wild_mutation_index_dir, tag, "wild-type mutation index").long()
    mutant_mutation_index = load_tensor(args.mutant_mutation_index_dir, tag, "mutant mutation index").long()

    if wild_structure.shape[0] != wild_positions.shape[0]:
        raise ValueError(f"{tag}: wild-type structure rows {wild_structure.shape[0]} != position rows {wild_positions.shape[0]}")
    if mutant_structure.shape[0] != mutant_positions.shape[0]:
        raise ValueError(f"{tag}: mutant structure rows {mutant_structure.shape[0]} != position rows {mutant_positions.shape[0]}")
    for name, tensor in (("wild-type structure", wild_structure), ("mutant structure", mutant_structure)):
        if tensor.ndim != 2 or tensor.shape[1] != args.structure_dimension:
            raise ValueError(f"{tag}: {name} shape {tuple(tensor.shape)} must end in {args.structure_dimension}")
    for name, tensor in (("wild-type sequence", wild_sequence), ("mutant sequence", mutant_sequence)):
        expected = (args.sequence_length, args.sequence_dimension)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{tag}: {name} shape {tuple(tensor.shape)} must equal {expected}")
    for name, index, row_count in (
        ("wild-type mutation index", wild_mutation_index, wild_positions.shape[0]),
        ("mutant mutation index", mutant_mutation_index, mutant_positions.shape[0]),
    ):
        if index.numel() != 1 or not 0 <= int(index.reshape(-1)[0]) < row_count:
            raise ValueError(f"{tag}: {name} must contain one zero-based index in [0, {row_count})")

    wild_graph = Data(
        res_stru_fea=wild_structure.float(),
        pos=wild_positions,
        edge_index=build_edges(wild_positions, args.edge_cutoff),
        y=torch.tensor(0.0),
        res_esm1v_fea=wild_sequence,
        num_residue=torch.tensor(wild_positions.shape[0]),
        tag=tag,
        mut_mpos=wild_mutation_index,
    )
    mutant_graph = Data(
        res_stru_fea=mutant_structure.float(),
        pos=mutant_positions,
        edge_index=build_edges(mutant_positions, args.edge_cutoff),
        res_esm1v_fea=mutant_sequence,
        num_residue=torch.tensor(mutant_positions.shape[0]),
        tag=tag,
        mut_mpos=mutant_mutation_index,
    )

    destination = args.output_dir / f"{tag}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        return destination
    temporary = destination.with_suffix(".pt.tmp")
    torch.save((wild_graph, mutant_graph), temporary)
    temporary.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", required=True, type=Path)
    parser.add_argument("--tag-column", default="Tag")
    parser.add_argument("--wild-structure-dir", required=True, type=Path)
    parser.add_argument("--mutant-structure-dir", required=True, type=Path)
    parser.add_argument("--wild-position-dir", required=True, type=Path)
    parser.add_argument("--mutant-position-dir", required=True, type=Path)
    parser.add_argument("--wild-sequence-dir", required=True, type=Path)
    parser.add_argument("--mutant-sequence-dir", required=True, type=Path)
    parser.add_argument("--wild-mutation-index-dir", required=True, type=Path)
    parser.add_argument("--mutant-mutation-index-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--edge-cutoff", type=float, default=8.0)
    parser.add_argument("--structure-dimension", type=int, default=1280)
    parser.add_argument("--sequence-length", type=int, default=161)
    parser.add_argument("--sequence-dimension", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tags = load_tags(args.tags, args.tag_column)
    completed: list[str] = []
    failed: dict[str, str] = {}
    for tag in tqdm(tags, desc="Assembling model features"):
        try:
            assemble_one(tag, args)
            completed.append(tag)
        except Exception as exc:
            failed[tag] = str(exc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Assembled {len(completed)}/{len(tags)} final feature files.")
    if failed:
        for tag, message in failed.items():
            print(f"{tag}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
