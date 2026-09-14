#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "sample_id",
    "sequence",
    "sequence_position",
    "wt_aa",
    "mut_aa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-csv", required=True, type=Path)
    parser.add_argument("--wild-type-output-fasta", required=True, type=Path)
    parser.add_argument("--mutant-output-fasta", required=True, type=Path)
    parser.add_argument(
        "--flank-length",
        type=int,
        default=80,
        help="Residues retained on each side; 80 produces the 161-residue model input.",
    )
    return parser.parse_args()


def sequence_window(sequence: str, position: int, flank_length: int) -> str:
    if position < 1 or position > len(sequence):
        raise ValueError(f"position {position} is outside sequence length {len(sequence)}")
    center = position - 1
    left = sequence[max(0, center - flank_length):center]
    right = sequence[center + 1:center + 1 + flank_length]
    return left.rjust(flank_length, "X") + sequence[center] + right.ljust(flank_length, "X")


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, sequence in records:
            handle.write(f">{sample_id}\n{sequence}\n")


def main() -> None:
    args = parse_args()
    if args.flank_length < 0:
        raise ValueError("--flank-length must be non-negative")

    dataframe = pd.read_csv(args.variants_csv, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required internal CSV columns: {missing}")

    wild_type_records: list[tuple[str, str]] = []
    mutant_records: list[tuple[str, str]] = []
    for index, row in dataframe.iterrows():
        row_number = index + 2
        sample_id = row["sample_id"]
        sequence = row["sequence"].upper()
        position = int(row["sequence_position"])
        wild_type = row["wt_aa"].upper()
        mutant = row["mut_aa"].upper()
        observed = sequence[position - 1] if 1 <= position <= len(sequence) else None
        if observed != wild_type:
            raise ValueError(
                f"Row {row_number}: sequence residue mismatch at position {position}; "
                f"expected {wild_type}, observed {observed}"
            )

        mutant_sequence = sequence[:position - 1] + mutant + sequence[position:]
        wild_type_records.append((sample_id, sequence_window(sequence, position, args.flank_length)))
        mutant_type_window = sequence_window(mutant_sequence, position, args.flank_length)
        mutant_records.append((sample_id, mutant_type_window))

    write_fasta(wild_type_records, args.wild_type_output_fasta)
    write_fasta(mutant_records, args.mutant_output_fasta)
    print(
        f"Prepared {len(wild_type_records)} paired windows of length "
        f"{2 * args.flank_length + 1}."
    )


if __name__ == "__main__":
    main()
