#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_generation.local_structure_utils import (
    FIRST_COORDINATION_THRESHOLD,
    SECOND_COORDINATION_THRESHOLD,
    prepare_local_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-csv", required=True, type=Path)
    parser.add_argument("--wild-type-pdb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--residue-count", type=int, default=128)
    parser.add_argument(
        "--first-coordination-threshold",
        "--metal-distance-threshold",
        dest="first_coordination_threshold",
        type=float,
        default=FIRST_COORDINATION_THRESHOLD,
    )
    parser.add_argument(
        "--second-coordination-threshold",
        type=float,
        default=SECOND_COORDINATION_THRESHOLD,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.residue_count < 1:
        raise ValueError("--residue-count must be at least 1")
    if args.first_coordination_threshold <= 0:
        raise ValueError("--first-coordination-threshold must be positive")
    if args.second_coordination_threshold <= 0:
        raise ValueError("--second-coordination-threshold must be positive")

    dataframe = pd.read_csv(args.variants_csv, dtype=str, keep_default_na=False)
    required = {"sample_id", "pdb_id", "pdb_position", "chain", "metal ion", "wt_aa"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required internal CSV columns: {missing}")

    failures: list[str] = []
    for index, row in dataframe.iterrows():
        sample_id = row["sample_id"]
        source = args.wild_type_pdb_dir / f"{row['pdb_id'].upper()}.pdb"
        destination = args.output_dir / f"{sample_id}.pdb"
        if destination.exists() and not args.overwrite:
            continue
        try:
            residue_count, metal_count = prepare_local_structure(
                full_pdb_path=source,
                metal_source_path=source,
                output_path=destination,
                chain_id=row["chain"],
                pdb_position=row["pdb_position"],
                expected_amino_acid=row["wt_aa"],
                specified_metal=row["metal ion"],
                residue_count=args.residue_count,
                first_coordination_threshold=args.first_coordination_threshold,
                second_coordination_threshold=args.second_coordination_threshold,
            )
            print(
                f"Saved {destination} ({residue_count} amino acids, "
                f"{metal_count} mutation-related metal ions)"
            )
        except Exception as exc:
            failures.append(f"row {index + 2} ({sample_id}): {exc}")

    if failures:
        raise RuntimeError("Failed to prepare wild-type local structures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
