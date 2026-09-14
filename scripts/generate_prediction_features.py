#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_generation.local_structure_utils import FIRST_COORDINATION_THRESHOLD, SECOND_COORDINATION_THRESHOLD


def run_stage(name: str, command: list[str], environment: dict[str, str]) -> None:
    print(f"\n[{name}]\n{' '.join(shlex.quote(part) for part in command)}", flush=True)
    subprocess.run(command, check=True, env=environment)


def generate_features(
    variants_csv: Path,
    workspace: Path,
    output_dir: Path,
    device: str,
    foldx_processes: int,
    saprot_cpu_workers: int,
) -> None:
    feature_scripts = PACKAGE_ROOT / "feature_generation"
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = environment.get("PATHOMENU_MKL_THREADING_LAYER", "GNU")
    paths = {
        "foldx": PACKAGE_ROOT / "tools" / "foldx",
        "foldseek": PACKAGE_ROOT / "tools" / "foldseek",
        "saprot": PACKAGE_ROOT / "weights" / "pretrained" / "SaProt_650M_AF2",
        "esm1v": PACKAGE_ROOT / "weights" / "pretrained" / "esm1v_t33_650M_UR90S_1.pt",
    }
    directories = {
        "sequence_windows": workspace / "sequence_windows",
        "wild_sequence": workspace / "sequence_embeddings" / "wild_type",
        "mutant_sequence": workspace / "sequence_embeddings" / "mutant",
        "wild_full": workspace / "full_structures" / "wild_type",
        "mutant_full": workspace / "full_structures" / "mutant",
        "wild_local": workspace / "local_structures" / "wild_type",
        "mutant_local": workspace / "local_structures" / "mutant",
        "wild_structure": workspace / "saprot_features" / "wild_type" / "embeddings",
        "mutant_structure": workspace / "saprot_features" / "mutant" / "embeddings",
        "wild_position": workspace / "saprot_features" / "wild_type" / "coordinates",
        "mutant_position": workspace / "saprot_features" / "mutant" / "coordinates",
        "wild_index": workspace / "saprot_features" / "wild_type" / "mutation_indices",
        "mutant_index": workspace / "saprot_features" / "mutant" / "mutation_indices",
        "logs": workspace / "logs",
        "foldx_work": workspace / "foldx_work",
    }
    wild_fasta = directories["sequence_windows"] / "wild_type.fasta"
    mutant_fasta = directories["sequence_windows"] / "mutant.fasta"
    run_stage("sequence_windows", [
        sys.executable, str(feature_scripts / "prepare_sequence_windows.py"),
        "--variants-csv", str(variants_csv),
        "--wild-type-output-fasta", str(wild_fasta),
        "--mutant-output-fasta", str(mutant_fasta),
        "--flank-length", "80",
    ], environment)
    for name, fasta, destination in (
        ("wild_type_sequence_embeddings", wild_fasta, directories["wild_sequence"]),
        ("mutant_sequence_embeddings", mutant_fasta, directories["mutant_sequence"]),
    ):
        run_stage(name, [
            sys.executable, str(feature_scripts / "extract_sequence_embeddings.py"),
            "--fasta", str(fasta), "--output-dir", str(destination),
            "--model", "esm1v", "--model-weights", str(paths["esm1v"]), "--device", device,
        ], environment)
    run_stage("wild_type_local_structures", [
        sys.executable, str(feature_scripts / "prepare_wild_type_local_structures.py"),
        "--variants-csv", str(variants_csv), "--wild-type-pdb-dir", str(directories["wild_full"]),
        "--output-dir", str(directories["wild_local"]), "--residue-count", "128",
        "--first-coordination-threshold", str(FIRST_COORDINATION_THRESHOLD),
        "--second-coordination-threshold", str(SECOND_COORDINATION_THRESHOLD),
    ], environment)
    run_stage("mutant_full_structures", [
        sys.executable, str(feature_scripts / "generate_mutant_structures_with_foldx.py"),
        "--variants-csv", str(variants_csv), "--wild-type-pdb-dir", str(directories["wild_full"]),
        "--mutant-pdb-output-dir", str(directories["mutant_full"]), "--foldx-executable", str(paths["foldx"]),
        "--work-dir", str(directories["foldx_work"]), "--error-log", str(directories["logs"] / "foldx.tsv"),
        "--num-processes", str(foldx_processes),
    ], environment)
    run_stage("mutant_local_structures", [
        sys.executable, str(feature_scripts / "prepare_mutant_local_structures.py"),
        "--variants-csv", str(variants_csv), "--mutant-pdb-dir", str(directories["mutant_full"]),
        "--wild-type-pdb-dir", str(directories["wild_full"]), "--output-dir", str(directories["mutant_local"]),
        "--residue-count", "128", "--first-coordination-threshold", str(FIRST_COORDINATION_THRESHOLD),
        "--second-coordination-threshold", str(SECOND_COORDINATION_THRESHOLD),
    ], environment)
    for name, structure_kind, local_dir, full_dir, prefix in (
        ("wild_type_saprot_features", "wild_type", directories["wild_local"], directories["wild_full"], "wild"),
        ("mutant_saprot_features", "mutant", directories["mutant_local"], directories["mutant_full"], "mutant"),
    ):
        run_stage(name, [
            sys.executable, str(feature_scripts / "extract_saprot_features.py"),
            "--structure-kind", structure_kind, "--local-structure-dir", str(local_dir),
            "--full-structure-dir", str(full_dir), "--embedding-output-dir", str(directories[f"{prefix}_structure"]),
            "--coordinate-output-dir", str(directories[f"{prefix}_position"]),
            "--mutation-index-output-dir", str(directories[f"{prefix}_index"]),
            "--error-log", str(directories["logs"] / f"saprot_{prefix}.log"),
            "--distance-threshold", str(FIRST_COORDINATION_THRESHOLD), "--saprot-model-dir", str(paths["saprot"]),
            "--foldseek-executable", str(paths["foldseek"]), "--num-cpu-workers", str(saprot_cpu_workers),
            "--gpu-device", device, "--gpu-batch-size", "1", "--queue-size", "16",
        ], environment)
    run_stage("assemble_model_features", [
        sys.executable, str(feature_scripts / "assemble_model_features.py"), "--tags", str(variants_csv),
        "--tag-column", "sample_id", "--wild-structure-dir", str(directories["wild_structure"]),
        "--mutant-structure-dir", str(directories["mutant_structure"]),
        "--wild-position-dir", str(directories["wild_position"]), "--mutant-position-dir", str(directories["mutant_position"]),
        "--wild-sequence-dir", str(directories["wild_sequence"]), "--mutant-sequence-dir", str(directories["mutant_sequence"]),
        "--wild-mutation-index-dir", str(directories["wild_index"]),
        "--mutant-mutation-index-dir", str(directories["mutant_index"]), "--output-dir", str(output_dir),
        "--edge-cutoff", "8.0", "--structure-dimension", "1280",
        "--sequence-length", "161", "--sequence-dimension", "1280",
    ], environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants_csv", type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--foldx-processes", type=int, default=18)
    parser.add_argument("--saprot-cpu-workers", type=int, default=8)
    arguments = parser.parse_args()
    generate_features(arguments.variants_csv, arguments.workspace, arguments.output_dir, arguments.device, arguments.foldx_processes, arguments.saprot_cpu_workers)


if __name__ == "__main__":
    main()
