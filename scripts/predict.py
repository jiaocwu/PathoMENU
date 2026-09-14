#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = os.environ.get("PATHOMENU_MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import requests
import torch
from Bio.PDB import PDBParser
from torch.utils.data import DataLoader
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from feature_generation.local_structure_utils import FIRST_COORDINATION_THRESHOLD, METAL_ELEMENTS, SECOND_COORDINATION_THRESHOLD, mutation_coordination_metals
from models.PathoMENU import PathoMENU
from models.structure_encoder import PathoMENUStructureEncoder
from scripts.generate_prediction_features import generate_features
from scripts.dataloader import PredictionPairDataset, graph_pair_collate
from scripts.score_calibration import apply_pathomenu_calibration, load_calibration_parameters


CHECKPOINT_PATH = PACKAGE_ROOT / "weights" / "PathoMENU_weight.pth"
CALIBRATION_PATH = PACKAGE_ROOT / "scripts" / "PathoMENU_smooth6_pchip.json"
DEFAULT_FEATURE_DIR = PACKAGE_ROOT / "features"
STRUCTURE_PARAMETERS = {
    "input_dim": 1280,
    "hidden_channels": 128,
    "num_layers": 2,
    "num_radial": 64,
    "cutoff": 8.0,
    "num_heads": 8,
    "lmax": 2,
    "y_mean": 0,
    "y_std": 1,
    "pos_require_grad": False,
    "readout": "sum",
    "dropout": 0.45,
}
FUSION_CONFIG = {
    "params": {
        "mlp_dropout": 0.45,
        "transformer_dropout": 0.45,
        "focal_loss_gamma": 2.0,
        "sequence_encoder_layers": 2,
        "sequence_encoder_heads": 8,
        "sequence_encoder_hidden_dim": 512,
        "esm1v_input_dim": 1280,
        "lambda_auxiliary_prediction": 0.01,
        "lambda_gate_ranking": 0.015,
        "lambda_orthogonality": 0.0005,
        "lambda_alignment": 0.002,
        "rank_margin": 0.015,
        "aux_warmup_epochs": 5,
        "aux_rampup_epochs": 10,
        "sequence_feature": "esm1v",
    }
}

REQUIRED_COLUMNS = (
    "pdb_id", "sequence", "pdb_position", "sequence_position", "metal ion", "wt_aa", "mut_aa",
)
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
THREE_TO_ONE = {three: one for one, three in ONE_TO_THREE.items()}
ALLOWED_SEQUENCE_CODES = set("ABCDEFGHIKLMNPQRSTVWXYZUOJ")
PDB_POSITION_PATTERN = re.compile(r"^(-?\d+)([A-Za-z]?)$")
PREDICTION_STATUS_COLUMN = "prediction_status"
PREDICTION_MESSAGE_COLUMN = "prediction_message"
INPUT_ROW_COLUMN = "input_row"


class PDBResidueIdentityMismatchError(ValueError):

    def __init__(self, message: str, fallback_chain: str = "NA") -> None:
        super().__init__(message)
        self.fallback_chain = fallback_chain


def input_row_text(row: pd.Series) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow([row[column] for column in REQUIRED_COLUMNS])
    return buffer.getvalue()


def required_text(value, column: str, row_number: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row_number}: {column} is empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_number}: {column} is empty")
    if re.fullmatch(r"-?[0-9]+\.0", text):
        text = text[:-2]
    return text


def normalize_sequence(value: str, row_number: int) -> str:
    text = required_text(value, "sequence", row_number).replace("\\n", "\n").replace("\\r", "\r")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        if any(line.startswith(">") for line in lines[1:]):
            raise ValueError(f"Row {row_number}: sequence must contain exactly one FASTA record")
        lines = lines[1:]
    sequence = re.sub(r"\s+", "", "".join(lines)).upper()
    if not sequence:
        raise ValueError(f"Row {row_number}: sequence contains no amino acids")
    invalid = sorted(set(sequence) - ALLOWED_SEQUENCE_CODES)
    if invalid:
        raise ValueError(f"Row {row_number}: sequence contains unsupported amino-acid codes: {invalid}")
    return sequence


def parse_pdb_position(value: str, row_number: int) -> tuple[int, str, str]:
    text = required_text(value, "pdb_position", row_number)
    match = PDB_POSITION_PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"Row {row_number}: pdb_position must be an integer with an optional insertion code")
    residue_number = int(match.group(1))
    insertion_code = match.group(2).upper() or " "
    return residue_number, insertion_code, f"{residue_number}{insertion_code.strip()}"


def download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    pdb_dir.mkdir(parents=True, exist_ok=True)
    destination = pdb_dir / f"{pdb_id}.pdb"
    if destination.is_file() and destination.stat().st_size > 100:
        return destination
    response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
    response.raise_for_status()
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in response.text.splitlines()):
        raise ValueError(f"Downloaded PDB file contains no atoms: {pdb_id}")
    temporary = destination.with_suffix(".pdb.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def resolve_structure_chain(pdb_path: Path, pdb_position: str, expected_wild_type: str, row_number: int) -> str:
    residue_number, insertion_code, _ = parse_pdb_position(pdb_position, row_number)
    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"Row {row_number}: PDB structure contains no models: {pdb_path}") from exc
    candidates = []
    matching_chains = []
    residue_id = (" ", residue_number, insertion_code)
    for chain in model:
        if residue_id not in chain:
            continue
        observed_three = chain[residue_id].get_resname().upper()
        observed_one = THREE_TO_ONE.get(observed_three, observed_three)
        candidates.append((chain.id, observed_one))
        if observed_one == expected_wild_type:
            matching_chains.append(chain.id)
    if not matching_chains:
        observed = ", ".join(f"chain {chain}: {amino_acid}" for chain, amino_acid in candidates)
        fallback_chain = candidates[0][0] if candidates else "NA"
        raise PDBResidueIdentityMismatchError(
            f"The residue at PDB position {pdb_position} does not match wt_aa {expected_wild_type} "
            f"(observed {observed or 'position absent from all chains'}).",
            fallback_chain=fallback_chain,
        )
    if len(matching_chains) > 1:
        warnings.warn(
            f"Row {row_number}: {pdb_path.stem}:{pdb_position} matches {expected_wild_type} "
            f"in multiple chains {matching_chains}; using the first chain ({matching_chains[0]}).",
            stacklevel=2,
        )
    return matching_chains[0]


def normalize_input_frame(input_csv: Path, wild_type_pdb_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    if frame.empty:
        raise ValueError(f"No variants were found in {input_csv}")
    normalized_rows = []
    for index, (_, row) in enumerate(frame.iterrows()):
        row_number = index + 2
        pdb_id = required_text(row["pdb_id"], "pdb_id", row_number).upper()
        if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
            raise ValueError(f"Row {row_number}: pdb_id must be a four-character PDB ID")
        sequence = normalize_sequence(row["sequence"], row_number)
        sequence_position_text = required_text(row["sequence_position"], "sequence_position", row_number)
        if not sequence_position_text.isdigit() or int(sequence_position_text) < 1:
            raise ValueError(f"Row {row_number}: sequence_position must be a positive integer")
        sequence_position = int(sequence_position_text)
        if sequence_position > len(sequence):
            raise ValueError(f"Row {row_number}: sequence_position {sequence_position} exceeds sequence length {len(sequence)}")
        wild_type = required_text(row["wt_aa"], "wt_aa", row_number).upper()
        mutant = required_text(row["mut_aa"], "mut_aa", row_number).upper()
        if wild_type not in ONE_TO_THREE or mutant not in ONE_TO_THREE:
            raise ValueError(f"Row {row_number}: wt_aa and mut_aa must be standard one-letter amino-acid codes")
        observed_sequence_residue = sequence[sequence_position - 1]
        identity_message = ""
        if observed_sequence_residue != wild_type:
            identity_message = (
                f"The residue at UniProt position {sequence_position} does not match "
                f"wt_aa {wild_type} (observed {observed_sequence_residue})."
            )
        residue_number, insertion_code, pdb_position = parse_pdb_position(row["pdb_position"], row_number)
        metal = required_text(row["metal ion"], "metal ion", row_number).upper()
        if metal not in METAL_ELEMENTS:
            raise ValueError(f"Row {row_number}: metal ion must be a supported metal element, for example CA or ZN")
        chain = "NA"
        related_metals = []
        if not identity_message:
            pdb_path = download_pdb(pdb_id, wild_type_pdb_dir)
            try:
                chain = resolve_structure_chain(pdb_path, pdb_position, wild_type, row_number)
            except PDBResidueIdentityMismatchError as exc:
                chain = exc.fallback_chain
                identity_message = str(exc)
            else:
                related_metals = mutation_coordination_metals(
                    pdb_path,
                    (chain, residue_number, insertion_code.strip()),
                    metal,
                    FIRST_COORDINATION_THRESHOLD,
                    SECOND_COORDINATION_THRESHOLD,
                )
        if identity_message:
            prediction_status = "skipped"
            prediction_message = identity_message
        elif not related_metals:
            prediction_status = "skipped"
            prediction_message = f"This residue {wild_type}{pdb_position} does not coordinate with the {metal} ion."
        else:
            prediction_status = ""
            prediction_message = ""
        normalized_rows.append({
            "pdb_id": pdb_id,
            "sequence": sequence,
            "pdb_position": pdb_position,
            "sequence_position": str(sequence_position),
            "metal ion": metal,
            "wt_aa": wild_type,
            "mut_aa": mutant,
            "chain": chain,
            PREDICTION_STATUS_COLUMN: prediction_status,
            PREDICTION_MESSAGE_COLUMN: prediction_message,
            INPUT_ROW_COLUMN: input_row_text(row),
        })
    return pd.DataFrame(normalized_rows)


def prepare_prediction_input(input_csv: Path, workspace: Path, wild_type_pdb_dir: Path):
    input_csv = input_csv.expanduser().resolve()
    frame = normalize_input_frame(input_csv, wild_type_pdb_dir)
    tags = []
    for _, row in frame.iterrows():
        mutation = f"{ONE_TO_THREE[row['wt_aa']]}{row['pdb_position']}{ONE_TO_THREE[row['mut_aa']]}"
        tags.append("_".join((row["pdb_id"], mutation, row["metal ion"], row["chain"])))
    result_frame = frame.copy()
    result_frame.insert(0, "sample_id", tags)
    eligible = result_frame[result_frame[PREDICTION_STATUS_COLUMN].eq("")].copy()
    if eligible["sample_id"].duplicated().any():
        duplicate_tags = eligible.loc[eligible["sample_id"].duplicated(False), "sample_id"].unique().tolist()
        raise ValueError(f"Duplicate feature names in prediction input: {duplicate_tags[:5]}")
    internal_dir = workspace / "input_variants"
    internal_dir.mkdir(parents=True, exist_ok=True)
    internal_csv = internal_dir / f"{input_csv.stem}.csv"
    eligible.to_csv(internal_csv, index=False)
    return result_frame, eligible["sample_id"].tolist(), internal_csv


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_model(device: torch.device) -> torch.nn.Module:
    model = PathoMENU(PathoMENUStructureEncoder(**STRUCTURE_PARAMETERS), FUSION_CONFIG).to(device)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=False)
    shared_keys = {f"classifier_shared.{key}" for key in model.classifier_shared.state_dict()}
    ranking_gate_keys = {
        f"delta_fusion.ranking_gate.{key}"
        for key in model.delta_fusion.ranking_gate.state_dict()
    }
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    allowed_missing = (set(), shared_keys, ranking_gate_keys, shared_keys | ranking_gate_keys)
    if unexpected_keys or missing_keys not in allowed_missing:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
        )
    if shared_keys.issubset(missing_keys):
        structure_state = model.classifier_struct.state_dict()
        sequence_state = model.classifier_esm.state_dict()
        shared_state = {
            key: 0.5 * (structure_state[key] + sequence_state[key])
            for key in model.classifier_shared.state_dict()
        }
        model.classifier_shared.load_state_dict(shared_state, strict=True)
    if ranking_gate_keys.issubset(missing_keys):
        fusion_gate = model.delta_fusion.gate
        ranking_gate = model.delta_fusion.ranking_gate
        hidden_dim = model.structure_model.output_dim
        with torch.no_grad():
            ranking_gate[0].weight.copy_(
                torch.cat([
                    fusion_gate[0].weight[:hidden_dim],
                    fusion_gate[0].weight[hidden_dim:],
                    0.5 * (
                        fusion_gate[0].weight[:hidden_dim]
                        + fusion_gate[0].weight[hidden_dim:]
                    ),
                ])
            )
            ranking_gate[0].bias.copy_(
                torch.cat([
                    fusion_gate[0].bias[:hidden_dim],
                    fusion_gate[0].bias[hidden_dim:],
                    0.5 * (
                        fusion_gate[0].bias[:hidden_dim]
                        + fusion_gate[0].bias[hidden_dim:]
                    ),
                ])
            )
            ranking_gate[1].weight[:, :hidden_dim].copy_(fusion_gate[1].weight[:, :hidden_dim])
            ranking_gate[1].weight[:, hidden_dim:2 * hidden_dim].copy_(fusion_gate[1].weight[:, hidden_dim:])
            ranking_gate[1].weight[:, 2 * hidden_dim:].copy_(
                0.5 * (
                    fusion_gate[1].weight[:, :hidden_dim]
                    + fusion_gate[1].weight[:, hidden_dim:]
                )
            )
            ranking_gate[1].bias.copy_(fusion_gate[1].bias)
            ranking_gate[4].weight[:2].copy_(fusion_gate[4].weight)
            ranking_gate[4].weight[2].copy_(fusion_gate[4].weight.mean(dim=0))
            ranking_gate[4].bias[:2].copy_(fusion_gate[4].bias)
            ranking_gate[4].bias[2].copy_(fusion_gate[4].bias.mean())
    if hasattr(model, "set_focal_loss_alpha"):
        model.set_focal_loss_alpha(0.5)
    if hasattr(model, "set_current_epoch"):
        model.set_current_epoch(int(checkpoint.get("epoch", 0)))
    return model.eval()


def predict_scores(tags: list[str], feature_dir: Path, device: torch.device, batch_size: int) -> dict[str, float]:
    if not tags:
        return {}
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dataset = PredictionPairDataset(tags, feature_dir, strict=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=graph_pair_collate,
    )
    model = load_model(device)
    calibration = load_calibration_parameters(CALIBRATION_PATH)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for collated in tqdm(loader, desc="Predicting"):
            if collated is None:
                continue
            wild, mutant, batch_tags = collated
            raw = model(wild.to(device), mutant.to(device))["predictions"].detach().cpu().numpy()
            calibrated = apply_pathomenu_calibration(raw, calibration)
            scores.update({tag: float(score) for tag, score in zip(batch_tags, calibrated, strict=True)})
    if len(scores) != len(tags):
        missing = [tag for tag in tags if tag not in scores]
        raise RuntimeError(f"Prediction failed for: {missing[:5]}")
    return scores


def write_predictions(output: Path, rows: pd.DataFrame, scores: dict[str, float]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["predicted_probability"])
        writer.writeheader()
        for _, row in rows.iterrows():
            status = str(row[PREDICTION_STATUS_COLUMN]).strip()
            writer.writerow({"predicted_probability": status or scores[str(row["sample_id"])]})
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--foldx-processes", type=int, default=18)
    parser.add_argument("--saprot-cpu-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    input_csv = arguments.variants_csv.expanduser().resolve()
    if not input_csv.is_file() or input_csv.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Prediction input must be an existing CSV file: {input_csv}")
    output = arguments.output.expanduser().resolve() if arguments.output else input_csv.parent / "predictions.csv"
    feature_dir = arguments.feature_dir.expanduser().resolve()
    device = resolve_device(arguments.device)
    with tempfile.TemporaryDirectory(prefix="pathomenu_predict_") as temporary_directory:
        workspace = Path(temporary_directory)
        rows, tags, internal_csv = prepare_prediction_input(input_csv, workspace, workspace / "full_structures" / "wild_type")
        skipped = rows[rows[PREDICTION_STATUS_COLUMN].ne("")]
        for _, row in skipped.iterrows():
            print(f"{json.dumps(row[INPUT_ROW_COLUMN], ensure_ascii=False)}: {row[PREDICTION_MESSAGE_COLUMN]}")
        missing_tags = [tag for tag in tags if not (feature_dir / f"{tag}.pt").is_file()]
        if missing_tags:
            missing_csv = workspace / "input_variants" / "missing_features.csv"
            frame = pd.read_csv(internal_csv, dtype=str, keep_default_na=False)
            frame[frame["sample_id"].isin(missing_tags)].to_csv(missing_csv, index=False)
            generate_features(missing_csv, workspace, feature_dir, str(device), arguments.foldx_processes, arguments.saprot_cpu_workers)
        scores = predict_scores(tags, feature_dir, device, arguments.batch_size)
        write_predictions(output, rows, scores)
    print(f"Prediction output: {output}")
    print(f"Final feature directory: {feature_dir}")


if __name__ == "__main__":
    main()
