#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from Bio import SeqIO
from tqdm import tqdm


ESM1V_LOADER = "esm1v_t33_650M_UR90S_1"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["esm1v"])
    parser.add_argument(
        "--model-weights",
        type=Path,
        help="Optional local fair-esm checkpoint; large ESM1v weights may be stored outside this package.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import esm

    if not hasattr(esm, "pretrained"):
        raise ImportError(
            "The installed 'esm' package is not fair-esm. Remove the conflicting esm package "
            "and install fair-esm==2.0.0."
        )

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if args.model_weights:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(args.model_weights))
    else:
        loader = getattr(esm.pretrained, ESM1V_LOADER)
        model, alphabet = loader()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(str(args.fasta), "fasta"))
    for record in tqdm(records, desc=f"Extracting {args.model} embeddings"):
        tag = str(record.id)
        destination = args.output_dir / f"{tag}.pt"
        if destination.exists() and not args.overwrite:
            continue
        sequence = str(record.seq).upper()
        _, _, tokens = batch_converter([(tag, sequence)])
        tokens = tokens.to(device)
        with torch.no_grad():
            representations = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        embedding = representations[0, 1 : len(sequence) + 1].detach().cpu().float()
        if embedding.shape[0] != len(sequence):
            raise ValueError(f"{tag}: embedding rows {embedding.shape[0]} != sequence length {len(sequence)}")
        torch.save(embedding, destination)


if __name__ == "__main__":
    main()
