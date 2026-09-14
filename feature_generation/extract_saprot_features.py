import os
import sys
from pathlib import Path
import time
import argparse
import traceback
import difflib
from functools import partial
import warnings
import subprocess
import tempfile

import torch
import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm
import multiprocessing as mp

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

warnings.simplefilter('ignore', PDBConstructionWarning)
mp.set_start_method('spawn', force=True)


def get_structure_sequences(foldseek_executable, pdb_path, chains=None):
    foldseek = Path(foldseek_executable)
    structure_path = Path(pdb_path)
    if not foldseek.is_file():
        raise FileNotFoundError(f"Foldseek executable does not exist: {foldseek}")
    if not structure_path.is_file():
        raise FileNotFoundError(f"PDB structure does not exist: {structure_path}")
    with tempfile.TemporaryDirectory(prefix="pathomenu_foldseek_") as temporary_directory:
        output_path = Path(temporary_directory) / "structure_sequences.tsv"
        subprocess.run([
            str(foldseek), "structureto3didescriptor", "-v", "0", "--threads", "1",
            "--chain-name-mode", "1", str(structure_path), str(output_path),
        ], check=True, capture_output=True, text=True)
        selected_chains = None if chains is None else set(chains)
        sequence_data = {}
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                description, sequence, structure_sequence = line.rstrip("\n").split("\t")[:3]
                description_name = description.split(" ", 1)[0]
                chain = description_name.replace(structure_path.name, "").split("_")[-1]
                if selected_chains is not None and chain not in selected_chains:
                    continue
                if chain in sequence_data:
                    continue
                combined = "".join(
                    amino_acid + structure_token.lower()
                    for amino_acid, structure_token in zip(sequence, structure_sequence)
                )
                sequence_data[chain] = (sequence, structure_sequence, combined)
    return sequence_data


def load_saprot_backbone(model_directory, device):
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

    model_path = Path(model_directory)
    checkpoint_path = model_path / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SaProt checkpoint does not exist: {checkpoint_path}")
    tokenizer = EsmTokenizer.from_pretrained(str(model_path))
    configuration = EsmConfig.from_pretrained(str(model_path))
    masked_language_model = EsmForMaskedLM(configuration)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    load_result = masked_language_model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "esm.embeddings.position_embeddings.weight",
        "esm.contact_head.regression.bias",
        "esm.contact_head.regression.weight",
    }
    allowed_unexpected = {"esm.embeddings.position_ids"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing
    unexpected_keys = set(load_result.unexpected_keys) - allowed_unexpected
    if unexpected_missing or unexpected_keys:
        raise RuntimeError(
            "SaProt checkpoint does not match its configuration: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(unexpected_keys)}"
        )
    backbone = masked_language_model.esm.to(device).eval()
    del masked_language_model, state_dict
    return backbone, tokenizer



def aa_three_to_one(three_letter):
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O'
    }
    return mapping.get(three_letter.upper(), 'X')


def get_full_residue_id(residue):
    res_id_tuple = residue.get_id()
    res_seq = res_id_tuple[1]
    insertion_code = res_id_tuple[2].strip()
    return f"{res_seq}{insertion_code}" if insertion_code else str(res_seq)


def create_aligned_tensor(a: str, b: str, tensor_b: torch.Tensor) -> torch.Tensor:
    if len(b) != tensor_b.shape[0]:
        raise ValueError(f"Length of string 'b' ({len(b)}) must match the first dimension of 'tensor_b' ({tensor_b.shape[0]}).")
    if tensor_b.ndim != 2:
        raise ValueError("'tensor_b' must be a two-dimensional tensor.")

    feature_dim = tensor_b.shape[1] if tensor_b.shape[0] > 0 else 0
    zero_vector = torch.zeros(feature_dim, dtype=tensor_b.dtype, device=tensor_b.device)

    if not a:
        return torch.empty((0, feature_dim), dtype=tensor_b.dtype, device=tensor_b.device)
    if not b:
        return zero_vector.unsqueeze(0).expand(len(a), -1)

    tensor_a_rows = []
    matcher = difflib.SequenceMatcher(None, b, a, autojunk=False)

    for tag, b_start, b_end, a_start, a_end in matcher.get_opcodes():
        if tag == 'equal':
            tensor_a_rows.append(tensor_b[b_start:b_end])
        elif tag == 'delete':
            pass
        else:
            num_fillers = a_end - a_start
            if num_fillers > 0:
                filler = zero_vector.unsqueeze(0).expand(num_fillers, -1)
                tensor_a_rows.append(filler)

    if not tensor_a_rows:
        return zero_vector.unsqueeze(0).expand(len(a), -1)

    final_tensor = torch.cat(tensor_a_rows, dim=0)
    assert final_tensor.shape[0] == len(a), \
        f"Internal error: final tensor length ({final_tensor.shape[0]}) does not match target string 'a' ({len(a)})."
    return final_tensor



def process_single_file(task_args, config):
    sample_id, local_structure_path, pdb_path, embedding_path, coords_path, mut_idx_path = task_args
    try:
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Corresponding standard PDB file was not found: {pdb_path}")

        parser = PDBParser()
        local_structure = parser.get_structure("local_structure", local_structure_path)

        local_to_feature_position, metal_ion_positions, local_structure_chains = {}, {}, []
        metal_residues, aa_residues = [], []
        all_residue_coords = []
        i = 0

        for chain in local_structure[0]:
            local_structure_chains.append(chain.id)
            for residue in chain:
                res_name = residue.get_resname().upper()
                is_metal = residue.get_id()[0].startswith('H_')
                is_res = is_aa(residue, standard=False)

                if is_res or is_metal:
                    coord = None
                    if 'CA' in residue:
                        coord = residue['CA'].get_coord()
                    else:
                        atom_coords = [atom.get_coord() for atom in residue.get_iterator()]
                        if atom_coords:
                            coord = np.mean(atom_coords, axis=0)

                    if coord is None:
                        coord = np.array([0.0, 0.0, 0.0])
                    all_residue_coords.append(coord)

                    full_residue_id = get_full_residue_id(residue)
                    tag = f"{chain.id}_{full_residue_id}"
                    local_to_feature_position[tag] = i
                    if is_metal:
                        metal_ion_positions[tag] = {'metal_type': res_name, 'position': i}
                        metal_residues.append(residue)
                    elif is_res:
                        aa_residues.append(residue)
                    i += 1

        if len(all_residue_coords) != i:
            raise ValueError(f"Coordinate count ({len(all_residue_coords)}) does not match residue count ({i}).")

        if i == 0:
            raise ValueError("No valid metal or amino-acid residues were found in the local-structure PDB file.")

        parts = sample_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"Sample tag '{sample_id}' has an invalid format and its mutation cannot be parsed.")

        mutation_info_str = parts[1]
        mutation_chain_id = parts[3]

        if len(mutation_info_str) < 7:
            raise ValueError(f"Mutation string '{mutation_info_str}' is too short to parse.")

        mutation_residue_pos = mutation_info_str[3:-3]

        if not mutation_residue_pos:
            raise ValueError(f"No mutation position could be parsed from '{mutation_info_str}'.")

        mutation_site_tag = f"{mutation_chain_id}_{mutation_residue_pos}"

        metal_neighbor_aas = {}
        for metal_res in metal_residues:
            metal_full_id = get_full_residue_id(metal_res)
            metal_tag = f"{metal_res.get_parent().id}_{metal_full_id}"
            metal_atom = next(metal_res.get_iterator())

            neighbor_list = []
            for aa_res in aa_residues:
                min_dist = min(np.linalg.norm(metal_atom.get_coord() - aa_atom.get_coord()) for aa_atom in aa_res)
                if min_dist < config['distance_threshold']:
                    aa_full_id = get_full_residue_id(aa_res)
                    neighbor_list.append(f"{aa_res.get_parent().id}_{aa_full_id}")
            metal_neighbor_aas[metal_tag] = neighbor_list

        pdb_id = parts[0].upper() if config['structure_kind'] == 'wild_type' else sample_id

        foldseek_data = get_structure_sequences(config['foldseek_path'], pdb_path, local_structure_chains)
        struct_seq_dict = {f"{pdb_id}_{chain_id}": data[2] for chain_id, data in foldseek_data.items()}
        foldseek_seq_dict = {f"{pdb_id}_{chain_id}": data[0] for chain_id, data in foldseek_data.items()}

        if not struct_seq_dict:
            raise ValueError("Foldseek did not extract a sequence for any chain in the PDB file.")

        standard_structure = parser.get_structure("standard_pdb", pdb_path)
        need_pdbchain_2_pdbpos = {}
        pdb_seq_dict = {}
        for chain in standard_structure[0]:
            pdbchain_tag = f"{pdb_id}_{chain.id}"
            if pdbchain_tag in struct_seq_dict:
                valid_residues = [res for res in chain if is_aa(res, standard=False)]
                need_pdbchain_2_pdbpos[pdbchain_tag] = ",".join([get_full_residue_id(res) for res in valid_residues])
                pdb_seq_dict[pdbchain_tag] = "".join([aa_three_to_one(res.get_resname()) for res in valid_residues])

        return {
            "status": "success",
            "data": {
                "prefix": sample_id, "res_num": i, "local_to_feature_position": local_to_feature_position,
                "metal_ion_positions": metal_ion_positions, "struct_seq_dict": struct_seq_dict,
                "need_pdbchain_2_pdbpos": need_pdbchain_2_pdbpos, "metal_neighbor_aas": metal_neighbor_aas,
                'foldseek_seq_dict': foldseek_seq_dict, 'pdb_seq_dict': pdb_seq_dict,
                "all_residue_coords": np.array(all_residue_coords),
                "mutation_site_tag": mutation_site_tag,
                "embedding_path": embedding_path,
                "coords_path": coords_path,
                "mut_idx_path": mut_idx_path,
            }
        }
    except Exception as e:
        return {'status': 'error', 'prefix': sample_id, 'error': str(e), 'traceback': traceback.format_exc()}


def cpu_worker_wrapper(task_args, task_queue, config):
    result = process_single_file(task_args, config)
    if result['status'] == 'success':
        task_queue.put(result['data'])
    return result



def gpu_worker(task_queue, config):
    try:
        device = torch.device(config['gpu_device'])
        model, tokenizer = load_saprot_backbone(config['model_config_path'], device)

        def get_hidden_states(inputs):
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            eos_positions = (inputs["input_ids"] == tokenizer.eos_token_id).int().argmax(dim=-1)
            return [
                hidden_states[index][1:int(eos_position.item())]
                for index, eos_position in enumerate(eos_positions)
            ]

        print("GPU worker: model loaded successfully.")
    except Exception:
        print("GPU worker fatal error: unable to load the model; exiting.")
        traceback.print_exc()
        return

    while True:
        batch = []
        try:
            while len(batch) < config['gpu_batch_size']:
                task = task_queue.get(timeout=10)
                if task is None:
                    task_queue.put(None)
                    break
                batch.append(task)
            if not batch and task is None:
                break
        except Exception:
            if not batch:
                if task_queue.empty():
                    time.sleep(5)
                    if task_queue.empty():
                        break
                continue

        if not batch:
            continue

        all_sequences_to_embed = [seq for item in batch for seq in item['struct_seq_dict'].values()]
        batch_succeeded = False
        batch_embeddings = None

        try:
            with torch.no_grad():
                inputs = tokenizer(all_sequences_to_embed, return_tensors="pt", padding=True).to(device)
                batch_embeddings = get_hidden_states(inputs)
            batch_succeeded = True
            print(f"GPU: processed a batch of {len(batch)} successfully.")
        except Exception as batch_error:
            error_info = traceback.format_exc()
            print(f"\n--- GPU batch inference failed; retrying items individually ---\n{error_info}\n----------------------\n")
            with open(config['error_log_file'], 'a') as f:
                f.write(f"--- GPU batch inference failed ---\n{error_info}Affected file prefixes:\n")
                for item in batch: f.write(f"- {item['prefix']}\n")
                f.write("----------------------\n\n")

        emb_idx = 0
        for item in batch:
            try:
                num_chains_in_item = len(item['struct_seq_dict'])
                item_embeddings_list = []

                if batch_succeeded:
                    item_embeddings_list = [batch_embeddings[i].cpu() for i in
                                            range(emb_idx, emb_idx + num_chains_in_item)]
                else:
                    try:
                        sequences_for_item = list(item['struct_seq_dict'].values())
                        with torch.no_grad():
                            inputs = tokenizer(sequences_for_item, return_tensors="pt", padding=True).to(device)
                            item_embeddings_tensor = get_hidden_states(inputs)
                            item_embeddings_list = [emb.cpu() for emb in item_embeddings_tensor]
                    except Exception:
                        error_info = traceback.format_exc()
                        print(
                            f"\n--- GPU per-sample fallback retry failed ---\nPrefix: {item['prefix']}\n{error_info}\n----------------------\n")
                        with open(config['error_log_file'], 'a') as f:
                            f.write(
                                f"--- Per-sample fallback retry failed ---\nFile prefix: {item['prefix']}\nError details: {error_info}\n----------------------\n\n")
                        emb_idx += num_chains_in_item
                        continue

                aligned_embeddings = {}
                pdbchain_tags = list(item['struct_seq_dict'].keys())
                for i, pdbchain_tag in enumerate(pdbchain_tags):
                    embedding_b = item_embeddings_list[i]
                    seq_b = item['foldseek_seq_dict'].get(pdbchain_tag, "")
                    seq_a = item['pdb_seq_dict'].get(pdbchain_tag, "")
                    if len(seq_b) != embedding_b.shape[0]:
                        raise ValueError(
                            f"Foldseek sequence length ({len(seq_b)}) for chain {pdbchain_tag} does not match its embedding length ({embedding_b.shape[0]}).")
                    aligned_emb = create_aligned_tensor(seq_a, seq_b, embedding_b)
                    pdbpos_list = item['need_pdbchain_2_pdbpos'].get(pdbchain_tag, "").split(',')
                    if '' in pdbpos_list and len(pdbpos_list) == 1: pdbpos_list = []
                    if aligned_emb.shape[0] != len(pdbpos_list):
                        raise ValueError(
                            f"Aligned embedding length ({aligned_emb.shape[0]}) for chain {pdbchain_tag} does not match the PDB residue count ({len(pdbpos_list)}).")
                    aligned_embeddings[pdbchain_tag] = (aligned_emb, pdbpos_list)

                final_embeddings = [None] * item['res_num']
                for key, pos_in_final in item['local_to_feature_position'].items():
                    if key in item['metal_ion_positions']: continue
                    chain_id_from_key, res_pos_from_key = key.split('_')
                    found = False
                    for pdbchain_tag, (emb, pos_list) in aligned_embeddings.items():
                        chain_id_from_list = pdbchain_tag.split('_')[-1]
                        if chain_id_from_key == chain_id_from_list:
                            try:
                                embed_idx_in_chain = pos_list.index(res_pos_from_key)
                                final_embeddings[pos_in_final] = emb[embed_idx_in_chain]
                                found = True
                                break
                            except ValueError:
                                continue
                    if not found:
                        pass

                embed_dim = next((t.shape[0] for t in final_embeddings if t is not None), 1280)
                for metal_tag, metal_info in item['metal_ion_positions'].items():
                    position = metal_info['position']
                    neighbor_aa_tags = item['metal_neighbor_aas'].get(metal_tag, [])
                    neighbor_embeddings = [final_embeddings[item['local_to_feature_position'][aa_tag]] for aa_tag in
                                           neighbor_aa_tags if aa_tag in item['local_to_feature_position'] and final_embeddings[
                                               item['local_to_feature_position'][aa_tag]] is not None]
                    if neighbor_embeddings:
                        final_embeddings[position] = torch.stack(neighbor_embeddings).mean(dim=0)
                    else:
                        final_embeddings[position] = torch.zeros(embed_dim)

                for i in range(len(final_embeddings)):
                    if final_embeddings[i] is None:
                        final_embeddings[i] = torch.zeros(embed_dim)

                embedding_tensor = torch.stack(final_embeddings).to(torch.float32)
                if embedding_tensor.shape[0] != item['res_num']:
                    raise ValueError(
                        f"Final embedding count validation failed: expected {item['res_num']}, got {embedding_tensor.shape[0]}.")

                coords_tensor = torch.from_numpy(item['all_residue_coords']).to(torch.float32)
                if coords_tensor.shape[0] != item['res_num']:
                    raise ValueError(f"Coordinate count validation failed: expected {item['res_num']}, got {coords_tensor.shape[0]}.")

                mutation_site_index = item['local_to_feature_position'].get(item['mutation_site_tag'])
                if mutation_site_index is None:
                    raise ValueError(f"Mutation site '{item['mutation_site_tag']}' was not found in the residue map.")
                mutation_idx_tensor = torch.tensor(mutation_site_index, dtype=torch.long)

                torch.save(embedding_tensor, item['embedding_path'])
                torch.save(coords_tensor, item['coords_path'])
                torch.save(mutation_idx_tensor, item['mut_idx_path'])

            except Exception:
                error_info = traceback.format_exc()
                print(
                    f"\n--- GPU per-sample postprocessing error ---\nPrefix: {item['prefix']}\n{error_info}\n----------------------\n")
                with open(config['error_log_file'], 'a') as f:
                    f.write(
                        f"--- GPU per-sample postprocessing error ---\nFile prefix: {item['prefix']}\nError details: {error_info}\n----------------------\n\n")
            finally:
                if batch_succeeded:
                    emb_idx += num_chains_in_item

    print("GPU worker: all queued tasks processed; exiting normally.")



def main():
    parser = argparse.ArgumentParser(description="Generate protein structure embeddings, coordinates, and mutation information from PDB files in parallel.")
    parser.add_argument('--structure-kind', choices=('wild_type', 'mutant'), required=True)
    parser.add_argument('--local-structure-dir', dest='local_structure_dir', type=str, required=True,
                        help="Directory containing 128-residue local-structure PDB files.")
    parser.add_argument('--full-structure-dir', dest='pdb_folder', type=str, required=True,
                        help="Directory containing standard PDB files.")

    parser.add_argument('--embedding-output-dir', dest='embedding_output_dir', type=str, required=True,
                        help="Directory for embedding files.")
    parser.add_argument('--coordinate-output-dir', dest='coords_output_dir', type=str, required=True,
                        help="Directory for coordinate files.")
    parser.add_argument('--mutation-index-output-dir', dest='mut_idx_output_dir', type=str, required=True,
                        help="Directory for mutation-site index files.")

    parser.add_argument('--error-log', dest='error_log_file', type=str, required=True,
                        help="Path to the error log file.")
    parser.add_argument('--distance-threshold', dest='distance_threshold', type=float, default=3.0, help="Strict first-coordination-shell distance threshold for metal-node embeddings.")
    parser.add_argument('--saprot-model-dir', dest='model_config_path', type=str, required=True,
                        help="Path to model configuration and weights.")
    parser.add_argument('--foldseek-executable', dest='foldseek_path', type=str, required=True, help="Path to the Foldseek executable.")
    parser.add_argument('--num-cpu-workers', dest='num_cpu_workers', type=int, default=8, help="Number of CPU preprocessing workers.")
    parser.add_argument('--gpu-device', dest='gpu_device', type=str, default="cuda:0", help="GPU device used for inference.")
    parser.add_argument('--gpu-batch-size', dest='gpu_batch_size', type=int, default=8, help="GPU inference batch size.")
    parser.add_argument('--queue-size', dest='queue_size', type=int, default=16, help="Maximum producer-consumer queue size.")

    args = parser.parse_args()
    config = vars(args)

    start_time = time.time()
    os.makedirs(config['embedding_output_dir'], exist_ok=True)
    os.makedirs(config['coords_output_dir'], exist_ok=True)
    os.makedirs(config['mut_idx_output_dir'], exist_ok=True)
    error_log_parent = os.path.dirname(config['error_log_file'])
    if error_log_parent:
        os.makedirs(error_log_parent, exist_ok=True)
    with open(config['error_log_file'], 'w') as f:
        f.write(f"--- Multiprocess embedding generation log ---\nStart time: {time.ctime()}\nConfiguration: {config}\n\n")

    tasks = []
    pdb_files = [os.path.join(root, f) for root, _, files in os.walk(config['local_structure_dir']) for f in files if
                 f.endswith('.pdb')]

    for pdb_file_path in pdb_files:
        sample_id = os.path.splitext(os.path.basename(pdb_file_path))[0]

        embedding_path = os.path.join(config['embedding_output_dir'], f"{sample_id}.pt")
        coords_path = os.path.join(config['coords_output_dir'], f"{sample_id}.pt")
        mut_idx_path = os.path.join(config['mut_idx_output_dir'], f"{sample_id}.pt")

        if os.path.exists(embedding_path) and os.path.exists(coords_path) and os.path.exists(mut_idx_path):
            continue

        try:
            pdb_id = sample_id.split('_')[0].upper() if config['structure_kind'] == 'wild_type' else sample_id
            pdb_path = os.path.join(config['pdb_folder'], f"{pdb_id}.pdb")
            tasks.append((sample_id, pdb_file_path, pdb_path, embedding_path, coords_path, mut_idx_path))
        except IndexError:
            print(f"Invalid filename format; skipping: {pdb_file_path}")
            continue

    if not tasks:
        print("No new tasks require processing.")
        return

    print(f"Found {len(tasks)} tasks to process.")

    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=config['queue_size'])

    gpu_process = mp.Process(target=gpu_worker, args=(task_queue, config))
    gpu_process.start()
    print("GPU consumer process started.")

    print(f"Starting {config['num_cpu_workers']} CPU workers for preprocessing...")
    cpu_success_count = 0
    cpu_error_count = 0
    with mp.Pool(processes=config['num_cpu_workers']) as pool:
        worker_func = partial(cpu_worker_wrapper, task_queue=task_queue, config=config)
        with tqdm(total=len(tasks), desc="CPU Pre-processing") as pbar:
            for result in pool.imap_unordered(worker_func, tasks):
                if result['status'] == 'success':
                    cpu_success_count += 1
                else:
                    cpu_error_count += 1
                    with open(config['error_log_file'], 'a') as f:
                        f.write(
                            f"--- CPU preprocessing error ---\nFile prefix: {result['prefix']}\nError: {result['error']}\n{result.get('traceback', '')}\n------------------\n\n")
                pbar.update(1)

    print("All CPU tasks completed. Sending the termination signal to the GPU process...")
    task_queue.put(None)
    gpu_process.join()
    print("GPU process has exited.")

    missing_outputs = [
        path
        for task in tasks
        for path in task[3:6]
        if not os.path.isfile(path)
    ]
    if missing_outputs:
        raise RuntimeError(f"SaProt outputs are missing: {missing_outputs}")

    end_time = time.time()
    print("\n--- All tasks completed ---")
    print(f"Total elapsed time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"CPU preprocessing: {cpu_success_count} succeeded, {cpu_error_count} failed.")
    print(f"See '{config['error_log_file']}' for detailed error information.")


if __name__ == '__main__':
    main()
