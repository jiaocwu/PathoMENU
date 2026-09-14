import argparse
import os
import shutil
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Pool, Manager
import pandas as pd
import subprocess
import time
import urllib.request
import urllib.error
import random

CONFIG = {
    'pdb_folder': None,
    'csv_file': None,
    'output_folder': None,
    'foldx_path': None,
    'base_work_dir': None,
    'error_log_file': None,
    'num_processes': 18,
    'overwrite': False,
}


def setup_directories():
    os.makedirs(CONFIG['output_folder'], exist_ok=True)
    os.makedirs(CONFIG['base_work_dir'], exist_ok=True)
    os.makedirs(CONFIG['pdb_folder'], exist_ok=True)
    error_log_parent = os.path.dirname(CONFIG['error_log_file'])
    if error_log_parent:
        os.makedirs(error_log_parent, exist_ok=True)

    if os.path.exists(CONFIG['error_log_file']):
        try:
            os.remove(CONFIG['error_log_file'])
        except OSError:
            pass


def download_pdb_from_rcsb_safe(pdb_code, save_dir):
    if len(pdb_code) != 4:
        return False, f"Invalid PDB code length: {pdb_code}"

    pdb_code = pdb_code.upper()
    target_path = os.path.join(save_dir, f"{pdb_code}.pdb")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return True, None

    pid = os.getpid()
    rnd = random.randint(1000, 9999)
    temp_path = os.path.join(save_dir, f"{pdb_code}_{pid}_{rnd}.tmp")

    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"

    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(temp_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        if os.path.getsize(temp_path) < 100:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, "Downloaded file too small"

        os.replace(temp_path, target_path)
        return True, None

    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return False, f"HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return False, str(e)


def load_mutation_data():
    try:
        mutation_data = pd.read_csv(CONFIG['csv_file'], dtype=str, keep_default_na=False)
        required = {'sample_id', 'pdb_id', 'chain', 'pdb_position', 'wt_aa', 'mut_aa'}
        missing = sorted(required - set(mutation_data.columns))
        if missing:
            raise ValueError(f"Missing required internal CSV columns: {missing}")
        mutations_by_pdb = defaultdict(list)
        for _, row in mutation_data.iterrows():
            mutations_by_pdb[row['sample_id']].append(row)
        print(f"Loaded {len(mutation_data)} mutations in {len(mutations_by_pdb)} samples")
        return mutations_by_pdb
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def test_foldx_installation():
    foldx_path = CONFIG['foldx_path']
    if not os.path.isfile(foldx_path):
        print(f"FoldX executable does not exist: {foldx_path}")
        return False
    if not os.access(foldx_path, os.X_OK):
        print(f"FoldX file is not executable: {foldx_path}")
        return False
    return True


def process_single_pdb(args):
    sample_id, mutations, _ = args

    current_pid = os.getpid()
    work_dir = os.path.join(CONFIG['base_work_dir'], f"work_{sample_id}_{current_pid}")
    os.makedirs(work_dir, exist_ok=True)

    mut_list_file = os.path.join(work_dir, 'individual_list.txt')

    try:
        target_file = os.path.join(CONFIG['output_folder'], f"{sample_id}.pdb")
        if os.path.exists(target_file) and os.path.getsize(target_file) > 0 and not CONFIG['overwrite']:
            return {'status': 'skipped', 'pdb_id': sample_id, 'error': 'Target file already exists'}

        pdb_code = str(mutations[0]['pdb_id']).upper()
        wildtype_pdb = os.path.join(CONFIG['pdb_folder'], f"{pdb_code}.pdb")

        if not os.path.exists(wildtype_pdb):
            success, msg = download_pdb_from_rcsb_safe(pdb_code, CONFIG['pdb_folder'])
            if not success:
                return {'status': 'error', 'pdb_id': sample_id, 'error': f"PDB not found/download failed: {msg}"}

        mut_strs = []
        for row in mutations:
            one_mut = f"{row['wt_aa']}{row['chain']}{row['pdb_position']}{row['mut_aa']}"
            mut_strs.append(one_mut)

        with open(mut_list_file, "w", encoding="utf-8") as f:
            f.write(",".join(mut_strs) + ";")

        clean_pdb_name = f"{pdb_code}.pdb"
        local_pdb = os.path.join(work_dir, clean_pdb_name)
        shutil.copy2(wildtype_pdb, local_pdb)

        cmd = [
            CONFIG['foldx_path'],
            "--command=BuildModel",
            f"--pdb={clean_pdb_name}",
            f"--mutant-file=individual_list.txt",
            "--output-dir=."
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=work_dir, timeout=1000)

        generated_file = os.path.join(work_dir, f"{pdb_code}_1.pdb")

        if os.path.exists(generated_file):
            shutil.move(generated_file, target_file)
            return {'status': 'success', 'pdb_id': sample_id, 'error': None}
        else:
            files = [f for f in os.listdir(work_dir) if f.endswith('_1.pdb')]
            if files:
                shutil.move(os.path.join(work_dir, files[0]), target_file)
                return {'status': 'success', 'pdb_id': sample_id, 'error': None}

            return {'status': 'error', 'pdb_id': sample_id,
                    'error': f"FoldX output missing. Stderr: {result.stderr[:200]}"}

    except subprocess.TimeoutExpired:
        return {'status': 'error', 'pdb_id': sample_id, 'error': 'FoldX timeout'}
    except subprocess.CalledProcessError as e:
        return {'status': 'error', 'pdb_id': sample_id, 'error': f"FoldX crash: {e.stderr}"}
    except Exception as e:
        return {'status': 'error', 'pdb_id': sample_id, 'error': f"Exception: {str(e)}"}
    finally:
        if os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except:
                pass


def initialize_worker(config):
    CONFIG.update(config)


def log_error(pdb_id, error_msg):
    try:
        with open(CONFIG['error_log_file'], "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {pdb_id}: {error_msg}\n")
    except:
        pass


def update_progress(result, progress_data):
    progress_data['processed'] += 1
    if result['status'] == 'success':
        progress_data['success'] += 1
    elif result['status'] == 'error':
        progress_data['error'] += 1
        log_error(result['pdb_id'], result['error'])
    elif result['status'] == 'skipped':
        progress_data['skipped'] += 1

    if progress_data['processed'] % 10 == 0:
        print(f"Progress: {progress_data['processed']} | Success: {progress_data['success']} | "
              f"Failed: {progress_data['error']} | Skipped: {progress_data['skipped']}")


def main():
    start_time = time.time()
    setup_directories()

    print(f"--- Processing started (processes: {CONFIG['num_processes']}) ---")
    if not test_foldx_installation():
        raise RuntimeError("FoldX is unavailable")

    mutations_by_pdb = load_mutation_data()
    if not mutations_by_pdb: return

    all_pdb_ids = list(mutations_by_pdb.keys())
    print(f"Total tasks: {len(all_pdb_ids)}")

    manager = Manager()
    progress_data = manager.dict({'processed': 0, 'success': 0, 'error': 0, 'skipped': 0})
    tasks = [(pdb_id, mutations_by_pdb[pdb_id], i) for i, pdb_id in enumerate(all_pdb_ids)]

    with Pool(
        processes=CONFIG['num_processes'],
        maxtasksperchild=20,
        initializer=initialize_worker,
        initargs=(dict(CONFIG),),
    ) as pool:
        for result in pool.imap_unordered(process_single_pdb, tasks):
            update_progress(result, progress_data)

    try:
        os.rmdir(CONFIG['base_work_dir'])
    except:
        pass

    duration = time.time() - start_time
    print(f"\n--- Completed (elapsed: {duration:.1f}s) ---")
    print(f"Success: {progress_data['success']}, failed: {progress_data['error']}, skipped: {progress_data['skipped']}")
    print(f"Error log: {CONFIG['error_log_file']}")
    if progress_data['error']:
        raise RuntimeError(f"FoldX failed for {progress_data['error']} sample(s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate full mutant PDB structures with FoldX BuildModel.")
    parser.add_argument("--variants-csv", required=True, help="Prepared internal prediction CSV.")
    parser.add_argument("--wild-type-pdb-dir", required=True)
    parser.add_argument("--mutant-pdb-output-dir", required=True)
    parser.add_argument("--foldx-executable", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--error-log", required=True)
    parser.add_argument("--num-processes", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.num_processes < 1:
        raise ValueError("--num-processes must be at least 1")
    CONFIG.update({
        "pdb_folder": arguments.wild_type_pdb_dir,
        "csv_file": arguments.variants_csv,
        "output_folder": arguments.mutant_pdb_output_dir,
        "foldx_path": arguments.foldx_executable,
        "base_work_dir": arguments.work_dir,
        "error_log_file": arguments.error_log,
        "num_processes": arguments.num_processes,
        "overwrite": arguments.overwrite,
    })
    mp.set_start_method('spawn', force=True)
    main()
