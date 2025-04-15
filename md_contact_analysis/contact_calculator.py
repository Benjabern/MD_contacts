import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
import h5py

from scipy.sparse import coo_matrix
from contact_map_cython import build_contact_map_pbc
import numpy as np


def setup_logging(log_file=None):
    os.makedirs('logs', exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/md_analysis_{timestamp}.log'
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def analyze_frame(args):
    try:
        universe, frame_index, cutoff, res_indices = args
        universe.trajectory[frame_index]
        coords = universe.atoms.positions
        box_vectors = universe.dimensions
        residue_contact_matrix = build_contact_map_pbc(coords, box_vectors, cutoff, res_indices)
        return residue_contact_matrix
    except Exception as e:
        logging.error(f"Error processing frame {frame_index}: {e}")
        logging.error(traceback.format_exc())
        return None

def append_to_hdf5(file_path, new_array, dataset_name='cmaps'):
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            dataset = f[dataset_name]
            dataset.resize(dataset.shape[0] + 1, axis=0)
            dataset[-1] = new_array
        else:
            maxshape = (None,) + new_array.shape
            f.create_dataset(dataset_name, data=[new_array], maxshape=maxshape, chunks=True, compression='gzip', compression_opts=9)

def run_contact_calculation(
        universe,
        output_file,
        cutoff: float = 3.5,
        n_jobs: int = 16,
        start_frame: int = None,
        end_frame: int = None,
        step: int = None,
        chunk_size: int = 1000
):
    total_frames = universe.trajectory.n_frames
    start = 0 if start_frame is None else max(0, start_frame)
    end = total_frames if end_frame is None else min(total_frames, end_frame)
    step = 1 if step is None else max(1, step)
    frame_indices = list(range(start, end, step))
    n_frames = len(frame_indices)
    if n_frames == 0:
        logging.error("No frames to analyze with current slice parameters")
        return None
    res_indices = [res.atoms.indices for res in universe.residues]
    total_contacts = np.zeros((len(res_indices), len(res_indices)), dtype=np.uint32)
    for chunk_start_idx in range(0, n_frames, chunk_size):
        chunk_end_idx = min(chunk_start_idx + chunk_size, n_frames)
        chunk_frame_numbers = frame_indices[chunk_start_idx:chunk_end_idx]
        logging.info(f"Processing frames {chunk_frame_numbers[0]} to {chunk_frame_numbers[-1]}")
        with mp.Pool(processes=n_jobs) as pool:
            start_time = time.time()
            chunk_args = [(universe, frame_num, cutoff, res_indices) for frame_num in chunk_frame_numbers]
            chunk_results = pool.map(analyze_frame, chunk_args)
            for result in chunk_results:
                if result is not None:
                    #append_to_hdf5(f"{output_file}.h5", result)
                    total_contacts += result
            end_time = time.time()
            logging.info(f"Chunk computation time: {end_time - start_time:.2f} seconds")
    return total_contacts

def write_contact_matrix(results, output_file, universe):
    try:
        residue_names = [res.resname for res in universe.residues]
        residue_numbers = [int(res.resid) for res in universe.residues]
        names = residue_names
        real_numbers = residue_numbers
        results = results.astype(float)
        cmap = results.tolist()
        nres = int(len(cmap))
        out_dict = {
            'NResidues': nres,
            'map': cmap,
            'real_numbers': real_numbers,
            'names': names
        }
        with open(f"{output_file}.json", "w") as json_file:
            json.dump(out_dict, json_file)
        logging.info(f"Contact matrix written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing contact matrix: {e}")