import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from MDAnalysis.analysis.distances import distance_array

def setup_logging(log_file=None):
    """
    Configure logging to both console and file with clean formatting

    Args:
        log_file (str, optional): Path to log file. If None, uses default name.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # If no log file provided, create a timestamped log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/md_analysis_{timestamp}.log'

    # Configure logging with cleaner format
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
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
    """
    Analyze a single frame with error handling.
    
    Args:
        args (tuple): Tuple containing universe, frame_index, cutoff, and res_indices
    """
    try:
        universe, frame_index, cutoff, res_indices = args
        n_groups = len(res_indices)    
        universe.trajectory[frame_index]
        groups = [universe.atoms[indices] for indices in res_indices]
        dimensions = universe.dimensions
        frame_min_dist = np.zeros((n_groups, n_groups))
        # Calculate distance arrays for all group pairs
        dist_arrays = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                dist_array = distance_array(groups[i].positions, groups[j].positions, dimensions)
                dist_arrays.append(dist_array)
        
        # Find minimum distances and contacts for all group pairs
        idx = 0
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                dist_array = dist_arrays[idx]
                min_idx = np.unravel_index(np.argmin(dist_array), dist_array.shape)
                frame_min_dist[i,j] = frame_min_dist[j,i] = dist_array[min_idx]
                idx += 1
        
        contacts = np.zeros((n_groups, n_groups))
        contacts[frame_min_dist <= cutoff] = 1
        return contacts
    
    except Exception as e:
        logging.error(f"Error processing frame {frame_index}: {e}")
        logging.error(traceback.format_exc())
        return None


def frame_analysis(matrix, row_start=None, row_end=None, col_start=None, col_end=None):
    """
    Count rows that contain at least one 1 in the specified range.
    Uses numpy operations for maximum performance.

    Args:
        matrix: 2D numpy array or list of lists
        row_start: Starting row index (inclusive), defaults to 0
        row_end: Ending row index (exclusive), defaults to matrix height
        col_start: Starting column index (inclusive), defaults to 0
        col_end: Ending column index (exclusive), defaults to matrix width

    Returns:
        int: Number of rows containing at least one 1
    """

    # Slice matrix to specified range
    submatrix = matrix[row_start:row_end, col_start:col_end]

    with open('time_res.xvg', "a") as out:
        out.write(f'{np.any(submatrix == 1, axis=1).sum()}\n')


def run_contact_calculation(
        universe,
        cutoff: float = 3.5,
        n_jobs: int = 16,
        start_frame: int = None,
        end_frame: int = None,
        step: int = None,
        chunk_size: int = 1000
):
    """
    Generate contact matrix for molecular dynamics trajectory

    Args:
        universe: MDAnalysis Universe object
        cutoff: Distance cutoff for contacts in Angstrom
        n_jobs: Number of parallel processes
        start_frame: Starting frame for analysis
        end_frame: Ending frame for analysis
        step: Frame step for analysis
        chunk_size: Number of frames to process in each chunk

    Returns:
        Contact matrix
    """
    # Handle trajectory slicing parameters
    total_frames = universe.trajectory.n_frames
    start = 0 if start_frame is None else max(0, start_frame)
    end = total_frames if end_frame is None else min(total_frames, end_frame)
    step = 1 if step is None else max(1, step)

    # Get frame indices after slicing
    frame_indices = list(range(start, end, step))
    n_frames = len(frame_indices)

    if n_frames == 0:
        logging.error("No frames to analyze with current slice parameters")
        return None

    # Use residue indices
    res_indices = [res.atoms.indices for res in universe.residues]
    total_contacts = np.zeros((len(res_indices), len(res_indices)))

    # Process frames in chunks
    for chunk_start_idx in range(0, n_frames, chunk_size):
        chunk_end_idx = min(chunk_start_idx + chunk_size, n_frames)
        # Get the actual frame numbers for this chunk from frame_indices
        chunk_frame_numbers = frame_indices[chunk_start_idx:chunk_end_idx]

        logging.info(f"Processing frames {chunk_frame_numbers[0]} to {chunk_frame_numbers[-1]}")

        with mp.Pool(processes=n_jobs) as pool:
            start_time = time.time()

            # Prepare arguments for pool.starmap using actual frame numbers
            chunk_args = [(universe, frame_num, cutoff, res_indices)
                          for frame_num in chunk_frame_numbers]

            # Process chunk in parallel
            chunk_results = pool.map(analyze_frame, chunk_args)

            # Sum valid results, skipping None values
            for result in chunk_results:
                if result is not None:
                    frame_analysis(result, row_start=2170, row_end=2291, col_start=0, col_end=2170)
                    total_contacts += result

            end_time = time.time()
            logging.info(f"Chunk computation time: {end_time - start_time:.2f} seconds")

    return total_contacts

def write_contact_matrix(results, output_file, universe):
    """
    Write contact matrix to JSON file with correct residue names and numbers
    
    Args:
        results: Contact matrix
        output_file: Path to output JSON file
        universe: MDAnalysis Universe object with original residue information
    """
    try:
        # Get unique residue names and numbers
        residue_names = [res.resname for res in universe.residues]
        residue_numbers = [int(res.resid) for res in universe.residues]

        names = residue_names
        real_numbers = residue_numbers
        cmap = results.tolist()
        nres = int(len(cmap))

        out_dict = {
            'NResidues': nres,
            'map': cmap, 
            'real_numbers': real_numbers,
            'names': names
        }

        with open(output_file, "w") as json_file:
            json.dump(out_dict, json_file)
        
        logging.info(f"Contact matrix written to {output_file}")
    
    except Exception as e:
        logging.error(f"Error writing contact matrix: {e}")
