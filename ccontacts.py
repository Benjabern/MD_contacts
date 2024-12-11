import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np
import multiprocessing as mp
from typing import List, Tuple
import time
import os
import json
import traceback
import sys
import argparse
import logging
from datetime import datetime

def setup_logging(log_file=None):
    """
    Configure logging to both console and file
    
    Args:
        log_file (str, optional): Path to log file. If None, uses default name.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # If no log file provided, create a timestamped log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'logs/md_analysis_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

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

def run_parallel_analysis(universe, cutoff: float, n_jobs: int, res_indices, chunk_size: int = 100):
    """
    Run parallel analysis with chunked processing
    
    Args:
        universe: MDAnalysis Universe object
        cutoff: Distance cutoff for contacts in Angstrom
        n_jobs: Number of parallel processes
        res_indices: Residue indices to analyze
        chunk_size: Number of frames to process in each chunk
    """
    n_frames = universe.trajectory.n_frames
    total_contacts = np.zeros((len(res_indices), len(res_indices)))
    
    # Process frames in chunks
    for start_frame in range(0, n_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, n_frames)
        chunk_frames = list(range(start_frame, end_frame))
        
        logging.info(f"Processing frames {start_frame} to {end_frame}")
        
        try:
            with mp.Pool(processes=n_jobs) as pool:
                start_time = time.time()
                
                # Prepare arguments for pool.starmap
                chunk_args = [(universe, frame, cutoff, res_indices) for frame in chunk_frames]
                
                # Process chunk in parallel
                chunk_results = pool.map(analyze_frame, chunk_args)
                
                # Sum valid results, skipping None values
                for result in chunk_results:
                    if result is not None:
                        total_contacts += result
                
                end_time = time.time()
                logging.info(f"Chunk computation time: {end_time-start_time:.2f} seconds")
        
        except Exception as e:
            logging.error(f"Error in chunk processing: {e}")
            logging.error(traceback.format_exc())
    
    return total_contacts

def write_distance_matrix(results, output_file):
    """
    Write contact matrix to JSON file
    
    Args:
        results: Contact matrix
        output_file: Path to output JSON file
    """
    try:
        names = ["x"] * len(results)
        cmap = results.tolist()
        out_dict = {
            'NResidues': len(cmap), 
            'map': cmap, 
            'real_numbers': list(range(1, len(cmap) + 1)), 
            'names': names
        }
    
        with open(output_file, "w") as json_file:
            json.dump(out_dict, json_file)
        
        logging.info(f"Contact matrix written to {output_file}")
    
    except Exception as e:
        logging.error(f"Error writing contact matrix: {e}")

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Molecular Dynamics Contact Analysis")
    
    parser.add_argument(
        "-s", "--gro", 
        required=True, 
        help="Input .gro file path"
    )
    parser.add_argument(
        "-f", "--xtc", 
        required=True, 
        help="Input .xtc file path"
    )
    parser.add_argument(
        "-o", "--output", 
        default="contacts.json", 
        help="Output JSON file path (default: micro_contacts.json)"
    )
    parser.add_argument(
        "-c", "--cutoff", 
        type=float, 
        default=3.5, 
        help="Distance cutoff for contacts (default: 3.5)"
    )
    parser.add_argument(
        "-j", "--jobs", 
        type=int, 
        default=16, 
        help="Number of parallel jobs (default: 16)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=100, 
        help="Number of frames to process in each chunk (default: 100)"
    )
    parser.add_argument(
        "-l", "--log", 
        help="Log file path (default: logs/md_analysis_TIMESTAMP.log)"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the molecular dynamics analysis
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log)
        
        logging.info("Starting Molecular Dynamics Contact Analysis")
        logging.info(f"Input GRO file: {args.gro}")
        logging.info(f"Input XTC file: {args.xtc}")
        logging.info(f"Output file: {args.output}")
        logging.info(f"Cutoff distance: {args.cutoff}")
        logging.info(f"Parallel jobs: {args.jobs}")
        logging.info(f"Chunk size: {args.chunk_size}")
        
        # Load trajectory
        u = mda.Universe(args.gro, args.xtc)
        
        # Run analysis
        res_indices = [res.atoms.indices for res in u.residues]
        results = run_parallel_analysis(
            universe=u,
            cutoff=args.cutoff,
            n_jobs=args.jobs,
            res_indices=res_indices,
            chunk_size=args.chunk_size
        )
        
        # Write results
        write_distance_matrix(results, args.output)
        
        logging.info("Molecular Dynamics Contact Analysis completed successfully")
    
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
