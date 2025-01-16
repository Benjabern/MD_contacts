import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array
from mpi4py import MPI


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
            for j in range(i + 1, n_groups):
                dist_array = distance_array(groups[i].positions, groups[j].positions, dimensions)
                dist_arrays.append(dist_array)

        # Find minimum distances and contacts for all group pairs
        idx = 0
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                dist_array = dist_arrays[idx]
                min_idx = np.unravel_index(np.argmin(dist_array), dist_array.shape)
                frame_min_dist[i, j] = frame_min_dist[j, i] = dist_array[min_idx]
                idx += 1

        contacts = np.zeros((n_groups, n_groups))
        contacts[frame_min_dist <= cutoff] = 1
        return contacts

    except Exception as e:
        logging.error(f"Error processing frame {frame_index}: {e}")
        logging.error(traceback.format_exc())
        return None

def run_analysis(universe, cutoff: float, res_indices, chunk_size: int = 100,
                 start_frame: int = None, end_frame: int = None, step: int = None):
    """
    Run hybrid MPI+multiprocessing analysis with trajectory slicing

    Args:
        universe: MDAnalysis Universe object
        cutoff: Distance cutoff for contacts in Angstrom
        res_indices: Residue indices to analyze
        chunk_size: Number of frames to process in each chunk
        start_frame: Starting frame for analysis (None = first frame)
        end_frame: End frame for analysis (None = last frame)
        step: Step size between frames (None = every frame)
    """
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up per-node multiprocessing to use all available CPUs
    n_cores_per_node = mp.cpu_count()

    # Handle trajectory slicing parameters
    total_frames = universe.trajectory.n_frames
    start = 0 if start_frame is None else max(0, start_frame)
    end = total_frames if end_frame is None else min(total_frames, end_frame)
    step = 1 if step is None else max(1, step)

    # Get frame indices after slicing
    frame_indices = list(range(start, end, step))
    n_frames = len(frame_indices)

    if n_frames == 0:
        if rank == 0:
            logging.error("No frames to analyze with current slice parameters")
        return None

    total_contacts = np.zeros((len(res_indices), len(res_indices)))

    # Distribute frames across MPI ranks
    frames_per_rank = n_frames // size
    rank_start_idx = rank * frames_per_rank
    rank_end_idx = rank_start_idx + frames_per_rank if rank < size - 1 else n_frames

    # Get actual frame numbers for this rank
    rank_frames = frame_indices[rank_start_idx:rank_end_idx]

    # Process frames in chunks on each rank
    for chunk_start_idx in range(0, len(rank_frames), chunk_size):
        chunk_end_idx = min(chunk_start_idx + chunk_size, len(rank_frames))
        chunk_frames = rank_frames[chunk_start_idx:chunk_end_idx]

        if rank == 0:
            logging.info(f"Processing frames {chunk_frames[0]} to {chunk_frames[-1]} on rank {rank}")

        try:
            with mp.Pool(processes=n_cores_per_node) as pool:
                chunk_args = [(universe, frame, cutoff, res_indices) for frame in chunk_frames]
                chunk_results = pool.map(analyze_frame, chunk_args)

                for result in chunk_results:
                    if result is not None:
                        total_contacts += result

        except Exception as e:
            logging.error(f"Frame chunk processing error on rank {rank}: {str(e)}")
            logging.debug(traceback.format_exc())

    # Gather results from all ranks
    global_contacts = np.zeros_like(total_contacts)
    comm.Reduce(total_contacts, global_contacts, op=MPI.SUM, root=0)

    return global_contacts if rank == 0 else None


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


def main():
    """
    Main function to run the molecular dynamics analysis
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Setup logging
        setup_logging(args.log)

        # Initial parameter logging
        if MPI.COMM_WORLD.Get_rank() == 0:
            logging.info("=" * 50)
            logging.info("Molecular Dynamics Contact Analysis")
            logging.info("=" * 50)
            logging.info(f"Configuration:")
            logging.info(f"  Input Structure: {args.gro}")
            logging.info(f"  Trajectory: {args.xtc}")
            logging.info(f"  Output: {args.output}")
            logging.info(f"  Cutoff: {args.cutoff} Å")
            logging.info(f"  Chunk size: {args.chunk_size} frames")
            if args.begin is not None:
                logging.info(f"  Start frame: {args.begin}")
            if args.end is not None:
                logging.info(f"  End frame: {args.end}")
            if args.step is not None:
                logging.info(f"  Frame step: {args.step}")
            logging.info("=" * 50)

        # Load trajectory
        u = mda.Universe(args.gro, args.xtc)

        # Run analysis
        res_indices = [res.atoms.indices for res in u.residues]
        start_time = time.time()
        results = run_analysis(u, args.cutoff, res_indices, args.chunk_size,
                               start_frame=args.begin, end_frame=args.end, step=args.step)

        # Write results
        if MPI.COMM_WORLD.Get_rank() == 0:
            write_distance_matrix(results, args.output)
            end_time = time.time()
            logging.info("=" * 50)
            logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
            logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Critical error in execution: {str(e)}")
        logging.debug(traceback.format_exc())
        sys.exit(1)


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
        help="Output JSON file path (default: contacts.json)"
    )
    parser.add_argument(
        "-c", "--cutoff",
        type=float,
        default=3.5,
        help="Distance cutoff for contacts (default: 3.5)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of frames to process in each chunk (default: 1000)"
    )
    parser.add_argument(
        "-l", "--log",
        help="Log file path (default: logs/md_analysis_TIMESTAMP.log)"
    )
    parser.add_argument(
        "-b", "--begin",
        type=int,
        default=None,
        help="Starting frame for analysis"
    )
    parser.add_argument(
        "-e", "--end",
        type=int,
        default=None,
        help="End frame for analysis"
    )
    parser.add_argument(
        "-df", "--step",
        type=int,
        default=None,
        help="Frame step for analysis (every nth frame)"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main()