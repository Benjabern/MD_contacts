import argparse
import json
import sys
import traceback
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from .contact_calculator import run_contact_calculation, write_contact_matrix, setup_logging
from .contact_analyzer import load_config, ContactAnalysis



def main():
    """
    Main function to run the molecular dynamics contact analysis suite
    """
    parser = argparse.ArgumentParser(
        description="Molecular Dynamics Contact Analysis Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add mode selection
    parser.add_argument('mode', choices=['calculate', 'analyze'],
                        help='Mode of operation: calculate contact matrix or analyze contacts')

    # Common arguments
    parser.add_argument('-s', '--structure', required=True,
                        help='Input structure file (.gro, .pdb)')
    parser.add_argument('--log', help='Log file path')

    # Contact calculation arguments
    contact_calc_group = parser.add_argument_group('Contact Generation')
    contact_calc_group.add_argument('-f', '--trajectory',
                                   help='Input trajectory file (.xtc)')
    contact_calc_group.add_argument('-o', '--output', default='contacts.json',
                        help='Output directory/file path')
    contact_calc_group.add_argument('--cutoff', type=float, default=3.5,
                                   help='Distance cutoff for contacts in Angstrom')
    contact_calc_group.add_argument('-j', '--jobs', type=int, default=16,
                                   help='Number of parallel jobs')
    contact_calc_group.add_argument('--chunk-size', type=int, default=1000,
                                   help='Number of frames to process in each chunk')
    contact_calc_group.add_argument('-b', '--begin', type=int,
                                   help='Starting frame for analysis')
    contact_calc_group.add_argument('-e', '--end', type=int,
                                   help='End frame for analysis')
    contact_calc_group.add_argument('-df', '--step', type=int,
                                   help='Frame step for analysis')

    # Contact analysis arguments
    contact_analysis_group = parser.add_argument_group('Contact Analysis')
    contact_analysis_group.add_argument('-m', '--contact_matrix',
                                        help='Json file containing contact matrix')
    contact_analysis_group.add_argument('-c', '--config',
                                        help='YAML configuration file for contact analysis')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log)

    try:

        if args.mode == 'calculate':
            if not args.trajectory:
                parser.error("Trajectory file is required in calculate mode")

            # Load universe
            universe = mda.Universe(args.structure, args.trajectory)

            # output file path
            contacts_path = Path(args.output)

            # Calculate contact matrix
            contact_matrix = run_contact_calculation(
                universe,
                contacts_path,
                cutoff=args.cutoff,
                n_jobs=args.jobs,
                start_frame=args.begin,
                end_frame=args.end,
                step=args.step,
                chunk_size=args.chunk_size
            )

            # Write contact matrix
            write_contact_matrix(contact_matrix, contacts_path, universe)

        elif args.mode == 'analyze':
            # Load contact matrix
            contacts_path = Path(args.contact_matrix)
            with open(contacts_path, 'r') as f:
                contacts = json.load(f)
            contact_matrix = np.array(contacts['map'], dtype=int)


            # Load configuration
            config = load_config(args.config)

            # Load universe
            universe = mda.Universe(args.structure)
            universe.add_TopologyAttr('tempfactors', range(len(universe.atoms)))
            universe.add_TopologyAttr('chainIDs', range(len(universe.atoms)))
            logger.info("topology processing complete")

            # Initialize and run contact analysis
            analysis = ContactAnalysis(universe, config, contact_matrix)

            # Perform analysis
            residue_contacts = analysis.analyze_contacts()
            avg_generic_contacts, avg_interest_contacts = analysis.write_results(residue_contacts)
            logger.info("molecule group contacts complete")

            # Export PDB files
            analysis.export_pdb(residue_contacts, avg_interest_contacts, avg_generic_contacts)
            logger.info("structure files export complete")

            # Generate visualizations
            enrichment_matrix, _ = analysis.sasa_norm_enrichment()
            analysis.plot_enrichments(enrichment_matrix)
            analysis.sasa_norm_generic_propensity()
            logger.info("visualizations complete")
            logger.info("contact analysis complete")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
