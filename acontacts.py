import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter
import json
import os
import warnings
# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')


def write_residue_contacts(residue_contacts, config):
    """
    Write residue contact information to files.

    Parameters:
    -----------
    residue_contacts : dict
        Dictionary containing contact information for each interest
    output_dir : str, optional
        Directory to save output files
    """
    cat_m_interest, m_interest_ranges, cat_m_generic, m_generic_ranges, project = parse_config_file(config)
    output_dir = project
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare lists to collect contact data for averaging
    all_generic_contacts = []
    all_interest_contacts = []

    # Write individual interest contact files
    for m_of_interest, contacts in residue_contacts.items():
        # Write generic contacts
        generic_file = os.path.join(output_dir, f"{m_of_interest}_{cat_m_generic}_contacts_{project}.txt")
        if len(contacts['generic_contacts']) > 0:
            with open(generic_file, 'w') as f:
                for contact_count in contacts['generic_contacts']:
                    f.write(f"{contact_count}\n")

        # Store generic contacts for averaging
        all_generic_contacts.append(contacts['generic_contacts'])

        # Write interest contacts
        interest_file = os.path.join(output_dir, f"{m_of_interest}_{cat_m_interest}_contacts_{project}.txt")
        with open(interest_file, 'w') as f:
            for contact_count in contacts['interest_contacts']:
                f.write(f"{contact_count}\n")

        # Store interest contacts for averaging
        all_interest_contacts.append(contacts['interest_contacts'])

    # Calculate and write average contacts
    # Convert lists to numpy arrays
    all_generic_contacts = np.array(all_generic_contacts)
    all_interest_contacts = np.array(all_interest_contacts)

    # Calculate average contacts
    avg_generic_contacts = np.mean(all_generic_contacts, axis=0)
    avg_interest_contacts = np.mean(all_interest_contacts, axis=0)

    # Write average generic contacts
    avg_generic_file = os.path.join(output_dir, f"{cat_m_interest}_{cat_m_generic}_av_contacts_{project}.avg")
    if len(avg_generic_contacts) > 0:
        with open(avg_generic_file, 'w') as f:
            for contact_count in avg_generic_contacts:
                f.write(f"{contact_count}\n")

    # Write average interest contacts
    avg_interest_file = os.path.join(output_dir, f"{cat_m_interest}_{cat_m_interest}_av_contacts_{project}.avg")
    with open(avg_interest_file, 'w') as f:
        for contact_count in avg_interest_contacts:
            f.write(f"{contact_count}\n")

    print(f"Residue contact files written to {output_dir}")
    return avg_generic_contacts, avg_interest_contacts


def analyze_residue_contacts(contact_matrix, m_interest_ranges, m_generic_ranges):
    """
    Analyze contacts for each residue in interests with generics and other interests.

    Parameters:
    -----------
    sub_matrix : numpy.ndarray
        2D matrix of binary contacts between residues
    component_ranges : dict
        Dictionary with keys as component names and values as lists of
        [start_index, end_index] for each component

    Returns:
    --------
    dict: Containing contact information for each molecule of interest
          containing both (group)internal and external contacts.
    """
    # Identify interest and generic components
    nframes = contact_matrix[0][0]
    interest_components = [c for c in m_interest_ranges.keys()]
    if len(m_generic_ranges) > 0:
        generic_components = [c for c in m_generic_ranges.keys()]

        generic_contact_ranges = [m_generic_ranges[d] for d in generic_components]
        generic_columns = np.concatenate([
            np.arange(d_start, d_end) for (d_start, d_end) in generic_contact_ranges
        ])

    # Prepare results dictionary
    residue_contacts = {}
    for m_of_interest in interest_components:
        # Get interest range
        p_start, p_end = m_interest_ranges[m_of_interest]

        if len(m_generic_ranges) > 0:
            # Analyze contacts with generics
            generic_contact_ranges = [m_generic_ranges[d] for d in generic_components]

            # Create a list of column indices for generic contacts
            generic_columns = np.concatenate([
                np.arange(d_start, d_end) for (d_start, d_end) in generic_contact_ranges
            ])
            # Extract submatrix for generic contacts
            generic_contact_matrix = contact_matrix[p_start:p_end, generic_columns]
            # Count contacts for each residue with generics
            generic_residue_contacts = np.sum(generic_contact_matrix, axis=1)
            generic_residue_contacts = generic_residue_contacts / nframes
        else:
            generic_residue_contacts = []


        # Analyze contacts with other molecules of interests
        other_m_of_interest = [p for p in interest_components if p != m_of_interest]

        other_interest_contact_ranges = [m_interest_ranges[p] for p in other_m_of_interest]
        if other_interest_contact_ranges:

            # Create a list of column indices for interest contacts
            interest_columns = np.concatenate([
                np.arange(p_start, p_end) for (p_start, p_end) in other_interest_contact_ranges
            ])

            # Extract submatrix for interest contacts
            interest_contact_matrix = contact_matrix[p_start:p_end, interest_columns]

            # Count contacts for each residue with other interests
            interest_residue_contacts = np.sum(interest_contact_matrix, axis=1)
            interest_residue_contacts = interest_residue_contacts / nframes
        else:
            p_mat = contact_matrix[p_start:p_end, p_start:p_end]
            interest_residue_contacts = np.sum(p_mat, axis=1)
            interest_residue_contacts = interest_residue_contacts / nframes
        # Store results
        residue_contacts[m_of_interest] = {
            'generic_contacts': generic_residue_contacts,
            'interest_contacts': interest_residue_contacts,
        }
    return residue_contacts

def export_pdb(universe, residue_contacts, config, av_interest_contacts, av_generic_contacts):
    """
    Export an MDAnalysis Universe to a PDB file with:
    - Sequentially numbered residues
    - Custom residue-wise B-factors

    """
    cat_m_interest, m_interest_ranges, cat_m_generic, m_generic_ranges, project = parse_config_file(config)
    output_dir = project
    for i, residue in enumerate(universe.residues, start=1):
        residue.resid = i
    seq_uni = universe
    for name_m_interest, (start, end) in m_interest_ranges.items():
        new_segment = universe.add_Segment(segid=name_m_interest)
        seq_uni.residues[start:end].segments = new_segment
        i = 0
        residues = universe.atoms.select_atoms(f'segid {name_m_interest}')
        if len(av_generic_contacts) > 0:
            for residue in residues.residues:
                residue_bfactor = residue_contacts[name_m_interest]['generic_contacts'][i]
                for atom in residue.atoms:
                    atom.tempfactor = residue_bfactor
                i += 1
            with PDBWriter(os.path.join(output_dir, f'{name_m_interest}_{cat_m_generic}_contacts_{project}.pdb')) as pdb_writer:
                pdb_writer.write(residues)
            i = 0
            for residue in residues.residues:
                residue_bfactor = av_generic_contacts[i]
                for atom in residue.atoms:
                    atom.tempfactor = residue_bfactor
                i += 1
            with PDBWriter(os.path.join(output_dir, f'{cat_m_interest}_{cat_m_generic}_av_contacts_{project}.pdb')) as pdb_writer:
                pdb_writer.write(residues)
            i = 0
        for residue in residues.residues:
            residue_bfactor = residue_contacts[name_m_interest]['interest_contacts'][i]
            for atom in residue.atoms:
                atom.tempfactor = residue_bfactor
            i += 1
        with PDBWriter(os.path.join(output_dir, f'{name_m_interest}_{cat_m_interest}_contacts_{project}.pdb')) as pdb_writer:
            pdb_writer.write(residues)
        i = 0
        for residue in residues.residues:
            residue_bfactor = av_interest_contacts[i]
            for atom in residue.atoms:
                atom.tempfactor = residue_bfactor
            i += 1
        with PDBWriter(os.path.join(output_dir, f'{cat_m_interest}_{cat_m_interest}_av_contacts_{project}.pdb')) as pdb_writer:
            pdb_writer.write(residues)
    for name_m_generic, (start, end) in m_generic_ranges.items():
        new_segment = universe.add_Segment(segid=name_m_generic)
        seq_uni.residues[start:end].segments = new_segment

def parse_config_file(filename):
    """
    Parse a molecule configuration file.

    Args:
        filename (str): Path to the configuration file

    Returns:
        tuple: (name_m_interest, m_interest_ranges, name_m_generic, m_generic_ranges)
    """
    # Initialize variables to store parsed data
    cat_m_interest = ''
    m_interest_ranges = {}
    cat_m_generic = ''
    m_generic_ranges = {}

    # Current section tracker
    current_section = None

    # Read the file
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace and convert tabs to spaces
            line = line.strip().replace('\t', ' ')

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            if line.startswith('Project:'):
                project = line.split(':', 1)[1].strip()
            # Check for Name: line
            if line.startswith('Name:'):
                name = line.split(':', 1)[1].strip()
                if current_section is None:
                    cat_m_interest = name
                    current_section = 'interest'
                else:
                    cat_m_generic = name
                    current_section = 'generic'
                continue

            # Parse range entries
            if '-' in line:
                try:
                    key, range_value = line.split(':', 1)
                    key = key.strip()
                    start, end = map(int, range_value.strip().split('-'))

                    if current_section == 'interest':
                        m_interest_ranges[key] = [start, end]
                    elif current_section == 'generic':
                        m_generic_ranges[key] = [start, end]
                except ValueError:
                    print(f"Warning: Skipping invalid range entry: {line}")

    return cat_m_interest, m_interest_ranges, cat_m_generic, m_generic_ranges, project
def enrichments(contact_matrix, m_interest_ranges, m_generic_ranges, project, cat_m_interest, seq):
    out_path = os.path.join(project, f'{cat_m_interest}_c_enrichments_{project}.pdf')
    interest_components = [c for c in m_interest_ranges.keys()]
    generic_components = [c for c in m_generic_ranges.keys()]
    combination_counts = {}
    total_contacts = 0
    total_pairs = 0
    for m_of_interest in interest_components:
        p_start, p_end = m_interest_ranges[m_of_interest]
        other_m_of_interest = [p for p in interest_components if p != m_of_interest]
        other_interest_contact_ranges = [m_interest_ranges[p] for p in other_m_of_interest]
        if other_interest_contact_ranges:

            # Create a list of column indices for interest contacts
            interest_columns = np.concatenate([
                np.arange(p_start, p_end) for (p_start, p_end) in other_interest_contact_ranges
            ])

            # Extract submatrix for interest contacts
            p_mat = contact_matrix[p_start:p_end, interest_columns]
            rlabels = seq[p_start:p_end]
            clabels = seq[interest_columns]
        else:
            p_mat = contact_matrix[p_start:p_end, p_start:p_end]
            rlabels = seq[p_start:p_end]
            clabels = seq[p_start:p_end]
        total_pairs += len(rlabels)*len(clabels)
        total_contacts += np.sum(p_mat)
        for i, row_label in enumerate(rlabels):
            for j, col_label in enumerate(clabels):
                combination = frozenset([row_label, col_label])
                count = p_mat[i, j]
                if combination not in combination_counts:
                    combination_counts[combination] = {
                        'total_count': 0,
                        'occurrences': 0
                    }

                # Update total count and increment occurrences
                combination_counts[combination]['total_count'] += count
                combination_counts[combination]['occurrences'] += 1
    # Extract unique labels
    labels = sorted({label for key in combination_counts.keys() for label in key})
    label_index = {label: i for i, label in enumerate(labels)}
    # Initialize the matrix
    size = len(labels)
    matrix = np.zeros((size, size))
    for key, values in combination_counts.items():
        count = values['total_count']
        occurrences = values['occurrences']
        # Calculate the value
        if occurrences > 0 and total_contacts > 0:
            value = ((count) / total_contacts) / (occurrences / total_pairs)
        else:
            value = 0

        # Update the matrix for each pair in the combination
        for label1 in key:
            for label2 in key:
                idx1, idx2 = label_index[label1], label_index[label2]
                matrix[idx1, idx2] = value
    c_enr_map = pd.DataFrame(data=matrix[::-1, :], index=labels[::-1], columns=labels)
    fig, axs = plt.subplots(figsize=(10, 10), constrained_layout=True)
    sns.heatmap(c_enr_map, annot=False, cmap="coolwarm", vmin=0, center=1)
    plt.title('Contacts enrichment matrix', fontsize=20)
    try:
        fig.savefig(out_path)
        print("File", out_path, "created\n")
    except:
        print("Error writing file", out_path + '\n')

def clean_seq(u):
    seq = []
    [seq.append(str(r).split()[1].rstrip(",")) for r in u.residues]
    seq = np.array(seq)
    replacements = {"GLH": "GLU", "CYX": "CYS", "HID": "HIS", "HIE": "HIS"}
    seq = np.vectorize(lambda x: replacements.get(x, x))(seq)
    return seq
config = sys.argv[2]
input = sys.argv[1]
structure = sys.argv[3]
with open(input) as f:
   contacts = json.load(f)
cmap = contacts['map']
contact_matrix = np.array(cmap, dtype=int)
contact_matrix = np.array(contact_matrix)
cat_m_interest, m_interest_ranges, name_m_generic, m_generic_ranges, project = parse_config_file(config)
u = mda.Universe(structure)
seq = clean_seq(u)
u.add_TopologyAttr('tempfactors', range(len(u.atoms)))
# Analyze residue contacts
residue_contacts = analyze_residue_contacts(contact_matrix, m_interest_ranges, m_generic_ranges)
av_generic_contacts, av_interest_contacts = write_residue_contacts(residue_contacts, config)
export_pdb(u, residue_contacts, config, av_interest_contacts, av_generic_contacts)
enrichments(contact_matrix, m_interest_ranges, m_generic_ranges, project, cat_m_interest, seq)
