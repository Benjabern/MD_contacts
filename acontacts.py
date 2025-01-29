import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from MDAnalysis.coordinates.PDB import PDBWriter
from dataclasses import dataclass

@dataclass
class MoleculeConfig:
    name: str
    components: Dict[str, List[int]]

@dataclass
class Config:
    project: str
    molecules_of_interest: MoleculeConfig
    generic_molecules: Optional[MoleculeConfig]

class ContactAnalysis:
    def __init__(self, universe: mda.Universe, config: Config, contact_matrix: np.ndarray):
        self.universe = universe
        self.config = config
        self.contact_matrix = contact_matrix
        self.seq = self._clean_sequence()
        self._renumber_residues()
        self.m_interest_ranges, self.m_generic_ranges = self._convert_atom_ranges()

    def _clean_sequence(self) -> np.ndarray:
        """Clean and standardize residue names in sequence."""
        seq = [str(r).split()[1].rstrip(",") for r in self.universe.residues]
        seq = np.array(seq)
        replacements = {
            "GLH": "GLU", 
            "CYX": "CYS", 
            "HID": "HIS", 
            "HIE": "HIS"
        }
        return np.vectorize(lambda x: replacements.get(x, x))(seq)

    def _renumber_residues(self):
        """Renumber residues in universe sequentially."""
        for i, residue in enumerate(self.universe.residues, start=1):
            residue.resid = i

    def _convert_atom_ranges(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Convert atom ranges to residue ranges."""
        m_interest_ranges = {}
        m_generic_ranges = {}
        
        def _convert_range(component_range: List[int]) -> List[int]:
            start = self.universe.atoms[component_range[0] - 1].residue.resid - 1
            end = self.universe.atoms[component_range[1] - 1].residue.resid
            return [start, end]

        for name, range_vals in self.config.molecules_of_interest.components.items():
            m_interest_ranges[name] = _convert_range(range_vals)

        if self.config.generic_molecules:
            for name, range_vals in self.config.generic_molecules.components.items():
                m_generic_ranges[name] = _convert_range(range_vals)

        return m_interest_ranges, m_generic_ranges

    def analyze_contacts(self) -> Dict:
        """Analyze residue contacts for all molecules."""
        residue_contacts = {}
        nframes = self.contact_matrix[0][0]
        
        for m_of_interest, (p_start, p_end) in self.m_interest_ranges.items():
            generic_contacts = self._analyze_generic_contacts(p_start, p_end, nframes)
            interest_contacts = self._analyze_interest_contacts(m_of_interest, p_start, p_end, nframes)
            
            residue_contacts[m_of_interest] = {
                'generic_contacts': generic_contacts,
                'interest_contacts': interest_contacts
            }
            
        return residue_contacts

    def _analyze_generic_contacts(self, p_start: int, p_end: int, nframes: int) -> np.ndarray:
        """Analyze contacts with generic molecules."""
        if not self.m_generic_ranges:
            return np.array([])

        generic_columns = np.concatenate([
            np.arange(start, end) 
            for start, end in self.m_generic_ranges.values()
        ])
        
        contact_matrix = self.contact_matrix[p_start:p_end, generic_columns]
        return np.sum(contact_matrix, axis=1) / nframes

    def _analyze_interest_contacts(self, current_mol: str, p_start: int, p_end: int, nframes: int) -> np.ndarray:
        """Analyze contacts with other molecules of interest."""
        other_ranges = [(s, e) for m, (s, e) in self.m_interest_ranges.items() 
                       if m != current_mol]
        
        if not other_ranges:
            # Self-contacts if no other molecules of interest
            contact_matrix = self.contact_matrix[p_start:p_end, p_start:p_end]
        else:
            interest_columns = np.concatenate([
                np.arange(start, end) for start, end in other_ranges
            ])
            contact_matrix = self.contact_matrix[p_start:p_end, interest_columns]
            
        return np.sum(contact_matrix, axis=1) / nframes

    def write_results(self, residue_contacts: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Write contact results to files and return averages."""
        output_dir = Path(self.config.project)
        output_dir.mkdir(exist_ok=True)

        all_generic_contacts = []
        all_interest_contacts = []

        for m_of_interest, contacts in residue_contacts.items():
            self._write_contact_file(output_dir, m_of_interest, contacts)
            
            if len(contacts['generic_contacts']) > 0:
                all_generic_contacts.append(contacts['generic_contacts'])
            all_interest_contacts.append(contacts['interest_contacts'])

        return self._write_average_contacts(output_dir, all_generic_contacts, all_interest_contacts)

    def _write_contact_file(self, output_dir: Path, molecule: str, contacts: Dict):
        """Write contact data for a single molecule."""
        generic_name = self.config.generic_molecules.name if self.config.generic_molecules else ""
        interest_name = self.config.molecules_of_interest.name

        if len(contacts['generic_contacts']) > 0:
            generic_file = output_dir / f"{molecule}_{generic_name}_contacts_{self.config.project}.txt"
            np.savetxt(generic_file, contacts['generic_contacts'])

        interest_file = output_dir / f"{molecule}_{interest_name}_contacts_{self.config.project}.txt"
        np.savetxt(interest_file, contacts['interest_contacts'])

    def _write_average_contacts(self, output_dir: Path, 
                              generic_contacts: List[np.ndarray], 
                              interest_contacts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate and write average contacts."""
        generic_name = self.config.generic_molecules.name if self.config.generic_molecules else ""
        interest_name = self.config.molecules_of_interest.name

        avg_generic_contacts = np.mean(generic_contacts, axis=0) if generic_contacts else np.array([])
        avg_interest_contacts = np.mean(interest_contacts, axis=0)

        if len(avg_generic_contacts) > 0:
            avg_generic_file = output_dir / f"{interest_name}_{generic_name}_av_contacts_{self.config.project}.avg"
            np.savetxt(avg_generic_file, avg_generic_contacts)

        avg_interest_file = output_dir / f"{interest_name}_{interest_name}_av_contacts_{self.config.project}.avg"
        np.savetxt(avg_interest_file, avg_interest_contacts)

        return avg_generic_contacts, avg_interest_contacts

    def export_pdb(self, residue_contacts: Dict, avg_interest_contacts: np.ndarray, 
                  avg_generic_contacts: np.ndarray):
        """Export PDB files with contact information in B-factors."""
        output_dir = Path(self.config.project)
        generic_name = self.config.generic_molecules.name if self.config.generic_molecules else ""
        interest_name = self.config.molecules_of_interest.name

        for name_m_interest, (start, end) in self.m_interest_ranges.items():
            # Create new segment for the molecule
            new_segment = self.universe.add_Segment(segid=name_m_interest)
            self.universe.residues[start:end].segments = new_segment
            residues = self.universe.atoms.select_atoms(f'segid {name_m_interest}')

            self._write_molecule_pdbs(
                output_dir, name_m_interest, residues, 
                residue_contacts[name_m_interest], 
                avg_generic_contacts, avg_interest_contacts,
                generic_name, interest_name
            )

    def _write_molecule_pdbs(self, output_dir: Path, molecule_name: str, residues: mda.AtomGroup,
                           contacts: Dict, avg_generic_contacts: np.ndarray, 
                           avg_interest_contacts: np.ndarray, generic_name: str, 
                           interest_name: str):
        """Write PDB files for a single molecule with different contact information."""
        def _write_pdb_with_contacts(filename: Path, contact_values: np.ndarray):
            for i, residue in enumerate(residues.residues):
                for atom in residue.atoms:
                    atom.tempfactor = contact_values[i]
            with PDBWriter(filename) as pdb_writer:
                pdb_writer.write(residues)

        # Write generic contacts if they exist
        if len(contacts['generic_contacts']) > 0:
            _write_pdb_with_contacts(
                output_dir / f"{molecule_name}_{generic_name}_contacts_{self.config.project}.pdb",
                contacts['generic_contacts']
            )
            _write_pdb_with_contacts(
                output_dir / f"{interest_name}_{generic_name}_av_contacts_{self.config.project}.pdb",
                avg_generic_contacts
            )

        # Write interest contacts
        _write_pdb_with_contacts(
            output_dir / f"{molecule_name}_{interest_name}_contacts_{self.config.project}.pdb",
            contacts['interest_contacts']
        )
        _write_pdb_with_contacts(
            output_dir / f"{interest_name}_{interest_name}_av_contacts_{self.config.project}.pdb",
            avg_interest_contacts
        )

    def calculate_enrichments(self) -> Tuple[pd.DataFrame, Dict]:
        """Calculate contact enrichments between residues."""
        combination_counts = {}
        total_contacts = 0
        total_pairs = 0

        # Collect contact data
        for m_of_interest, (p_start, p_end) in self.m_interest_ranges.items():
            matrix_data = self._get_contact_matrix_for_molecule(m_of_interest, p_start, p_end)
            if matrix_data is None:
                continue

            p_mat, rlabels, clabels = matrix_data
            total_pairs += len(rlabels) * len(clabels)
            total_contacts += np.sum(p_mat)

            self._update_combination_counts(combination_counts, p_mat, rlabels, clabels)

        # Calculate enrichment matrix
        enrichment_matrix = self._calculate_enrichment_matrix(
            combination_counts, total_contacts, total_pairs)
        
        return enrichment_matrix, combination_counts

    def _get_contact_matrix_for_molecule(self, m_of_interest: str, p_start: int, p_end: int) -> Optional[Tuple]:
        """Get contact matrix and labels for a molecule."""
        other_m_of_interest = [p for p in self.m_interest_ranges.keys() if p != m_of_interest]
        other_ranges = [self.m_interest_ranges[p] for p in other_m_of_interest]

        if other_ranges:
            interest_columns = np.concatenate([
                np.arange(p_start, p_end) for (p_start, p_end) in other_ranges
            ])
            p_mat = self.contact_matrix[p_start:p_end, interest_columns]
            rlabels = self.seq[p_start:p_end]
            clabels = self.seq[interest_columns]
        else:
            p_mat = self.contact_matrix[p_start:p_end, p_start:p_end]
            rlabels = self.seq[p_start:p_end]
            clabels = self.seq[p_start:p_end]

        return p_mat, rlabels, clabels

    def _update_combination_counts(self, combination_counts: Dict, 
                                 contact_matrix: np.ndarray,
                                 row_labels: np.ndarray, 
                                 col_labels: np.ndarray):
        """Update combination counts dictionary with contact data."""
        for i, row_label in enumerate(row_labels):
            for j, col_label in enumerate(col_labels):
                combination = frozenset([row_label, col_label])
                count = contact_matrix[i, j]
                
                if combination not in combination_counts:
                    combination_counts[combination] = {
                        'total_count': 0,
                        'occurrences': 0
                    }
                
                combination_counts[combination]['total_count'] += count
                combination_counts[combination]['occurrences'] += 1

    def _calculate_enrichment_matrix(self, combination_counts: Dict, 
                                   total_contacts: int, 
                                   total_pairs: int) -> pd.DataFrame:
        """Calculate enrichment matrix from combination counts."""
        # Extract unique labels
        labels = sorted({label for key in combination_counts.keys() for label in key})
        label_index = {label: i for i, label in enumerate(labels)}
        
        # Initialize matrix
        size = len(labels)
        matrix = np.zeros((size, size))
        
        # Fill matrix
        for combination, values in combination_counts.items():
            if values['occurrences'] > 0 and total_contacts > 0:
                value = (values['total_count'] / total_contacts) / (values['occurrences'] / total_pairs)
            else:
                value = 0
                
            # Update matrix symmetrically
            for label1 in combination:
                for label2 in combination:
                    idx1, idx2 = label_index[label1], label_index[label2]
                    matrix[idx1, idx2] = value
        
        return pd.DataFrame(data=matrix[::-1, :], index=labels[::-1], columns=labels)

    def plot_enrichments(self, enrichment_matrix: pd.DataFrame):
        """Plot contact enrichment heatmap."""
        output_dir = Path(self.config.project)
        out_path = output_dir / f'{self.config.molecules_of_interest.name}_c_enrichments_{self.config.project}.pdf'
        
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        sns.heatmap(enrichment_matrix, annot=False, cmap="coolwarm", vmin=0, center=1)
        plt.title('Contacts enrichment matrix', fontsize=20)
        
        try:
            fig.savefig(out_path)
            print(f"File {out_path} created")
        except Exception as e:
            print(f"Error writing file {out_path}: {str(e)}")
        finally:
            plt.close(fig)


    def plot_scale(self):
            """Plot contact scale visualization comparing generic and interest molecules."""
            if not self.m_generic_ranges:
                return

            output_dir = Path(self.config.project)
            out_path = output_dir / f'{self.config.project}_scale.png'
            out_path_txt = output_dir / f'{self.config.project}_scale.txt'

            # Calculate contacts and collect sequences
            combination_counts = {}
            total_contacts = 0
            total_pairs = 0

            # Get sequences for generic molecules and molecules of interest
            generic_residues = set()
            interest_residues = set()

            # Collect generic residues
            for _, (start, end) in self.m_generic_ranges.items():
                generic_residues.update(self.seq[start:end])

            # Collect residues of interest
            for m_of_interest, (p_start, p_end) in self.m_interest_ranges.items():
                interest_residues.update(self.seq[p_start:p_end])

                # Calculate contacts between interest and generic molecules
                generic_columns = np.concatenate([
                    np.arange(start, end) for start, end in self.m_generic_ranges.values()
                ])

                p_mat = self.contact_matrix[p_start:p_end, generic_columns]
                rlabels = self.seq[p_start:p_end]
                clabels = self.seq[generic_columns]

                total_pairs += len(rlabels) * len(clabels)
                total_contacts += np.sum(p_mat)

                # Update combination counts
                for i, row_label in enumerate(rlabels):
                    for j, col_label in enumerate(clabels):
                        combination = frozenset([row_label, col_label])
                        count = p_mat[i, j]

                        if combination not in combination_counts:
                            combination_counts[combination] = {
                                'total_count': 0,
                                'occurrences': 0
                            }

                        combination_counts[combination]['total_count'] += count
                        combination_counts[combination]['occurrences'] += 1

            # Create enrichment matrix
            all_residues = sorted(generic_residues | interest_residues)
            matrix_size = len(all_residues)
            enrichment_matrix = np.zeros((matrix_size, matrix_size))
            residue_to_index = {res: idx for idx, res in enumerate(all_residues)}

            for combination, values in combination_counts.items():
                if values['occurrences'] > 0 and total_contacts > 0:
                    value = (values['total_count'] / total_contacts) / (values['occurrences'] / total_pairs)
                else:
                    value = 0

                for res1 in combination:
                    for res2 in combination:
                        idx1 = residue_to_index[res1]
                        idx2 = residue_to_index[res2]
                        enrichment_matrix[idx1, idx2] = value

            # Create DataFrames for the full matrix
            enrichment_df = pd.DataFrame(
                enrichment_matrix,
                index=all_residues,
                columns=all_residues
            )

            # Filter matrix to show only generic vs interest interactions
            generic_residues = sorted(generic_residues)
            interest_residues = sorted(interest_residues)
            filtered_matrix = enrichment_df.loc[generic_residues, interest_residues]

            # Save numerical data
            filtered_matrix.to_csv(out_path_txt, sep=' ', mode='a')

            # Create visualization
            height = max(len(generic_residues) * 0.5, 2)
            fig, ax = plt.subplots(figsize=(12, height), constrained_layout=True)

            sns.heatmap(filtered_matrix,
                       annot=False,
                       cmap="coolwarm",
                       vmin=0,
                       center=1,
                       ax=ax,
                       xticklabels=True,
                       yticklabels=True)

            plt.xticks(rotation=45, ha='right')

            try:
                fig.savefig(out_path)
                print(f"File {out_path} created")
            except Exception as e:
                print(f"Error writing file {out_path}: {str(e)}")
            finally:
                plt.close(fig)
def load_config(config_path: str) -> Config:
    """Load and parse YAML configuration file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    return Config(
        project=config_data['project'],
        molecules_of_interest=MoleculeConfig(**config_data['molecules_of_interest']),
        generic_molecules=MoleculeConfig(**config_data['generic_molecules']) 
        if 'generic_molecules' in config_data else None
    )



def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <contacts.json> <config.yaml> <structure.pdb>")
        sys.exit(1)

    # Load input data
    with open(sys.argv[1]) as f:
        contacts = json.load(f)
    contact_matrix = np.array(contacts['map'], dtype=int)
    
    config = load_config(sys.argv[2])
    universe = mda.Universe(sys.argv[3])
    universe.add_TopologyAttr('tempfactors', range(len(universe.atoms)))

    # Initialize analysis
    analysis = ContactAnalysis(universe, config, contact_matrix)
    
    # Perform analysis
    residue_contacts = analysis.analyze_contacts()
    avg_generic_contacts, avg_interest_contacts = analysis.write_results(residue_contacts)
    
    # Export PDB files
    analysis.export_pdb(residue_contacts, avg_interest_contacts, avg_generic_contacts)
    
    # Generate visualizations
    enrichment_matrix, _ = analysis.calculate_enrichments()
    analysis.plot_enrichments(enrichment_matrix)
    analysis.plot_scale()

if __name__ == "__main__":
    main()
