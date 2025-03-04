import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import freesasa
from MDAnalysis.coordinates.PDB import PDBWriter
from dataclasses import dataclass

# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

@dataclass
class ChainConfig:
    chain_id: str
    atom_range: List[int]

@dataclass
class MoleculeConfig:
    name: str
    components: Dict[str, List[int]]
    chains: Optional[Dict[str, List[ChainConfig]]] = None  # Each component can have multiple chains

@dataclass
class Config:
    project: str
    molecules_of_interest: MoleculeConfig
    generic_molecules: Optional[MoleculeConfig]

    def validate_chain_ranges(self):
        """Validate that chain atom ranges are within their component bounds."""

        def _validate_component_chains(component_name: str,
                                       component_range: List[int],
                                       chains: List[ChainConfig]) -> List[str]:
            errors = []
            start, end = component_range
            for chain in chains:
                chain_start, chain_end = chain.atom_range
                if chain_start < start or chain_end > end:
                    errors.append(
                        f"Chain {chain.chain_id} range [{chain_start}, {chain_end}] in component "
                        f"{component_name} exceeds component bounds [{start}, {end}]"
                    )
                # Check for overlapping chains
                for other_chain in chains:
                    if chain != other_chain:
                        if (chain_start <= other_chain.atom_range[1] and
                                chain_end >= other_chain.atom_range[0]):
                            errors.append(
                                f"Overlapping chain ranges detected in {component_name}: "
                                f"Chain {chain.chain_id} [{chain_start}, {chain_end}] overlaps with "
                                f"Chain {other_chain.chain_id} {other_chain.atom_range}"
                            )
            return errors

        errors = []

        # Validate molecules of interest
        if self.molecules_of_interest.chains:
            for comp_name, comp_range in self.molecules_of_interest.components.items():
                if comp_name in self.molecules_of_interest.chains:
                    errors.extend(_validate_component_chains(
                        comp_name,
                        comp_range,
                        self.molecules_of_interest.chains[comp_name]
                    ))

        # Validate generic molecules
        if self.generic_molecules and self.generic_molecules.chains:
            for comp_name, comp_range in self.generic_molecules.components.items():
                if comp_name in self.generic_molecules.chains:
                    errors.extend(_validate_component_chains(
                        comp_name,
                        comp_range,
                        self.generic_molecules.chains[comp_name]
                    ))

        if errors:
            raise ValueError("Chain validation errors:\n" + "\n".join(errors))

class ContactAnalysis:
    def __init__(self, universe: mda.Universe, config: Config, contact_matrix: np.ndarray):
        self.universe = universe
        self.config = config
        self.contact_matrix = contact_matrix
        self.seq = self._clean_sequence()
        self.original_resids = self._store_original_resids()
        self._renumber_residues()
        self.sasa_list = self._calc_sasa()
        self.m_interest_ranges, self.m_generic_ranges = self._convert_atom_ranges()

    def _calc_sasa(self):
        structure = freesasa.Structure()
        freesasa.setVerbosity(1)
        for a in self.universe.atoms:
            x, y, z = a.position
            resname = a.resname
            structure.addAtom(a.type.rjust(2), resname, a.resid.item(), a.segid, x, y, z)

        parameters = freesasa.Parameters()
        result = freesasa.calc(structure, parameters)
        residue_areas = [result.residueAreas()[s][r] for s in list(result.residueAreas().keys()) for r in
                         list(result.residueAreas()[s].keys())]
        sasa_list = [r.total for r in residue_areas]
        return sasa_list

    def _store_original_resids(self) -> Dict[int, int]:
        """Store mapping between sequential and original residue IDs."""
        return {i + 1: res.resid for i, res in enumerate(self.universe.residues)}

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
        """Renumber residues in universe sequentially for internal processing."""
        for i, residue in enumerate(self.universe.residues, start=1):
            residue.resid = i

    def _get_residue_identifier(self, resid: int) -> str:
        """Get the original residue number and name for a given residue ID."""
        residue = self.universe.residues[resid - 1]  # -1 for 0-based indexing
        original_resid = self.original_resids[resid]
        resname = residue.resname
        return f"{original_resid}{resname}"

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

    def _assign_chains(self, universe: mda.Universe, molecule_config: MoleculeConfig,
                       component_name: str, start_idx: int, end_idx: int):
        """Assign chain IDs to atoms based on config."""
        # Default to no chain ID if not specified
        for atom in universe.atoms[start_idx:end_idx]:
            atom.chainID = ""

        if not molecule_config.chains or component_name not in molecule_config.chains:
            return

        for chain_config in molecule_config.chains[component_name]:
            chain_start = chain_config.atom_range[0] - 1  # Convert to 0-based indexing
            chain_end = chain_config.atom_range[1]
            for atom in universe.atoms[chain_start:chain_end]:
                atom.chainID = chain_config.chain_id

    def export_pdb(self, residue_contacts: Dict, avg_interest_contacts: np.ndarray,
                   avg_generic_contacts: np.ndarray):
        """Export PDB files with contact information in B-factors and specified chain IDs."""
        output_dir = Path(self.config.project)
        generic_name = self.config.generic_molecules.name if self.config.generic_molecules else ""
        interest_name = self.config.molecules_of_interest.name

        # Create a copy of the universe to preserve original residue IDs
        temp_universe = self.universe.copy()
        # Restore original residue IDs
        for residue in temp_universe.residues:
            residue.resid = self.original_resids[residue.resid]

        for name_m_interest, (start, end) in self.m_interest_ranges.items():
            # Create new segment for the molecule
            new_segment = temp_universe.add_Segment(segid=name_m_interest)
            residues = temp_universe.residues[start:end]
            residues.segments = new_segment
            # Get the atom indices for this component
            component_range = self.config.molecules_of_interest.components[name_m_interest]
            start_idx = component_range[0] - 1
            end_idx = component_range[1]

            # Assign chain IDs based on atom ranges
            self._assign_chains(temp_universe, self.config.molecules_of_interest,
                                name_m_interest, start_idx, end_idx)

            self._write_molecule_pdbs(
                output_dir, name_m_interest, residues,
                residue_contacts[name_m_interest],
                avg_generic_contacts, avg_interest_contacts,
                generic_name, interest_name, start
            )

    def _write_molecule_pdbs(self, output_dir: Path, molecule_name: str, residues: mda.AtomGroup,
                             contacts: Dict, avg_generic_contacts: np.ndarray,
                             avg_interest_contacts: np.ndarray, generic_name: str,
                             interest_name: str, start: int):
        """Write PDB files and corresponding text files for a single molecule with contact information."""
        def _write_pdb_with_contacts(filename: Path, contact_values: np.ndarray):
            for i, residue in enumerate(residues.residues):
                for atom in residue.atoms:
                    atom.tempfactor = contact_values[i]
            with PDBWriter(filename) as pdb_writer:
                pdb_writer.write(residues)

            # Write corresponding text file
            text_filename = filename.with_suffix('.txt')
            self._write_contact_text_file(text_filename, contact_values, start)

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
        """Write analysis results and return average contacts."""
        output_dir = Path(self.config.project)
        output_dir.mkdir(exist_ok=True)

        # Calculate average contacts across all molecules of interest
        avg_interest_contacts = np.zeros(max(len(contacts['interest_contacts'])
                                             for contacts in residue_contacts.values()))
        count_interest = np.zeros_like(avg_interest_contacts)

        for contacts in residue_contacts.values():
            interest_data = contacts['interest_contacts']
            avg_interest_contacts[:len(interest_data)] += interest_data
            count_interest[:len(interest_data)] += 1

        # Avoid division by zero
        mask = count_interest > 0
        avg_interest_contacts[mask] /= count_interest[mask]

        # Calculate average generic contacts if they exist
        avg_generic_contacts = np.array([])
        if any(len(contacts['generic_contacts']) > 0 for contacts in residue_contacts.values()):
            avg_generic_contacts = np.zeros(max(len(contacts['generic_contacts'])
                                                for contacts in residue_contacts.values()))
            count_generic = np.zeros_like(avg_generic_contacts)

            for contacts in residue_contacts.values():
                generic_data = contacts['generic_contacts']
                if len(generic_data) > 0:
                    avg_generic_contacts[:len(generic_data)] += generic_data
                    count_generic[:len(generic_data)] += 1

            # Avoid division by zero
            mask = count_generic > 0
            avg_generic_contacts[mask] /= count_generic[mask]

        return avg_generic_contacts, avg_interest_contacts

    def _write_contact_text_file(self, filename: Path, contact_values: np.ndarray,
                               start_resid: int):
        """Write contact data to a text file with index, value, and residue identifier."""
        with open(filename, 'w') as f:
            f.write("@    title \"Contacts profile\"\n")
            f.write("@    yaxis  label \"Contact Frequency\"\n")
            f.write("@    xaxis  label \"Residue\"\n")
            f.write("@TYPE xy\n")
            for i, value in enumerate(contact_values, 1):
                residue_id = self._get_residue_identifier(start_resid + i)
                f.write(f"{i}\t{value:.3f}\t\"{residue_id}\"\n")


    def sasa_norm_enrichment(self) -> Tuple[pd.DataFrame, Dict]:
        """Calculate contact enrichments between residue types in molecules of interest group."""
        combination_counts = {}
        total_contacts = 0
        total_pairs = 0
        # Collect contact data
        for m_of_interest, (p_start, p_end) in self.m_interest_ranges.items():
            matrix_data = self._get_contact_matrix_for_molecule(m_of_interest, p_start, p_end)
            if matrix_data is None:
                continue

            p_mat, rlabels, clabels, row_indices, col_indices = matrix_data
            total_pairs += len(rlabels) * len(clabels)
            total_contacts += np.sum(p_mat)

            self._update_combination_counts(combination_counts, p_mat, rlabels, clabels,
                                            row_indices, col_indices)

        # Calculate enrichment matrix
        enrichment_matrix = self._calculate_enrichment_matrix(
            combination_counts, total_contacts, total_pairs)
        
        return enrichment_matrix, combination_counts

    def _get_contact_matrix_for_molecule(self, m_of_interest: str, p_start: int, p_end: int) -> Optional[Tuple]:
        """Get contact matrix and labels for a molecule of interest."""
        other_m_of_interest = [p for p in self.m_interest_ranges.keys() if p != m_of_interest]
        other_ranges = [self.m_interest_ranges[p] for p in other_m_of_interest]

        if other_ranges:
            interest_columns = np.concatenate([
                np.arange(p_start, p_end) for (p_start, p_end) in other_ranges
            ])
            p_mat = self.contact_matrix[p_start:p_end, interest_columns]
            rlabels = self.seq[p_start:p_end]
            clabels = self.seq[interest_columns]
            row_indices = np.arange(p_start, p_end)
            col_indices = interest_columns
        else:
            p_mat = self.contact_matrix[p_start:p_end, p_start:p_end]
            rlabels = self.seq[p_start:p_end]
            clabels = self.seq[p_start:p_end]
            row_indices = np.arange(p_start, p_end)
            col_indices = np.arange(p_start, p_end)

        return p_mat, rlabels, clabels, row_indices, col_indices

    def _update_combination_counts(self, combination_counts: Dict, 
                                 contact_matrix: np.ndarray,
                                 row_labels: np.ndarray, 
                                 col_labels: np.ndarray,
                                 row_indices: np.ndarray,
                                 col_indices: np.ndarray):
        """Update combination counts dictionary with contact data for contacts between molecules of interest."""
        total_row_sasa = sum(np.array(self.sasa_list)[row_indices])
        total_col_sasa = sum(np.array(self.sasa_list)[col_indices])
        sasa_interest = total_row_sasa + total_col_sasa

        for i, row_label in enumerate(row_labels):
            row_idx = row_indices[i]  # Get the actual index in the original sequence
            for j, col_label in enumerate(col_labels):
                col_idx = col_indices[j]  # Get the actual index in the original sequence
                combination = frozenset([row_label, col_label])
                count = contact_matrix[i, j]

                # Get product of Residue SASA for every residue pair
                sasa_product = ((self.sasa_list[row_idx] * self.sasa_list[col_idx])/((sasa_interest-total_row_sasa)*sasa_interest))


                if combination not in combination_counts:
                    combination_counts[combination] = {
                        'total_count': 0,
                        'occurrences': 0,
                        'sasa_norm': 0
                    }
                
                combination_counts[combination]['total_count'] += count
                combination_counts[combination]['occurrences'] += 1
                combination_counts[combination]['sasa_norm'] += sasa_product

    def _calculate_enrichment_matrix(self, combination_counts: Dict, 
                                   total_contacts: int, 
                                   total_pairs: int) -> pd.DataFrame:
        """Calculate sasa normalized enrichment matrix between residues in molecules of interest from combination count dictionary."""
        # Extract unique labels
        labels = sorted({label for key in combination_counts.keys() for label in key})
        label_index = {label: i for i, label in enumerate(labels)}
        
        # Initialize matrix
        size = len(labels)
        matrix = np.zeros((size, size))
        
        # Fill matrix
        for combination, values in combination_counts.items():
            if values['occurrences'] > 0 and total_contacts > 0:
                value = (values['total_count'] / total_contacts) / values['sasa_norm']
            else:
                value = 0
                
            # Update matrix symmetrically
            for label1 in combination:
                for label2 in combination:
                    idx1, idx2 = label_index[label1], label_index[label2]
                    matrix[idx1, idx2] = value
        
        return pd.DataFrame(data=matrix[::-1, :], index=labels[::-1], columns=labels)

    def plot_enrichments(self, enrichment_matrix: pd.DataFrame):
        """Plot heatmap for sasa normalized enrichments between residues in molecules of interest."""
        output_dir = Path(self.config.project)
        out_path = output_dir / f'{self.config.molecules_of_interest.name}_c_enrichments_{self.config.project}.pdf'
        
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        sns.heatmap(enrichment_matrix, annot=False, cmap="coolwarm", vmin=0, center=1)
        plt.title('Contacts enrichment matrix', fontsize=20)
        
        try:
            fig.savefig(out_path)
        except Exception as e:
            print(f"Error writing file {out_path}: {str(e)}")
        finally:
            plt.close(fig)


    def sasa_norm_generic_propensity(self):
            """Calculates, plots and writes to text, solvent exposure normalized propensities between all residues in the molecule of interest group and the generic group"""
            if not self.m_generic_ranges:
                return
            output_dir = Path(self.config.project)
            out_path = output_dir / f'{self.config.project}_propensity.png'
            out_path_txt = output_dir / f'{self.config.project}_scale.txt'

            # Calculate contacts and collect sequences
            combination_counts = {}
            total_contacts = 0
            total_pairs = 0

            # Get sequences for generic molecules and molecules of interest
            generic_residues = set()
            interest_residues = set()
            sasa_interest = 0
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
                row_indices = np.arange(p_start, p_end)
                col_indices = generic_columns
                total_pairs += len(rlabels) * len(clabels)
                total_contacts += np.sum(p_mat)

                total_row_sasa = sum(np.array(self.sasa_list)[row_indices])
                total_col_sasa = sum(np.array(self.sasa_list)[col_indices])

                sasa_interest += total_row_sasa
                sasa_generic = total_col_sasa

                # Update combination counts
                for i, row_label in enumerate(rlabels):
                    row_idx = row_indices[i]
                    for j, col_label in enumerate(clabels):
                        col_idx = col_indices[j]
                        combination = frozenset([row_label, col_label])
                        count = p_mat[i, j]

                        # Get product of Residue SASA for every residue pair
                        sasa_product = self.sasa_list[row_idx] * self.sasa_list[col_idx]

                        if combination not in combination_counts:
                            combination_counts[combination] = {
                                'total_count': 0,
                                'occurrences': 0,
                                'total_sasa': 0
                            }

                        combination_counts[combination]['total_count'] += count
                        combination_counts[combination]['occurrences'] += 1
                        combination_counts[combination]['total_sasa'] += sasa_product

            # Create sasa normalized enrichment matrix
            all_residues = sorted(generic_residues | interest_residues)
            matrix_size = len(all_residues)
            enrichment_matrix = np.zeros((matrix_size, matrix_size))
            enrichment_matrix_log = np.zeros((matrix_size, matrix_size))
            residue_to_index = {res: idx for idx, res in enumerate(all_residues)}

            for combination, values in combination_counts.items():
                if values['occurrences'] > 0 and total_contacts > 0:
                    value = (values['total_count'] / total_contacts) / (values['total_sasa']/(sasa_interest*sasa_generic))
                    log_value = -1 * (math.log(value))
                else:
                    value = 0
                    log_value = 0

                for res1 in combination:
                    for res2 in combination:
                        idx1 = residue_to_index[res1]
                        idx2 = residue_to_index[res2]
                        enrichment_matrix[idx1, idx2] = value
                        enrichment_matrix_log[idx1, idx2] = log_value

            # Create DataFrames for the full matrix
            enrichment_df = pd.DataFrame(
                enrichment_matrix,
                index=all_residues,
                columns=all_residues
            )
            enrichment_df_ln = pd.DataFrame(
                enrichment_matrix_log,
                index=all_residues,
                columns=all_residues
            )
            # Filter matrix to show only generic vs interest interactions
            generic_residues = sorted(generic_residues)
            interest_residues = sorted(interest_residues)
            filtered_matrix = enrichment_df.loc[generic_residues, interest_residues]
            filtered_matrix_log = enrichment_df_ln.loc[generic_residues, interest_residues]

            # Create visualization
            height = max(len(generic_residues) * 0.5, 2)
            fig, ax = plt.subplots(figsize=(12, height), constrained_layout=True)

            sns.heatmap(filtered_matrix,
                       annot=False,
                       cmap="coolwarm",
                       ax=ax,
                       xticklabels=True,
                       yticklabels=True,
                       vmin=0,
                       center=1)

            plt.xticks(rotation=45, ha='right')

            try:
                fig.savefig(out_path)
            except Exception as e:
                print(f"Error writing file {out_path}: {str(e)}")
            finally:
                plt.close(fig)

            # Save numerical data
            filtered_matrix_log.to_csv(out_path_txt, sep=' ', mode='w')

def load_config(config_path: str) -> Config:
    """Load and parse YAML configuration file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    def _process_chains(chains_data: Dict) -> Dict[str, List[ChainConfig]]:
        processed_chains = {}
        for comp_name, chains_list in chains_data.items():
            processed_chains[comp_name] = [
                ChainConfig(
                    chain_id=chain['chain_id'],
                    atom_range=chain['atom_range']
                )
                for chain in chains_list
            ]
        return processed_chains

    # Process molecules_of_interest
    mol_interest = config_data['molecules_of_interest']
    mol_interest_config = MoleculeConfig(
        name=mol_interest['name'],
        components=mol_interest['components'],
        chains=_process_chains(mol_interest['chains']) if 'chains' in mol_interest else None
    )

    # Process generic_molecules if present
    generic_mol_config = None
    if 'generic_molecules' in config_data:
        generic_mol = config_data['generic_molecules']
        generic_mol_config = MoleculeConfig(
            name=generic_mol['name'],
            components=generic_mol['components'],
            chains=_process_chains(generic_mol['chains']) if 'chains' in generic_mol else None
        )

    config = Config(
        project=config_data['project'],
        molecules_of_interest=mol_interest_config,
        generic_molecules=generic_mol_config
    )

    # Validate chain ranges
    config.validate_chain_ranges()

    return config
