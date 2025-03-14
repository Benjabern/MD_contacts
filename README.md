# MD Contact Analysis
Calculates and analyzes all against all residue wise contact counts from MD simulation trajectories. The Tool aims to provide paralelizable and scalable calculation of contact matrices, as well as customizable analysis functionality.

## Installation

```bash
git clone https://github.com/Benjabern/MD_contacts
cd MD_contacts/
pip install .
```

## Usage

### Calculate Contact Matrix
To generate a json file containing an all against all count of contacts between residues, using 128 cores run:

```bash
md_contacts calculate -s structure.gro -f trajectory.xtc -o contact_map.json -j 128
```
It is recomended to do a test run with a small number of parallel jobs (-j or --jobs) to estimate the memory requirements per job. 
The chunk size parameter is usually fine at default value but lower values specified with --chunk-size can be used in a test run to estimate the time requirements of the calculations.
A cutoff in Angstrom can be specified using the -c/--cutoff argument (default: 3.5).
A structure (.gro) and trajectory (.xtc) file need to be specified with the -s and -f flags respectively.
The -o/--output flag allows to specify the name of the system for the output file.

### Analyze Contacts
To analyse the contact matrix, in addition to the .json output from the calculate module and a corresponding .gro file,
a config file specifying atom ranges of interest needs to be provided (see template yaml file). 

```bash
md_contacts analyze -s structure.gro -c config.yaml -m contact_map.json
```
Additionally, to preserve proper bonds in output .pdb files, chain identifiers for non consecutively numbered residues should be present in the config file, i.e. if two residues in the same components have the same residue number they need to be specified to belong to different chains.
All specified molecules of interest will be analysed individually as well as on average. A residue wise profile
of contacts will be created for contacts to the ligand group as well as to all other members of the molecule of interest group.
A pdb structure for each member of the molecule of interst group containing the the contacts to other molecules of interst as well as the ligand as b-factors will be created in addition to a matrix of pairwise enrichments in contacts bewteen molecules of interest.

