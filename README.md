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
To generate a json file containing an all-against-all count of contacts between residues as well as a h5 file containing a binary contact matrix for every frame, using 128 cores run:

```bash
md_contacts calculate -s structure.gro -f trajectory.xtc -o contact_map.json -j 128
```
It is recomended to do a test run with a small number of parallel jobs (-j or --jobs) to estimate the memory requirements per job. 
The chunk size parameter is usually fine at default value but lower values specified with --chunk-size can be used in a test run to estimate the time requirements of the calculations.
A cutoff in Angstrom can be specified using the -c/--cutoff argument (default: 3.5).
A structure (.gro) and trajectory (.xtc) file need to be specified with the -s and -f flags respectively.
The -o/--output flag allows to specify the name of the system for the output file. 
The trajectory can be sliced using the -b, -e and -df flags, for the first and last frame index to consider as well as the step (every nth frame) to use.
E.g. using -b 0 -e 100000 -df 10 will calculate contacts from the beginning of the trajectory up until frame 100000 for every 10th frame.
For further analysis, the binary residue wise contact matrix for every frame is written to a h5 file (contained in the "cmaps" dataset).

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

### Output files
The files in the example_output folder provided follow the example_config.yaml file. For every copy of a protein (P1-P5) outlined in the config file, a contacts profile with other molecules of interest (other protein copies), the specified generic molecules (Adenine), and an average profile (denoted by _av_ in filename) is created in text format (xmgrace formatted). For each of these profiles, a .pdb file with the contacts per frame added to the B-factor column is created. Lastly, a matrix of solvent accesibility normalized contact enrichments between residues of the molecule of interest group (P1-P5) as well as to the generic molecules residues is output. The negative natural logarithm of these enrichments is additionally output to a text file. 
