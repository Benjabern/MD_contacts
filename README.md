## Calculation of contacts
Two main scripts are provided. To calculate an all against all contact matrix in the form of a json file, ccontacts.py is used.
The script is highly parralelizable and scales well with number of provided cores.
It is recomended to do a test run with a small number of parallel jobs (-j or --jobs) to estimate the memory requirements
per job. The chunk size parameter is usually fine at default value but lower values specified with --chunk-size can be used in
a test run to estimate the time requirements of the calculations.
A cutoff in Angstrom can be specified using the -c/--cutoff argument (default: 3.5).
A structure (.gro) and trajectory (.xtc) file need to be specified with the -s and -f flags respectively.
The -o/--output flag allows to specify the name of the system for the output file.
# Example usage
Using 128 cores
> python3 ccontacts.py -s mystructure.gro -f mytrajectory.xtc -j 128 -o mysystem -l log_mysystem

## Analysis of contacts
To analyse the contact matrix, in addition to the .json output from ccontacts.py and a corresponding .gro file,
a config file specifying residue index ranges of interest needs to be provided (see template config file).

***Crucially, the nature of the contacts calculation and consequently the analysis script converts all original residue numberings into one continous numbering scheme which might deviate from the residue numberings in the original topology. This is releveant for multicopy systems or systems specifying multiple protein chains. The new indexes correspond to .gro line numbers -1 (ignoring box vectors, titles, comments, atoms count, etc.)***

All specified molecules of interest will be analysed individually as well as on average. A residue wise profile
of contacts will be created for contacts to the ligand group as well as to all other members of the molecule of interest group.
A pdb structure for each member of the molecule of interst group containing the the contacts to other molecules of interst as well
as the ligand as b-factors will be created in addition to a matrix of pairwise enrichments in contacts bewteen molecules of interest.
# Example usage
> python3 acontacts.py mysystem_contacts.json mysystem.cfg mystructure.gro

## Requirements
python3

numpy

MDAnalysis

seaborn

matplotlib
