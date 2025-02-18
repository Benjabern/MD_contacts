# MD Contact Analysis

A comprehensive tool for molecular dynamics contact analysis.

## Installation



```bash
git clone https://github.com/Benjabern/MD_contacts
cd MD_contacts/
pip install .
```

## Usage

### Generate Contact Matrix
```bash
md_contacts calculate -s structure.gro -f trajectory.xtc -o contact_map.json
```

### Analyze Contacts
```bash
md_contacts analyze -s structure.gro -c config.yaml -m contact_map.json
```
