# MD Contact Analysis

A comprehensive tool for molecular dynamics contact analysis.

## Installation

```bash
pip install .
```

## Usage

### Generate Contact Matrix
```bash
md_contacts calculate -s structure.gro -f trajectory.xtc -o contact_map.json
```

### Analyze Contacts
```bash
md-contacts analyze -s structure.gro -c config.yaml -m contact_map.json
```
