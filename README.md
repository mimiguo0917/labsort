# labsort

A CLI tool for scanning, classifying, and organizing lab files. Built for computational biology workflows with smart handling of sequencing data, spreadsheets, scripts, and more.

## Features

- **Scan & classify** files into 25+ categories (sequencing, flow cytometry, scripts, images, etc.)
- **Full-text search** with SQLite FTS5 indexing
- **Smart organize** into a clean hierarchy: `Data/`, `Documents/`, `Images/`, `Code/`, `Other/`
- **Duplicate detection** (exact SHA256 + near-duplicate by name/size)
- **Interactive review** mode for sequencing data — mark analyzed data for cleanup
- **Fully undoable** — every organize/clean operation can be reversed with `labsort undo`
- **Never deletes** — disposable files go to `Save to Delete/`, not the trash

## Installation

Requires Python 3.10+.

```bash
pip install -e /path/to/labsort
```

## Usage

### Scan a directory
```bash
labsort scan ~/Downloads
```

### Search indexed files
```bash
labsort search "plasmid map"
```

### View file info and category breakdown
```bash
labsort info ~/Downloads
labsort tree ~/Downloads
```

### Organize files into a clean folder structure
```bash
# Dry run (preview only)
labsort organize ~/Downloads

# Execute
labsort organize ~/Downloads --execute

# With interactive sequencing review
labsort organize ~/Downloads --review
```

### Clean up duplicates and orphaned index files
```bash
labsort clean ~/Downloads
```

### Undo the last operation
```bash
labsort undo
```

### View operation history
```bash
labsort history
```

## Organize Structure

```
Data/
├── Sequencing/
│   ├── Sanger Sequencing/
│   ├── Sequences/
│   ├── FASTQ Reads/
│   └── Cloning & Plasmid Maps/
├── Spreadsheets & Data/
├── Flow Cytometry/
Documents/
Images/
Code/
Other/
Save to Delete/
```

## Dependencies

- [Click](https://click.palletsprojects.com/) — CLI framework
- [Rich](https://github.com/Textualize/rich) — terminal formatting
- [PyYAML](https://pyyaml.org/) — configuration
