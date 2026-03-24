"""File classification engine.

Rule priority:
  1. Compound extension match
  2. Simple extension match
  3. Disambiguation (context-aware for ambiguous extensions)
  4. Name/path pattern match
  5. Fallback to 'other'

Each category is a dict with:
  - extensions: list of extensions (with dot)
  - name_patterns: list of regex patterns to match against filename
  - dir_patterns: list of regex patterns to match against parent dir name
  - group_with_extensions: companion extensions that travel with this type
  - disambiguate: callable(filepath, magic_bytes, siblings) -> bool
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .utils import get_compound_extension, read_magic_bytes


@dataclass
class Category:
    name: str
    label: str
    extensions: list[str] = field(default_factory=list)
    compound_extensions: list[str] = field(default_factory=list)
    name_patterns: list[str] = field(default_factory=list)
    dir_patterns: list[str] = field(default_factory=list)
    group_with_extensions: list[str] = field(default_factory=list)
    disambiguate: Callable | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Disambiguation helpers
# ---------------------------------------------------------------------------

def _is_genomics_h5(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """h5 in a genomics context (cellranger, anndata, 10x)."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    genomics_hints = [
        "filtered_feature_bc_matrix", "raw_feature_bc_matrix",
        "molecule_info", "cellranger", "cellbender", "anndata",
    ]
    if any(h in name or h in parent for h in genomics_hints):
        return True
    # Check siblings for genomics files
    genomics_exts = {".bam", ".bai", ".bed", ".gtf", ".mtx", ".tsv.gz"}
    sibling_exts = {Path(s).suffix.lower() for s in siblings}
    return bool(genomics_exts & sibling_exts)


def _is_microscopy_h5(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """h5/hdf5 in a microscopy context."""
    parent = fp.parent.name.lower()
    micro_hints = ["microscopy", "imaging", "confocal", "timelapse", "zstack"]
    return any(h in parent for h in micro_hints)


def _is_genomics_fasta(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """FASTA used as a reference genome — requires positive signal, not just size.

    Large FASTA files from pipelines (USEARCH centroids, dereplication, etc.)
    should NOT be classified as reference genomes.
    """
    name = fp.name.lower()
    parent = fp.parent.name.lower()

    # Keyword match in filename or parent dir
    ref_hints = ["genome", "reference", "chr", "assembly", "grch", "mm10", "hg38",
                 "hg19", "mm39", "t2t", "mfas", "macfas", "rhemac"]
    has_ref_keyword = any(h in name or h in parent for h in ref_hints)

    # Pipeline intermediates — definitely NOT reference genomes
    pipeline_hints = ["uncategorized", "centroids", "cluster", "derep",
                      "consensus", "filtered", "merged", "chimera"]
    is_pipeline_output = any(h in name for h in pipeline_hints)
    if is_pipeline_output:
        return False

    if has_ref_keyword:
        return True

    # Check for genome index siblings (.fai, .dict, .bwt) — strong signal
    genome_idx_exts = {".fai", ".dict", ".bwt", ".pac", ".sa", ".amb", ".ann"}
    sibling_exts = {Path(s).suffix.lower() for s in siblings}
    if genome_idx_exts & sibling_exts:
        return True

    return False


def _is_paper_pdf(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """PDF that is a paper/publication vs a presentation export."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    paper_hints = ["paper", "publication", "manuscript", "preprint", "journal",
                   "review", "doi", "arxiv", "biorxiv", "medrxiv"]
    pres_hints = ["slide", "presentation", "talk", "poster", "lecture"]
    if any(h in name or h in parent for h in pres_hints):
        return False
    if any(h in name or h in parent for h in paper_hints):
        return True
    # Check for companion .pptx — if present, this PDF is likely an export
    sibling_stems = {Path(s).stem.lower() for s in siblings}
    my_stem = fp.stem.lower()
    sibling_names = [s.lower() for s in siblings]
    if any(s.endswith(".pptx") and Path(s).stem.lower() == my_stem for s in sibling_names):
        return False
    return True  # Default: PDFs are papers


def _is_microscopy_tif(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """TIF/TIFF from microscopy vs a generic image."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    # ome.tif is always microscopy (handled by compound ext)
    micro_hints = ["confocal", "zstack", "timelapse", "microscop", "fluoresc",
                   "dapi", "fitc", "tritc", "cy5", "gfp", "rfp", "bf_", "dic_"]
    if any(h in name or h in parent for h in micro_hints):
        return True
    # Large TIFs (>50MB) are likely microscopy
    size = fp.stat().st_size if fp.exists() else 0
    if size > 50 * 1024 * 1024:
        return True
    # Siblings with .scn, .nd2, .lif, .czi suggest microscopy
    micro_exts = {".scn", ".nd2", ".lif", ".czi", ".vsi", ".svs"}
    sibling_exts = {Path(s).suffix.lower() for s in siblings}
    return bool(micro_exts & sibling_exts)


def _is_figure_image(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """PNG/JPG/SVG that is a figure vs a random screenshot/photo."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    fig_hints = ["figure", "fig_", "fig.", "panel", "plot", "graph", "chart",
                 "heatmap", "volcano", "pca", "umap", "tsne", "barplot"]
    if any(h in name or h in parent for h in fig_hints):
        return True
    return False


def _is_gel_image(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """Image that is a gel/blot."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    gel_hints = ["gel", "blot", "western", "southern", "northern",
                 "coomassie", "sypro", "ethidium", "sybr"]
    return any(h in name or h in parent for h in gel_hints)


def _is_compressed_genomics(fp: Path, magic: bytes, siblings: list[str]) -> bool:
    """Generic .gz that is genomics data (not .fastq.gz etc which are compound)."""
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    genomics_hints = ["genome", "alignment", "variant", "snp", "indel",
                      "coverage", "peaks", "chipseq", "atacseq", "rnaseq"]
    return any(h in name or h in parent for h in genomics_hints)


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

CATEGORIES: list[Category] = [
    # --- Sequencing / Genomics ---
    Category(
        name="fastq",
        label="FASTQ Reads",
        compound_extensions=[".fastq.gz", ".fq.gz"],
        extensions=[".fastq", ".fq"],
        group_with_extensions=[".fastq.gz", ".fq.gz"],
        description="Raw sequencing reads",
    ),
    Category(
        name="bam_cram",
        label="Alignments",
        extensions=[".bam", ".cram", ".sam"],
        group_with_extensions=[".bai", ".crai", ".bam.bai"],
        description="Aligned sequencing reads",
    ),
    Category(
        name="vcf",
        label="Variant Calls",
        compound_extensions=[".vcf.gz", ".vcf.bgz"],
        extensions=[".vcf", ".bcf", ".gvcf"],
        group_with_extensions=[".tbi", ".csi"],
        description="Variant call files",
    ),
    Category(
        name="bed_annotation",
        label="Genome Annotations",
        compound_extensions=[".bed.gz", ".gff.gz", ".gtf.gz"],
        extensions=[".bed", ".gff", ".gff3", ".gtf", ".bedgraph", ".narrowpeak",
                    ".broadpeak", ".bigbed", ".bb"],
        description="Genomic intervals and annotations",
    ),
    Category(
        name="reference_genome",
        label="Reference Genomes",
        extensions=[".fa", ".fasta", ".fna", ".2bit"],
        group_with_extensions=[".fai", ".dict", ".amb", ".ann", ".bwt", ".pac", ".sa"],
        disambiguate=_is_genomics_fasta,
        description="Reference genome sequences",
    ),
    Category(
        name="sequences",
        label="Sequences",
        extensions=[".fa", ".fasta", ".fna", ".fas"],
        description="DNA/RNA sequence files (FASTA)",
    ),
    Category(
        name="cloning_sequence",
        label="Cloning & Plasmid Maps",
        extensions=[".gb", ".gbk", ".genbank", ".dna", ".ape", ".snapgene", ".xdna"],
        group_with_extensions=[".gb", ".gbk", ".dna", ".fasta", ".fa"],
        description="GenBank, SnapGene, and plasmid map files",
    ),
    Category(
        name="sequence_protein",
        label="Protein Sequences",
        extensions=[".faa", ".pdb", ".mmcif"],
        description="Protein sequences and structures",
    ),
    Category(
        name="counts_matrix",
        label="Count Matrices",
        extensions=[".mtx", ".h5ad", ".loom"],
        name_patterns=[r"counts?[_\.]", r"expression[_\.]", r"umi[_\.]"],
        description="Gene expression count matrices",
    ),
    Category(
        name="genomics_index",
        label="Genomics Indices",
        extensions=[".bai", ".crai", ".tbi", ".csi", ".fai", ".dict",
                    ".amb", ".ann", ".bwt", ".pac", ".sa", ".idx"],
        description="Index files for genomics data",
    ),
    Category(
        name="genomics_h5",
        label="Genomics HDF5",
        extensions=[".h5", ".hdf5"],
        disambiguate=_is_genomics_h5,
        description="HDF5 files from single-cell / genomics pipelines",
    ),

    # --- Bioinformatics intermediates ---
    Category(
        name="usearch_output",
        label="USEARCH/VSEARCH Output",
        extensions=[".uc"],
        description="USEARCH/VSEARCH cluster and mapping output",
    ),
    Category(
        name="numpy_data",
        label="NumPy Arrays",
        extensions=[".npy", ".npz"],
        description="NumPy serialized arrays (embeddings, matrices)",
    ),
    Category(
        name="parquet_data",
        label="Parquet Data",
        extensions=[".parquet", ".pq"],
        description="Apache Parquet columnar data files",
    ),

    # --- Microscopy / Imaging ---
    Category(
        name="microscopy_proprietary",
        label="Microscopy (Proprietary)",
        extensions=[".nd2", ".lif", ".czi", ".vsi", ".oib", ".oif",
                    ".ets", ".scn", ".svs", ".mrxs", ".qptiff"],
        description="Proprietary microscope image formats",
    ),
    Category(
        name="microscopy_tif",
        label="Microscopy TIFF",
        compound_extensions=[".ome.tif", ".ome.tiff"],
        extensions=[".tif", ".tiff"],
        disambiguate=_is_microscopy_tif,
        description="TIFF images from microscopy",
    ),
    Category(
        name="microscopy_h5",
        label="Microscopy HDF5",
        extensions=[".h5", ".hdf5"],
        disambiguate=_is_microscopy_h5,
        description="HDF5 files from microscopy/imaging pipelines",
    ),

    # --- Gel / Blot images ---
    Category(
        name="gel_images",
        label="Gel/Blot Images",
        extensions=[".gel", ".scn", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        name_patterns=[r"(?i)gel", r"(?i)blot", r"(?i)western", r"(?i)southern", r"(?i)northern"],
        dir_patterns=[r"(?i)gel", r"(?i)blot", r"(?i)western"],
        disambiguate=_is_gel_image,
        description="Gel and blot images",
    ),

    # --- Figures and plots ---
    Category(
        name="figures",
        label="Figures & Plots",
        extensions=[".svg", ".eps", ".ai", ".png", ".jpg", ".jpeg"],
        name_patterns=[r"(?i)fig(ure)?[_\s.-]?\d", r"(?i)panel[_\s]?[a-z]",
                       r"(?i)(plot|graph|chart|heatmap|volcano|pca|umap)"],
        disambiguate=_is_figure_image,
        description="Publication figures and data plots",
    ),

    # --- Documents ---
    Category(
        name="papers_pdf",
        label="Papers & PDFs",
        extensions=[".pdf"],
        disambiguate=_is_paper_pdf,
        description="Papers, publications, and PDF documents",
    ),
    Category(
        name="presentations",
        label="Presentations",
        extensions=[".pptx", ".ppt", ".key", ".odp"],
        dir_patterns=[r"(?i)slide", r"(?i)presentation", r"(?i)talk", r"(?i)poster"],
        description="Slide decks and presentations",
    ),
    Category(
        name="documents",
        label="Documents",
        extensions=[".docx", ".doc", ".odt", ".rtf", ".pages", ".md", ".tex", ".rst"],
        description="Word processing documents and markup",
    ),
    Category(
        name="spreadsheets",
        label="Spreadsheets & Data",
        extensions=[".xlsx", ".xls", ".ods", ".numbers",
                    ".csv", ".tsv", ".dat", ".tab"],
        name_patterns=[r"(?i)(counts?|results?|summary|metadata|manifest|sample)"],
        description="Spreadsheets and tabular data files",
    ),
    Category(
        name="r_data",
        label="R Data",
        extensions=[".rds", ".rda", ".rdata", ".robj"],
        description="R serialized data objects",
    ),
    Category(
        name="pickle_data",
        label="Python Pickles",
        extensions=[".pkl", ".pickle", ".joblib"],
        description="Python serialized objects",
    ),
    Category(
        name="json_yaml",
        label="JSON/YAML/Config",
        extensions=[".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"],
        description="Configuration and structured data",
    ),

    # --- Code / Scripts ---
    Category(
        name="scripts",
        label="Scripts & Code",
        extensions=[".py", ".r", ".rmd", ".qmd", ".jl", ".m",
                    ".sh", ".bash", ".zsh", ".pl", ".rb"],
        description="Source code and scripts",
    ),
    Category(
        name="notebooks",
        label="Notebooks",
        extensions=[".ipynb", ".rmd", ".qmd"],
        description="Computational notebooks",
    ),
    Category(
        name="snakemake_nextflow",
        label="Workflow Pipelines",
        extensions=[".smk", ".nf"],
        name_patterns=[r"(?i)^snakefile$", r"(?i)^nextflow\.config$"],
        description="Workflow manager files (Snakemake, Nextflow)",
    ),

    # --- Sanger / Chromatograms ---
    Category(
        name="sanger",
        label="Sanger Sequencing",
        extensions=[".ab1", ".abi", ".scf", ".seq"],
        group_with_extensions=[".ab1", ".seq"],
        description="Sanger sequencing chromatograms and sequences",
    ),

    # --- Flow cytometry ---
    Category(
        name="flow_cytometry",
        label="Flow Cytometry",
        extensions=[".fcs", ".wsp", ".lmd"],
        description="Flow cytometry data",
    ),

    # --- Archives / Installers ---
    Category(
        name="archives",
        label="Archives",
        compound_extensions=[".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"],
        extensions=[".zip", ".7z", ".rar", ".gz", ".bz2", ".xz", ".zst",
                    ".dmg", ".iso", ".pkg"],
        description="Compressed archives and disk images",
    ),

    # --- Images (general) ---
    Category(
        name="images",
        label="Images",
        extensions=[".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic"],
        description="General image files",
    ),

    # --- NIfTI / Neuroimaging ---
    Category(
        name="neuroimaging",
        label="Neuroimaging",
        compound_extensions=[".nii.gz"],
        extensions=[".nii", ".nii.gz", ".mgz", ".mgh"],
        description="Neuroimaging data (NIfTI, FreeSurfer)",
    ),

    # --- Media ---
    Category(
        name="media",
        label="Audio/Video",
        extensions=[".mp4", ".m4a", ".mp3", ".wav", ".avi", ".mov", ".mkv",
                    ".flac", ".ogg", ".aac", ".webm"],
        description="Audio and video files",
    ),

    # --- Web / HTML reports ---
    Category(
        name="web_reports",
        label="HTML Reports",
        extensions=[".html", ".htm"],
        description="HTML report and result pages",
    ),

    # --- Logs ---
    Category(
        name="logs",
        label="Log Files",
        extensions=[".log", ".err", ".out"],
        name_patterns=[r"(?i)(slurm|nextflow|snakemake)[_-]\d+"],
        description="Execution logs and error output",
    ),
]

# Build lookup tables
_EXT_MAP: dict[str, list[Category]] = {}
_COMPOUND_EXT_MAP: dict[str, Category] = {}

for _cat in CATEGORIES:
    for _cext in _cat.compound_extensions:
        _COMPOUND_EXT_MAP[_cext] = _cat
    for _ext in _cat.extensions:
        _EXT_MAP.setdefault(_ext, []).append(_cat)


def _get_siblings(filepath: Path) -> list[str]:
    """List sibling filenames in the same directory."""
    try:
        return [f.name for f in filepath.parent.iterdir() if f != filepath]
    except (OSError, PermissionError):
        return []


def classify(filepath: Path, siblings: list[str] | None = None) -> str:
    """Classify a file and return the category name.

    Args:
        filepath: Path to the file.
        siblings: Optional pre-fetched list of sibling filenames.

    Returns:
        Category name string (e.g. 'fastq', 'bam_cram', 'other').
    """
    name = filepath.name
    ext = get_compound_extension(name)

    if siblings is None:
        siblings = _get_siblings(filepath)

    magic = b""  # Lazy — only read if needed

    # 1. Compound extension match (unambiguous)
    if ext in _COMPOUND_EXT_MAP:
        return _COMPOUND_EXT_MAP[ext].name

    # 2. Extension match — may need disambiguation
    candidates = _EXT_MAP.get(ext, [])

    if len(candidates) == 1:
        cat = candidates[0]
        if cat.disambiguate:
            if not magic:
                magic = read_magic_bytes(filepath)
            if cat.disambiguate(filepath, magic, siblings):
                return cat.name
            # Disambiguation failed — fall through
        else:
            return cat.name

    if len(candidates) > 1:
        # Multiple categories claim this extension — disambiguate
        if not magic:
            magic = read_magic_bytes(filepath)
        for cat in candidates:
            if cat.disambiguate and cat.disambiguate(filepath, magic, siblings):
                return cat.name
        # No disambiguator matched — return first without a disambiguator (generic),
        # or first candidate as last resort
        for cat in candidates:
            if not cat.disambiguate:
                return cat.name
        return candidates[0].name

    # 3. Name/path pattern match
    for cat in CATEGORIES:
        for pat in cat.name_patterns:
            if re.search(pat, name):
                return cat.name
        for pat in cat.dir_patterns:
            if re.search(pat, filepath.parent.name):
                return cat.name

    return "other"


def get_category(name: str) -> Category | None:
    """Look up a Category by name."""
    for cat in CATEGORIES:
        if cat.name == name:
            return cat
    return None


def all_category_names() -> list[str]:
    """Return all recognized category names."""
    return [c.name for c in CATEGORIES]
