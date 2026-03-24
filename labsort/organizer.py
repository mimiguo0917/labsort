"""File organization engine: move, organize by category, clean duplicates.

All operations are logged and reversible via undo.
Safe-delete moves files to a trash staging folder instead of permanent deletion.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import get_category
from .config import CONFIG_DIR, load_config
from .indexer import Index
from .searcher import find_duplicates, find_orphaned_indices
from .utils import format_size, get_stem, iso_timestamp

TRASH_DIR = CONFIG_DIR / "trash"

# Map categories → broad top-level super-categories.
# Sequencing is folded into Data (as a sub-level).
SUPER_CATEGORIES: dict[str, str] = {
    # Data (includes all sequencing & analysis)
    "fastq": "Data",
    "bam_cram": "Data",
    "vcf": "Data",
    "bed_annotation": "Data",
    "reference_genome": "Data",
    "sequences": "Data",
    "cloning_sequence": "Data",
    "sequence_protein": "Data",
    "counts_matrix": "Data",
    "genomics_index": "Data",
    "genomics_h5": "Data",
    "usearch_output": "Data",
    "sanger": "Data",
    "flow_cytometry": "Data",
    "spreadsheets": "Data",
    "r_data": "Data",
    "pickle_data": "Data",
    "numpy_data": "Data",
    "parquet_data": "Data",
    "json_yaml": "Data",
    "tabular_data": "Data",
    # Documents
    "papers_pdf": "Documents",
    "presentations": "Documents",
    "documents": "Documents",
    # Images
    "microscopy_proprietary": "Images",
    "microscopy_tif": "Images",
    "microscopy_h5": "Images",
    "gel_images": "Images",
    "figures": "Images",
    "images": "Images",
    "neuroimaging": "Images",
    # Code
    "scripts": "Code",
    "notebooks": "Code",
    "snakemake_nextflow": "Code",
    # Other
    "archives": "Other",
    "media": "Other",
    "web_reports": "Other",
    "logs": "Other",
}

# Sequencing-related categories get nested under Data/Sequencing/ instead of Data/
SEQUENCING_CATEGORIES = {
    "fastq", "bam_cram", "vcf", "bed_annotation", "reference_genome",
    "sequences", "cloning_sequence", "sequence_protein", "counts_matrix",
    "genomics_index", "genomics_h5", "usearch_output", "sanger",
}

# Keywords that indicate a stem-group cluster is sequencing-related
SEQUENCING_KEYWORDS = {
    "contig", "trace", "alignment", "reads", "sample",
    "sanger", "sequencing", "fastq", "barcode",
}

# Categories with results from sequencing services (worth asking "already analyzed?")
# These are typically downloaded once, analyzed, and the results are what matter.
REVIEWABLE_SEQ_CATEGORIES = {
    "sanger",       # .ab1, .seq trace files from Quintara/Genewiz/Azenta
    "sequences",    # .fasta, .gbk contigs & alignments (derived results)
    "fastq",        # raw reads (Illumina, nanopore)
}

# Categories to SKIP during review (user actively needs these)
# cloning_sequence: .dna, .gb plasmid maps — actively used for cloning work
# reference_genome, genomics_index: reference files, not disposable

# Keywords from sequencing service providers (helps identify service results)
SERVICE_KEYWORDS = {
    "quintara", "genewiz", "azenta", "eurofins", "eton", "psomagen",
    "gsl",  # UC Berkeley Genomics Sequencing Lab
}

# Files that are likely safe to delete (installers, temp downloads, etc.)
DISPOSABLE_EXTENSIONS = {
    ".dmg", ".pkg", ".msi", ".exe", ".deb", ".rpm",   # installers
    ".crdownload", ".download", ".part", ".tmp",        # partial/temp downloads
    ".iso", ".img",                                     # disk images
}


@dataclass
class MoveAction:
    """A single planned file move."""
    from_path: str
    to_path: str
    size: int = 0
    category: str = ""


@dataclass
class SequencingGroup:
    """A group of related sequencing files for interactive review."""
    name: str
    files: list[dict] = field(default_factory=list)
    total_size: int = 0
    category: str = ""
    oldest_modified: str = ""
    newest_modified: str = ""
    extensions: list[str] = field(default_factory=list)


@dataclass
class OrganizePlan:
    """A plan for organizing files, to be previewed before execution."""
    op_type: str
    description: str
    moves: list[MoveAction] = field(default_factory=list)
    total_size: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_root: str = ""
    target_root: str = ""


def _safe_move(src: Path, dst: Path) -> None:
    """Move a file, creating parent directories as needed.

    Uses shutil.move which handles cross-device moves.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _resolve_conflict(dst: Path) -> Path:
    """If dst exists, append a numeric suffix to avoid overwriting."""
    if not dst.exists():
        return dst
    stem = dst.stem
    ext = dst.suffix
    parent = dst.parent
    counter = 1
    while True:
        new_name = f"{stem} ({counter}){ext}"
        candidate = parent / new_name
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Organize by category
# ---------------------------------------------------------------------------

def plan_organize_by_category(
    idx: Index,
    source_dir: str,
    *,
    target_dir: str | None = None,
    keep_groups: bool = True,
) -> OrganizePlan:
    """Plan organizing files from source_dir into category subfolders.

    Args:
        idx: Database index.
        source_dir: Directory whose files to organize.
        target_dir: Where to create category folders. Defaults to source_dir.
        keep_groups: If True, move file group members together with their primary.
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve() if target_dir else source

    plan = OrganizePlan(
        op_type="organize",
        description=f"Organize by category: {source} → {target}",
        source_root=str(source),
        target_root=str(target),
    )

    # Get all indexed files under source_dir
    files = idx.query_files(parent_dir=str(source), limit=100_000)
    if not files:
        plan.errors.append(f"No indexed files found under {source}")
        return plan

    # Build set of files already planned for move (to handle groups)
    planned_paths: set[str] = set()

    # Load groups for group-aware moving
    groups_by_file: dict[str, list[str]] = {}
    if keep_groups:
        groups = idx.get_groups(scan_root=str(source))
        for g in groups:
            members = json.loads(g["members"]) if isinstance(g["members"], str) else g["members"]
            for m in members:
                groups_by_file.setdefault(m, []).extend(
                    [x for x in members if x != m]
                )

    for f in files:
        fp = f["filepath"]
        if fp in planned_paths:
            continue

        cat_name = f["category"]
        cat_obj = get_category(cat_name)
        folder_name = cat_obj.label if cat_obj else cat_name

        src_path = Path(fp)
        # Preserve relative path under source_dir for nested structures
        try:
            rel = src_path.relative_to(source)
        except ValueError:
            plan.skipped.append(fp)
            continue

        # If file is already in the right category folder, skip
        if rel.parts and rel.parts[0] == folder_name:
            plan.skipped.append(fp)
            planned_paths.add(fp)
            continue

        dst_path = _resolve_conflict(target / folder_name / src_path.name)

        plan.moves.append(MoveAction(
            from_path=fp,
            to_path=str(dst_path),
            size=f["size"],
            category=cat_name,
        ))
        plan.total_size += f["size"]
        planned_paths.add(fp)

        # Move group members together
        if fp in groups_by_file:
            for member_path in groups_by_file[fp]:
                if member_path in planned_paths:
                    continue
                member_src = Path(member_path)
                if not member_src.exists():
                    continue
                member_dst = _resolve_conflict(target / folder_name / member_src.name)
                member_size = member_src.stat().st_size if member_src.exists() else 0
                plan.moves.append(MoveAction(
                    from_path=member_path,
                    to_path=str(member_dst),
                    size=member_size,
                    category=cat_name,
                ))
                plan.total_size += member_size
                planned_paths.add(member_path)

    return plan


# ---------------------------------------------------------------------------
# Organize by project (stem-based clustering)
# ---------------------------------------------------------------------------

def _cluster_by_stem(files: list[dict]) -> dict[str, list[dict]]:
    """Group files by their stem (case-insensitive).

    Returns {lowercase_stem: [file_dicts]}.
    Files with the same stem but different extensions are grouped together.
    """
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        stem = get_stem(f["filename"]).lower()
        by_stem[stem].append(f)
    return dict(by_stem)


def _pick_project_name(cluster: list[dict]) -> str:
    """Pick the best display name for a project cluster.

    Prefers the original casing of the shortest stem variant.
    """
    stems = [get_stem(f["filename"]) for f in cluster]
    # Pick the shortest (most "base") name, preserving original case
    stems.sort(key=len)
    return stems[0]


def plan_organize_by_project(
    idx: Index,
    source_dir: str,
    *,
    target_dir: str | None = None,
    min_group_size: int = 2,
) -> OrganizePlan:
    """Plan organizing files into project folders based on shared stems.

    Files with the same stem (case-insensitive) are grouped into a project folder.
    Singletons (files with no stem-match) are placed into category subfolders
    under an 'Ungrouped/' directory.

    Args:
        idx: Database index.
        source_dir: Directory whose files to organize.
        target_dir: Where to create project folders. Defaults to source_dir.
        min_group_size: Minimum files to form a project folder (default 2).
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve() if target_dir else source

    plan = OrganizePlan(
        op_type="organize",
        description=f"Organize by project: {source} → {target}",
        source_root=str(source),
        target_root=str(target),
    )

    # Get all indexed files under source_dir
    files = idx.query_files(parent_dir=str(source), limit=100_000)
    if not files:
        plan.errors.append(f"No indexed files found under {source}")
        return plan

    # Cluster files by stem
    clusters = _cluster_by_stem(files)

    planned_paths: set[str] = set()

    for stem_lower, cluster in clusters.items():
        if len(cluster) >= min_group_size:
            # Project folder — named after the shared stem
            folder_name = _pick_project_name(cluster)

            for f in cluster:
                fp = f["filepath"]
                if fp in planned_paths:
                    continue

                src_path = Path(fp)
                try:
                    rel = src_path.relative_to(source)
                except ValueError:
                    plan.skipped.append(fp)
                    continue

                # Skip if already in the right project folder
                if rel.parts and rel.parts[0] == folder_name:
                    plan.skipped.append(fp)
                    planned_paths.add(fp)
                    continue

                dst_path = _resolve_conflict(target / folder_name / src_path.name)

                plan.moves.append(MoveAction(
                    from_path=fp,
                    to_path=str(dst_path),
                    size=f["size"],
                    category=f.get("category", ""),
                ))
                plan.total_size += f["size"]
                planned_paths.add(fp)
        else:
            # Singleton — organize by category under Ungrouped/
            for f in cluster:
                fp = f["filepath"]
                if fp in planned_paths:
                    continue

                src_path = Path(fp)
                try:
                    rel = src_path.relative_to(source)
                except ValueError:
                    plan.skipped.append(fp)
                    continue

                cat_name = f.get("category", "other")
                cat_obj = get_category(cat_name)
                cat_label = cat_obj.label if cat_obj else cat_name

                # Skip if already in Ungrouped/Category/
                if rel.parts and rel.parts[0] == "Ungrouped":
                    plan.skipped.append(fp)
                    planned_paths.add(fp)
                    continue

                dst_path = _resolve_conflict(
                    target / "Ungrouped" / cat_label / src_path.name
                )

                plan.moves.append(MoveAction(
                    from_path=fp,
                    to_path=str(dst_path),
                    size=f["size"],
                    category=cat_name,
                ))
                plan.total_size += f["size"]
                planned_paths.add(fp)

    return plan


# ---------------------------------------------------------------------------
# Token-based filename clustering
# ---------------------------------------------------------------------------

def _tokenize_filename(filename: str) -> set[str]:
    """Extract meaningful tokens from a filename for similarity grouping."""
    stem = get_stem(filename)
    # Split camelCase: "R3VariantBreakdown" → "R3 Variant Breakdown"
    stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    parts = re.split(r'[_\-\s.()]+', stem)
    tokens = set()
    for p in parts:
        if re.match(r'^\d{4,8}$', p):   # skip dates/years (YYYY, YYYYMMDD)
            continue
        if re.match(r'^\d{1,3}$', p):    # skip short numbers
            continue
        if len(p) < 3:                   # skip tiny fragments
            continue
        tokens.add(p.lower())
    return tokens


def _find_token_clusters(
    files: list[dict], min_shared: int = 2
) -> tuple[list[tuple[str, list[dict]]], list[dict]]:
    """Cluster files by shared filename tokens using union-find.

    Returns (clusters, singletons) where clusters = [(label, [files])].
    """
    file_tokens = [(f, _tokenize_filename(f["filename"])) for f in files]
    n = len(file_tokens)

    if n < 2:
        return [], files

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build edges: connect files sharing enough tokens
    for i in range(n):
        for j in range(i + 1, n):
            shared = file_tokens[i][1] & file_tokens[j][1]
            if len(shared) >= min_shared:
                union(i, j)

    # Collect components
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    clusters = []
    singletons = []

    for indices in components.values():
        if len(indices) < 2:
            singletons.append(file_tokens[indices[0]][0])
            continue

        cluster_files = [file_tokens[i][0] for i in indices]

        # Find shared tokens across all files in cluster for label
        shared = file_tokens[indices[0]][1].copy()
        for i in indices[1:]:
            shared &= file_tokens[i][1]

        if shared:
            # Pick longest tokens first (most descriptive)
            label = " ".join(sorted(shared, key=lambda t: -len(t))[:3])
        else:
            # Fallback: most common tokens across cluster
            all_tokens: Counter[str] = Counter()
            for i in indices:
                all_tokens.update(file_tokens[i][1])
            common = [t for t, c in all_tokens.most_common(3) if c >= 2]
            label = " ".join(common) if common else "Related"

        clusters.append((label.title(), cluster_files))

    return clusters, singletons


# ---------------------------------------------------------------------------
# Merge related stem groups by token similarity
# ---------------------------------------------------------------------------

def _merge_stem_groups(
    stem_groups: dict[str, list[dict]],
) -> list[tuple[str, list[dict]]]:
    """Merge stem groups that share filename tokens into bigger clusters.

    Two-pass approach to avoid over-merging from transitive chains:
      Pass 1: Merge groups sharing ≥2 tokens (strict, prevents chaining).
      Pass 2: Absorb remaining small groups into existing large clusters
              if they share ≥1 token (no new chains between small groups).

    Returns [(label, [all files in merged group])].
    """
    group_names = list(stem_groups.keys())
    n = len(group_names)

    if n < 2:
        return [(name, files) for name, files in stem_groups.items()]

    group_tokens = [_tokenize_filename(name) for name in group_names]

    # --- Pass 1: strict merge (≥2 shared tokens) via Union-Find ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            shared = group_tokens[i] & group_tokens[j]
            if len(shared) >= 2:
                union(i, j)

    # Collect pass-1 components
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    # Build pass-1 clusters with labels and combined tokens
    clusters: list[tuple[str, list[dict], set[str]]] = []  # (label, files, all_tokens)
    small_groups: list[tuple[int, str, list[dict], set[str]]] = []  # leftover small groups

    for indices in components.values():
        all_files: list[dict] = []
        combined_tokens: set[str] = set()
        for i in indices:
            all_files.extend(stem_groups[group_names[i]])
            combined_tokens |= group_tokens[i]

        if len(indices) == 1:
            # Not merged — save for pass 2
            small_groups.append((indices[0], group_names[indices[0]], all_files, combined_tokens))
        else:
            # Merged cluster — compute label from shared tokens
            shared = group_tokens[indices[0]].copy()
            for i in indices[1:]:
                shared &= group_tokens[i]
            if shared:
                label = " ".join(sorted(shared, key=lambda t: -len(t))[:3])
            else:
                all_t: Counter[str] = Counter()
                for i in indices:
                    all_t.update(group_tokens[i])
                common = [t for t, c in all_t.most_common(3) if c >= 2]
                label = " ".join(common) if common else group_names[indices[0]]
            clusters.append((label.title(), all_files, combined_tokens))

    # --- Pass 2: absorb small groups into large clusters (≥1 shared token) ---
    still_small = []
    for idx_orig, name, files, tokens in small_groups:
        absorbed = False
        # Find best matching large cluster (most shared tokens)
        best_match = -1
        best_overlap = 0
        for ci, (clabel, cfiles, ctokens) in enumerate(clusters):
            overlap = len(tokens & ctokens)
            if overlap >= 1 and overlap > best_overlap:
                best_overlap = overlap
                best_match = ci
        if best_match >= 0:
            clabel, cfiles, ctokens = clusters[best_match]
            cfiles.extend(files)
            ctokens |= tokens
            clusters[best_match] = (clabel, cfiles, ctokens)
            absorbed = True
        if not absorbed:
            still_small.append((name, files))

    # Convert to output format
    result = [(label, files) for label, files, _ in clusters]
    result.extend(still_small)
    return result


# ---------------------------------------------------------------------------
# Helpers for smart organize
# ---------------------------------------------------------------------------

def _is_disposable(f: dict) -> bool:
    """Check if a file is likely safe to delete (installers, temp downloads, etc.)."""
    ext = f.get("extension", "").lower()
    return ext in DISPOSABLE_EXTENSIONS


def _category_path(cat_name: str) -> tuple[str, ...]:
    """Get the folder path components for a category.

    Sequencing categories get an extra nesting level under Data/Sequencing/.
    Other categories go directly under their super-category.

    Examples:
        "sanger"     → ("Data", "Sequencing", "Sanger Sequencing")
        "spreadsheets" → ("Data", "Spreadsheets & Data")
        "papers_pdf" → ("Documents", "Papers & PDFs")
    """
    cat_obj = get_category(cat_name)
    cat_label = cat_obj.label if cat_obj else cat_name
    super_cat = SUPER_CATEGORIES.get(cat_name, "Other")

    if cat_name in SEQUENCING_CATEGORIES:
        return (super_cat, "Sequencing", cat_label)
    return (super_cat, cat_label)


def _is_group_sequencing(group_files: list[dict], group_label: str = "") -> bool:
    """Check if a stem group is sequencing-related (by keywords or file categories)."""
    # Check label tokens
    label_tokens = {t.lower() for t in re.split(r'[\s_\-]+', group_label) if len(t) >= 3}
    if label_tokens & SEQUENCING_KEYWORDS:
        return True

    # Check filename tokens
    all_tokens: set[str] = set()
    for f in group_files:
        all_tokens |= _tokenize_filename(f["filename"])
    if all_tokens & SEQUENCING_KEYWORDS:
        return True

    # Majority of files from sequencing categories?
    seq_count = sum(1 for f in group_files
                    if f.get("category", "other") in SEQUENCING_CATEGORIES)
    return seq_count > len(group_files) / 2


def _group_super_category(group_files: list[dict]) -> str:
    """Determine the super-category for a group of files by majority vote."""
    cat_counts: Counter[str] = Counter()
    for f in group_files:
        cat = f.get("category", "other")
        super_cat = SUPER_CATEGORIES.get(cat, "Other")
        cat_counts[super_cat] += 1
    if cat_counts:
        return cat_counts.most_common(1)[0][0]
    return "Other"


def _group_path(group_files: list[dict], group_label: str) -> tuple[str, ...]:
    """Get the folder path for a merged stem group.

    Sequencing groups → ("Data", "Sequencing", label)
    Other groups → (super_cat, label)
    """
    super_cat = _group_super_category(group_files)

    if super_cat == "Data" and _is_group_sequencing(group_files, group_label):
        return ("Data", "Sequencing", group_label)

    return (super_cat, group_label)


# ---------------------------------------------------------------------------
# Smart organize (two-step: clean then organize)
# ---------------------------------------------------------------------------

def find_sequencing_candidates(
    idx: Index,
    source_dir: str,
    min_group_size: int = 3,
) -> list[SequencingGroup]:
    """Find groups of sequencing service results that might be already analyzed.

    Only reviews categories that are typically "download once, analyze, done":
    Sanger traces (.ab1/.seq), contigs/alignments (.fasta/.gbk), and raw reads (.fastq).

    Cloning sequences (.dna/.gb) are excluded — those are actively used.
    """
    source = Path(source_dir).resolve()
    files = idx.query_files(parent_dir=str(source), limit=100_000)

    # Only review sanger results, contigs/alignments, and raw reads
    # Skip cloning sequences, reference genomes, indexes (user needs those)
    seq_files = [f for f in files if f.get("category", "other") in REVIEWABLE_SEQ_CATEGORIES]
    if not seq_files:
        return []

    # Cluster by stem
    stem_clusters = _cluster_by_stem(seq_files)
    stem_groups: dict[str, list[dict]] = {}
    singletons: list[dict] = []

    for stem_lower, cluster in stem_clusters.items():
        if len(cluster) >= 2:
            folder_name = _pick_project_name(cluster)
            stem_groups[folder_name] = cluster
        else:
            singletons.extend(cluster)

    # Merge related stem groups
    merged = _merge_stem_groups(stem_groups)

    # Also token-cluster the singletons
    token_clusters, leftovers = _find_token_clusters(singletons)

    # Build SequencingGroup objects
    groups: list[SequencingGroup] = []

    for name, group_files in merged:
        if len(group_files) < min_group_size:
            continue
        mods = sorted(f.get("modified", "") for f in group_files if f.get("modified"))
        exts = sorted(set(f.get("extension", "") for f in group_files))
        groups.append(SequencingGroup(
            name=name,
            files=group_files,
            total_size=sum(f["size"] for f in group_files),
            category=Counter(f.get("category", "") for f in group_files).most_common(1)[0][0],
            oldest_modified=mods[0] if mods else "",
            newest_modified=mods[-1] if mods else "",
            extensions=exts,
        ))

    for label, cluster_files in token_clusters:
        if len(cluster_files) < min_group_size:
            continue
        mods = sorted(f.get("modified", "") for f in cluster_files if f.get("modified"))
        exts = sorted(set(f.get("extension", "") for f in cluster_files))
        groups.append(SequencingGroup(
            name=label,
            files=cluster_files,
            total_size=sum(f["size"] for f in cluster_files),
            category=Counter(f.get("category", "") for f in cluster_files).most_common(1)[0][0],
            oldest_modified=mods[0] if mods else "",
            newest_modified=mods[-1] if mods else "",
            extensions=exts,
        ))

    # Sort by size descending (biggest groups first — most space to reclaim)
    groups.sort(key=lambda g: g.total_size, reverse=True)
    return groups


def add_reviewed_to_plan(
    plan: OrganizePlan,
    approved_groups: list[SequencingGroup],
    target: Path,
) -> None:
    """Add user-approved sequencing groups to a plan as moves to Save to Delete/."""
    for group in approved_groups:
        for f in group.files:
            fp = f["filepath"]
            src_path = Path(fp)
            dst_path = _resolve_conflict(
                target / "Save to Delete" / "Data Already Analyzed" / group.name / src_path.name
            )
            plan.moves.append(MoveAction(
                from_path=fp, to_path=str(dst_path),
                size=f["size"], category=f.get("category", ""),
            ))
            plan.total_size += f["size"]


def plan_organize_smart(
    idx: Index,
    source_dir: str,
    *,
    target_dir: str | None = None,
) -> OrganizePlan:
    """Smart two-step organization.

    Step 1 — Clean:
        Disposable files (installers, temp downloads) → Save to Delete/
        Duplicate files (same SHA256) → Save to Delete/Duplicates/

    Step 2 — Organize (file type → project/topic):
        target/
        ├── Data/
        │   ├── Sequencing/           (extra level for seq categories)
        │   │   ├── Sanger/
        │   │   │   ├── Ligation/     (project/topic cluster)
        │   │   │   └── ...
        │   │   ├── Contigs/          (stem group cluster)
        │   │   └── FASTQ/
        │   ├── Spreadsheets/
        │   │   ├── AAV Analysis/     (topic cluster)
        │   │   └── ...
        │   └── Flow Cytometry/
        ├── Documents/
        │   ├── Papers/
        │   └── Presentations/
        ├── Images/
        │   ├── Microscopy/
        │   └── Gels/
        ├── Code/
        ├── Other/
        └── Save to Delete/
    """
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve() if target_dir else source

    plan = OrganizePlan(
        op_type="organize",
        description=f"Smart organize: {source} → {target}",
        source_root=str(source),
        target_root=str(target),
    )

    files = idx.query_files(parent_dir=str(source), limit=100_000)
    if not files:
        plan.errors.append(f"No indexed files found under {source}")
        return plan

    planned_paths: set[str] = set()

    # ===================================================================
    # STEP 1: CLEAN — move disposable & duplicate files to Save to Delete/
    # ===================================================================

    # 1a. Disposable files (installers, temp downloads)
    non_disposable: list[dict] = []
    for f in files:
        if _is_disposable(f):
            fp = f["filepath"]
            src_path = Path(fp)
            try:
                src_path.relative_to(source)
            except ValueError:
                plan.skipped.append(fp)
                continue

            ext = f.get("extension", "").lower()
            if ext in {".dmg", ".pkg", ".msi", ".exe", ".deb", ".rpm"}:
                sub = "Installers"
            elif ext in {".iso", ".img"}:
                sub = "Disk Images"
            else:
                sub = "Temp Downloads"

            dst_path = _resolve_conflict(
                target / "Save to Delete" / sub / src_path.name
            )
            plan.moves.append(MoveAction(
                from_path=fp, to_path=str(dst_path),
                size=f["size"], category=f.get("category", ""),
            ))
            plan.total_size += f["size"]
            planned_paths.add(fp)
        else:
            non_disposable.append(f)

    # 1b. Duplicate files (same SHA256 → keep newest, move rest)
    dup_groups = find_duplicates(idx, directory=str(source))
    for group in dup_groups:
        dup_files = group["files"]
        if len(dup_files) < 2:
            continue
        # Keep newest
        dup_files.sort(key=lambda f: f.get("modified", ""), reverse=True)
        for dup in dup_files[1:]:
            fp = dup["filepath"]
            if fp in planned_paths:
                continue
            src_path = Path(fp)
            if not src_path.exists():
                continue
            dst_path = _resolve_conflict(
                target / "Save to Delete" / "Duplicates" / src_path.name
            )
            plan.moves.append(MoveAction(
                from_path=fp, to_path=str(dst_path),
                size=dup["size"], category=dup.get("category", ""),
            ))
            plan.total_size += dup["size"]
            planned_paths.add(fp)

    # ===================================================================
    # STEP 2: ORGANIZE — file type → project/topic
    # ===================================================================

    # Group remaining files by category (file type)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for f in non_disposable:
        if f["filepath"] in planned_paths:
            continue
        cat = f.get("category", "other")
        by_category[cat].append(f)

    for cat_name, cat_files in by_category.items():
        # Determine the folder path for this file type
        cat_path_parts = _category_path(cat_name)  # e.g. ("Data", "Sequencing", "Sanger")
        cat_base = target.joinpath(*cat_path_parts)

        # Within this file type: find project/topic clusters
        # First try stem-based grouping
        stem_clusters = _cluster_by_stem(cat_files)
        stem_groups: dict[str, list[dict]] = {}
        remaining: list[dict] = []

        for stem_lower, cluster in stem_clusters.items():
            if len(cluster) >= 2:
                folder_name = _pick_project_name(cluster)
                stem_groups[folder_name] = cluster
            else:
                remaining.extend(cluster)

        # Merge related stem groups by token similarity
        merged = _merge_stem_groups(stem_groups)

        # Large merged groups → named project sub-folder under this file type
        MIN_PROJECT = 5
        for group_name, group_files in merged:
            if len(group_files) < MIN_PROJECT:
                remaining.extend(group_files)
                continue

            for f in group_files:
                fp = f["filepath"]
                if fp in planned_paths:
                    continue
                src_path = Path(fp)
                try:
                    src_path.relative_to(source)
                except ValueError:
                    plan.skipped.append(fp)
                    continue

                dst_path = _resolve_conflict(cat_base / group_name / src_path.name)
                plan.moves.append(MoveAction(
                    from_path=fp, to_path=str(dst_path),
                    size=f["size"], category=cat_name,
                ))
                plan.total_size += f["size"]
                planned_paths.add(fp)

        # Remaining files: try token sub-clustering for topic groups
        token_clusters, singletons = _find_token_clusters(remaining)

        for label, cluster_files in token_clusters:
            for f in cluster_files:
                fp = f["filepath"]
                if fp in planned_paths:
                    continue
                src_path = Path(fp)
                try:
                    src_path.relative_to(source)
                except ValueError:
                    plan.skipped.append(fp)
                    continue

                dst_path = _resolve_conflict(cat_base / label / src_path.name)
                plan.moves.append(MoveAction(
                    from_path=fp, to_path=str(dst_path),
                    size=f["size"], category=cat_name,
                ))
                plan.total_size += f["size"]
                planned_paths.add(fp)

        # True singletons go flat into file type folder
        for f in singletons:
            fp = f["filepath"]
            if fp in planned_paths:
                continue
            src_path = Path(fp)
            try:
                src_path.relative_to(source)
            except ValueError:
                plan.skipped.append(fp)
                continue

            dst_path = _resolve_conflict(cat_base / src_path.name)
            plan.moves.append(MoveAction(
                from_path=fp, to_path=str(dst_path),
                size=f["size"], category=cat_name,
            ))
            plan.total_size += f["size"]
            planned_paths.add(fp)

    return plan


# ---------------------------------------------------------------------------
# Clean duplicates
# ---------------------------------------------------------------------------

def plan_clean_duplicates(
    idx: Index,
    directory: str | None = None,
    *,
    trash_dir: str | None = None,
    keep: str = "newest",
) -> OrganizePlan:
    """Plan moving duplicate files to trash, keeping one copy.

    Args:
        idx: Database index.
        directory: Limit to files under this directory.
        trash_dir: Trash staging folder. Defaults to ~/.labsort/trash/.
        keep: Which copy to keep: 'newest' (default), 'oldest', 'shortest_path'.
    """
    trash = Path(trash_dir) if trash_dir else TRASH_DIR

    plan = OrganizePlan(
        op_type="clean",
        description=f"Clean duplicates → {trash}",
        source_root=directory or "",
        target_root=str(trash),
    )

    dup_groups = find_duplicates(idx, directory=directory)
    if not dup_groups:
        return plan

    for group in dup_groups:
        files = group["files"]
        if len(files) < 2:
            continue

        # Sort to determine which to keep
        if keep == "newest":
            files.sort(key=lambda f: f.get("modified", ""), reverse=True)
        elif keep == "oldest":
            files.sort(key=lambda f: f.get("modified", ""))
        elif keep == "shortest_path":
            files.sort(key=lambda f: len(f["filepath"]))

        # Keep the first, trash the rest
        for dup in files[1:]:
            fp = dup["filepath"]
            src = Path(fp)
            if not src.exists():
                continue
            # Preserve directory structure in trash
            dst = trash / "duplicates" / src.name
            dst = _resolve_conflict(dst)

            plan.moves.append(MoveAction(
                from_path=fp,
                to_path=str(dst),
                size=dup["size"],
                category=dup.get("category", ""),
            ))
            plan.total_size += dup["size"]

    return plan


# ---------------------------------------------------------------------------
# Clean orphaned index files
# ---------------------------------------------------------------------------

def plan_clean_orphans(
    idx: Index,
    *,
    trash_dir: str | None = None,
) -> OrganizePlan:
    """Plan moving orphaned index files (.bai without .bam, etc.) to trash."""
    trash = Path(trash_dir) if trash_dir else TRASH_DIR

    plan = OrganizePlan(
        op_type="clean",
        description=f"Clean orphaned indices → {trash}",
        target_root=str(trash),
    )

    orphans = find_orphaned_indices(idx)
    if not orphans:
        return plan

    for f in orphans:
        fp = f["filepath"]
        src = Path(fp)
        if not src.exists():
            continue
        dst = trash / "orphans" / src.name
        dst = _resolve_conflict(dst)

        plan.moves.append(MoveAction(
            from_path=fp,
            to_path=str(dst),
            size=f["size"],
            category=f.get("category", ""),
        ))
        plan.total_size += f["size"]

    return plan


# ---------------------------------------------------------------------------
# Execute a plan
# ---------------------------------------------------------------------------

def execute_plan(idx: Index, plan: OrganizePlan) -> int:
    """Execute an organize/clean plan. Returns the operation ID.

    All moves are logged in the database for undo support.
    """
    if not plan.moves:
        return -1

    op_id = idx.create_operation(
        op_type=plan.op_type,
        description=plan.description,
        source_root=plan.source_root,
        target_root=plan.target_root,
    )

    moved_count = 0
    moved_size = 0
    path_updates: list[tuple[str, str]] = []

    for action in plan.moves:
        src = Path(action.from_path)
        dst = Path(action.to_path)

        if not src.exists():
            plan.errors.append(f"Source missing: {src}")
            continue

        # Resolve any new conflicts (files may have changed since planning)
        dst = _resolve_conflict(dst)

        try:
            _safe_move(src, dst)
            idx.log_move(
                operation_id=op_id,
                from_path=action.from_path,
                to_path=str(dst),
                size=action.size,
                category=action.category,
            )
            path_updates.append((action.from_path, str(dst)))
            moved_count += 1
            moved_size += action.size
        except (OSError, shutil.Error) as e:
            plan.errors.append(f"Failed to move {src}: {e}")

    # Update index paths
    idx.batch_update_paths(path_updates)

    # Complete the operation
    idx.complete_operation(op_id, moved_count, moved_size)

    # Clean up empty directories left behind after moves
    if plan.source_root:
        _cleanup_empty_dirs(Path(plan.source_root))

    return op_id


# ---------------------------------------------------------------------------
# Undo an operation
# ---------------------------------------------------------------------------

def undo_operation(idx: Index, operation_id: int) -> tuple[int, list[str]]:
    """Undo an operation by reversing all its moves.

    Returns (count_restored, errors).
    """
    op = idx.get_operation(operation_id)
    if not op:
        return 0, [f"Operation {operation_id} not found"]
    if op["status"] == "undone":
        return 0, [f"Operation {operation_id} already undone"]
    if op["status"] not in ("completed", "partial"):
        return 0, [f"Operation {operation_id} has status '{op['status']}', cannot undo"]

    moves = idx.get_operation_moves(operation_id)
    if not moves:
        return 0, [f"No moves found for operation {operation_id}"]

    restored = 0
    errors: list[str] = []
    path_updates: list[tuple[str, str]] = []

    # Reverse moves in reverse order
    for move in reversed(moves):
        if move["undone"]:
            continue

        src = Path(move["to_path"])    # current location
        dst = Path(move["from_path"])  # original location

        if not src.exists():
            errors.append(f"Cannot restore {src}: file missing from trash")
            continue

        if dst.exists():
            errors.append(f"Cannot restore to {dst}: file already exists there")
            continue

        try:
            _safe_move(src, dst)
            path_updates.append((move["to_path"], move["from_path"]))
            restored += 1
        except (OSError, shutil.Error) as e:
            errors.append(f"Failed to restore {src} → {dst}: {e}")

    # Update index paths back to originals
    idx.batch_update_paths(path_updates)

    # Mark operation as undone
    idx.mark_operation_undone(operation_id)

    # Clean up empty directories left behind in trash
    if op.get("target_root"):
        _cleanup_empty_dirs(Path(op["target_root"]))

    return restored, errors


def _cleanup_empty_dirs(root: Path) -> None:
    """Remove empty directories under root (bottom-up)."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        dp = Path(dirpath)
        if dp == root:
            continue
        if not any(dp.iterdir()):
            try:
                dp.rmdir()
            except OSError:
                pass
