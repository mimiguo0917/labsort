"""Detect related files: same-stem pairs, numbered series, index pairs."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import get_category
from .utils import get_compound_extension, get_stem, strip_copy_suffixes


@dataclass
class FileGroup:
    """A group of related files."""
    group_type: str  # 'same_stem', 'index_pair', 'numbered_series', 'companions'
    primary: str     # filepath of the "main" file
    members: list[str] = field(default_factory=list)  # all filepaths in group
    description: str = ""


# Known index-pair relationships: {data_ext: [index_exts]}
INDEX_PAIRS = {
    ".bam": [".bai", ".bam.bai"],
    ".cram": [".crai"],
    ".vcf.gz": [".tbi", ".csi"],
    ".vcf.bgz": [".tbi", ".csi"],
    ".fa": [".fai", ".dict"],
    ".fasta": [".fai", ".dict"],
    ".bed.gz": [".tbi"],
}


def detect_index_pairs(files: list[dict]) -> list[FileGroup]:
    """Find data+index file pairs (e.g., .bam + .bai).

    Args:
        files: List of dicts with at least 'filepath', 'filename', 'extension', 'parent_dir'.
    """
    groups = []
    # Build lookup: (parent_dir, stem_lower) -> list of files
    by_dir_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in files:
        stem = get_stem(f["filename"]).lower()
        by_dir_stem[(f["parent_dir"], stem)].append(f)

    seen = set()
    for (pdir, stem), cluster in by_dir_stem.items():
        if len(cluster) < 2:
            continue
        ext_map = {f["extension"]: f for f in cluster}
        for data_ext, idx_exts in INDEX_PAIRS.items():
            if data_ext in ext_map:
                for idx_ext in idx_exts:
                    if idx_ext in ext_map:
                        data_fp = ext_map[data_ext]["filepath"]
                        idx_fp = ext_map[idx_ext]["filepath"]
                        key = tuple(sorted([data_fp, idx_fp]))
                        if key not in seen:
                            seen.add(key)
                            groups.append(FileGroup(
                                group_type="index_pair",
                                primary=data_fp,
                                members=[data_fp, idx_fp],
                                description=f"{data_ext} + {idx_ext}",
                            ))
    return groups


def detect_same_stem_pairs(files: list[dict]) -> list[FileGroup]:
    """Find files with the same stem but different extensions in the same directory.

    E.g., report.pptx + report.pdf, sample.ab1 + sample.seq
    Case-insensitive: SMUCR074.dna and smucr074.gb will be grouped.
    """
    groups = []
    by_dir_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in files:
        stem = get_stem(f["filename"]).lower()
        by_dir_stem[(f["parent_dir"], stem)].append(f)

    for (pdir, stem), cluster in by_dir_stem.items():
        if len(cluster) < 2:
            continue
        # Skip if these are already index pairs
        exts = {f["extension"] for f in cluster}
        fps = [f["filepath"] for f in cluster]
        groups.append(FileGroup(
            group_type="same_stem",
            primary=fps[0],
            members=fps,
            description=f"Same stem '{stem}': {', '.join(sorted(exts))}",
        ))
    return groups


def detect_numbered_series(files: list[dict]) -> list[FileGroup]:
    """Find files that are copies/versions of each other.

    E.g., report.pdf, report (1).pdf, report_v2.pdf
    """
    groups = []
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        by_dir[f["parent_dir"]].append(f)

    for pdir, dir_files in by_dir.items():
        # Group by base name after stripping copy suffixes (case-insensitive)
        by_base: dict[str, list[dict]] = defaultdict(list)
        for f in dir_files:
            base = strip_copy_suffixes(f["filename"]).lower()
            by_base[base].append(f)

        for base, cluster in by_base.items():
            if len(cluster) < 2:
                continue
            fps = [f["filepath"] for f in cluster]
            names = [f["filename"] for f in cluster]
            groups.append(FileGroup(
                group_type="numbered_series",
                primary=fps[0],
                members=fps,
                description=f"Versions of '{base}': {', '.join(names)}",
            ))
    return groups


def detect_companion_files(files: list[dict]) -> list[FileGroup]:
    """Find companion files declared by category rules (group_with_extensions)."""
    groups = []
    by_dir_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in files:
        stem = get_stem(f["filename"]).lower()
        by_dir_stem[(f["parent_dir"], stem)].append(f)

    seen = set()
    for (pdir, stem), cluster in by_dir_stem.items():
        if len(cluster) < 2:
            continue
        for f in cluster:
            cat = get_category(f.get("category", ""))
            if cat and cat.group_with_extensions:
                companion_exts = set(cat.group_with_extensions)
                companions = [
                    g for g in cluster
                    if g["extension"] in companion_exts and g["filepath"] != f["filepath"]
                ]
                if companions:
                    fps = [f["filepath"]] + [c["filepath"] for c in companions]
                    key = tuple(sorted(fps))
                    if key not in seen:
                        seen.add(key)
                        groups.append(FileGroup(
                            group_type="companions",
                            primary=f["filepath"],
                            members=fps,
                            description=f"{cat.label} + companions",
                        ))
    return groups


def detect_all_groups(files: list[dict]) -> list[FileGroup]:
    """Run all group detection algorithms and return deduplicated groups.

    Args:
        files: List of file dicts with keys: filepath, filename, extension,
               parent_dir, category.
    """
    all_groups = []
    all_groups.extend(detect_index_pairs(files))
    all_groups.extend(detect_companion_files(files))
    all_groups.extend(detect_same_stem_pairs(files))
    all_groups.extend(detect_numbered_series(files))

    # Deduplicate: if two groups have the same member set, keep the more specific one
    priority = {"index_pair": 0, "companions": 1, "same_stem": 2, "numbered_series": 3}
    seen_members: dict[tuple, FileGroup] = {}
    for g in all_groups:
        key = tuple(sorted(g.members))
        existing = seen_members.get(key)
        if existing is None or priority.get(g.group_type, 9) < priority.get(existing.group_type, 9):
            seen_members[key] = g

    return list(seen_members.values())
