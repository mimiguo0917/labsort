"""Click CLI: scan, status, search, info, tree, reindex, organize, clean, undo, history."""

from __future__ import annotations

import json as json_mod
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from . import __version__
from .classifier import all_category_names, get_category
from .config import load_config
from .file_groups import detect_all_groups
from .indexer import Index
from .scanner import scan_directory
from .searcher import (
    find_by_groups,
    find_duplicates,
    find_large_files,
    find_near_duplicates,
    find_orphaned_indices,
    find_recent_files,
    fts_search,
)
from .organizer import (
    SequencingGroup,
    add_reviewed_to_plan,
    execute_plan,
    find_sequencing_candidates,
    plan_clean_duplicates,
    plan_clean_orphans,
    plan_organize_by_category,
    plan_organize_by_project,
    plan_organize_smart,
    undo_operation,
)
from .utils import format_size

console = Console()
err_console = Console(stderr=True)


def _json_output(data):
    """Print JSON to stdout."""
    click.echo(json_mod.dumps(data, indent=2, default=str))


@click.group()
@click.version_option(__version__, prog_name="labsort")
def cli():
    """labsort — scan, classify, index, and search your lab files."""
    pass


# -------------------------------------------------------------------------
# scan
# -------------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--thorough", is_flag=True, help="Hash files above the size limit too.")
@click.option("--dry-run", is_flag=True, help="Count files without full metadata collection.")
@click.option("--incremental/--full", default=True, help="Skip files unchanged since last scan.")
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON.")
def scan(directory: str, thorough: bool, dry_run: bool, incremental: bool, as_json: bool):
    """Scan a directory tree and index all files."""
    root = str(Path(directory).resolve())

    with Index() as idx:
        # Get last-indexed times for incremental scan
        last_indexed = idx.get_last_indexed(root) if incremental else {}

        result = scan_directory(
            root,
            thorough=thorough,
            dry_run=dry_run,
            last_indexed=last_indexed,
            show_progress=not as_json,
        )

        if not dry_run:
            # Index files
            idx.upsert_files(result.files)

            # Detect and save groups
            file_dicts = [
                {
                    "filepath": f.filepath,
                    "filename": f.filename,
                    "extension": f.extension,
                    "parent_dir": f.parent_dir,
                    "category": f.category,
                }
                for f in result.files
            ]
            groups = detect_all_groups(file_dicts)
            idx.save_groups(groups, scan_root=root)

            # Clean up deleted files
            removed = idx.remove_missing_files()

            # Log the scan
            idx.log_scan(result)

        if as_json:
            _json_output({
                "root": result.root,
                "files_scanned": len(result.files),
                "total_size": result.total_size,
                "total_size_human": format_size(result.total_size),
                "skipped_unchanged": result.skipped_unchanged,
                "skipped_ignored": result.skipped_ignored,
                "errors": result.errors,
                "groups_detected": len(groups) if not dry_run else 0,
                "removed_missing": removed if not dry_run else 0,
                "dry_run": dry_run,
            })
        else:
            console.print()
            panel_lines = [
                f"[bold]Root:[/] {result.root}",
                f"[bold]Files scanned:[/] {len(result.files)}",
                f"[bold]Total size:[/] {format_size(result.total_size)}",
            ]
            if result.skipped_unchanged:
                panel_lines.append(f"[dim]Skipped (unchanged):[/] {result.skipped_unchanged}")
            if result.skipped_ignored:
                panel_lines.append(f"[dim]Skipped (ignored):[/] {result.skipped_ignored}")
            if not dry_run:
                panel_lines.append(f"[bold]Groups detected:[/] {len(groups)}")
                if removed:
                    panel_lines.append(f"[yellow]Removed missing:[/] {removed}")
            if result.errors:
                panel_lines.append(f"[red]Errors:[/] {len(result.errors)}")
            if dry_run:
                panel_lines.append("[dim italic]Dry run — nothing was indexed.[/]")

            console.print(Panel("\n".join(panel_lines), title="Scan Complete", border_style="green"))


# -------------------------------------------------------------------------
# status
# -------------------------------------------------------------------------

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(as_json: bool):
    """Show index status: category breakdown, total size, last scan."""
    with Index() as idx:
        total = idx.total_files()
        size = idx.total_size()
        cats = idx.get_category_counts()
        last = idx.last_scan()
        roots = idx.get_scan_roots()

        if as_json:
            _json_output({
                "total_files": total,
                "total_size": size,
                "total_size_human": format_size(size),
                "categories": [
                    {"category": c, "count": n, "size": s, "size_human": format_size(s)}
                    for c, n, s in cats
                ],
                "scan_roots": roots,
                "last_scan": dict(last) if last else None,
            })
            return

        if total == 0:
            console.print("[dim]No files indexed yet. Run [bold]labsort scan <directory>[/bold] first.[/]")
            return

        console.print(Panel(
            f"[bold]{total:,}[/] files, [bold]{format_size(size)}[/] indexed",
            title="labsort index",
            border_style="blue",
        ))

        table = Table(title="Categories", show_lines=False)
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Size", justify="right")
        for cat_name, count, cat_size in cats:
            cat_obj = get_category(cat_name)
            label = cat_obj.label if cat_obj else cat_name
            table.add_row(label, str(count), format_size(cat_size))
        console.print(table)

        if last:
            console.print(f"\n[dim]Last scan:[/] {last['root']} at {last['finished_at']}")
        if roots:
            console.print(f"[dim]Scan roots:[/] {', '.join(roots)}")


# -------------------------------------------------------------------------
# search
# -------------------------------------------------------------------------

@cli.command()
@click.argument("query", required=False)
@click.option("--category", "-c", help="Filter by category name.")
@click.option("--extension", "-e", help="Filter by extension (e.g. '.fastq.gz').")
@click.option("--directory", "-d", help="Limit to files under this directory.")
@click.option("--duplicates", is_flag=True, help="Find exact duplicates (same SHA256).")
@click.option("--near-duplicates", is_flag=True, help="Find near-duplicate filenames.")
@click.option("--large", type=float, default=None, help="Find files larger than N MB.")
@click.option("--recent", type=int, default=None, help="Find files modified within N days.")
@click.option("--orphaned", is_flag=True, help="Find orphaned index files.")
@click.option("--groups", is_flag=True, help="Show file groups.")
@click.option("--group-type", help="Filter groups by type (index_pair, same_stem, etc.).")
@click.option("--limit", "-n", default=50, help="Max results.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def search(
    query: str | None,
    category: str | None,
    extension: str | None,
    directory: str | None,
    duplicates: bool,
    near_duplicates: bool,
    large: float | None,
    recent: int | None,
    orphaned: bool,
    groups: bool,
    group_type: str | None,
    limit: int,
    as_json: bool,
):
    """Search the file index. Provide a query for full-text search, or use flags."""
    with Index() as idx:
        if idx.total_files() == 0:
            err_console.print("[dim]No files indexed yet. Run [bold]labsort scan <directory>[/bold] first.[/]")
            return

        # Special modes
        if duplicates:
            results = find_duplicates(idx, directory=directory)
            if as_json:
                _json_output(results)
            else:
                _print_duplicate_groups(results)
            return

        if near_duplicates:
            results = find_near_duplicates(idx, directory=directory, limit=limit)
            if as_json:
                _json_output(results)
            else:
                _print_near_duplicate_groups(results)
            return

        if large is not None:
            results = find_large_files(idx, min_size_mb=large, directory=directory, limit=limit)
            if as_json:
                _json_output(results)
            else:
                _print_file_table(results, title=f"Files > {large} MB")
            return

        if recent is not None:
            results = find_recent_files(idx, days=recent, directory=directory, limit=limit)
            if as_json:
                _json_output(results)
            else:
                _print_file_table(results, title=f"Modified in last {recent} days")
            return

        if orphaned:
            results = find_orphaned_indices(idx)
            if as_json:
                _json_output(results)
            else:
                _print_file_table(results, title="Orphaned Index Files")
            return

        if groups or group_type:
            results = find_by_groups(idx, group_type=group_type)
            if as_json:
                _json_output(results)
            else:
                _print_groups(results)
            return

        # Category/extension-only filter (no FTS query)
        if not query and (category or extension or directory):
            filters = {}
            if category:
                filters["category"] = category
            if extension:
                filters["extension"] = extension
            if directory:
                filters["parent_dir"] = str(Path(directory).resolve())
            filters["limit"] = limit
            results = idx.query_files(**filters)
            if as_json:
                _json_output(results)
            else:
                title = "Filtered results"
                if category:
                    title += f" (category={category})"
                if extension:
                    title += f" (ext={extension})"
                _print_file_table(results, title=title)
            return

        # Full-text search
        if not query:
            err_console.print("[red]Provide a search query or use a flag (--duplicates, --large, etc.).[/]")
            return

        results = fts_search(
            idx, query,
            category=category,
            extension=extension,
            directory=str(Path(directory).resolve()) if directory else None,
            limit=limit,
        )
        if as_json:
            _json_output(results)
        else:
            _print_file_table(results, title=f'Search: "{query}"')


# -------------------------------------------------------------------------
# info
# -------------------------------------------------------------------------

@cli.command()
@click.argument("filepath", type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def info(filepath: str, as_json: bool):
    """Show detailed information about an indexed file."""
    resolved = str(Path(filepath).resolve())
    with Index() as idx:
        f = idx.get_file(resolved)
        if not f:
            err_console.print(f"[red]File not in index:[/] {resolved}")
            err_console.print("[dim]Run 'labsort scan' on the parent directory first.[/]")
            return

        if as_json:
            _json_output(f)
            return

        cat_obj = get_category(f["category"])
        cat_label = cat_obj.label if cat_obj else f["category"]

        lines = [
            f"[bold]Path:[/]      {f['filepath']}",
            f"[bold]Name:[/]      {f['filename']}",
            f"[bold]Category:[/]  {cat_label} [dim]({f['category']})[/]",
            f"[bold]Extension:[/] {f['extension']}",
            f"[bold]Size:[/]      {format_size(f['size'])}",
            f"[bold]Modified:[/]  {f['modified']}",
            f"[bold]Created:[/]   {f['created']}",
        ]
        if f["sha256"]:
            lines.append(f"[bold]SHA256:[/]    {f['sha256'][:16]}...")
        if f["is_symlink"]:
            lines.append("[bold]Symlink:[/]   Yes")
        if f["sidecar_of"]:
            lines.append(f"[bold]Sidecar of:[/] {f['sidecar_of']}")
        if f["preview"]:
            lines.append(f"\n[bold]Preview:[/]\n[dim]{f['preview'][:300]}[/]")

        lines.append(f"\n[dim]Indexed at: {f['indexed_at']}[/]")

        console.print(Panel("\n".join(lines), title="File Info", border_style="cyan"))


# -------------------------------------------------------------------------
# tree
# -------------------------------------------------------------------------

@cli.command()
@click.argument("directory", required=False, type=click.Path(exists=True, file_okay=False))
@click.option("--depth", "-d", default=3, help="Max tree depth.")
@click.option("--category", "-c", help="Only show files of this category.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def tree(directory: str | None, depth: int, category: str | None, as_json: bool):
    """Show a tree view of indexed files by directory."""
    with Index() as idx:
        if idx.total_files() == 0:
            err_console.print("[dim]No files indexed. Run 'labsort scan' first.[/]")
            return

        filters = {"limit": 5000}
        if directory:
            filters["parent_dir"] = str(Path(directory).resolve())
        if category:
            filters["category"] = category

        files = idx.query_files(**filters)

        if as_json:
            _json_output(_build_tree_dict(files, depth))
            return

        if not files:
            console.print("[dim]No matching files found.[/]")
            return

        # Build a Rich tree
        root_label = directory or "Index"
        rich_tree = Tree(f"[bold]{root_label}[/]")
        _build_rich_tree(rich_tree, files, depth)
        console.print(rich_tree)


# -------------------------------------------------------------------------
# reindex
# -------------------------------------------------------------------------

@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def reindex(as_json: bool):
    """Rebuild the FTS5 search index."""
    with Index() as idx:
        idx.rebuild_fts()
        total = idx.total_files()
        if as_json:
            _json_output({"status": "ok", "total_files": total})
        else:
            console.print(f"[green]FTS5 index rebuilt.[/] {total:,} files indexed.")


# -------------------------------------------------------------------------
# organize (Phase 2)
# -------------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--by", "organize_by", type=click.Choice(["smart", "category", "project"]),
              default="smart", help="Strategy: smart (auto-detect + token clustering), project (stem groups), category (file type).")
@click.option("--into", "target_dir", type=click.Path(), default=None,
              help="Target directory for organized files. Defaults to same directory.")
@click.option("--review", is_flag=True,
              help="Interactively review sequencing data groups — mark analyzed data for deletion.")
@click.option("--dry-run", is_flag=True, help="Preview what would happen without moving files.")
@click.option("--no-groups", is_flag=True, help="Don't keep file groups together (category mode).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def organize(directory: str, organize_by: str, target_dir: str | None,
             review: bool, dry_run: bool, no_groups: bool, as_json: bool):
    """Organize files into subfolders.

    Default mode (--by smart): Auto-detects the best strategy. Groups related
    files by stem first, then sub-clusters remaining files by name similarity
    within their category. Best of both worlds.

    Project mode (--by project): Groups files by shared stem into project folders.
    Singletons go into 'Ungrouped/<Category>/'.

    Category mode (--by category): Sorts files purely by file type.

    Use --review to interactively mark sequencing data as "already analyzed"
    before organizing. Confirmed groups go to Save to Delete/Data Already Analyzed/.

    Use --dry-run to preview the plan first.
    """
    source = str(Path(directory).resolve())
    target = Path(target_dir).resolve() if target_dir else Path(source)

    with Index() as idx:
        if idx.total_files() == 0:
            err_console.print("[dim]No files indexed. Run 'labsort scan' first.[/]")
            return

        # --- Interactive sequencing review ---
        approved_groups: list[SequencingGroup] = []
        if review:
            approved_groups = _interactive_sequencing_review(idx, source)

        if organize_by == "smart":
            plan = plan_organize_smart(idx, source, target_dir=target_dir)
        elif organize_by == "project":
            plan = plan_organize_by_project(idx, source, target_dir=target_dir)
        else:
            plan = plan_organize_by_category(
                idx, source,
                target_dir=target_dir,
                keep_groups=not no_groups,
            )

        # Add reviewed sequencing groups to the plan
        if approved_groups:
            # Remove any moves for files that the user marked as analyzed
            approved_paths = {f["filepath"] for g in approved_groups for f in g.files}
            plan.moves = [m for m in plan.moves if m.from_path not in approved_paths]
            # Recalculate total size after removing
            plan.total_size = sum(m.size for m in plan.moves)
            # Add the "save to delete" moves
            add_reviewed_to_plan(plan, approved_groups, target)

        if not plan.moves:
            console.print("[dim]Nothing to organize — files are already sorted or no files found.[/]")
            if plan.errors:
                for e in plan.errors:
                    err_console.print(f"[red]{e}[/]")
            return

        if as_json and dry_run:
            _json_output({
                "dry_run": True,
                "moves": [{"from": m.from_path, "to": m.to_path,
                           "size": m.size, "category": m.category}
                          for m in plan.moves],
                "total_files": len(plan.moves),
                "total_size": plan.total_size,
                "total_size_human": format_size(plan.total_size),
                "skipped": len(plan.skipped),
            })
            return

        if dry_run:
            _print_organize_plan(plan)
            return

        # Execute for real
        op_id = execute_plan(idx, plan)

        if as_json:
            _json_output({
                "operation_id": op_id,
                "moved": len(plan.moves),
                "total_size": plan.total_size,
                "total_size_human": format_size(plan.total_size),
                "errors": plan.errors,
            })
        else:
            console.print(Panel(
                f"[bold]Moved:[/] {len(plan.moves)} files ({format_size(plan.total_size)})\n"
                f"[bold]Operation ID:[/] {op_id}\n"
                f"[dim]Use 'labsort undo {op_id}' to reverse this operation.[/]"
                + (f"\n[red]Errors:[/] {len(plan.errors)}" if plan.errors else ""),
                title="Organize Complete",
                border_style="green",
            ))
            if plan.errors:
                for e in plan.errors[:5]:
                    err_console.print(f"  [red]{e}[/]")
                if len(plan.errors) > 5:
                    err_console.print(f"  [dim]... and {len(plan.errors) - 5} more[/]")


# -------------------------------------------------------------------------
# clean (Phase 2)
# -------------------------------------------------------------------------

@cli.command()
@click.option("--duplicates", is_flag=True, help="Move duplicate files to trash (keeps newest).")
@click.option("--orphaned", is_flag=True, help="Move orphaned index files to trash.")
@click.option("--directory", "-d", help="Limit to files under this directory.")
@click.option("--trash-dir", type=click.Path(), default=None,
              help="Trash folder (default: ~/.labsort/trash/).")
@click.option("--keep", type=click.Choice(["newest", "oldest", "shortest_path"]),
              default="newest", help="Which duplicate to keep.")
@click.option("--dry-run", is_flag=True, help="Preview what would be trashed.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def clean(duplicates: bool, orphaned: bool, directory: str | None,
          trash_dir: str | None, keep: str, dry_run: bool, as_json: bool):
    """Move duplicates or orphaned files to a trash staging folder.

    Files are NOT permanently deleted — they're moved to ~/.labsort/trash/
    (or a custom --trash-dir). Use 'labsort undo' to restore them.

    You must specify at least one of --duplicates or --orphaned.
    """
    if not duplicates and not orphaned:
        err_console.print("[red]Specify --duplicates and/or --orphaned.[/]")
        return

    with Index() as idx:
        if idx.total_files() == 0:
            err_console.print("[dim]No files indexed. Run 'labsort scan' first.[/]")
            return

        plans = []

        if duplicates:
            dup_plan = plan_clean_duplicates(
                idx, directory=directory, trash_dir=trash_dir, keep=keep,
            )
            if dup_plan.moves:
                plans.append(dup_plan)

        if orphaned:
            orph_plan = plan_clean_orphans(idx, trash_dir=trash_dir)
            if orph_plan.moves:
                plans.append(orph_plan)

        if not plans:
            console.print("[dim]Nothing to clean.[/]")
            return

        for plan in plans:
            if dry_run:
                if as_json:
                    _json_output({
                        "dry_run": True,
                        "type": plan.op_type,
                        "description": plan.description,
                        "moves": [{"from": m.from_path, "to": m.to_path,
                                   "size": m.size} for m in plan.moves],
                        "total_files": len(plan.moves),
                        "total_size": plan.total_size,
                        "total_size_human": format_size(plan.total_size),
                    })
                else:
                    _print_clean_plan(plan)
            else:
                op_id = execute_plan(idx, plan)
                if as_json:
                    _json_output({
                        "operation_id": op_id,
                        "moved_to_trash": len(plan.moves),
                        "total_size": plan.total_size,
                        "total_size_human": format_size(plan.total_size),
                        "trash_dir": plan.target_root,
                        "errors": plan.errors,
                    })
                else:
                    console.print(Panel(
                        f"[bold]{plan.description}[/]\n"
                        f"[bold]Moved to trash:[/] {len(plan.moves)} files "
                        f"({format_size(plan.total_size)})\n"
                        f"[bold]Trash folder:[/] {plan.target_root}\n"
                        f"[bold]Operation ID:[/] {op_id}\n"
                        f"[dim]Use 'labsort undo {op_id}' to restore files.[/]"
                        + (f"\n[red]Errors:[/] {len(plan.errors)}" if plan.errors else ""),
                        title="Clean Complete",
                        border_style="yellow",
                    ))


# -------------------------------------------------------------------------
# undo (Phase 2)
# -------------------------------------------------------------------------

@cli.command()
@click.argument("operation_id", type=int, required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def undo(operation_id: int | None, as_json: bool):
    """Undo a previous organize or clean operation.

    Without an ID, undoes the most recent completed operation.
    With an ID, undoes that specific operation.
    """
    with Index() as idx:
        if operation_id is None:
            # Find the most recent completed operation
            ops = idx.get_operations(limit=10)
            completed = [o for o in ops if o["status"] == "completed"]
            if not completed:
                err_console.print("[dim]No completed operations to undo.[/]")
                return
            operation_id = completed[0]["id"]

        op = idx.get_operation(operation_id)
        if not op:
            err_console.print(f"[red]Operation {operation_id} not found.[/]")
            return

        if op["status"] == "undone":
            err_console.print(f"[dim]Operation {operation_id} was already undone.[/]")
            return

        restored, errors = undo_operation(idx, operation_id)

        if as_json:
            _json_output({
                "operation_id": operation_id,
                "restored": restored,
                "errors": errors,
            })
        else:
            console.print(Panel(
                f"[bold]Operation:[/] {op['op_type']} (#{operation_id})\n"
                f"[bold]Restored:[/] {restored} files\n"
                + (f"[red]Errors:[/] {len(errors)}" if errors else "[green]All files restored.[/]"),
                title="Undo Complete",
                border_style="green",
            ))
            for e in errors[:5]:
                err_console.print(f"  [red]{e}[/]")


# -------------------------------------------------------------------------
# history (Phase 2)
# -------------------------------------------------------------------------

@cli.command()
@click.option("--limit", "-n", default=20, help="Max operations to show.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def history(limit: int, as_json: bool):
    """Show history of organize and clean operations."""
    with Index() as idx:
        ops = idx.get_operations(limit=limit)

        if not ops:
            console.print("[dim]No operations yet.[/]")
            return

        if as_json:
            _json_output(ops)
            return

        table = Table(title="Operation History", show_lines=False)
        table.add_column("ID", style="bold", justify="right")
        table.add_column("Type", style="cyan")
        table.add_column("Status")
        table.add_column("Files", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Date", style="dim")
        table.add_column("Description", max_width=40)

        for op in ops:
            status_style = {
                "completed": "[green]completed[/]",
                "undone": "[yellow]undone[/]",
                "pending": "[dim]pending[/]",
                "partial": "[red]partial[/]",
            }.get(op["status"], op["status"])

            table.add_row(
                str(op["id"]),
                op["op_type"],
                status_style,
                str(op["file_count"]),
                format_size(op["total_size"]),
                (op["completed_at"] or op["created_at"])[:16],
                op["description"][:40],
            )
        console.print(table)


# -------------------------------------------------------------------------
# Helpers for Rich output
# -------------------------------------------------------------------------

def _interactive_sequencing_review(idx, source: str) -> list[SequencingGroup]:
    """Walk the user through sequencing data groups and ask which are analyzed."""
    from .classifier import get_category as _get_cat
    from .organizer import SERVICE_KEYWORDS

    candidates = find_sequencing_candidates(idx, source)
    if not candidates:
        console.print("[dim]No sequencing data groups found to review.[/]")
        return []

    total_size = sum(g.total_size for g in candidates)
    total_files = sum(len(g.files) for g in candidates)

    console.print(Panel(
        f"Found [bold]{len(candidates)}[/] sequencing result groups "
        f"({total_files} files, {format_size(total_size)}).\n"
        f"[dim]Reviewing: Sanger traces, contigs, alignments, raw reads.\n"
        f"Skipped: Cloning sequences (.dna/.gb) — you still need those.[/]\n\n"
        f"For each group, answer whether the data has been analyzed.\n"
        f"Analyzed data → [cyan]Save to Delete/Data Already Analyzed/[/]\n\n"
        f"[dim]y = already analyzed (ok to delete)  "
        f"n = still need it  "
        f"s = skip remaining  "
        f"q = quit review[/]",
        title="Sequencing Data Review",
        border_style="magenta",
    ))

    approved: list[SequencingGroup] = []

    for i, group in enumerate(candidates, 1):
        cat_obj = _get_cat(group.category)
        cat_label = cat_obj.label if cat_obj else group.category
        ext_str = ", ".join(group.extensions[:5])
        age = group.oldest_modified[:10] if group.oldest_modified else "unknown"

        # Check if group looks like sequencing service results
        name_lower = group.name.lower()
        all_tokens = set()
        for f in group.files:
            all_tokens.add(f.get("parent_dir", "").lower())
            all_tokens.add(f.get("filename", "").lower())
        all_text = " ".join(all_tokens)
        is_service = any(kw in all_text or kw in name_lower for kw in SERVICE_KEYWORDS)

        # Show sample filenames
        sample_names = [f["filename"] for f in group.files[:4]]
        sample_str = ", ".join(sample_names)
        if len(group.files) > 4:
            sample_str += f", ... +{len(group.files) - 4} more"

        console.print(f"\n[bold]({i}/{len(candidates)})[/] [cyan]{group.name}[/]"
                      + (" [magenta](sequencing service)[/]" if is_service else ""))
        console.print(
            f"  [bold]Type:[/] {cat_label}  "
            f"[bold]Files:[/] {len(group.files)}  "
            f"[bold]Size:[/] {format_size(group.total_size)}  "
            f"[bold]Extensions:[/] {ext_str}"
        )
        console.print(f"  [bold]Date range:[/] {age} → {group.newest_modified[:10] if group.newest_modified else '?'}")
        console.print(f"  [dim]{sample_str}[/]")

        while True:
            answer = click.prompt(
                "  Already analyzed?",
                type=click.Choice(["y", "n", "s", "q"]),
                default="n",
                show_choices=True,
            )
            if answer == "y":
                approved.append(group)
                console.print(f"  [yellow]→ Marked for deletion[/]")
                break
            elif answer == "n":
                console.print(f"  [green]→ Keeping[/]")
                break
            elif answer == "s":
                console.print(f"\n[dim]Skipping remaining groups.[/]")
                return approved
            elif answer == "q":
                console.print(f"\n[dim]Review cancelled.[/]")
                return []

    if approved:
        total = sum(g.total_size for g in approved)
        total_files = sum(len(g.files) for g in approved)
        console.print(Panel(
            f"[bold]{len(approved)} groups[/] marked as analyzed "
            f"({total_files} files, {format_size(total)})\n"
            f"These will be moved to [cyan]Save to Delete/Data Already Analyzed/[/]",
            title="Review Summary",
            border_style="yellow",
        ))
    else:
        console.print("\n[dim]No groups marked for deletion.[/]")

    return approved


def _print_file_table(files: list[dict], title: str = "Results"):
    if not files:
        console.print("[dim]No results.[/]")
        return

    table = Table(title=title, show_lines=False)
    table.add_column("Name", style="white", max_width=40)
    table.add_column("Category", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")
    table.add_column("Path", style="dim", max_width=50)

    for f in files:
        cat_obj = get_category(f["category"])
        cat_label = cat_obj.label if cat_obj else f["category"]
        table.add_row(
            f["filename"],
            cat_label,
            format_size(f["size"]),
            f["modified"][:10] if f.get("modified") else "",
            f["parent_dir"],
        )
    console.print(table)
    console.print(f"[dim]{len(files)} result(s)[/]")


def _print_duplicate_groups(groups: list[dict]):
    if not groups:
        console.print("[dim]No duplicates found.[/]")
        return

    total_waste = 0
    for g in groups:
        table = Table(title=f"SHA256: {g['sha256'][:16]}...", show_lines=False, border_style="yellow")
        table.add_column("File", style="white")
        table.add_column("Size", justify="right")
        table.add_column("Path", style="dim")
        for f in g["files"]:
            table.add_row(f["filename"], format_size(f["size"]), f["parent_dir"])
        console.print(table)
        waste = g["size"] * (g["count"] - 1)
        total_waste += waste
        console.print(f"  [dim]Wasted space: {format_size(waste)}[/]\n")

    console.print(f"[bold]{len(groups)} duplicate group(s)[/], {format_size(total_waste)} total wasted space.")


def _print_near_duplicate_groups(groups: list[dict]):
    if not groups:
        console.print("[dim]No near-duplicates found.[/]")
        return
    for g in groups:
        console.print(f"\n[bold]{g['base_name']}[/] ({g['count']} versions) in {g['directory']}")
        for f in g["files"]:
            console.print(f"  {f['filename']}  [dim]{format_size(f['size'])}  {f['modified'][:10]}[/]")

    console.print(f"\n[bold]{len(groups)} near-duplicate group(s)[/]")


def _print_groups(groups: list[dict]):
    if not groups:
        console.print("[dim]No file groups found.[/]")
        return
    for g in groups:
        console.print(f"\n[cyan]{g['group_type']}[/] — {g['description']}")
        for member in g["members"]:
            console.print(f"  {member}")

    console.print(f"\n[bold]{len(groups)} group(s)[/]")


def _build_rich_tree(parent: Tree, files: list[dict], max_depth: int, current_depth: int = 0):
    """Recursively build a Rich tree from file records."""
    if current_depth >= max_depth:
        return

    # Group files by immediate subdirectory
    by_dir: dict[str, list[dict]] = {}
    for f in files:
        parts = Path(f["parent_dir"]).parts
        # Determine the key at current depth
        by_dir.setdefault(f["parent_dir"], []).append(f)

    # Deduplicate directories and sort
    dirs = sorted(by_dir.keys())
    for d in dirs:
        dir_files = by_dir[d]
        dir_name = Path(d).name or d
        branch = parent.add(f"[bold blue]{dir_name}/[/] [dim]({len(dir_files)} files)[/]")
        for f in dir_files[:10]:  # limit files per dir
            cat_obj = get_category(f["category"])
            cat_label = cat_obj.label if cat_obj else f["category"]
            branch.add(f"{f['filename']}  [cyan]{cat_label}[/]  [dim]{format_size(f['size'])}[/]")
        if len(dir_files) > 10:
            branch.add(f"[dim]... and {len(dir_files) - 10} more[/]")


def _print_organize_plan(plan):
    """Print a dry-run preview of an organize plan with folder structure."""
    from collections import Counter, defaultdict

    console.print(Panel(
        f"[bold]Plan:[/] {plan.description}\n"
        f"[bold]Files to move:[/] {len(plan.moves)}\n"
        f"[bold]Total size:[/] {format_size(plan.total_size)}\n"
        f"[bold]Already organized:[/] {len(plan.skipped)}\n"
        f"[dim italic]Dry run — no files were moved.[/]",
        title="Organize Preview",
        border_style="blue",
    ))

    # Build 3-level folder structure for display
    target_root = Path(plan.target_root) if plan.target_root else None

    # Nested dict: level1 → level2 → level3 → count
    folder_tree: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for m in plan.moves:
        dst = Path(m.to_path)
        if target_root:
            try:
                rel = dst.relative_to(target_root)
                parts = rel.parts
                l1 = parts[0] if len(parts) > 0 else "root"
                l2 = parts[1] if len(parts) > 2 else "(files)"
                l3 = parts[2] if len(parts) > 3 else "(files)"
                folder_tree[l1][l2][l3] += 1
            except ValueError:
                folder_tree["other"]["(files)"]["(files)"] += 1
        else:
            folder_tree["other"]["(files)"]["(files)"] += 1

    # Print folder tree
    tree = Tree(f"[bold]{plan.target_root or plan.source_root}[/]")
    for l1 in sorted(folder_tree.keys()):
        l2_map = folder_tree[l1]
        l1_total = sum(sum(l3.values()) for l3 in l2_map.values())
        branch1 = tree.add(f"[bold cyan]{l1}/[/] [dim]({l1_total} files)[/]")

        for l2, l3_map in sorted(l2_map.items(), key=lambda x: -sum(x[1].values())):
            l2_total = sum(l3_map.values())
            if l2 == "(files)":
                branch1.add(f"[dim]{l2_total} file(s) directly[/]")
                continue

            branch2 = branch1.add(f"{l2}/ [dim]({l2_total} files)[/]")

            # Show up to 8 sub-folders, collapse rest
            items = sorted(l3_map.items(), key=lambda x: -x[1])
            shown = 0
            for l3, count in items:
                if l3 == "(files)":
                    if len(items) > 1:
                        branch2.add(f"[dim]{count} file(s) directly[/]")
                    continue
                if shown >= 8:
                    remaining = sum(c for n, c in items[shown+1:] if n != "(files)")
                    if remaining > 0:
                        branch2.add(f"[dim]... +{len(items) - shown - 1} more folders ({remaining} files)[/]")
                    break
                branch2.add(f"{l3}/ [dim]({count})[/]")
                shown += 1

    console.print(tree)

    # Show first few moves as examples
    if plan.moves:
        console.print("\n[bold]Sample moves:[/]")
        for m in plan.moves[:10]:
            console.print(f"  {Path(m.from_path).name} [dim]→[/] {m.to_path}")
        if len(plan.moves) > 10:
            console.print(f"  [dim]... and {len(plan.moves) - 10} more[/]")


def _print_clean_plan(plan):
    """Print a dry-run preview of a clean plan."""
    console.print(Panel(
        f"[bold]Plan:[/] {plan.description}\n"
        f"[bold]Files to trash:[/] {len(plan.moves)}\n"
        f"[bold]Space to reclaim:[/] {format_size(plan.total_size)}\n"
        f"[dim italic]Dry run — no files were moved.[/]",
        title="Clean Preview",
        border_style="yellow",
    ))

    if plan.moves:
        console.print("\n[bold]Files to be trashed:[/]")
        for m in plan.moves[:15]:
            console.print(
                f"  {Path(m.from_path).name}  "
                f"[dim]{format_size(m.size)}  →  {m.to_path}[/]"
            )
        if len(plan.moves) > 15:
            console.print(f"  [dim]... and {len(plan.moves) - 15} more[/]")


def _build_tree_dict(files: list[dict], max_depth: int) -> dict:
    """Build a JSON-serializable tree structure."""
    tree: dict = {"directories": {}}
    for f in files:
        d = f["parent_dir"]
        tree["directories"].setdefault(d, []).append({
            "filename": f["filename"],
            "category": f["category"],
            "size": f["size"],
        })
    return tree
