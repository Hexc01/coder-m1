"""Index a target repository for code context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from src.engineering.code_index import RepoIndexer

console = Console()


@click.command()
@click.argument("repo_path")
def main(repo_path: str):
    """Index a repository and print summary."""
    console.print(f"[bold]Indexing:[/] {repo_path}")

    indexer = RepoIndexer(repo_path)
    indexer.index()

    console.print(f"\n[green]Files indexed: {len(indexer.files)}[/]")
    console.print(f"[green]Symbols found: {len(indexer.symbols)}[/]")

    # Summary by kind
    kinds = {}
    for s in indexer.symbols:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    for kind, count in sorted(kinds.items()):
        console.print(f"  {kind}: {count}")

    # Top symbols
    console.print(f"\n[bold]Top symbols:[/]")
    for s in indexer.symbols[:20]:
        console.print(f"  {s.kind:10} {s.name:30} {s.file_path}:{s.line_start}")


if __name__ == "__main__":
    main()
