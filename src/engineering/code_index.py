"""AST-based repo-level code indexer."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class CodeSymbol:
    name: str
    kind: str  # "function", "class", "method", "variable"
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None
    parent: str | None  # class name for methods


@dataclass
class FileIndex:
    path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    line_count: int = 0


class RepoIndexer:
    """Index a repository's code structure using AST parsing."""

    SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx"}

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.files: dict[str, FileIndex] = {}
        self.symbols: list[CodeSymbol] = []

    def index(self) -> None:
        """Walk the repo and index all supported files."""
        for file_path in self._walk_files():
            if file_path.suffix == ".py":
                self._index_python(file_path)
        logger.info(f"Indexed {len(self.files)} files, {len(self.symbols)} symbols")

    def search_symbols(self, query: str) -> list[CodeSymbol]:
        """Search for symbols matching a query."""
        query_lower = query.lower()
        return [s for s in self.symbols if query_lower in s.name.lower()]

    def get_file_context(self, file_path: str, line_start: int, line_end: int) -> str:
        """Get code context around a specific range."""
        full_path = self.repo_path / file_path
        if not full_path.exists():
            return ""
        lines = full_path.read_text().splitlines()
        start = max(0, line_start - 10)
        end = min(len(lines), line_end + 10)
        return "\n".join(lines[start:end])

    def _walk_files(self):
        """Walk the repo and yield supported files."""
        skip_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv", "data"}
        for path in self.repo_path.rglob("*"):
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.is_file() and path.suffix in self.SUPPORTED_EXTENSIONS:
                yield path

    def _index_python(self, file_path: Path):
        """Parse a Python file and extract symbols."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            rel_path = str(file_path.relative_to(self.repo_path))
            symbols = []
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(CodeSymbol(
                        name=node.name, kind="function",
                        file_path=rel_path, line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node), parent=None,
                    ))
                elif isinstance(node, ast.ClassDef):
                    symbols.append(CodeSymbol(
                        name=node.name, kind="class",
                        file_path=rel_path, line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node), parent=None,
                    ))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.dump(node))
            self.files[rel_path] = FileIndex(
                path=rel_path, language="python",
                symbols=symbols, imports=imports,
                line_count=len(content.splitlines()),
            )
            self.symbols.extend(symbols)
        except SyntaxError:
            pass
        except Exception as e:
            logger.debug(f"Failed to index {file_path}: {e}")
