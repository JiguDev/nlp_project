"""I/O helpers for JSONL, text, and directory management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path if needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding=encoding)


def write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write a UTF-8 text file, creating parent directories if needed."""
    file_path = ensure_parent_dir(path)
    file_path.write_text(content, encoding=encoding)


def read_jsonl(path: str | Path) -> List[dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: List[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to JSONL format."""
    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records one by one."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def list_files(path: str | Path, suffixes: Sequence[str] | None = None) -> List[Path]:
    """Return a sorted list of files under a directory."""
    root = Path(path)
    files = [candidate for candidate in root.rglob("*") if candidate.is_file()]
    if suffixes is not None:
        files = [candidate for candidate in files if candidate.suffix.lower() in {suffix.lower() for suffix in suffixes}]
    return sorted(files)
