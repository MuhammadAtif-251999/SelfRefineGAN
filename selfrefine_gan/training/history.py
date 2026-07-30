"""CSV training-history writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def append_history(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
