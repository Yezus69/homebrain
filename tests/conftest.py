from __future__ import annotations

import sys
from pathlib import Path
import inspect

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

if "newline" not in inspect.signature(Path.write_text).parameters:
    _original_write_text = Path.write_text

    def _write_text_with_newline(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if newline is None:
            return _original_write_text(self, data, encoding=encoding, errors=errors)
        with self.open("w", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.write(data)

    Path.write_text = _write_text_with_newline  # type: ignore[method-assign]
