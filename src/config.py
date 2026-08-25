"""Project configuration — the levers, kept out of the code.

`config.toml` holds every knob that changes what a run costs or how good it is. This
module is the only thing that reads it. Two rules it enforces so that "read from config"
means something:

* **no defaults for a lever.** A missing `ingest.frames.fps` raises, it does not fall back
  to 0.2. A silent default is a hardcoded value wearing a hat, and the acceptance criterion
  for VRAG-005 is precisely that the sampling rate is not hardcoded.
* **the file that produced a run is recorded with it.** `Config.fingerprint()` returns the
  path and a sha256 of the bytes, which ingest writes into every media.json. Six weeks from
  now, "why does this run have 4× the frames" is answerable.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("config.toml")


class ConfigError(Exception):
    """config.toml is missing, unparseable, or missing a lever the caller needs."""


@dataclass(frozen=True)
class Config:
    path: Path
    raw: bytes
    data: dict[str, Any]

    def get(self, dotted: str) -> Any:
        """Fetch `a.b.c`. Raises rather than defaulting — see the module docstring."""
        node: Any = self.data
        walked: list[str] = []
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                where = ".".join(walked) or "(root)"
                raise ConfigError(
                    f"{self.path}: no {dotted!r} — {where} has no key {key!r}. "
                    f"This is a lever, so there is no default; add it to {self.path}."
                )
            node = node[key]
            walked.append(key)
        return node

    def section(self, dotted: str) -> dict[str, Any]:
        node = self.get(dotted)
        if not isinstance(node, dict):
            raise ConfigError(f"{self.path}: {dotted!r} is {type(node).__name__}, not a table")
        return dict(node)

    def fingerprint(self) -> dict[str, str]:
        """Printable identity of the exact config bytes behind a run."""
        return {
            "path": str(self.path).replace("\\", "/"),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }


def load(path: Path | str = DEFAULT_PATH) -> Config:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    return Config(path=path, raw=raw, data=data)
