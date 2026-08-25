"""Read the project's env files.

Two quirks this has to survive:

* the shared `.env` is written in shell syntax (`export GROQ_API_KEY=...`), because it is
  also `source`-d by hand. A plain `KEY=VALUE` parser turns that into a key literally
  named "export GROQ_API_KEY".
* secrets are meant to live in `~/.config/`, not the repo (CLAUDE.md). Both locations are
  read, and every lookup reports *where* the value came from so the doctor can show it.

Never returns or logs a secret value — see `fingerprint`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Highest precedence last-word goes to the real process environment; files fill the gaps
# in the order listed.
ENV_FILES = (
    Path(".env"),
    Path.home() / ".config" / "ai-course-vrag.env",
    Path.home() / ".config" / "ai-course-board.env",
)


def parse_env_file(text: str) -> dict[str, str]:
    """Parse dotenv-ish text. Tolerates `export ` prefixes, comments, quoted values.

    Deliberately does not strip inline comments: `KEY=a#b` is the value `a#b`, because
    `#` is legal in a token and guessing wrong silently corrupts a credential.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def load_env(files: tuple[Path, ...] = ENV_FILES) -> dict[str, tuple[str, str]]:
    """Merge process env and env files into {key: (value, source)}.

    `source` is "environment" or the file path — printable, unlike the value.
    """
    merged: dict[str, tuple[str, str]] = {}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for key, value in parse_env_file(text).items():
            merged.setdefault(key, (value, str(path)))
    for key, value in os.environ.items():
        merged[key] = (value, "environment")
    return merged


def fingerprint(value: str) -> str:
    """A printable, non-reversible stand-in for a secret.

    Length plus a short digest: enough to confirm two machines hold the *same* key
    without ever putting the key on a terminal.
    """
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"len={len(value)} sha256:{digest}"
