"""Env doctor - VRAG-001.

Prints PASS/FAIL for every dependency the pipeline needs and exits non-zero if any
required one fails. Offline by design: it inspects what is installed and what
credentials are present, and never calls a paid API. The only network call is to the
local Ollama daemon.

    make doctor
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.env import fingerprint, load_env

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

MIN_PYTHON = (3, 10)
OLLAMA_TIMEOUT_S = 3.0

# Credentials the MVP needs, with the prefix each provider issues. A wrong-shaped key
# fails at the first real call, hours later - cheaper to catch here.
CREDENTIALS = (
    ("HF_TOKEN", "hf_", "Hugging Face - model + dataset access"),
    ("GROQ_API_KEY", "gsk_", "Groq free tier - hosted whisper arm"),
    ("NVIDIA_API_KEY", "nvapi-", "NVIDIA NIM free tier - hosted fallback"),
)

OPTIONAL_CREDENTIALS = (
    ("LANGFUSE_PUBLIC_KEY", "pk-", "tracing, not needed for the MVP gate"),
    ("LANGFUSE_SECRET_KEY", "sk-", "tracing, not needed for the MVP gate"),
)

# Substring match - Ollama tags carry a registry prefix and a quant suffix.
EXPECTED_OLLAMA_MODELS = (
    ("nomic-embed-text", "embeddings for VRAG-015; HF repo id nomic-ai/nomic-embed-text-v1.5"),
)


@dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str


def _run(cmd: list[str]) -> str | None:
    """First line of a command's output, or None if it cannot be run."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else ""


def check_python() -> Check:
    got = sys.version_info[:3]
    want = ".".join(str(n) for n in MIN_PYTHON)
    ok = got[: len(MIN_PYTHON)] >= MIN_PYTHON
    version = ".".join(str(n) for n in got)
    return Check(
        "runtime",
        "python",
        PASS if ok else FAIL,
        f"{version} (need >={want})" if ok else f"{version} - too old, need >={want}",
    )


def check_binary(name: str, version_args: list[str], why: str) -> Check:
    path = shutil.which(name)
    if not path:
        return Check("binaries", name, FAIL, f"not on PATH - {why}")
    line = _run([path, *version_args])
    return Check("binaries", name, PASS, _short_version(name, line) or path)


# Banners are not uniform: "uv 0.12.5 (hash ... target)", "ollama version is 0.32.14",
# "ffmpeg version 6.1.1-build". Matching the number beats matching the word "version".
VERSION_RE = re.compile(r"\d+\.\d+(?:\.\d+)?[\w.+-]*")


def _short_version(name: str, line: str | None) -> str:
    """Pull the version number out of a --version banner, else return nothing."""
    if not line:
        return ""
    match = VERSION_RE.search(line)
    return match.group(0)[:40] if match else ""


def _ollama_host(env: dict[str, tuple[str, str]]) -> str:
    host = env.get("OLLAMA_HOST", ("", ""))[0] or "127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def check_ollama_daemon(env: dict[str, tuple[str, str]]) -> tuple[Check, list[str]]:
    """Reachability of the local daemon, plus the model tags it reports."""
    host = _ollama_host(env)
    url = f"{host}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=OLLAMA_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Check("services", "ollama daemon", FAIL, f"{host} unreachable ({exc}) - run `ollama serve`"), []
    tags = [m.get("name", "") for m in payload.get("models", [])]
    return Check("services", "ollama daemon", PASS, f"{host} - {len(tags)} model(s)"), tags


def check_ollama_models(tags: list[str]) -> list[Check]:
    checks = []
    for needle, why in EXPECTED_OLLAMA_MODELS:
        hit = next((t for t in tags if needle in t), None)
        checks.append(
            Check("services", f"model {needle}", PASS, hit)
            if hit
            else Check("services", f"model {needle}", WARN, f"not pulled - {why}")
        )
    return checks


def check_credential(env, key: str, prefix: str, why: str, required: bool) -> Check:
    section = "credentials" if required else "credentials (optional)"
    missing = FAIL if required else WARN
    found = env.get(key)
    if not found or not found[0]:
        return Check(section, key, missing, f"not set - {why}")
    value, source = found
    detail = f"{source:<28} {fingerprint(value)}"
    if not value.startswith(prefix):
        return Check(section, key, WARN, f"{detail}  (expected prefix {prefix!r})")
    return Check(section, key, PASS, detail)


def collect() -> list[Check]:
    env = load_env()
    checks = [check_python()]
    checks.append(check_binary("uv", ["--version"], "the env manager; see https://astral.sh/uv"))
    checks.append(check_binary("ffmpeg", ["-version"], "audio extraction (VRAG-005)"))
    checks.append(check_binary("ffprobe", ["-version"], "media metadata (VRAG-005)"))
    checks.append(check_binary("ollama", ["--version"], "local model arm; see https://ollama.com"))

    daemon, tags = check_ollama_daemon(env)
    checks.append(daemon)
    checks.extend(check_ollama_models(tags))

    for key, prefix, why in CREDENTIALS:
        checks.append(check_credential(env, key, prefix, why, required=True))
    for key, prefix, why in OPTIONAL_CREDENTIALS:
        checks.append(check_credential(env, key, prefix, why, required=False))
    return checks


def report(checks: list[Check], out=sys.stdout) -> int:
    """Print the table. Returns the process exit code: non-zero if anything FAILed."""
    print("VRAG env doctor", file=out)
    for section in dict.fromkeys(c.section for c in checks):
        print(f"\n{section}", file=out)
        for c in (c for c in checks if c.section == section):
            print(f"  {c.status:<4}  {c.name:<22} {c.detail}", file=out)

    tally = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL)}
    print(
        f"\n{tally[PASS]} PASS  {tally[WARN]} WARN  {tally[FAIL]} FAIL",
        file=out,
    )
    if tally[FAIL]:
        print(f"FAIL - {tally[FAIL]} required check(s) failed. Fix the FAIL lines above.", file=out)
        return 1
    print("PASS - environment is ready." + (" WARNs are non-blocking." if tally[WARN] else ""), file=out)
    return 0


def main() -> int:
    return report(collect())


if __name__ == "__main__":
    raise SystemExit(main())
