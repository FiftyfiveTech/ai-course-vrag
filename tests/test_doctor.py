"""Unit tests for the env doctor (VRAG-001).

The two behaviours worth locking down: the parser survives the shell-syntax `.env` this
project actually ships, and a secret never reaches a terminal.
"""

import io

from src.doctor import FAIL, PASS, WARN, Check, check_credential, report
from src.env import fingerprint, load_env, parse_env_file

SECRET = "gsk_thisIsNotARealKeyJustAFixture"


def test_parses_shell_export_prefix():
    """The shared .env is `source`-able, so keys arrive as `export KEY=value`."""
    parsed = parse_env_file("export GROQ_API_KEY=abc\nHF_TOKEN=def\n")
    assert parsed == {"GROQ_API_KEY": "abc", "HF_TOKEN": "def"}


def test_skips_comments_and_blanks_and_strips_quotes():
    parsed = parse_env_file(
        "\n# GROQ CONFIG\n"
        'export A="quoted"\n'
        "export B='single'\n"
        "  \n"
        "not_an_assignment\n"
    )
    assert parsed == {"A": "quoted", "B": "single"}


def test_keeps_hash_inside_a_value():
    """`#` is legal in a token; guessing it starts a comment corrupts the credential."""
    assert parse_env_file("K=a#b")["K"] == "a#b"


def test_process_env_beats_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("export VRAG_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("VRAG_TEST_KEY", "from_environment")
    value, source = load_env((env_file,))["VRAG_TEST_KEY"]
    assert (value, source) == ("from_environment", "environment")


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env((tmp_path / "nope.env",)) is not None


def test_fingerprint_hides_the_value():
    printed = fingerprint(SECRET)
    assert SECRET not in printed
    assert f"len={len(SECRET)}" in printed
    assert printed == fingerprint(SECRET)  # stable across runs


def test_credential_missing_is_fail_but_optional_is_warn():
    assert check_credential({}, "GROQ_API_KEY", "gsk_", "why", required=True).status == FAIL
    assert check_credential({}, "LANGFUSE_PUBLIC_KEY", "pk-", "why", required=False).status == WARN


def test_credential_wrong_prefix_warns_and_never_prints_the_secret():
    check = check_credential({"HF_TOKEN": (SECRET, ".env")}, "HF_TOKEN", "hf_", "why", required=True)
    assert check.status == WARN
    assert SECRET not in check.detail


def test_credential_present_and_well_formed_passes():
    check = check_credential({"GROQ_API_KEY": (SECRET, ".env")}, "GROQ_API_KEY", "gsk_", "why", required=True)
    assert check.status == PASS
    assert SECRET not in check.detail


def test_report_exits_non_zero_on_fail_and_names_the_dependency():
    out = io.StringIO()
    code = report([Check("binaries", "ffmpeg", FAIL, "not on PATH")], out=out)
    text = out.getvalue()
    assert code == 1
    assert "FAIL" in text and "ffmpeg" in text


def test_report_exits_zero_when_only_passes_and_warns():
    out = io.StringIO()
    code = report(
        [Check("binaries", "ffmpeg", PASS, "6.1"), Check("services", "model x", WARN, "not pulled")],
        out=out,
    )
    assert code == 0
    assert "1 PASS  1 WARN  0 FAIL" in out.getvalue()
