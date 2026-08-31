#!/bin/sh
# Pull the embedding model into the shared ollama volume, once.
#
# THE TAG IS THE WHOLE POINT OF THIS FILE.
#
# `ollama pull hf.co/<repo>` with no tag fetches the repo's *smallest* file. For
# nomic-embed-text-v1.5-GGUF that is Q2_K — 2-bit weights on a 137 M-parameter encoder — and
# the measurement is recorded in config.toml against the [embed] model lever:
#
#   Q2_K   no prefixes     5/12 = 0.4167   recall@5 on evals/dev
#   F16    no prefixes    11/12 = 0.9167
#
# Same code, same chunks, same questions. 225 MB of weights is the difference between a
# system that works and one that does not, and the untagged pull is the one that looks
# correct in a script. So the tag is written out here, checked against config.toml below,
# and never templated.
set -eu

MODEL="hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16"

# config.toml is the source of truth for the lever; this script must not drift from it.
# It is mounted read-only in compose.yaml, so if the two disagree, stop rather than pull
# something the pipeline will not ask for.
CONFIG="/app/config.toml"
if [ -f "$CONFIG" ]; then
    want="$(sed -n 's/^model *= *"\(nomic[^"]*\)".*/\1/p' "$CONFIG" | head -1)"
    if [ -n "$want" ] && [ "hf.co/$want" != "$MODEL" ]; then
        echo "FAIL: config.toml [embed] model is 'hf.co/$want'" >&2
        echo "      but docker/ollama-init.sh pulls '$MODEL'." >&2
        echo "      Change both, or the index is built with weights nothing queries with." >&2
        exit 1
    fi
fi

echo "ollama-init: pulling $MODEL"
ollama pull "$MODEL"

# Prove it landed under the name src/embed.py will ask for. `_hf_to_ollama_tag` builds
# 'hf.co/<repo id>' from the [embed] model in config.toml, so a pull that succeeded under a
# different name is a failure that would only surface as an empty index later.
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "FAIL: $MODEL pulled but is not in \`ollama list\`" >&2
    exit 1
fi

echo "ollama-init: ok"
ollama list
