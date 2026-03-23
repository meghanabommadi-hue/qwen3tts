#!/usr/bin/env bash
# Qwen3TTS setup — installs dependencies and downloads model weights.
#
# Usage:
#   bash qwen3tts/setup/setup.sh
#   bash qwen3tts/setup/setup.sh --skip-model    # skip model download
#   bash qwen3tts/setup/setup.sh --model-dir /path/to/models/Qwen3-TTS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_MODEL=0
MODEL_DIR=""

for arg in "$@"; do
    case "$arg" in
        --skip-model) SKIP_MODEL=1 ;;
        --model-dir=*) MODEL_DIR="${arg#--model-dir=}" ;;
        --model-dir) shift; MODEL_DIR="${1:-}" ;;
    esac
done

echo ""
echo "======================================================="
echo "  Qwen3TTS Setup"
echo "======================================================="
echo "  repo root  : $REPO_ROOT"
echo "  model dir  : ${MODEL_DIR:-~/models/Qwen3-TTS (default)}"
echo "  skip model : $SKIP_MODEL"
echo "======================================================="
echo ""

# ── 1. Install torch + torchaudio (must come before sglang) ──────────────────
echo "==> [1/5] Installing torch + torchaudio"
uv pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

# ── 2. Install sglang — pinned to 0.5.2 (matches FlowTTS; 0.5.9+ breaks with
#        transformers 4.57.x due to DeepseekVL2Config dataclass incompatibility)
echo "==> [2/5] Installing sglang==0.5.2"
uv pip install "sglang[all]==0.5.2"

# ── 3. Install flashinfer kernel (sglang may already pull this; ensures it) ───
echo "==> [3/5] Installing flashinfer-python"
uv pip install flashinfer-python || echo "  (flashinfer-python already pulled by sglang)"

# ── 4. Install transformers — pinned to 4.57.3 first (compatible with sglang 0.5.2),
#        then upgrade to git HEAD so qwen3_tts arch is registered. The git HEAD build
#        must come after sglang so sglang's import-time checks pass against 4.57.3,
#        but the running process picks up qwen3_tts support from the git version.
echo "==> [4/6] Installing transformers==4.57.3 (sglang-compatible base)"
uv pip install "transformers==4.57.3"
echo "==> [5/6] Upgrading transformers to git HEAD (adds qwen3_tts architecture)"
uv pip install "git+https://github.com/huggingface/transformers.git"

# ── 6. Install remaining Python dependencies ──────────────────────────────────
echo "==> [6/6] Installing requirements.txt"
uv pip install -r "$REPO_ROOT/requirements.txt"

# ── Patch sglang's DeepseekVL2Config ─────────────────────────────────────────
# transformers git HEAD makes PretrainedConfig a @dataclass, which breaks sglang
# 0.5.2's DeepseekVL2Config: bare class-level annotations like
#   vision_config: DeepseekVL2VisionEncoderConfig
# are treated as required dataclass fields, causing:
#   TypeError: non-default argument 'vision_config' follows default argument
# Fix: convert those three annotations to ClassVar[type] so dataclass ignores them.
echo "==> Patching sglang config files for transformers git HEAD compatibility"
# transformers git HEAD makes PretrainedConfig a @dataclass. Any bare class-level
# annotation like `vision_config: SomeConfig` inside a PretrainedConfig subclass
# is treated as a required dataclass field, causing:
#   TypeError: non-default argument 'vision_config' follows default argument
# Fix: replace all such bare annotations with ClassVar[type] across all sglang
# config files (deepseekvl2.py, janus_pro.py, and any future additions).
python3 - <<'PYEOF'
import sys, re, pathlib

sglang_dir = next(
    (p for p in pathlib.Path(sys.prefix).rglob("sglang") if p.is_dir() and (p / "__init__.py").exists()),
    None,
)
if not sglang_dir:
    print("  sglang package not found, skipping")
    sys.exit(0)

# Match any bare class-body annotation at 4-space indent with no default value.
# These become required dataclass fields when transformers git HEAD makes
# PretrainedConfig a @dataclass, breaking sglang at import time.
bare_ann = re.compile(r'^(\s{4})(\w+): ([A-Za-z_]\w*)$', re.MULTILINE)
patched = 0
for pyfile in sglang_dir.rglob("*.py"):
    src = pyfile.read_text()
    new_src = bare_ann.sub(r'\1\2: "ClassVar[type]"', src)
    if new_src == src:
        continue
    if "ClassVar" not in new_src:
        if "from typing import" in new_src:
            new_src = new_src.replace("from typing import", "from typing import ClassVar,", 1)
        else:
            new_src = "from typing import ClassVar\n" + new_src
    pyfile.write_text(new_src)
    patched += 1

print(f"  {patched} file(s) patched")
PYEOF

# ── Clear stale .pyc caches ───────────────────────────────────────────────────
# After pinning sglang + sgl_kernel versions, stale bytecode from a different
# version causes ImportError: cannot import name 'fused_marlin_moe' from sgl_kernel.
echo "==> Clearing stale .pyc caches for sglang / sgl_kernel"
find "$(python3 -c 'import site; print(site.getsitepackages()[0])')/sglang" -name "*.pyc" -delete 2>/dev/null || true
find "$(python3 -c 'import site; print(site.getsitepackages()[0])')/sgl_kernel" -name "*.pyc" -delete 2>/dev/null || true

# ── 6. Download Qwen3-TTS model weights ───────────────────────────────────────
if [[ "$SKIP_MODEL" -eq 1 ]]; then
    echo "==> [7/7] Skipping model download (--skip-model)"
else
    echo "==> [7/7] Downloading Qwen3-TTS model weights (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
    if [[ -n "$MODEL_DIR" ]]; then
        python3 "$SCRIPT_DIR/download_models.py" --model-dir "$MODEL_DIR"
    else
        python3 "$SCRIPT_DIR/download_models.py"
    fi
fi

echo ""
echo "======================================================="
echo "  Setup complete."
echo ""
echo "  Run the server:"
echo "    cd $REPO_ROOT"
echo "    bash run.sh --ports 10 --ctrl-port 9764"
echo ""
echo "  Benchmark:"
echo "    python3 -m qwen3tts.test.benchmark --base-port 9765 --n-ports 10"
echo "======================================================="
echo ""
