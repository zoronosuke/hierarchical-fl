#!/usr/bin/env bash
# =============================================================================
# Jetson Orin Nano 環境構築スクリプト (uv版)
# JetPack 6.x (Ubuntu 22.04 / L4T) 向け
# =============================================================================
set -e

echo "=== Jetson Orin Nano HFL Setup (uv) ==="

# --- 1. uv インストール ---
if ! command -v uv &> /dev/null; then
    echo "[1/3] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/3] uv already installed: $(uv --version)"
fi

# --- 2. 依存関係インストール ---
echo "[2/3] Installing dependencies with uv..."
# pyproject.toml の [tool.uv.sources] で aarch64 の場合は
# jetson-pytorch インデックスから torch をインストールする設定済み
uv sync

# インストール確認
echo "  Checking torch..."
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# --- 3. 電力モード ---
echo "[3/3] Power mode info:"
echo "  Current: $(sudo nvpmodel -q 2>/dev/null || echo 'N/A')"
echo "  MAXN mode: sudo nvpmodel -m 0"
echo "  15W mode:  sudo nvpmodel -m 1"

echo ""
echo "=== Setup complete ==="
echo "Run dry-run: uv run python run.py --dry-run"
