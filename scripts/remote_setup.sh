#!/usr/bin/env bash
set -euo pipefail

repo_url="${1:-https://github.com/zoronosuke/hierarchical-fl.git}"
revision="${2:?revision is required}"
install_dir="${3:-$HOME/hierarchical-fl-persistent}"

if [[ -e "$install_dir" && ! -d "$install_dir/.git" ]]; then
    echo "ERROR: $install_dir exists but is not a Git repository" >&2
    exit 1
fi

if [[ ! -d "$install_dir/.git" ]]; then
    git clone "$repo_url" "$install_dir"
fi

if ! git -C "$install_dir" diff --quiet || ! git -C "$install_dir" diff --cached --quiet; then
    echo "ERROR: $install_dir has tracked local changes; refusing to overwrite them" >&2
    git -C "$install_dir" status --short >&2
    exit 1
fi

git -C "$install_dir" fetch origin --prune
git -C "$install_dir" checkout --detach "$revision"

cd "$install_dir"
# JetPack 6.2.1 stores cuDSS outside the default dynamic-linker path.
jetson_lib_dir="/usr/lib/aarch64-linux-gnu/libcudss/12"
if [[ -d "$jetson_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$jetson_lib_dir:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
bash scripts/setup_jetson.sh
mkdir -p logs
echo "Ready: $(hostname) revision=$(git rev-parse --short HEAD) dir=$install_dir"
