#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get -y install --no-install-recommends \
            git \
            make \
            cmake \
            build-essential \
            python3-dev \
            python3-pip \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            liblzma-dev \
            libffi-dev \
            curl 
python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Setup completed."
