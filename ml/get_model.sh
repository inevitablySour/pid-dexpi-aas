#!/usr/bin/env bash
# Download the trained symbol/connection model for the review app's image and hybrid
# modes into ml/models/ (published as a GitHub release asset - too big for git).
# Needs the GitHub CLI logged in to an account with access to the repo: gh auth login
set -euo pipefail
TAG="${MODEL_TAG:-model-fold1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/ml/models"
cd "$ROOT"
gh release download "$TAG" --pattern "pid2graph_fold1.pt" --dir ml/models --clobber
echo "model -> ml/models/pid2graph_fold1.pt"
