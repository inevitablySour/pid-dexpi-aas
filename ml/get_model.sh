#!/usr/bin/env bash
# Download the trained model for the review app's image and hybrid modes into ml/models/
# (published as a GitHub release asset - too big for git). The app uses
# ml/models/pid2graph_final.pt when present, else pid2graph_fold1.pt.
# Needs the GitHub CLI logged in to an account with access to the repo: gh auth login
#   ml/get_model.sh                      # the final model (all data)
#   MODEL_TAG=model-fold1 ml/get_model.sh   # the older fold-1 model
set -euo pipefail
TAG="${MODEL_TAG:-model-final}"
FILE="pid2graph_${TAG#model-}.pt"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/ml/models"
cd "$ROOT"
gh release download "$TAG" --pattern "$FILE" --dir ml/models --clobber
echo "model -> ml/models/$FILE"
