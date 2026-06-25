#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python scripts/collect_phospholipid_msms.py \
  --out "./expanded_phospholipids" \
  --work "./_downloads"
