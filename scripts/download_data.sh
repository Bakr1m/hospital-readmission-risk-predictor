#!/usr/bin/env bash
# Fetch the project datasets into data/.
# The data is public (UCI ML Repository, "Diabetes 130-US hospitals",
# dataset 296, years 1999-2008) but too large to version-control,
# so it is gitignored — run this script after cloning.
set -euo pipefail
cd "$(dirname "$0")/.."

need_download=0
for f in data/diabetes_raw.csv data/diabetes_clean.csv; do
  if [ ! -f "$f" ]; then
    echo "missing: $f"
    need_download=1
  fi
done

if [ "$need_download" -eq 0 ]; then
  echo "data/ already present — nothing to do."
  exit 0
fi

cat <<'EOF'
Automatic download is not wired up for this dataset.
Please obtain the files manually:

  1. Visit the UCI ML Repository and open dataset 296
     ("Diabetes 130-US hospitals for years 1999-2008").
  2. Download the encounters CSV and save it as:
       data/diabetes_raw.csv
  3. Run the cleaning/feature-engineering notebooks (02_cleaning)
     or the training pipeline to produce:
       data/diabetes_clean.csv

Expected layout afterwards:
  data/diabetes_raw.csv    (~19 MB)
  data/diabetes_clean.csv  (~26 MB, includes engineered columns)
EOF
exit 1
