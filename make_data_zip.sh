#!/usr/bin/env bash
# Build svg_data.zip — the file you upload to Google Drive for Colab.
# Run from the repo root.
#
# What goes in the zip:
#   part2/data/train.bin   (~210 MB)
#   part2/data/val.bin     (~2 MB)
#   part2/data/test.bin    (~2 MB)
#   part2/data/stats.json
#
# Usage:
#   bash make_data_zip.sh

set -e

VENV_"python="./venv/bin/"python"
ZIP_NAME="svg_data.zip"

echo "=== Step 1: tokenize JSONL splits → binary files ==="
if [ -f part2/data/train.bin ]; then
    echo "  part2/data/train.bin already exists — skipping prepare.py"
else
    echo "  Running part2/prepare.py ..."
    cd part2
    $VENV_"python prepare.py
    cd ..
fi

echo ""
echo "=== Step 2: create $ZIP_NAME ==="
rm -f "$ZIP_NAME"
zip "$ZIP_NAME" \
    part2/data/train.bin \
    part2/data/val.bin \
    part2/data/test.bin \
    part2/data/stats.json

echo ""
echo "Done!"
ls -lh "$ZIP_NAME"
echo ""
echo "Next: upload $ZIP_NAME to Google Drive, then in Colab run:"
echo "  from google.colab import drive"
echo "  drive.mount('/content/drive')"
echo "  !unzip /content/drive/MyDrive/svg_data.zip -d /content/ml-proj"