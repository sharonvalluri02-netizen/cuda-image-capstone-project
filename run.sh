#!/bin/bash
set -e

echo "=============================================="
echo " CUDA GPU IMAGE PROCESSING PROJECT"
echo "=============================================="
echo

echo "[1/5] Checking GPU..."
nvidia-smi

echo
echo "[2/5] Generating dataset..."
python3 generate_dataset.py

echo
echo "[3/5] Compiling CUDA program..."
make

echo
echo "[4/5] Running GPU image processing..."
./image_processing input output

echo
echo "[5/5] Counting generated output files..."
COUNT=$(find output -name '*_edges.pgm' | wc -l)
echo "Output images generated: $COUNT"

echo
echo "Project execution completed successfully."
