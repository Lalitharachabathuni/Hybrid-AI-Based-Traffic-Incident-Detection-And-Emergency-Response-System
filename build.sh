#!/usr/bin/env bash
# Render Build Script
# Exit on any error
set -o errexit

echo "=== [1/3] Installing Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/3] Building Synthesized Feature Dataset for NH16 Corridor ==="
cd hybrid_system/backend/ml
python preprocess.py

echo "=== [3/3] Training XGBoost Incident Severity Model ==="
python train_model.py
cd ../../..

echo "=== Build Complete! Ready to launch FastAPI on Render ==="
