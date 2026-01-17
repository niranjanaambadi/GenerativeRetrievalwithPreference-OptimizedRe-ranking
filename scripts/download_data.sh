#!/bin/bash
# Download datasets for training and evaluation

set -e

DATA_DIR="data"
mkdir -p $DATA_DIR

echo "========================================"
echo "Downloading datasets..."
echo "========================================"

# MS MARCO (small subset for training)
echo "Downloading MS MARCO..."
mkdir -p $DATA_DIR/msmarco

# Download from BEIR
if ! command -v wget &> /dev/null; then
    echo "wget not found, using curl..."
    curl -L "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip" -o $DATA_DIR/msmarco.zip
else
    wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip" -O $DATA_DIR/msmarco.zip
fi

unzip -q $DATA_DIR/msmarco.zip -d $DATA_DIR/
rm $DATA_DIR/msmarco.zip
echo "MS MARCO downloaded!"

# TREC-COVID for evaluation
echo "Downloading TREC-COVID..."
mkdir -p $DATA_DIR/beir

if ! command -v wget &> /dev/null; then
    curl -L "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip" -o $DATA_DIR/trec-covid.zip
else
    wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip" -O $DATA_DIR/trec-covid.zip
fi

unzip -q $DATA_DIR/trec-covid.zip -d $DATA_DIR/beir/
rm $DATA_DIR/trec-covid.zip
echo "TREC-COVID downloaded!"

# Natural Questions (optional)
echo "Downloading Natural Questions..."
if ! command -v wget &> /dev/null; then
    curl -L "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip" -o $DATA_DIR/nq.zip
else
    wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip" -O $DATA_DIR/nq.zip
fi

unzip -q $DATA_DIR/nq.zip -d $DATA_DIR/beir/
rm $DATA_DIR/nq.zip
echo "Natural Questions downloaded!"

# Create preference data directory
mkdir -p $DATA_DIR/preferences

echo "========================================"
echo "All datasets downloaded successfully!"
echo "========================================"
echo ""
echo "Directory structure:"
ls -la $DATA_DIR/
echo ""
echo "Next steps:"
echo "1. Generate preference pairs: python src/preference/generate_pairs.py"
echo "2. Train dense encoder: bash scripts/train_dense.sh"
echo "3. Train DPO reranker: bash scripts/dpo_pref.sh"
