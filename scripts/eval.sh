#!/bin/bash
# Evaluation script for the retrieval pipeline
# Runs ablation study on BEIR datasets

set -e

# Configuration
DATASET=${DATASET:-"trec-covid"}
CONFIG=${CONFIG:-"configs/eval.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"results"}

echo "========================================"
echo "Pipeline Evaluation"
echo "========================================"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "========================================"

mkdir -p $OUTPUT_DIR

# Run ablation studies
echo ""
echo "Running ablation studies..."
echo ""

# 1. BM25 only
echo "[1/4] Evaluating BM25 only..."
python -m src.eval.evaluate \
    --config $CONFIG \
    --dataset $DATASET \
    --stages bm25 \
    --output $OUTPUT_DIR/${DATASET}_bm25.json

# 2. BM25 + Dense
echo "[2/4] Evaluating BM25 + Dense..."
python -m src.eval.evaluate \
    --config $CONFIG \
    --dataset $DATASET \
    --stages bm25 dense \
    --output $OUTPUT_DIR/${DATASET}_bm25_dense.json

# 3. BM25 + Dense + LLM Re-rank (base)
echo "[3/4] Evaluating BM25 + Dense + LLM Re-rank (base)..."
python -m src.eval.evaluate \
    --config $CONFIG \
    --dataset $DATASET \
    --stages bm25 dense rerank_base \
    --output $OUTPUT_DIR/${DATASET}_bm25_dense_rerank.json

# 4. BM25 + Dense + LLM Re-rank (DPO)
echo "[4/4] Evaluating BM25 + Dense + LLM Re-rank (DPO)..."
python -m src.eval.evaluate \
    --config $CONFIG \
    --dataset $DATASET \
    --stages bm25 dense rerank_dpo \
    --output $OUTPUT_DIR/${DATASET}_bm25_dense_dpo.json

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"

# Summarize results
echo ""
echo "Results Summary:"
echo "----------------"

for f in $OUTPUT_DIR/${DATASET}_*.json; do
    echo ""
    echo "$(basename $f):"
    python -c "
import json
with open('$f') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    print(f\"  nDCG@10: {metrics.get('ndcg@10', 'N/A'):.4f}\")
    print(f\"  MRR@10:  {metrics.get('mrr@10', 'N/A'):.4f}\")
    print(f\"  Recall@100: {metrics.get('recall@100', 'N/A'):.4f}\")
" 2>/dev/null || echo "  (Could not parse results)"
done

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
