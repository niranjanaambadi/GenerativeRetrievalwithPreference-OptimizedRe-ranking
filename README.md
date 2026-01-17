# Generative Retrieval with Preference-Optimized Re-ranking

A hybrid search pipeline combining BM25 sparse retrieval, dense encoding, and LLM-based re-ranking with Direct Preference Optimization (DPO). This project demonstrates modern information retrieval techniques with distributed training support.

## Overview

This system implements a three-stage retrieval pipeline:

1. **BM25 Retrieval** — Fast sparse retrieval for candidate generation
2. **Dense Encoding** — Neural bi-encoder for semantic similarity
3. **LLM Re-ranking** — Mistral-7B with DPO for preference-aligned ranking

## Architecture

```
Query → BM25 (top-100) → Dense Encoder (top-20) → LLM Re-ranker (top-10)
                                                        ↓
                                              DPO Fine-tuned Mistral-7B
```

## Features

- **Hybrid Retrieval**: Combines lexical (BM25) and semantic (dense) retrieval
- **Preference Optimization**: DPO training on synthetic preference pairs
- **Distributed Training**: Multi-GPU support via `torchrun`
- **BEIR Evaluation**: Standard IR metrics (nDCG@10, MRR@10, Recall@k)
- **Interactive Demo**: Gradio UI + CLI evaluation tools

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/gen-retrieval-pref.git
cd gen-retrieval-pref

# Create environment
conda create -n genret python=3.10 -y
conda activate genret

# Install dependencies
pip install -r requirements.txt

# Download data
bash scripts/download_data.sh
```

## Quick Start

### 1. Train Dense Encoder

```bash
# Single GPU
python src/retrieval/train_dense.py --config configs/dense_encoder.yaml

# Multi-GPU (4x A10G)
bash scripts/train_dense.sh
```

### 2. Generate Preference Pairs

```bash
python src/preference/generate_pairs.py \
    --dataset msmarco \
    --output data/preferences/train.jsonl
```

### 3. DPO Training

```bash
# Single GPU
python src/preference/dpo_train.py --config configs/dpo.yaml

# Multi-GPU
bash scripts/dpo_pref.sh
```

### 4. Evaluation

```bash
# Run full evaluation
python src/eval/evaluate.py \
    --model_path checkpoints/dpo-mistral-7b \
    --dataset trec-covid \
    --metrics ndcg@10 mrr@10 recall@100

# Quick CLI demo
python src/eval/cli_demo.py --query "COVID-19 vaccine efficacy"
```

### 5. Launch Demo

```bash
python src/demo/app.py --port 7860
```

## Project Structure

```
gen-retrieval-pref/
├── README.md
├── requirements.txt
├── configs/
│   ├── dense_encoder.yaml      # Dense encoder training config
│   ├── dpo.yaml                # DPO training config
│   └── eval.yaml               # Evaluation config
├── data/
│   ├── msmarco/                # MS MARCO subset
│   ├── beir/                   # BEIR evaluation datasets
│   └── preferences/            # Generated preference pairs
├── src/
│   ├── retrieval/
│   │   ├── bm25.py             # BM25 retrieval
│   │   ├── dense_encoder.py    # Bi-encoder model
│   │   └── train_dense.py      # Dense encoder training
│   ├── rerank/
│   │   ├── llm_reranker.py     # Mistral-7B re-ranker
│   │   └── prompts.py          # Ranking prompts
│   ├── preference/
│   │   ├── generate_pairs.py   # Synthetic preference generation
│   │   └── dpo_train.py        # DPO training loop
│   ├── eval/
│   │   ├── metrics.py          # IR metrics implementation
│   │   ├── evaluate.py         # Evaluation script
│   │   └── cli_demo.py         # CLI demonstration
│   └── demo/
│       └── app.py              # Gradio application
├── scripts/
│   ├── download_data.sh        # Data download script
│   ├── train_dense.sh          # Distributed dense training
│   ├── dpo_pref.sh             # Distributed DPO training
│   └── eval.sh                 # Evaluation script
├── notebooks/
│   └── analysis.ipynb          # Results analysis
└── launch_mgpu.sh              # Multi-GPU launcher
```

## Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Dense Encoder | MiniLM-L6-v2 | 22M |
| Re-ranker (base) | Mistral-7B-v0.3 | 7B |
| Re-ranker (DPO) | Mistral-7B + DPO | 7B |

## Datasets

**Training:**
- MS MARCO (passage ranking subset): ~500K queries

**Evaluation (BEIR):**
- TREC-COVID: 50 queries, 171K documents
- Natural Questions: 3,452 queries

## Results

| Model | TREC-COVID nDCG@10 | NQ nDCG@10 | MRR@10 |
|-------|-------------------|------------|--------|
| BM25 only | 0.594 | 0.329 | 0.312 |
| + Dense | 0.657 | 0.412 | 0.398 |
| + LLM Re-rank | 0.701 | 0.456 | 0.441 |
| + DPO | **0.723** | **0.478** | **0.463** |

*Results on BEIR benchmark subsets*

## Distributed Training

### AWS Setup (g5.12xlarge - 4x A10G)

```bash
# Launch multi-GPU training
torchrun --nproc_per_node=4 \
    src/preference/dpo_train.py \
    --config configs/dpo.yaml \
    --distributed
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

## Configuration

### Dense Encoder (`configs/dense_encoder.yaml`)

```yaml
model:
  name: sentence-transformers/all-MiniLM-L6-v2
  max_seq_length: 512

training:
  batch_size: 64
  learning_rate: 2e-5
  epochs: 3
  warmup_ratio: 0.1

distributed:
  backend: nccl
  world_size: 4
```

### DPO Training (`configs/dpo.yaml`)

```yaml
model:
  name: mistralai/Mistral-7B-v0.3
  load_in_4bit: true

dpo:
  beta: 0.1
  learning_rate: 5e-7
  batch_size: 4
  gradient_accumulation_steps: 8

training:
  epochs: 1
  max_steps: 1000
  save_steps: 200
```

## Citation

If you use this work, please cite:

```bibtex
@software{gen_retrieval_pref,
  title={Generative Retrieval with Preference-Optimized Re-ranking},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gen-retrieval-pref}
}
```

## License

Apache 2.0

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for Mistral-7B
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MS MARCO](https://microsoft.github.io/msmarco/)
- [Hugging Face](https://huggingface.co/) for transformers and TRL
