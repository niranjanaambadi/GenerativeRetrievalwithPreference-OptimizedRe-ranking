#!/bin/bash
# Distributed DPO training script for Mistral-7B re-ranker
# Supports multi-GPU training with torchrun

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-4}
CONFIG=${CONFIG:-"configs/dpo.yaml"}
MASTER_PORT=${MASTER_PORT:-29501}
MAX_SAMPLES=${MAX_SAMPLES:-}

echo "========================================"
echo "DPO Re-ranker Training (Mistral-7B)"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Master Port: $MASTER_PORT"
echo "========================================"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. GPU required for DPO training."
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Check VRAM
echo ""
echo "GPU Memory Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# For Flash Attention
export CUDA_LAUNCH_BLOCKING=0

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create directories
mkdir -p checkpoints/dpo-mistral-7b
mkdir -p logs

# Build command
CMD="src/preference/dpo_train.py --config $CONFIG"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Starting distributed DPO training with $NUM_GPUS GPUs..."
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $CMD \
        --distributed \
        2>&1 | tee logs/dpo_training_$(date +%Y%m%d_%H%M%S).log
else
    echo "Starting single-GPU DPO training..."
    
    python $CMD \
        2>&1 | tee logs/dpo_training_$(date +%Y%m%d_%H%M%S).log
fi

echo "========================================"
echo "DPO Training complete!"
echo "Model saved to: checkpoints/dpo-mistral-7b"
echo "========================================"
