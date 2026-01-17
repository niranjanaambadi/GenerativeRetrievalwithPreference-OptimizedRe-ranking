#!/bin/bash
# Distributed training script for dense encoder
# Supports multi-GPU training with torchrun

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-4}
CONFIG=${CONFIG:-"configs/dense_encoder.yaml"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "========================================"
echo "Dense Encoder Training"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Master Port: $MASTER_PORT"
echo "========================================"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Running on CPU."
    NUM_GPUS=1
else
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Available GPUs: $AVAILABLE_GPUS"
    
    if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
        NUM_GPUS=$AVAILABLE_GPUS
    fi
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export MASTER_ADDR=localhost
export MASTER_PORT=$MASTER_PORT
export NCCL_DEBUG=INFO

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create output directory
mkdir -p checkpoints/dense-encoder
mkdir -p logs

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Starting distributed training with $NUM_GPUS GPUs..."
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        src/retrieval/train_dense.py \
        --config $CONFIG \
        --distributed \
        2>&1 | tee logs/dense_training_$(date +%Y%m%d_%H%M%S).log
else
    echo "Starting single-GPU training..."
    
    python src/retrieval/train_dense.py \
        --config $CONFIG \
        2>&1 | tee logs/dense_training_$(date +%Y%m%d_%H%M%S).log
fi

echo "========================================"
echo "Training complete!"
echo "Model saved to: checkpoints/dense-encoder"
echo "========================================"
