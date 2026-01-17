#!/bin/bash
# Multi-GPU Training Launcher
# Generic launcher for distributed training jobs

set -e

# Parse arguments
SCRIPT=${1:-""}
shift || true

if [ -z "$SCRIPT" ]; then
    echo "Usage: ./launch_mgpu.sh <script.py> [args...]"
    echo ""
    echo "Examples:"
    echo "  ./launch_mgpu.sh src/retrieval/train_dense.py --config configs/dense_encoder.yaml"
    echo "  ./launch_mgpu.sh src/preference/dpo_train.py --config configs/dpo.yaml"
    exit 1
fi

# Configuration
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

echo "========================================"
echo "Multi-GPU Training Launcher"
echo "========================================"
echo "Script: $SCRIPT"
echo "Arguments: $@"
echo "========================================"
echo "Distributed Config:"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  NNODES: $NNODES"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "========================================"

# Set environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=false

# For AWS instances
if [ -f /opt/amazon/efa/bin/fi_info ]; then
    echo "EFA detected, configuring for AWS..."
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

# Launch
if [ $NUM_GPUS -gt 1 ] || [ $NNODES -gt 1 ]; then
    echo ""
    echo "Launching distributed training..."
    echo ""
    
    torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $SCRIPT \
        --distributed \
        "$@"
else
    echo ""
    echo "Launching single-GPU training..."
    echo ""
    
    python $SCRIPT "$@"
fi

echo ""
echo "========================================"
echo "Training job complete!"
echo "========================================"
