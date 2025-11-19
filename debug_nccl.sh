#!/bin/bash
# Debug script to see detailed NCCL errors

echo "Running trainer with NCCL debug output..."
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

cd /home/antim/gokul/verifiers-multimodal
source .venv/bin/activate

uv run python scripts/vf-train-multimodal.py @ configs/vf-rl/test-multimodal.toml

