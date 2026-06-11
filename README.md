
# Self-SoftCoT: A Self-Consistent Framework via Position-Aware Latent Space Reinforcement Learning

This repository contains the official PyTorch implementation for the paper **"Self-SoftCoT: A Self-Consistent Framework via Position-Aware Latent Space Reinforcement Learning"**.

> **Terminology Note:** In this codebase, the project is internally referred to as `Unified-SoftCoT`. All class names (e.g., `UnifiedSoftCoT`) and utility files correspond to the `Self-SoftCoT` framework described in the paper.

## Overview

**Self-SoftCoT** is a single-stream latent reasoning framework that enables frozen Large Language Models (LLMs) to generate and consume internal thought tokens without external assistant models.

This repository implements the **Group Sequence Policy Optimization (GSPO)** algorithm enhanced with an **Experience Replay (Buffer)** mechanism. This "Collect-then-Train" strategy stabilizes gradients for position-aware independent projection layers.

## Note on the Qwen3-LongCoT Baseline

In the paper, **Qwen3-LongCoT** refers to Qwen3-8B evaluated with thinking mode enabled, where the model generates explicit long-chain reasoning traces before producing the final answer.

It is not a separate method proposed by this work, nor does it denote a new official model name. The citation to Qwen Team (2025) is used to reference the Qwen3 model family and its thinking-mode mechanism.

The Qwen3-LongCoT results reported in the paper are our own evaluations under the same benchmark protocol, using Qwen3-8B with thinking mode enabled across five random seeds.

## Implementation Note

The released implementation should be treated as the main reference for reproduction. The main GSPO results reported in the paper are obtained using sequence-level probability-ratio clipping, KL regularization, and experience replay, as implemented in the training scripts.

The DPO/KTO ablations are implemented as DPO-style and KTO-style variants under the same Self-SoftCoT architecture and are adapted to the projection-layer training setting, rather than being standard offline DPO/KTO implementations. These preference-style variants use optimization objectives that differ from GSPO and do not adopt GSPO’s advantage-weighted sequence-level probability-ratio clipping mechanism. Therefore, they may be less stable than GSPO in this projection-layer training setting.DPO-style optimization is more prone to generation degeneration, output-length inflation, repetitive boilerplate content, and degraded response quality, while KTO-style optimization is relatively more stable. These preference-style variants are provided only as auxiliary ablations, and the main reproducible results of the paper should be based on the GSPO implementation.

The main GSPO results reported in the paper do not rely on explicit gradient-norm clipping.

## Requirements

Please ensure the following core dependencies are installed. We recommend using a virtual environment (Conda).

```
pip install torch==2.7.0
pip install transformers==4.51.0
pip install fastNLP==0.7.0
pip install tqdm
Project Structure
Plaintext

.
├── train_gspo_buffer_multitask.py    # [CORE] Main training script with Buffer & GSPO
├── evaluate_unified.py               # Evaluation script for all tasks
├── unified_llm_model.py              # Model architecture (Independent Projections)
├── unified_utils.py                  # Preprocessing templates and utility functions
├── data_loader.py                    # Data loading logic for GSM8K, AQuA, etc.
└── data/                             # Dataset directory
    ├── GSM8K/
    ├── ASDiv-Aug/
    ├── AQuA/
    ├── StrategyQA/
    └── DU/                           
Training (Buffer Configuration)
We support training on Qwen2.5 and LLaMA-3.1 backbones. The training script uses a Buffer-based GSPO strategy: it collects a batch of episodes first, then updates the policy using experience replay.

1. Key Buffer Arguments
To strictly reproduce the Config B (Robust Recipe) from the paper, ensure you set these arguments:

--episodes_per_round 16: Buffer Size. Collects 16 episodes before performing gradient updates.

--update_epochs 3: Replay Ratio. The collected buffer is reused for 3 internal update epochs.

--group_size 5: GSPO Group. Samples 5 outputs per question to calculate sequence-level advantages.

--beta_kl 0.01: KL Penalty. Keeps the policy close to the behavior policy during replay.

2. Training Commands
Qwen-2.5-7B-Instruct (Math Reasoning)


python train_gspo_buffer_multitask.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --data_path "./data/GSM8K" \
    --output_dir "./output_qwen_gsm8k_buffer" \
    --task_name "gsm8k" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 1e-5 \
    --beta_kl 0.01 \
    --max_data_epochs 1
LLaMA-3.1-8B-Instruct (Commonsense QA)


python train_gspo_buffer_multitask.py \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --data_path "./data/StrategyQA" \
    --output_dir "./output_llama_sqa_buffer" \
    --task_name "strategyqa" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 5e-6 \
    --beta_kl 0.01
Evaluation
To evaluate a trained checkpoint (e.g., final.bin) on the test set across 5 random seeds (41-45) as reported in the paper, use the following  script:

#!/bin/
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="./data/GSM8K"
CKPT_PATH="./output_qwen_gsm8k_buffer/ckpt/final.bin"
TASK="gsm8k"

for SEED in {41..45}; do
    echo "Running Evaluation with Seed $SEED..."
    python evaluate_unified.py \
        --model_id "$MODEL_ID" \
        --data_path "$DATA_PATH" \
        --params_file_name "$CKPT_PATH" \
        --task_name "$TASK" \
        --dataset_split "test" \
        --num_thought_tokens 2 \
        --seed $SEED
done
