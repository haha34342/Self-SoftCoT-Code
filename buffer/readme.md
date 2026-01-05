Unified SoftCoT/GSPO Training & Evaluation Guide
本项目用于训练和评估基于投影层（Projection Layer）的强化学习模型（GSPO）。支持 Qwen-2.5 和 Llama-3 两个基座模型在 GSM8K, ASDiv-Aug, AQuA, StrategyQA 四个数据集上的实验。

1. 环境依赖 (Requirements)
请确保在运行代码前安装以下特定版本的核心库：

Bash

pip install fastNLP==0.7.0
pip install torch==2.7.0
pip install transformers==4.51.0
2. 项目结构 (Project Structure)
请确保代码与数据目录的相对位置如下所示：

Plaintext

.
├── train_gspo_buffer_multitask.py    # 训练主脚本
├── evaluate_unified.py               # 测评主脚本
├── unified_llm_model.py              # 模型定义
├── unified_utils.py                  # 工具函数
├── data_loader.py                    # 数据加载器
└── data/                             # 数据根目录
    ├── GSM8K/
    ├── ASDiv-Aug/
    ├── AQuA/
    └── StrategyQA/
3. 训练指令 (Training)
3.1 Qwen-2.5-7B-Instruct (Learning Rate: 1e-5)
注意: 请根据您的实际模型路径修改 --model_id。

(1) GSM8K (Budget: 1 Epoch)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --data_path "./data/GSM8K" \
    --output_dir "./output_qwen2.5_gsm8k_n2_v2_budget_1epoch_lr1e5_150steps" \
    --task_name "gsm8k" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 1e-5 \
    --max_data_epochs 1
(2) ASDiv-Aug (Budget: 1 Epoch)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --data_path "./data/ASDiv-Aug" \
    --output_dir "./output_qwen2.5_asdiv_n2_v2_budget_1epoch_lr1e5_150steps" \
    --task_name "asdiv-aug" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 1e-5 \
    --max_data_epochs 1
(3) AQuA (Budget: Unlimited)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --data_path "./data/AQuA" \
    --output_dir "./output_qwen2.5_aqua_n2_v2_budget_unlim_lr1e5_150steps" \
    --task_name "aqua" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 1e-5 \
    --max_data_epochs 0
(4) StrategyQA (Budget: Unlimited)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
    --data_path "./data/StrategyQA" \
    --output_dir "./output_qwen2.5_sqa_n2_v2_budget_unlim_lr1e5_150steps" \
    --task_name "strategyqa" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 1e-5 \
    --max_data_epochs 0
3.2 Llama-3.1-8B-Instruct (Learning Rate: 5e-6)
注意: 请根据您的实际模型路径修改 --model_id。

(1) GSM8K (Budget: 1 Epoch)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --data_path "./data/GSM8K" \
    --output_dir "./output_llama3_gsm8k_n2_v2_budget_1epoch_150steps" \
    --task_name "gsm8k" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 5e-6 \
    --max_data_epochs 1
(2) ASDiv-Aug (Budget: 1 Epoch)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --data_path "./data/ASDiv-Aug" \
    --output_dir "./output_llama3_asdiv_n2_v2_budget_1epoch_150steps" \
    --task_name "asdiv-aug" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 5e-6 \
    --max_data_epochs 1
(3) AQuA (Budget: Unlimited)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --data_path "./data/AQuA" \
    --output_dir "./output_llama3_aqua_n2_v2_budget_unlim_150steps" \
    --task_name "aqua" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 5e-6 \
    --max_data_epochs 0
(4) StrategyQA (Budget: Unlimited)
Bash

python train_gspo_buffer_multitask.py \
    --model_id "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --data_path "./data/StrategyQA" \
    --output_dir "./output_llama3_sqa_n2_v2_budget_unlim_150steps" \
    --task_name "strategyqa" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --episodes_per_round 16 \
    --update_epochs 3 \
    --train_steps 150 \
    --save_every 10 \
    --lr 5e-6 \
    --max_data_epochs 0
4. 测评指令 (Evaluation)
批量运行 Seed 41-45
要对训练好的权重进行 5 次随机种子（41, 42, 43, 44, 45）的完整测评，请使用以下脚本模板。

参数说明：

MODEL_ID: 基座模型路径 (Qwen 或 Llama)

DATA_PATH: 测试数据路径

TASK_NAME: 任务名称 (gsm8k, asdiv-aug, aqua, strategyqa)

CKPT_PATH: 训练生成的权重文件 (例如 ./output_.../ckpt/final.bin)

通用测评脚本示例 (Shell)
将以下代码保存为 run_eval.sh，修改头部的变量后运行：

Bash

#!/bin/bash

# ================= 配置区域 =================
# 1. 设置基座模型路径
MODEL_ID="/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

# 2. 设置任务和数据路径
TASK_NAME="gsm8k"  # 可选: gsm8k, asdiv-aug, aqua, strategyqa
DATA_PATH="./data/GSM8K"

# 3. 设置训练好的权重路径 (final.bin)
CKPT_PATH="./output_qwen2.5_gsm8k_n2_v2_budget_1epoch_lr1e5_150steps/ckpt/final.bin"

# 4. 日志保存目录
LOG_DIR="./logs_eval_${TASK_NAME}"
mkdir -p $LOG_DIR
# ===========================================

echo "开始测评 Task: $TASK_NAME"
echo "权重: $CKPT_PATH"

for SEED in {41..45}; do
    echo "--------------------------------"
    echo "Running Seed: $SEED"
    
    python -u evaluate_unified.py \
        --model_id "$MODEL_ID" \
        --data_path "$DATA_PATH" \
        --params_file_name "$CKPT_PATH" \
        --task_name "$TASK_NAME" \
        --dataset_split "test" \
        --num_thought_tokens 2 \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/eval_seed${SEED}.log"
done

echo "所有 Seed 运行完毕。请检查 $LOG_DIR 下的日志文件。"
快速提取结果
运行完上述脚本后，可以使用以下命令快速查看 5 个 Seed 的平均准确率：

Bash

grep "Final Accuracy" ./logs_eval_gsm8k/*.log