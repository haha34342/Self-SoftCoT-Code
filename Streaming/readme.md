📄 README 1: Qwen2.5-7B-Instruct (Online Policy)
Markdown

# Qwen2.5-7B-Instruct 独立部署与运行指南

本项目基于 **Qwen2.5** 体系，采用 **Online Policy (流式训练)** 策略。
该策略利用 Qwen 模型较强的泛化能力，使用恒等初始化 (Identity Init) 和流式更新，训练收敛速度快。

## 🛠️ 1. 环境依赖 (Requirements)

请务必确保您的 Python 环境安装了以下指定版本的核心库：

```bash
pip install fastNLP==0.7.0
pip install torch==2.7.0
pip install transformers==4.51.0
(注：如找不到特定版本，请使用兼容的最新版本，但建议保持一致以复现结果)

📂 2. 项目文件结构
请将文件组织如下（建议项目根目录为 zuizhong）：

Plaintext

/your/custom/path/zuizhong/            <-- 项目根目录 (可自定义)
├── data_loader.py                     # [核心] 数据加载器 (不可缺失)
├── evaluate_unified.py                # [核心] 评测脚本
├── train_gspo_all_tasks.py            # [Qwen] Online Policy 训练脚本
├── unified_llm_model.py               # 模型定义
├── unified_utils.py                   # 工具函数
├── data/                              # [数据] 数据集文件夹 (相对路径需固定)
│   ├── GSM8K/
│   ├── ASDiv-Aug/
│   ├── AQuA/
│   ├── StrategyQA/
│   └── DU/
└── quanzhong/                         # [权重] 存放位置 (相对路径需固定)
    ├── gsm8K/step5000.bin
    ├── strageqa/step6800.bin
    ├── asdiv+du/step5800.bin
    └── aqua/step800.bin
🏋️ 3. 独立训练指令 (Qwen)
Qwen 使用 train_gspo_all_tasks.py 进行训练。 注意： 请将 --model_id 修改为您本地的 Qwen 模型绝对路径。

(1) 训练 GSM8K
Bash

nohup python train_gspo_all_tasks.py \
    --model_id "/path/to/your/Qwen2.5-7B-Instruct" \
    --data_path "./data/GSM8K" \
    --task_name "gsm8k" \
    --output_dir "./output_gsm8k" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --mini_batch_size 5 \
    --train_steps 7000 \
    --save_every 200 \
    --log_every 1 \
    --lr 1e-5 > train_gsm8k.log 2>&1 &
(2) 训练 StrategyQA
Bash

nohup python train_gspo_all_tasks.py \
    --model_id "/path/to/your/Qwen2.5-7B-Instruct" \
    --data_path "./data/StrategyQA" \
    --task_name "strategyqa" \
    --output_dir "./output_sqa" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --mini_batch_size 5 \
    --train_steps 7000 \
    --save_every 200 \
    --log_every 1 \
    --lr 1e-5 > train_sqa.log 2>&1 &
(3) 训练 AQuA
Bash

nohup python train_gspo_all_tasks.py \
    --model_id "/path/to/your/Qwen2.5-7B-Instruct" \
    --data_path "./data/AQuA" \
    --task_name "aqua" \
    --output_dir "./output_aqua" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --mini_batch_size 5 \
    --train_steps 7000 \
    --save_every 200 \
    --log_every 1 \
    --lr 1e-5 > train_aqua.log 2>&1 &
(4) 训练 ASDiv-Aug
Bash

nohup python train_gspo_all_tasks.py \
    --model_id "/path/to/your/Qwen2.5-7B-Instruct" \
    --data_path "./data/ASDiv-Aug" \
    --task_name "asdiv-aug" \
    --output_dir "./output_asdiv" \
    --path_to_projection_module "None" \
    --num_thought_tokens 2 \
    --group_size 5 \
    --mini_batch_size 5 \
    --train_steps 7000 \
    --save_every 200 \
    --log_every 1 \
    --lr 1e-5 > train_asdiv.log 2>&1 &
📊 4. 独立评测指令 (Qwen)
以下指令包含：自动创建脚本 -> 设置路径 -> 运行 Seed 41-45 -> 自动汇总结果。 请在终端直接复制粘贴运行。

(1) 评测 GSM8K (加载 Step 5000)
Bash

cat << 'EOF' > run_eval_qwen_gsm8k.sh
#!/bin/bash
# === 请修改此处路径 ===
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct"
# ======================
WEIGHT="./quanzhong/gsm8K/step5000.bin"
DATA="./data/GSM8K"
LOG_DIR="./logs_qwen_gsm8k"

mkdir -p $LOG_DIR
echo "Seed,Accuracy" > $LOG_DIR/summary.csv
for SEED in {41..45}; do
    python evaluate_unified.py --model_id "$MODEL_PATH" --task_name "gsm8k" --data_path "$DATA" \
    --params_file_name "$WEIGHT" --num_thought_tokens 2 --seed $SEED --test_k 0 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
    ACC=$(grep "Final Accuracy" "$LOG_DIR/seed_${SEED}.log" | tail -n 1 | awk -F'= ' '{print $2}' | sed 's/%//')
    echo "$SEED,$ACC" >> $LOG_DIR/summary.csv
done
python3 -c "import pandas as pd; df=pd.read_csv('$LOG_DIR/summary.csv'); print(f'Mean: {df.Accuracy.mean():.2f}%')"
EOF
chmod +x run_eval_qwen_gsm8k.sh && ./run_eval_qwen_gsm8k.sh
(2) 评测 StrategyQA (加载 Step 6800)
Bash

cat << 'EOF' > run_eval_qwen_sqa.sh
#!/bin/bash
# === 请修改此处路径 ===
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct"
# ======================
WEIGHT="./quanzhong/strageqa/step6800.bin"
DATA="./data/StrategyQA"
LOG_DIR="./logs_qwen_sqa"

mkdir -p $LOG_DIR
echo "Seed,Accuracy" > $LOG_DIR/summary.csv
for SEED in {41..45}; do
    python evaluate_unified.py --model_id "$MODEL_PATH" --task_name "strategyqa" --data_path "$DATA" \
    --params_file_name "$WEIGHT" --num_thought_tokens 2 --seed $SEED --test_k 0 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
    ACC=$(grep "Final Accuracy" "$LOG_DIR/seed_${SEED}.log" | tail -n 1 | awk -F'= ' '{print $2}' | sed 's/%//')
    echo "$SEED,$ACC" >> $LOG_DIR/summary.csv
done
python3 -c "import pandas as pd; df=pd.read_csv('$LOG_DIR/summary.csv'); print(f'Mean: {df.Accuracy.mean():.2f}%')"
EOF
chmod +x run_eval_qwen_sqa.sh && ./run_eval_qwen_sqa.sh
(3) 评测 ASDiv-Aug (加载 Step 5800)
Bash

cat << 'EOF' > run_eval_qwen_asdiv.sh
#!/bin/bash
# === 请修改此处路径 ===
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct"
# ======================
WEIGHT="./quanzhong/asdiv+du/step5800.bin"
DATA="./data/ASDiv-Aug"
LOG_DIR="./logs_qwen_asdiv"

mkdir -p $LOG_DIR
echo "Seed,Accuracy" > $LOG_DIR/summary.csv
for SEED in {41..45}; do
    python evaluate_unified.py --model_id "$MODEL_PATH" --task_name "asdiv-aug" --data_path "$DATA" \
    --params_file_name "$WEIGHT" --num_thought_tokens 2 --seed $SEED --test_k 0 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
    ACC=$(grep "Final Accuracy" "$LOG_DIR/seed_${SEED}.log" | tail -n 1 | awk -F'= ' '{print $2}' | sed 's/%//')
    echo "$SEED,$ACC" >> $LOG_DIR/summary.csv
done
python3 -c "import pandas as pd; df=pd.read_csv('$LOG_DIR/summary.csv'); print(f'Mean: {df.Accuracy.mean():.2f}%')"
EOF
chmod +x run_eval_qwen_asdiv.sh && ./run_eval_qwen_asdiv.sh
(4) 评测 DU (加载 Step 5800)
Bash

cat << 'EOF' > run_eval_qwen_du.sh
#!/bin/bash
# === 请修改此处路径 ===
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct"
# ======================
WEIGHT="./quanzhong/asdiv+du/step5800.bin"
DATA="./data/DU"
LOG_DIR="./logs_qwen_du"

mkdir -p $LOG_DIR
echo "Seed,Accuracy" > $LOG_DIR/summary.csv
for SEED in {41..45}; do
    python evaluate_unified.py --model_id "$MODEL_PATH" --task_name "du" --data_path "$DATA" \
    --params_file_name "$WEIGHT" --num_thought_tokens 2 --seed $SEED --test_k 0 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
    ACC=$(grep "Final Accuracy" "$LOG_DIR/seed_${SEED}.log" | tail -n 1 | awk -F'= ' '{print $2}' | sed 's/%//')
    echo "$SEED,$ACC" >> $LOG_DIR/summary.csv
done
python3 -c "import pandas as pd; df=pd.read_csv('$LOG_DIR/summary.csv'); print(f'Mean: {df.Accuracy.mean():.2f}%')"
EOF
chmod +x run_eval_qwen_du.sh && ./run_eval_qwen_du.sh
(5) 评测 AQuA (加载 Step 800)
Bash

cat << 'EOF' > run_eval_qwen_aqua.sh
#!/bin/bash
# === 请修改此处路径 ===
MODEL_PATH="/path/to/your/Qwen2.5-7B-Instruct"
# ======================
WEIGHT="./quanzhong/aqua/step800.bin"
DATA="./data/AQuA"
LOG_DIR="./logs_qwen_aqua"

mkdir -p $LOG_DIR
echo "Seed,Accuracy" > $LOG_DIR/summary.csv
for SEED in {41..45}; do
    python evaluate_unified.py --model_id "$MODEL_PATH" --task_name "aqua" --data_path "$DATA" \
    --params_file_name "$WEIGHT" --num_thought_tokens 2 --seed $SEED --test_k 0 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
    ACC=$(grep "Final Accuracy" "$LOG_DIR/seed_${SEED}.log" | tail -n 1 | awk -F'= ' '{print $2}' | sed 's/%//')
    echo "$SEED,$ACC" >> $LOG_DIR/summary.csv
done
python3 -c "import pandas as pd; df=pd.read_csv('$LOG_DIR/summary.csv'); print(f'Mean: {df.Accuracy.mean():.2f}%')"
EOF
chmod +x run_eval_qwen_aqua.sh && ./run_eval_qwen_aqua.sh