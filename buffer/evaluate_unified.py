import re
import argparse
import os
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

# 引入当前目录路径
sys.path.append(os.getcwd())

from unified_llm_model import UnifiedSoftCoT
from unified_utils import (
    pre_process_gsm8k_unified,
    pre_process_strategy_qa_unified,
    pre_process_aqua_unified,
    pre_process_du_unified
)
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--params_file_name', type=str, default=None)
    parser.add_argument('--num_thought_tokens', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--task_name', type=str, default='gsm8k', choices=['gsm8k', 'asdiv-aug', 'strategyqa', 'aqua', 'du'])
    parser.add_argument('--print_input', action='store_true', default=False)
    parser.add_argument('--print_response', action='store_true', default=False)
    
    parser.add_argument('--test_k', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tune_base_model', action='store_true', default=False)
    
    # [恢复] 允许用户选择跑 train / dev / test
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'dev', 'test'], help='Which split to evaluate on')
    
    return parser.parse_args()

# === 答案提取函数 ===
def extract_answer_math(response_text):
    cleaned_str = response_text.replace(',', '').replace('%', '').replace('$', '')
    match = re.findall(r'(-?[\d,]+(?:\.\d+)?)', cleaned_str)
    if match:
        try:
            val_str = match[-1].replace(',', '')
            return round(float(val_str), 2) if '.' in val_str else int(val_str)
        except: return None
    return None

def extract_answer_boolean(response_text):
    raw_lower = response_text.lower()
    rev_text = raw_lower[::-1]
    last_yes = re.search(r'\bsey\b', rev_text)
    idx_yes = last_yes.start() if last_yes else len(rev_text)
    last_no = re.search(r'\bon\b', rev_text)
    idx_no = last_no.start() if last_no else len(rev_text)
    if idx_yes == len(rev_text) and idx_no == len(rev_text): return None
    return 'Yes' if idx_yes < idx_no else 'No'

def extract_answer_option(response_text):
    rev_text = response_text.lower()[::-1]
    match = re.search(r'\b[a-f]\b', rev_text)
    if match: return match.group(0).upper()
    return None

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_id):
        raise ValueError(f"Model path does not exist: {args.model_id}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    logger.info(f"Loading model from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = UnifiedSoftCoT(
        model_id=args.model_id,
        num_thought_tokens=args.num_thought_tokens,
        tune_base_model=args.tune_base_model,
        path_to_projection_module=args.params_file_name,
    )
    model.eval()
    logger.info(f"Model loaded. N={args.num_thought_tokens}")
    
    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
        extract_fn = extract_answer_math
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
        extract_fn = extract_answer_math
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
        preprocess_fn = pre_process_strategy_qa_unified
        extract_fn = extract_answer_boolean
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
        preprocess_fn = pre_process_aqua_unified
        extract_fn = extract_answer_option
    elif args.task_name == 'du':
        db = DULoader().load(args.data_path)
        preprocess_fn = pre_process_du_unified
        extract_fn = extract_answer_option
    else:
        raise NotImplementedError
    
    # [恢复] 使用参数控制读取哪个 split
    if args.dataset_split not in db.datasets:
        logger.error(f"Split {args.dataset_split} not found in dataset! Available: {list(db.datasets.keys())}")
        return
        
    ds = db.get_dataset(args.dataset_split)
    logger.info(f"🚀 Evaluating on split: {args.dataset_split} (Size: {len(ds)})")
    
    if args.test_k > 0:
        ds = ds[:args.test_k]
    
    generation_config = GenerationConfig.from_pretrained(args.model_id)
    # [CRITICAL FIX] Llama Pad Token
    if 'llama' in args.model_id.lower():
        generation_config.pad_token_id = 128009 
    else:
        generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
        
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.top_p = 1.0
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = 1024
    generation_config.do_sample = True 
    
    correct_count = 0
    
    for idx, ins in enumerate(tqdm(ds, desc=f'Eval {args.dataset_split}', ncols=100)):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        raw_gt = None
        if 'correct' in ins: raw_gt = ins['correct']
        elif 'answer' in ins: raw_gt = ins['answer']
        
        gt_val = None
        if args.task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if isinstance(raw_gt, str):
                    ans_part = raw_gt.split('\n')[-1].replace(',', '').replace('####', '').strip()
                    gt_val = float(ans_part) if '.' in ans_part else int(ans_part)
            except: gt_val = None
        elif args.task_name == 'strategyqa':
            # StrategyQA GT处理增强
            if isinstance(raw_gt, bool): gt_val = 'Yes' if raw_gt else 'No'
            else: gt_val = str(raw_gt)
        elif args.task_name in ['aqua', 'du']:
            try:
                if isinstance(raw_gt, str) and '####' in raw_gt:
                    gt_val = raw_gt.split('####')[-1].strip()
                else:
                    gt_val = raw_gt.strip() if isinstance(raw_gt, str) else None
            except: gt_val = None

        inputs = preprocess_fn(
            ins, tokenizer, num_thought_tokens=args.num_thought_tokens,
            split='test', device=model.device,
        )
        
        if args.print_input:
            logger.info(f'Decoded Prompt: {tokenizer.decode(inputs["input_ids"][0])}')
        
        if args.num_thought_tokens > 0:
            inputs_embeds = model.get_inputs_embeds_for_unified_model(
                inputs['input_ids'], inputs['attention_mask'], inputs['thought_index'], args.print_input,
            )
        else:
            inputs_embeds = model.model.get_input_embeddings()(inputs['input_ids'])
            
        with torch.no_grad():
            outputs = model.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config,
                num_return_sequences=args.num_return_sequences,
                use_cache=True
            )
        
        response = outputs[0]
        full_text = tokenizer.decode(response, skip_special_tokens=True)
        
        if args.print_response:
            logger.info(f'Full Response: {full_text}')

        pred_val = extract_fn(full_text)
        
        is_correct = False
        if pred_val is not None and gt_val is not None:
            if args.task_name in ['gsm8k', 'asdiv-aug']:
                if abs(pred_val - gt_val) < 1e-4: is_correct = True
            else:
                if str(pred_val).upper() == str(gt_val).upper(): is_correct = True
        
        if is_correct: correct_count += 1
        
        if not args.print_input and not args.print_response:
             logger.info(f"[{idx + 1}] GT: {gt_val} | Pred: {pred_val} | Acc: {correct_count / (idx + 1) * 100:.2f}%")
    
    acc = correct_count / len(ds) * 100
    logger.info(f'Final Accuracy on {args.dataset_split}: {correct_count}/{len(ds)} = {acc:.2f}%')

if __name__ == '__main__':
    main()