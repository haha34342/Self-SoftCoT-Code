from typing import List, Dict, Any
import torch
import re
from fastNLP import logger

print("DEBUG: Loaded Unified Utils V4 (Full Multi-Task Support)")

def get_soft_thoughts(num_thought_tokens):
    return '<|box_start|>' + '<|endoftext|>' * num_thought_tokens + '<|box_end|>'

def _split_question_options(instance):
    """
    智能分离问题和选项。
    返回: (clean_question, options_formatted_string)
    """
    raw_q = instance['question']
    options_str = ""

    # 1. 尝试从字符串中分割 Options
    if "Options:" in raw_q:
        parts = raw_q.split("Options:")
        clean_q = parts[0].strip()
        # 保留 Options: 前缀，并确保格式整洁
        options_str = "Options:" + parts[1]
    else:
        clean_q = raw_q.strip()
        # 2. 如果字符串没 Options，尝试从 instance['options'] 列表获取
        if 'options' in instance:
            options_str = "Options:\n" + "\n".join(instance['options'])
    
    return clean_q, options_str

def _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len):
    """通用打包函数"""
    input_messages = [{'role': 'user', 'content': input_content}]
    
    input_ids = tokenizer.apply_chat_template(input_messages, tokenize=True)
    if max_len > 0 and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    
    attention_mask = [1] * len(input_ids)
    
    if num_thought_tokens > 0:
        box_start_tokens = tokenizer.encode('<|box_start|>', add_special_tokens=False)
        box_end_tokens = tokenizer.encode('<|box_end|>', add_special_tokens=False)
        
        box_start_id = box_start_tokens[0] if box_start_tokens else None
        box_end_id = box_end_tokens[0] if box_end_tokens else None
        
        input_ids_tensor = torch.tensor(input_ids)
        
        if box_start_id is not None and box_end_id is not None:
            box_start_positions = (input_ids_tensor == box_start_id).nonzero(as_tuple=True)[0]
            box_end_positions = (input_ids_tensor == box_end_id).nonzero(as_tuple=True)[0]
            
            if len(box_start_positions) > 0 and len(box_end_positions) > 0:
                # 寻找最后一个 Box
                thought_start_idx = box_start_positions[-1].item() + 1
                thought_end_idx = box_end_positions[-1].item()
            else:
                # logger.warning(f"Special tokens not found. Fallback.")
                thought_start_idx = 0
                thought_end_idx = 0
        else:
            thought_start_idx = 0
            thought_end_idx = 0
    else:
        thought_start_idx = 0
        thought_end_idx = 0
    
    labels = [-100] * len(input_ids)
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'thought_index': [thought_start_idx, thought_end_idx, 0, 0],
    }
    
    # 兼容 AQuA/DU 的 GT 字段
    if 'answer' in instance:
        inputs['answer'] = instance['answer']
    elif 'correct' in instance:
        inputs['answer'] = instance['correct']
        
    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, List) else v
            for k, v in inputs.items()
        }
    
    return inputs

# === 1. GSM8K / ASDiv 预处理 ===
def pre_process_gsm8k_unified(instance, tokenizer, num_thought_tokens=4, device=None, split='train', max_len=-1, **kwargs):
    reasoning_list = instance['answer'].split('\n')
    answer = reasoning_list[-1]
    question = instance['question']
    
    input_template = (
        f'Solve the following math problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief calculations]\n'
        f'## Step 2: [Brief calculations]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is just the final number or expression that solves the problem.\n\n'
        f'Problem: {question}'
    )

    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens)
        input_content = f'{input_template}{soft_thoughts}\n\n'
    else:
        input_content = f'{input_template}\n\n'
    
    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

# === 2. StrategyQA 预处理 ===
def pre_process_strategy_qa_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    question = instance['question']
    
    # 策略QA不需要处理选项
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens)
        input_content = (
            f'You are required to answer the following question with `Yes` or `No`: {question}{soft_thoughts}\n\n'
            f'Therefore, the final answer is `Yes` or `No`?'
        )
    else:
        input_content = (
            f'You are required to answer the following question with `Yes` or `No`: {question}\n\n'
            f'Therefore, the final answer is `Yes` or `No`?'
        )

    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

# === 3. AQuA 预处理 ===
def pre_process_aqua_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    # 分离问题和选项
    clean_q, options_str = _split_question_options(instance)
    
    input_template_head = (
        f'You are required to solve the following math multiple choices questions.\n'
        f'In the multiple choices question, there are five options: A, B, C, D, and E, respectively.\n'
        f'The correct answer that solves the problem is one of these options.\n'
        f'Your job is to solve the problem and find the correct option.\n'
        f'Here are the instructions for solving the problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief calculations]\n'
        f'## Step 2: [Brief calculations]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with the following sentence:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is the option from A, B, C, D, and E.\n'
        f'Only one letter from A to E is accepted in the answer span.\n\n'
        f'Problem: {clean_q}'
    )

    # 1. 拼接 Soft Tokens (紧跟纯问题)
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens)
        input_content = f'{input_template_head}{soft_thoughts}'
    else:
        input_content = f'{input_template_head}'

    # 2. 拼接 Options (如果存在)
    if options_str:
        input_content += '\n' + options_str

    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

# === 4. DU (Date Understanding) 预处理 ===
def pre_process_du_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    # 分离问题和选项
    clean_q, options_str = _split_question_options(instance)
    
    input_template_head = (
        f'You are required to solve the following math multiple choices questions.\n'
        f'In the multiple choices question, there are five options: A, B, C, D, E, and F, respectively.\n'
        f'The correct answer that solves the problem is one of these options.\n'
        f'Your job is to solve the problem and find the correct option.\n'
        f'Here are the instructions for solving the problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal description.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief reasoning step]\n'
        f'## Step 2: [Brief reasoning step]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with the following sentence:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is the option from A, B, C, D, E, and F.\n'
        f'Only one letter from A to F is accepted in the answer span.\n\n'
        f'Problem: {clean_q}'
    )

    # 1. 拼接 Soft Tokens (紧跟纯问题)
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens)
        input_content = f'{input_template_head}{soft_thoughts}'
    else:
        input_content = f'{input_template_head}'

    # 2. 拼接 Options (如果存在)
    if options_str:
        input_content += '\n' + options_str

    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)
