#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSPO Training Script - Multi-Task Parallel Version (Fixed Instance.get error)
适配: GSM8K, AQuA, DU, StrategyQA
"""
import os
import re
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm
import sys

# 导入路径
sys.path.append(os.getcwd())
# 导入所有 Loader
from data_loader import GSM8KLoader, AQuALoader, DULoader, StrategyQALoader, AugASDivLoader
from unified_llm_model import UnifiedSoftCoT
# 导入所有 Preprocessors
from unified_utils import (
    pre_process_gsm8k_unified,
    pre_process_aqua_unified,
    pre_process_du_unified,
    pre_process_strategy_qa_unified
)

def enforce_eager_backend():
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def seed_init(s: int):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# === 多任务答案提取 ===
def extract_answer(text: str, task_name: str):
    text = text.strip()
    # 1. Math (GSM8K, ASDiv)
    if task_name in ['gsm8k', 'asdiv-aug']:
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        t = match.group(1) if match else text
        t = t.replace(",", "").replace("%", "").replace("$", "")
        # 匹配数字 (含负数)
        m = re.findall(r"([-+]?\d+(?:\.\d+)?)", t)
        if not m: return None
        try: return int(m[-1]) if "." not in m[-1] else round(float(m[-1]), 2)
        except: return None
        
    # 2. Option (AQuA, DU)
    elif task_name in ['aqua', 'du']:
        # 优先看 Box
        match = re.search(r'\\boxed\{([A-Fa-f])\}', text)
        if match: return match.group(1).upper()
        # 否则反向找单个字母
        rev_text = text.lower()[::-1]
        m = re.search(r'\b[a-f]\b', rev_text)
        if m: return m.group(0).upper()
        return None
        
    # 3. Boolean (StrategyQA)
    elif task_name == 'strategyqa':
        raw_lower = text.lower()
        rev_text = raw_lower[::-1]
        last_yes = re.search(r'\bsey\b', rev_text)
        idx_yes = last_yes.start() if last_yes else 999999
        last_no = re.search(r'\bon\b', rev_text)
        idx_no = last_no.start() if last_no else 999999
        
        if idx_yes == 999999 and idx_no == 999999: return None
        return 'Yes' if idx_yes < idx_no else 'No'
    
    return None

def compute_reward(pred, gt, task_name):
    if pred is None or gt is None:
        return 0.0
    
    if task_name in ['gsm8k', 'asdiv-aug']:
        # Math: Float Tolerance
        return 1.0 if abs(pred - gt) < 1e-4 else 0.0
    else:
        # Option/Boolean: String Match
        return 1.0 if str(pred).upper() == str(gt).upper() else 0.0

def build_prompt_embeddings_indep(model, input_ids, attention_mask, thought_index, use_projection_grad=False):
    with torch.no_grad():
        base_embeds = model.model.get_input_embeddings()(input_ids)
        outputs_p1 = model.model(inputs_embeds=base_embeds, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = outputs_p1.hidden_states[-1]

    inputs_embeds_p2 = base_embeds.clone()
    batch_size = input_ids.size(0)
    
    for b in range(batch_size):
        s_idx = thought_index[b, 0].item()
        e_idx = thought_index[b, 1].item()
        
        soft_thoughts_raw = hidden_states[b, s_idx:e_idx]
        
        if soft_thoughts_raw.size(0) != model.num_thought_tokens:
            continue

        projected_list = []
        for i in range(model.num_thought_tokens):
            token_vec = soft_thoughts_raw[i]
            if use_projection_grad:
                token_vec = model.dropout(token_vec)
                proj_vec = model.projections[i](token_vec)
            else:
                with torch.no_grad():
                    token_vec = model.dropout(token_vec) 
                    proj_vec = model.projections[i](token_vec)
            projected_list.append(proj_vec)
            
        projected_thoughts = torch.stack(projected_list)
        inputs_embeds_p2[b, s_idx:e_idx] = projected_thoughts
        
    return inputs_embeds_p2

def get_logprobs(model, input_ids, attention_mask, thought_index, response_ids, use_grad=False):
    prompt_embeds = build_prompt_embeddings_indep(model, input_ids, attention_mask, thought_index, use_projection_grad=use_grad)
    resp_embeds = model.model.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([prompt_embeds, resp_embeds], dim=1)
    
    resp_mask = torch.ones_like(response_ids)
    full_mask = torch.cat([attention_mask, resp_mask], dim=1)
    
    if use_grad: 
        full_embeds.requires_grad_(True) 
        outputs = model.model(inputs_embeds=full_embeds, attention_mask=full_mask)
    else:
        with torch.no_grad(): 
            outputs = model.model(inputs_embeds=full_embeds, attention_mask=full_mask)
            
    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len-1 : -1, :] 
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--path_to_projection_module", default="None")
    ap.add_argument("--task_name", type=str, required=True, choices=['gsm8k', 'asdiv-aug', 'aqua', 'du', 'strategyqa'])
    ap.add_argument("--num_thought_tokens", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=5) 
    ap.add_argument("--epsilon", type=float, default=3e-4)
    ap.add_argument("--train_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5) 
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--update_old_every", type=int, default=4) 
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mini_batch_size", type=int, default=5)
    return ap.parse_args()

def main():
    enforce_eager_backend()
    args = parse_args()
    seed_init(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    flog = open(os.path.join(args.output_dir, "logs", "train_stream.log"), "a", buffering=1)
    def log_msg(msg):
        try:
            tqdm.write(msg)
            flog.write(msg + "\n")
            flog.flush()
        except: pass

    log_msg(f"🔧 Config: Task={args.task_name} | N={args.num_thought_tokens} | G={args.group_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = UnifiedSoftCoT(args.model_id, args.num_thought_tokens, path_to_projection_module=args.path_to_projection_module)
    model.eval()
    dev = model.device
    
    opt = torch.optim.AdamW(model.projections.parameters(), lr=args.lr)
    
    proj_old = deepcopy(model.projections)
    for p in proj_old.parameters(): p.requires_grad = False
    
    # === 1. 根据任务加载数据和选择预处理 ===
    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
        preprocess_fn = pre_process_aqua_unified
    elif args.task_name == 'du':
        db = DULoader().load(args.data_path)
        preprocess_fn = pre_process_du_unified
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
        preprocess_fn = pre_process_strategy_qa_unified
    else:
        raise ValueError(f"Unknown task: {args.task_name}")

    train_ds = db.get_dataset("train")
    log_msg(f"Loaded {len(train_ds)} training samples from {args.task_name}.")
    
    pbar = tqdm(range(args.train_steps), ascii=True, ncols=100)
    
    gen_config = GenerationConfig(
        do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens, 
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 151643, 
        eos_token_id=tokenizer.eos_token_id, use_cache=True,
        top_p=1.0 
    )
    
    original_state = None
    is_swapped = False

    try:
        for step in pbar:
            opt.zero_grad()
            
            if step % args.update_old_every == 0:
                proj_old.load_state_dict(model.projections.state_dict())
            
            data_idx = step % len(train_ds)
            ins = train_ds[data_idx]
            
            # === 2. GT 解析 (兼容 Instance 对象, 修复 .get 错误) ===
            gt_val = None
            if args.task_name in ['gsm8k', 'asdiv-aug']:
                try:
                    txt = ins['answer'].split('\n')[-1].replace(',', '').replace('####', '').strip()
                    gt_val = float(txt) if '.' in txt else int(txt)
                except: pass
            elif args.task_name == 'strategyqa':
                # StrategyQA
                if 'answer' in ins:
                    raw = ins['answer']
                    if isinstance(raw, bool): gt_val = 'Yes' if raw else 'No'
                    else: gt_val = str(raw)
            elif args.task_name in ['aqua', 'du']:
                # AQuA/DU 优先找 correct，其次 answer
                raw = None
                if 'correct' in ins:
                    raw = ins['correct']
                elif 'answer' in ins:
                    raw = ins['answer']
                
                if raw:
                    if '####' in raw: gt_val = raw.split('####')[-1].strip()
                    else: gt_val = raw.strip()

            processed = preprocess_fn(ins, tokenizer, args.num_thought_tokens, device=dev, split='test')
            input_ids, attention_mask, thought_index = processed['input_ids'], processed['attention_mask'], processed['thought_index']
            
            # === Parallel Rollout ===
            G = args.group_size
            input_ids_g = input_ids.repeat(G, 1)
            attention_mask_g = attention_mask.repeat(G, 1)
            thought_index_g = thought_index.repeat(G, 1)
            
            model.model.gradient_checkpointing_disable()
            model.model.config.use_cache = True
            model.eval() 
            
            original_state = deepcopy(model.projections.state_dict())
            model.projections.load_state_dict(proj_old.state_dict())
            is_swapped = True
            
            with torch.no_grad():
                prompt_embeds_g = build_prompt_embeddings_indep(model, input_ids_g, attention_mask_g, thought_index_g, False)
                gen_output_batch = model.model.generate(
                    inputs_embeds=prompt_embeds_g, 
                    attention_mask=attention_mask_g, 
                    generation_config=gen_config
                )
            
            model.projections.load_state_dict(original_state)
            is_swapped = False
            
            # ============================================================
            # [CRITICAL FIX]: generate with inputs_embeds returns ONLY new tokens.
            # Do NOT slice!
            # ============================================================
            response_ids = gen_output_batch
            
            # === 3. Reward 计算 ===
            rewards = []
            decoded_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            for txt in decoded_texts:
                pred = extract_answer(txt, args.task_name)
                r = compute_reward(pred, gt_val, args.task_name)
                rewards.append(r)
            
            r_tensor = torch.tensor(rewards, device=dev)
            r_mean, r_std = r_tensor.mean(), r_tensor.std() + 1e-8
            
            if r_std < 1e-6:
                pbar.set_description(f"Step {step+1}: R={r_mean:.2f} (Skip)")
                # log_msg(f"[Step {step+1}] Skip due to low variance (R={r_mean})") 
            else:
                # === Training Phase ===
                model.train()
                model.model.gradient_checkpointing_enable()
                model.model.config.use_cache = False
                
                adv = (r_tensor - r_mean) / r_std
                
                model.projections.load_state_dict(proj_old.state_dict())
                is_swapped = True
                with torch.no_grad(): 
                    logp_old = get_logprobs(model, input_ids_g, attention_mask_g, thought_index_g, response_ids)
                model.projections.load_state_dict(original_state)
                is_swapped = False
                
                mini_bs = args.mini_batch_size
                total_loss_val = 0.0
                approx_kl_val = 0.0
                
                for i in range(0, G, mini_bs):
                    end_idx = min(i + mini_bs, G)
                    sub_sl = slice(i, end_idx)
                    
                    sub_logp_new = get_logprobs(model, input_ids_g[sub_sl], attention_mask_g[sub_sl], thought_index_g[sub_sl], response_ids[sub_sl], use_grad=True)
                    
                    sub_logp_old_slice = logp_old[sub_sl]
                    sub_adv_slice = adv[sub_sl]
                    
                    mask = (response_ids[sub_sl] != tokenizer.pad_token_id).float()
                    log_diff = sub_logp_new - sub_logp_old_slice
                    
                    with torch.no_grad():
                        sub_kl = 0.5 * ((log_diff * mask) ** 2).sum() / (mask.sum() + 1e-8)
                        approx_kl_val += sub_kl.item() * (end_idx - i)

                    ratio = torch.exp((log_diff * mask).sum(1) / (mask.sum(1) + 1e-8))
                    loss = -torch.min(ratio * sub_adv_slice, torch.clamp(ratio, 1-args.epsilon, 1+args.epsilon) * sub_adv_slice).mean()
                    
                    loss_backward = loss / (G / (end_idx - i))
                    loss_backward.backward()
                    total_loss_val += loss.item() * (end_idx - i)
                
                total_loss_val /= G
                approx_kl_val /= G
                opt.step()
                
                pbar.set_description(f"Step {step+1}: R={r_mean:.2f} L={total_loss_val:.4f}")
                if (step+1) % args.log_every == 0:
                    log_msg(f"[UPDATE] Step={step+1} Task={args.task_name} Loss={total_loss_val:.4f} Reward={r_mean:.2f} KL={approx_kl_val:.4f}")

            if (step+1) % args.save_every == 0:
                save_path = os.path.join(args.output_dir, "ckpt", f"step{step+1}.bin")
                log_msg(f"💾 Saving checkpoint to {save_path} ...")
                torch.save(model.projections.state_dict(), save_path)

    except KeyboardInterrupt:
        save_path = os.path.join(args.output_dir, "ckpt", "interrupted.bin")
        log_msg(f"\n🛑 Interrupted! Checking weights...")
        if is_swapped and original_state is not None:
            model.projections.load_state_dict(original_state)
        try:
            torch.save(model.projections.state_dict(), save_path)
        except: pass
        sys.exit(0)

    torch.save(model.projections.state_dict(), os.path.join(args.output_dir, "ckpt", "final.bin"))
    flog.close()

if __name__ == "__main__":
    main()