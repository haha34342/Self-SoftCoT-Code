"""
Unified SoftCoT Model - Independent Projection Version
配套 "only <box_start>" 创新点使用
修改1: Dropout 设为 0.0 以稳定训练
修改2: 初始化改为全 0 (Zero Init) 以复现高分权重策略
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache
from fastNLP import logger


class UnifiedSoftCoT(nn.Module):
    
    def __init__(
        self,
        model_id,
        num_thought_tokens=4,
        tune_base_model=False,
        path_to_projection_module=None,
        device_map='auto',
        **kwargs,
    ):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            _fast_init=False,
        )
        
        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.num_thought_tokens = num_thought_tokens
        self.tune_base_model = tune_base_model
        
        # === 独立投影层 ===
        if num_thought_tokens > 0:
            # 创建 N 个独立的 Linear 层
            self.projections = nn.ModuleList([
                nn.Linear(
                    self.model.config.hidden_size, 
                    self.model.config.hidden_size,
                    dtype=torch.bfloat16
                ) for _ in range(num_thought_tokens)
            ])
            
            # [Modified] 彻底关闭 Dropout
            self.dropout = nn.Dropout(0.0)
            
            # [Modified] 初始化为全 0 映射 (Zero Initialization)
            # 这样初始输出为 0 向量，对 Base Model 干扰最小
            with torch.no_grad():
                for proj in self.projections:
                    proj.weight.data.zero_()
                    proj.bias.data.zero_()
        else:
            self.projections = nn.ModuleList([])
            self.dropout = nn.Dropout(0.0)

        for n, p in self.model.named_parameters():
            p.requires_grad = tune_base_model
        
        # 加载权重 (适配 ModuleList)
        if path_to_projection_module is not None and path_to_projection_module not in ['None']:
            try:
                state_dict = torch.load(path_to_projection_module, map_location='cpu', weights_only=True)
                self.projections.load_state_dict(state_dict)
                logger.info(f'Loaded weights from `{path_to_projection_module}`.')
            except Exception as e:
                logger.warning(f'Failed to load weights: {e}. Using zero initialization.')
        
        self.projections.to(self.model.device)

    @property
    def device(self):
        return self.model.device

    def save_pretrained(self, save_model_dir_root: str, **kwargs):
        os.makedirs(save_model_dir_root, exist_ok=True)
        projection_file = os.path.join(save_model_dir_root, 'projection.bin')
        torch.save(self.projections.state_dict(), projection_file)
        logger.info(f'Saved projection module to `{projection_file}`.')

    def get_inputs_embeds_for_unified_model(
        self,
        input_ids,
        attention_mask,
        thought_index,
        print_index=False,
    ):
        batch_size, seq_len = input_ids.size()
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
        
        for b in range(batch_size):
            s_idx = thought_index[b, 0].item()
            e_idx = thought_index[b, 1].item()
            
            soft_thoughts_raw = hidden_states[b, s_idx:e_idx]
            
            if soft_thoughts_raw.size(0) != self.num_thought_tokens:
                continue

            # 逐个 Token 应用对应的投影层
            projected_list = []
            for i in range(self.num_thought_tokens):
                token_vec = soft_thoughts_raw[i] 
                token_vec = self.dropout(token_vec)
                proj_vec = self.projections[i](token_vec) 
                projected_list.append(proj_vec)
            
            projected_thoughts = torch.stack(projected_list)
            inputs_embeds[b, s_idx:e_idx] = projected_thoughts
            
            if print_index:
                logger.info(f'Processed soft thoughts at index {s_idx}-{e_idx}')
        
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        thought_index: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        print_index=False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_len = input_ids.size()
        
        if seq_len > 1 and self.num_thought_tokens > 0:
            inputs_embeds = self.get_inputs_embeds_for_unified_model(
                input_ids, attention_mask, thought_index, print_index,
            )
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        
        return outputs
