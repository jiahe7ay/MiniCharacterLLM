# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
from pathlib import Path
import sys

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from conversation import get_conv_template


from functools import partial
from datasets import Dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    conv_template: str = field(default = "vicuna-1.1")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(
        default = None
    )
    mask_user: bool = field(
        default = True    
    )


local_rank = None



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
        sample,
        conv_template,
        max_length,
        tokenizer,
        mask_user = True,
    ) -> Dict:

  
        source = sample["conversations"]


        conv = get_conv_template(conv_template)
        # scores roles =  query score
        roles = {"user": conv.roles[0], "llm": conv.roles[1]}

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []

        
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            assert role == conv.roles[j % 2], breakpoint()
            #    self.messages.append([role, message])
            conv.append_message(role, sentence["value"])
#       SeparatorStyle.SCORER
        conversation = conv.get_prompt()
        # query:xx score:xx 

        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            max_length=max_length, 
            padding=True,
            truncation=True,
        ).input_ids
        
        input_ids = input_ids.flatten()
        
        if mask_user:
        
            targets = input_ids.clone()

            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                    
                role = roles[sentence["from"]]
                # assert role == conv.roles[j % 2], f"{i}"
                #如果当前对话不符合他的角色就退出
                assert role == conv.roles[j % 2], breakpoint()
                
                if role == conv.roles[1]:
                    #如果角色是gpt就直接添加
                    conv.append_message(role, sentence["value"])
                
                if role != conv.roles[1]:
                    if j == 0:
                        #如果human在开头的话，对话起始的index就是0
                        conv_start_idx = 0
                    else:
                        #获取前面的对话的长度
                        conv_last = conv.get_prompt()
                        #根据前面的长度，获取当前huamn对话的起始index
                        conv_start_idx = tokenizer(
                            conv_last,
                            return_tensors="pt",
                            max_length=max_length,
                            padding=True,
                            truncation=True,
                        ).input_ids.shape[1]
                    #将human的对话放入到message list  
                    conv.append_message(role, sentence["value"])
                    #将human的对话拼接成prompt 
                    conv_so_far = conv.get_prompt()
                    #获取human对话结束的长度(如果不是第一轮对话，这个长度包括前面的对话)，就是结束Index
                    conv_end_idx = tokenizer(
                        conv_so_far,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length, 
                    ).input_ids.shape[1]
                    
                    # conv_end_idx -= 1   # hard offset for llama model
                    #将human对话片段不算loss
                    targets[conv_start_idx:conv_end_idx] = IGNORE_TOKEN_ID
                    
                    if conv_end_idx >= max_length:
                        break
        
        attention_mask = torch.ones_like(input_ids)
        
        return dict(
            input_ids=input_ids,
            labels = targets,
            attention_mask = attention_mask,
        )


def get_datasets(data, preprocess_func, num_proc):

    
    conversations = [{"conversations": item["cov"]} for item in data]
    
    raw_dataset = Dataset.from_list(conversations)

    tokenized_datasets = raw_dataset.map(
        preprocess_func,
        batched = False,
        num_proc = num_proc,
        remove_columns = ["conversations"],
        desc = "Tokenizing and reformatting instruction data"
    )  
    
    return tokenized_datasets


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, max_length, fwd_batch_size, data_args, mask_user = True
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    conv_template = data_args.conv_template
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    preprocess_func = partial(preprocess, 
                            conv_template = conv_template,
                            max_length = max_length,
                            tokenizer = tokenizer,
                            mask_user = mask_user)
    
    train_dataset = get_datasets(raw_data, preprocess_func, 12)

    return dict(train_dataset=train_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank
    world_size = training_args.world_size
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        #padding=True,
        truncation=True,
        use_fast=False,
        trust_remote_code=True

    )
    tokenizer.pad_token_id = tokenizer.im_end_id
    


    
    fwd_batch_size = training_args.per_device_train_batch_size * world_size
    
    train_dataset = make_supervised_data_module(
                                              tokenizer=tokenizer,
                                              max_length = training_args.model_max_length,
                                              fwd_batch_size = fwd_batch_size,
                                              data_args=data_args,
                                              mask_user = training_args.mask_user)
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **train_dataset
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir = training_args.output_dir)


if __name__ == "__main__":
    train()
