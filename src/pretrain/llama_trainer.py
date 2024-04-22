'''
author:        zhangjl19 <zhangjl19@spdb.com.cn>
date:          2024-03-11 10:02:44
'''

import sys
import os
import random

import pathlib
import logging
from transformers import HfArgumentParser
# from transformers import LlamaForCausalLM, LlamaConfig
import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import IterableDataset

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cur_path, '../../'))

from src.utils.arg_parsers import ModelArguments, DataArguments, TrainingArguments
from src.utils.logging import log_dist
from src.utils.dist_utils import is_rank_0
from src.tokenizer.my_tokenizer import tokenizer
from model.modeling_llama2 import LlamaConfig, LlamaForCausalLM
from src.dataset.data_iter import create_shard_kwargs, create_data_iter
from src.dataset.pretrain_dataset import preprocess_the_pile_gen, \
    preprocess_wudao_gen, pretrain_collate_fn_gen

def init_env(training_args:TrainingArguments):
    ################################
    ###### Create Exp. Env #########
    ################################
    if training_args.checkpoint_dir:
        log_dist("Creating Training Env",
                 ranks=[0],
                 level=logging.INFO)
        # checkpoint_dir = pathlib.Path(checkpoint_dir)
        pathlib.Path(training_args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        exp_dir = os.path.join(training_args.checkpoint_dir, 'exp')
        if not is_rank_0():
            return None, exp_dir
        pathlib.Path(exp_dir).mkdir(exist_ok=True)
    else:
        log_dist("checkpoint_dir not exists ",
                 ranks=[0],
                 level=logging.WARNING)
        
    # Tensorboard writer
    if is_rank_0():
        tb_dir = os.path.join(training_args.checkpoint_dir, "tb_dir")
        pathlib.Path(tb_dir).mkdir(exist_ok=True)
        summary_writer = SummaryWriter(log_dir=tb_dir)
    
    log_dist("Training Env Created",
                 ranks=[0],
                 level=logging.INFO)
    return summary_writer, exp_dir

def get_data_loader(data_args:DataArguments, training_args: TrainingArguments, model_engine,  ds_config):
    ################################
    ###### Create Datasets #########
    ################################
    log_dist("Creating Datasets", ranks=[0], level=logging.INFO)

    paths = create_shard_kwargs([data_args.train_datas_path_pattern])
    log_dist(f"read total {len(paths)} for train", ranks=[0], level=logging.INFO)
    
    random.shuffle(paths)
    transform_dict = {
        'wudao': preprocess_wudao_gen(tokenizer, training_args.max_length), 
        # 'pile': preprocess_the_pile_gen(tokenizer, max_length)
    }
    data_set = IterableDataset.from_generator(create_data_iter, gen_kwargs={
        'paths': paths, 
        'transform_dict': transform_dict,
        'process_index': model_engine.global_rank,
        'num_processes': model_engine.world_size
    })
    
    train_batch_size = ds_config['train_micro_batch_size_per_gpu'] *  model_engine.world_size * ds_config['gradient_accumulation_steps']
    train_loader = DataLoader(data_set, batch_size=train_batch_size, num_workers=model_engine.world_size, 
        collate_fn=pretrain_collate_fn_gen(tokenizer, training_args.max_length), drop_last=True)
    log_dist("Dataset Creation Done", ranks=[0], level=logging.INFO)
    
    return train_loader

def get_tokenizer():
    return tokenizer

def get_model(model_args:ModelArguments, tokenizer):
    ################################
    ###### Create Model ############
    ################################
    log_dist("Creating Model", ranks=[0], level=logging.INFO)
    model =  LlamaForCausalLM(LlamaConfig(vocab_size=tokenizer.vocab_size, 
                                         initializer_range=1e-2, 
                                         pad_token_id=tokenizer.pad_id, 
                                         hidden_size=1024,
                                         num_hidden_layers=8,
                                         num_attention_heads=8,
                                         rms_norm_eps=1e-5, 
                                         hidden_dropout_prob=0.1, 
                                         attention_dropout_prob=0.1, 
                                         use_stable_embedding=True, 
                                         shared_input_output_embedding=True))
    log_dist(
        f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}",
        ranks=[0],
        level=logging.INFO)
        
    log_dist("Model Creation Done", ranks=[0], level=logging.INFO)
    return model


def init_ds_engine(model, ds_config):
    ################################
    ###### DeepSpeed engine ########
    ################################
    log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)

    model_engine, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config,
                                          dist_init_required=True)
    log_dist("DeepSpeed engine created", ranks=[0], level=logging.INFO)
    return model_engine

def train(model_args:ModelArguments, data_args:DataArguments, training_args:TrainingArguments):
    device = (torch.device(get_accelerator().device_name()) if get_accelerator().is_available() else torch.device("cpu"))
    with open(training_args.ds_config, "r", encoding="utf-8") as jf:
        ds_config_obj = json.load(jf)
    
    summary_writer, exp_dir = init_env(training_args)
    model = get_model(model_args, tokenizer)   
    model_engine = init_ds_engine(model, ds_config_obj)
    train_loader = get_data_loader(data_args, training_args, model_engine, ds_config_obj)
    
    ################################
    ####### The Training Loop ######
    ################################
    model_engine.train()
    losses = []
    for step, batch in enumerate(train_loader, start=1):
        step = step * model_engine.world_size
        if step >= training_args.num_iterations:
            break
        
        for k, v in batch.items():
            batch[k] = v.to(device)
        labels = batch['input_ids'].clone()
        labels[labels==tokenizer.pad_id] = -100
        out = model_engine(**batch, labels=labels)
        total_loss = out.loss
        # Backward pass
        model_engine.backward(total_loss)
        # Optimizer Step
        model_engine.step()
        losses.append(total_loss.item())
        
        if step % training_args.log_steps == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                     ranks=[0],
                     level=logging.INFO)
            if is_rank_0():
                summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        if step % training_args.check_point_steps == 0:
            model_engine.save_checkpoint(save_dir=exp_dir,
                                  client_state={'checkpoint_step': step})
            log_dist("Saved model to {0}".format(exp_dir),
                     ranks=[0],
                     level=logging.INFO)
            
    # Save the last checkpoint if not saved yet
    if step % training_args.check_point_steps != 0:
        model_engine.save_checkpoint(save_dir=exp_dir,
                              client_state={'checkpoint_step': step})
        log_dist("Saved model to {0}".format(exp_dir),
                 ranks=[0],
                 level=logging.INFO)
        
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
    

if __name__ == '__main__':
    main()