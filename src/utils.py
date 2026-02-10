import threading 
import psutil 
import argparse
import torch
import torch.nn.functional as F
import gc 
import os
import random
import uuid
import json

## general argument parser that works for all environments
def general_arguments():
    parser = argparse.ArgumentParser()
    ## general setup
    parser.add_argument("--base_model_name_or_path", default="HunterJiang97/PABU-Agent-8B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    return parser

## detailed arguments for SFT training environments
def sft_arguments():
    parser = general_arguments()
    ## train-related
    parser.add_argument("--saved_path", type=str, default="") # if length longer than 0 then will read such adapter
    parser.add_argument("--save_by_step", type=bool, default=False)
    parser.add_argument("--save_path_format", default="../model/llama-31-8b-e{}-s{}-c{}/")
    parser.add_argument("--datafile_train", default="../train/train.csv")
    parser.add_argument("--train_col", default="all")
    parser.add_argument("--datafile_test", default="../train/sampled_test.csv")
    parser.add_argument("--test_col", default="prompt")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=1)    
    parser.add_argument("--batch_size", type=int, default=4)
    return parser

## detailed arguments for alfworld playing environments
def play_arguments():
    parser = general_arguments()
    ## inference-related
    parser.add_argument("--saved_path", default="")   
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--task_file", type=str, default="")
    parser.add_argument("--play_save_path", type=str, default="")
    parser.add_argument("--do_actor_sample", type=bool, default=False)
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--working_type", type = str, default = "step")
    parser.add_argument("--play_port", type = str, default = "36001")
    parser.add_argument("--instances", type = int, default = 8)
    parser.add_argument("--max_iter", type = int, default = 30)
    parser.add_argument("--game_type", type = str, default = "alfworld")
    parser.add_argument("--log_save_path", type = str, default = "../logs/")
    return parser

def b2mb(x):
    """
    Bytes to MegaBytes
    """
    return int(x / 2**20)
    
class TorchTracemalloc:
    def __enter__(self):
        gc.collect() # Clean unreferenced objects
        torch.cuda.empty_cache() # Avoid OOM
        torch.cuda.reset_max_memory_allocated() # reset peak gauge to zero 
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process(os.getpid())
        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True 
        peak_monitor_thread = threading.Thread(target = self.peak_monitor_func)
        peak_monitor_thread.daemon = True 
        peak_monitor_thread.start()
        return self
    def cpu_mem_used(self):
        return self.process.memory_info().rss 
    def peak_monitor_func(self):
        self.cpu_peak = -1 
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break
    def __exit__(self, *exc):
        self.peak_monitoring = False 
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

def preprocess_function(examples, tokenizer, column, max_length, train = True):
    if train:
        str_use = [x + "<|end_of_text|><|im_end|><|endoftext|><|end_of_text|><|im_end|><|endoftext|><|end_of_text|><|im_end|><|endoftext|>" for x in examples[column]]
    else:
        str_use = [x for x in examples[column]]
        
    return tokenizer(
        str_use, 
        padding = "max_length",
        max_length = max_length,
        truncation=True
    )