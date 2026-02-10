from concurrent.futures import ThreadPoolExecutor
import argparse
import psutil
from datetime import datetime
import json
import time
import sys
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy

from accelerate import Accelerator
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    DataCollatorForLanguageModeling,
)
import torch

from utils import TorchTracemalloc, play_arguments
from utils_agentenv import (
    AlfWorldEnvClient, 
    SciworldEnvClient, 
    MazeEnvClient, 
    WordleEnvClient, 
    TextCraftEnvClient, 
    BabyAIEnvClient,
    MovieEnvClient,
    WeatherEnvClient,
    Env_Conn_Full
    )

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def update_host_status(args):
    pred_iter, host_id = args
    return host_id, hosts[host_id]["env"].step(pred_iter)

def update_game_status(args):
    idx, task_id, save_dir, conn_function, epsilon, game_type, max_iter = args
    return idx, task_id, conn_function(
            hosts[idx]["client"],
            task_id,
            save_path= save_dir,
            max_iter = max_iter,
            game_type = game_type
        )

def main(args):
    ## Init Accelerator
    accelerator = Accelerator()
    set_seed(args.seed)

    ## set connection function
    ALF_Env_Conn_sel = Env_Conn_Full
    accelerator.print("using bridging class: {}".format(str(ALF_Env_Conn_sel)))

    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    ## figure out tasks
    tasks_all = [int(x) for x in list(np.load(args.task_file))]
    accelerator.print(np.load(args.task_file))

    ## saved time var
    time_model = 0
    time_env = 0

    ## create dir if not exist
    if len(args.play_save_path) > 0:
        save_path = args.play_save_path if args.play_save_path[-1] == "/" else args.play_save_path + "/"
    else:
        save_path = args.base_model_name_or_path if args.base_model_name_or_path[-1] == "/" else args.base_model_name_or_path + "/"
        save_path = save_path.replace("/model/", "/eval/")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
        accelerator.print("Create a new dir:", save_path)
    else:
        accelerator.print("Dir:", save_path, "exists")

    ## tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    except:
        accelerator.print("Local tokenizer not found, fetching from hub")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ## load model
    peft_model_id = args.saved_path
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, trust_remote_code=True
    )

    # init host env
    current_task_idx = 0
    all_done = False
    prepare_flag = False
    while not all_done:
        ## update host
        current_template_list = []
        current_host_list = []
        all_done = True
        t1 = time.time()
        idx_iter, task_idx_iter, save_path_iter, conn_fun_iter, epsilon_iter, type_iter = [], [], [], [], [], []
        max_iter = []
        for idx, _ in enumerate(hosts):
            if (hosts[idx]["status"] == True) and (current_task_idx < len(tasks_all)):
                idx_iter.append(idx)
                task_idx_iter.append(tasks_all[current_task_idx])
                save_path_iter.append(deepcopy(save_path))
                conn_fun_iter.append(deepcopy(ALF_Env_Conn_sel))
                epsilon_iter.append(deepcopy(args.epsilon))
                type_iter.append(deepcopy(args.game_type))
                max_iter.append(deepcopy(args.max_iter))
                #hosts_iter.append(deepcopy(hosts))
                current_task_idx += 1

        if len(idx_iter) > 0:
            with ThreadPoolExecutor() as executor:
                results = executor.map(update_game_status, zip(
                    idx_iter, task_idx_iter, save_path_iter, 
                    conn_fun_iter, epsilon_iter, type_iter, max_iter))

        # create new game
        for idx, game_id, conn in results:
            hosts[idx]["env"] = conn
            hosts[idx]["status"] = False
            hosts[idx]["idx"] = game_id

        for idx, _ in enumerate(hosts):
            # load data
            if hosts[idx]["status"] == False:
                current_template_list.append(hosts[idx]["env"].history[-1]["content"])
                current_host_list.append(idx)
                all_done = False

        ## dataset (current step)
        with accelerator.main_process_first():
            test_dataset = Dataset.from_dict(
                {"template": current_template_list, "hosts": current_host_list}
            ).map(
                lambda examples: tokenizer(
                    examples["template"],
                    padding="max_length",
                    max_length=args.max_length,
                    truncation=True,
                    return_token_type_ids=False,
                ),
                batched=True,
                num_proc=1,
                load_from_cache_file=True,
                remove_columns=["template", "hosts"],
                desc="Running Tokenizer on test dataset",
            )
        accelerator.wait_for_everyone()

        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=data_collator,
            batch_size=args.batch_size,
            pin_memory=True,
        )
        if not prepare_flag:
            model, test_dataloader = accelerator.prepare(
                model, test_dataloader  # , optimizer, lr_scheduler
            )
            prepare_flag = True

        time_env += time.time() - t1

        t2 = time.time()

        model.eval()
        eval_preds = []
        BEAM_SIZE = 8
        with TorchTracemalloc() as tracemalloc:

            for _, batch in enumerate(tqdm(test_dataloader)):

                batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch,
                        synced_gpus=False,
                        max_new_tokens=256,
                        eos_token_id=128001,
                        do_sample=True,
                        top_k=100,
                        temperature=0.6
                    )
                outputs = accelerator.pad_across_processes(
                    outputs, dim=1, pad_index=tokenizer.pad_token_id
                )
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, args.max_length :].detach().cpu().numpy()
                preds = tokenizer.batch_decode(preds, skip_special_token=True)
                
                accelerator.print(preds)
                eval_preds.extend(preds)
        accelerator.wait_for_everyone()

        ## host step
        time_model += time.time() - t2
        t3 = time.time()

        with ThreadPoolExecutor() as executor:
            results = executor.map(update_host_status, zip(eval_preds, current_host_list))

        for host_id, status in results:
            hosts[host_id]["status"] = status

        time_env += time.time() - t3

        accelerator.print("Env.  Time: {} s\nModel Time: {} s".format(
            str(round(time_env, 2)), 
            str(round(time_model, 2))
            ))

    ## save time
    log_save = vars(args)
    log_save["time_env"] = time_env
    log_save["time_model"] = time_model

    filename = args.log_save_path + args.base_model_name_or_path.split("/")[-1] + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(filename, 'w') as f:
        json.dump(log_save, f, indent=2)



if __name__ == "__main__":
    args = play_arguments().parse_args()
    print("working with ", args)

    ## connect to hosts
    ports = [*range(int(args.play_port), int(args.play_port) + int(args.instances))]
    hosts = []

    SERVER_IP = "127.0.0.1"
    if args.game_type == "alfworld":
        Client_Use = AlfWorldEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "sciworld":
        Client_Use = SciworldEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "textcraft":
        Client_Use = TextCraftEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "babyai":
        Client_Use = BabyAIEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "weather":
        Client_Use = WeatherEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "movie":
        Client_Use = MovieEnvClient
        Link_use = "http://{}:{}"
    elif args.game_type == "maze":
        Client_Use = MazeEnvClient
        Link_use = "http://{}:{}/maze"
    elif args.game_type == "wordle":
        Client_Use = WordleEnvClient
        Link_use = "http://{}:{}/wordle"
    
    for idx, port in enumerate(ports):
        hosts.append(
            {
                "status": True,
                "client": Client_Use(
                    env_server_base=Link_use.format(SERVER_IP, str(port)),
                ),
                "env": None,
            }
        )

    main(args)
