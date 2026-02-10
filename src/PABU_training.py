import gc
import os
import sys
import threading
import argparse
import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    DataCollatorForLanguageModeling,
)

from utils import preprocess_function, sft_arguments


def main(args):
    ## Init Accelerator
    accelerator = Accelerator()
    set_seed(args.seed)

    ## Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    except:
        accelerator.print("Local tokenizer not found, fetching from hub")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0  # unk. different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    ## Dataset
    dataset = load_dataset(args.datafile_train).flatten()
    with accelerator.main_process_first():
        train_dataset = dataset["train"].map(
            preprocess_function,
            fn_kwargs={
                "tokenizer": tokenizer,
                "column": args.train_col,
                "max_length": args.max_length,
                "train": True,
            },
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running Tokenizer on train dataset",
        )
    accelerator.wait_for_everyone()

    ## Dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    print(next(iter(train_dataloader)))

    ## Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, trust_remote_code=True
    )

    ## optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    ## Accelerator Prep
    model, train_dataloader, optimizer, lr_scheduler = (
        accelerator.prepare(
            model, train_dataloader, optimizer, lr_scheduler
        )
    )
    accelerator.print(model)

    ## DS Stage 3
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3


    ## save information
    #save_info_list = [str(epoch), str(step), args.train_col, str(args.lr)]
    template_format = len(args.save_path_format.split("{}"))

    ## Start Training and Evaluation
    for epoch in range(args.num_epochs):
        # loss backward
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # save by step
            if (step % 1000 == 0) and (step > 0) and (args.save_by_step):
                save_info_list = [str(epoch), str(step), args.train_col, str(args.lr)]
                model.save_pretrained(
                    args.save_path_format.format(*save_info_list[:template_format])
                )
                tokenizer.save_pretrained(
                    args.save_path_format.format(*save_info_list[:template_format])
                )

        # save after each epoch
        save_info_list = [str(epoch), str(step), args.train_col, str(args.lr)]
        model.save_pretrained(
            args.save_path_format.format(*save_info_list[:template_format])
        )
        tokenizer.save_pretrained(
            args.save_path_format.format(*save_info_list[:template_format])
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

if __name__ == "__main__":
    # parser
    args = sft_arguments().parse_args()
    print("working with ", args)
    # run with arguments
    main(args)
