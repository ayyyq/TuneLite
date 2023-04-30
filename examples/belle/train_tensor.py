import copy
import os
import sys
import copy
import json
import re
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser, AutoTokenizer
from transformers import set_seed
from dataclasses import asdict
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)

from collie.models import llama
from collie.log import print
from arguments import ModelArguments, DataArguments, MyCollieArguments
from mydatasets import MyDataset
from collie.trainer import InplaceTensorTrainer, InplaceZeroTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM

IGNORE_INDEX = -100


def train():
    # ========== 1. logs and args ==========
    local_rank, world_size = llama.setup_model_parallel()
    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    set_seed(collie_args.seed)
    assert local_rank == collie_args.local_rank
    assert world_size == collie_args.world_size

    model_name = model_args.model_name_or_path.split('/')[-1]
    tag_name = '_'.join([data_args.dataset_name, model_name, collie_args.tag] if collie_args.tag else [data_args.dataset_name, model_name])

    hparam_name = 'output'
    if collie_args.optim != 'sgd':
        hparam_name += '_' + collie_args.optim
    if collie_args.learning_rate != 3e-2:
        hparam_name += '_lr' + str(collie_args.learning_rate)
    if collie_args.per_device_train_batch_size != 1:
        hparam_name += '_bs' + str(collie_args.per_device_train_batch_size)
    if collie_args.lr_scheduler_type != 'linear':
        hparam_name += '_' + collie_args.lr_scheduler_type
    if collie_args.warmup != 0:
        hparam_name += '_warmup' + str(collie_args.warmup)
    if collie_args.clip_grad_norm and collie_args.clip_grad_norm > 0:
        hparam_name += '_clipgradnorm' + str(collie_args.clip_grad_norm)
    if collie_args.clip_grad_value:
        hparam_name += '_clipgrad' + str(collie_args.clip_grad_value)
    if collie_args.clip_loss_value:
        hparam_name += '_cliploss' + str(collie_args.clip_loss_value)
    collie_args.output_dir = os.path.join('outputs', tag_name, hparam_name)
    if 'eval' in collie_args.tag:
        assert collie_args.resume_from_checkpoint is not None
        collie_args.output_dir = f"{collie_args.resume_from_checkpoint}-eval"
    if not os.path.exists(collie_args.output_dir):
        os.makedirs(collie_args.output_dir, exist_ok=True)

    if 'debug' in collie_args.tag:
        os.environ['WANDB_MODE'] = 'offline'
    if collie_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(collie_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        if 'eval' in collie_args.tag:
            wandb_name = collie_args.resume_from_checkpoint.replace('outputs', 'eval').replace('/', '_')
        elif 'zero-shot' in collie_args.tag or hparam_name == 'output':
            wandb_name = tag_name
        else:
            wandb_name = '_'.join([tag_name, hparam_name.replace('output_', '')])
        wandb.init(
            project="belle",
            entity='collie_exp',
            name=wandb_name,
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    model, _ = llama.load_model(
        ckpt_dir=os.path.join(model_args.cache_dir, model_args.model_name_or_path.split('-')[-1]) if collie_args.resume_from_checkpoint is None else collie_args.resume_from_checkpoint,  # 7B, 13B, 30B, 65B
        tokenizer_path=os.path.join(model_args.cache_dir, 'tokenizer.model'),
        local_rank=collie_args.local_rank,
        world_size=collie_args.world_size,
        froze_embeddings=False,
        zero=False,
        tensor_parallel=True,
        pipeline_parallel=False,
        max_batch_size=collie_args.per_device_eval_batch_size,
        max_seq_len=data_args.max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'cache/llama-65b',
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    torch.cuda.empty_cache()

    # ========== 3. Preprocessing the datasets. ==========
    train_dataset = MyDataset(data_args, tokenizer, 'train')
    eval_dataset = MyDataset(data_args, tokenizer, 'eval')
    test_dataset = MyDataset(data_args, tokenizer, 'test')

    # ========== 4. Initialize our Trainer. ==========
    def compute_metrics(decoded_preds, dataset, save_prefix='eval'):
        preds, golds = [], []
        if collie_args.local_rank in [-1, 0]:
            output_filename = os.path.join(collie_args.output_dir, f"{save_prefix}_predictions.jsonl")
            index = 1
            while os.path.exists(output_filename):
                output_filename = os.path.join(collie_args.output_dir, f"{save_prefix}_predictions_{index}.jsonl")
                index += 1
            with open(output_filename, "w") as fout:
                for gold_instance, pred_text in zip(dataset.data, decoded_preds):
                    temp = pred_text.split('我的回答是')
                    pred = temp[1].strip() if len(temp) > 1 else ''
                    if '我回答完了，请您过目。' in pred:
                        pred = pred.split('我回答完了，请您过目。')[0].strip()
                    preds.append(pred)

                    fout.write(json.dumps({
                        "question": gold_instance['question'],
                        "std_answer": gold_instance['std_answer'],
                        'user_answer': pred,
                        'prediction': pred_text,
                        'class': gold_instance['class'],
                    }, ensure_ascii=False) + "\n")
                print(f"Saved {save_prefix} predictions to {output_filename}.")
        torch.distributed.barrier()

        result = {'score': 0}
        return result

    trainer = InplaceTensorTrainer(
        model=model,
        collie_args=collie_args,
        data_collator=DataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        cache_dir=os.path.join(model_args.cache_dir, model_args.model_name_or_path.split('-')[-1]),
    )

    if 'zero-shot' in collie_args.tag or 'eval' in collie_args.tag:
        trainer.eval(trainer.global_step, 0, trainer.eval_dataset, trainer.eval_dataloader, 'eval')
    else:
        trainer.train()

        test_dataloader = trainer.get_eval_dataloader(test_dataset)
        trainer.eval(trainer.global_step, 0, test_dataset, test_dataloader, 'test')


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
