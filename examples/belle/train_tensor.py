import copy
import os
import sys
import copy
import json
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

    wandb_config = copy.deepcopy(asdict(collie_args))
    wandb_config.update(asdict(model_args))
    wandb_config.update(asdict(data_args))
    if collie_args.tag == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    if collie_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(collie_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        wandb.init(
            project="belle",
            entity='collie_exp',
            name=tag_name if collie_args.tag == 'zero-shot' or hparam_name == 'output' else '_'.join([tag_name, hparam_name.replace('output_', '')]),
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
    # eval_dataset = MyDataset(data_args, tokenizer, 'eval')
    set_seed(collie_args.seed)

    # ========== 4. Initialize our Trainer. ==========
    def compute_metrics(decoded_preds, dataset, save_prefix=None):
        preds, golds = [], []
        with open(os.path.join(collie_args.output_dir, f"{save_prefix}_predictions.jsonl"), "w") as fout:
            for gold_instance, pred_text in zip(dataset.data, decoded_preds):
                temp = pred_text.split('Assistant: ')
                pred = temp[1].strip() if len(temp) > 1 else ''
                preds.append(pred)
                gold = gold_instance['target'][:gold_instance['target'].index(tokenizer.eos_token)]
                golds.append(gold)

                fout.write(json.dumps({
                    "source": gold_instance['source'],
                    "target": gold,
                    "pred": pred,
                    'pred_text': pred_text,
                }) + "\n")
            print(f"Saved {save_prefix} predictions to {collie_args.output_dir}.")

        result = {'score': 0}
        return result

    set_seed(collie_args.seed)
    trainer = InplaceTensorTrainer(
        model=model,
        collie_args=collie_args,
        data_collator=DataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        cache_dir=os.path.join(model_args.cache_dir, model_args.model_name_or_path.split('-')[-1]),
    )

    if 'zero-shot' in collie_args.tag or 'eval' in collie_args.tag:
        trainer.eval(trainer.global_step, 1, trainer.eval_dataset['test'], trainer.eval_dataloader['test'], 'test')
    else:
        trainer.train()


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
