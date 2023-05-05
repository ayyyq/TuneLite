import copy
import os
import sys
import json

import torch
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.deepspeed import HfDeepSpeedConfig
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)
from collie.log import print
from arguments import ModelArguments, DataArguments, MyCollieArguments
from mydatasets import MyDataset
from mytrainer import MyInplaceZeroTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM
from collie.trainer import InplaceZeroTrainer


def train():
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.float16)
    # torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyCollieArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, collie_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, collie_args = parser.parse_args_into_dataclasses()
    set_seed(collie_args.seed)

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
    ds_config = collie_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = collie_args.gradient_checkpointing
    if collie_args.resume_from_checkpoint is not None:
        print(f'Load checkpoint from {collie_args.resume_from_checkpoint}.')
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path if collie_args.resume_from_checkpoint is None else collie_args.resume_from_checkpoint,
        cache_dir=model_args.cache_dir,
        # local_files_only=True,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    train_dataset = MyDataset(data_args, tokenizer, 'train')
    eval_dataset = MyDataset(data_args, tokenizer, 'eval')
    test_dataset = MyDataset(data_args, tokenizer, 'test')

    def compute_metrics(decoded_preds, dataset, save_prefix='eval'):
        preds, golds = [], []
        if collie_args.local_rank in [-1, 0]:
            output_filename = os.path.join(collie_args.output_dir, f"{save_prefix}_predictions.jsonl")
            index = 1
            while os.path.exists(output_filename):
                output_filename = os.path.join(collie_args.output_dir, f"{save_prefix}_predictions_{index}.jsonl")
                index += 1
            with open(output_filename, "w", encoding='utf-8') as fout:
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
                print(f"Save {save_prefix} predictions to {output_filename}.")
        torch.distributed.barrier()

        result = {'score': 0}
        return result

    # ========== 4. Initialize our Trainer. ==========
    trainer = InplaceZeroTrainer(
        model=model,
        collie_args=collie_args,
        data_collator=DataCollatorForCauselLM(tokenizer, max_length=data_args.max_length, padding_side='left'),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if collie_args.do_train:
        trainer.train()

    if collie_args.do_predict:
        test_dataloader = trainer.get_eval_dataloader(test_dataset)
        trainer.eval(trainer.global_step, collie_args.num_train_epochs - 1, test_dataset, test_dataloader, 'test')


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
