import os
import sys
sys.path.append("/mnt/lustre/zhangshuo/projects/TuneLite")

from datasets import load_dataset

from tunelite.models.llama_colossalai import HFLikeTokenizer, Tokenizer, ModelArgs, get_7B_llama, load_state_dict, get_13B_llama, get_30B_llama
from tunelite.trainer.colossalai_trainer import ColossalaiTrainer, TrainerArgs

import torch
from torch.utils.data import DataLoader


def collate_fn(batch, tokenizer, max_length=1024, bos=True, eos=True):
    text = [e['text'] for e in batch]
    tknz_batch = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        bos=bos,
        eos=eos
    )
    tknz_batch['input_ids'] = tknz_batch['input_ids'][:, :max_length-1]
    return {
        'input_ids': tknz_batch['input_ids'].long()
    }, tknz_batch['input_ids'].long()
    

def main():
    tokenizer = HFLikeTokenizer(
        tokenizer=Tokenizer(model_path='/mnt/petrelfs/zhangshuo/projects/OptiLLM/colossalai/llama/tokenizer.model'))
    def compute_metrics(result, label, epoch, step):
        print("\n")
        print("\n".join([tokenizer.decode(token.tolist()) for token in result][:1]))
        print("\n")
    model_args = ModelArgs()
    model_args.pp_size = 8
    model_args.micro_batch_num = 128
    model_args.fp16 = True
    model_args.checkpoint = True
    model_args.dense = "fused"
    model_args.attention = "flash"
    model_args.rotary_emb = "fused"
    
    trainer_args = TrainerArgs()
    trainer_args.eval_max_length = 30
    trainer_args.eval_per_steps = 100
    trainer_args.eval_per_epoches = 1
    trainer_args.learning_rate = 2e-5
    trainer_args.inplace = False
    
    model = get_7B_llama(model_args)
    state_dict = load_state_dict(model_args=model_args, s3_folder="hdd:s3://opennlplab_hdd/models/llama/llama-7b-hf")
    model.load_state_dict(state_dict)
    train_dataloader = DataLoader(
        [{"text": " ".join(["love" for i in range(512)])} for _ in range(64000)],
        batch_size=128,
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024),
    )
    eval_dataloader = DataLoader(
        [{"text": "My name is MOSS, and my responsibility is to"} for _ in range(128)],
        batch_size=128,
        collate_fn=lambda x: collate_fn(x, tokenizer, 1024, eos=False),
    )
    trainer = ColossalaiTrainer(model=model,
                                train_dataloader=train_dataloader,
                                eval_dataloader=eval_dataloader,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metrics,
                                trainer_args=trainer_args)
    trainer.eval()
    
    
# Command: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 train_colossalai.py
if __name__ == "__main__":
    try:
        main()
    except:
        import rich
        console = rich.console.Console()
        console.print_exception(show_locals=True)