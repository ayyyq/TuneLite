set -x
port=$(shuf -i25000-30000 -n1)

# for tensor trainer with inplace sgd
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port="$port" examples/mcqa/train_tensor.py examples/mcqa/hf_args_tensor.yaml

# for zero trainer with inplace sgd
#WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0 train_zero_lora.py hf_args_zero_lora.yaml

#deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 train_zero_lora.py hf_args_zero_v100_lora.yaml

#deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 train_zero_lora.py hf_args_zero_v100_lora_1.yaml

#deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 train_zero_lora.py hf_args_zero_v100_lora_2.yaml


#deepspeed --master_port "$port" --include localhost:1 train_zero.py hf_args_zero.yaml
