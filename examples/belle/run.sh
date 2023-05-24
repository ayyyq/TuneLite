set -x
port=$(shuf -i25000-30000 -n1)

# for tensor trainer with inplace sgd
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port="$port" examples/belle/train_tensor.py examples/belle/hf_args_tensor.yaml

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port="$port" examples/belle/train_tensor.py examples/belle/eval_hf_args_tensor.yaml

# for zero trainer with inplace sgd
#WANDB_MODE=disabled \
deepspeed --master_port "$port" --include localhost:0,1,2,3 examples/belle/train_zero.py examples/belle/hf_args_zero.yaml
