from dataclasses import dataclass, field
from typing import Optional
from collie.arguments import CollieArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llama-7B")
    cache_dir: Optional[str] = field(default='../llama/checkpoint')
    # llama_dir: Optional[str] = field(default='/remote-home/klv/exps/MossOn3090/llama')


@dataclass
class DataArguments:
    data_dir: str = field(default='data')
    dataset_name: str = field(default='openbookqa')
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})

    data_tag: str = field(default='src')
    prompt_type: str = field(default='natural', metadata={"help": "The type of prompt, including [natural, brown]."})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on input."})
    max_length: int = field(default=1024)
    few_shot_size: int = field(default=-1)
    in_context_learning: bool = field(default=False, metadata={"help": "Whether to use in-context learning."})


@dataclass
class MyCollieArguments(CollieArguments):
    length_normalization: bool = field(default=True, metadata={"help": "Whether to normalize the loss by the length of the input."})
    unconditional_normalization: bool = field(default=False, metadata={"help": "Whether to normalize the loss by the length of the input."})

    hf_learning_rate: float = field(default=5e-4, metadata={"help": "The learning rate for the HF optimizer."})
    hf_weight_decay: float = field(default=0.0, metadata={"help": "The weight decay for the HF optimizer."})
    hf_lr_scheduler_type: str = field(default='linear', metadata={"help": "The lr scheduler type for the HF optimizer."})
    hf_warmup: int = field(default=0, metadata={"help": "The warmup steps for the HF optimizer."})

    # lora hyperparams
    peft_type: str = field(default=None, metadata={
        "help": "The type of PEFT, including [lora, prefix-tuning, prompt-tuning, p-tuning]."})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})
