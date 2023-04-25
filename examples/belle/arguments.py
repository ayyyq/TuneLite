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
    dataset_name: str = field(default='belle')
    data_path: dict = field(default=None)
    data_save_dir: str = field(default='data')
    data_cache_dir: str = field(default=None)
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})

    data_tag: str = field(default='src')
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on input."})
    max_length: int = field(default=1024)
    sample_size: int = field(default=-1, metadata={"help": "The number of samples to use."})


@dataclass
class MyCollieArguments(CollieArguments):
    length_normalization: bool = field(default=True, metadata={"help": "Whether to normalize the loss by the length of the input."})
    unconditional_normalization: bool = field(default=False, metadata={"help": "Whether to normalize the loss by the length of the input."})
