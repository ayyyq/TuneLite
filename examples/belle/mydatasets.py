import os
import copy
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass
from fastNLP import print

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Optional, Union, Callable


IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0
user_prompt = """[用户] 从现在开始，你是一个回答问题，完成指令的人工智能机器人，你要解答各种类型的问题，包括编程，算数，回答常识或专业知识。请提供尽可能详尽且对用户有帮助的回答。问题可能会包含很多换行符进行格式化，你可以把文本传来起来理解。另外回答算式，代码的时候可以采用markdown的格式进行回答。
{0:}
[机器人] """
agent_prompt = """我的回答是
{0:}
我回答完了，请您过目。"""


class MyDataset(Dataset):
    def __init__(self, data_args, tokenizer, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split

        save_dir = os.path.join(data_args.data_save_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            if split == 'train':
                if data_args.sample_size > 0:
                    dataset = load_from_disk(os.path.join(data_args.data_cache_dir, f'Belle_2M_CN_1w'))  # 1w
                else:
                    dataset = load_dataset(data_args.data_path[split], split=split)  # 1M
                # if data_args.sample_size > 0:
                #     # random挑选1w条数据
                #     random.seed(REPRODUCIBILITY_SEED)
                #     possible_indices = list(range(len(dataset)))
                #     sampled_indices = random.sample(possible_indices, data_args.sample_size)
                #     dataset = dataset.select(sampled_indices)
                #     dataset.save_to_disk(os.path.join(data_args.data_cache_dir, f'Belle_2M_CN_1w'))
                # dataset = load_from_disk(os.path.join(data_args.data_cache_dir, f'Belle_2M_CN_1w'))
            else:
                dataset = load_dataset('json', data_files=data_args.data_path[split], split='train')

            self.data = self.process(dataset, save_file)
        else:
            print(f'Load data from {save_file}.')
            self.data = torch.load(save_file)
        # if split == 'train':
        #     self.data = self.data[:5000]
        if split == 'eval':
            self.data = self.data[:20]
        print('Data size:', len(self.data))
        print('Data format:', self.data[0])
        print('Max length:', max([len(d['input_ids']) for d in self.data]))

    def process(self, dataset, save_file):
        data = []
        for instance in tqdm(dataset):
            # "Human: "+sample['instruction']+sample['input']+"\n Assistant: "+sample['output']
            if self.split == 'train':
                source = user_prompt.format(instance['instruction'].strip())
                target = f"{agent_prompt.format(instance['output'].strip())}{self.tokenizer.eos_token}"
            else:
                source = f"{user_prompt.format(instance['question'].strip())}我的回答是\n"
                target = f"{instance['std_answer'].strip()}{self.tokenizer.eos_token}"

            example = source + target
            example_tokenized = self.tokenizer(example, truncation=True, max_length=self.data_args.max_length)
            source_tokenized = self.tokenizer(source, truncation=True, max_length=self.data_args.max_length)

            if self.split == 'train':
                input_ids = example_tokenized['input_ids']
                labels = copy.deepcopy(input_ids)
                if not self.data_args.train_on_inputs:
                    labels = np.array(labels)
                    labels[:len(source_tokenized['input_ids']) - 1] = IGNORE_INDEX
                data.append({'input_ids': input_ids,
                             'labels': labels,
                             'source': source,
                             'target': target})
            else:
                input_ids = source_tokenized['input_ids']
                labels = copy.deepcopy(input_ids)
                data.append({'input_ids': input_ids,
                             'labels': labels,
                             'source': source,
                             'question': instance['question'],
                             'std_answer': instance['std_answer'],
                             'class': instance['class']})

        torch.save(data, save_file)
        print(f'Save data to {save_file}.')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }


@dataclass
class DataCollatorForCauselLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features


@dataclass
class EvalDataCollatorForCauselLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        split_size = []
        new_features = []
        assert "labels" in features[0].keys()
        for feature in features:
            split_size.append(len(feature["labels"]))
            for op_input_ids, op_labels in zip(feature["input_ids"], feature["labels"]):
                new_features.append({"input_ids": op_input_ids, "labels": op_labels})

        labels = [feature["labels"] for feature in new_features]
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in new_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        new_features = self.tokenizer.pad(
            new_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        new_features["split_size"] = split_size

        return new_features


if __name__ == '__main__':
    from transformers import HfArgumentParser, LlamaTokenizer
    from arguments import ModelArguments, DataArguments

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    data_args.dataset_name = 'belle'
    data_args.data_path = {'train': 'BelleGroup/train_2M_CN',
                           'eval': 'examples/belle/eval/eval_set.json'}
    data_args.data_cache_dir = 'cache'
    data_args.refresh = True
    data_args.data_tag = 'base'
    data_args.train_on_inputs = True
    data_args.max_length = 1024
    data_args.sample_size = 10000

    tokenizer = LlamaTokenizer.from_pretrained(
        '../finetuneLLM/cache/llama-65b',
        cache_dir=model_args.cache_dir,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    # train_dataset = MyDataset(data_args, tokenizer, split='train')
    eval_dataset = MyDataset(data_args, tokenizer, split='eval')
