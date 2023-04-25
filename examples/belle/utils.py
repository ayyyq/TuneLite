import copy
from dataclasses import dataclass

import numpy as np
from transformers.utils import PaddingStrategy
from transformers.trainer import *


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

    tokenizer: Any
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = 'right'

    def __call__(self, features, return_tensors=None):
        padding_side = self.padding_side

        # if return_tensors is None:
        #     return_tensors = self.return_tensors
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

        max_length = max(len(feature['input_ids']) for feature in features)
        if padding_side == 'right':
            input_ids = [feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids']))
                         for feature in features]
            attention_mask = [[1] * len(feature['input_ids']) + [0] * (max_length - len(feature['input_ids'])) for
                              feature in features]
        elif padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])) + feature['input_ids']
                         for feature in features]
            attention_mask = [[0] * (max_length - len(feature['input_ids'])) + [1] * len(feature['input_ids']) for
                              feature in features]
        else:
            raise ValueError("Invalid padding strategy:" + str(padding_side))

        features = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'labels': torch.tensor(np.array([feature['labels'] for feature in features])).long()
        }
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

    tokenizer: Any
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = 'left'
    unconditional_normalization: bool = False

    def __call__(self, features, return_tensors=None):
        padding_side = self.padding_side

        split_size = []
        new_features = []
        assert "labels" in features[0].keys()
        for feature in features:
            split_size.append(len(feature["labels"]))
            for op_input_ids, op_labels in zip(feature["input_ids"], feature["labels"]):
                un_mask = np.zeros_like(op_labels)
                un_mask_index = np.where(op_labels == self.label_pad_token_id, 1, 0).sum() - 2
                un_mask[:un_mask_index] = 1
                new_features.append({"input_ids": op_input_ids, "labels": op_labels, "un_mask": un_mask})

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

            for feature in new_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    feature["un_mask"] = np.concatenate([feature["un_mask"], np.ones_like(remainder)]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    feature["un_mask"] = np.concatenate([np.ones_like(remainder), feature["un_mask"]]).astype(np.int64)

        max_length = max(len(feature['input_ids']) for feature in new_features)
        if padding_side == 'right':
            input_ids = [feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids']))
                         for feature in new_features]
            attention_mask = [[1] * len(feature['input_ids']) + [0] * (max_length - len(feature['input_ids'])) for
                              feature in new_features]
        elif padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])) + feature['input_ids']
                         for feature in new_features]
            attention_mask = [[0] * (max_length - len(feature['input_ids'])) + [1] * len(feature['input_ids']) for
                              feature in new_features]
        else:
            raise ValueError("Invalid padding strategy:" + str(padding_side))

        batched_features = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'labels': torch.tensor(np.array([feature['labels'] for feature in new_features])).long(),
            'split_size': split_size
        }
        if self.unconditional_normalization:
            batched_features['un_mask'] = torch.tensor(np.array([feature['un_mask'] for feature in new_features])).bool()

        return batched_features
