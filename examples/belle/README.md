# CoLLiE example on Belle

Support on Belle full-parameter finetuning and evaluation.

## How to run the example

```sh examples/belle/run.sh```

```hf_args_tensor.yaml``` for training , ```eval_hf_args_tensor.yaml``` for evaluation. After generating ```eval_predictions.jsonl```, run
```
python examples/belle/evaluation_chatgpt.py
``` 
to get the final scores.

May need single device of GPU to finetune 7B, 2 devices for 13B, 4 devices for 30B and 8 devices for 65B.