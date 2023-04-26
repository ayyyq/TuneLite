import os
import json
import time
import random
import argparse

import numpy as np
import torch
import backoff
import openai
openai.api_key = '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'
# openai.api_key = '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'

import wandb
from fastNLP import print, logger
os.environ['WANDB_MODE'] = 'offline'


def get_input_prompt(prompt, instance):
    if instance['class'] in ["generation", "brainstorming", "rewrite"]:
        return f"{prompt}\n\n问题：{instance['question']}\n模型回答：{instance['user_answer']}。\n\n请针对模型回答给出得分，顺便给出理由："
    else:
        return f"{prompt}\n\n问题：{instance['question']}\n标准回答：{instance['std_answer']}\n模型回答：{instance['user_answer']}。\n\n请针对模型回答给出得分，顺便给出理由："


# Sentence Generator (Decoder) for GPT-3 ...
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def decoder_for_gpt3(args, input):
    response = openai.ChatCompletion.create(
        max_tokens=3000,
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': input},
        ],
        temperature=args.temperature,
    )

    return response['choices'][0]['message']['content']


class Decoder():
    def __init__(self):
        # print_now()
        pass

    def decode(self, args, input):
        response = decoder_for_gpt3(args, input)
        return response


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument('--wandb_name', type=str, default=None)
    # model
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for GPT-3"
    )
    # data
    parser.add_argument("--data_dir", type=str, default='outputs/belle_llama-13B_eval-debug/output_lr0.01_warmup0.05_clipgradnorm5.0')
    parser.add_argument(
        "--resume_id", type=int, default=-1,
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    args = parser.parse_args()
    fix_seed(args.random_seed)
    wandb.init(project='chatgpt', name=args.wandb_name, config=args)

    decoder = Decoder()
    data = [json.loads(line) for line in open(os.path.join(args.data_dir, 'eval_predictions.jsonl'), 'r')]
    assert len(data) == 1000
    prompts = {}
    for line in open('examples/belle/eval/eval_prompt.json', 'r'):
        prompt = json.loads(line)
        prompts[prompt['class'].lower()] = prompt['prompt']
    print('Writing scores to {}'.format(os.path.join(args.data_dir, 'chatgpt_scores.jsonl')))
    with open(os.path.join(args.data_dir, 'chatgpt_scores.jsonl'), 'w') as output:
        for i, instance in enumerate(data):
            if i < args.resume_id:
                continue

            # minibatch size should be 1 because GPT-3 API takes only 1 input for each request
            print('*************************')
            print("{}st data".format(i + 1))
            input_prompt = get_input_prompt(prompts[instance['class']], instance)
            print(input_prompt)

            score = decoder.decode(args, input_prompt)
            new_instance = instance
            new_instance['score'] = score
            output.write(json.dumps(new_instance) + '\n')

            print(score)
            print('*************************')


if __name__ == '__main__':
    main()
