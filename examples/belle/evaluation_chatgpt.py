import os
import re
import json
import time
import random
import argparse

import numpy as np
import torch
import backoff
import openai

import wandb
from fastNLP import print, logger
# os.environ['WANDB_MODE'] = 'offline'


def get_input_prompt(prompt, instance):
    user_answer = instance['user_answer']
    if '我的回答是' in user_answer:
        user_answer = instance['user_answer'].split('我的回答是')
        user_answer = user_answer[1].strip() if len(user_answer) > 1 else ''
    if '我回答完了，请您过目。' in user_answer:
        user_answer = user_answer.split('我回答完了，请您过目。')[0].strip()

    if instance['class'] in ["generation", "brainstorming", "rewrite"]:
        return f"{prompt}\n\n问题：{instance['question']}\n模型回答：{user_answer}。\n\n请针对模型回答给出得分，顺便给出理由："
    else:
        return f"{prompt}\n\n问题：{instance['question']}\n标准回答：{instance['std_answer']}\n模型回答：{user_answer}。\n\n请针对模型回答给出得分，顺便给出理由："


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_chat_response(args, input, key):
    response = openai.ChatCompletion.create(
        model=args.model,
        api_key=key,
        messages=[
            {'role': 'user', 'content': input},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    prediction = response['choices'][0]['message']['content'].strip()
    return prediction


class Decoder():
    def __init__(self):
        self.keys = [
            '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',
            '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',
            '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',
            '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',
            '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'
        ]

    def decode(self, args, input, key_id):
        response = get_chat_response(args, input, self.keys[key_id])
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
    # model
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    # data
    parser.add_argument("--data_tag", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='outputs/belle_llama-65B_sft-1w/output_lr0.01_warmup0.05_clipgradnorm5.0/checkpoint-0-eval')
    parser.add_argument("--eval_file", type=str, default='eval_predictions.jsonl')
    parser.add_argument(
        "--resume_id", type=int, default=-1,
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    args = parser.parse_args()
    fix_seed(args.random_seed)
    wandb.init(project='chatgpt',
               name=args.data_tag if args.data_tag else '-'.join(args.data_dir.split('/')[1:]),
               config=vars(args))

    decoder = Decoder()
    data = [json.loads(line) for line in open(os.path.join(args.data_dir, args.eval_file), 'r')]
    # assert len(data) == 1000
    prompts = {}
    for line in open('examples/belle/eval/eval_prompt.json', 'r'):
        prompt = json.loads(line)
        prompts[prompt['class'].lower()] = prompt['prompt']

    output_filename = os.path.join(args.data_dir, 'chatgpt_scores.jsonl') if args.resume_id < 0 else os.path.join(args.data_dir, f'chatgpt_scores_{args.resume_id}.jsonl')
    print('Writing scores to {}'.format(output_filename))
    with open(output_filename, 'w', encoding='utf-8') as output:
        for i, instance in enumerate(data):
            if i < args.resume_id - 1:
                continue

            # minibatch size should be 1 because GPT-3 API takes only 1 input for each request
            print('*************************')
            print("{}st data".format(i + 1))
            input_prompt = get_input_prompt(prompts[instance['class']], instance)
            print(input_prompt)

            score = decoder.decode(args, input_prompt, i % 5)
            new_instance = instance
            new_instance['score'] = score
            output.write(json.dumps(new_instance, ensure_ascii=False) + '\n')

            print(score)
            print('*************************')


def post_process(filename):
    data = [json.loads(line) for line in open(filename, 'r')]
    avg_score = 0.0
    pattern = r'得分[^0-9]*(\d+[.]?\d*)'
    for instance in data:
        score = re.match(pattern, instance['score']).group(1)
        avg_score += float(score)
    avg_score = avg_score / len(data)
    print(f'Average score: {avg_score}')


if __name__ == '__main__':
    main()
    # post_process('outputs/guobo/chatgpt_scores.jsonl')  # 59.76
