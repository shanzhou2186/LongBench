import os, csv, json
import argparse
import time
from tqdm import tqdm
import re
import urllib.request
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

DEFAULT_URL = os.environ.get("OPENAI_BASE_URL", os.environ.get("SGLANG_URL", "http://127.0.0.1:3000/v1"))
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
DEFAULT_MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", os.environ.get("SGLANG_MAX_TOTAL_TOKENS", "64000")))
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()


def strip_think_blocks(text):
    text = text or ''
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def prepare_prompt(prompt, model, enable_thinking=False):
    prompt = prompt.strip()
    if enable_thinking:
        return prompt
    if 'qwen3' in model.lower() and not prompt.startswith('/no_think') and not prompt.startswith('/think'):
        return '/no_think\n' + prompt
    return prompt


def ensure_server_ready(base_url, timeout=5):
    models_url = base_url.rstrip('/') + '/models'
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception as exc:
        raise RuntimeError(
            f'Cannot reach OpenAI-compatible server at {models_url}. '
            'Start your sglang/OpenAI-compatible server first, or pass the correct '
            f'--url/--api_key values. Current --url is {base_url}.'
        ) from exc


def encode_prompt(prompt, model, tokenizer):
    if model in model_map:
        return tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.encode(prompt, disallowed_special=())


def decode_prompt(token_ids, model, tokenizer):
    if model in model_map:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return tokenizer.decode(token_ids)


def truncate_prompt(prompt, model, tokenizer, max_len=None):
    if max_len is None:
        max_len = maxlen_map[model]
    input_ids = encode_prompt(prompt, model, tokenizer)
    if len(input_ids) <= max_len:
        return prompt

    head_len = max_len // 2
    tail_len = max_len - head_len
    truncated_ids = input_ids[:head_len] + input_ids[-tail_len:]
    return decode_prompt(truncated_ids, model, tokenizer)


def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None, max_prompt_tokens=None):
    prompt = truncate_prompt(prompt, model, tokenizer, max_len=max_prompt_tokens)
    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return strip_think_blocks(completion.choices[0].message.content)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..." % (str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''


def extract_answer(response):
    response = strip_think_blocks(response).replace('*', '').strip()
    patterns = [
        r'The correct answer is\s*\(([A-D])\)',
        r'The correct answer is\s*([A-D])\b',
        r'correct answer\s*[:：]\s*\(?([A-D])\)?',
        r'answer\s*[:：]\s*\(?([A-D])\)?',
        r'option\s*([A-D])\b',
        r'^\(?([A-D])\)?[\s\.:。!]?$',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    standalone = re.findall(r'\b([A-D])\b', response.upper())
    if len(standalone) == 1:
        return standalone[0]
    return None


def load_input_items(input_jsonl):
    with open(input_jsonl, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def build_output_path(args):
    input_stem = os.path.splitext(os.path.basename(args.input_jsonl))[0]
    model_stem = args.model.split("/")[-1]
    if args.rag > 0:
        return os.path.join(args.save_dir, model_stem + f"_{input_stem}_rag_{str(args.rag)}.jsonl")
    if args.no_context:
        return os.path.join(args.save_dir, model_stem + f"_{input_stem}_no_context.jsonl")
    if args.cot:
        return os.path.join(args.save_dir, model_stem + f"_{input_stem}_cot.jsonl")
    return os.path.join(args.save_dir, model_stem + f"_{input_stem}.jsonl")


def get_pred(data, args, out_file, write_lock):
    model = args.model
    max_prompt_tokens = min(maxlen_map[model], args.max_prompt_tokens)
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
        tokenizer.model_max_length = max(tokenizer.model_max_length, 10 ** 9)
    client = OpenAI(
        base_url=args.url,
        api_key=args.api_key,
        timeout=600.0,
    )
    for item in tqdm(data):
        if item.get("_summarize_error"):
            continue
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        prompt = prepare_prompt(prompt, model, enable_thinking=args.cot)
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024, max_prompt_tokens=max_prompt_tokens)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, max_prompt_tokens=max_prompt_tokens)
        if output == '':
            continue
        if args.cot:
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            prompt = prepare_prompt(prompt, model, enable_thinking=False)
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128, max_prompt_tokens=max_prompt_tokens)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        with write_lock:
            with open(out_file, 'a', encoding='utf-8') as fout:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    ensure_server_ready(args.url)
    out_file = build_output_path(args)

    data_all = load_input_items(args.input_jsonl)

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f if line.strip()}
    data = []
    for item in data_all:
        if item.get("_id") not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    write_lock = mp.Lock()
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, write_lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", "-i", type=str, default="results/longbench_v2_summary_5000_test.jsonl")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--max_prompt_tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args()
    main()