import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


WORKER_TOKENIZER = None
WORKER_TOKENIZER_KIND = 'hf'
WORKER_MODEL_NAME = ''
WORKER_TEMPLATE = ''


def load_prompt_template(prompt_style):
    template_map = {
        '0shot': 'prompts/0shot.txt',
        'cot': 'prompts/0shot_cot.txt',
        'no_context': 'prompts/0shot_no_context.txt',
    }
    template_path = template_map[prompt_style]
    with open(template_path, encoding='utf-8') as file:
        return file.read()


def load_model_map(config_path):
    with open(config_path, encoding='utf-8') as file:
        return json.load(file)


def load_jsonl(path):
    items = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = item.get('_id')
            if item_id is not None:
                items[item_id] = item
    return items


def load_hf_items(dataset_name, split_name):
    if load_dataset is None:
        raise RuntimeError('datasets 未安装，且未提供 --raw_jsonl 或 --raw_local_json。')
    dataset = load_dataset(dataset_name, split=split_name)
    items = {}
    for item in dataset:
        item_id = item.get('_id')
        if item_id is not None:
            items[item_id] = dict(item)
    return items


def load_local_json_items(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file)

    items = {}
    for item in data:
        item_id = item.get('_id')
        if item_id is not None:
            items[item_id] = item
    return items


def find_local_longbench_cache():
    candidates = [
        '/root/.cache/huggingface/hub/datasets--THUDM--LongBench-v2/snapshots/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json',
        '/root/.cache/huggingface/hub/datasets--zai-org--LongBench-v2/snapshots/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ''


def resolve_model_path(model_name, model_map):
    return model_map.get(model_name, model_name)


def derive_input_label(path, default_label):
    if not path:
        return default_label
    filename = os.path.basename(path)
    label, _ = os.path.splitext(filename)
    return label or default_label


def resolve_summary_sources(summary_paths, summary_labels):
    if summary_labels and len(summary_labels) != len(summary_paths):
        raise ValueError('--summary_input_label 的数量必须和 --summary_jsonl 一致。')

    sources = []
    for index, path in enumerate(summary_paths):
        label = summary_labels[index] if summary_labels else derive_input_label(path, f'summary_input_{index + 1}')
        sources.append({
            'path': path,
            'label': label,
        })
    return sources


def count_tokens(tokenizer, text):
    text = text or ''
    if WORKER_TOKENIZER_KIND == 'hf':
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(tokenizer.encode(text, disallowed_special=()))


def build_prompt(template, item, context):
    return (
        template
        .replace('$DOC$', (context or '').strip())
        .replace('$Q$', (item.get('question', '') or '').strip())
        .replace('$C_A$', (item.get('choice_A', '') or '').strip())
        .replace('$C_B$', (item.get('choice_B', '') or '').strip())
        .replace('$C_C$', (item.get('choice_C', '') or '').strip())
        .replace('$C_D$', (item.get('choice_D', '') or '').strip())
    )


def prepare_prompt(prompt, model_name, enable_thinking=False):
    prompt = (prompt or '').strip()
    if enable_thinking:
        return prompt
    if 'qwen3' in model_name.lower() and not prompt.startswith('/no_think') and not prompt.startswith('/think'):
        return '/no_think\n' + prompt
    return prompt


def decode_tokens(tokenizer, token_ids):
    if WORKER_TOKENIZER_KIND == 'hf':
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return tokenizer.decode(token_ids)


def truncate_prompt(prompt, tokenizer, max_len):
    if not max_len or max_len <= 0:
        return prompt
    token_ids = tokenizer.encode(prompt, add_special_tokens=False) if WORKER_TOKENIZER_KIND == 'hf' else tokenizer.encode(prompt, disallowed_special=())
    if len(token_ids) <= max_len:
        return prompt

    head_len = max_len // 2
    tail_len = max_len - head_len
    truncated_ids = token_ids[:head_len] + token_ids[-tail_len:]
    return decode_tokens(tokenizer, truncated_ids)


def init_worker(model_name, model_path, template, use_tiktoken):
    global WORKER_TOKENIZER, WORKER_TOKENIZER_KIND, WORKER_MODEL_NAME, WORKER_TEMPLATE
    WORKER_MODEL_NAME = model_name
    WORKER_TEMPLATE = template
    if use_tiktoken:
        if tiktoken is None:
            raise RuntimeError('tiktoken 未安装，无法为 GPT/o1 系列模型统计 token。')
        WORKER_TOKENIZER = tiktoken.encoding_for_model('gpt-4o-2024-08-06')
        WORKER_TOKENIZER_KIND = 'tiktoken'
        return

    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    WORKER_TOKENIZER.model_max_length = max(getattr(WORKER_TOKENIZER, 'model_max_length', 0), 10 ** 9)
    WORKER_TOKENIZER_KIND = 'hf'


def process_item(task):
    item_id, raw_item, summary_item, max_prompt_tokens, enable_thinking, raw_input_label, raw_input_source, summary_input_label, summary_input_source = task
    raw_context = raw_item.get('context', '')
    summary_context = summary_item.get('context', '')
    raw_tokens = count_tokens(WORKER_TOKENIZER, raw_item.get('context', ''))
    summary_tokens = count_tokens(WORKER_TOKENIZER, summary_item.get('context', ''))
    raw_prompt = prepare_prompt(build_prompt(WORKER_TEMPLATE, raw_item, raw_context), WORKER_MODEL_NAME, enable_thinking=enable_thinking)
    summary_prompt = prepare_prompt(build_prompt(WORKER_TEMPLATE, raw_item, summary_context), WORKER_MODEL_NAME, enable_thinking=enable_thinking)
    raw_prompt_full_tokens = count_tokens(WORKER_TOKENIZER, raw_prompt)
    summary_prompt_full_tokens = count_tokens(WORKER_TOKENIZER, summary_prompt)
    raw_prompt_final = truncate_prompt(raw_prompt, WORKER_TOKENIZER, max_prompt_tokens)
    summary_prompt_final = truncate_prompt(summary_prompt, WORKER_TOKENIZER, max_prompt_tokens)
    raw_prompt_tokens = count_tokens(WORKER_TOKENIZER, raw_prompt_final)
    summary_prompt_tokens = count_tokens(WORKER_TOKENIZER, summary_prompt_final)
    delta_tokens = raw_tokens - summary_tokens
    ratio = round(summary_tokens / raw_tokens, 6) if raw_tokens else ''
    prompt_delta_tokens = raw_prompt_tokens - summary_prompt_tokens
    prompt_ratio = round(summary_prompt_tokens / raw_prompt_tokens, 6) if raw_prompt_tokens else ''
    return {
        '_id': item_id,
        'raw_input_label': raw_input_label,
        'raw_input_source': raw_input_source,
        'summary_input_label': summary_input_label,
        'summary_input_source': summary_input_source,
        'domain': raw_item.get('domain', ''),
        'sub_domain': raw_item.get('sub_domain', ''),
        'difficulty': raw_item.get('difficulty', ''),
        'length': raw_item.get('length', ''),
        'original_input_context_tokens': raw_tokens,
        'summary_input_context_tokens': summary_tokens,
        'raw_context_tokens': raw_tokens,
        'summary_context_tokens': summary_tokens,
        'delta_tokens': delta_tokens,
        'summary_to_raw_ratio': ratio,
        'original_input_prompt_tokens_full': raw_prompt_full_tokens,
        'summary_input_prompt_tokens_full': summary_prompt_full_tokens,
        'raw_prompt_tokens_full': raw_prompt_full_tokens,
        'summary_prompt_tokens_full': summary_prompt_full_tokens,
        'original_input_prompt_tokens': raw_prompt_tokens,
        'summary_input_prompt_tokens': summary_prompt_tokens,
        'raw_prompt_tokens': raw_prompt_tokens,
        'summary_prompt_tokens': summary_prompt_tokens,
        'prompt_delta_tokens': prompt_delta_tokens,
        'summary_prompt_to_raw_ratio': prompt_ratio,
    }


def resolve_worker_count(requested_workers, task_count):
    cpu_count = os.cpu_count() or 1
    if task_count <= 0:
        return 1
    return max(1, min(requested_workers, cpu_count, task_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_jsonl', default='', help='Original input jsonl with full raw context')
    parser.add_argument('--raw_local_json', default='', help='Local Hugging Face cached data.json path for LongBench-v2')
    parser.add_argument('--summary_jsonl', nargs='+', required=True, help='One or more summarized input jsonl files with replaced context')
    parser.add_argument('--raw_input_label', default='original_input', help='Label written to CSV for the original input side')
    parser.add_argument('--summary_input_label', nargs='*', default=[], help='Labels written to CSV for summarized inputs; defaults to each summary jsonl filename')
    parser.add_argument('--model', default='Qwen3-30B-A3B-local', help='Model name in config/model2path.json or a direct model path')
    parser.add_argument('--model_config', default='config/model2path.json')
    parser.add_argument('--max_prompt_tokens', type=int, default=64000, help='Use the same truncation limit as pred.py')
    parser.add_argument('--prompt_style', choices=['0shot', 'cot', 'no_context'], default='0shot')
    parser.add_argument('--raw_hf_dataset', default='THUDM/LongBench-v2', help='HF dataset to use when --raw_jsonl is not provided')
    parser.add_argument('--raw_hf_split', default='train', help='HF split to use when --raw_jsonl is not provided')
    parser.add_argument('--output_csv', default='results/token_input_comparison.csv')
    parser.add_argument('--progress_every', type=int, default=20, help='Print progress every N compared samples')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of worker processes for token counting')
    args = parser.parse_args()

    model_map = load_model_map(args.model_config)
    model_path = resolve_model_path(args.model, model_map)
    prompt_template = load_prompt_template(args.prompt_style)
    use_tiktoken = 'gpt' in args.model.lower() or 'o1' in args.model.lower()
    enable_thinking = args.prompt_style == 'cot'
    raw_input_label = args.raw_input_label
    summary_sources = resolve_summary_sources(args.summary_jsonl, args.summary_input_label)

    local_cache_path = args.raw_local_json or find_local_longbench_cache()

    if args.raw_jsonl:
        raw_items = load_jsonl(args.raw_jsonl)
        raw_input_source = args.raw_jsonl
    elif local_cache_path:
        raw_items = load_local_json_items(local_cache_path)
        raw_input_source = local_cache_path
    else:
        raw_items = load_hf_items(args.raw_hf_dataset, args.raw_hf_split)
        raw_input_source = f'{args.raw_hf_dataset}:{args.raw_hf_split}'
    tasks = []
    summary_sample_counts = {}
    for source in summary_sources:
        summary_items = load_jsonl(source['path'])
        common_ids = [item_id for item_id in raw_items if item_id in summary_items]
        summary_sample_counts[source['label']] = len(common_ids)
        tasks.extend([
            (
                item_id,
                raw_items[item_id],
                summary_items[item_id],
                args.max_prompt_tokens,
                enable_thinking,
                raw_input_label,
                raw_input_source,
                source['label'],
                source['path'],
            )
            for item_id in common_ids
        ])

    start_time = time.time()

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)

    total_raw_tokens = 0
    total_summary_tokens = 0
    total_raw_prompt_tokens_full = 0
    total_summary_prompt_tokens_full = 0
    total_raw_prompt_tokens = 0
    total_summary_prompt_tokens = 0
    rows_per_label = defaultdict(int)
    worker_count = resolve_worker_count(args.num_workers, len(tasks))

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            '_id',
            'raw_input_label',
            'raw_input_source',
            'summary_input_label',
            'summary_input_source',
            'domain',
            'sub_domain',
            'difficulty',
            'length',
            'original_input_context_tokens',
            'summary_input_context_tokens',
            'raw_context_tokens',
            'summary_context_tokens',
            'delta_tokens',
            'summary_to_raw_ratio',
            'original_input_prompt_tokens_full',
            'summary_input_prompt_tokens_full',
            'raw_prompt_tokens_full',
            'summary_prompt_tokens_full',
            'original_input_prompt_tokens',
            'summary_input_prompt_tokens',
            'raw_prompt_tokens',
            'summary_prompt_tokens',
            'prompt_delta_tokens',
            'summary_prompt_to_raw_ratio',
        ])

        if worker_count <= 1:
            init_worker(args.model, model_path, prompt_template, use_tiktoken)
            for index, task in enumerate(tasks, start=1):
                result = process_item(task)
                total_raw_tokens += result['raw_context_tokens']
                total_summary_tokens += result['summary_context_tokens']
                total_raw_prompt_tokens_full += result['raw_prompt_tokens_full']
                total_summary_prompt_tokens_full += result['summary_prompt_tokens_full']
                total_raw_prompt_tokens += result['raw_prompt_tokens']
                total_summary_prompt_tokens += result['summary_prompt_tokens']
                rows_per_label[result['summary_input_label']] += 1
                writer.writerow([
                    result['_id'],
                    result['raw_input_label'],
                    result['raw_input_source'],
                    result['summary_input_label'],
                    result['summary_input_source'],
                    result['domain'],
                    result['sub_domain'],
                    result['difficulty'],
                    result['length'],
                    result['original_input_context_tokens'],
                    result['summary_input_context_tokens'],
                    result['raw_context_tokens'],
                    result['summary_context_tokens'],
                    result['delta_tokens'],
                    result['summary_to_raw_ratio'],
                    result['original_input_prompt_tokens_full'],
                    result['summary_input_prompt_tokens_full'],
                    result['raw_prompt_tokens_full'],
                    result['summary_prompt_tokens_full'],
                    result['original_input_prompt_tokens'],
                    result['summary_input_prompt_tokens'],
                    result['raw_prompt_tokens'],
                    result['summary_prompt_tokens'],
                    result['prompt_delta_tokens'],
                    result['summary_prompt_to_raw_ratio'],
                ])
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(tasks)):
                    elapsed = round(time.time() - start_time, 2)
                    print(f'progress={index}/{len(tasks)} elapsed_s={elapsed}')
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_worker,
                initargs=(args.model, model_path, prompt_template, use_tiktoken),
            ) as executor:
                futures = [executor.submit(process_item, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    total_raw_tokens += result['raw_context_tokens']
                    total_summary_tokens += result['summary_context_tokens']
                    total_raw_prompt_tokens_full += result['raw_prompt_tokens_full']
                    total_summary_prompt_tokens_full += result['summary_prompt_tokens_full']
                    total_raw_prompt_tokens += result['raw_prompt_tokens']
                    total_summary_prompt_tokens += result['summary_prompt_tokens']
                    rows_per_label[result['summary_input_label']] += 1
                    writer.writerow([
                        result['_id'],
                        result['raw_input_label'],
                        result['raw_input_source'],
                        result['summary_input_label'],
                        result['summary_input_source'],
                        result['domain'],
                        result['sub_domain'],
                        result['difficulty'],
                        result['length'],
                        result['original_input_context_tokens'],
                        result['summary_input_context_tokens'],
                        result['raw_context_tokens'],
                        result['summary_context_tokens'],
                        result['delta_tokens'],
                        result['summary_to_raw_ratio'],
                        result['original_input_prompt_tokens_full'],
                        result['summary_input_prompt_tokens_full'],
                        result['raw_prompt_tokens_full'],
                        result['summary_prompt_tokens_full'],
                        result['original_input_prompt_tokens'],
                        result['summary_input_prompt_tokens'],
                        result['raw_prompt_tokens'],
                        result['summary_prompt_tokens'],
                        result['prompt_delta_tokens'],
                        result['summary_prompt_to_raw_ratio'],
                    ])
                    if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(tasks)):
                        elapsed = round(time.time() - start_time, 2)
                        print(f'progress={index}/{len(tasks)} elapsed_s={elapsed}')

    sample_count = len(tasks)
    avg_raw_tokens = round(total_raw_tokens / sample_count, 3) if sample_count else 0
    avg_summary_tokens = round(total_summary_tokens / sample_count, 3) if sample_count else 0
    avg_saved_tokens = round((total_raw_tokens - total_summary_tokens) / sample_count, 3) if sample_count else 0
    compression_ratio = round(total_summary_tokens / total_raw_tokens, 6) if total_raw_tokens else 0
    avg_raw_prompt_tokens_full = round(total_raw_prompt_tokens_full / sample_count, 3) if sample_count else 0
    avg_summary_prompt_tokens_full = round(total_summary_prompt_tokens_full / sample_count, 3) if sample_count else 0
    avg_raw_prompt_tokens = round(total_raw_prompt_tokens / sample_count, 3) if sample_count else 0
    avg_summary_prompt_tokens = round(total_summary_prompt_tokens / sample_count, 3) if sample_count else 0
    avg_saved_prompt_tokens = round((total_raw_prompt_tokens - total_summary_prompt_tokens) / sample_count, 3) if sample_count else 0
    prompt_compression_ratio_full = round(total_summary_prompt_tokens_full / total_raw_prompt_tokens_full, 6) if total_raw_prompt_tokens_full else 0
    prompt_compression_ratio = round(total_summary_prompt_tokens / total_raw_prompt_tokens, 6) if total_raw_prompt_tokens else 0

    print(f'model_path={model_path}')
    print(f'num_workers={worker_count}')
    print(f'prompt_style={args.prompt_style}')
    print(f'max_prompt_tokens={args.max_prompt_tokens}')
    print(f'raw_input_label={raw_input_label}')
    print(f'raw_input_source={raw_input_source}')
    print(f'summary_inputs={len(summary_sources)}')
    for source in summary_sources:
        print(f'summary_input={source["label"]} path={source["path"]} matched_samples={summary_sample_counts[source["label"]]} written_rows={rows_per_label[source["label"]]}')
    if local_cache_path:
        print(f'raw_local_json={local_cache_path}')
    print(f'compared_samples={sample_count}')
    print(f'total_raw_tokens={total_raw_tokens}')
    print(f'total_summary_tokens={total_summary_tokens}')
    print(f'avg_raw_tokens={avg_raw_tokens}')
    print(f'avg_summary_tokens={avg_summary_tokens}')
    print(f'avg_saved_tokens={avg_saved_tokens}')
    print(f'compression_ratio={compression_ratio}')
    print(f'total_raw_prompt_tokens_full={total_raw_prompt_tokens_full}')
    print(f'total_summary_prompt_tokens_full={total_summary_prompt_tokens_full}')
    print(f'avg_raw_prompt_tokens_full={avg_raw_prompt_tokens_full}')
    print(f'avg_summary_prompt_tokens_full={avg_summary_prompt_tokens_full}')
    print(f'prompt_compression_ratio_full={prompt_compression_ratio_full}')
    print(f'total_raw_prompt_tokens={total_raw_prompt_tokens}')
    print(f'total_summary_prompt_tokens={total_summary_prompt_tokens}')
    print(f'avg_raw_prompt_tokens={avg_raw_prompt_tokens}')
    print(f'avg_summary_prompt_tokens={avg_summary_prompt_tokens}')
    print(f'avg_saved_prompt_tokens={avg_saved_prompt_tokens}')
    print(f'prompt_compression_ratio={prompt_compression_ratio}')
    print(f'output_csv={args.output_csv}')
    print(f'total_elapsed_s={round(time.time() - start_time, 2)}')


if __name__ == '__main__':
    main()