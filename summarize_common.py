#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List

import requests

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


DEFAULT_BASE_URL = os.environ.get("SGLANG_URL", os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1"))


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def ensure_server_ready(base_url: str, timeout: int = 5) -> None:
    models_url = normalize_base_url(base_url) + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Unexpected HTTP status {response.status} from {models_url}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach sglang OpenAI-compatible endpoint at {models_url}. "
            "Check your local server, port, and --base_url setting."
        ) from exc


def maybe_disable_qwen3_thinking(text: str, disable_thinking: bool) -> str:
    text = (text or "").strip()
    if not disable_thinking:
        return text
    if text.startswith("/no_think") or text.startswith("/think"):
        return text
    return "/no_think\n" + text


def strip_think_blocks(text: str) -> str:
    text = text or ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def sanitize_summary(text: str) -> str:
    text = strip_think_blocks(text)
    forbidden_line_patterns = [
        r'正确答案',
        r'答案是',
        r'the correct answer',
        r'correct answer',
        r'option\s*[A-D]',
        r'选项\s*[A-DＡ-Ｄ]',
        r'选\s*[A-DＡ-Ｄ]',
        r'^[A-D][\s\.:：、）)]',
        r'^\(?[A-D]\)?$',
    ]
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append(raw_line)
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in forbidden_line_patterns):
            continue
        cleaned_lines.append(raw_line)

    text = '\n'.join(cleaned_lines).strip()
    substitutions = [
        (r'选项\s*[A-DＡ-Ｄ]', '候选说法'),
        (r'option\s*[A-D]', 'candidate statement'),
        (r'\b[A-D]\b(?=\s*选项)', ''),
    ]
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clip_text_by_chars(text: str, char_limit: int) -> str:
    text = sanitize_summary(text)
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    clipped = text[:char_limit].rstrip()
    last_break = max(clipped.rfind("\n"), clipped.rfind("。"), clipped.rfind("；"))
    if last_break >= char_limit // 2:
        clipped = clipped[: last_break + 1].rstrip()
    return clipped


def get_tokenizer(model_path: str):
    if AutoTokenizer is None:
        raise RuntimeError("transformers 未安装，无法按 token 控制输出长度。")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.model_max_length = max(getattr(tokenizer, 'model_max_length', 0), 10 ** 9)
    return tokenizer


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def clip_text_by_tokens(text: str, tokenizer, token_limit: int) -> str:
    text = sanitize_summary(text)
    if token_limit <= 0:
        return text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) <= token_limit:
        return text
    clipped = tokenizer.decode(input_ids[:token_limit], skip_special_tokens=True)
    return sanitize_summary(clipped)


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = text or ""
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]
    overlap_chars = max(0, min(overlap_chars, chunk_chars // 2))
    step = max(1, chunk_chars - overlap_chars)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def call_chat_completion(
    base_url: str,
    chat_path: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 900,
    timeout_s: int = 600,
    disable_thinking: bool = True,
) -> str:
    base_url = normalize_base_url(base_url)
    if chat_path.startswith("/v1/"):
        chat_path = chat_path[3:]
    url = base_url.rstrip("/") + "/" + chat_path.lstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_messages = list(messages)
    if request_messages:
        last_message = dict(request_messages[-1])
        if last_message.get("role") == "user":
            last_message["content"] = maybe_disable_qwen3_thinking(last_message.get("content", ""), disable_thinking)
            request_messages[-1] = last_message

    payload = {
        "model": model,
        "messages": request_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} from {url}\n"
            f"Response: {response.text[:2000]}\n"
            "Hint: check --chat_path and whether your SGLang server exposes OpenAI-compatible endpoint."
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]


def load_input_items(args) -> List[Dict[str, Any]]:
    if args.input_jsonl:
        items = []
        with open(args.input_jsonl, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if not load_dataset:
        raise RuntimeError("datasets 未安装，且未提供 --input_jsonl。请 pip install datasets 或提供本地 jsonl。")

    dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    return [dict(item) for item in dataset]


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--hf_dataset", type=str, default="THUDM/LongBench-v2")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--input_jsonl", type=str, default="")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--chat_path", type=str, default="/chat/completions")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--sleep_s", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=20)
    return parser


def run_pipeline(args, summarize_fn) -> None:
    ensure_server_ready(args.base_url)
    items = load_input_items(args)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    output_root, output_ext = os.path.splitext(args.output_jsonl)
    error_jsonl = output_root + "_errors" + (output_ext or ".jsonl")
    success_count = 0
    error_count = 0
    progress = tqdm(total=len(items), desc="Summarizing", unit="item") if tqdm is not None else None

    try:
        with open(args.output_jsonl, "w", encoding="utf-8") as output_file, open(error_jsonl, "w", encoding="utf-8") as error_file:
            for index, item in enumerate(items, start=1):
                try:
                    summary = summarize_fn(item, args)
                    output_item = dict(item)
                    output_item["context"] = summary
                    output_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    output_file.flush()
                    success_count += 1
                except Exception as exc:
                    output_item = dict(item)
                    output_item["_summarize_error"] = str(exc)
                    error_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    error_file.flush()
                    error_count += 1

                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(success=success_count, error=error_count)
                elif args.progress_every > 0 and index % args.progress_every == 0:
                    print(f"[OK] processed {index}/{len(items)} | success={success_count} error={error_count}")
    finally:
        if progress is not None:
            progress.close()

    print(f"[DONE] wrote summaries: {args.output_jsonl}")
    print(f"[DONE] wrote errors: {error_jsonl}")
    print(f"[STATS] success={success_count} error={error_count} total={len(items)}")