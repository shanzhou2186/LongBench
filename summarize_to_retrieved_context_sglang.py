#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""直接摘要版脚本。

这是最直接的一种方法：把整篇原文连同问题和选项一起送给小模型，让它一次性生成高密度摘要。
它不做分块、不做层次合并，流程最简单，通常可作为其他更复杂方法的对照基线。
"""

import os
import json
import time
import argparse
import re
import urllib.request
from typing import List, Dict, Any, Optional

import requests

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# -----------------------
# Prompt for direct summary generation
# -----------------------
SUMMARIZER_SYSTEM = (
    "你是“长文档阅读助手”。你的任务是：只基于给定的原文内容，"
    "为多选题生成一份高密度摘要，供后续人工或模型直接阅读。"
    "严格要求：不要编造原文没有的信息；如果原文不足以支持某个结论，"
    "请明确写“原文未提及/不确定”。输出必须是纯文本摘要，不要输出 JSON，不要输出代码块。"
    "绝对不要判断哪个选项正确，不要出现 A/B/C/D 选项字母，不要出现“正确答案”“答案是”“选项A/选项B/选项C/选项D”等表述。"
)

SUMMARIZER_USER_TMPL = """【输入】
问题：{question}
选项：
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

原文（很长）：{context}

【输出要求】
请直接输出一份中文摘要，长度控制在 {summary_char_limit} 字符以内，目标接近但不要明显超过该长度。
摘要应尽量包含：
1. 与问题直接相关的核心事实
2. 能区分各候选说法的关键信息，但只能描述内容差异，不能引用选项字母
3. 必要的条件关系、时间线、因果链
4. 原文未提及或无法确定的信息

不要输出 JSON，不要复述题目模板，不要输出“以下是摘要”。
不要写“选项A/选项B/选项C/选项D”、不要写“正确答案”、不要写“答案是”、不要给出最终选项判断。
"""


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def ensure_server_ready(base_url: str, timeout: int = 5) -> None:
    # 运行前先确认本地 OpenAI-compatible 服务已经启动。
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
    # 清理思维链和直接答案表达，避免摘要中泄露选项结论。
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
    substitution_patterns = [
        (r'选项\s*[A-DＡ-Ｄ]', '候选说法'),
        (r'option\s*[A-D]', 'candidate statement'),
        (r'\b[A-D]\b(?=\s*选项)', ''),
    ]
    for pattern, replacement in substitution_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clip_summary(text: str, summary_char_limit: int) -> str:
    # 直接摘要版只做字符级裁剪，尽量把输出限制在设定长度附近。
    text = sanitize_summary(text)
    if len(text) <= summary_char_limit:
        return text
    clipped = text[:summary_char_limit].rstrip()
    last_break = max(clipped.rfind("\n"), clipped.rfind("。"), clipped.rfind("；"))
    if last_break >= summary_char_limit // 2:
        clipped = clipped[: last_break + 1].rstrip()
    return clipped


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
    # 调用本地 SGLang/OpenAI-compatible 接口获取单次摘要结果。
    base_url = normalize_base_url(base_url)
    if chat_path.startswith("/v1/"):
        chat_path = chat_path[3:]
    url = base_url.rstrip("/") + "/" + chat_path.lstrip("/")
    headers = {"Content-Type": "application/json"}
    # Many local services ignore api_key, but keep it for compatibility
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

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code} from {url}\n"
            f"Response: {resp.text[:2000]}\n"
            f"Hint: check --chat_path and whether your SGLang server exposes OpenAI-compatible endpoint."
        )
    data = resp.json()
    # OpenAI-compatible format:
    return data["choices"][0]["message"]["content"]


def summarize_one_item(
    item: Dict[str, Any],
    base_url: str,
    chat_path: str,
    api_key: str,
    model: str,
    max_summary_tokens: int,
    summary_char_limit: int,
    disable_thinking: bool,
) -> str:
    # 直接把整篇上下文和题目一起送入模型，让模型一次性输出高密度摘要。
    user_prompt = SUMMARIZER_USER_TMPL.format(
        question=item.get("question", ""),
        choice_A=item.get("choice_A", ""),
        choice_B=item.get("choice_B", ""),
        choice_C=item.get("choice_C", ""),
        choice_D=item.get("choice_D", ""),
        context=item.get("context", ""),
        summary_char_limit=summary_char_limit,
    )
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    raw = call_chat_completion(
        base_url=base_url,
        chat_path=chat_path,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_summary_tokens,
        disable_thinking=disable_thinking,
    )
    return clip_summary(raw, summary_char_limit)


def load_input_items(args) -> List[Dict[str, Any]]:
    # 支持从本地 jsonl 读取，也支持直接从 Hugging Face 数据集读取。
    if args.input_jsonl:
        items = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if not load_dataset:
        raise RuntimeError("datasets 未安装，且未提供 --input_jsonl。请 pip install datasets 或提供本地 jsonl。")

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    return [dict(x) for x in ds]


def main():
    # 这个脚本没有复杂流程，主要参数就是输入来源、输出文件和摘要长度控制。
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", type=str, default="THUDM/LongBench-v2")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--input_jsonl", type=str, default="")
    ap.add_argument("--output_jsonl", type=str, required=True)

    ap.add_argument("--base_url", type=str, default=os.environ.get("SGLANG_URL", os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")))
    ap.add_argument("--chat_path", type=str, default="/chat/completions",
                    help="SGLang/OpenAI-compatible chat endpoint path. Default: /chat/completions")
    ap.add_argument("--api_key", type=str, default="EMPTY")
    ap.add_argument("--model", type=str, required=True,
                    help="Use the model name exposed by your SGLang server (served-model-name if set).")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Allow Qwen3 thinking mode. By default the script prefixes /no_think to keep JSON output stable.")

    ap.add_argument("--summary_char_limit", type=int, default=5000)
    ap.add_argument("--max_summary_tokens", type=int, default=1800)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()
    ensure_server_ready(args.base_url)

    items = load_input_items(args)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for i, item in enumerate(items, 1):
            try:
                summary = summarize_one_item(
                    item=item,
                    base_url=args.base_url,
                    chat_path=args.chat_path,
                    api_key=args.api_key,
                    model=args.model,
                    max_summary_tokens=args.max_summary_tokens,
                    summary_char_limit=args.summary_char_limit,
                    disable_thinking=not args.enable_thinking,
                )
                output_item = dict(item)
                output_item["context"] = summary
                out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            except Exception as e:
                output_item = dict(item)
                output_item["_summarize_error"] = str(e)
                out.write(json.dumps(output_item, ensure_ascii=False) + "\n")

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

            if i % 20 == 0:
                print(f"[OK] processed {i}/{len(items)}")

    print(f"[DONE] wrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()