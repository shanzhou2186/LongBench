#!/usr/bin/env python3

import json

from summarize_common import build_parser, call_chat_completion, clip_text_by_tokens, get_tokenizer, run_pipeline


SYSTEM_PROMPT = (
    "你是多选题证据压缩助手。"
    "你的输出必须优先满足 token 长度预算，而不是尽量保留全文信息。"
    "只保留与问题和候选说法判别直接相关的事实。"
    "不要输出 JSON，不要判断正确答案，不要出现 A/B/C/D。"
)


USER_TEMPLATE = """【问题】
{question}

【候选说法】
1. {choice_A}
2. {choice_B}
3. {choice_C}
4. {choice_D}

【原文】
{context}

【任务】
请输出一份中文压缩摘要，目标是让最终摘要尽量不超过 {target_tokens} 个 token。
优先保留：
1. 能直接支持或排除候选说法的事实
2. 时间、条件、因果、数量、主体关系
3. 原文未提及/不确定的点

不要给出最终选项判断。
"""


TOKENIZER = None


def summarize_one_item(item, args):
    user_prompt = USER_TEMPLATE.format(
        question=item.get("question", ""),
        choice_A=item.get("choice_A", ""),
        choice_B=item.get("choice_B", ""),
        choice_C=item.get("choice_C", ""),
        choice_D=item.get("choice_D", ""),
        context=item.get("context", ""),
        target_tokens=args.target_tokens,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_chat_completion(
        base_url=args.base_url,
        chat_path=args.chat_path,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        temperature=0.0,
        max_tokens=args.max_summary_tokens,
        disable_thinking=not args.enable_thinking,
    )
    return clip_text_by_tokens(raw, TOKENIZER, args.target_tokens)


def main():
    global TOKENIZER
    parser = build_parser("Constrain summary output by token budget, such as 8k, 16k, or 32k tokens.")
    parser.add_argument("--target_tokens", type=int, default=8192)
    parser.add_argument("--max_summary_tokens", type=int, default=12000)
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    model_path = args.model_path or args.model
    if args.model_path:
        TOKENIZER = get_tokenizer(args.model_path)
    else:
        try:
            with open("config/model2path.json", encoding="utf-8") as file:
                model_map = json.load(file)
            TOKENIZER = get_tokenizer(model_map.get(args.model, args.model))
        except FileNotFoundError:
            TOKENIZER = get_tokenizer(model_path)

    run_pipeline(args, summarize_one_item)


if __name__ == "__main__":
    main()