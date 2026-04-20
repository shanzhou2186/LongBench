#!/usr/bin/env python3

from summarize_common import build_parser, call_chat_completion, clip_text_by_chars, run_pipeline


SYSTEM_PROMPT = (
    "你是选择题相关证据过滤助手。"
    "你的任务不是总结全文，而是只保留与题目和候选说法判别直接相关的证据。"
    "所有无关背景、铺垫、重复叙述、宽泛结论都要删掉。"
    "不要输出 JSON，不要给出最终选项，不要出现 A/B/C/D。"
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
请只输出与题目和候选说法直接相关的证据摘要，忽略全文概括。
摘要应只包含：
1. 可以支持或排除候选说法的事实
2. 数字、主体、时间、条件、比较关系
3. 原文没有提供、因此无法判断的部分

输出格式：
- 关键证据：...
- 关键限制：...
- 不确定信息：...

不要给出最终选项判断。
请尽量压缩到 {summary_char_limit} 字符以内。
"""


def summarize_one_item(item, args):
    user_prompt = USER_TEMPLATE.format(
        question=item.get("question", ""),
        choice_A=item.get("choice_A", ""),
        choice_B=item.get("choice_B", ""),
        choice_C=item.get("choice_C", ""),
        choice_D=item.get("choice_D", ""),
        context=item.get("context", ""),
        summary_char_limit=args.summary_char_limit,
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
    return clip_text_by_chars(raw, args.summary_char_limit)


def main():
    parser = build_parser("Keep only evidence directly relevant to the question and candidate statements.")
    parser.add_argument("--summary_char_limit", type=int, default=2500)
    parser.add_argument("--max_summary_tokens", type=int, default=1000)
    args = parser.parse_args()
    run_pipeline(args, summarize_one_item)


if __name__ == "__main__":
    main()