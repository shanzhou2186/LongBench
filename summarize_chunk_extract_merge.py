#!/usr/bin/env python3

from summarize_common import build_parser, call_chat_completion, chunk_text, clip_text_by_chars, run_pipeline


CHUNK_SYSTEM_PROMPT = (
    "你是多选题证据抽取助手。"
    "给你原文中的一个片段时，你只抽取和问题及候选说法直接相关的证据。"
    "不要概括整篇文章，不要保留无关背景，不要判断正确答案，不要出现 A/B/C/D。"
)

CHUNK_USER_TEMPLATE = """【问题】
{question}

【候选说法】
1. {choice_A}
2. {choice_B}
3. {choice_C}
4. {choice_D}

【原文片段】
{chunk}

【任务】
只抽取这个片段中与解题直接相关的事实、数字、时间、条件、主体关系、否定信息。
如果这个片段没有关键证据，输出“无关键证据”。
不要给出最终选项判断。
"""

MERGE_SYSTEM_PROMPT = (
    "你是多选题证据整理助手。"
    "你会收到多个片段证据，请去重、压缩、合并成一份可直接用于答题的短摘要。"
    "只保留关键证据，不给出最终选项判断，不出现 A/B/C/D。"
)

MERGE_USER_TEMPLATE = """【问题】
{question}

【候选说法】
1. {choice_A}
2. {choice_B}
3. {choice_C}
4. {choice_D}

【片段证据】
{evidence_text}

【任务】
请去重并合并成一份中文短摘要，只保留能区分候选说法的关键信息。
若某些信息仍无法判断，请写“原文未提及/不确定”。
输出控制在 {summary_char_limit} 字符以内。
"""


def summarize_one_item(item, args):
    chunks = chunk_text(item.get("context", ""), args.chunk_chars, args.overlap_chars)
    evidence_parts = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = CHUNK_USER_TEMPLATE.format(
            question=item.get("question", ""),
            choice_A=item.get("choice_A", ""),
            choice_B=item.get("choice_B", ""),
            choice_C=item.get("choice_C", ""),
            choice_D=item.get("choice_D", ""),
            chunk=chunk,
        )
        chunk_messages = [
            {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
            {"role": "user", "content": chunk_prompt},
        ]
        chunk_summary = call_chat_completion(
            base_url=args.base_url,
            chat_path=args.chat_path,
            api_key=args.api_key,
            model=args.model,
            messages=chunk_messages,
            temperature=0.0,
            max_tokens=args.chunk_extract_tokens,
            disable_thinking=not args.enable_thinking,
        )
        evidence_parts.append(f"片段{index}: {chunk_summary}")

    merge_prompt = MERGE_USER_TEMPLATE.format(
        question=item.get("question", ""),
        choice_A=item.get("choice_A", ""),
        choice_B=item.get("choice_B", ""),
        choice_C=item.get("choice_C", ""),
        choice_D=item.get("choice_D", ""),
        evidence_text="\n\n".join(evidence_parts),
        summary_char_limit=args.summary_char_limit,
    )
    merge_messages = [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": merge_prompt},
    ]
    merged = call_chat_completion(
        base_url=args.base_url,
        chat_path=args.chat_path,
        api_key=args.api_key,
        model=args.model,
        messages=merge_messages,
        temperature=0.0,
        max_tokens=args.max_summary_tokens,
        disable_thinking=not args.enable_thinking,
    )
    return clip_text_by_chars(merged, args.summary_char_limit)


def main():
    parser = build_parser("Split document into chunks, extract question-relevant evidence per chunk, then merge.")
    parser.add_argument("--chunk_chars", type=int, default=12000)
    parser.add_argument("--overlap_chars", type=int, default=1000)
    parser.add_argument("--chunk_extract_tokens", type=int, default=400)
    parser.add_argument("--summary_char_limit", type=int, default=3000)
    parser.add_argument("--max_summary_tokens", type=int, default=1200)
    args = parser.parse_args()
    run_pipeline(args, summarize_one_item)


if __name__ == "__main__":
    main()