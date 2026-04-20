#!/usr/bin/env python3

"""严格压缩版摘要脚本。

这个脚本的目标不是生成一份通用、可读性优先的摘要，而是用小模型把长上下文压缩成
“仍然足够支持后续答题模型推理”的证据摘要。

核心思路有两条路径：
1. 文本较短时，直接单次压缩。
2. 文本较长时，先分块抽取证据，再分层合并，最后按下游答题模型的 token 预算裁剪。

因此，这个脚本的压缩目标是“适配下游答题 prompt 的真实预算”，而不是只看固定字符数。
"""

import json
import os

from summarize_common import build_parser, call_chat_completion, chunk_text, clip_text_by_chars, clip_text_by_tokens, get_tokenizer, run_pipeline


SYSTEM_PROMPT = (
    "你是多选题长文压缩助手。"
    "目标不是保留尽量多信息，而是在不编造的前提下尽可能强压缩。"
    "只保留解题真正需要的事实、条件、时间线、因果关系和排除信息。"
    "删除背景铺垫、重复论述、修辞、泛泛总结和与题目无关的细节。"
    "不要输出 JSON，不要输出代码块，不要判断正确答案，不要出现 A/B/C/D 或‘正确答案’。"
)

CHUNK_SYSTEM_PROMPT = (
    "你是多选题证据抽取助手。"
    "给你原文中的一个片段时，你只抽取和问题及候选说法直接相关的证据。"
    "不要概括整篇文章，不要保留无关背景，不要判断正确答案，不要出现 A/B/C/D。"
)

MERGE_SYSTEM_PROMPT = (
    "你是多选题长文压缩助手。"
    "你会收到多个片段证据，请去重、压缩、合并成一份可直接用于答题的中文短摘要。"
    "只保留解题真正需要的事实、条件、时间线、因果关系和排除信息。"
    "不要给出最终选项判断，不要出现 A/B/C/D 或‘正确答案’。"
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
请把原文强压缩成一份中文摘要，尽量短，但保留解题必要信息。
要求：
1. 只保留与问题和候选说法判别直接相关的信息
2. 优先保留可区分候选说法的事实
3. 明确保留否定信息、限制条件、时间关系、因果关系
4. 对无法确定的部分写“原文未提及/不确定”
5. 不给出最终选项判断
6. 输出尽量压缩到 {summary_char_limit} 字符以内
"""

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
请去重并强压缩成一份中文短摘要，只保留能区分候选说法的关键信息。
若某些信息仍无法判断，请写“原文未提及/不确定”。
输出控制在 {summary_char_limit} 字符以内。
"""

INTERMEDIATE_MERGE_USER_TEMPLATE = """【问题】
{question}

【候选说法】
1. {choice_A}
2. {choice_B}
3. {choice_C}
4. {choice_D}

【部分片段证据】
{evidence_text}

【任务】
请先对这一部分片段证据去重并强压缩，只保留能区分候选说法的关键信息。
不要给出最终选项判断。
输出尽量压缩，便于后续继续合并。
"""


MODEL_MAP = {}
MAXLEN_MAP = {}
TARGET_TOKENIZER = None
TARGET_TEMPLATE = ''


def load_json_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as file:
        return json.load(file)


def load_prompt_template(prompt_style):
    template_map = {
        '0shot': 'prompts/0shot.txt',
        'cot': 'prompts/0shot_cot.txt',
        'no_context': 'prompts/0shot_no_context.txt',
    }
    with open(template_map[prompt_style], encoding='utf-8') as file:
        return file.read()


def resolve_target_model_path(args):
    if args.target_model_path:
        return args.target_model_path
    return MODEL_MAP.get(args.target_model, args.target_model)


def resolve_target_max_prompt_tokens(args):
    if args.target_max_prompt_tokens > 0:
        return args.target_max_prompt_tokens
    return MAXLEN_MAP.get(args.target_model, 0)


def prepare_target_prompt(prompt, model_name, enable_thinking=False):
    prompt = (prompt or '').strip()
    if enable_thinking:
        return prompt
    if 'qwen3' in model_name.lower() and not prompt.startswith('/no_think') and not prompt.startswith('/think'):
        return '/no_think\n' + prompt
    return prompt


def build_target_prompt(item, context, args):
    # 用下游答题模型的真实模板来估算 prompt 开销，避免摘要长度和实际答题输入预算脱节。
    prompt = (
        TARGET_TEMPLATE
        .replace('$DOC$', (context or '').strip())
        .replace('$Q$', (item.get('question', '') or '').strip())
        .replace('$C_A$', (item.get('choice_A', '') or '').strip())
        .replace('$C_B$', (item.get('choice_B', '') or '').strip())
        .replace('$C_C$', (item.get('choice_C', '') or '').strip())
        .replace('$C_D$', (item.get('choice_D', '') or '').strip())
    )
    return prepare_target_prompt(prompt, args.target_model, enable_thinking=args.target_prompt_style == 'cot')


def get_effective_summary_token_limit(item, args):
    if args.summary_token_limit > 0:
        return args.summary_token_limit
    if TARGET_TOKENIZER is None:
        return 0
    max_prompt_tokens = resolve_target_max_prompt_tokens(args)
    if max_prompt_tokens <= 0:
        return 0
    # 先扣掉题目、选项、模板本身占用的 token，再把剩余预算留给摘要正文。
    prompt_overhead = len(TARGET_TOKENIZER.encode(build_target_prompt(item, '', args), add_special_tokens=False))
    return max(0, max_prompt_tokens - prompt_overhead)


def get_effective_generation_tokens(item, args):
    if args.max_summary_tokens > 0:
        return args.max_summary_tokens
    token_limit = get_effective_summary_token_limit(item, args)
    if token_limit > 0:
        return token_limit
    return 900


def finalize_summary(text, item, args):
    # 生成结束后再按下游 tokenizer 做一次硬裁剪，确保最终摘要真的能塞进目标 prompt 预算。
    result = text
    token_limit = get_effective_summary_token_limit(item, args)
    if token_limit > 0 and TARGET_TOKENIZER is not None:
        result = clip_text_by_tokens(result, TARGET_TOKENIZER, token_limit)
    if args.summary_char_limit > 0:
        result = clip_text_by_chars(result, args.summary_char_limit)
    elif token_limit <= 0:
        result = clip_text_by_chars(result, 10 ** 9)
    return result


def merge_evidence_batch(evidence_parts, item, args, final_merge=False):
    # 长文不是一次性全部合并，而是先按批次压缩，避免 merge 阶段本身再次超长。
    template = MERGE_USER_TEMPLATE if final_merge else INTERMEDIATE_MERGE_USER_TEMPLATE
    merge_prompt = template.format(
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
    return call_chat_completion(
        base_url=args.base_url,
        chat_path=args.chat_path,
        api_key=args.api_key,
        model=args.model,
        messages=merge_messages,
        temperature=0.0,
        max_tokens=get_effective_generation_tokens(item, args),
        disable_thinking=not args.enable_thinking,
    )


def hierarchical_merge_evidence(evidence_parts, item, args):
    if not evidence_parts:
        return ""
    # 分层合并：把 chunk 级证据逐层压缩，直到只剩一份最终摘要。
    current_level = list(evidence_parts)
    level = 0
    while len(current_level) > 1:
        level += 1
        print(f"[MERGE] id={item.get('_id', '')} level={level} groups={len(current_level)}")
        next_level = []
        for batch_start in range(0, len(current_level), args.merge_batch_size):
            batch = current_level[batch_start:batch_start + args.merge_batch_size]
            merged_batch = merge_evidence_batch(batch, item, args, final_merge=False)
            next_level.append(merged_batch)
            if args.chunk_progress_every > 0:
                batch_index = batch_start // args.merge_batch_size + 1
                total_batches = (len(current_level) + args.merge_batch_size - 1) // args.merge_batch_size
                if batch_index % args.chunk_progress_every == 0 or batch_index == total_batches:
                    print(f"[MERGE] id={item.get('_id', '')} level={level} batch={batch_index}/{total_batches}")
        current_level = next_level
    final_text = merge_evidence_batch(current_level, item, args, final_merge=True)
    return finalize_summary(final_text, item, args)


def summarize_single_pass(item, args):
    # 如果原文还不算太长，直接单次压缩，减少分块带来的信息割裂。
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
        max_tokens=get_effective_generation_tokens(item, args),
        disable_thinking=not args.enable_thinking,
    )
    return finalize_summary(raw, item, args)


def summarize_chunked(item, args):
    # 对超长文本，先做“分块证据抽取”，再做“证据合并”，这是脚本处理长上下文的主路径。
    chunks = chunk_text(item.get("context", ""), args.chunk_chars, args.overlap_chars)
    print(f"[CHUNK] id={item.get('_id', '')} total_chunks={len(chunks)}")
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
        if args.chunk_progress_every > 0 and (index % args.chunk_progress_every == 0 or index == len(chunks)):
            print(f"[CHUNK] id={item.get('_id', '')} processed={index}/{len(chunks)}")
    return hierarchical_merge_evidence(evidence_parts, item, args)


def summarize_one_item(item, args):
    context = item.get("context", "") or ""
    # 这里按原始上下文长度切换策略：短文走单次压缩，长文走分块抽取+层次合并。
    if len(context) <= args.auto_chunk_chars:
        return summarize_single_pass(item, args)
    return summarize_chunked(item, args)


def main():
    global MODEL_MAP, MAXLEN_MAP, TARGET_TOKENIZER, TARGET_TEMPLATE
    # model 是负责生成摘要的小模型；target_model 是后续真正拿摘要去答题的大模型。
    # 这个脚本会按 target_model 的 tokenizer 和 prompt 模板来控制最终摘要预算。
    parser = build_parser("Strongly compress context while preserving only question-relevant evidence.")
    parser.add_argument("--auto_chunk_chars", type=int, default=12000)
    parser.add_argument("--chunk_chars", type=int, default=12000)
    parser.add_argument("--overlap_chars", type=int, default=1000)
    parser.add_argument("--chunk_extract_tokens", type=int, default=400)
    parser.add_argument("--chunk_progress_every", type=int, default=20)
    parser.add_argument("--merge_batch_size", type=int, default=24)
    parser.add_argument("--summary_char_limit", type=int, default=0,
                        help="Optional secondary char cap. Set 0 to disable fixed char clipping.")
    parser.add_argument("--summary_token_limit", type=int, default=0,
                        help="Hard token cap for the final summary under the target model tokenizer. Set 0 to auto-fit target prompt budget.")
    parser.add_argument("--max_summary_tokens", type=int, default=0,
                        help="Generation max tokens for merge/final summary. Set 0 to auto-follow the effective summary token limit.")
    parser.add_argument("--target_model", type=str, default=os.environ.get("TARGET_MODEL", "Qwen3-1.7B-local"),
                        help="Downstream answer model whose prompt budget the final summary should fit.")
    parser.add_argument("--target_model_path", type=str, default="",
                        help="Optional explicit tokenizer path for the downstream answer model.")
    parser.add_argument("--target_max_prompt_tokens", type=int, default=0,
                        help="Override downstream prompt token budget. If 0, read from config/model2maxlen.json.")
    parser.add_argument("--target_prompt_style", choices=["0shot", "cot", "no_context"], default="0shot",
                        help="Prompt template used by the downstream answer model when estimating remaining summary budget.")
    args = parser.parse_args()

    MODEL_MAP = load_json_file('config/model2path.json')
    MAXLEN_MAP = load_json_file('config/model2maxlen.json')
    TARGET_TEMPLATE = load_prompt_template(args.target_prompt_style)
    target_model_path = resolve_target_model_path(args)
    if target_model_path:
        TARGET_TOKENIZER = get_tokenizer(target_model_path)

    run_pipeline(args, summarize_one_item)


if __name__ == "__main__":
    main()