#!/usr/bin/env python3

import argparse
import csv
import os
from collections import defaultdict


NUMERIC_FIELDS = [
    'raw_context_tokens',
    'summary_context_tokens',
    'delta_tokens',
    'raw_prompt_tokens_full',
    'summary_prompt_tokens_full',
    'raw_prompt_tokens',
    'summary_prompt_tokens',
    'prompt_delta_tokens',
]


DEFAULT_GROUPS = ['summary_input_label', 'difficulty', 'length', 'domain', 'sub_domain']


def safe_ratio(numerator, denominator):
    if not denominator:
        return 0
    return round(numerator / denominator, 6)


def to_int(row, field):
    value = row.get(field, '')
    if value == '':
        return 0
    return int(value)


def load_rows(input_csv):
    with open(input_csv, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)


def summarize_group(rows, group_field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(group_field, '')].append(row)

    summary_rows = []
    for group_value in sorted(grouped):
        items = grouped[group_value]
        sample_count = len(items)
        metrics = {field: sum(to_int(row, field) for row in items) for field in NUMERIC_FIELDS}
        raw_truncated_samples = sum(to_int(row, 'raw_prompt_tokens_full') > to_int(row, 'raw_prompt_tokens') for row in items)
        summary_truncated_samples = sum(to_int(row, 'summary_prompt_tokens_full') > to_int(row, 'summary_prompt_tokens') for row in items)

        summary_rows.append({
            group_field: group_value,
            'sample_count': sample_count,
            'total_raw_context_tokens': metrics['raw_context_tokens'],
            'total_summary_context_tokens': metrics['summary_context_tokens'],
            'saved_context_tokens': metrics['delta_tokens'],
            'context_compression_ratio': safe_ratio(metrics['summary_context_tokens'], metrics['raw_context_tokens']),
            'avg_raw_context_tokens': round(metrics['raw_context_tokens'] / sample_count, 3),
            'avg_summary_context_tokens': round(metrics['summary_context_tokens'] / sample_count, 3),
            'avg_saved_context_tokens': round(metrics['delta_tokens'] / sample_count, 3),
            'total_raw_prompt_tokens_full': metrics['raw_prompt_tokens_full'],
            'total_summary_prompt_tokens_full': metrics['summary_prompt_tokens_full'],
            'saved_prompt_tokens_full': metrics['raw_prompt_tokens_full'] - metrics['summary_prompt_tokens_full'],
            'prompt_compression_ratio_full': safe_ratio(metrics['summary_prompt_tokens_full'], metrics['raw_prompt_tokens_full']),
            'avg_raw_prompt_tokens_full': round(metrics['raw_prompt_tokens_full'] / sample_count, 3),
            'avg_summary_prompt_tokens_full': round(metrics['summary_prompt_tokens_full'] / sample_count, 3),
            'total_raw_prompt_tokens': metrics['raw_prompt_tokens'],
            'total_summary_prompt_tokens': metrics['summary_prompt_tokens'],
            'saved_prompt_tokens': metrics['prompt_delta_tokens'],
            'prompt_compression_ratio': safe_ratio(metrics['summary_prompt_tokens'], metrics['raw_prompt_tokens']),
            'avg_raw_prompt_tokens': round(metrics['raw_prompt_tokens'] / sample_count, 3),
            'avg_summary_prompt_tokens': round(metrics['summary_prompt_tokens'] / sample_count, 3),
            'avg_saved_prompt_tokens': round(metrics['prompt_delta_tokens'] / sample_count, 3),
            'raw_truncated_samples': raw_truncated_samples,
            'summary_truncated_samples': summary_truncated_samples,
        })
    return summary_rows


def write_csv(output_csv, rows, group_field):
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = [
        group_field,
        'sample_count',
        'total_raw_context_tokens',
        'total_summary_context_tokens',
        'saved_context_tokens',
        'context_compression_ratio',
        'avg_raw_context_tokens',
        'avg_summary_context_tokens',
        'avg_saved_context_tokens',
        'total_raw_prompt_tokens_full',
        'total_summary_prompt_tokens_full',
        'saved_prompt_tokens_full',
        'prompt_compression_ratio_full',
        'avg_raw_prompt_tokens_full',
        'avg_summary_prompt_tokens_full',
        'total_raw_prompt_tokens',
        'total_summary_prompt_tokens',
        'saved_prompt_tokens',
        'prompt_compression_ratio',
        'avg_raw_prompt_tokens',
        'avg_summary_prompt_tokens',
        'avg_saved_prompt_tokens',
        'raw_truncated_samples',
        'summary_truncated_samples',
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_output_path(output_dir, input_csv, group_field):
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    return os.path.join(output_dir, f'{base_name}_by_{group_field}.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='results/token_input_comparison_merged.csv')
    parser.add_argument('--output_dir', default='results/grouped_token_stats')
    parser.add_argument('--group_by', nargs='+', default=DEFAULT_GROUPS)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    for group_field in args.group_by:
        summary_rows = summarize_group(rows, group_field)
        output_csv = build_output_path(args.output_dir, args.input_csv, group_field)
        write_csv(output_csv, summary_rows, group_field)
        print(f'group_by={group_field} output_csv={output_csv} groups={len(summary_rows)}')


if __name__ == '__main__':
    main()