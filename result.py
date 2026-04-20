import argparse
import csv
import json
import os

from group_token_comparison import DEFAULT_GROUPS, build_output_path, summarize_group, write_csv


def load_prediction_file(filename):
    if not os.path.isfile(filename):
        return []
    if os.path.getsize(filename) == 0:
        return []

    with open(filename, encoding='utf-8') as file:
        text = file.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                return []
        return data

    if isinstance(data, list):
        return data
    return []


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return "NA"
    return str(round(100 * numerator / denominator, 1))


def collect_accuracy_rows(results_dir, compensated=False):
    files = sorted(os.listdir(results_dir))
    output = [["Model", "Overall", "Easy", "Hard", "Short", "Medium", "Long"]]

    for file in files:
        if file.startswith('.'):
            continue
        filename = os.path.join(results_dir, file)
        if not file.endswith('.json') and not file.endswith('.jsonl'):
            continue
        pred_data = load_prediction_file(filename)
        if not pred_data:
            continue
        sample = pred_data[0]
        required_fields = {"judge", "pred", "difficulty", "length"}
        if not required_fields.issubset(sample.keys()):
            continue

        easy, hard, short, medium, long = 0, 0, 0, 0, 0
        easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
        for pred in pred_data:
            acc = int(pred['judge'])
            if compensated and pred['pred'] is None:
                acc = 0.25
            if pred['difficulty'] == 'easy':
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred['length'] == 'short':
                short += 1
                short_acc += acc
            elif pred['length'] == 'medium':
                medium += 1
                medium_acc += acc
            else:
                long += 1
                long_acc += acc

        name = '.'.join(file.split('.')[:-1])
        output.append([
            name,
            safe_percentage(easy_acc + hard_acc, len(pred_data)),
            safe_percentage(easy_acc, easy),
            safe_percentage(hard_acc, hard),
            safe_percentage(short_acc, short),
            safe_percentage(medium_acc, medium),
            safe_percentage(long_acc, long),
        ])
    return output


def write_accuracy_table(rows, output_path):
    lines = ['\t'.join(row) for row in rows]
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def load_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as file:
        return list(csv.DictReader(file))


def generate_grouped_token_csvs(token_csv, output_dir, group_fields):
    rows = load_csv_rows(token_csv)
    if not rows:
        return {}

    generated = {}
    for group_field in group_fields:
        summary_rows = summarize_group(rows, group_field)
        output_csv = build_output_path(output_dir, token_csv, group_field)
        write_csv(output_csv, summary_rows, group_field)
        generated[group_field] = {
            'output_csv': output_csv,
            'rows': summary_rows,
        }
    return generated


def format_markdown_table(headers, rows):
    table = [
        ' | '.join(headers),
        ' | '.join(['---'] * len(headers)),
    ]
    for row in rows:
        table.append(' | '.join(row))
    return '\n'.join(table)


def format_integer(value):
    return f'{int(value):,}'


def format_ratio(value):
    return f'{float(value) * 100:.4f}%'


def unique_nonempty_values(rows, field):
    return sorted({row.get(field, '').strip() for row in rows if row.get(field, '').strip()})


def aggregate_token_rows(rows):
    total_raw_context_tokens = sum(int(row['raw_context_tokens']) for row in rows)
    total_summary_context_tokens = sum(int(row['summary_context_tokens']) for row in rows)
    total_raw_prompt_tokens_full = sum(int(row['raw_prompt_tokens_full']) for row in rows)
    total_summary_prompt_tokens_full = sum(int(row['summary_prompt_tokens_full']) for row in rows)
    total_raw_prompt_tokens = sum(int(row['raw_prompt_tokens']) for row in rows)
    total_summary_prompt_tokens = sum(int(row['summary_prompt_tokens']) for row in rows)
    raw_truncated_samples = sum(int(row['raw_prompt_tokens_full']) > int(row['raw_prompt_tokens']) for row in rows)
    summary_truncated_samples = sum(int(row['summary_prompt_tokens_full']) > int(row['summary_prompt_tokens']) for row in rows)
    return {
        'samples': len(rows),
        'raw_context_tokens': total_raw_context_tokens,
        'summary_context_tokens': total_summary_context_tokens,
        'raw_prompt_tokens_full': total_raw_prompt_tokens_full,
        'summary_prompt_tokens_full': total_summary_prompt_tokens_full,
        'raw_prompt_tokens': total_raw_prompt_tokens,
        'summary_prompt_tokens': total_summary_prompt_tokens,
        'raw_truncated_samples': raw_truncated_samples,
        'summary_truncated_samples': summary_truncated_samples,
    }


def deduplicate_original_rows(rows):
    deduped = {}
    for row in rows:
        item_id = row.get('_id', '')
        if item_id and item_id not in deduped:
            deduped[item_id] = row
    return list(deduped.values())


def build_accuracy_section(accuracy_rows):
    headers = accuracy_rows[0]
    rows = accuracy_rows[1:]
    return '\n'.join([
        '## Accuracy',
        '',
        format_markdown_table(headers, rows),
    ])


def build_token_overall_section(token_rows):
    if not token_rows:
        return '## Token Overall\n\nToken comparison file not found.'

    raw_input_sources = unique_nonempty_values(token_rows, 'raw_input_source')
    summary_labels = sorted({row.get('summary_input_label', '').strip() for row in token_rows if row.get('summary_input_label', '').strip()})
    if len(summary_labels) > 1:
        original_rows = deduplicate_original_rows(token_rows)
        original_aggregate = aggregate_token_rows(original_rows)
        headers = ['Input Variant', 'Samples', 'Context Tokens', 'Prompt Full', 'Prompt Final', 'Truncated Samples']
        table_rows = [
            [
                original_rows[0].get('raw_input_label', 'original_input') if original_rows else 'original_input',
                str(original_aggregate['samples']),
                format_integer(original_aggregate['raw_context_tokens']),
                format_integer(original_aggregate['raw_prompt_tokens_full']),
                format_integer(original_aggregate['raw_prompt_tokens']),
                str(original_aggregate['raw_truncated_samples']),
            ]
        ]
        for label in summary_labels:
            aggregate = aggregate_token_rows([row for row in token_rows if row.get('summary_input_label', '').strip() == label])
            table_rows.append([
                label,
                str(aggregate['samples']),
                format_integer(aggregate['summary_context_tokens']),
                format_integer(aggregate['summary_prompt_tokens_full']),
                format_integer(aggregate['summary_prompt_tokens']),
                str(aggregate['summary_truncated_samples']),
            ])

        notes = []
        if raw_input_sources:
            notes.append(f'- Raw input source: {raw_input_sources[0]}')
        for label in summary_labels:
            aggregate = aggregate_token_rows([row for row in token_rows if row.get('summary_input_label', '').strip() == label])
            summary_sources = unique_nonempty_values(
                [row for row in token_rows if row.get('summary_input_label', '').strip() == label],
                'summary_input_source',
            )
            notes.append(
                f'- {label}: final prompt is {format_ratio(aggregate["summary_prompt_tokens"] / original_aggregate["raw_prompt_tokens"] if original_aggregate["raw_prompt_tokens"] else 0)} of original input'
            )
            if summary_sources:
                notes.append(f'- {label} source: {summary_sources[0]}')

        return '\n'.join([
            '## Token Overall',
            '',
            format_markdown_table(headers, table_rows),
            '',
            f'- Summary inputs compared: {len(summary_labels)}',
            f'- Unique samples: {len(original_rows)}',
            *notes,
        ])

    aggregate = aggregate_token_rows(token_rows)

    headers = ['Metric', 'Raw', 'Summary', 'Saved', 'Summary/Raw']
    rows = [
        [
            'Context tokens',
            format_integer(aggregate['raw_context_tokens']),
            format_integer(aggregate['summary_context_tokens']),
            format_integer(aggregate['raw_context_tokens'] - aggregate['summary_context_tokens']),
            format_ratio(aggregate['summary_context_tokens'] / aggregate['raw_context_tokens'] if aggregate['raw_context_tokens'] else 0),
        ],
        [
            'Prompt tokens full',
            format_integer(aggregate['raw_prompt_tokens_full']),
            format_integer(aggregate['summary_prompt_tokens_full']),
            format_integer(aggregate['raw_prompt_tokens_full'] - aggregate['summary_prompt_tokens_full']),
            format_ratio(aggregate['summary_prompt_tokens_full'] / aggregate['raw_prompt_tokens_full'] if aggregate['raw_prompt_tokens_full'] else 0),
        ],
        [
            'Prompt tokens final',
            format_integer(aggregate['raw_prompt_tokens']),
            format_integer(aggregate['summary_prompt_tokens']),
            format_integer(aggregate['raw_prompt_tokens'] - aggregate['summary_prompt_tokens']),
            format_ratio(aggregate['summary_prompt_tokens'] / aggregate['raw_prompt_tokens'] if aggregate['raw_prompt_tokens'] else 0),
        ],
    ]
    notes = [
        f'- Samples: {aggregate["samples"]}',
        f'- Raw truncated samples: {aggregate["raw_truncated_samples"]}',
        f'- Summary truncated samples: {aggregate["summary_truncated_samples"]}',
    ]
    if raw_input_sources:
        notes.append(f'- Raw input source: {raw_input_sources[0]}')
    summary_sources = unique_nonempty_values(token_rows, 'summary_input_source')
    if summary_sources:
        notes.append(f'- Summary input source: {summary_sources[0]}')
    return '\n'.join([
        '## Token Overall',
        '',
        format_markdown_table(headers, rows),
        '',
        *notes,
    ])


def build_group_section(title, group_field, rows):
    headers = [group_field, 'sample_count', 'total_raw_prompt_tokens', 'total_summary_prompt_tokens', 'saved_prompt_tokens', 'prompt_compression_ratio']
    table_rows = [
        [
            row[group_field],
            str(row['sample_count']),
            format_integer(row['total_raw_prompt_tokens']),
            format_integer(row['total_summary_prompt_tokens']),
            format_integer(row['saved_prompt_tokens']),
            format_ratio(row['prompt_compression_ratio']),
        ]
        for row in rows
    ]
    return '\n'.join([
        f'## {title}',
        '',
        format_markdown_table(headers, table_rows),
    ])


def write_overview_file(output_path, accuracy_rows, token_rows, grouped_results):
    sections = [
        '# Result Overview',
        '',
        build_accuracy_section(accuracy_rows),
        '',
        build_token_overall_section(token_rows),
    ]

    section_titles = {
        'summary_input_label': 'Token By Summary Input',
        'difficulty': 'Token By Difficulty',
        'length': 'Token By Length',
        'domain': 'Token By Domain',
        'sub_domain': 'Token By Sub Domain',
    }
    for group_field in DEFAULT_GROUPS:
        grouped = grouped_results.get(group_field)
        if not grouped:
            continue
        sections.extend([
            '',
            build_group_section(section_titles.get(group_field, group_field), group_field, grouped['rows']),
        ])

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(sections).strip() + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--accuracy_output', default='result.txt')
    parser.add_argument('--overview_output', default='result_overview.md')
    parser.add_argument('--token_csv', default='results/token_input_comparison_merged.csv')
    parser.add_argument('--grouped_token_dir', default='results/grouped_token_stats')
    parser.add_argument('--compensated', action='store_true')
    args = parser.parse_args()

    accuracy_rows = collect_accuracy_rows(args.results_dir, compensated=args.compensated)
    write_accuracy_table(accuracy_rows, args.accuracy_output)

    token_rows = load_csv_rows(args.token_csv)
    grouped_results = generate_grouped_token_csvs(args.token_csv, args.grouped_token_dir, DEFAULT_GROUPS)
    write_overview_file(args.overview_output, accuracy_rows, token_rows, grouped_results)


if __name__ == '__main__':
    main()
