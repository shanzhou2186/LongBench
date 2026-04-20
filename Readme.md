This is the PoC benchmark based on LongBench
Summarize with small model
python compare-input.py \
  --summary_jsonl data/longbench_v2_with_rc_0.6B.jsonl data/longbench_v2_with_rc_1.7B.jsonl \
  --summary_input_label 0.6B_summary 1.7B_summary \
  --raw_input_label original_input \
  --output_csv results/token_input_comparison_multi.csv


