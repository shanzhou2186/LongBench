This is the PoC benchmark based on LongBench
### summarize the result with small model 

'''
python summarize_strict_compression.py \
  --model Qwen3-0.6B-local \
  --api_key EMPTY \
  --base_url http://127.0.0.1:30000/v1 \
  --output_jsonl data/longbench_v2_with_rc_0.6B.jsonl \
  --target_model Qwen3-30B-A3B-local \
  --target_prompt_style 0shot
'''


### Use the output of small model as the input of big model

'''
python pred-opt.py \
  --model Qwen3-30B-A3B-local \
  --url http://127.0.0.1:30000/v1 \
  --api_key EMPTY \
  --input_jsonl data/longbench_v2_with_rc_1.7B.jsonl \
  --save_dir results/run_1_7b \
  --prompt_token_margin 512
'''

### Compare the result of token number
'''
python compare-input.py \
  --summary_jsonl data/longbench_v2_with_rc_0.6B.jsonl data/longbench_v2_with_rc_1.7B.jsonl \
  --summary_input_label 0.6B_summary 1.7B_summary \
  --raw_input_label original_input \
  --output_csv results/token_input_comparison_multi.csv
'''
### The final result
'''
python result.py \
  --token_csv results/token_input_comparison_multi.csv \
  --overview_output results/result_overview_multi.md
'''
