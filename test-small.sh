python pred-opt.py \
  --model Qwen3-30B-A3B-local \
  --url http://127.0.0.1:30000/v1 \
  --api_key EMPTY \
  --input_jsonl data/longbench_v2_with_rc_1.7B.jsonl \
  --save_dir results/run_1_7b \
  --prompt_token_margin 512
