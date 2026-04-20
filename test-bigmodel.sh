export HF_ENDPOINT=https://hf-mirror.com
python pred.py \
  --model Qwen3-30B-A3B-local \
  --url http://127.0.0.1:30000/v1 \
  --api_key EMPTY \
  --max_prompt_tokens 40000 
