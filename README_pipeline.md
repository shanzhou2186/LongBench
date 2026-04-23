# 🗜️ Context Summarization Pipeline

使用小模型（Qwen3-0.6B / 1.7B）压缩长上下文，再送入大答题模型，在减少 prompt token 的同时尽量保留答题准确率。

## Scripts Overview

| 脚本 | 用途 |
|------|------|
| `summarize_strict_compression.py` | 小模型压缩上下文（单次压缩或分块抽取+层次合并） |
| `compare-input.py` | 对比原始输入与各摘要输入的 prompt token 量 |
| `result.py` | 聚合准确率 + token 统计，生成 `result_overview.md` |
| `benchmark_summarize.py` | 测试小模型摘要服务的吞吐量 |
| `sample_data.py` | 按类别随机抽样，生成快速测试用输入文件 |
| `pred-opt.py` | 大模型答题推理（支持原始输入或摘要输入） |

---

## Step 1: 抽取测试样本

从原始数据的 5 个类别（easy / hard / short / medium / long）中各随机抽取 N 条，合并去重后生成测试用 jsonl：

```bash
# 每类各抽 2 条（默认），共最多 10 条（_id 去重）
python sample_data.py --output_jsonl data/sample_test.jsonl

# 每类抽 5 条，固定随机种子（可复现）
python sample_data.py --n 5 --seed 42 --output_jsonl data/sample_test.jsonl

# 指定原始数据路径
python sample_data.py \
    --raw_local_json /path/to/data.json \
    --output_jsonl data/sample_test.jsonl
```

---

## Step 2: 启动模型服务（SGLang）

使用 SGLang 启动 OpenAI-compatible 推理服务。以下示例均假设在 **96 物理核 CPU** 环境下运行。

### 启动大答题模型（Qwen3-30B-A3B，3路张量并行）

```bash
SGLANG_USE_CPU_ENGINE=1 \
SGLANG_CPU_OMP_THREADS_BIND='0-31|32-63|64-95' \
python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --model /home/shan/Qwen/Qwen3-30B-A3B \
    --trust-remote-code \
    --device cpu \
    --disable-overlap-schedule \
    --tp 3 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 65536
```

### 启动小摘要模型（Qwen3-0.6B 示例，单路）

```bash
SGLANG_USE_CPU_ENGINE=1 \
SGLANG_CPU_OMP_THREADS_BIND='0-95' \
python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --model /home/shan/Qwen/Qwen3-0.6B \
    --trust-remote-code \
    --device cpu \
    --disable-overlap-schedule \
    --tp 1 \
    --mem-fraction-static 0.8 \
    --max-total-tokens 65536
```

**参数说明：**
- `SGLANG_USE_CPU_ENGINE=1`：启用 CPU 推理引擎
- `SGLANG_CPU_OMP_THREADS_BIND`：将各张量并行 worker 绑定到对应核段，`|` 分隔每个 worker 的核范围；`--tp 3` 时分三段（0-31 / 32-63 / 64-95）
- `--tp`：张量并行度，需与 `SGLANG_CPU_OMP_THREADS_BIND` 中的段数一致
- `--max-total-tokens`：KV cache 总 token 容量上限，答题模型设为 65536（对应 64K 上下文窗口）
- `--mem-fraction-static`：静态内存占比，CPU 模式下控制 KV cache 预分配比例
- `--disable-overlap-schedule`：CPU 模式下禁用 overlap 调度，避免兼容性问题

服务默认监听 `http://0.0.0.0:30000/v1`，启动后用以下命令验证：

```bash
curl http://127.0.0.1:30000/v1/models
```

---

## Step 3: 运行小模型摘要压缩

先按上方说明启动小模型服务，再运行摘要脚本：

```bash
# 测试样本（快速验证）
python summarize_strict_compression.py \
    --model Qwen3-0.6B-local \
    --api_key EMPTY \
    --base_url http://127.0.0.1:30000/v1 \
    --input_jsonl data/sample_test.jsonl \
    --output_jsonl data/sample_test_summarized.jsonl \
    --target_model Qwen3-30B-A3B-local \
    --target_prompt_style 0shot

# 全量数据（503 条，0.6B 模型）
python summarize_strict_compression.py \
    --model Qwen3-0.6B-local \
    --api_key EMPTY \
    --base_url http://127.0.0.1:30000/v1 \
    --output_jsonl data/longbench_v2_with_rc_0.6B.jsonl \
    --target_model Qwen3-30B-A3B-local \
    --target_prompt_style 0shot

# 全量数据（1.7B 模型）
python summarize_strict_compression.py \
    --model Qwen3-1.7B-local \
    --api_key EMPTY \
    --base_url http://127.0.0.1:30000/v1 \
    --output_jsonl data/longbench_v2_with_rc_1.7B.jsonl \
    --target_model Qwen3-30B-A3B-local \
    --target_prompt_style 0shot
```

**关键参数说明：**
- `--model`：摘要小模型（负责压缩的模型）
- `--target_model`：下游答题大模型，用于估算 prompt 开销从而确定摘要 token 预算
- `--target_prompt_style`：答题时用的 prompt 模板（`0shot` / `cot`），影响 token 预算估算
- `--auto_chunk_chars`（默认 12000）：context 超过此长度走分块路径，否则走单次压缩

---

## Step 3: 吞吐量压测

测试小模型摘要服务的处理速度，获取输出吞吐和总吞吐：

```bash
# 快速连通性验证（10 条，单并发）
python benchmark_summarize.py \
    --model Qwen3-0.6B-local \
    --base_url http://127.0.0.1:30000/v1 \
    --input_jsonl data/sample_test.jsonl \
    --output_jsonl /dev/null

# 压测（50 条，8 并发，保存统计结果）
python benchmark_summarize.py \
    --model Qwen3-0.6B-local \
    --base_url http://127.0.0.1:30000/v1 \
    --concurrency 8 \
    --limit 50 \
    --output_jsonl /dev/null \
    --stats_output results/benchmark_stats.txt
```

**输出指标：**
| 指标 | 含义 |
|------|------|
| `output_throughput` | completion tokens / 挂钟时间 |
| `total_throughput` | (prompt + completion) tokens / 挂钟时间 |
| 分项统计 | `single_pass` / `chunk_extract` / `merge` 各自的调用次数、token 量、占总输出比例 |

---

## Step 4: 大模型答题推理

用大模型对三种输入（原始 / 0.6B 摘要 / 1.7B 摘要）分别跑推理：

```bash
# 原始输入
python pred-opt.py --model Qwen3-30B-A3B-local \
    --output_file results/Qwen3-30B-A3B-local_opt_org.jsonl

# 0.6B 摘要输入
python pred-opt.py --model Qwen3-30B-A3B-local \
    --summary_jsonl data/longbench_v2_with_rc_0.6B.jsonl \
    --output_file results/Qwen3-30B-A3B-local_opt_0.6B.jsonl

# 1.7B 摘要输入
python pred-opt.py --model Qwen3-30B-A3B-local \
    --summary_jsonl data/longbench_v2_with_rc_1.7B.jsonl \
    --output_file results/Qwen3-30B-A3B-local_opt_1.7B.jsonl
```

---

## Step 5: 对比 Token 用量

统计三种输入方式在答题 prompt 上的 token 总量差异：

```bash
python compare-input.py \
    --summary_jsonl data/longbench_v2_with_rc_0.6B.jsonl \
                    data/longbench_v2_with_rc_1.7B.jsonl \
    --output_csv results/token_input_comparison_multi.csv
```

输出 CSV 包含每条样本在三种输入下的 context token 数、prompt token 数、压缩比等字段。

---

## Step 6: 生成结果汇总

```bash
python result.py \
    --token_csv results/token_input_comparison_multi.csv \
    --overview_output result_overview_multi.md
```

生成 `result_overview_multi.md`，包含：
- 准确率表（Overall / Easy / Hard / Short / Medium / Long）
- Token 三路对比表（原始输入 vs 0.6B 摘要 vs 1.7B 摘要）
- 按 difficulty / length / domain / sub_domain 分组的 token 压缩统计

---

## Config

模型名称到本地路径的映射放在 `config/model2path.json`，模型最大 prompt token 数放在 `config/model2maxlen.json`。

```json
// config/model2path.json（示例）
{
    "Qwen3-0.6B-local":      "/home/shan/Qwen/Qwen3-0.6B",
    "Qwen3-1.7B-local":      "/home/shan/Qwen/Qwen3-1.7B",
    "Qwen3-30B-A3B-local":   "/home/shan/Qwen/Qwen3-30B-A3B"
}

// config/model2maxlen.json（示例）
{
    "Qwen3-30B-A3B-local":   65536
}
```
