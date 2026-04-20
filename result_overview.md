# Result Overview

## Accuracy

Model | Overall | Easy | Hard | Short | Medium | Long
--- | --- | --- | --- | --- | --- | ---
Qwen3-30B-A3B-local_longbench_v2_with_rc | 31.2 | 36.4 | 28.6 | 31.2 | NA | NA
Qwen3-30B-A3B-local_opt | 26.6 | 31.2 | 23.8 | 32.2 | 19.5 | 31.5
Qwen3-30B-A3B-local_opt_1.7B | 31.4 | 33.9 | 29.9 | 37.2 | 27.0 | 30.6
Qwen3-30B-A3B-local_opt_org | 30.4 | 31.2 | 29.9 | 37.8 | 25.6 | 27.8

## Token Overall

Metric | Raw | Summary | Saved | Summary/Raw
--- | --- | --- | --- | ---
Context tokens | 130,976,720 | 91,130 | 130,885,590 | 0.0696%
Prompt tokens full | 131,090,003 | 204,401 | 130,885,602 | 0.1559%
Prompt tokens final | 18,270,685 | 204,401 | 18,066,284 | 1.1187%

- Samples: 503
- Raw truncated samples: 370
- Summary truncated samples: 0

## Token By Difficulty

difficulty | sample_count | total_raw_prompt_tokens | total_summary_prompt_tokens | saved_prompt_tokens | prompt_compression_ratio
--- | --- | --- | --- | --- | ---
easy | 192 | 7,040,552 | 69,619 | 6,970,933 | 0.9888%
hard | 311 | 11,230,133 | 134,782 | 11,095,351 | 1.2002%

## Token By Length

length | sample_count | total_raw_prompt_tokens | total_summary_prompt_tokens | saved_prompt_tokens | prompt_compression_ratio
--- | --- | --- | --- | --- | ---
long | 108 | 4,423,653 | 54,807 | 4,368,846 | 1.2390%
medium | 215 | 8,806,393 | 84,203 | 8,722,190 | 0.9562%
short | 180 | 5,040,639 | 65,391 | 4,975,248 | 1.2973%

## Token By Domain

domain | sample_count | total_raw_prompt_tokens | total_summary_prompt_tokens | saved_prompt_tokens | prompt_compression_ratio
--- | --- | --- | --- | --- | ---
Code Repository Understanding | 50 | 2,011,441 | 21,858 | 1,989,583 | 1.0867%
Long In-context Learning | 81 | 3,189,717 | 37,915 | 3,151,802 | 1.1887%
Long Structured Data Understanding | 33 | 1,329,811 | 9,295 | 1,320,516 | 0.6990%
Long-dialogue History Understanding | 39 | 1,424,662 | 6,768 | 1,417,894 | 0.4751%
Multi-Document QA | 125 | 4,413,601 | 62,702 | 4,350,899 | 1.4207%
Single-Document QA | 175 | 5,901,453 | 65,863 | 5,835,590 | 1.1160%

## Token By Sub Domain

sub_domain | sample_count | total_raw_prompt_tokens | total_summary_prompt_tokens | saved_prompt_tokens | prompt_compression_ratio
--- | --- | --- | --- | --- | ---
Academic | 94 | 3,114,596 | 39,179 | 3,075,417 | 1.2579%
Agent history QA | 20 | 646,421 | 2,302 | 644,119 | 0.3561%
Code repo QA | 50 | 2,011,441 | 21,858 | 1,989,583 | 1.0867%
Detective | 22 | 879,277 | 5,745 | 873,532 | 0.6534%
Dialogue history QA | 19 | 778,241 | 4,466 | 773,775 | 0.5739%
Event ordering | 20 | 815,140 | 7,377 | 807,763 | 0.9050%
Financial | 37 | 1,409,295 | 20,678 | 1,388,617 | 1.4673%
Governmental | 41 | 1,414,113 | 21,756 | 1,392,357 | 1.5385%
Knowledge graph reasoning | 15 | 614,400 | 2,071 | 612,329 | 0.3371%
Legal | 33 | 928,205 | 14,096 | 914,109 | 1.5186%
Literary | 30 | 1,112,221 | 9,747 | 1,102,474 | 0.8764%
Many-shot learning | 21 | 860,160 | 7,697 | 852,463 | 0.8948%
Multi-news | 23 | 642,207 | 9,987 | 632,220 | 1.5551%
New language translation | 20 | 819,198 | 9,712 | 809,486 | 1.1855%
Table QA | 18 | 715,411 | 7,224 | 708,187 | 1.0098%
User guide QA | 40 | 1,510,359 | 20,506 | 1,489,853 | 1.3577%
