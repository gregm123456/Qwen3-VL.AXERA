# Qwen3-VL-4B-Instruct Video Understanding Time Estimates

## Input
input video: 8 frames, 384x384

prefill token num 620
image token num 144*4 = 576

## Measured timings
image encoder: 
158 ms * 4 = 632ms 

Number of chunks required in prefill stage: math.ceil(620/128) = 5

hidden layers = 36 

Single-layer prefill chunk 1 time: 11.1 ms
Single-layer prefill chunk 2 time: 11.8 ms
Single-layer prefill chunk 3 time: 12.3 ms
Single-layer prefill chunk 4 time: 13.2 ms
Single-layer prefill chunk 5 time: 13.6 ms


Single-layer prefill (5 chunks) total: 11.1 + 11.8 + 12.3 + 13.2 + 13.6 = 62 ms
36-layer prefill (5 chunks) total: 62 * 36 = 2232 ms
Post-layer time: 20 ms

Prefill total time: 2232 + 20 = 2252 ms

Decode per-layer time: 6.0 ms
36-layer decode time: 36 * 6 = 216 ms

Decode total time per pass: 216 + 20 = 236 ms

## Summary
Image encoder time: 632 ms
LLM prefill time: 2252 ms
Decode speed: 1000 / 236 = 4.2 tokens/s

