# Qwen3-VL-2B-Instruct Video Understanding Time Estimates

## Input
input video: 8 frames, 384x384

prefill token num 620
image token num 144*4 = 576

## Measured timings
image encoder: 
157 ms * 4 = 628ms 

Number of chunks required in prefill stage: math.ceil(620/128) = 5

hidden layers = 28 

Single-layer prefill chunk 1 time: 5.8 ms
Single-layer prefill chunk 2 time: 6.3 ms
Single-layer prefill chunk 3 time: 6.5 ms
Single-layer prefill chunk 4 time: 6.9 ms
Single-layer prefill chunk 5 time: 7.3 ms


Single-layer prefill (5 chunks) total: 5.8 + 6.3 + 6.5 + 6.9 + 7.3 = 32.8 ms
28-layer prefill (5 chunks) total: 32.8 * 28 = 918.4 ms
Post-layer time: 16.2 ms

Prefill total time: 918.4 + 16.2 = 934.6 ms

Decode per-layer time: 3.2 ms
28-layer decode time: 3.2 * 28 = 89.6 ms

Decode total time per pass: 89.6 + 16.2 = 105.8 ms

## Summary
Image encoder time: 628 ms
LLM prefill time: 934.6 ms
Decode speed: 1000 / 105.8 = 9.5 tokens/s

