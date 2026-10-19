# SenseNova Phase 3 Validation Checklist

Use this checklist on the target GPU and Ascend NPU hosts. It validates the
UMM-default pipeline after the no-runtime migration. Baseline comparison is
against commit `706053fa0` with the same checkpoint, seeds, hardware,
parameters, concurrency, and warmup.

## Environment

- Checkpoint: `sensenova/SenseNova-U1.5-8B-MoT` (or the local copy).
- Current branch: `wes/sensenova-umm-adapter`.
- Record driver, CUDA/CANN, PyTorch, SGLang, and model revision.
- For GPU: record `nvidia-smi -q` and the GPU count.
- For NPU: record `npu-smi info`, CANN version, and the installed operator
  probes logged at startup.
- Clear stale server processes and ports before each run.

## Functional matrix

| Case | Configuration | Pass condition |
| --- | --- | --- |
| T2I short | 512x512, 10 steps, n=1 | HTTP 200, valid PNG, sane image |
| T2I long | 1024x1024, 50 steps, n=1 | HTTP 200, valid PNG, no OOM |
| T2I multi-output | n=10 | 10 images, order matches seeds, usage.image_count=10 |
| I2I one reference | explicit size, n=1 | HTTP 200, PNG, reference semantics preserved |
| I2I multi reference | 3 references, size omitted | aspect ratio preserved, usage image_tokens > 0 |
| Batching | two compatible 1024x1024 T2I requests, max_batch_size=2 | log contains `T2I batch: size=2` |
| TP | `sensenova_u1_tp2.yaml`, n=1 | result matches TP=1 baseline within tolerance |
| DP | `sensenova_u1_dp2.yaml`, concurrent requests | both replicas serve; outputs associated per request |
| DP+TP | `sensenova_u1_dp2_tp2.yaml` | both replicas and all ranks complete |
| Cache-DiT | `enable_cache_dit=true`, 50 steps | 200, output close to baseline, state isolated across requests |
| Offload | `dit_layerwise_offload=true`, CUDA only | 200, no weight leak, later request succeeds |
| Cancel/disconnect | cancel during denoise | request ends, no stuck worker, next request succeeds |
| Deadline | short `timeout_s`, long request | deadline error, state released, next request succeeds |
| Failure recovery | force one failed generation | next request succeeds, weights remain loaded |

## Commands

Default launch:

```bash
python -m sglang_omni.cli serve \
  --config examples/configs/sensenova_u1_umm.yaml \
  --model-path "$MODEL_PATH" \
  --generate.gpu 0 \
  --port 8000
```

T2I:

```bash
curl -sS http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A blue cat by a window","size":"512x512","n":1,"seed":42,
       "num_inference_steps":10,"guidance_scale":4.0}' > t2i.json
```

I2I:

```bash
curl -sS http://127.0.0.1:8000/v1/images/edits \
  -F image=@reference.png -F prompt='Replace the background with snow' \
  -F size=512x512 -F n=1 -F seed=42 -F num_inference_steps=10 \
  -F guidance_scale=4.0 > i2i.json
```

TP / DP use the example YAMLs:

```bash
python -m sglang_omni.cli serve --config examples/configs/sensenova_u1_tp2.yaml \
  --model-path "$MODEL_PATH" --port 8000
python -m sglang_omni.cli serve --config examples/configs/sensenova_u1_dp2.yaml \
  --model-path "$MODEL_PATH" --port 8000
python -m sglang_omni.cli serve --config examples/configs/sensenova_u1_dp2_tp2.yaml \
  --model-path "$MODEL_PATH" --port 8000
```

Cache-DiT and offload:

```bash
python -m sglang_omni.cli serve --model-path "$MODEL_PATH" \
  --generate.factory.enable_cache_dit true --generate.gpu 0 --port 8000
python -m sglang_omni.cli serve --model-path "$MODEL_PATH" \
  --generate.factory.dit_layerwise_offload true --generate.gpu 0 --port 8000
```

Metrics and usage:

```bash
curl -sS http://127.0.0.1:8000/metrics | grep -E 'generated_images|stage|http_requests'
```

Record `usage` from both Images endpoints, including
`input_tokens_details.text_tokens` and `image_tokens`.

## Baseline comparison

- Store JSON responses and decoded PNGs for baseline and current runs.
- Compare image count, dimensions, format, seed/output association, and usage
  fields.
- For image quality, record an agreed metric (for example SSIM or LPIPS) and
  keep the same prompt/seed/size/steps.
- Compare request latency, throughput, GPU/NPU memory, and startup/shutdown
  time. A passing unit test is not a performance claim.

## NPU-specific checks

- Confirm startup logs show FIA / RMSNorm / SwiGLU availability.
- Exercise I2I with 1 and 3 references; confirm per-branch prefix lengths and
  the selected cache layout are logged.
- Opt in to fused RoPE with `SGLANG_SENSENOVA_NPU_FUSED_ROPE=1`; compare
  outputs against eager and capture an Ascend trace proving the kernel ran.
- Record any fallback path (FIA -> SDPA, fused RoPE -> eager).

## Result record

For each case record: date, host, device, checkpoint revision, launch command,
request, HTTP status, output hash/quality, metrics delta, and any fallback or
error. Mark unexecuted cases explicitly; do not infer them from another
hardware or from the native pipeline.
