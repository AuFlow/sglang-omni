# SenseNova Remote Test Report

Date: 2026-10-19. Host: single RTX 4090. The host has the full SGLang stack
and a SenseNova-U1.5-8B-MoT checkpoint; it has no Ascend NPU.

Environment:

```text
Python 3.12.15
torch 2.13.0+cu130
SGLang 0.5.21
transformers 5.12.1
FastAPI 0.147.0
pytest 9.1.1
```

## Results

```text
tests/unit_test/sensenova                         225 passed, 1 failed
tests/unit_test/serve                             876 passed, 3 failed, 2 skipped
tests/unit_test/pipeline                          929 passed, 10 failed, 6 skipped
local CPU/Gloo regression                         241 passed, 5 skipped
```

The SenseNova failure is
`test_native_tp.py::test_native_qwen3_prefix_and_image_denoising_tp_parity[dtype1]`
(`torch.bfloat16`); float32 passes. It also fails at baseline `cd4b310fa`.
The full model's bf16 CPU SDPA output is already numerically unstable in this
torch build, so this is an environment/dtype limitation, not a UMM migration
regression.

All 10 pipeline failures reproduce at baseline `cd4b310fa`. Two of the three
serve failures reproduce at baseline; the health test passes when rerun alone
and is timing-flaky. No UMM/SenseNova-specific regression was observed.

## End-to-end serving

Checkpoint: `/workspace/model/SenseNova-U1.5-8B-MoT`. Served through the
UMM-default config with layerwise offload:

```text
dit_layerwise_offload=true
dit_layerwise_resident_layers=8
dit_offload_prefetch_size=2
orchestrator timeout_s=1800
```

The first attempt with `resident_layers=0` exceeded the default 600 s deadline
before completing a 512x512, 10-step request. Keeping 8 DiT layers resident
brought 512x512 4-step requests to completion in roughly tens of seconds.

All of these returned HTTP 200 with valid PNG and `image_usage`:

| Case | Result | Usage |
| --- | --- | --- |
| T2I 512x512, 4 steps | 326708-byte PNG | text_tokens=12, image_count=1, 512x512 |
| T2I 512x512, n=2, 4 steps | 2 PNGs | text_tokens=9, image_count=2, 512x512 |
| I2I 512x512, 4 steps | 314162-byte PNG | text_tokens=10, image_tokens=4096, image_count=1 |
| T2I 1024x1024, 4 steps | 892748-byte PNG | text_tokens=12, image_count=1, 1024x1024 |

Artifacts on the remote host: `/workspace/t2i.png`, `/workspace/i2i.png`,
`/workspace/t2i_n2_0.png`, `/workspace/t2i_n2_1.png`,
`/workspace/t2i_1024.png`. Server log: `/workspace/sensenova-server.log`.

`/metrics` is disabled by default. Restarting with `--enable-metrics` makes
it return 200, and after one T2I request it reports:

```text
sglang_omni:generated_images_total{model_name=...,stage="generate"} 1.0
```

## Batching, cache, cancellation, deadline

Server settings: `max_batch_size=2`, `max_batch_wait_ms=500`,
`enable_cache_dit` per request, layerwise offload resident=8.

- Batching: two concurrent 512x512/4-step requests produced
  `SenseNova-U1 T2I batch: size=2, image_size=512x512, steps=4, cfg=4.0`;
  both returned HTTP 200.
- Cache-DiT: an 8-step request with `enable_cache_dit=true` returned HTTP 200.
  The log shows `[Cache-DiT] ... Collected Context Config:
  DBCache_F8B0_W8I1M0MC0_R0.08_N8_CFG1` and `Match Blocks ...
  CachedBlocks_Pattern_3_4_5`.
- Cancel/disconnect: a 50-step request with `curl -m 5` exited 28; the server
  logged `asyncio.CancelledError`, remained healthy, and a following 4-step
  request returned HTTP 200.
- Deadline: with `timeout_s=15`, a 50-step request returned HTTP 500 with
  `UMM request deadline exceeded after timeout_s=15`. After restoring
  `timeout_s=1800`, a 4-step request returned HTTP 200.
- TP/DP: not run, because the host exposes a single RTX 4090.

## Boundary, concurrency, long-step, failure, stream

- T2I n=10: HTTP 200, 10 valid 512x512 PNGs, `image_count=10`.
- T2I 768x512: HTTP 200, `width=768`, `height=512`.
- I2I n=2 with one reference: HTTP 200, 2 images, `image_count=2`.
- I2I with three references: HTTP 500 CUDA OOM for both n=1 and n=2 on the 24 GiB
  RTX 4090. One reference n=2 fits; multi-reference is not memory-safe here.
- I2I with one reference and omitted size: HTTP 200, 2048x2048,
  `image_tokens=4096`; the default size path works but is memory-heavy.
- Cache-DiT sequential: two 8-step requests both HTTP 200; outputs differ, so
  the second request did not reuse the first response.
- Cache-DiT + batching: two concurrent 8-step requests logged
  `T2I batch: size=2` and both returned HTTP 200.
- Four concurrent 2-step requests: all HTTP 200, batched in pairs.
- 50-step T2I: HTTP 200; completed within the 1800 s deadline.
- Failure recovery: invalid size returned HTTP 400, and the next valid request
  returned HTTP 200.
- `/generate` streaming: HTTP 200 with two segments (image and usage), a
  complete event, and `[DONE]`; the usage segment carried `stage_name=generate`.
  The request took several minutes for the 2-step stage params, so stage-param
  passthrough is not confirmed and needs follow-up.

Metrics after these runs:

```text
sglang_omni:generated_images_total{stage="generate"} 24
sglang_omni:generated_images_total{stage="unknown"}   1
```

The `unknown` sample is the streamed `/generate` result: its final segments have
their data stripped and the client cannot recover the usage segment's stage
label from the final result. This is a metrics labeling gap for streamed UMM.

## Baseline quality and offload lifecycle

The pre-UMM baseline was launched from a detached worktree at `706053fa0`
(native single-stage pipeline) on port 8001, with the same checkpoint and
offload settings. T2I and I2I used the same prompt, seed, size, steps, and
reference image as the current UMM runs.

- T2I: MSE=0, PSNR=inf, SSIM=1.0 against the current UMM output.
- I2I: MSE=0, PSNR=inf, SSIM=1.0 against the current UMM output.
- Usage fields match: T2I text_tokens=12/image_count=1; I2I
  text_tokens=10/image_tokens=4096/image_count=1.

Layerwise offload lifecycle on the current server:

```text
after start                          ~8.9 GiB GPU
after one 4-step request             ~11.6 GiB GPU
after stop                           1 MiB GPU
after restart                        ~8.9 GiB GPU
request after restart                HTTP 200
```

No residual GPU memory was observed after shutdown.

## Fixes found remotely

- FastAPI 0.147 parsed an omitted optional boolean form field as `False`; the
  I2I endpoint now parses `enable_cache_dit` as an optional string and accepts
  explicit true/false.
- The fused RoPE tests patched `modeling_qwen3` aliases; the dispatch actually
  reads `rotary_embeddings`, so the tests now patch the active module.
- The factory-precedence round-trip test reset the UMM-required
  `cooperative_cancellation` flag; the test now preserves it.

## Not run

- NPU FIA/RMSNorm/SwiGLU/fused RoPE and I2I: no Ascend NPU.
- TP/DP real multi-GPU inference: the host has one GPU.
- NPU execution: no Ascend NPU.
- Multi-reference I2I on 24 GiB: currently OOM.
- `/generate` streamed UMM metrics stage labeling and stage-param passthrough.
- Baseline comparison at larger scale and other prompts/seeds.
