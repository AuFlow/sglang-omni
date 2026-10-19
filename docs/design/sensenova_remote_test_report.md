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
The `/metrics` path returns 404 in this serving configuration.

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
- Phase 3 batching, TP/DP, Cache-DiT, offload lifecycle, cancellation, and
  baseline image comparison at scale.
- Metrics endpoint validation: `/metrics` returns 404 here.
