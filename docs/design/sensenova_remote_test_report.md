# SenseNova Remote Test Report

Date: 2026-10-19. Host: single RTX 4090. The host has the full SGLang stack
but no SenseNova checkpoint and no Ascend NPU.

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

## Fixes found remotely

- FastAPI 0.147 parsed an omitted optional boolean form field as `False`; the
  I2I endpoint now parses `enable_cache_dit` as an optional string and accepts
  explicit true/false.
- The fused RoPE tests patched `modeling_qwen3` aliases; the dispatch actually
  reads `rotary_embeddings`, so the tests now patch the active module.
- The factory-precedence round-trip test reset the UMM-required
  `cooperative_cancellation` flag; the test now preserves it.

## Not run

- Real T2I/I2I inference: no SenseNova checkpoint on the host.
- NPU FIA/RMSNorm/SwiGLU/fused RoPE and I2I: no NPU.
- Phase 3 performance, cancellation, and baseline image comparison.
