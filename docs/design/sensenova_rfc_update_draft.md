# SenseNova RFC Update Draft

Status: issue-ready text for #2831 and #2908 as of 2026-10-19. This file does
not post anything; copy the relevant section into the issue comment.

## Comment for #2831

### Progress update

The integration branch `wes/sensenova-umm-adapter` has merged main at
`cd4b310fa` (including #2918 and #2919) and now runs SenseNova T2I/I2I on the
shared UMM pipeline by default, using the model's existing `generate` stage.
The previous single-stage native pipeline is no longer the default serving
path.

Current implementation status:

- `pipeline/umm.py`, `scheduling/message.py`, and the shared TP scheduler are
  unchanged from main. SenseNova does not add UMM runtime options or
  completion metadata.
- The adapter returns ordered image segments plus one `kind="usage"` segment.
  Images endpoints read `image_usage` from that segment; `/generate` clients
  also see the usage segment.
- Admission limits are fixed model constants
  (`DEFAULT_UMM_ADMISSION_LIMITS`); `max_outputs` is 11 for up to 10 images
  plus the usage segment.
- Deadline is carried in model metadata
  (`sensenova_deadline_monotonic_s`); TP cancellation uses a SenseNova
  scheduler subclass and a model exception.
- CPU/Gloo regression currently passes 237 tests (2 skipped), covering T2I,
  multi-output, UMM routing/output, Images usage, metrics, TP config and
  cancellation, Cache-DiT, sampling, and usage. GPU/NPU acceptance has not
  been run.

### Proposed phase/scope edits

1. Make UMM the default serving contract for T2I/I2I; keep the native
   single-stage pipeline only as a migration/deprecation path.
2. Replace the Phase 3 acceptance table with the UMM matrix below.
3. Add a Phase 4 for thinking mode and interleaved text/image output. It
   depends on #2908 progress streaming, SGLang #40702 (thinking), and
   SGLang #35038 (interleave). Until progress streaming lands, Phase 4 is
   planned/blocked, not an acceptance gate.

| Capability | GPU | NPU | Current evidence |
| --- | --- | --- | --- |
| T2I | Partial | Partial | PR-level checks in #2325/#2351; no CI model lane |
| I2I | Partial | Partial | PR-level checks in #2351/#2354; no CI model lane |
| n>1 multi-output | Unverified | Partial | NPU check in #2653; GPU unverified |
| Dynamic batching | Unverified | Unverified | CPU/mock tests only |
| TP | Unverified | Unverified | Gloo CPU parity; NCCL/HCCL inference unverified |
| DP | Unverified | Unverified | Config tests and docs only |
| Cache-DiT | Unverified | Unverified | Unit tests only |
| Layerwise offload | Unverified | N/A | CUDA-only; no lifecycle test |
| Cancellation/deadline | Unverified | Unverified | CPU controller tests only |
| UMM path | Unexecuted | Unexecuted | CPU contracts plus fake coordinator |
| Images metrics | Partial | Partial | Native evidence in #2728; UMM segment counting added |

### Mainline follow-ups

- #2723 (shared Images response) and #2727 (shared metrics) are still open.
- Split the current work into runtime migration, adapter/usage, config
  default, client/metrics, and docs/tests commits before opening mainline
  PRs.
- Do not claim hardware acceptance from the native pipeline for the UMM
  default.

## Comment for #2908

### SenseNova row

Update the SenseNova-U1 row to:

> T2I/I2I on the shared UMM controller, with one generation phase and one
> usage segment. Thinking and interleaved output are a later #2831 phase and
> are blocked on phase progress streaming. The shared generation stage is not
> used by the current SenseNova implementation.

### Runtime additions

No new UMM runtime additions are required for the current SenseNova
integration. In particular:

- `UMMStep.completion_metadata` is not used; `image_usage` is carried as a
  `kind="usage"` output segment.
- `create_umm_controller(adapter_options=...)` is not used; admission limits
  are model constants.
- `UMM_DEADLINE_METADATA_KEY` is not added to `pipeline/umm.py`; the adapter
  writes a model-owned deadline key.
- `forget()` keeps main behavior.
- TP cancellation is a SenseNova `TensorParallelScheduler` subclass, not a
  shared scheduler mode.

This means the adapter follows the “Adding a model changes no shared runtime
code” rule, except for the client/serve result handling needed to read the
usage segment and count image segments in metrics.

### Shared generation stage

The #2919 native generation stage is not a drop-in host yet:

- SGLang multimodal_gen T2I is merged (#36606), but I2I (#39081) and
  interleaved/thinking support (#35038/#40702) remain open.
- #2919's inline media limit is 8 MB, below what SenseNova admission allows,
  so a qualification change would be needed before switching.

The current model-owned `generate` stage remains the supported path.

## Evidence

- Branch: `wes/sensenova-umm-adapter`, merge `cd4b310fa`.
- CPU/Gloo regression: 237 passed, 2 skipped (43 subtests).
- Not run in this environment: full SGLang platform tests, GPU/NPU inference,
  NCCL/HCCL cancellation, business acceptance.
