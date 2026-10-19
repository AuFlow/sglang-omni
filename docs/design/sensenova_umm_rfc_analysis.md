# SenseNova 在 SGLang-Omni 的 RFC 关联与分支符合度分析

分析对象：`wes/sensenova-umm-adapter` 分支（HEAD `cd4b310fa`），
[#2831 SenseNova Adaptation RFC](https://github.com/sgl-project/sglang-omni/issues/2831)，
[#2908 Shared Serving Runtime for UMMs RFC](https://github.com/sgl-project/sglang-omni/issues/2908)。
本文记录分析与设计；实现阶段的改动以工作区代码为准。

> **2026-10-19 决策更新（以此为准）**
>
> 1. **UMM path 替换 native pipeline**，不再保留两套并行的默认 serving 路径。
> 2. **修改 #2831，增加后续 Phase 承载 thinking/interleaved**；不再新建独立 RFC。
> 3. **不改动 UMM runtime controller**（`sglang_omni/pipeline/umm.py` 及其共享 controller
>    契约）。已确认范围**不包括** client/serve：UMM 结果处理、Images usage、metrics 允许
>    做最小共享适配。
> 4. **状态更新**：分支已在 `cd4b310fa` merge main，包含 #2918/#2919；此前的 rebase
>    冲突已解决。按第 3 条，仍需撤回 branch 对 controller/scheduler 的改动。
>
> 前文（尤其 1.2、4.2、6.1、6.2.2、6.5）记录的是决策前/merge 前的分析；与本节冲突处，
> 以第 8 节为准。
>
> **实现状态（2026-10-19）**：本轮已开始按第 8 节改代码；工作区代码状态以此为准，
> 本文件中的“当前分支”描述是分析时的基线。

## 结论摘要

- **可以关联，但只覆盖 T2I/I2I。** #2908 明确把 SenseNova-U1 的 generation/editing 指向 #2831，
  但 #2831 的 scope 又明确排除了 thinking mode 和 interleaved generation。
  #2908 要求的 thinking/interleaved 目前没有 RFC 承载，属于“悬空要求”。
- **#2831 的 Phase 1/2 代码基本齐全，Phase 3 未完成。** 分支实现了 T2I/I2I、Images endpoints、
  usage/metrics、batching/multi-output、Cache-DiT、offload、TP/DP 和 NPU 路径；
  但 UMM 路径的 GPU/NPU、TP/DP、Cache-DiT、offload 验收均未执行。
- **#2908 的 adapter 契约只满足了单 phase 图像生成的部分。** 分支实现了 `start/advance/usage/close`、
  五个 bounds、segment envelope 和协作式取消；但没有 `progress` 回调、没有 thinking/interleaved、
  没有使用 #2919 的 shared generation stage。
- **分支修改了共享 UMM runtime，违反 #2908 “Adding a model 不改共享 runtime”的约定。**
  具体是 `completion_metadata`、`adapter_options`、deadline metadata 和 `forget` 的 close reason。
- **分支已 merge main。** `cd4b310fa` 把 main（含 #2918/#2919）合入分支，
  `pipeline/umm.py`/`docs/index.rst` 的文本冲突已解决；但按 no-runtime 决策，branch 的
  controller/scheduler 改动仍要撤回。
- **2026-10-19 决策已确认：UMM path 替换 native、#2831 增加后续 Phase、不改 UMM runtime。**
  目标架构和最小改动方案见第 8 节；前文与第 8 节冲突处，以第 8 节为准。

## 1. 分析对象与当前状态

### 1.1 两个 RFC 的定位

| RFC | 定位 | SenseNova 在其中的位置 |
| --- | --- | --- |
| [#2831](https://github.com/sgl-project/sglang-omni/issues/2831) | SenseNova-U1.5-8B-MoT 的模型适配、feature migration、硬件验收 | 模型级 T2I/I2I、Images API、GPU/NPU、优化项 |
| [#2908](https://github.com/sgl-project/sglang-omni/issues/2908) | UMM（统一多模态模型）共享运行时 | 在 model integration 表中把 SenseNova-U1 作为后续 consumer |

#2831 的 scope 原文覆盖：

- 模型：SenseNova-U1.5-8B-MoT。
- 任务：T2I 和 I2I。
- 硬件：GPU 和 Ascend NPU。
- Serving：`/v1/images/generations`、`/v1/images/edits`、Images usage、Prometheus metrics。
- Feature migration：从 SGLang 迁移选定的 SenseNova 能力和优化。

#2831 的 out of scope 原文包括：

- 其他 SenseNova 模型或任务，包括 text understanding、**thinking mode**、**interleaved generation**。
- 与 SGLang 的完整 feature parity。
- 新的优化研究。
- 新测试框架。

#2908 的 model integration 表对 SenseNova-U1 的描述是：

> Generation and editing as in #2831. For thinking and interleaved output, one phase runs the
> model's own interleaving loop and streams its segments, since generated pixels re-enter the
> model's context. Native multimodal_gen already serves SenseNova-U1, which makes the shared
> generation stage a candidate host, subject to qualification.

#2908 的 implementation plan 又写明：

> SenseNova-U1 thinking and interleaved | One streaming phase on the model's stage |
> Phase progress streaming | With the #2831 owners

### 1.2 分支与 main 的状态

- 分支 HEAD：`cd4b310fa`（2026-10-19 02:25 -0700），提交信息
  “Merge main into wes/sensenova-umm-adapter”，parents 是 `d6b9348f2` 和 `72bc179af`。
- main tip：`72bc179af`。分支现在包含 main；`git rev-list --left-right --count` 为 42/0，
  即领先 main 42 个 commit、落后 0。
- #2918 `8ad61f161` 和 #2919 `797ca2d09` 已经随 merge 进入分支。
- `sglang_omni/pipeline/umm.py` 的 merge 结果同时保留了 main 的 `dispatches`/三元组
  `apply()` 和分支的 `completion_metadata`/deadline key/`adapter_options`；`docs/index.rst`
  的冲突也已合并。**之前报告的“不能直接 merge/rebase”已经过期。**
- 但按 2026-10-19“不改 UMM runtime controller”的决策，这些分支 runtime 改动仍然需要
  撤回（见第 8 节），所以 merge 解决的是文本冲突，不是目标架构问题。
- 分支有 42 个 commit 领先 main，包含 #2325、#2351、#2354、#2621、#2653、#2654、#2670、
  #2672、#2724、#2728、#2766 等 #2831 表中的实现，以及 #2928/#2929/#2935 重构和
  UMM 集成。
- main 上**没有** `sglang_omni/models/sensenova_u1/`，也没有 SenseNova 的 docs/configs。
  这符合 #2831 “先在分支验收、再拆 PR 进 main”的安排，但不满足其最终目标。
- 分支当前没有 open PR 指向 main（GitHub API 查询为 0）。

### 1.3 本次实际执行的验证

在本地 CPU 环境（Python 3.12 + torch 2.9.1 + transformers 5.12.1）执行：

```text
tests/unit_test/sensenova/test_umm.py                         31 passed
tests/unit_test/sensenova/test_cache_dit.py
tests/unit_test/sensenova/test_sampling.py
tests/unit_test/sensenova/test_image_usage.py
tests/unit_test/sensenova/test_tp_config.py                   43 passed, 2 skipped
tests/unit_test/serve/test_images_usage_api.py                14 passed（用最小 sglang LazyImport stub）
以上合并执行                                                 88 passed, 2 skipped, 43 subtests passed
```

`test_images_usage_api.py` 覆盖了真实 controller 产生的 UMM final result 转成 Images
response、n>1、I2I、usage 和 400/503 rejection；为了绕开本地缺少完整 `sglang` 安装的问题，
给它注入了一个只提供 `LazyImport` 的最小 stub。这验证的是本仓库的 UMM/Images 契约，不是
完整 SGLang 服务安装。

`ruff check`、`black --check`、仓库的 `check_leading_underscore.py` 和 `check_if_else.py`
在本次检查的改动文件上均通过。完整 SenseNova 模型套件（`test_t2i_pipeline.py`、
`test_npu_adaptation.py`、`test_multi_output.py` 等）需要真实的 `sglang` 模型/平台依赖，
本地环境没有安装，因此**没有**执行；这与分支文档中“CPU contract tests 不等于完整服务验收”
的表述一致。

在 `cd4b310fa` merge main 之后，又在同一环境重跑了上面这组测试：仍是
88 passed, 2 skipped, 43 subtests passed。说明 merge 保留了 branch 的 UMM controller
改动后，现有 CPU contract 仍然通过；这不代表“no-runtime”目标已经达成。

实现阶段（2026-10-19）在本地 stub 环境下扩大回归到 T2I、multi-output、native module、
denoise、TP/Gloo、Images API 和 metrics：**237 passed, 2 skipped, 43 subtests passed**。
`test_npu_adaptation.py` 仍需要真实 SGLang platform，`test_fused_rope.py` 的 NPU 分支在本
地 stub 下不成立；这两组留给完整 CI/NPU 环境。

## 2. #2831 与 #2908 的关联性

### 2.1 关联矩阵

| #2908 对 SenseNova 的要求 | #2831 是否覆盖 | 分支实现 | 判定 |
| --- | --- | --- | --- |
| T2I generation | 是（Phase 1/2） | 是 | 已关联 |
| I2I editing | 是（Phase 1/2） | 是 | 已关联 |
| `/v1/images/generations`、`/v1/images/edits` | 是（Phase 1） | 是 | 已关联 |
| Images usage、metrics | 是（Phase 2） | 分支有，main 未合 | 部分关联 |
| Adapter `start/advance/usage/close` | 未提及（#2908 定义） | 是 | 单 phase 可用 |
| UMM segment envelope | 未提及（#2908 定义） | 是 | 单 phase 可用 |
| 五个 controller bounds | 未提及（#2908 定义） | 是 | 可用 |
| Progress streaming | 未提及（#2908 planned） | 否 | 未关联 |
| Thinking mode | **明确 out of scope** | 否 | **悬空** |
| Interleaved output | **明确 out of scope** | 否 | **悬空** |
| Shared generation stage | 未提及（#2908 #2919） | 否，使用自定义 stage | 分歧 |
| 不修改共享 runtime | #2908 “Adding a model” 要求 | **否** | 不符合 |
| Parallel phases | #2908 #2918 | 已合入（merge），adapter 单 phase 不使用 | 目标采用 main controller |
| Request-scoped context release | #2908 planned | 不涉及（单 phase） | N/A |

### 2.2 关联结论

1. **生成/编辑部分可以关联。** #2908 把 SenseNova 的 generation/editing 委托给 #2831，
   分支的 T2I/I2I 实现也确实落在 #2831 的 Phase 1/2 范围内。
2. **thinking/interleaved 不能直接关联到当前 #2831。** #2831 明确把它们排除在 scope 外，
   而 #2908 又把它们列为 SenseNova 的后续设计。这不是实现问题，是 RFC 边界问题：
   需要一个 follow-up RFC/issue，或修改 #2831 的 scope 和 phase 计划。
3. **#2908 的两个运行时前提并未满足。**
   - `progress` streaming 仍是 Proposed，没有合入的 PR；没有它，thinking text 和
     interleaved segments 无法在 phase 结束前到达客户端。
   - shared generation stage #2919 已合入 main，但当前分支没有使用；而且 SGLang 上游
     `multimodal_gen` 对 SenseNova 的支持是分项的：T2I 已合入
     [#36606](https://github.com/sgl-project/sglang/pull/36606)，I2I
     [#39081](https://github.com/sgl-project/sglang/pull/39081)、thinking
     [#40702](https://github.com/sgl-project/sglang/pull/40702)、interleaved
     [#35038](https://github.com/sgl-project/sglang/pull/35038) 仍 open。
     因此 #2908 的 “Native multimodal_gen already serves SenseNova-U1” 只对 T2I 成立，
     不能据此要求当前分支直接切到 shared generation stage。
4. **分支的 UMM T2I/I2I 路径本身不在 #2831 的 planned work 里。** #2831 描述的是 native
   pipeline（`sensenova_generate`），分支最近新增的是 opt-in 的共享 UMM controller 路径。
   这是一条“同一模型、第二套 serving 编排”的路径；#2831 没有它的验收项，#2908 也没有把它
   列为单独的 implementation unit。需要明确它是替代 native path，还是 experimental 的第二路径。

## 3. 分支对 #2831 的符合度

### 3.1 Phase 1：native model and serving adaptation

| #2831 planned work | 代码证据 | 测试/文档 | 判定 |
| --- | --- | --- | --- |
| 模型/tokenizer 加载、dtype、device placement | [`stages.py`](../../sglang_omni/models/sensenova_u1/stages.py#L159) 的 `create_generation_executor`；[`config.py`](../../sglang_omni/models/sensenova_u1/config.py#L21) | `test_tp_config.py`、`test_t2i_pipeline.py`、cookbook | 符合 |
| T2I 与 I2I 执行路径 | [`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L35) 的 `generate_image`、[`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L274) 的 `generate_image_edit` | `test_t2i_pipeline.py`、`test_multi_output.py` | 符合 |
| Images endpoints | [`openai_api.py`](../../sglang_omni/serve/openai_api.py#L1509)、[`openai_api.py`](../../sglang_omni/serve/openai_api.py#L1576) | `test_t2i_pipeline.py`、`test_images_usage_api.py` | 符合 |
| prompt、reference、参数、seed、输出 | [`sampling.py`](../../sglang_omni/models/sensenova_u1/sampling.py)、[`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L139) | `test_multi_output.py`、`test_t2i_pipeline.py` | 符合 |
| GPU/NPU 路径、平台算子与 fallback | [`attention_backend.py`](../../sglang_omni/models/sensenova_u1/neo_unify/attention_backend.py#L24)、[`qwen_feed_forward.py`](../../sglang_omni/models/sensenova_u1/neo_unify/qwen_feed_forward.py#L14)、[`rotary_embeddings.py`](../../sglang_omni/models/sensenova_u1/neo_unify/rotary_embeddings.py#L39) | `test_npu_adaptation.py`、`test_fused_rope.py`、cookbook | 代码符合；硬件验收部分完成 |
| focused unit tests 和 deployment docs | `tests/unit_test/sensenova/`、[`sensenova_u1.md`](../cookbook/sensenova_u1.md#L1) | — | 基本符合；部分覆盖缺口见 3.4 |

### 3.2 Phase 2：selected feature and optimization migration

| #2831 planned work | 代码证据 | 测试/文档 | 判定 |
| --- | --- | --- | --- |
| T2I batching、native multi-output、prefix-KV reuse、seed/output ordering | [`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L55)、[`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L197)、`neo_unify/text_image_generation.py` | `test_multi_output.py`、`test_t2i_pipeline.py`、`test_native_module_compat.py` | 符合 |
| NPU attention/KV layout/MLP/timestep preparation | [`attention_backend.py`](../../sglang_omni/models/sensenova_u1/neo_unify/attention_backend.py#L149)、[`conditioned_image_denoise.py`](../../sglang_omni/models/sensenova_u1/neo_unify/conditioned_image_denoise.py#L185)、[`text_image_denoise.py`](../../sglang_omni/models/sensenova_u1/neo_unify/text_image_denoise.py#L168) | `test_npu_adaptation.py` | 代码符合；trace 未执行 |
| shared RMSNorm dispatch、减少 host-device sync | [`qwen_feed_forward.py`](../../sglang_omni/models/sensenova_u1/neo_unify/qwen_feed_forward.py#L14) 的 `SenseNovaRMSNorm` | `test_npu_adaptation.py` | 符合 |
| RoPE table 复用、GPU/NPU fused RoPE | [`rotary_embeddings.py`](../../sglang_omni/models/sensenova_u1/neo_unify/rotary_embeddings.py#L115)、[`modeling_qwen3.py`](../../sglang_omni/models/sensenova_u1/neo_unify/modeling_qwen3.py#L284) | `test_denoise_optimizations.py`、`test_fused_rope.py` | 代码符合；NPU fuse 为 opt-in |
| Cache-DiT 的 request/branch state isolation | [`cache_dit.py`](../../sglang_omni/models/sensenova_u1/cache_dit.py#L30) | `test_cache_dit.py` | 符合 |
| layerwise offload 的 loading/prefetch/cleanup | [`offload.py`](../../sglang_omni/models/sensenova_u1/offload.py#L102) | 只有 config 测试 | **部分符合**：缺少真实 offload 生命周期回归 |
| DP/TP serving、worker lifecycle、orderly shutdown | [`tensor_parallel.py`](../../sglang_omni/models/sensenova_u1/tensor_parallel.py#L149)、[`config.py`](../../sglang_omni/models/sensenova_u1/config.py#L70)、`scheduling/tensor_parallel_scheduler.py` | `test_tensor_parallel.py`、`test_native_tp.py`、`test_tp_config.py`、`test_t2i_pipeline.py` | 代码/CPU 测试符合；真实多卡未验收 |
| 两个 Images endpoints 的 usage | [`usage.py`](../../sglang_omni/models/sensenova_u1/usage.py#L10)、[`openai_api.py`](../../sglang_omni/serve/openai_api.py#L1509) | `test_image_usage.py`、`test_images_usage_api.py` | 符合；mainline #2723 仍 open |
| request/scheduling/stage/image metrics | [`request_metrics.py`](../../sglang_omni/serve/request_metrics.py#L164)、`docs/metrics.md` | `test_metrics*.py`、`test_images_usage_api.py` | 分支符合；mainline #2727 仍 open |

### 3.3 Phase 3：hardware validation

#2831 的 Phase 3 全部未勾选。分支文档也明确写了：

> GPU/NPU launch, baseline image comparison, batching throughput, TP/DP inference,
> Cache-DiT, offload, and business acceptance remain unexecuted for this UMM path.
> The existing pipeline's hardware results do not establish these properties for the new
> configuration.

也就是说：

- **native pipeline** 有部分历史硬件证据（GPU T2I/I2I、Ascend I2I、NPU multi-output、
  usage/metrics），但不够关闭 Phase 3。
- **opt-in UMM path** 的 GPU/NPU 推理完全未执行；CPU contract tests 不能替代。
- 因此不能声称 #2831 已完成；只能声称 Phase 1/2 的代码已具备，Phase 3 待验收。

**CI 覆盖缺口**

仓库的 [`.github/workflows/test.yaml`](../../.github/workflows/test.yaml#L23) 分成两个 job：

- `unit-test`：`CUDA_VISIBLE_DEVICES=""`，跑 `pytest tests/ -m "not benchmark and not accelerator"`。
- `unit-test-accelerator`：跑 `pytest tests/ -m "accelerator and not benchmark"`。

SenseNova 的硬件测试没有 `accelerator` marker，所以：

- `test_fused_rope.py::test_cuda_bf16_rope_matches_eager` 用
  `@pytest.mark.skipif(not torch.cuda.is_available())`，在 `unit-test` 里被 skip；在
  `unit-test-accelerator` 里因为没有 marker 被 deselect。它实际不会在 CI 跑。
- `test_npu_adaptation.py::test_randn_fallback_preserves_accelerator_rng` 在没有 accelerator
  时 `pytest.skip`，同样没有 marker，因此也不会在 accelerator job 跑。
- `.github/` 里没有 SenseNova 的 model CI lane / label（对比 TTS、ASR、Qwen3-Omni 都有
  独立的 `test-*-ci.yaml`）。

结论：**CPU contract tests 会跑；真实 GPU/NPU 的 SenseNova 测试没有 CI 入口**。这直接
影响 #2831 “focused regression coverage” 和 Phase 3 的证据链。

**Phase 3 验收矩阵**

| 能力 | GPU | NPU | 当前证据/缺口 |
| --- | --- | --- | --- |
| T2I | 部分 | 部分 | #2325/#2351 的 PR 级检查；没有 CI model lane |
| I2I | 部分 | 部分 | #2351/#2354 的 PR 级检查；没有 CI model lane |
| n>1 multi-output | 未验证 | 部分 | #2653 的 NPU multi-output 检查；GPU 未验证 |
| Dynamic batching | 未验证 | 未验证 | 只有 CPU/mock 测试；文档说 opt-in |
| TP | 未验证 | 未验证 | Gloo CPU parity；NCCL/HCCL 推理未验证 |
| DP | 未验证 | 未验证 | 只有 config 测试和文档 |
| Cache-DiT | 未验证 | 未验证 | 只有 unit tests；真实模型/硬件未验证 |
| Layerwise offload | 未验证 | N/A（CUDA only） | 只有 config 测试；没有生命周期测试 |
| Cancellation/deadline | 未验证 | 未验证 | CPU controller tests；NCCL/HCCL 未验证 |
| UMM path（全部） | 未执行 | 未执行 | 只有 CPU contract + fake coordinator |
| Images metrics | 部分 | 部分 | #2728 的 native 证据；UMM metrics 漏计 |

这张表也解释了为什么 #2831 Phase 3 不能勾选：不是代码不存在，而是**每条能力在不同
hardware/path 组合上的证据层级不一致**，且没有 CI 或硬件报告把它们串起来。

### 3.4 完成标准与 mainline

#2831 的 completion criteria 要求“selected capabilities 有 focused regression coverage 和
documented limitations”。当前有两个明确缺口：

1. **layerwise offload 没有真实生命周期测试。** `offload.py` 有完整的 load/prefetch/release/close
   实现，但测试只覆盖配置传递，没有覆盖权重复位、失败清理、并发拒绝等行为。
2. **NPU fused RoPE 的 opt-in 开关没有文档。** 代码使用
   `SGLANG_SENSENOVA_NPU_FUSED_ROPE`，但 cookbook 没有说明如何开启、限制是什么；
   而 #2831 的 compatibility requirements 要求每个迁移能力记录 hardware/configuration 组合。
   Cache-DiT 一节也没有明确 GPU/NPU 支持矩阵，只描述了 guidance 限制。

mainline 方面：

- #2723（shared Images response）和 #2727（shared metrics）仍 open。
- 分支包含这两个实现，但 main 没有 SenseNova，也没有这些 shared foundation。
- 分支还在 `wes/sensenova-umm-adapter` 上合入了 #2928/#2929/#2935 等分支内重构；
  这些不是 mainline PR，后续需要拆分/重放。
- 当前没有 open PR 指向 main（GitHub API 查询为 0），因此 main 的 PR CI gate 没有覆盖
  这个分支；结合上面的 accelerator marker 缺口，硬件测试证据不会自动积累。
- #2831 的最终目标“bring the validated implementation into main through reviewable PRs”
  目前没有开始。
- #2831 的 Progress Tracking 仍写着 “reviewed at `d69e0381b`”（2026-10-15）；当前分支
  HEAD `cd4b310fa` 已经包含 #2928/#2929/#2935、UMM 集成和 main 的 #2918/#2919，RFC 的
  进度记录已经过期，
  也没有跟踪 opt-in UMM path。

**两条路径的行为差异**

分支文档说 UMM path “targets the same T2I/I2I workloads”，但它不是 native path 的严格等价：

| 行为 | native path | UMM path | 影响 |
| --- | --- | --- | --- |
| 请求 deadline | 没有 controller deadline | `timeout_s=600` | 长请求会在 UMM path 上 500/expired；native path 不会 |
| 输入/输出 admission | 只有 endpoint/model 自身校验 | 六个 `SenseNovaAdmissionLimits`（images/pixels/references/bytes/steps） | 同一请求可能在 UMM path 被 400，在 native path 成功 |
| TP 取消 | 文档：各 rank 跑完 forward，stage 丢弃结果 | `cooperative_cancellation=True`，在 checkpoint 停止 | 两条路径的 TP 取消语义不同 |
| 并发 | stage scheduler 的队列/批处理 | controller `max_active=8` + stage 队列 | 并发上限和多请求排队行为不同 |
| Images metrics | 从 chunk 的 `images_b64` 统计 | UMM segments 可能不被 metrics 识别（见 5.2） | 同一请求的 `generated_images_total` 可能不同 |

因此 #2831 的 Phase 3 验收不能只测 native path；如果保留 UMM path，必须把上述边界单独
测试并在文档中说明。**决策后 UMM path 替换 native，这些差异不再是“两条路径”的问题，
而是迁移时必须收敛或明确接受的新默认行为**（见 8.3/8.7）。

## 4. 分支对 #2908 的符合度

### 4.1 Adapter contract

| #2908 契约 | 分支实现 | 判定 |
| --- | --- | --- |
| `start(context) -> UMMStep` | [`umm.py`](../../sglang_omni/models/sensenova_u1/umm.py#L47)：校验 admission、记录请求、派发 `generate` | 符合 |
| `advance(context, phase, result) -> UMMStep` | [`umm.py`](../../sglang_omni/models/sensenova_u1/umm.py#L66)：校验 phase、图片数量、封装 image segments | 符合（单 phase） |
| `usage(context) -> bytes` | [`umm.py`](../../sglang_omni/models/sensenova_u1/umm.py#L93)：输入字节 + 输出预留 | 符合 |
| `close(context, reason)` | [`umm.py`](../../sglang_omni/models/sensenova_u1/umm.py#L101)：释放 per-request state | 符合 |
| 五个 bounds | [`umm_config.py`](../../sglang_omni/models/sensenova_u1/umm_config.py#L41)：`max_dispatches=1`、`max_outputs=10`、`max_active=8`、`max_retained_bytes=256MiB`、`timeout_s=600` | 符合 |
| pipeline config 校验 | [`umm_config.py`](../../sglang_omni/models/sensenova_u1/umm_config.py#L71) + 基类 `UMMPipelineConfig` | 符合 |
| phase stage 返回原 request | `generate_image*` 只改 `payload.data`，metadata/continuation 保留 | 符合 |
| segment envelope | [`image_media.py`](../../sglang_omni/client/image_media.py#L13)：`kind/mime_type/size_bytes/sha256/url`；controller 补 `execution_id/segment_index` | 符合 |
| 取消/超时/close once | controller `release/expire` + [`cancellation.py`](../../sglang_omni/models/sensenova_u1/cancellation.py#L22) | 基本符合；见 4.4 |
| parallel phases (`dispatches`) | 已合入（merge）；adapter 单 phase 不使用 | 目标采用 main controller |
| `progress` streaming | 无 | 不符合 #2908 的 SenseNova 后续要求 |
| request-scoped context release | 不涉及 | N/A |

### 4.2 共享 runtime 的改动（关键不符合项）

#2908 在 “Adding a model” 中写明：

> No step changes shared runtime code.

分支对共享 runtime 做了 6 类改动：

| 改动 | 位置 | 用途 | 问题 |
| --- | --- | --- | --- |
| `UMMStep.completion_metadata` + final `metadata` | [`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L110)、[`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L669) | 把 `image_usage` 带回客户端 | 不在 #2908 契约中；客户端可见结果多了一个字段；#2908 的 final result 只定义 segments + joined text |
| `UMM_DEADLINE_METADATA_KEY` | [`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L46)、[`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L689) | 让 phase stage 在 checkpoint 自取消 | phase stage 因此知道 controller 的内部 key；#2908 说 phase stage “need no knowledge of the controller” |
| `adapter_options` 透传 | [`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L777) | 把 admission limits 传给 adapter | 共享 factory 的新参数；不在 #2908 契约中 |
| `forget` 区分 expired/aborted | [`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L298) | close reason 更准确 | 共享行为变化；需要契约/测试 |
| client 读取 `metadata.image_usage` | [`client.py`](../../sglang_omni/client/client.py#L1122) | UMM final result 的 usage | 与 `completion_metadata` 绑定；属于客户端契约扩展 |
| `RequestCancelledError` + `TensorParallelScheduler.cooperative_cancellation` | [`message.py`](../../sglang_omni/scheduling/message.py#L16)、[`tensor_parallel_scheduler.py`](../../sglang_omni/scheduling/tensor_parallel_scheduler.py#L42) | 让 SenseNova TP 在 checkpoint 停止 | 共享 scheduler 的新模式/异常；默认关闭，但不在 #2908 契约中 |

这些改动**不是模型名或 tuned values**，技术上可以做成通用能力；但按 #2908 的流程，它们应该
作为 RFC 的 runtime additions 明确列出，而不是随模型 adapter 一起改。当前分支文档也承认
“does not claim upstream RFC approval”。
**2026-10-19 决策后不再采用“进 runtime”方案**：按 8.2/8.5，deadline/TP 放模型侧，
`completion_metadata` 改成 usage segment，`adapter_options` 改成固定 limits；本节只保留作
决策前的分析记录。

### 4.3 输出、usage、bounds、取消

- **输出契约**：分支把每张图封装为 `InlineImage`，controller 生成连续 `segment_index`；
  Images endpoint 再还原成 `images_b64`/`image_b64`。这与 #2908 的 segment envelope 一致。
- **usage**：`image_usage` 走 `completion_metadata`，不进入 segment；`adapter.usage()` 报
  input bytes + 输出预留。两者语义分开，和分支文档一致，但 `completion_metadata` 本身是
  RFC 外扩展。这里还有一个语义偏差：#2908 定义 `usage` 为 adapter 的 retained state，
  而分支把 future output reservation 也算进去；这更保守，但会让“尚未存在的输出”提前消耗
  `max_retained_bytes`。需要在 RFC 里明确这是允许的保守估计，或把 reservation 放回
  admission/config 层。
- **bounds**：`max_outputs`、`max_active`、`max_retained_bytes`、`timeout_s` 都由 controller
  检查；`max_dispatches=1` 与单 phase 匹配；`max_images` 与 `max_outputs` 在 config 里交叉校验。
- **取消**：[`cancellation.py`](../../sglang_omni/models/sensenova_u1/cancellation.py#L66) 的
  `GenerationScope.checkpoint()` 用 `all(...)` 判断 batch 内是否全部取消/过期；TP 下用
  `all_reduce(MAX)` 让所有 rank 一致。一个 cancelled member 不会单独中断 live members，
  符合分支文档的约束。
- **超时**：controller 生成 `UMMDeadlineError`，`generation_error_status_code` 对它返回 500。
  #2908 只要求“deadline expiry 停止后续 dispatch 并 abort”，没有规定 HTTP 状态；
  但“超时返回 500”是否可接受需要明确。shed 路径返回 503，两者语义不同。

### 4.4 未满足 #2908 的 SenseNova 设计要求

1. **thinking mode 未接入。** [`sampling.py`](../../sglang_omni/models/sensenova_u1/sampling.py)
   对 `think_mode=True` 直接报 “not supported yet”；[`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L139)
   和 I2I 路径都硬编码 `think_mode=False`；cookbook 也写明 “thinking/SRT KV transfer are not
   exposed by the Omni stage”。
2. **interleaved generation 未接入。** `neo_unify` 里有 `interleaved_generation.py`、
   `interleaved_image_denoise.py`、`interleaved_text_image_denoise.py` 等模型代码，但
   Omni stage 没有调用它们；UMM adapter 只有 `generate` 一个 phase，输出只有 image segments。
3. **progress streaming 未实现。** 共享 controller 仍要求 phase stage 的 `stream_to` 为空
   （`UMMPipelineConfig` 校验），也没有 `progress(context, phase, item)` 回调；因此
   #2908 设想的“one phase runs the model's own interleaving loop and streams its segments”
   在当前 runtime 上无法实现。
4. **shared generation stage 未使用。** 分支使用自定义 `sensenova_generate` stage
   （[`umm_config.py`](../../sglang_omni/models/sensenova_u1/umm_config.py#L53)）；#2919 的
   `sglang_omni/native_generation/stage.py` 已在 main 上，merge 后也在分支里，但 SenseNova
   没有使用它。是否切换取决于 SGLang `multimodal_gen` 对 I2I/TP/Cache-DiT 的支持，当前上游
   I2I 仍 open。另外 #2919 的 inline media 上限是 8MB，而 SenseNova 的 admission 允许更大的
   输出；直接切换会被 8MB 上限拒绝或截断，需要先调整/qualification。
5. **phase stage 知道 controller 内部 key。** [`cancellation.py`](../../sglang_omni/models/sensenova_u1/cancellation.py#L13)
   从 `pipeline.umm` 导入 `UMM_DEADLINE_METADATA_KEY`；这违反 #2908 “phase stages return the
   request they received, so they need no knowledge of the controller”。可以改成 adapter 在
   request metadata 里写模型自己的 deadline key，phase stage 只读模型 key。

## 5. 具体问题与风险

### 5.1 P0：阻塞合入/阻塞目标

1. **与 main 的 `pipeline/umm.py` 冲突已经 merge 解决，但 runtime 改动仍要撤回。**
   `cd4b310fa` 已把 main 的 `dispatches`/三元组 `apply()` 和分支的
   `completion_metadata`/deadline key/`adapter_options` 合并在一起，CPU 测试仍通过。
   按“不改 UMM runtime controller”的决策，这些 branch 改动不能保留；目标是把
   `pipeline/umm.py` revert 到 main，而不是继续维护 merge 后的混合版本。
2. **RFC scope 未对齐。**
   #2831 排除 thinking/interleaved；#2908 要求 SenseNova 的 thinking/interleaved。
   **决策已确认修改 #2831 增加后续 Phase**；在 RFC 文本更新前，实现仍没有验收依据。
3. **共享 runtime 改动没有 RFC 归属。**
   `completion_metadata`、`adapter_options`、deadline metadata、`forget` 语义变化需要
   作为 #2908 的 runtime additions 讨论，或者改成模型侧实现；否则每个模型 adapter 都会
   继续扩展 controller。**决策已确认“不改 UMM runtime”**，所以这些改动必须按 8.2
   移除或替换，不能原样合入。
4. **UMM path 的硬件验收为零。**
   #2831 Phase 3 和分支文档都承认未执行；在 GPU/NPU 上至少要做 T2I/I2I、n>1、batching、
   TP/DP、Cache-DiT、offload、取消/超时、失败恢复的对比验收。**决策后 UMM 替换 native，
   这项从“可选”变成“必须”。**

### 5.2 P1：功能/一致性

1. **UMM 路径的 image metrics 可能漏计。**
   `RequestMetrics.generate` 只看 `chunk.images_b64` / `chunk.image_b64`
   （[`request_metrics.py`](../../sglang_omni/serve/request_metrics.py#L213)）；UMM 流式 chunk
   用 `segment` 携带图片，final chunk 的 segments 又去掉了 data。`image_result_with_disconnect_watch`
   是在 metrics 包装之后才把 segments 还原成 `images_b64`
   （[`openai_api.py`](../../sglang_omni/serve/openai_api.py#L1746)）。因此走 UMM 的
   `/v1/images/*` 可能不会增加 `generated_images_total`。如果 UMM path 要保留 #2831 的
   metrics 验收，需要在 metrics 层识别 segments，或在 client 层更早还原。
   实测 `umm_result_chunk()` 对带 image segment 的 final result 返回
   `images_b64=None`、`image_b64=None`，只有 `segments` 带 data；这确认了 metrics 看不到图片。
2. **UMM pipeline 的端到端 routing 没有集成测试。** `test_umm.py` 直接调用
   `controller.transition()`；`test_images_usage_api.py` 用 fake `GenerationCoordinator`；
   两者都没有跑真正的 `Stage`/coordinator、`route_umm`、`UMMOutputRelay`、`stream_to` 和
   local edge。配置校验通过不等于 pipeline 能正确路由 final result 和 stream segments。
3. **UMM + DP/TP 没有配置示例或运行测试。** native path 有 `sensenova_u1_dp2.yaml`、
   `sensenova_u1_tp2.yaml`、`sensenova_u1_dp2_tp2.yaml` 和对应测试；UMM path 只有
   `sensenova_u1_umm.yaml`。配置构造本身可用（本次用 `SenseNovaU1UMMPipelineConfig`
   手工加了 `tp_size=2` 和 `processes`，能展开成 `generate@r0/r1`），但没有 UMM + TP/DP
   的 stage 运行、路由或硬件测试。
4. **offload 回归覆盖不足。** 只有配置测试；`offload.py` 的 load/prefetch/release/close
   没有真实生命周期测试。
5. **NPU fused RoPE 文档缺失。** `SGLANG_SENSENOVA_NPU_FUSED_ROPE` 在代码中出现，但
   cookbook 没有说明；#2831 要求记录 hardware/configuration 组合。
6. **admission 的图片校验异常类型不完整。** [`admission.py`](../../sglang_omni/models/sensenova_u1/admission.py#L43)
   只把 `Image.DecompressionBombError` 转成 `ValueError`；`image.verify()` 可能抛
   `SyntaxError` 等，controller 只捕获 `(ValueError, OSError)`，会变成 500 而不是 400。
   同样的 `verify()` 异常在 [`read_image_edit_upload`](../../sglang_omni/serve/openai_api.py#L1810)
   里也只捕获 `(OSError, Image.DecompressionBombError)`；malformed upload 可能走 500。
7. **deadline 的 HTTP 状态需要明确。** 当前是 500；如果客户端应重试，503/504 更合适。
   需要和 #2908 的 shedding 503 语义一起定义。
8. **deadline metadata key 与 #2919 重复。** #2919 的 shared generation stage 用
   `generation_deadline`，UMM controller 用 `umm_deadline_monotonic_s`；两者含义/时钟基准
   不同。如果以后 SenseNova 走 shared generation stage，会出现两套 deadline 约定。建议统一
   成一个通用 key，或把它降级为模型侧 metadata。
9. **TP 下每个 denoise step 都 all-reduce 一次。** [`cancellation.py`](../../sglang_omni/models/sensenova_u1/cancellation.py#L75)
   的 checkpoint 在 TP 模式每步同步；功能正确，但可能抵消 TP 收益。需要性能测量，或改成
   低频检查/只在对齐的 collective 边界检查。
10. **mainline 依赖未合并。** #2723/#2727 仍 open；UMM 的 `image_usage` 依赖 #2723 的
    Images response foundation，metrics 依赖 #2727。mainline SenseNova 落地前必须先合这两个
    （或把它们的 diff 一起带上）。
11. **分支内重构未拆 PR。** #2928/#2929/#2935 已合入分支，但 base 是 main；这些重构和
    SenseNova 功能混在一起，后续拆 PR 时需要单独 review。

### 5.3 P2：清理项

- `client/types.py` 从 `typing.TypedDict` 改成 `typing_extensions.TypedDict` 看起来与
  SenseNova 无关；确认是否必要，否则缩小 diff。
- `InlineImage` 与 #2919 `native_generation/stage.py` 的 inline media dict 有重复的字段
  定义；移除 runtime 改动/复用 #2919 helper 后应尽量共用一个 schema。
- `stages.py` 的 `with cancellation, offload.request() if offload is not None else nullcontext():`
  可读性较差；不是功能问题，但建议拆成局部变量。
- 分支文档的 “Shared UMM Integration Contract” 是分支本地契约，不是 RFC 契约；在 PR
  描述里要避免让读者误以为 #2908 已批准。

## 6. 建议的下一步

### 6.1 先做决策（不写代码）

**已确认（2026-10-19）**：UMM path 替换 native；#2831 增加后续 Phase；不改 UMM runtime。
以下保留为决策记录，具体设计以第 8 节为准。

1. **确认 UMM path 的定位**：它是 #2831 native path 的替代，还是 experimental 的第二路径？
   - 如果替代：需要把 native path 的验收项迁移到 UMM path，并说明何时删除 native path。
   - 如果并存：在 #2831 增加一节 “opt-in UMM path”，明确其验收范围和默认关闭。
2. **确认 thinking/interleaved 的归属**：
   - 修改 #2831 scope，增加 Phase 4（thinking/interleaved），并依赖 #2908 的 progress
     streaming；或
   - 新建 SenseNova-U1 thinking/interleaved RFC，由 #2908 引用。
3. **确认共享 runtime 扩展的归属**：
   - 把 `completion_metadata`、`adapter_options`、deadline metadata、`forget` 变化作为
     #2908 的 runtime additions 提案；或
   - 改成模型侧：adapter 在 request metadata 里写自己的 deadline key；usage 走一个明确的
     final-result 字段（但需要客户端契约）；adapter options 用模型专用 factory。

### 6.2 再 migrate/重构

1. merge 已在 `cd4b310fa` 完成（含 #2918/#2919），不再需要 rebase。按 no-runtime 决策，
   撤回 branch 对 `pipeline/umm.py`/scheduler 的改动，采用 main 的
   `UMMStep`/`apply()` 三元组；不要再维护 merge 后的混合 controller。
2. `completion_metadata` 不保留，改用 usage segment（8.2）；`dispatches` 保留在 main
   的 `UMMStep` 里，adapter 单 phase 不使用它。
3. 让 phase stage 不再 import `pipeline.umm`：adapter 在 `UMMStep.request.metadata` 里写
   模型自己的 deadline key（例如 `sensenova_deadline_monotonic_s`），`cancellation.py` 只读
   这个 key。
4. 复用 #2919 的 inline media schema/helper，必要时把 `InlineImage` 上移为共享客户端类型。
5. 把 #2928/#2929/#2935 的分支内重构拆成独立 PR 或至少独立 commit，避免和 UMM 功能混在一起。

**6.2.1 与 #2918 的具体冲突点**

merge 已在 `cd4b310fa` 完成；下表保留为差异说明。目标 no-runtime 是直接采用 main 的
controller，所以这些差异只影响 adapter/test 适配，不需要保留 branch 的 controller 实现。

| main（#2918）的现状 | 分支现状 | 合并动作 |
| --- | --- | --- |
| `UMMStep` 有 `dispatches: tuple[UMMDispatch, ...]` | 只有单 `phase`；新增 `completion_metadata` | 保留 `dispatches`，把 `completion_metadata` 加成并存的字段 |
| `UMMContext.expected` 是 `dict[str, Continuation]`，按 stage 记录多个 running phase | 是单个 `Continuation?` | 采用 main 的 dict；分支的“expected 单个”逻辑删除 |
| `apply()` 返回 `(result, added, started)`，等待时为 `None` | 返回 `(result, added)`，只支持单 phase | 采用 main 的三元组；`completion_metadata` 只在 `result` 为 final `StagePayload` 时写入 |
| `transition()` 用 `started` 和 `context.expected` 判断 `completed` | 用 `step.phase is None` 判断 | 采用 main 的 `completed` 逻辑；分支的 `step.phase is None` 不能覆盖“等待并行 phase”情况 |
| 每个 dispatch 独立构造 metadata | 只给一个 request 加 deadline | 把 `UMM_DEADLINE_METADATA_KEY` 加到 dispatch 循环的每个 metadata 里 |
| `create_umm_controller` 无 `adapter_options` | 有 `adapter_options` | 保留该参数，或改成模型专用 factory；需在 #2908 记录 |
| `forget()` 固定 `"aborted"` | 区分 `"expired"`/`"aborted"` | 保留分支改动，但补一个 close reason 的契约测试 |
| `emit_result` 处理 `RoutedPayloads` | 只处理 `StagePayload?` | 采用 main 的 `RoutedPayloads` 支持；`UMMOutputRelay` 不变 |

对应的测试调整：`tests/unit_test/sensenova/test_umm.py` 里依赖
`UMM_DEADLINE_METADATA_KEY`、`controller.transition()` 两段式、`controller.contexts` 的断言
需要在移除 branch controller 改动后重跑；`test_outputs_and_usage_survive_controller_completion`
应能在 main 的单 phase 路径上继续通过，但 `test_completed...` 一类多 phase 测试要按 #2918
的语义补充。

**6.2.2 共享 runtime 改动的替代方案**

不是所有 branch 的共享改动都必须进 #2908。逐项判断：

| 共享改动 | 是否必须进共享层 | 模型侧替代 | 建议 |
| --- | --- | --- | --- |
| `completion_metadata` | 需要把 `image_usage` 带回客户端；没有 controller 之外的 final-result hook | 用 `UMMOutput(kind="usage")` segment 或扩展 `GenerateChunk` | 两种都需要契约变更；建议把 `completion_metadata` 作为 #2908 的通用 addition 提案 |
| `UMM_DEADLINE_METADATA_KEY` | 不必：adapter 可以在 `start()` 返回的 request metadata 里写自己的 key | `SenseNovaUMMAdapter.start()` 写入 `sensenova_deadline_monotonic_s`，`cancellation.py` 只读这个 key | **移到模型侧**，删除共享 controller 的 deadline key |
| `adapter_options` | 不必：adapter 可以硬编码默认 limits | 模型专用 factory/限制常量；但配置覆盖会变难 | 保留为通用参数更合理，但要在 #2908 记录 |
| `forget()` 区分 expired | 不必：`expire()` 已经用 expired 释放；`forget()` 多数时候看到的是 aborted | adapter 的 `close()` 可以自己判断 deadline | 可删掉，减少共享行为变化 |
| `RequestCancelledError` + `TensorParallelScheduler` 模式 | 不必：可以定义 `SenseNovaTensorParallelScheduler` 子类 | 模型自己的异常 + 子类覆盖 `abort/consume_if_aborted/run_single` | **移到模型侧**，不修改共享 scheduler |

这张表给出的是“最小共享改动”方案：deadline 和 TP cancellation 完全可以模型侧做；
`completion_metadata` 和 `adapter_options` 则更适合作为 #2908 的显式 runtime additions。
把这两类分开，移除 runtime 改动后的 diff 会小很多。
**决策后进一步收紧为“不改 UMM runtime”**：`completion_metadata` 改成 usage segment，
`adapter_options` 改成固定 limits；以 8.2 为准。

### 6.3 再补验收

1. **CPU contract**：保持现有 31 个 UMM tests；补 UMM metrics、UMM cancellation 与真实
   stage/coordinator 的集成测试。可以直接借鉴 Cosmos3 的
   `tests/unit_test/serve/test_umm_serving.py`，改成 SenseNova 的 UMM pipeline。
2. **CI 入口**：给真实硬件测试加 `@pytest.mark.accelerator`，并新增 SenseNova 的 GPU
   nightly/PR lane（至少 T2I/I2I + TP/DP）；NPU lane 单独跑 Ascend。否则现有硬件测试
   会被 `unit-test` skip、被 `unit-test-accelerator` deselect。
3. **GPU/NPU functional**：T2I/I2I、n>1、batching、TP、DP、Cache-DiT、offload、取消/超时、
   失败恢复；与 `706053fa0` 对比（同 checkpoint、seed、hardware、参数、并发、warmup）。
4. **性能**：至少测 UMM path 的 TP 每步 all-reduce 开销、batching 吞吐、首图延迟；
   没有测量就不要声称性能。
5. **文档**：补 NPU fused RoPE 开关、UMM path 的 hardware/configuration 支持矩阵、
   未验证组合列表。可执行的验收步骤见
   [sensenova_phase3_checklist.md](sensenova_phase3_checklist.md)。

### 6.4 最后拆 mainline PR

建议顺序：

1. #2723 shared Images response。
2. #2727 shared metrics。
3. SenseNova 模型加载 + T2I stage。
4. I2I + NPU。
5. batching/multi-output/prefix-KV。
6. Cache-DiT、offload、TP/DP。
7. usage/metrics 集成。
8. opt-in UMM adapter（如果 RFC 接受）。

### 6.5 建议的 RFC 修改（草案）

这一节是给 #2831/#2908 owner 的提案文本，不是已经批准的 RFC 内容。
**决策后以 8.9 为准**：不再新增 UMM runtime additions，改为 #2831 Phase 4 + UMM 默认 +
usage segment/fixed limits 的说明。

**#2831 建议增加：**

- Scope 里增加一条：“An opt-in UMM-serving path may be developed on
  `wes/sensenova-umm-adapter` for the same T2I/I2I workloads. It reuses the shared UMM
  controller and stays non-default until Phase 3 accepts it.”
- Out of scope 保留 thinking/interleaved，但增加指向：“Thinking mode and interleaved
  generation remain out of scope here; track them in #2908 and a follow-up RFC.”
- Phase 3 增加：“If the opt-in UMM path is kept, validate its deadline, admission bounds,
  TP cancellation, Images metrics and output association separately from the native path.”

**#2908 建议二选一：**

- 方案 A（推荐，若这些能力要共享）：在 Runtime additions 表增加
  `completion_metadata`、phase deadline metadata、`adapter_options`，并明确
  `completion_metadata` 的客户端契约、计数方式和测试；`forget` 的 expired/aborted
  语义也要写进 close reason 契约。
- 方案 B（更保守）：这些能力留在 SenseNova 模型侧：adapter 在 request metadata 里写自己的
  deadline key、usage 走模型自己的 final-result 字段或专用 segment，不修改共享 controller。

**#2908 SenseNova row 建议修正：**

- “Native multimodal_gen already serves SenseNova-U1” 应限定为 T2I（SGLang #36606 已合入）；
  I2I（#39081）、thinking（#40702）、interleaved（#35038）仍 open。
- shared generation stage 目前只能覆盖 T2I；在 I2I 上游支持合入前，不能把它作为 SenseNova
  I2I 的验收前提。
- thinking/interleaved 明确依赖 progress streaming；在 progress streaming 落地前，这一行应
  保持 “blocked” 而不是 “candidate”。

## 7. 证据附录

### 7.1 关键代码位置

- UMM adapter：[`umm.py`](../../sglang_omni/models/sensenova_u1/umm.py#L41)。
- UMM pipeline config：[`umm_config.py`](../../sglang_omni/models/sensenova_u1/umm_config.py#L21)。
- UMM controller：[`pipeline/umm.py`](../../sglang_omni/pipeline/umm.py#L97)。
- SenseNova stage：[`stages.py`](../../sglang_omni/models/sensenova_u1/stages.py#L159)。
- 生成路径：[`generation.py`](../../sglang_omni/models/sensenova_u1/generation.py#L35)。
- 取消：[`cancellation.py`](../../sglang_omni/models/sensenova_u1/cancellation.py#L22)。
- Cache-DiT：[`cache_dit.py`](../../sglang_omni/models/sensenova_u1/cache_dit.py#L30)。
- offload：[`offload.py`](../../sglang_omni/models/sensenova_u1/offload.py#L102)。
- TP：[`tensor_parallel.py`](../../sglang_omni/models/sensenova_u1/tensor_parallel.py#L149)。
- Images API：[`openai_api.py`](../../sglang_omni/serve/openai_api.py#L1509)。
- inline media：[`image_media.py`](../../sglang_omni/client/image_media.py#L13)。
- cookbook：[`sensenova_u1.md`](../cookbook/sensenova_u1.md#L226)。

### 7.2 RFC/PR 链接

- [#2831 SenseNova RFC](https://github.com/sgl-project/sglang-omni/issues/2831)
- [#2908 UMM RFC](https://github.com/sgl-project/sglang-omni/issues/2908)
- [#2918 parallel phases](https://github.com/sgl-project/sglang-omni/pull/2918)
- [#2919 shared generation stage](https://github.com/sgl-project/sglang-omni/pull/2919)
- [#2723 shared Images response](https://github.com/sgl-project/sglang-omni/pull/2723)
- [#2727 shared metrics](https://github.com/sgl-project/sglang-omni/pull/2727)
- [#1502 one segment contract / LLaDA2](https://github.com/sgl-project/sglang-omni/pull/1502)
- [SGLang #36606 SenseNova T2I](https://github.com/sgl-project/sglang/pull/36606)
- [SGLang #39081 SenseNova I2I](https://github.com/sgl-project/sglang/pull/39081)
- [SGLang #40702 SenseNova thinking](https://github.com/sgl-project/sglang/pull/40702)
- [SGLang #35038 SenseNova interleave](https://github.com/sgl-project/sglang/pull/35038)

### 7.3 一句话判断

**#2831 是“模型能不能在 Omni 里跑对”的 RFC，#2908 是“UMM 模型怎么共享一套编排”的 RFC；
当前分支把两者在 T2I/I2I 上接了起来，并且已经 merge 了 #2918/#2919。下一步不是 rebase，
而是按 no-runtime 决策撤回 branch 的 controller/scheduler 改动、改用 usage segment，再按
#2831 的 Phase 3 做硬件验收。**

## 8. 按决策的目标架构（UMM 替换 native，不改 UMM runtime）

> 本节依据 2026-10-19 的三个决策；前文与本节冲突处，以本节为准。

### 8.1 约束推导

「不改动 UMM runtime」按以下范围理解（2026-10-19 已确认）：

- `sglang_omni/pipeline/umm.py` 保持 main（含 #2918）的版本：`UMMStep` 没有
  `completion_metadata`，`create_umm_controller` 没有 `adapter_options`，`forget` 固定
  `aborted`，没有 `UMM_DEADLINE_METADATA_KEY`。
- `UMMStep` 只能用 main 已有的字段：`phase`、`request`、`outputs`、`context`、
  `dispatches`。
- `client/` 与 `serve/` 的 UMM 结果处理、Images usage 和 metrics **允许**最小适配；这是
  保留 #2831 usage/metrics 的前提。

在这个约束下，目标形态是：

```text
Images API -> UMM orchestrator (unchanged) -> SenseNova generate stage
           -> UMM output stage (unchanged) -> Images response
```

- SenseNova 只有一个 `generate` phase；T2I/I2I 都在这个 phase 里完成。
- `image_usage` 不再走 `completion_metadata`，而是作为最后一个 `UMMOutput(kind="usage")`
  segment 返回。
- deadline、TP cancellation、limits 全部模型侧实现；不修改 controller/scheduler。

### 8.2 Adapter 设计（不改 runtime）

**当前分支 → 目标实现的对应关系**

| 当前分支 | 目标实现 | 动作 |
| --- | --- | --- |
| `UMMStep.completion_metadata` | `UMMOutput(kind="usage")` | 删除共享字段；adapter 追加 usage segment |
| `adapter_options` | 模型常量/环境变量 | 删除共享参数；`SenseNovaUMMAdapter(model_path)` 自带 limits |
| `UMM_DEADLINE_METADATA_KEY` | `sensenova_deadline_monotonic_s` | adapter 写入；`cancellation.py` 只读模型 key |
| `RequestCancelledError` + 共享 TP scheduler 模式 | `SenseNovaCancelledError` + `SenseNovaTensorParallelScheduler` | revert 共享 scheduler；模型子类 |
| `forget()` 区分 expired | 不改 | revert；adapter close 不依赖 reason |
| native `SenseNovaU1PipelineConfig` 默认 | UMM config 默认 | 合并 stages；保留别名/迁移 |
| opt-in UMM path | 唯一默认 path | 更新 docs/tests/DP/TP YAML |

| 回调/字段 | 目标实现 | 说明 |
| --- | --- | --- |
| `start()` | 校验 admission；在 request metadata 写 `sensenova_deadline_monotonic_s`；派发 `generate` | deadline 用模型自己的 key，不再 import `UMM_DEADLINE_METADATA_KEY` |
| `advance()` | 校验图片数；返回 image segments + 一个 `usage` segment | 不再使用 `completion_metadata` |
| `usage()` | input bytes + 输出预留 | 保持保守估计；需要在 #2831 文档中说明 |
| `close()` | 释放 `executions` | 不依赖 close reason；expired/aborted 都可清理 |
| `__init__` | `SenseNovaUMMAdapter(model_path)`，limits 用模型常量/环境变量 | 不再有 `adapter_options`；配置不能逐部署调 limits |

`usage` segment 的形状建议：

```json
{
  "type": "segment",
  "kind": "usage",
  "data": {
    "image_usage": {
      "input_tokens": 15,
      "total_tokens": 15,
      "input_tokens_details": {"text_tokens": 15, "image_tokens": 4096}
    },
    "stage_name": "generate"
  }
}
```

这样做的代价：

- `max_outputs` 要把 usage segment 算进去：`max_outputs >= max_images + 1`。
- `max_retained_bytes` 会计入这个很小的 segment。
- `/generate` 的客户端会多看到一个 `kind="usage"` 的 segment；`/v1/images/*` 过滤它。
- `UMMGenerateChunk.image_usage` 不会自动有值；推荐在 `umm_result_chunk` 里解析 usage
  segment 并设置 `image_usage`（必要时也设置 `stage_name`）。这是 client 的最小改动，已确认
  允许；Images endpoint 和 `/generate` 都能拿到 usage，`image_result_with_disconnect_watch`
  只负责把 image segment 还原成 `images_b64`。

已用 **`origin/main` 的 `UMMController`**（没有 `completion_metadata`/`adapter_options`/
deadline key）模拟验证：adapter 返回 `image + usage` 两个 segment 时，final result 是
`[image, usage]`，没有 `metadata` 字段，context 正常关闭；`max_outputs=2` 通过，
`max_outputs=1` 会报 `UMM request exceeds max_outputs=1`。这确认了“usage segment +
max_outputs+1”在 no-runtime 方案下的可行性，而不只是 branch 混合 controller 上可行。
n=10 的极端情况也验证了：`max_outputs=10` 会失败，`max_outputs=11` 会产出 11 个 segment
（10 张图 + usage），所以默认配置必须从 10 改成 11。

如果 usage segment 不可接受，另一个选择是把 `image_usage` 放在每张图片 segment 的 data 里；
但那会破坏 `InlineImage` 的 strict schema，不推荐。

### 8.3 配置与 native path 的替换

当前 `EntryClass = SenseNovaU1PipelineConfig`（native），`SenseNovaU1UMMPipelineConfig`
是 opt-in。替换 native 后建议：

1. 让 `SenseNovaU1PipelineConfig` 直接继承 `UMMPipelineConfig`，把 orchestrator/output
   stages、`phases={"generate": "generate"}` 和五个 bounds 作为默认值。
2. 保留 `SenseNovaU1UMMPipelineConfig` 作为兼容别名，或删除并更新 YAML。
3. `examples/configs/sensenova_u1_dp2.yaml`、`_tp2.yaml`、`_dp2_tp2.yaml` 仍然只覆盖
   `generate` stage；因为 UMM config 的 generate stage 名字不变，理论上可以复用，但需要
   重新跑 DP/TP 展开测试。注意现有 `test_tp_example_make_config` 断言
   `all(stage.tp_size == 2)`，UMM 默认多了 `orchestrator`/`output`（tp_size=1），这个断言
   会失败；需要改成只检查 `generate@rN`。`test_tp_config.py` 的 `make_config` 还用
   `stages[0]` 当 generate；UMM 默认 `stages[0]` 是 orchestrator，也要改成按 name 查找。
4. `SenseNovaU1PipelineConfig.model_post_init` 需要同时执行 UMM 的 wiring 校验和原 TP/DP
   校验；注意 MRO 和 `super()` 调用顺序。
5. `docs/cookbook/sensenova_u1.md` 的默认启动命令从 native 改成 UMM；`Current Scope` 和
   `Shared UMM Integration Contract` 合并成唯一的 serving contract。
6. 所有 native-only 测试（entry stage、terminal stage、single-stage scheduler）要么迁移到
   UMM，要么删除；不能保留一套“默认是 native、测试也是 native”的假象。

这是一个 breaking change：现有 `sensenova_u1.yaml` 用户会看到三个 stage 和新的 deadline/
admission 行为。需要 migration note 和至少一个 deprecation 周期；如果不想 breaking，可以
先提供 `--variant umm` 默认、native 显式 opt-in，再在下个版本删除 native。

### 8.4 Images / usage / metrics

- **Images endpoint**：`image_result_with_disconnect_watch` 已经是 UMM chunk 的汇聚点；
  在这里从 `result.segments` 找 `kind == "usage"`，把 `data["image_usage"]` 设到
  `result.image_usage`，再走 `build_image_response`。T2I/I2I 都不需要改 controller。
- **I2I 多输出**：endpoint 目前对每个 output 发一个独立请求；每个请求一个 usage segment。
  保持“取第一个 output 的 usage”还是累加，需要明确；现有 native 行为是取第一个。
- **`/generate`**：usage segment 会出现在 segments 里；客户端如果只关心图片，按 kind 过滤。
- **metrics**：`RequestMetrics` 目前只看 `images_b64`/`image_b64`。UMM path 替换 native 后，
  必须在 `RequestMetrics.generate` 里识别 `UMMGenerateChunk.segments` 的 image kind，
  否则 `generated_images_total` 永远不增长。这个改动在 `serve/request_metrics.py`，不在
  UMM controller。另一个标签问题：UMM final chunk 的 `stage_name` 目前是 None，metrics 会
  落到 `unknown`，需要在 result/client 侧补 `generate` stage label，否则 #2728 的 stage
  label 语义不成立。
- **`UMMGenerateChunk.image_usage`**：推荐让 `umm_result_chunk` 解析 usage segment，
  设置 `image_usage` 和 stage label；这是 client 改动，已确认允许。Images endpoint 就不需要
  自己解析，`/generate` 也能拿到 usage。实测当前 `umm_result_chunk` 对 usage segment 的
  `image_usage` 仍是 None，但 segment data 可以被 `parse_image_usage` 正常解析；改动很小。
- **图片 segment 的 `segment_index`**：usage segment 会占一个 index；Images endpoint 过滤后
  仍按图片顺序返回；客户端如果按 segments 顺序展示，需要跳过 usage。

### 8.5 Deadline / TP cancellation / forget

- **Deadline**：`SenseNovaUMMAdapter.start` 写
  `sensenova_deadline_monotonic_s = context.deadline`；`cancellation.py` 读取这个模型 key。
  这样 `pipeline/umm.py` 不需要新增 `UMM_DEADLINE_METADATA_KEY`。
- **TP cancellation**：在 `sensenova_u1/tensor_parallel.py` 或 `stages.py` 定义
  `SenseNovaTensorParallelScheduler(TensorParallelScheduler)`，覆盖 `abort`、
  `consume_if_aborted`、`run_single`，使用模型自己的 `SenseNovaCancelledError`。
  `create_generation_executor` 在 `cooperative_cancellation=True` 时实例化这个子类。
  `scheduling/message.py` 和 `scheduling/tensor_parallel_scheduler.py` 不改。
- **forget**：不改 controller。`expire()` 已经用 `"expired"` 释放；`forget()` 走
  `"aborted"`。`SenseNovaUMMAdapter.close` 不依赖 reason；如果确实需要区分，可以在 close
  里比较 `context.deadline` 和 `time.monotonic()`，但这只影响模型日志，不影响协议。

### 8.6 thinking / interleaved 的后续 Phase

按决策 2，修改 #2831 增加 Phase 4：

- Phase 4 scope：thinking mode、interleaved text/image output。
- 依赖：#2908 的 progress streaming（目前仍是 Proposed）、SGLang #40702（thinking）和
  #35038（interleave）。
- 在 UMM 上，SenseNova 仍是一个 phase 内跑模型自己的 interleaving loop；需要 `progress`
  回调把 text/segment 在 phase 结束前送到客户端。在没有 progress streaming 之前，Phase 4
  只能标为 blocked/planned，不能进验收。
- Phase 4 不要求修改 UMM controller；如果 progress streaming 最终以 runtime addition 落地，
  那是 #2908 的变更，不是 SenseNova 自己的。

### 8.7 风险与取舍

1. **替换 native 是 breaking change。** 三个 stage、deadline、admission bounds、usage segment
   都会改变现有行为；需要 migration note/deprecation。
2. **固定 limits 失去 per-deployment 调参。** 没有 `adapter_options` 后，limits 只能写常量
   或读环境变量；生产上如果要按 GPU 内存调 `max_active`/`max_retained_bytes`，会不方便。
3. **usage segment 是新 kind。** 现有客户端可能只认识 text/image；`/generate` 会多一个
   segment；需要客户端兼容和文档。
4. **metrics 仍需 serve 改动。** 已确认 client/serve 允许最小适配，因此
   `RequestMetrics` 识别 UMM image segments、Images endpoint 提取 usage segment 是必须做的
   工作，不是阻塞；如果不做，UMM path 替换 native 后会丢失 #2831 的 usage/metrics 验收。
5. **#2918/#2919 已 merge，但 controller 改动仍要撤回。** `cd4b310fa` 解决的是文本冲突；
   no-runtime 目标要求把 `pipeline/umm.py` 恢复到 main，adapter/tests 适配 main 的
   `UMMStep.dispatches`、`expected` dict、三元组 `apply()` 和 `RoutedPayloads`。
6. **硬件验收仍是最大缺口。** 替换 native 后，原 native 的硬件证据不能直接复用；Phase 3
   必须按 UMM path 重跑。

### 8.8 实施顺序（分析结论，不是代码）

1. merge 已在 `cd4b310fa` 完成；删除 branch 对 `pipeline/umm.py`、
   `scheduling/message.py`、`scheduling/tensor_parallel_scheduler.py` 的改动，恢复到 main。
2. 把 deadline 移到模型 metadata；把 TP cancellation 改成 SenseNova 子类；删除
   `completion_metadata`，改用 usage segment。
3. 把 `SenseNovaU1PipelineConfig` 改成 UMM 默认；迁移 DP/TP YAML、docs、tests。
4. 在 `umm_result_chunk`（或 Images endpoint）提取 usage segment；在 `RequestMetrics`
   识别 UMM image segments 和 stage label。
5. 跑 CPU/Gloo 回归，再按 Phase 3 矩阵跑 GPU/NPU。
6. 更新 #2831（Phase 4 + UMM 默认）和 #2908（SenseNova row + 不改 runtime 的说明）。

### 8.9 决策后的 RFC 修改点

- **#2831**：把 UMM path 写成默认 serving contract；增加 Phase 4（thinking/interleaved）；
  Phase 3 验收矩阵按 UMM path 重写；明确 usage segment 和 fixed limits。
- **#2908**：SenseNova row 改为 “T2I/I2I on UMM controller；thinking/interleaved Phase 4，
  blocked on progress streaming”；明确不新增 runtime additions；说明 usage segment 是
  model/result 层约定，不是 controller 契约。
- **#2723/#2727**：仍然需要，因为 Images usage 和 metrics 的 shared foundation 还没进 main；
  但 UMM path 的 usage segment 解析和 metrics 识别可以在 SenseNova 集成 PR 里做。

可直接贴到 issue 的英文草案见
[sensenova_rfc_update_draft.md](sensenova_rfc_update_draft.md)。

### 8.10 文件级改动地图

这里的“动作”是实现阶段的建议，不是本轮已执行的改动。

| 文件/目录 | 动作 | 备注 |
| --- | --- | --- |
| `sglang_omni/pipeline/umm.py` | revert 到 main（#2918）版本 | 不保留 `completion_metadata`、`adapter_options`、deadline key、`forget` 改动 |
| `sglang_omni/scheduling/message.py` | revert `RequestCancelledError` | 用模型自己的 `SenseNovaCancelledError` |
| `sglang_omni/scheduling/tensor_parallel_scheduler.py` | revert `cooperative_cancellation`/abort 改动 | 用 `SenseNovaTensorParallelScheduler` 子类 |
| `sglang_omni/client/client.py` | 去掉 `metadata` 解析；在 `umm_result_chunk` 解析 usage segment | 设置 `image_usage`、可选 `stage_name`；这是 client 最小适配 |
| `sglang_omni/client/types.py` | 评估 `typing_extensions.TypedDict` 改动是否必要 | 与 SenseNova 无关则 revert |
| `sglang_omni/client/image_media.py` | 保留 | `InlineImage`/`parse_image_usage` 仍需使用 |
| `sglang_omni/serve/openai_api.py` | 保留 UMM chunk 转换和错误映射；确认 usage 由 client 解析 | `image_result_with_disconnect_watch` 只还原 image segments |
| `sglang_omni/serve/request_metrics.py` | 识别 `UMMGenerateChunk.segments` 的 image kind | 否则 `generated_images_total` 不增长；同时设置 stage label |
| `sglang_omni/models/sensenova_u1/umm.py` | 删除 `completion_metadata`；追加 usage segment；固定 limits；写模型 deadline key | 不 import `pipeline.umm` 的 deadline key |
| `sglang_omni/models/sensenova_u1/umm_config.py` | 删除 `adapter_options`；`max_outputs = max_images + 1`；默认 limits | 与 `umm.py` 的常量保持一致 |
| `sglang_omni/models/sensenova_u1/config.py` | 让 `SenseNovaU1PipelineConfig` 成为 UMM 默认；保留别名/迁移 | native `entry_stage="generate"`/单 stage 要删除 |
| `sglang_omni/models/sensenova_u1/cancellation.py` | 用模型 deadline key；定义 `SenseNovaCancelledError` | 不再 import `UMM_DEADLINE_METADATA_KEY` |
| `sglang_omni/models/sensenova_u1/tensor_parallel.py` 或 `stages.py` | 增加 `SenseNovaTensorParallelScheduler` | 覆盖 `abort/consume_if_aborted/run_single` |
| `sglang_omni/models/sensenova_u1/stages.py` | 使用模型 scheduler；保留 `cooperative_cancellation` 选择 | 不修改共享 `TensorParallelScheduler` |
| `sglang_omni/models/sensenova_u1/admission.py` | 定义默认 UMM limits 常量 | `SenseNovaU1UMMPipelineConfig` 和 adapter 共用 |
| `docs/cookbook/sensenova_u1.md` | native 启动改成 UMM；usage segment/fixed limits/Phase 4 | 删除“opt-in”表述 |
| `docs/metrics.md` | 说明 UMM image segment 计数和 stage label | 与 `request_metrics.py` 同步 |
| `examples/configs/sensenova_u1_umm.yaml` | 变成默认/别名配置 | `config_cls` 与 `SenseNovaU1PipelineConfig` 对齐 |
| `examples/configs/sensenova_u1_dp2*.yaml` | 保持 stage 覆盖，验证 UMM 展开 | 现有测试的 `stages[0]`/`all(tp_size==2)` 要改 |
| `tests/unit_test/sensenova/test_umm.py` | 删 metadata/deadline key 测试；加 usage segment/max_outputs 测试 | 依赖 main 的 `UMMStep` |
| `tests/unit_test/serve/test_images_usage_api.py` | 用 usage segment 验证 Images usage | 真实 controller + fake coordinator 仍不够 |
| `tests/unit_test/sensenova/test_t2i_pipeline.py` | 迁移 native entry/stage 断言到 UMM | API 行为保持不变 |
| `tests/unit_test/sensenova/test_tp_config.py` | 按 stage name 取 generate；只检查 generate@rN | 见 8.3 |
| `tests/unit_test/serve/test_umm_serving.py` | 改成 SenseNova UMM pipeline 的全链路测试 | 补 route/output/stream |
| `.github/workflows/test.yaml` 或新 workflow | 给硬件测试加 `accelerator` marker；新增 SenseNova lane | GPU nightly/PR + NPU |

### 8.11 测试迁移地图

| 测试 | 迁移动作 |
| --- | --- |
| `test_umm.py` | `completion_metadata` 改成 usage segment；`UMM_DEADLINE_METADATA_KEY` 改成模型 key；`max_outputs` 10→11 |
| `tests/unit_test/pipeline/test_umm.py` | 删除 merge 新增的 `test_parallel_completion_metadata_and_deadline`，或回退到 main 的版本 |
| `tests/unit_test/fixtures/sensenova_umm.py` | `adapter_options` 改成 `SenseNovaUMMAdapter("unused")`；limits 与模型默认常量一致 |
| `test_images_usage_api.py` | fake coordinator 的 UMM result 带 usage segment；断言 `image_usage` 仍到 Images response |
| `test_t2i_pipeline.py` | `entry_stage`/stage list 改成 UMM；API 层行为不变 |
| `test_tp_config.py` | `stages[0]` 改成 `stage_named("generate")`；`all(tp_size==2)` 只检查 generate |
| `test_tensor_parallel.py`/`test_native_tp.py` | 改用 `SenseNovaTensorParallelScheduler`；覆盖取消后复用 |
| `test_npu_adaptation.py` | config 迁移；accelerator 分支加 marker |
| `test_fused_rope.py` | CUDA 用例加 `@pytest.mark.accelerator` |
| `test_images_usage_api.py`（metrics 部分） | 增加 UMM metrics 计数断言（如果 metrics 改动在本 PR） |
| 新增 `test_umm_serving_sensenova.py` | 真 `Stage`/coordinator + route/output/stream + usage segment |
| CI | `unit-test-accelerator` 能选中 SenseNova 硬件用例；新增 GPU/NPU lane |

这两张表把“分析结论”落到文件和测试层面；下一步实现时可以直接按表执行，不需要再重新推导。

### 8.12 no-runtime 的符号引用清单

merge 后这些符号已经分布在 pipeline、SenseNova、tests、docs 中；撤回时要一起处理，不能只改
`pipeline/umm.py`。

| 符号 | 引用数（grep 近似，含本报告） | 需要处理的文件 |
| --- | --- | --- |
| `completion_metadata` | 41 | `pipeline/umm.py`、`models/sensenova_u1/umm.py`、`tests/unit_test/pipeline/test_umm.py`、`tests/unit_test/sensenova/test_umm.py`、`docs/cookbook/sensenova_u1.md` |
| `adapter_options` | 24 | `pipeline/umm.py`、`models/sensenova_u1/umm_config.py`、`tests/unit_test/fixtures/sensenova_umm.py`、`docs/cookbook/sensenova_u1.md` |
| `UMM_DEADLINE_METADATA_KEY` | 20 | `pipeline/umm.py`、`models/sensenova_u1/cancellation.py`、`tests/unit_test/pipeline/test_umm.py`、`tests/unit_test/sensenova/test_umm.py` |
| `RequestCancelledError` | 14 | `scheduling/message.py`、`scheduling/tensor_parallel_scheduler.py`、`models/sensenova_u1/cancellation.py`、`tests/unit_test/sensenova/test_tensor_parallel.py`、`tests/unit_test/sensenova/test_umm.py` |
| `cooperative_cancellation` | 23 | `scheduling/tensor_parallel_scheduler.py`、`models/sensenova_u1/config.py`、`models/sensenova_u1/umm_config.py`、`models/sensenova_u1/stages.py`、对应 tests |

其中两个容易漏掉的点：

- `tests/unit_test/pipeline/test_umm.py` 里 merge 新增了
  `test_parallel_completion_metadata_and_deadline`；它引用了 `completion_metadata` 和
  `UMM_DEADLINE_METADATA_KEY`，no-runtime 后必须删除或回退到 main 的测试。
- `tests/unit_test/fixtures/sensenova_umm.py` 用 `controller_options["adapter_options"]`
  构造 adapter；固定 limits 后要改成 `SenseNovaUMMAdapter("unused")`，并让 fixture 的
  controller limits 与模型默认常量一致。

`cooperative_cancellation` 在 `tests/unit_test/cosmos3/test_native_placement.py` 里也出现，
但那是 native generation 的 plugin cancellation，不是这次 SenseNova 的 flag，不能一起删。
