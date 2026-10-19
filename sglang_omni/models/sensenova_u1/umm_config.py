# SPDX-License-Identifier: Apache-2.0
"""Opt-in shared UMM orchestration for SenseNova T2I and I2I."""

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.schema import FactoryArgs, StageConfig
from sglang_omni.models.sensenova_u1.admission import SenseNovaAdmissionLimits
from sglang_omni.models.sensenova_u1.config import (
    SenseNovaGenerationFactoryArgs,
    SenseNovaGenerationStageConfig,
    SenseNovaU1PipelineConfig,
)
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)


class SenseNovaU1UMMPipelineConfig(UMMPipelineConfig, SenseNovaU1PipelineConfig):
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = [
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                adapter="sglang_omni.models.sensenova_u1.umm.SenseNovaUMMAdapter",
                phases={"generate": "generate"},
                adapter_options={
                    "limits": {
                        "max_images": 10,
                        "max_output_pixels": 10 * 2048 * 2048,
                        "max_reference_pixels": 32 * 1024 * 1024,
                        "max_reference_images": 8,
                        "max_steps": 100,
                        "max_input_bytes": 32 * 1024 * 1024,
                    }
                },
                max_dispatches=1,
                max_outputs=10,
                max_active=8,
                max_retained_bytes=256 * 1024 * 1024,
                timeout_s=600.0,
            ),
            next=["generate", "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        SenseNovaGenerationStageConfig(
            name="generate",
            process="sensenova_generate",
            factory_path="sglang_omni.models.sensenova_u1.stages.create_generation_executor",
            factory=SenseNovaGenerationFactoryArgs(
                enable_cache_dit=False, cooperative_cancellation=True
            ),
            gpu=0,
            next="orchestrator",
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: object = None, /) -> None:
        super().model_post_init(__context)
        controller = resolve_stage_typed_kwargs(self.stage_named("orchestrator"))
        limits = SenseNovaAdmissionLimits.model_validate(
            controller["adapter_options"]["limits"]
        )
        if controller["max_outputs"] < limits.max_images:
            raise ValueError("orchestrator.max_outputs must cover adapter max_images")
        elif not self.stage_named("generate").factory.cooperative_cancellation:
            raise ValueError("SenseNova UMM requires cooperative_cancellation")
        else:
            pass
