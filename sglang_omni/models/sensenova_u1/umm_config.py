# SPDX-License-Identifier: Apache-2.0
"""Opt-in shared UMM orchestration for SenseNova T2I and I2I."""

from sglang_omni.config.schema import FactoryArgs, StageConfig
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
                max_dispatches=1,
                max_outputs=10,
                max_active=8,
                max_retained_bytes=128 * 1024 * 1024,
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
            factory=SenseNovaGenerationFactoryArgs(enable_cache_dit=False),
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
