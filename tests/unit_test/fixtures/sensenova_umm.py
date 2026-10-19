# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for SenseNova controller and serving contracts."""

import base64
import io

import pytest
from PIL import Image

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.sensenova_u1.umm import SenseNovaUMMAdapter
from sglang_omni.models.sensenova_u1.umm_config import SenseNovaU1UMMPipelineConfig
from sglang_omni.pipeline.umm import UMMController, UMMLimits


@pytest.fixture
def encoded_images() -> list[str]:
    images = []
    for color in ("red", "green", "blue"):
        image_buffer = io.BytesIO()
        Image.new("RGB", (64, 32), color=color).save(image_buffer, format="PNG")
        images.append(base64.b64encode(image_buffer.getvalue()).decode("ascii"))
    return images


@pytest.fixture
def encoded_image(encoded_images: list[str]) -> str:
    return encoded_images[0]


@pytest.fixture
def controller() -> UMMController:
    configuration = SenseNovaU1UMMPipelineConfig(model_path="unused")
    controller_options = resolve_stage_typed_kwargs(
        configuration.stage_named("orchestrator")
    )
    return UMMController(
        SenseNovaUMMAdapter("unused", **controller_options["adapter_options"]),
        limits=UMMLimits(
            max_dispatches=1,
            max_outputs=10,
            max_active=2,
            max_retained_bytes=256 * 1024 * 1024,
            timeout_s=60,
        ),
        phases={"generate": "generate"},
        output_stage="output",
    )
