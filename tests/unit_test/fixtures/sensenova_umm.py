# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for SenseNova controller and serving contracts."""

import base64
import io

import pytest
from PIL import Image

from sglang_omni.models.sensenova_u1.umm import SenseNovaUMMAdapter
from sglang_omni.pipeline.umm import UMMController, UMMLimits


@pytest.fixture
def encoded_image() -> str:
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 32), color="red").save(image_buffer, format="PNG")
    return base64.b64encode(image_buffer.getvalue()).decode("ascii")


@pytest.fixture
def controller() -> UMMController:
    return UMMController(
        SenseNovaUMMAdapter("unused"),
        limits=UMMLimits(
            max_dispatches=1,
            max_outputs=10,
            max_active=2,
            max_retained_bytes=1024 * 1024,
            timeout_s=60,
        ),
        phases={"generate": "generate"},
        output_stage="output",
    )
