# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from sglang_omni.serve.protocol import ImageGenerationRequest


@pytest.mark.parametrize("dimensions", [(1024, 512), ("1024", "512")])
def test_dimensions_accept_integers_and_numeric_strings(dimensions):
    request = ImageGenerationRequest(
        prompt="a cat", size="2048x2048", width=dimensions[0], height=dimensions[1]
    )
    assert (request.width, request.height) == (1024, 512)


@pytest.mark.parametrize("fields", [{"width": "1024"}, {"height": "512"}])
def test_dimensions_can_be_provided_independently(fields):
    request = ImageGenerationRequest(prompt="a cat", **fields)
    assert request.width == (1024 if "width" in fields else None)
    assert request.height == (512 if "height" in fields else None)
    assert request.size == "2048x2048"


def test_size_and_default_remain_supported():
    assert ImageGenerationRequest(prompt="a cat").size == "2048x2048"
    assert ImageGenerationRequest(prompt="a cat", size="1024x512").size == "1024x512"


@pytest.mark.parametrize(
    "fields",
    [
        {"width": 0, "height": 512},
        {"width": "invalid", "height": 512},
        {"width": 1024, "height": -1},
    ],
)
def test_invalid_dimensions_are_rejected(fields):
    with pytest.raises(ValidationError):
        ImageGenerationRequest(prompt="a cat", **fields)


def test_extra_fields_are_accepted_and_preserved():
    extras = {"num_frames": 1, "bot_task": "text_to_image", "extra_body": {"seed": 7}}
    request = ImageGenerationRequest(prompt="a cat", **extras)
    assert request.model_extra == extras
    assert request.seed is None
