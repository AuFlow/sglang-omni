# SPDX-License-Identifier: Apache-2.0
import base64
import io
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.serve.openai_api import create_app


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize("fail_after_image", [False, True])
def test_images_usage_and_complete_result_consumption(
    editing: bool, outputs: int, fail_after_image: bool
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    class ImageClient(Client):
        def __init__(self) -> None:
            super().__init__(None)
            self.completed: int = 0

        async def generate(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncIterator[GenerateChunk]:
            if not editing:
                assert request.extra_params == {
                    "width": 64,
                    "height": 32,
                    "num_inference_steps": 50,
                    "guidance_scale": 4.0,
                    "seed": 42,
                    "n": outputs,
                }
            yield self._default_result_builder(
                request_id,
                {
                    "modality": "image",
                    "image_b64": encoded_image,
                    "images_b64": [encoded_image] * request.extra_params["n"],
                    "image_usage": {
                        "input_tokens": 3,
                        "total_tokens": 3,
                        "input_tokens_details": {
                            "text_tokens": 3,
                            "image_tokens": 16 if editing else 0,
                        },
                    },
                    "finish_reason": "stop",
                },
            )
            if fail_after_image:
                raise RuntimeError("Backend completion failed")
            else:
                self.completed += 1

    backend = ImageClient()
    with TestClient(
        create_app(backend, model_name="test", architectures=["NEOChatModel"])
    ) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data={"prompt": "a red mug", "size": "64x32", "n": outputs},
                files={"image": ("reference.png", buffer.getvalue(), "image/png")},
            )
        else:
            response = client.post(
                "/v1/images/generations",
                json={
                    "prompt": "a red mug",
                    "size": "64x32",
                    "n": outputs,
                    "return_stage_metrics": True,
                    "bot_task": "think",
                    "sys_type": "en_unified",
                    "system_prompt": "custom prompt",
                },
            )
    assert response.status_code == (500 if fail_after_image else 200)
    if fail_after_image:
        assert backend.completed == 0
    else:
        result = response.json()
        assert backend.completed == (outputs if editing else 1)
        assert len(result["data"]) == outputs
        assert all(image["output_format"] == "png" for image in result["data"])
        assert result["usage"]["input_tokens"] == result["usage"]["total_tokens"] == 3
        assert result["usage"]["input_tokens_details"] == {
            "text_tokens": 3,
            "image_tokens": 16 if editing else 0,
        }
        assert result["usage"]["image_count"] == outputs
        assert (result["usage"]["width"], result["usage"]["height"]) == (64, 32)


@pytest.mark.parametrize(
    "fields",
    [
        {"n": 0},
        {"n": 11},
        {"seed": -1},
        {"num_inference_steps": 0},
        {"guidance_scale": -1},
        {"response_format": "url"},
    ],
)
def test_unknown_extensions_do_not_bypass_known_field_validation(fields):
    class UnusedClient(Client):
        def __init__(self):
            super().__init__(None)

        async def generate(self, *args, **kwargs):
            raise AssertionError("Invalid requests must not reach inference")

    with TestClient(
        create_app(UnusedClient(), model_name="test", architectures=["NEOChatModel"])
    ) as client:
        response = client.post(
            "/v1/images/generations",
            json={"prompt": "a red mug", "return_stage_metrics": True, **fields},
        )
    assert response.status_code == 422
