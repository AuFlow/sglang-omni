# SPDX-License-Identifier: Apache-2.0
"""Single-phase SenseNova generation on the shared UMM controller."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from sglang_omni.client.image_media import InlineImage, parse_image_usage
from sglang_omni.client.types import ImageUsageDict
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)


class SenseNovaImageResult(BaseModel):
    model_config = ConfigDict(strict=True)

    modality: Literal["image"]
    image_b64: str | None = None
    images_b64: list[str] | None = None
    image_usage: ImageUsageDict | None = None


@dataclass(kw_only=True)
class SenseNovaExecution:
    expected_images: int
    retained_bytes: int


class SenseNovaUMMAdapter:
    def __init__(self, model_path: str) -> None:
        """The shared factory supplies model_path; the generation stage loads it."""
        self.executions: dict[str, SenseNovaExecution] = {}

    def start(self, context: UMMContext) -> UMMStep:
        try:
            inputs = TypeAdapter(JsonValue).validate_python(context.request.inputs)
            parameters = TypeAdapter(dict[str, JsonValue]).validate_python(
                context.request.params
            )
            if isinstance(inputs, str) and inputs.strip():
                options = SenseNovaU1Sampling.from_params(parameters)
                expected_images = options.n
            elif (
                isinstance(inputs, dict)
                and inputs.get("task") == "image_edit"
                and isinstance(inputs.get("prompt"), str)
                and inputs["prompt"].strip()
            ):
                SenseNovaU1ImageEditSampling.from_params(parameters)
                references = inputs.get("image_b64")
                if not (
                    isinstance(references, str)
                    and references
                    or isinstance(references, list)
                    and references
                    and all(
                        isinstance(reference, str) and reference
                        for reference in references
                    )
                ):
                    raise ValueError("Image editing requires encoded reference images")
                else:
                    expected_images = 1
            else:
                raise ValueError(
                    "SenseNova UMM supports a text prompt or an image_edit request"
                )
        except ValueError as error:
            raise UMMRequestError(str(error)) from error
        self.executions[context.execution_id] = SenseNovaExecution(
            expected_images=expected_images,
            retained_bytes=len(
                json.dumps([inputs, parameters], ensure_ascii=False).encode("utf-8")
            ),
        )
        return UMMStep(phase="generate", request=context.request)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        if phase != "generate":
            raise ValueError(f"Unexpected SenseNova phase: {phase}")
        else:
            generation = SenseNovaImageResult.model_validate(result)
        images = generation.images_b64
        if images is None:
            images = [generation.image_b64] if generation.image_b64 is not None else []
        else:
            pass
        execution = self.executions[context.execution_id]
        if len(images) != execution.expected_images:
            raise ValueError("SenseNova output count does not match the request")
        else:
            pass
        return UMMStep(
            outputs=tuple(
                UMMOutput("image", InlineImage.from_png(encoded_image).model_dump())
                for encoded_image in images
            ),
            image_usage=parse_image_usage(generation.image_usage),
        )

    def usage(self, context: UMMContext) -> int:
        execution = self.executions.get(context.execution_id)
        return execution.retained_bytes if execution is not None else 0

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.executions.pop(context.execution_id, None)
