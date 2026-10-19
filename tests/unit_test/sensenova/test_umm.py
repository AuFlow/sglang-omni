# SPDX-License-Identifier: Apache-2.0
"""SenseNova phase, output, and Images API contracts on the real controller."""

from dataclasses import replace

import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import UMMGenerateChunk
from sglang_omni.models.sensenova_u1.cancellation import (
    GenerationScope,
    generation_checkpoint,
)
from sglang_omni.models.sensenova_u1.umm_config import SenseNovaU1UMMPipelineConfig
from sglang_omni.pipeline.umm import (
    UMM_DEADLINE_METADATA_KEY,
    UMMController,
    UMMRequestError,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import RequestCancelledError

pytest_plugins = ["tests.unit_test.fixtures.sensenova_umm"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("image_count", [1, 3])
def test_outputs_and_usage_survive_controller_completion(
    controller: UMMController, encoded_image: str, streaming: bool, image_count: int
) -> None:
    request = OmniRequest("a red mug", params={"n": image_count, "stream": streaming})
    phase = controller.transition(StagePayload("request", request, None))
    assert phase is not None
    assert phase.request.params["stream"] is False
    phase.data = {
        "modality": "image",
        "images_b64": [encoded_image] * image_count,
        "image_usage": {"input_tokens": 3, "total_tokens": 3},
    }
    result = controller.transition(phase)
    assert result is not None
    completion = Client.default_result_builder("request", result.data)
    assert isinstance(completion, UMMGenerateChunk)
    assert completion.image_usage == {"input_tokens": 3, "total_tokens": 3}
    assert completion.segments is not None
    assert [segment.segment_index for segment in completion.segments] == list(
        range(image_count)
    )
    if streaming:
        segments = [controller.outbox.get_nowait().data for _ in range(image_count)]
        assert all(segment.data is None for segment in completion.segments)
        assert [
            InlineImage.model_validate(segment["data"]).encoded_image()
            for segment in segments
        ] == [encoded_image] * image_count
    else:
        assert [
            InlineImage.model_validate(segment.data).encoded_image()
            for segment in completion.segments
        ] == [encoded_image] * image_count
    assert not controller.contexts
    assert not controller.adapter.executions


def test_bad_result_releases_adapter_state(
    controller: UMMController, encoded_image: str
) -> None:
    phase = controller.transition(
        StagePayload("request", OmniRequest("prompt", {"n": 2}), None)
    )
    assert phase is not None
    phase.data = {"modality": "image", "image_b64": encoded_image}
    with pytest.raises(ValueError, match="output count"):
        controller.transition(phase)
    assert not controller.contexts
    assert not controller.adapter.executions


def test_usage_bytes_are_part_of_controller_bound(
    controller: UMMController, encoded_image: str
) -> None:
    phase = controller.transition(StagePayload("request", OmniRequest("prompt"), None))
    assert phase is not None
    controller.limits = replace(controller.limits, max_retained_bytes=1)
    phase.data = {
        "modality": "image",
        "image_b64": encoded_image,
        "image_usage": {"total_tokens": 3},
    }
    with pytest.raises(RuntimeError, match="max_retained_bytes"):
        controller.transition(phase)
    assert controller.outbox.empty()
    assert not controller.adapter.executions


@pytest.mark.parametrize("parameters", [{"think_mode": True}, {"n": 11}])
def test_invalid_request_never_dispatches(
    controller: UMMController, parameters: dict[str, int | bool]
) -> None:
    with pytest.raises(UMMRequestError):
        controller.transition(
            StagePayload("request", OmniRequest("prompt", parameters), None)
        )
    assert not controller.contexts
    assert not controller.adapter.executions


def test_umm_config_retains_tp_constraints() -> None:
    configuration = SenseNovaU1UMMPipelineConfig(model_path="unused")
    assert configuration.resolved_entry_stage == "orchestrator"
    assert configuration.stage_named("generate").next == "orchestrator"
    assert configuration.stage_named("output").terminal
    configuration.stage_named("generate").tp_size = 3
    configuration.stage_named("generate").gpu = [0, 1, 2]
    with pytest.raises(ValueError, match="tp_size"):
        SenseNovaU1UMMPipelineConfig.model_validate(configuration.model_dump())


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"width": 65536, "height": 65536}, "max_output_pixels"),
        ({"num_inference_steps": 101}, "max_steps"),
    ],
)
def test_admission_limits_reject_before_generation(
    controller: UMMController, parameters: dict[str, int], message: str
) -> None:
    with pytest.raises(UMMRequestError, match=message):
        controller.transition(
            StagePayload("request", OmniRequest("prompt", parameters), None)
        )
    assert not controller.contexts
    assert controller.outbox.empty()


def test_output_reservation_rejects_before_generation(
    controller: UMMController,
) -> None:
    controller.limits = replace(controller.limits, max_retained_bytes=1024)
    with pytest.raises(UMMRequestError, match="max_retained_bytes"):
        controller.transition(StagePayload("request", OmniRequest("prompt"), None))
    assert not controller.contexts
    assert controller.outbox.empty()


def test_batch_cancellation_preserves_live_members_and_clears_scope() -> None:
    cancelled_requests = {"first"}
    payloads = [
        StagePayload(request_id, OmniRequest("prompt"), None)
        for request_id in ("first", "second")
    ]
    with GenerationScope(
        payloads, cancelled_requests.__contains__, torch.device("cpu"), None
    ):
        generation_checkpoint()
        cancelled_requests.add("second")
        with pytest.raises(RequestCancelledError):
            generation_checkpoint()
    generation_checkpoint()
    with GenerationScope(
        [StagePayload("next", OmniRequest("prompt"), None)],
        cancelled_requests.__contains__,
        torch.device("cpu"),
        None,
    ):
        generation_checkpoint()


def test_expired_phase_stops_at_checkpoint() -> None:
    payload = StagePayload(
        "expired",
        OmniRequest("prompt", metadata={UMM_DEADLINE_METADATA_KEY: 0.0}),
        None,
    )
    with GenerationScope(
        [payload], lambda request_id: False, torch.device("cpu"), None
    ):
        with pytest.raises(RequestCancelledError):
            generation_checkpoint()
