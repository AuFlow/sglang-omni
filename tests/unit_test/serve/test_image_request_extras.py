# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling
from sglang_omni.serve.image_request import (
    EXTRA_CONTAINERS,
    image_request_extra_value,
)
from sglang_omni.serve.protocol import ImageGenerationRequest


@pytest.mark.parametrize("container", EXTRA_CONTAINERS)
@pytest.mark.parametrize("as_json", [False, True])
def test_cache_extensions_are_resolved_and_validated(container, as_json):
    values = {
        "enable_cache_dit": True,
        "cache_dit_params": {"Fn_compute_blocks": 2},
    }
    request = ImageGenerationRequest(
        prompt="a cat", **{container: json.dumps(values) if as_json else values}
    )
    resolved = {name: image_request_extra_value(request, name) for name in values}
    options = SenseNovaU1Sampling.from_params(resolved)
    assert options.enable_cache_dit is True
    assert options.cache_dit_params == {"Fn_compute_blocks": 2}


def test_top_level_false_overrides_nested_true():
    request = ImageGenerationRequest(
        prompt="a cat", enable_cache_dit=False, extra_body={"enable_cache_dit": True}
    )
    assert image_request_extra_value(request, "enable_cache_dit") is False


@pytest.mark.parametrize("first", range(len(EXTRA_CONTAINERS)))
def test_container_precedence_and_null_fallback(first):
    fields = {
        name: {"cache_dit_params": None if i < first else {"Fn_compute_blocks": i + 1}}
        for i, name in enumerate(EXTRA_CONTAINERS)
    }
    request = ImageGenerationRequest(prompt="a cat", **fields)
    assert image_request_extra_value(request, "cache_dit_params") == {
        "Fn_compute_blocks": first + 1
    }


def test_nested_extra_params_and_direct_precedence():
    request = ImageGenerationRequest(
        prompt="a cat",
        extra_body={
            "enable_cache_dit": False,
            "extra_params": '{"enable_cache_dit": true, "cache_dit_params": {"Fn_compute_blocks": 2}}',
        },
    )
    assert image_request_extra_value(request, "enable_cache_dit") is False
    assert image_request_extra_value(request, "cache_dit_params") == {"Fn_compute_blocks": 2}


@pytest.mark.parametrize("malformed", ["{", "[]", [], 1, None])
def test_malformed_container_is_skipped(malformed):
    request = ImageGenerationRequest(
        prompt="a cat", extra_body=malformed, extra_params={"enable_cache_dit": True}
    )
    assert image_request_extra_value(request, "enable_cache_dit") is True


@pytest.mark.parametrize("container", [None, "extra_body"])
def test_invalid_resolved_option_is_rejected(container):
    fields = {"enable_cache_dit": "invalid"}
    request = ImageGenerationRequest(
        prompt="a cat", **({container: fields} if container else fields)
    )
    with pytest.raises(ValueError, match="enable_cache_dit must be a boolean"):
        SenseNovaU1Sampling.from_params(
            {"enable_cache_dit": image_request_extra_value(request, "enable_cache_dit")}
        )


def test_ordinary_fields_are_not_promoted_from_extras():
    request = ImageGenerationRequest(prompt="a cat", extra_body={"seed": 7, "width": 512})
    assert request.seed is None
    assert request.width is None
