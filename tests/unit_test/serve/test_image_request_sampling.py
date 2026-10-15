# SPDX-License-Identifier: Apache-2.0
import pytest

from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling
from sglang_omni.serve.image_request import resolve_image_generation_params
from sglang_omni.serve.protocol import ImageGenerationRequest


@pytest.mark.parametrize(
    "fields",
    [
        {"think_mode": True},
        {"extra_body": {"think_mode": True}},
        {"extra_json": '{"think_mode": true}'},
        {"extra_args": {"think_mode": True}},
        {"extra_params": {"think_mode": True}},
    ],
)
def test_think_mode_is_rejected_before_inference(fields):
    request = ImageGenerationRequest(prompt="cat", **fields)
    with pytest.raises(ValueError, match="think_mode is not supported"):
        resolve_image_generation_params(request)


def test_explicit_false_think_mode_overrides_nested_true():
    request = ImageGenerationRequest(
        prompt="cat", think_mode=False, extra_body={"think_mode": True}
    )
    assert resolve_image_generation_params(request)["think_mode"] is False


@pytest.mark.parametrize("n, expected", [(None, 1), (0, 1), (-3, 1), (1, 1), (5, 5), (11, 10)])
def test_n_is_clamped(n, expected):
    params = resolve_image_generation_params(ImageGenerationRequest(prompt="cat", n=n))
    assert params["n"] == expected


def test_explicit_null_uses_model_defaults():
    request = ImageGenerationRequest(
        prompt="cat", size=None, width=None, height=None, seed=None,
        num_inference_steps=None, guidance_scale=None, n=None, response_format=None,
    )
    assert resolve_image_generation_params(request) == {
        "n": 1, "width": 2048, "height": 2048, "seed": 42,
        "num_inference_steps": 50, "guidance_scale": 4.0,
    }


def test_null_size_with_one_dimension():
    params = resolve_image_generation_params(
        ImageGenerationRequest(prompt="cat", size=None, width="512")
    )
    assert (params["width"], params["height"]) == (512, 2048)


@pytest.mark.parametrize("seed, n, expected", [(7, 3, [7, 8, 9]), ([0, 9, 4], 3, [0, 9, 4]), ([0], 1, [0])])
def test_seed_order(seed, n, expected):
    params = resolve_image_generation_params(ImageGenerationRequest(prompt="cat", seed=seed, n=n))
    assert SenseNovaU1Sampling.from_params(params).output_seeds == expected


@pytest.mark.parametrize("seed,n", [([], 1), ([1, 2], 1), ([1], 2), ([-1], 1), (-1, 1)])
def test_invalid_seed_is_rejected_before_inference(seed, n):
    request = ImageGenerationRequest(prompt="cat", seed=seed, n=n)
    with pytest.raises(ValueError, match="seed"):
        resolve_image_generation_params(request)


def test_null_fields_do_not_override_explicit_zero_values():
    params = resolve_image_generation_params(
        ImageGenerationRequest(prompt="cat", seed=0, guidance_scale=0, num_inference_steps=1)
    )
    assert params["seed"] == params["guidance_scale"] == 0


@pytest.mark.parametrize("fields", [{"num_inference_steps": 0}, {"guidance_scale": -1}])
def test_invalid_sampling_is_rejected(fields):
    with pytest.raises(ValueError):
        resolve_image_generation_params(ImageGenerationRequest(prompt="cat", **fields))
