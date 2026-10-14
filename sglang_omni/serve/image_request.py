# SPDX-License-Identifier: Apache-2.0
"""SGLang-compatible transport helpers for image extension parameters."""

import json
from typing import Any

from pydantic import BaseModel


EXTRA_CONTAINERS = ("extra_body", "extra_json", "extra_args", "extra_params")


def _flatten_extra_params(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    extra_params = payload.pop("extra_params", None)
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params)
        except (ValueError, TypeError):
            extra_params = None
    if isinstance(extra_params, dict):
        for key, value in extra_params.items():
            payload.setdefault(key, value)
    return payload


def _parse_container(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return _flatten_extra_params(value) if isinstance(value, dict) else {}


def image_request_extra_value(request: BaseModel, field_name: str) -> Any:
    """Read an extension without promoting ordinary protocol fields."""
    direct_value = getattr(request, field_name, None)
    if direct_value is not None:
        return direct_value
    extra = request.model_extra or {}
    direct = _flatten_extra_params(
        {key: value for key, value in extra.items() if key not in EXTRA_CONTAINERS}
    )
    if direct.get(field_name) is not None:
        return direct[field_name]
    for name in EXTRA_CONTAINERS:
        nested = _parse_container(extra.get(name))
        if nested.get(field_name) is not None:
            return nested[field_name]
    return None


def resolve_image_generation_params(request: BaseModel) -> dict[str, Any]:
    """Resolve nullable protocol fields before model validation."""
    from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling

    params = {"n": max(1, min(int(request.n or 1), 10))}
    if request.size:
        try:
            width, height = request.size.split("x")
            params.update(width=int(width), height=int(height))
        except ValueError as exc:
            raise ValueError("size must be formatted as positive WIDTHxHEIGHT") from exc
        if params["width"] <= 0 or params["height"] <= 0:
            raise ValueError("size must be formatted as positive WIDTHxHEIGHT")
    for name in ("width", "height", "seed", "num_inference_steps", "guidance_scale"):
        value = getattr(request, name)
        if value is not None:
            params[name] = value
    for name in ("enable_cache_dit", "cache_dit_params"):
        value = image_request_extra_value(request, name)
        if value is not None:
            params[name] = value
    options = SenseNovaU1Sampling.from_params(params)
    for name in ("width", "height", "seed", "num_inference_steps", "guidance_scale"):
        params[name] = getattr(options, name)
    return params
