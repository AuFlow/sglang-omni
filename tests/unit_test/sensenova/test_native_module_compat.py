# SPDX-License-Identifier: Apache-2.0
"""Check standalone native modules without loading accelerator dependencies."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def native_modules(monkeypatch):
    name = "_sensenova_review_modules"
    package = ModuleType(name)
    package.__path__ = [
        str(
            Path(__file__).resolve().parents[3]
            / "sglang_omni/models/sensenova_u1/neo_unify"
        )
    ]
    monkeypatch.setitem(sys.modules, name, package)
    try:
        yield lambda module: importlib.import_module(f"{name}.{module}")
    finally:
        for key in list(sys.modules):
            if key.startswith(f"{name}."):
                sys.modules.pop(key)
            else:
                pass


def test_moe_config_detects_expert_count_without_model_type(native_modules):
    module = native_modules("configuration_neo_chat")
    config = module._build_llm_config(
        {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32}
    )
    assert isinstance(config, module.NEOMoELLMConfig)
    assert config.num_experts == 4
    assert not module._is_moe_llm_config({"num_experts": 1})


def test_model_input_compat_is_available_on_installed_transformers(native_modules):
    module = native_modules("transformers_compat")
    assert callable(module.model_input_compat)


def test_deep_flow_head_constructs_and_accepts_timesteps(native_modules):
    module = native_modules("modeling_fm_modules")
    head = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    with torch.inference_mode():
        outputs = head(torch.randn(5, 8), torch.linspace(0, 1, 5))
    assert outputs.shape == (5, 12)
    assert torch.isfinite(outputs).all()
    assert "net.time_embed.mlp.0.weight" in head.state_dict()
    assert "net.final_layer.adaLN_modulation.1.weight" in head.state_dict()
