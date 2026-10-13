# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from sglang_omni.config import ProcessConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages


def _config(size=2, batch=1):
    data = SenseNovaU1PipelineConfig(model_path="/model/sensenova").model_dump()
    data["stages"][0]["tp_size"] = size
    data["stages"][0]["gpu"] = list(range(size))
    data["stages"][0]["factory"]["max_batch_size"] = batch
    return data


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_supported_tp_sizes(size):
    config = SenseNovaU1PipelineConfig.model_validate(_config(size))
    assert config.stages[0].tp_size == size


def test_tp_rejects_unsupported_size_and_dynamic_batching():
    with pytest.raises(ValueError, match="tp_size=1, 2, 4, or 8"):
        SenseNovaU1PipelineConfig.model_validate(_config(3))
    with pytest.raises(ValueError, match="max_batch_size=1"):
        SenseNovaU1PipelineConfig.model_validate(_config(batch=2))
    assert SenseNovaU1PipelineConfig.model_validate(_config(1, 2))


def test_dp_tp_expands_into_independent_gpu_groups():
    data = _config()
    data["processes"] = {
        "sensenova_generate": ProcessConfig(
            num_replicas=2, replica_devices=[0, 1, 2, 3]
        )
    }
    config = SenseNovaU1PipelineConfig.model_validate(data)
    plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, plan)
    assert [(stage.name, stage.tp_size, stage.gpu) for stage in expanded] == [
        ("generate@r0", 2, [0, 1]),
        ("generate@r1", 2, [2, 3]),
    ]


def test_dp_tp_rejects_overlapping_replica_devices():
    data = _config()
    data["processes"] = {
        "sensenova_generate": ProcessConfig(
            num_replicas=2, replica_devices=[0, 1, 1, 2]
        )
    }
    with pytest.raises(ValueError, match="distinct GPUs"):
        SenseNovaU1PipelineConfig.model_validate(data)


@pytest.mark.parametrize("name", ["sensenova_u1_tp2.yaml", "sensenova_u1_dp2_tp2.yaml"])
def test_tp_example_config(name):
    path = Path(__file__).resolve().parents[3] / "examples" / "configs" / name
    config = ConfigManager.from_file(str(path)).config
    plan, stages = compile_logical_processes(config)
    expanded, _ = expand_replica_stages(stages, plan)
    assert all(stage.tp_size == 2 for stage in expanded)


def test_dotted_cli_enables_tp_and_rejects_batching():
    manager = ConfigManager(SenseNovaU1PipelineConfig(model_path="/model/sensenova"))
    config = manager.merge_config(
        [("generate.tp_size", "2"), ("generate.gpu", "[0, 1]")]
    )
    assert config.stages[0].tp_size == 2
    with pytest.raises(ValueError, match="max_batch_size=1"):
        ConfigManager(config).merge_config([("generate.factory.max_batch_size", "2")])
