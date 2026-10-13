# SPDX-License-Identifier: Apache-2.0
"""Inference-only tensor parallelism for the dense SenseNova Qwen3 backbone.

Hidden states stay replicated. Q/K/V and gate/up projections shard their
output features; attention output and down projections sum partial results.
Vision encoders, embeddings, norms, and the pixel head stay replicated.
"""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

SUPPORTED_TP_SIZES = (1, 2, 4, 8)


def validate_tp_settings(tp_rank: int, tp_size: int, max_batch_size: int) -> None:
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError("SenseNova TP supports tp_size=1, 2, 4, or 8")
    if not 0 <= tp_rank < tp_size:
        raise ValueError("SenseNova tp_rank must be in [0, tp_size)")
    if tp_size > 1 and max_batch_size != 1:
        raise ValueError("SenseNova TP requires max_batch_size=1")


def validate_tp_device(device: torch.device, nccl_port: int | None) -> None:
    if device.type not in {"cuda", "npu"}:
        raise ValueError("SenseNova TP requires CUDA or Ascend NPU")
    if nccl_port is None:
        raise ValueError("SenseNova TP requires nccl_port from the stage launcher")


def initialize_tp(
    device: torch.device, tp_rank: int, tp_size: int, nccl_port: int | None
) -> dist.ProcessGroup:
    """Create the process-local TP group on the launcher's reserved port."""
    validate_tp_device(device, nccl_port)
    if dist.is_initialized():
        raise RuntimeError("SenseNova TP requires its own distributed worker process")
    getattr(torch, device.type).set_device(device)
    dist.init_process_group(
        backend="hccl" if device.type == "npu" else "nccl",
        init_method=f"tcp://127.0.0.1:{nccl_port}",
        rank=tp_rank,
        world_size=tp_size,
        timeout=timedelta(minutes=10),
    )
    return dist.group.WORLD


class RowParallelLinear(nn.Linear):
    """Linear with a local input shard and a replicated, summed output."""

    def __init__(self, linear: nn.Linear, rank: int, size: int, group):
        width = linear.in_features // size
        super().__init__(
            width,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.weight = nn.Parameter(
            linear.weight.detach()[:, rank * width : (rank + 1) * width]
            .contiguous()
            .clone(),
            requires_grad=False,
        )
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.tp_group = group
        self.train(linear.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight)
        dist.all_reduce(output, group=self.tp_group)
        # Add the replicated bias once, after summing the partial products.
        if self.bias is not None:
            output = output + self.bias
        return output


def _column_parallel(linear: nn.Linear, rank: int, size: int) -> nn.Linear:
    width = linear.out_features // size
    shard = nn.Linear(
        linear.in_features,
        width,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    shard.weight = nn.Parameter(
        linear.weight.detach()[rank * width : (rank + 1) * width].clone(),
        requires_grad=False,
    )
    if linear.bias is not None:
        shard.bias = nn.Parameter(
            linear.bias.detach()[rank * width : (rank + 1) * width].clone(),
            requires_grad=False,
        )
    shard.train(linear.training)
    return shard


def apply_tensor_parallel(model: nn.Module, rank: int, size: int, group) -> None:
    """Shard loaded CPU weights before moving the model to its accelerator."""
    validate_tp_settings(rank, size, 1)
    if size == 1:
        return
    if getattr(model, "sensenova_tp_size", 1) != 1:
        raise ValueError("SenseNova tensor parallelism has already been applied")
    backbone = model.language_model.model
    config = backbone.config
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(
            "SenseNova TP currently supports the dense Qwen3 backbone only"
        )
    for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(config, name) % size:
            raise ValueError(f"SenseNova {name} must be divisible by tp_size={size}")

    # Validate every projection before changing any module.
    columns: list[tuple[nn.Module, str]] = []
    rows: list[tuple[nn.Module, str]] = []
    for layer in backbone.layers:
        attention = layer.self_attn
        for suffix in ("", "_mot_gen"):
            columns.extend(
                (attention, f"{name}{suffix}")
                for name in ("q_proj", "k_proj", "v_proj")
            )
            rows.append((attention, f"o_proj{suffix}"))
        for mlp in (layer.mlp, layer.mlp_mot_gen):
            columns.extend((mlp, name) for name in ("gate_proj", "up_proj"))
            rows.append((mlp, "down_proj"))
    for projections, dimension in ((columns, "out_features"), (rows, "in_features")):
        for module, name in projections:
            linear = getattr(module, name)
            if not isinstance(linear, nn.Linear) or getattr(linear, dimension) % size:
                raise ValueError(
                    f"SenseNova {name} cannot be sharded with tp_size={size}"
                )

    for module, name in columns:
        setattr(module, name, _column_parallel(getattr(module, name), rank, size))
    for module, name in rows:
        setattr(
            module, name, RowParallelLinear(getattr(module, name), rank, size, group)
        )
    for layer in backbone.layers:
        for mlp in (layer.mlp, layer.mlp_mot_gen):
            mlp.intermediate_size //= size
            mlp._npu_gate_up_weight = None
    model.sensenova_tp_size = size
    model.sensenova_tp_group = group


def synchronize_tp_noise(model: nn.Module, noise: torch.Tensor) -> None:
    """Use rank zero's initial image noise on every rank, on every backend."""
    if getattr(model, "sensenova_tp_size", 1) > 1:
        dist.broadcast(noise, src=0, group=model.sensenova_tp_group)
