# SPDX-License-Identifier: Apache-2.0
"""Compatibility alias for the SenseNova UMM image generation pipeline."""

from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig


class SenseNovaU1UMMPipelineConfig(SenseNovaU1PipelineConfig):
    """Deprecated alias; the UMM pipeline is now the model's default."""
