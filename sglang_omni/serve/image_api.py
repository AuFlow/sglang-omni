# SPDX-License-Identifier: Apache-2.0
"""Model-owned Images API registration contract."""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI


class ImageApiRegistrar(Protocol):
    """Register a model's Images API routes on an initialized app."""

    def __call__(self, app: FastAPI) -> None: ...
