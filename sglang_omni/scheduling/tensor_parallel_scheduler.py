# SPDX-License-Identifier: Apache-2.0
"""Serial execution of leader-dispatched work containing TP collectives."""

from __future__ import annotations

import threading
from collections.abc import Callable

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class TensorParallelScheduler(SimpleScheduler):
    """Finish every dispatched forward, including aborted and queued requests.

    The stage suppresses aborted results. Rank-local cancellation must never
    remove a forward from the collective sequence. An ordered stop marker
    drains dispatched work before cleanup runs on the scheduler thread.

    Stage calls the optional synchronous validator on the leader before work
    fanout. It must only inspect request settings, without running collectives
    or modifying shared model state. ValueError rejects that request safely.
    """

    preserve_tp_work_on_abort = True

    def __init__(
        self,
        compute_fn: Callable,
        *,
        validate_request: Callable | None = None,
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(compute_fn)
        self.validate_request = validate_request
        self.wait_for_shutdown = True
        self._tp_shutdown_callback = shutdown_callback
        self._stop_marker = IncomingMessage(request_id="", type="stream_done")
        self._stop_requested = False
        self._stop_lock = threading.Lock()
        self._tp_compute_error: Exception | None = None

    def abort(self, request_id: str) -> None:
        # Cancellation is handled by Stage; every rank still runs this work.
        pass

    def _next_message(self) -> IncomingMessage | None:
        msg = super()._next_message()
        if msg is self._stop_marker:
            self._running = False
            return None
        return msg

    def start(self) -> None:
        try:
            super().start()
            if self._tp_compute_error is not None:
                raise RuntimeError(
                    "TP forward failed; the collective sequence cannot be reused"
                ) from self._tp_compute_error
        finally:
            self._running = False
            callback = self._tp_shutdown_callback
            self._tp_shutdown_callback = None
            if callback is not None:
                callback()

    def _run_single(self, msg, loop) -> None:
        try:
            super()._run_single(msg, loop)
        except Exception as exc:
            # A rank-local failure may occur between collectives. Continuing
            # with the next request could pair different layers across ranks.
            self._tp_compute_error = exc
            self._running = False
            raise

    def stop(self) -> None:
        with self._stop_lock:
            if not self._stop_requested:
                self._stop_requested = True
                self.inbox.put(self._stop_marker)
