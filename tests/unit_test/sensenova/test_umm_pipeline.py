# SPDX-License-Identifier: Apache-2.0
"""Minimal SenseNova UMM routing and output-stage contracts."""

import threading
import time

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.pipeline.umm import (
    CONTINUATION_METADATA_KEY,
    Continuation,
    UMMOutputRelay,
    route_umm,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage


def test_route_umm_uses_the_continuation_stage() -> None:
    continuation = Continuation("execution", 0, "generate", "generate", "nonce")
    payload = StagePayload(
        "request",
        OmniRequest("prompt", metadata={CONTINUATION_METADATA_KEY: continuation}),
        None,
    )
    assert route_umm("request", payload) == "generate"


def test_output_relay_forwards_stream_and_result() -> None:
    relay = UMMOutputRelay()
    relay.inbox.put(
        IncomingMessage(
            "request",
            "stream_chunk",
            StreamItem(0, {"kind": "image"}, "generate", {"modality": "image"}),
        )
    )
    relay.inbox.put(IncomingMessage("request", "new_request", {"type": "umm_result"}))
    thread = threading.Thread(target=relay.start)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while relay.outbox.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert relay.outbox.qsize() == 2
        stream = relay.outbox.get_nowait()
        result = relay.outbox.get_nowait()
    finally:
        relay.stop()
        thread.join(timeout=2)
    assert stream.type == "stream"
    assert stream.data == {"kind": "image"}
    assert stream.metadata == {"modality": "image"}
    assert result.type == "result"
    assert result.data == {"type": "umm_result"}
