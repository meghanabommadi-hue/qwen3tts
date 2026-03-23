"""Pipeline position: NEURAL CODEC — speech tokens ↔ audio tensor.

Role in pipeline:
  Sits between the LLM output and the WAV encoder.
  - encode(audio_path)              → context_tokens + ref_speech_tokens
  - decode_async(tokens, ctx)       → PCM tensor  (batched, async-safe)
  - decode(tokens, ctx)             → PCM tensor  (sync, for warmup/offline)
  - format_prompt(text, ctx, ref)   → LLM input string

Batching strategy:
  Requests are accumulated in an internal queue (max_batch_size OR batch_timeout_ms)
  then dispatched as a single GPU forward pass via ThreadPoolExecutor.
  This coalesces concurrent requests from all WS ports into one kernel launch.

Stub note:
  The actual ONNX/PyTorch codec weights are loaded from the model checkpoint.
  Replace the _load_model() body with the real Qwen3-TTS codec loader.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _PendingRequest:
    tokens: str
    context: str
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


class TTSCodec:
    """Batched neural codec for Qwen3-TTS.

    Thread-safe: decode_async() may be called from any asyncio coroutine.
    The internal batch worker runs in a single ThreadPoolExecutor thread to
    avoid CUDA / ONNX concurrency issues.
    """

    def __init__(
        self,
        max_batch_size: int = 128,
        batch_timeout_ms: float = 1.0,
        gpu_chunk_size: int = 90,
        onnx_workers: int = 1,
        use_trt: bool = False,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.gpu_chunk_size = gpu_chunk_size
        self.use_trt = use_trt

        self._queue: list[_PendingRequest] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=onnx_workers, thread_name_prefix="codec")
        self._decoder_model = None
        self._encoder_model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading — replace with real Qwen3-TTS codec loader
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load ONNX / PyTorch codec weights. Stub — replace with real loader."""
        # Example (real implementation):
        #   import onnxruntime as ort
        #   self._decoder_model = ort.InferenceSession("path/to/decoder.onnx", ...)
        #   self._encoder_model = ort.InferenceSession("path/to/encoder.onnx", ...)
        pass

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(
        self,
        text: str,
        context_tokens: str,
        ref_speech_tokens: Optional[str] = None,
    ) -> str:
        """Build the LLM input prompt for Qwen3-TTS.

        Format:
          <|task_tts|><|start_text|>{text}<|end_text|>
          {context_tokens}
          [<|prompt_speech_start|>{ref_speech_tokens}]
          <|start_speech|>
        """
        parts = [
            "<|task_tts|>",
            "<|start_text|>",
            text,
            "<|end_text|>",
            context_tokens,
        ]
        if ref_speech_tokens:
            parts += ["<|prompt_speech_start|>", ref_speech_tokens]
        parts.append("<|start_speech|>")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Encoding (reference audio → context tokens)
    # ------------------------------------------------------------------

    def encode(self, audio_path: str) -> tuple[str, str] | str:
        """Encode reference audio → (ref_speech_tokens, context_tokens).

        Returns a tuple (ref_speech_tokens, context_tokens) if the encoder
        produces both, or just context_tokens as a string.

        Replace with real encoder inference.
        """
        # Stub — real implementation calls self._encoder_model
        return self._default_context()

    # ------------------------------------------------------------------
    # Decoding (speech tokens → PCM tensor)
    # ------------------------------------------------------------------

    def decode(self, tokens: str, context: str):
        """Synchronous decode — for warmup / offline tools."""
        import numpy as np
        # Stub — real implementation calls self._decoder_model
        token_count = tokens.count("<|speech_token_")
        samples = token_count * 320  # 1 token = 320 samples @ 16kHz
        return np.zeros(samples, dtype=np.float32)

    async def decode_async(self, tokens: str, context: str):
        """Async batched decode — primary path used by server.py.

        Enqueues this request, waits for the batch worker to process it,
        and returns the PCM tensor.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        req = _PendingRequest(tokens=tokens, context=context, future=future, loop=loop)

        with self._lock:
            self._queue.append(req)
            should_flush = len(self._queue) >= self.max_batch_size

        if should_flush:
            self._executor.submit(self._flush_batch)
        else:
            # Schedule a timed flush after batch_timeout_ms
            loop.call_later(
                self.batch_timeout_ms / 1000.0,
                lambda: self._executor.submit(self._flush_batch),
            )

        return await future

    def _flush_batch(self) -> None:
        """Drain the queue and run one GPU forward pass for all pending requests."""
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[: self.max_batch_size]
            self._queue = self._queue[len(batch):]

        try:
            results = self._decode_batch(batch)
            for req, result in zip(batch, results):
                req.loop.call_soon_threadsafe(req.future.set_result, result)
        except Exception as e:
            for req in batch:
                req.loop.call_soon_threadsafe(req.future.set_exception, e)

    def _decode_batch(self, batch: list[_PendingRequest]):
        """Run codec on a batch of requests. Replace with real GPU inference."""
        import numpy as np
        results = []
        for req in batch:
            token_count = req.tokens.count("<|speech_token_")
            samples = token_count * 320
            results.append(np.zeros(samples, dtype=np.float32))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_context() -> str:
        return (
            "<|context_token_3991|><|context_token_1250|><|context_token_2828|>"
            "<|context_token_3303|><|context_token_1187|><|context_token_3021|>"
            "<|context_token_355|><|context_token_3767|><|context_token_3663|>"
            "<|context_token_837|><|context_token_731|><|context_token_3656|>"
            "<|context_token_757|><|context_token_3360|><|context_token_3250|>"
            "<|context_token_3626|><|context_token_1244|><|context_token_526|>"
            "<|context_token_3829|><|context_token_205|><|context_token_1619|>"
            "<|context_token_268|><|context_token_4024|><|context_token_3375|>"
            "<|context_token_3032|><|context_token_2180|><|context_token_3278|>"
            "<|context_token_1609|><|context_token_3685|><|context_token_1359|>"
            "<|context_token_2817|><|context_token_3999|>"
        )
