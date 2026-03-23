"""Pipeline position: CONFIGURATION (read by every module at import time).

Role in pipeline:
  Single source of truth for all tunable parameters. Every pipeline stage
  imports `settings` from here rather than reading env-vars directly.

Key sections and their pipeline consumers:
  TtsModelSettings  → synthesis/models.py (sglang engine init, sampling params)
                       synthesis/engine.py (model_dir, ref_audio paths)
  DecoderSettings   → api/websockets.py   (enabled flag, to_wav flag)
                       decoder/decoder.py  (sample_rate)
  RedisSettings     → api/websockets.py   (queue publish, pub/sub subscribe)
                       worker.py           (queue consume, pub/sub publish)
  WebSocketSettings → main.py             (uvicorn host/port)

All values can be overridden via environment variables (QWEN3TTS_ prefix).
No env-var → sensible defaults used (greedy decoding, 16 kHz, Redis localhost).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import ClassVar, Literal

_MODELS_DIR = str(Path.home() / "models")


class TtsModelSettings(BaseModel):
    """TTS model configuration for Qwen3-TTS."""

    # Path to the Qwen3-TTS model checkpoint (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
    model_dir: str = f"{_MODELS_DIR}/Qwen3-TTS"

    # Warmup sentence — spoken at startup to prime JIT caches.
    warmup_sentence: str = "Hello, how can I help you today?"

    # Reference audio for speaker style/timbre (leave empty to use model default)
    ref_audio: str = f"{_MODELS_DIR}/Qwen3-TTS/ref_audio.wav"

    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # sglang engine parameters
    mem_fraction_static: float = 0.65
    attention_backend: str = "triton"
    chunked_prefill_size: int = -1
    schedule_policy: str = "lpm"
    cuda_graph_max_bs: int = 160
    disable_radix_cache: bool = False
    num_continuous_decode_steps: int = 4

    # Generation / sampling parameters
    max_tokens: int = 600
    temperature: float = 0.0               # greedy — fastest, deterministic
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.6
    min_p: float = 0.05


class DecoderSettings(BaseModel):
    """Decoder / vocoder configuration."""

    model_gpu_id: int = 0
    decoder_gpu_id: int = 0

    sample_rate: int = 16000

    # Set to False to skip ncodec decoding and forward raw LLM tokens instead.
    enabled: bool = False

    # Set to False to decode to raw PCM bytes only (skip WAV encoding).
    to_wav: bool = True

    # TTSCodec batch queue settings
    max_batch: int = 128
    batch_timeout_ms: float = 1.0
    gpu_chunk_size: int = 90
    onnx_workers: int = 1
    use_trt: bool = False


class StreamingSettings(BaseModel):
    """Streaming audio chunk configuration."""

    enabled: bool = True

    # Speech tokens accumulated before decoding and sending a chunk.
    chunk_tokens: int = 30

    # Linear crossfade overlap between consecutive chunks (samples at 16 kHz).
    crossfade_samples: int = 400

    # Linear fade-out applied to the tail of each non-final chunk.
    fade_out_samples: int = 160


class WebSocketSettings(BaseModel):
    """Gateway WebSocket server settings."""

    host: str = "0.0.0.0"
    port: int = 9765


class Settings(BaseSettings):
    """Top-level Qwen3TTS settings."""

    model_config = SettingsConfigDict(env_prefix="QWEN3TTS_", env_nested_delimiter="__")

    ws: WebSocketSettings = WebSocketSettings()
    tts_model: TtsModelSettings = TtsModelSettings()
    decoder: DecoderSettings = DecoderSettings()
    streaming: StreamingSettings = StreamingSettings()

    # Redis queue / pubsub configuration
    class RedisSettings(BaseModel):
        host: str = "localhost"
        port: int = 6379
        db: int = 0
        password: str | None = None

        tts_queue_name: str = "qwen3tts:tts_queue"
        results_channel_prefix: str = "qwen3tts:audio"
        decoded_channel_prefix: str = "qwen3tts:decoded"
        worker_concurrency: int = 32

    redis: RedisSettings = RedisSettings()


settings = Settings()
