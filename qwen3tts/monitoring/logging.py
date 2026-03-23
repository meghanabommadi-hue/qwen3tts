"""Pipeline position: OBSERVABILITY — structured logging for all stages.

Role in pipeline:
  Called once at startup (main.py lifespan / server.py __main__) via
  configure_logging(). Every module then does:

      logger = structlog.get_logger(__name__)
      logger.info("tts_job_completed", call_id=…, synth_latency=…)

Output formats:
  json_logs=False  → coloured key=value console output (default/dev)
  json_logs=True   → one JSON object per line (production / log aggregators)

Key events to grep across the pipeline:
  qwen3tts_gateway_starting      main.py      process starting
  redis_connection_ready         main.py      Redis ready
  websocket_connected            websockets   call accepted
  job_published_to_queue         websockets   text pushed to Redis queue
  tts_job_completed              worker       sglang done, tokens published
  result_forwarded_to_client     websockets   audio sent back to caller
  websocket_disconnected         websockets   call ended
"""

from __future__ import annotations

from typing import Any, Iterable

import structlog


def _build_processors(json_logs: bool) -> Iterable[Any]:
    base: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if json_logs:
        base.append(structlog.processors.JSONRenderer())
    else:
        base.append(structlog.dev.ConsoleRenderer())
    return base


def configure_logging(*, json_logs: bool = False) -> None:
    """Configure structlog for Qwen3TTS.

    Call once during application startup before importing modules that use
    structlog.get_logger().
    """
    structlog.configure(
        processors=_build_processors(json_logs),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(20),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name) if name is not None else structlog.get_logger()
