"""Pipeline position: GATEWAY (Redis-backed multi-process mode).

Role in pipeline:
  1. Accepts WebSocket connections from callers (one connection per call_id).
  2. Receives synthesize requests, publishes them to the Redis TTS queue.
  3. Subscribes to the per-call Redis Pub/Sub channel and forwards audio
     token results (+ optional decoded WAV) back to the caller.
  4. Exposes /health and /ports HTTP endpoints for ops/discovery.

When to use this vs server.py:
  Use main.py when you want the full Redis-backed multi-process architecture:
  one gateway process per port, separate worker process(es) for GPU inference.

  Use server.py (via `./run.sh`) when you want a simpler single-process
  setup — no Redis, no worker, model loaded once in-process. (RECOMMENDED)

Port discovery:
  Set QWEN3TTS_KNOWN_PORTS=9765,9766,… so /ports can report which gateway
  ports are live without scanning the full range.
"""

from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qwen3tts.api.websockets import router as websocket_router, manager
from qwen3tts.core.config import settings
from qwen3tts.monitoring.logging import configure_logging


logger = structlog.get_logger(__name__)

_KNOWN_PORTS: List[int] = [
    int(p)
    for p in os.environ.get("QWEN3TTS_KNOWN_PORTS", "").split(",")
    if p.strip().isdigit()
]


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("qwen3tts_gateway_starting", host=settings.ws.host, port=settings.ws.port)

    try:
        await manager.initialize_redis()
        logger.info("redis_connection_ready")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))

    yield

    if manager.redis_client is not None:
        await manager.redis_client.aclose()
        logger.info("redis_connection_closed")

    logger.info("qwen3tts_gateway_stopped")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Qwen3TTS",
        description="Qwen3-powered text-to-speech gateway over WebSocket.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(websocket_router, tags=["websocket"])

    @app.get("/health")
    async def health() -> dict:
        return {
            "service": "Qwen3TTS",
            "status": "running",
            "ws_host": settings.ws.host,
            "ws_port": settings.ws.port,
        }

    @app.get("/ports")
    async def ports() -> dict:
        scan = _KNOWN_PORTS or list(range(9765, 9775))
        live = [p for p in scan if _port_open(p)]
        return {"known": scan, "live": live, "count": len(live)}

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(
        "qwen3tts.main:app",
        host=settings.ws.host,
        port=settings.ws.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
