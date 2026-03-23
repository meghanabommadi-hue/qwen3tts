"""Pipeline position: WORKER (Redis-backed multi-process mode only).

Role in pipeline:
  Sits between the Redis TTS queue and the Redis Pub/Sub result channel.
  Pops one job at a time (or concurrently up to worker_concurrency), runs
  sglang inference via Qwen3TtsSynthesizer, and publishes audio_tokens back
  so the gateway can forward them to the waiting WebSocket client.

Data flow:
  Redis queue (tts_queue_name)
    → blpop job  {call_id, text_id, text, published_at}
    → synthesis_service.synthesize(text)   [sglang GPU inference]
    → audio_tokens string
    → client.publish(results_channel_prefix:{call_id}, payload)
    → Gateway WebSocket listener receives result

Not used by server.py:
  server.py calls synthesis_service.synthesize() directly — no Redis, no
  worker process needed.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import redis.asyncio as redis
import structlog

from qwen3tts.core.config import settings
from qwen3tts.monitoring.metrics import record_synthesis_latency
from qwen3tts.synthesis.engine import synthesis_service


logger = structlog.get_logger(__name__)


async def _initialize_redis() -> redis.Redis:
    cfg = settings.redis
    redis_url = f"redis://{cfg.host}:{cfg.port}/{cfg.db}"
    client = await redis.from_url(
        redis_url,
        password=cfg.password,
        decode_responses=False,
    )
    logger.info("worker_redis_connected", host=cfg.host, port=cfg.port)
    return client


async def _process_job(client: redis.Redis, job_data: bytes) -> None:
    try:
        job: dict[str, Any] = json.loads(job_data)
        call_id = job["call_id"]
        text_id = job["text_id"]
        text = job["text"]

        if not synthesis_service.is_initialized:
            await synthesis_service.initialize()

        t0 = time.time()
        audio_tokens = await synthesis_service.synthesize(text)
        synth_latency = time.time() - t0

        record_synthesis_latency(call_id, text_id, synth_latency)

        payload = {
            "call_id": call_id,
            "text_id": text_id,
            "audio_tokens": audio_tokens,
            "is_final": True,
            "generated_at": time.time(),
            "llm_s": round(synth_latency, 4),
        }
        channel = f"{settings.redis.results_channel_prefix}:{call_id}"
        await client.publish(channel, json.dumps(payload))

        logger.info(
            "tts_job_completed",
            call_id=call_id,
            text_id=text_id,
            token_length=len(audio_tokens),
            synth_latency=round(synth_latency, 3),
        )
    except Exception as e:
        logger.error("worker_job_failed", error=str(e))


async def run_worker() -> None:
    client = await _initialize_redis()
    logger.info("qwen3tts_worker_started", queue=settings.redis.tts_queue_name)
    while True:
        try:
            result = await client.blpop(settings.redis.tts_queue_name, timeout=1)
            if result is None:
                continue
            _, job_data = result
            await _process_job(client, job_data)
        except Exception as e:
            logger.error("worker_loop_error", error=str(e))
            await asyncio.sleep(1.0)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()


class SynthesizerWorker:
    """Class-based concurrent worker — supports up to worker_concurrency parallel jobs."""

    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        self.running: bool = False
        self.semaphore: asyncio.Semaphore | None = None
        self.processing_tasks: set[asyncio.Task] = set()

    async def initialize(self) -> None:
        cfg = settings.redis
        redis_url = f"redis://{cfg.host}:{cfg.port}/{cfg.db}"
        self.redis_client = await redis.from_url(
            redis_url, password=cfg.password, decode_responses=False,
        )
        ping_response = self.redis_client.ping()
        if asyncio.iscoroutine(ping_response):
            await ping_response
        logger.info("worker_redis_connected", host=cfg.host, port=cfg.port)

        if not synthesis_service.is_initialized:
            await synthesis_service.initialize()
        logger.info("tts_model_loaded")

        self.semaphore = asyncio.Semaphore(cfg.worker_concurrency)
        self.running = True

    async def shutdown(self) -> None:
        self.running = False
        pending = list(self.processing_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.redis_client:
            await self.redis_client.aclose()

    async def _process_job_wrapper(self, job_data: bytes) -> None:
        if self.semaphore is None:
            return
        await self.semaphore.acquire()
        try:
            job: dict[str, Any] = json.loads(job_data)
            await self._process_single_job(job)
        except json.JSONDecodeError as e:
            logger.error("invalid_job_format", error=str(e))
        except Exception as e:
            logger.error("wrapper_processing_failed", error=str(e))
        finally:
            self.semaphore.release()

    async def _process_single_job(self, job: dict[str, Any]) -> None:
        try:
            job_received_time = time.time()
            call_id = job["call_id"]
            text_id = job["text_id"]
            text = job["text"]
            published_at = job["published_at"]
            queueing_latency = job_received_time - published_at

            start_synth = time.time()
            audio_tokens = await synthesis_service.synthesize(text)
            synth_latency_val = time.time() - start_synth

            record_synthesis_latency(call_id, text_id, synth_latency_val)

            payload = {
                "call_id": call_id,
                "text_id": text_id,
                "audio_tokens": audio_tokens,
                "is_final": True,
                "generated_at": time.time(),
                "queueing_latency": queueing_latency,
                "synthesis_latency": synth_latency_val,
            }
            channel = f"{settings.redis.results_channel_prefix}:{call_id}"
            assert self.redis_client is not None
            await self.redis_client.publish(channel, json.dumps(payload))

        except Exception as e:
            logger.error(
                "job_processing_failed",
                call_id=job.get("call_id"),
                text_id=job.get("text_id"),
                error=str(e),
            )

    async def run(self) -> None:
        assert self.redis_client is not None
        while self.running:
            try:
                result = await self.redis_client.blpop(
                    settings.redis.tts_queue_name, timeout=1,
                )
                if result is None:
                    continue
                _, job_data = result
                task = asyncio.create_task(self._process_job_wrapper(job_data))
                self.processing_tasks.add(task)
                task.add_done_callback(self.processing_tasks.discard)
            except Exception as e:
                logger.error("worker_loop_error", error=str(e))
                await asyncio.sleep(1.0)
