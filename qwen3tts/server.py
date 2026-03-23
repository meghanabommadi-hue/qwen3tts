#!/usr/bin/env python3
"""
Pipeline position: SINGLE-PROCESS GATEWAY (primary production entry point via run.sh).

Role in pipeline:
  Self-contained TTS server — no Redis, no worker process, no uvicorn per port.
  Loads sglang + codec once, then handles all WebSocket ports in one asyncio
  event loop. This is the recommended way to run Qwen3TTS in production.

  Client
    │  WebSocket (text) on port 9765…9765+N
    ▼
  server.py  (one process, one GPU load)
    │  normalize_text()
    │  Qwen3TtsSynthesizer.synthesize_stream(text)  [sglang in-process]
    │  → audio_tokens string / stream
    │  TTSCodec.decode_async(tokens, ctx)  → PCM tensor
    │  tensor_to_wav()  → WAV bytes
    ▼
  Client  (JSON metadata + binary WAV frames, streamed)

Port model:
  --ports N opens N consecutive ports starting at --base-port (default 9765).
  All ports share the same Qwen3TtsSynthesizer singleton (one model load).
  Concurrent requests from different ports are handled by asyncio concurrency
  — sglang's async_generate serialises GPU work internally.

Warmup:
  On startup, 40 sentences are synthesized concurrently to prime the GPU
  and JIT caches before real traffic arrives.

Usage (preferred):
    ./run.sh --ports 100
    ./run.sh --ports 3 --port 9000

Direct:
    python -m qwen3tts.server --ports 3
    python -m qwen3tts.server --ports 100 --base-port 9765
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import json
import re
import time

import numpy as np
import uuid
from pathlib import Path

import logging
from aiohttp import web

import websockets
from websockets.exceptions import WebSocketException

from qwen3tts.core.config import settings
from qwen3tts.decoder.decoder import tensor_to_wav, SAMPLE_RATE
from qwen3tts.monitoring.metrics import (
    record_call,
    record_ws_connection_open,
    record_ws_connection_close,
    record_ws_error,
    record_ws_done,
    ws_log_snapshot,
    record_port_change,
)
from qwen3tts.processing.audio_processing import crossfade, fade_out  # noqa: F401
from qwen3tts.processing.text_normalize import normalize_text
from qwen3tts.synthesis.models import Qwen3TtsSynthesizer

logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

_synthesizer: Qwen3TtsSynthesizer | None = None
_wav_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="wav_enc")
_RE_SPEECH = re.compile(r"<\|speech_token_\d+\|>", re.ASCII)
_STREAM_CHUNK_TOKENS = settings.streaming.chunk_tokens
_CROSSFADE_SAMPLES   = settings.streaming.crossfade_samples
_FADE_OUT_SAMPLES    = settings.streaming.fade_out_samples
_audio_out_dir: Path | None = None
_open_ports: set[int] = set()

# Rolling RTF tracking
_rtf_count: int   = 0
_rtf_sum:   float = 0.0


def _record_rtf(total_s: float, tokens: int) -> float:
    """Record RTF for one request, return current avg RTF."""
    global _rtf_count, _rtf_sum
    if tokens <= 0:
        return 0.0
    audio_s = tokens * 320 / 16000
    rtf = total_s / audio_s
    _rtf_count += 1
    _rtf_sum   += rtf
    return _rtf_sum / _rtf_count


_llm_log: Path = Path(__file__).parents[1] / "llm.log"
_llm_log_file = None
_llm_out_log: Path = Path(__file__).parents[1] / "monitoring" / "llm_outputs.jsonl"
_llm_out_log_file = None


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _tsms() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(line: str) -> None:
    if _llm_log_file is not None:
        _llm_log_file.write(line + "\n")
        _llm_log_file.flush()


async def _get_synthesizer() -> Qwen3TtsSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = Qwen3TtsSynthesizer()
        print(f"[{_ts()}] loading Qwen3 model...", flush=True)
        await _synthesizer.initialize()
        print(f"[{_ts()}] model ready", flush=True)
    return _synthesizer


# ---------------------------------------------------------------------------
# Streaming request handler
# ---------------------------------------------------------------------------

async def _handle_streaming_request(
    ws: websockets.ServerConnection,
    synth: Qwen3TtsSynthesizer,
    text: str,
    call_id: str,
    text_id: str,
    port: int,
    ts_text_recv: str,
) -> None:
    """Stream audio chunks to the client as the LLM produces speech tokens.

    Every _STREAM_CHUNK_TOKENS speech tokens the buffer is decoded to PCM
    and sent as two frames:
      - JSON  { type:"audio_chunk", chunk_index, call_id, text_id, is_final }
      - bytes  raw WAV for that chunk
    A final { type:"audio_done", ... } JSON frame is sent after all chunks.
    """
    codec = synth._tts_codec
    ctx   = synth._context_tokens

    t0             = time.perf_counter()
    ts_llm_start   = _tsms()
    _log(f"{ts_llm_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")

    buffer         = ""
    token_buf      = []
    overlap_tokens: list = []
    _OVERLAP       = 4
    chunk_index    = 0
    total_tokens   = 0
    total_wav_b    = 0
    decode_total   = 0.0
    wav_total      = 0.0
    first_chunk_sent = False

    loop = asyncio.get_event_loop()

    async def _flush_chunk(is_final: bool) -> None:
        nonlocal chunk_index, total_tokens, total_wav_b, decode_total, wav_total, first_chunk_sent, overlap_tokens
        if not token_buf:
            return

        real_tokens   = list(token_buf)
        token_buf.clear()
        decode_tokens = overlap_tokens + real_tokens
        chunk_tokens  = "".join(decode_tokens)
        n_overlap     = len(overlap_tokens)
        overlap_tokens = real_tokens[-_OVERLAP:]

        td = time.perf_counter()
        wav_tensor = await codec.decode_async(chunk_tokens, ctx)
        decode_total += time.perf_counter() - td

        tw = time.perf_counter()
        pcm = np.asarray(wav_tensor, dtype=np.float32).squeeze()
        if n_overlap > 0:
            discard = n_overlap * 320
            pcm = pcm[discard:]

        decoded = await loop.run_in_executor(_wav_executor, tensor_to_wav, pcm)
        wav_total += time.perf_counter() - tw

        n_tok = len(real_tokens)
        total_tokens += n_tok
        total_wav_b  += len(decoded.wav_bytes)

        ts_chunk = _tsms()
        if not first_chunk_sent:
            ttft = round((time.perf_counter() - t0) * 1000)
            print(f"[{ts_chunk}] :{port} {call_id}  first_chunk  ttft={ttft}ms  tokens={n_tok}", flush=True)
            first_chunk_sent = True

        await ws.send(json.dumps({
            "type":        "audio_chunk",
            "call_id":     call_id,
            "text_id":     text_id,
            "chunk_index": chunk_index,
            "sample_rate": SAMPLE_RATE,
            "wav_bytes":   len(decoded.wav_bytes),
            "tokens":      n_tok,
            "is_final":    is_final,
        }))
        await ws.send(decoded.wav_bytes)
        chunk_index += 1

    try:
        async for delta in synth.synthesize_stream(text):
            if not delta:
                await _flush_chunk(is_final=True)
                break

            buffer += delta
            last_end = 0
            for m in _RE_SPEECH.finditer(buffer):
                token_buf.append(m.group())
                last_end = m.end()
            if last_end:
                buffer = buffer[last_end:]

            while len(token_buf) >= _STREAM_CHUNK_TOKENS:
                await _flush_chunk(is_final=False)

        llm_s   = round(time.perf_counter() - t0, 4)
        llm_ms  = round(llm_s * 1000)
        total_s = round(time.perf_counter() - t0, 4)
        ts_done = _tsms()
        _log(f"{ts_done}  OUT  port={port}  text_id={text_id}  call_id={call_id}  llm_ms={llm_ms}")

        avg_rtf = _record_rtf(total_s, total_tokens)
        audio_s = total_tokens * 320 / 16000
        rtf     = total_s / audio_s if audio_s > 0 else 0.0

        await ws.send(json.dumps({
            "type":            "audio_done",
            "call_id":         call_id,
            "text_id":         text_id,
            "text":            text,
            "chunks":          chunk_index,
            "total_tokens":    total_tokens,
            "total_wav_bytes": total_wav_b,
            "sample_rate":     SAMPLE_RATE,
            "llm_s":           llm_s,
            "decode_s":        round(decode_total, 4),
            "rtf":             round(rtf, 3),
            "avg_rtf":         round(avg_rtf, 3),
        }))

        print(
            f"[{ts_done}] :{port} {call_id}  stream_done"
            f"  chunks={chunk_index}"
            f"  tokens={total_tokens}"
            f"  llm={llm_ms}ms"
            f"  decode={round(decode_total*1000)}ms"
            f"  wav_enc={round(wav_total*1000)}ms"
            f"  total={round(total_s*1000)}ms"
            f"  wav={total_wav_b}B"
            f"  rtf={rtf:.3f}"
            f"  avg_rtf={avg_rtf:.3f}",
            flush=True,
        )

    except Exception as e:
        ts_err = _tsms()
        print(f"[{ts_err}] :{port} {call_id}  STREAM ERROR: {e}", flush=True)
        record_ws_error(call_id, port=port, text_id=text_id, error=str(e))
        try:
            await ws.send(json.dumps({
                "type": "error", "call_id": call_id, "text_id": text_id, "error": str(e),
            }))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------

async def handle_connection(ws: websockets.ServerConnection, port: int) -> None:
    """Handle one persistent WebSocket connection (one call = one socket)."""
    peer = ws.remote_address
    conn_id = f"{peer[0]}:{peer[1]}"
    print(f"[{_ts()}] :{port} connected  peer={conn_id}", flush=True)
    record_ws_connection_open(conn_id, port=port)

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                continue

            text    = (data.get("text") or "").strip()
            call_id = data.get("call_id") or f"{peer[0]}:{peer[1]}"
            text_id = data.get("text_id") or str(uuid.uuid4())
            if not text:
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id,
                    "error": "Missing text",
                }))
                continue

            text      = normalize_text(text)
            streaming = bool(data.get("streaming", True))

            ts_text_recv = _tsms()
            _log(f"{ts_text_recv}  RECV port={port}  text_id={text_id}  call_id={call_id}  streaming={streaming}  text={text[:60]!r}")
            print(f"[{ts_text_recv}] :{port} {call_id}  {'stream' if streaming else 'req'}  {text[:60]!r}", flush=True)

            synth = await _get_synthesizer()

            if streaming:
                await _handle_streaming_request(ws, synth, text, call_id, text_id, port, ts_text_recv)
                continue

            # Non-streaming path
            try:
                t0 = time.perf_counter()
                ts_llm_start = _tsms()
                _log(f"{ts_llm_start}  IN   port={port}  text_id={text_id}  call_id={call_id}  text={text}")
                audio_tokens = await asyncio.wait_for(synth.synthesize(text), timeout=30.0)
                llm_s    = round(time.perf_counter() - t0, 4)
                llm_ms   = round(llm_s * 1000)
                ts_tokens_ready = _tsms()
                _log(f"{ts_tokens_ready}  OUT  port={port}  text_id={text_id}  call_id={call_id}  llm_ms={llm_ms}")

                token_count = audio_tokens.count("<|speech_token_")

                if _llm_out_log_file is not None:
                    _llm_out_log_file.write(json.dumps({
                        "ts": ts_tokens_ready, "call_id": call_id, "text_id": text_id,
                        "port": port, "text": text,
                        "audio_tokens": audio_tokens, "token_count": token_count,
                        "llm_ms": llm_ms,
                    }, ensure_ascii=False) + "\n")

                codec = synth._tts_codec
                ctx   = synth._context_tokens
                td = time.perf_counter()
                wav_tensor = await asyncio.wait_for(codec.decode_async(audio_tokens, ctx), timeout=30.0)
                decode_s = round(time.perf_counter() - td, 4)

                tw = time.perf_counter()
                decoded = tensor_to_wav(wav_tensor)
                wav_s = round(time.perf_counter() - tw, 4)

                record_call(
                    call_id=call_id, text_id=text_id, port=port, text=text,
                    token_count=token_count, llm_s=llm_s, decode_s=decode_s,
                    wav_bytes=len(decoded.wav_bytes), ts=ts_tokens_ready,
                )

                if _audio_out_dir is not None:
                    wav_file = _audio_out_dir / f"{text_id}.wav"
                    wav_file.write_bytes(decoded.wav_bytes)
                    print(f"[{_ts()}] :{port}  saved → {wav_file}", flush=True)

                await ws.send(json.dumps({
                    "type": "audio", "call_id": call_id, "text_id": text_id,
                    "text": text, "audio_tokens": audio_tokens,
                    "sample_rate": SAMPLE_RATE, "wav_bytes": len(decoded.wav_bytes),
                    "is_final": True, "llm_s": llm_s, "decode_s": decode_s,
                }))
                await ws.send(decoded.wav_bytes)
                ts_audio_sent = _tsms()

                total_s = llm_s + decode_s + wav_s
                avg_rtf = _record_rtf(total_s, token_count)
                audio_s = token_count * 320 / 16000
                rtf     = total_s / audio_s if audio_s > 0 else 0.0
                print(
                    f"[{ts_audio_sent}] :{port} {call_id}  done"
                    f"  llm={llm_ms}ms  decode={round(decode_s*1000)}ms"
                    f"  wav_enc={round(wav_s*1000)}ms  total={round(total_s*1000)}ms"
                    f"  tokens={token_count}  wav={len(decoded.wav_bytes)}B"
                    f"  rtf={rtf:.3f}  avg_rtf={avg_rtf:.3f}",
                    flush=True,
                )

                record_ws_done(
                    call_id, port=port, text_id=text_id, token_count=token_count,
                    llm_ms=llm_ms, decode_ms=round(decode_s * 1000),
                    total_ms=round(total_s * 1000), wav_bytes=len(decoded.wav_bytes),
                    ts_text_recv=ts_text_recv, ts_llm_start=ts_llm_start,
                    ts_tokens_ready=ts_tokens_ready, ts_audio_sent=ts_audio_sent,
                )

            except Exception as e:
                ts_err = _tsms()
                print(f"[{ts_err}] :{port} {call_id}  ERROR: {e}", flush=True)
                record_ws_error(call_id, port=port, text_id=text_id, error=str(e))
                await ws.send(json.dumps({
                    "type": "error", "call_id": call_id, "text_id": text_id, "error": str(e),
                }))

    except WebSocketException:
        pass
    except Exception as e:
        print(f"[{_ts()}] :{port} connection error: {e}", flush=True)
    finally:
        record_ws_connection_close(conn_id, port=port)
        print(f"[{_ts()}] :{port} disconnected  peer={conn_id}", flush=True)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

_WARMUP_SENTENCES = [
    "Hello, how can I help you today?",
    "Please hold on for a moment.",
    "Your request has been processed successfully.",
    "Can you please confirm your name?",
    "Thank you for calling. Have a great day.",
    "I'm sorry, could you repeat that?",
    "Your account balance is currently up to date.",
    "We will send you a confirmation email shortly.",
    "Is there anything else I can help you with?",
    "Your payment has been received. Thank you.",
    "The support team will contact you within twenty four hours.",
    "Please enter your four digit PIN to continue.",
    "Your order number is ready for dispatch.",
    "We appreciate your patience while we process your request.",
    "Your subscription has been renewed successfully.",
    "The estimated delivery time is three to five business days.",
    "Please verify your identity before proceeding.",
    "Your new password has been set. Please log in again.",
    "We have received your complaint and will investigate promptly.",
    "Thank you for choosing our service. Goodbye.",
]


async def _warmup(synth: Qwen3TtsSynthesizer) -> None:
    if not _WARMUP_SENTENCES:
        return
    batch_size = 40
    sentences = [_WARMUP_SENTENCES[i % len(_WARMUP_SENTENCES)] for i in range(batch_size)]
    print(f"[{_ts()}] warmup: running {batch_size} sentences concurrently...", flush=True)
    t0 = time.perf_counter()

    async def _one(sentence: str) -> bool:
        try:
            await synth.synthesize(normalize_text(sentence))
            return True
        except Exception as e:
            print(f"[{_ts()}] warmup sentence failed: {e}", flush=True)
            return False

    results = await asyncio.gather(*[_one(s) for s in sentences])
    ok = sum(results)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[{_ts()}] warmup done  {ok}/{batch_size} ok  ({elapsed:.0f}ms total)", flush=True)


async def _warmup_port(port: int, sentence: str) -> None:
    url = f"ws://127.0.0.1:{port}/ws/warmup"
    try:
        async with websockets.connect(url, ping_interval=None, max_size=100 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(json.dumps({
                "type": "synthesize", "call_id": "warmup", "text_id": "warmup", "text": sentence,
            }))
            await ws.recv()
    except Exception as e:
        print(f"[{_ts()}] warmup port {port} failed: {e}", flush=True)


async def _warmup_all_ports(ports: list[int]) -> None:
    sentence = settings.tts_model.warmup_sentence
    if not sentence or not ports:
        return
    print(f"[{_ts()}] warming up {len(ports)} port(s) concurrently...", flush=True)
    t0 = time.perf_counter()
    await asyncio.gather(*[_warmup_port(p, sentence) for p in ports])
    print(f"[{_ts()}] all ports warmed up  ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)


# ---------------------------------------------------------------------------
# Port binding
# ---------------------------------------------------------------------------

async def _bind_ws_port(port: int) -> bool:
    if port in _open_ports:
        return False

    async def handler(ws: websockets.ServerConnection, p: int = port) -> None:
        await handle_connection(ws, p)

    await websockets.serve(
        handler, "0.0.0.0", port,
        ping_interval=30, ping_timeout=30,
        max_size=100 * 1024 * 1024,
    )
    _open_ports.add(port)
    record_port_change(_open_ports)
    print(f"[{_ts()}] opened ws://0.0.0.0:{port}", flush=True)
    return True


# ---------------------------------------------------------------------------
# HTTP control API
# ---------------------------------------------------------------------------

async def _http_add_port(req: web.Request) -> web.Response:
    try:
        port = int(req.rel_url.query["port"])
    except (KeyError, ValueError):
        return web.Response(status=400, text="missing or invalid ?port=N")
    if port < 1024 or port > 65535:
        return web.Response(status=400, text="port out of range")
    already = port in _open_ports
    if not already:
        await _bind_ws_port(port)
    return web.json_response({"port": port, "opened": not already})


async def _http_list_ports(req: web.Request) -> web.Response:
    return web.json_response({"ports": sorted(_open_ports)})


async def _http_ready(req: web.Request) -> web.Response:
    if _synthesizer is None:
        return web.Response(status=503, text="loading")
    return web.json_response({"ready": True, "ports": sorted(_open_ports)})


async def _http_ws_log(req: web.Request) -> web.Response:
    return web.json_response(ws_log_snapshot())


async def _http_metrics(req: web.Request) -> web.Response:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return web.Response(body=generate_latest(), content_type=ct)


async def _run_control_api(ctrl_port: int) -> None:
    app = web.Application()
    app.router.add_post("/ports/add", _http_add_port)
    app.router.add_get("/ports",      _http_list_ports)
    app.router.add_get("/ready",      _http_ready)
    app.router.add_get("/metrics",    _http_metrics)
    app.router.add_get("/ws/log",     _http_ws_log)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", ctrl_port)
    await site.start()
    print(f"[{_ts()}] control API  http://127.0.0.1:{ctrl_port}", flush=True)


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

async def run_server(base_port: int, n_ports: int, ctrl_port: int | None = None) -> None:
    synth = await _get_synthesizer()
    await _warmup(synth)

    if ctrl_port:
        await _run_control_api(ctrl_port)

    initial_ports = [base_port + i for i in range(n_ports)]
    for p in initial_ports:
        await _bind_ws_port(p)

    await _warmup_all_ports(initial_ports)

    print(f"\n[{_ts()}] Qwen3TTS  {len(_open_ports)} port(s) ready:", flush=True)
    for p in sorted(_open_ports):
        print(f"  ws://0.0.0.0:{p}", flush=True)
    print(flush=True)

    await asyncio.Future()  # run forever


def main() -> None:
    global _audio_out_dir

    parser = argparse.ArgumentParser(description="Qwen3TTS single-process WebSocket server")
    parser.add_argument("--base-port", type=int, default=settings.ws.port,
                        help=f"First port to bind (default: {settings.ws.port})")
    parser.add_argument("--ports", type=int, default=1,
                        help="Number of WebSocket ports to open (default: 1)")
    parser.add_argument("--save-audio", type=str, default=None, metavar="DIR",
                        help="Directory to save decoded WAV files (one per request)")
    parser.add_argument("--ctrl-port", type=int, default=None, metavar="PORT",
                        help="HTTP control API port for on-demand WS port binding")
    args = parser.parse_args()

    global _llm_log_file, _llm_out_log_file
    _llm_log_file = open(_llm_log, "w", buffering=1)
    _llm_out_log.parent.mkdir(parents=True, exist_ok=True)
    _llm_out_log_file = open(_llm_out_log, "a", buffering=1)

    if args.save_audio:
        _audio_out_dir = Path(args.save_audio)
        _audio_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Qwen3TTS] Saving audio to {_audio_out_dir}/", flush=True)

    try:
        asyncio.run(run_server(args.base_port, args.ports, args.ctrl_port))
    except KeyboardInterrupt:
        print("\n[Qwen3TTS] Stopped.")


if __name__ == "__main__":
    main()
