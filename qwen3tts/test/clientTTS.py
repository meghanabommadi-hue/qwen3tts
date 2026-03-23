#!/usr/bin/env python3
"""Simple Qwen3TTS WebSocket test client.

Usage:
    python qwen3tts/test/clientTTS.py --text "Hello, how are you?"
    python qwen3tts/test/clientTTS.py --port 9765 --text "Your account balance is ready."
    python qwen3tts/test/clientTTS.py --streaming  # use streaming mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import websockets


async def run_client(host: str, port: int, text: str, streaming: bool, save_wav: str | None) -> None:
    url = f"ws://{host}:{port}/ws/test_client"
    call_id = "test_client"
    text_id = str(uuid.uuid4())

    print(f"Connecting to {url}", flush=True)
    print(f"Text: {text!r}", flush=True)
    print(f"Mode: {'streaming' if streaming else 'non-streaming'}\n", flush=True)

    async with websockets.connect(url, max_size=100 * 1024 * 1024, open_timeout=10) as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "type":      "synthesize",
            "call_id":   call_id,
            "text_id":   text_id,
            "text":      text,
            "streaming": streaming,
        }))
        print(f"Sent request ({time.perf_counter()-t0:.3f}s)", flush=True)

        wav_chunks: list[bytes] = []
        total_tokens = 0

        if streaming:
            # Receive chunks until audio_done
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    wav_chunks.append(raw)
                    print(f"  Received WAV chunk: {len(raw)} bytes", flush=True)
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "audio_chunk":
                    print(
                        f"  chunk {msg.get('chunk_index')}  "
                        f"tokens={msg.get('tokens')}  "
                        f"wav={msg.get('wav_bytes')}B  "
                        f"final={msg.get('is_final')}",
                        flush=True,
                    )
                    total_tokens += msg.get("tokens", 0)
                elif msg_type == "audio_done":
                    elapsed = time.perf_counter() - t0
                    print(f"\nDone!  chunks={msg.get('chunks')}  total_tokens={msg.get('total_tokens')}  "
                          f"llm={msg.get('llm_s')}s  rtf={msg.get('rtf')}  "
                          f"wall={elapsed:.3f}s", flush=True)
                    break
                elif msg_type == "error":
                    print(f"ERROR: {msg.get('error')}", flush=True)
                    return
        else:
            # Non-streaming: JSON frame + binary WAV frame
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("type") == "error":
                print(f"ERROR: {msg.get('error')}", flush=True)
                return

            elapsed_json = time.perf_counter() - t0
            total_tokens = msg.get("audio_tokens", "").count("<|speech_token_")
            print(f"Metadata received ({elapsed_json:.3f}s)  tokens={total_tokens}  "
                  f"llm={msg.get('llm_s')}s  decode={msg.get('decode_s')}s", flush=True)

            wav_data = await ws.recv()
            wav_chunks.append(wav_data)
            elapsed_wav = time.perf_counter() - t0
            print(f"WAV received ({elapsed_wav:.3f}s)  {len(wav_data)} bytes", flush=True)

        # Save WAV
        if save_wav and wav_chunks:
            all_wav = b"".join(wav_chunks)
            Path(save_wav).write_bytes(all_wav)
            print(f"\nSaved {len(all_wav)} bytes → {save_wav}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3TTS test client")
    parser.add_argument("--host",      default="localhost")
    parser.add_argument("--port",      type=int, default=9765)
    parser.add_argument("--text",      default="Hello, how can I assist you today?")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="streaming", action="store_false")
    parser.add_argument("--save-wav",  default=None, metavar="FILE",
                        help="Save output audio to a WAV file")
    args = parser.parse_args()

    asyncio.run(run_client(args.host, args.port, args.text, args.streaming, args.save_wav))


if __name__ == "__main__":
    main()
