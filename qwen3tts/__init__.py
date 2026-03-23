"""Qwen3TTS — single-process WebSocket TTS gateway powered by Qwen3 + neural codec.

Pipeline overview:
  1. Client connects via WebSocket on port 9765…9765+N.
  2. Client sends: {"type": "synthesize", "call_id": "…", "text_id": "…", "text": "…"}
  3. server.py normalises text → formats prompt → runs sglang (Qwen3) inference
     → yields speech token string: "<|speech_token_42|><|speech_token_7|>…"
  4. speech tokens → batched neural codec (TTSCodec) → PCM tensor → WAV bytes
  5. Server sends JSON metadata frame + binary WAV frame back to client.

Streaming mode (default):
  Every CHUNK_TOKENS speech tokens the partial audio is decoded and sent as
  an audio_chunk frame, with a final audio_done frame after all tokens.

Two deployment modes:
  server.py  (RECOMMENDED) — single process, no Redis, one GPU load, 100+ ports.
  main.py                  — Redis-backed multi-process for horizontal scaling.

Run:
  python -m qwen3tts.server --ports 100
"""
