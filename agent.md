# Qwen3TTS Agent Context

## What This Project Is
Qwen3TTS is a production-grade WebSocket TTS (text-to-speech) gateway powered by Qwen3 + neural codec, modelled exactly after FlowTTS (`/root/FlowTTS`). It supports 100+ concurrent ports, streaming audio chunks, Prometheus metrics, and both single-process and Redis-backed multi-process modes.

## Architecture
```
Client (WebSocket)
  │  text → JSON {type, call_id, text_id, text}
  ▼
server.py  (single-process, recommended)
  │  normalize_text()
  │  Qwen3TtsSynthesizer.synthesize_stream(text)
  │    └─ sglang Engine (Qwen3 model) → speech token string
  │  TTSCodec.decode_async(tokens, ctx) → PCM tensor
  │  tensor_to_wav() → WAV bytes
  ▼
Client  (JSON metadata + binary WAV, streamed in chunks)
```

## Deployment Modes
- **server.py** (recommended): single process, no Redis, one GPU load, all ports share one sglang Engine
- **main.py**: Redis-backed gateway + separate worker process(es) for horizontal scaling

## Port Model
- Default base port: **9765** (vs FlowTTS 8765)
- `--ports N` opens N consecutive ports from `--base-port`
- All ports share the same `Qwen3TtsSynthesizer` singleton

## Environment Variables
All settings overridable with `QWEN3TTS_` prefix, e.g.:
- `QWEN3TTS_TTS_MODEL__MODEL_DIR=/path/to/Qwen3-TTS`
- `QWEN3TTS_TTS_MODEL__REF_AUDIO=/path/to/ref.wav`
- `QWEN3TTS_DECODER__ENABLED=true`
- `QWEN3TTS_STREAMING__CHUNK_TOKENS=30`

## Current Build Status

### Completed Files
| File | Status | Notes |
|------|--------|-------|
| `qwen3tts/__init__.py` | ✅ | Package docstring / overview |
| `qwen3tts/core/__init__.py` | ✅ | |
| `qwen3tts/core/config.py` | ✅ | Pydantic settings, QWEN3TTS_ prefix, port 9765 |
| `qwen3tts/synthesis/__init__.py` | ✅ | |
| `qwen3tts/synthesis/models.py` | ✅ | Qwen3TtsSynthesizer: sglang + codec init, synthesize, synthesize_stream |
| `qwen3tts/synthesis/engine.py` | ✅ | SynthesisService singleton |
| `qwen3tts/api/__init__.py` | ✅ | |
| `qwen3tts/decoder/__init__.py` | ✅ | |
| `qwen3tts/decoder/ncodec/__init__.py` | ✅ | |
| `qwen3tts/processing/__init__.py` | ✅ | |
| `qwen3tts/monitoring/__init__.py` | ✅ | |
| `qwen3tts/test/__init__.py` | ✅ | |
| `qwen3tts/setup/__init__.py` | ✅ | |

### All Files — Completed
| File | Status | Notes |
|------|--------|-------|
| `qwen3tts/decoder/decoder.py` | ✅ | tensor_to_wav, DecodedAudio, SAMPLE_RATE=16000 |
| `qwen3tts/decoder/ncodec/codec.py` | ✅ | TTSCodec: batched decode queue, ThreadPoolExecutor, format_prompt, encode stub |
| `qwen3tts/api/models.py` | ✅ | SynthesizeRequest, AudioMessage, ErrorMessage Pydantic schemas |
| `qwen3tts/api/websockets.py` | ✅ | Redis-backed ConnectionManager + WebSocket endpoint |
| `qwen3tts/monitoring/metrics.py` | ✅ | Prometheus counters (qwen3tts_* prefix) + JSONL call log |
| `qwen3tts/monitoring/logging.py` | ✅ | structlog configure_logging() |
| `qwen3tts/processing/audio_processing.py` | ✅ | crossfade, fade_out, resample_audio, process_audio_pipeline |
| `qwen3tts/processing/text_normalize.py` | ✅ | normalize_text() — English-first, currency + number expansion, abbreviations |
| `qwen3tts/server.py` | ✅ | Single-process WS gateway: streaming + non-streaming, RTF, warmup, control API |
| `qwen3tts/main.py` | ✅ | Redis-backed multi-process gateway (FastAPI + CORS) |
| `qwen3tts/worker.py` | ✅ | Redis queue consumer + SynthesizerWorker (concurrent) |
| `qwen3tts/test/benchmark.py` | ✅ | Load test: RTF + p50/p95/p99 latencies, token stream log, per-port stats |
| `qwen3tts/test/clientTTS.py` | ✅ | Simple WS test client (streaming + non-streaming, --save-wav) |
| `qwen3tts/setup/download_models.py` | ✅ | HuggingFace snapshot_download for Qwen/Qwen3-TTS |
| `requirements.txt` | ✅ | Pinned deps (torch, sglang, websockets, aiohttp, pydantic, structlog, prometheus-client, …) |
| `run.sh` | ✅ | `python -m qwen3tts.server "$@"` launcher |
| `agent.md` | ✅ | This file — updated on every code change |
| `commands.md` | ✅ | All CLI commands: launch, benchmark, client, metrics, tuning, model download |
| `qwen3tts/setup/setup.sh` | ✅ | Full install: torch → sglang==0.5.2 → flashinfer → transformers==4.57.3 → transformers@git HEAD → deps → patch DeepseekVL2Config (ClassVar fix) → pyc cache clear → model download |

## Key Differences vs FlowTTS
| Aspect | FlowTTS | Qwen3TTS |
|--------|---------|----------|
| Model | MiraTTS (Hindi/Telugu) | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| Default port | 8765 | 9765 |
| Env prefix | `FLOWTTS_` | `QWEN3TTS_` |
| Package name | `flowtts` | `qwen3tts` |
| Text normalize | Hindi/English | English-first, extensible |
| Warmup sentence | Hindi | English |
| Class name | FlowTtsSynthesizer | Qwen3TtsSynthesizer |

## Data Flow (Request Lifecycle)
1. Client sends `{"type":"synthesize","call_id":"…","text_id":"…","text":"Hello world"}`
2. `normalize_text()` → cleans punctuation, expands numbers
3. `Qwen3TtsSynthesizer.synthesize_stream(text)` → async generator of speech token deltas
4. Tokens accumulated in rolling buffer; every `chunk_tokens` (default 30) tokens:
   - `TTSCodec.decode_async(chunk_tokens, context)` → PCM tensor
   - `tensor_to_wav(pcm)` → WAV bytes
   - Send `{type:"audio_chunk", chunk_index, tokens, is_final}` + raw WAV bytes
5. After EOS: send `{type:"audio_done", total_tokens, llm_s, decode_s, rtf}`

## Metrics Tracked
- `tts_requests_total` — successful requests
- `tts_llm_ms_total` — cumulative LLM inference time
- `tts_decode_ms_total` — cumulative codec decode time
- `tts_e2e_ms_total` — cumulative end-to-end time
- `tts_tokens_total` — cumulative speech tokens generated
- `tts_active_websockets` — live connections gauge
- `tts_open_ports` — currently bound WS ports
- RTF (real-time factor) — per-request and rolling average
- `monitoring/calls.jsonl` — append-only JSONL call log
