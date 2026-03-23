# Qwen3TTS Commands

## Launch server

```bash
cd /root/Qwen3TTS && bash run.sh --ctrl-port 9764
```

## Open N ports

```bash
python3 -m qwen3tts.server --ports 40 --ctrl-port 9764
```

Or via the control API while the server is already running:
```bash
# Add one port dynamically
curl -X POST "http://127.0.0.1:9764/ports/add?port=9805"

# List all currently open ports
curl http://127.0.0.1:9764/ports
```

---

## Run benchmark (N requests across live ports)

```bash
# Medium sentences, 3 requests per port, scan default range
python3 -m qwen3tts.test.benchmark --base-port 9765 --n-ports 40 --requests 3

# Total 100 requests distributed round-robin
python3 -m qwen3tts.test.benchmark --total 100 --n-ports 40

# Short sentences — pure throughput test
python3 -m qwen3tts.test.benchmark --total 100 --n-ports 40 --length short

# Long sentences — stress token generation
python3 -m qwen3tts.test.benchmark --requests 5 --n-ports 40 --length long

# Auto-discover live ports from /ports endpoint
python3 -m qwen3tts.test.benchmark --auto-ports --requests 3

# Don't save JSON output files, measure timing only
python3 -m qwen3tts.test.benchmark --total 50 --no-save
```

- Each port gets a **persistent WebSocket connection** (phone-call model)
- All ports run **concurrently**; requests within a port are sequential
- Reports: throughput (req/s), avg/min/max/std/p50/p95/p99 for total and LLM latency, RTF per port
- JSON outputs saved to `test/bench_YYYYMMDD_HHMMSS/`
- Token stream timeline saved to `test/llm_streaming_benchmark.log`

---

## Send a single test request

```bash
# Streaming (default) — receives audio chunks as they are generated
python3 -m qwen3tts.test.clientTTS --text "Hello, how can I help you today?"

# Non-streaming — waits for full audio before receiving
python3 -m qwen3tts.test.clientTTS --no-stream --text "Your payment has been received."

# Custom port
python3 -m qwen3tts.test.clientTTS --port 9766 --text "Please hold on for a moment."

# Save output audio to a WAV file
python3 -m qwen3tts.test.clientTTS --text "Thank you for calling." --save-wav /tmp/out.wav
```

---

## Check open ports

```bash
# Via control API
curl http://127.0.0.1:9764/ports

# Via TCP scan (requires ss)
ss -tlnp | grep python3 | awk '{print $4}' | sort -t: -k2 -n

# Readiness check (200 once model is loaded)
curl http://127.0.0.1:9764/ready
```

---

## Prometheus metrics

```bash
curl http://127.0.0.1:9764/metrics
```

Key metrics:
- `qwen3tts_requests_total` — successful TTS calls
- `qwen3tts_llm_ms_total` — cumulative LLM inference time
- `qwen3tts_decode_ms_total` — cumulative codec decode time
- `qwen3tts_e2e_ms_total` — cumulative end-to-end time
- `qwen3tts_tokens_total` — cumulative speech tokens generated
- `qwen3tts_active_websockets` — live connections (gauge)
- `qwen3tts_open_ports` — currently bound WS ports (gauge)
- `qwen3tts_errors_total` — failed requests

---

## View recent WebSocket events

```bash
curl http://127.0.0.1:9764/ws/log
```

Returns last 20 events (open / done / error / close) with per-milestone timestamps.

---

## Kill server

```bash
kill $(ss -tlnp | grep :9764 | grep -oP 'pid=\K[0-9]+')
```

Or by PID directly:
```bash
kill -9 <pid>
```

---

## Enable decoder (tokens → WAV in-process)

By default the server returns raw speech tokens only (decoder disabled).
To enable the neural codec decoder:

```bash
# Via env var
QWEN3TTS_DECODER__ENABLED=true bash run.sh --ports 10 --ctrl-port 9764

# Or edit qwen3tts/core/config.py → DecoderSettings → enabled: bool = True
```

---

## Enable TensorRT decoder (faster, first run ~60s compile)

```bash
QWEN3TTS_DECODER__USE_TRT=true bash run.sh --ports 1 --ctrl-port 9764
```

- Engine cached to disk next to model weights after first compile
- Subsequent starts load cache instantly

---

## Tuning streaming chunk size

Smaller chunks = lower latency to first audio; larger = fewer round-trips.

```bash
# 20 tokens per chunk (~400ms audio @ 16kHz)
QWEN3TTS_STREAMING__CHUNK_TOKENS=20 bash run.sh --ports 10

# Disable crossfade (raw chunks, no blending)
QWEN3TTS_STREAMING__CROSSFADE_SAMPLES=0 bash run.sh --ports 10

# All streaming settings at once
QWEN3TTS_STREAMING__CHUNK_TOKENS=20 \
QWEN3TTS_STREAMING__CROSSFADE_SAMPLES=0 \
QWEN3TTS_STREAMING__FADE_OUT_SAMPLES=0 \
bash run.sh --ports 10 --ctrl-port 9764
```

---

## First-time setup

```bash
# Full install: deps → sglang → flashinfer → model download
bash qwen3tts/setup/setup.sh

# Skip model download (if already downloaded)
bash qwen3tts/setup/setup.sh --skip-model

# Custom model directory
bash qwen3tts/setup/setup.sh --model-dir /path/to/models/Qwen3-TTS
```

## Download model weights only

```bash
# Default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
python3 -m qwen3tts.setup.download_models

# Available repo IDs:
#   Qwen/Qwen3-TTS-12Hz-1.7B-Base          (default, general purpose)
#   Qwen/Qwen3-TTS-12Hz-0.6B-Base          (smaller / faster)
#   Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   (custom voice cloning)
#   Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   (voice design)
python3 -m qwen3tts.setup.download_models --repo-id Qwen/Qwen3-TTS-12Hz-0.6B-Base
python3 -m qwen3tts.setup.download_models --model-dir ~/models/Qwen3-TTS --repo-id Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

---

## Multi-process mode (Redis-backed)

Only needed for horizontal scaling across machines.

```bash
# Terminal 1 — gateway
uvicorn qwen3tts.main:app --host 0.0.0.0 --port 9765

# Terminal 2 — worker (GPU inference)
python3 -m qwen3tts.worker
```

Set known ports for /ports discovery:
```bash
QWEN3TTS_KNOWN_PORTS=9765,9766,9767 uvicorn qwen3tts.main:app --port 9765
```

---

## Notes

- Ctrl API runs on `127.0.0.1:9764` (when `--ctrl-port 9764` is passed)
- WS ports start at `9765` by default (override with `--base-port`)
- All requests run fully parallel — sglang batches LLM, TTSCodec batches decode
- WAV outputs (when `--save-audio` is set) go to the specified directory, one file per `text_id`
- Call log appended to `monitoring/calls.jsonl` (one JSON line per completed call)
- LLM token outputs appended to `monitoring/llm_outputs.jsonl`
- Decoder config: `DecoderSettings` (`max_batch`, `gpu_chunk_size`, `onnx_workers`, `use_trt`)
  - Override via env: `QWEN3TTS_DECODER__MAX_BATCH=64`, `QWEN3TTS_DECODER__GPU_CHUNK_SIZE=120`
- Streaming config: `StreamingSettings` (`chunk_tokens`, `crossfade_samples`, `fade_out_samples`)
  - Override via env: `QWEN3TTS_STREAMING__CHUNK_TOKENS=30`
