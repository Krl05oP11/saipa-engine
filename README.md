# saipa-engine

FastAPI service that does the AI work for the SAIPA / EVAL-IA Moodle plugins:
role-aware course chat (RAG), dropout-risk scoring, exam rubric/question
generation and grading, and the Telegram channel.

- **Port:** 8052
- **Consumed by:** [`local_saipa`](https://github.com/Krl05oP11/moodle-local_saipa),
  [`block_saipa`](https://github.com/Krl05oP11/moodle-block_saipa),
  [`local_evalia`](https://github.com/Krl05oP11/moodle-local_evalia)
- **License:** proprietary (Schaller & Ponce). Not for redistribution.

## Endpoints (routers mounted in `main.py`)

| Router | Paths |
|---|---|
| health | `GET /health` |
| chat | `POST /chat`, `POST /chat/advisor` |
| index | `POST /index`, `POST /index-pdf-bytes`, `POST /index-pptx-bytes` |
| analytics | `POST /analytics/risk`, `POST /analytics/risk/batch` |
| eval | `POST /eval/{rubric,questions,exam,portfolio,feedback}/…`, `GET /eval/sources/{course_id}` |
| telegram | `POST /telegram/webhook`, `POST /telegram/notify`, `GET /telegram/status` |

## Run

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in the values — SAIPA_API_TOKEN is REQUIRED
uvicorn main:app --host 0.0.0.0 --port 8052
```

Or with Docker:

```bash
docker build -t saipa-engine .
docker run --env-file .env -p 8052:8052 saipa-engine
```

## Configuration

All settings come from environment variables / `.env` (see `.env.example`).
Key ones:

- `SAIPA_API_TOKEN` — **required**. Bearer token Moodle must send. The engine
  refuses to start with an empty token unless `SAIPA_DEV_MODE=true` (local dev only).
- `SAIPA_LICENSE_KEY` — license validation at startup (30-day offline grace).
- `MOODLE_URL` / `MOODLE_TOKEN` — Moodle web-services endpoint.
- `LLM_PROVIDER` — **required**, `ollama` or `anthropic` (this is the university
  admin's decision — who to contract and pay for the model, not ours). The
  engine refuses to start with it unset unless `SAIPA_DEV_MODE=true` (falls
  back to `ollama`).
  - `ollama`: `OLLAMA_BASE_URL` / `OLLAMA_MODEL`. Self-hosted, no API key.
  - `anthropic`: `ANTHROPIC_API_KEY` — **required** if this provider is chosen
    (same fail-closed rule as above). `ANTHROPIC_MODEL_CHAT` /
    `ANTHROPIC_MODEL_GRADING` — recommended defaults (Haiku for SAIPA
    chat/RAG/alerts, Sonnet+ for EVAL-IA grading — a weak model gives
    defensible-looking wrong scores on free-text answers), override if needed.
- Embeddings are always `sentence-transformers` (CPU, no GPU or extra API key —
  `SENTENCE_TRANSFORMERS_MODEL`, defaults to a small multilingual model). Not
  configurable per install like the LLM — there's no admin decision here, it
  just runs.
  It's loaded into process memory and warmed up at startup (see `main.py`) —
  with `--workers N` (the multi-worker example above), each worker process
  loads its own independent copy, so memory use and restart time scale with
  worker count. Consider `--workers 1` unless you've sized the instance for
  N copies of the model.
  **Switching this model (or upgrading it later) requires deleting and
  recreating every course's ChromaDB collection** — ChromaDB hard-rejects
  adding a different-dimension vector to an existing collection
  (`InvalidDimensionException`), it doesn't just degrade silently. Re-index
  every course from Moodle afterward.
- `CHROMA_PERSIST_DIR` — vector store location.
- `TELEGRAM_BOT_TOKEN` (+ webhook vars) — leave empty to disable the Telegram channel.

## Notes

- The dropout-risk model (`analytics/risk_model.py`) is a **deterministic
  rule-based heuristic** over 11 engagement signals — not a trained ML model.
  Feature vectors are computed by `local_saipa` and sent in the request.
- WhatsApp provider wiring is currently inconsistent across the codebase
  (Twilio / Evolution API) — pick one before enabling it.
- Moodle blocks RFC1918/loopback hosts (`curlsecurityblockedhosts`) by
  default — affects any install where this engine sits on a private IP,
  which is the recommended topology. `local_saipa`/`local_evalia` already
  ship a narrow allowlist-of-one-host fix (`engine_security_helper`); if you
  install the engine some other way, you'll need the equivalent on the
  Moodle side.

## Tests

```bash
pip install pytest
pytest
```
