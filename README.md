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
| index | `POST /index` |
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
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` — LLM + embeddings.
- `CHROMA_PERSIST_DIR` — vector store location.
- `TELEGRAM_BOT_TOKEN` (+ webhook vars) — leave empty to disable the Telegram channel.

## Notes

- The dropout-risk model (`analytics/risk_model.py`) is a **deterministic
  rule-based heuristic** over 11 engagement signals — not a trained ML model.
  Feature vectors are computed by `local_saipa` and sent in the request.
- WhatsApp provider wiring is currently inconsistent across the codebase
  (Twilio / Evolution API) — pick one before enabling it.

## Tests

```bash
pip install pytest
pytest
```
