"""
Telegram Bot webhook router for SAIPA.

Endpoints:
  POST /telegram/webhook  — receives updates from Telegram (registered via setWebhook)
  POST /telegram/notify   — protected endpoint for Moodle cron to push alerts
  GET  /telegram/status   — health check (is bot connected?)

Flow:
  /start <token>  → linking handshake via Moodle WS
  /help           → usage instructions
  /desconectar    → unlink account
  free text       → RAG+LLM response (same pipeline as web chat)
"""
import asyncio
import logging
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from config import get_settings
from rag.retriever import retrieve
from telegram.client import get_telegram_client
from telegram.moodle import confirm_link, get_session, mark_alert_responded, unlink_by_telegram_id

log    = logging.getLogger("saipa_engine.telegram")
router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# In-memory conversation history: telegram_id → list[{role, content}]
_sessions: dict[int, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = (
    "Eres SAIPA, un asistente pedagógico inteligente diseñado para acompañar a estudiantes "
    "universitarios. Respondés siempre en español de manera clara, cálida y motivadora. "
    "Tu rol es el de un tutor que guía al estudiante, lo acompaña en su proceso de aprendizaje "
    "y lo ayuda a no sentirse solo. Si disponés de información del curso en el contexto "
    "a continuación, usala para fundamentar tus respuestas.\n\n"
    "{context}"
)

HELP_TEXT = (
    "🤖 <b>SAIPA — Asistente Pedagógico</b>\n\n"
    "Puedo ayudarte con preguntas sobre tu cursada, explicar conceptos y acompañarte en tu aprendizaje.\n\n"
    "<b>Comandos disponibles:</b>\n"
    "/help — Mostrar este mensaje\n"
    "/desconectar — Desvincular tu cuenta de Moodle\n\n"
    "Simplemente escribime tu pregunta y te responderé."
)


# ── Token auth dependency (for /telegram/notify) ─────────────────────────────

def _verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = get_settings()
    if not settings.saipa_api_token:
        return
    if credentials is None or credentials.credentials != settings.saipa_api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@router.post("/telegram/webhook", tags=["telegram"])
async def telegram_webhook(request: Request):
    """
    Receives all Telegram updates (POST from Telegram servers).
    Validates X-Telegram-Bot-Api-Secret-Token, dispatches async tasks,
    returns 200 immediately.
    """
    settings = get_settings()

    # Validate secret header
    if settings.telegram_webhook_secret:
        received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if received != settings.telegram_webhook_secret:
            log.warning("Telegram webhook received with invalid secret")
            return JSONResponse({"status": "forbidden"}, status_code=403)

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "bad_request"}, status_code=400)

    message = data.get("message")
    if not message:
        return {"status": "ignored", "reason": "no_message"}

    # Skip messages from bots
    from_user = message.get("from", {})
    if from_user.get("is_bot"):
        return {"status": "ignored", "reason": "from_bot"}

    chat_id  = int(message["chat"]["id"])
    text     = (message.get("text") or "").strip()
    username = from_user.get("username", "")

    if not text:
        return {"status": "ignored", "reason": "no_text"}

    log.info("Telegram incoming from chat_id=%s: %d chars", chat_id, len(text))

    if text.startswith("/start"):
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) > 1 else None
        asyncio.create_task(_handle_start(chat_id, username, token))
    elif text == "/help":
        asyncio.create_task(_send_help(chat_id))
    elif text in ("/desconectar", "/disconnect", "/unlink"):
        asyncio.create_task(_handle_unlink(chat_id))
    else:
        asyncio.create_task(_handle_message(chat_id, text))

    return {"status": "ok"}


async def _dispatch_update(update: dict) -> None:
    """
    Shared dispatcher used by both the webhook endpoint and the polling loop.
    Extracts message fields and calls the appropriate handler.
    """
    message = update.get("message")
    if not message:
        return

    from_user = message.get("from", {})
    if from_user.get("is_bot"):
        return

    chat_id  = int(message["chat"]["id"])
    text     = (message.get("text") or "").strip()
    username = from_user.get("username", "")

    if not text:
        return

    log.info("Telegram [poll] from chat_id=%s: %d chars", chat_id, len(text))

    if text.startswith("/start"):
        parts = text.split(maxsplit=1)
        token = parts[1].strip() if len(parts) > 1 else None
        await _handle_start(chat_id, username, token)
    elif text == "/help":
        await _send_help(chat_id)
    elif text in ("/desconectar", "/disconnect", "/unlink"):
        await _handle_unlink(chat_id)
    else:
        await _handle_message(chat_id, text)


# ── Notify endpoint (called by Moodle cron / risk_evaluation) ────────────────

class NotifyRequest(BaseModel):
    telegram_id: int
    message: str


@router.post("/telegram/notify", tags=["telegram"])
async def telegram_notify(req: NotifyRequest, _: None = Depends(_verify_token)):
    """
    Sends a proactive message to a student or teacher via Telegram.
    Called by Moodle's risk_evaluation cron task when a student escalates to high risk.
    """
    client = get_telegram_client()
    if not client.enabled():
        return {"sent": False, "reason": "telegram_disabled"}
    sent = await client.send_message(req.telegram_id, req.message)
    return {"sent": sent}


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/telegram/status", tags=["telegram"])
async def telegram_status(_: None = Depends(_verify_token)):
    """Returns Telegram bot connection status."""
    client = get_telegram_client()
    if not client.enabled():
        return {"status": "disabled", "detail": "TELEGRAM_BOT_TOKEN not configured"}
    me = await client.get_me()
    if me.get("ok"):
        bot = me.get("result", {})
        return {
            "status":   "ok",
            "username": bot.get("username"),
            "name":     bot.get("first_name"),
        }
    return {"status": "error", "detail": me.get("error")}


# ── Command handlers ──────────────────────────────────────────────────────────

async def _handle_start(chat_id: int, username: str, token: str | None) -> None:
    client = get_telegram_client()

    if not token:
        await client.send_message(
            chat_id,
            "👋 Hola. Para vincular tu cuenta de Moodle con SAIPA, "
            "hacé clic en el botón <b>Vincular Telegram</b> dentro del bloque SAIPA en tu curso."
        )
        return

    try:
        result = await confirm_link(token, chat_id, username)
    except RuntimeError as exc:
        log.warning("Telegram link failed for chat_id=%s: %s", chat_id, exc)
        await client.send_message(
            chat_id,
            "⚠️ El enlace no es válido o ya expiró. "
            "Generá uno nuevo desde el bloque SAIPA en Moodle."
        )
        return

    fullname = result.get("fullname", "")
    await client.send_message(
        chat_id,
        f"✅ ¡Listo, {fullname}! Tu cuenta de Moodle fue vinculada correctamente.\n\n"
        "Ahora podés escribirme desde Telegram para hacer preguntas sobre tus cursos. "
        "Escribí /help para ver los comandos disponibles."
    )
    log.info("Telegram account linked: chat_id=%s → user_id=%s", chat_id, result.get("user_id"))


async def _send_help(chat_id: int) -> None:
    await get_telegram_client().send_message(chat_id, HELP_TEXT)


async def _handle_unlink(chat_id: int) -> None:
    client = get_telegram_client()
    try:
        await unlink_by_telegram_id(chat_id)
        await client.send_message(
            chat_id,
            "✅ Tu cuenta de Moodle fue desvinculada. "
            "Para volver a vincularla, usá el bloque SAIPA en tu curso."
        )
        log.info("Telegram account unlinked: chat_id=%s", chat_id)
    except Exception as exc:
        log.warning("Unlink failed for chat_id=%s: %s", chat_id, exc)
        await client.send_message(chat_id, "⚠️ No pude desvincular la cuenta. Intentá más tarde.")


# ── Two-way chat handler ──────────────────────────────────────────────────────

async def _handle_message(chat_id: int, text: str) -> None:
    client   = get_telegram_client()
    settings = get_settings()

    # Resolve Moodle user + course from Telegram chat_id
    try:
        session = await get_session(chat_id)
    except Exception as exc:
        log.error("get_session failed for chat_id=%s: %s", chat_id, exc)
        await client.send_message(chat_id, "⚠️ Error interno. Intentá en unos minutos.")
        return

    if not session.get("linked"):
        await client.send_message(
            chat_id,
            "Para chatear conmigo primero tenés que vincular tu cuenta de Moodle. "
            "Buscá el bloque SAIPA en tu curso y hacé clic en <b>Vincular Telegram</b>."
        )
        return

    user_id   = session["user_id"]
    course_id = int(session.get("course_id") or 0)

    log.info("Telegram chat: chat_id=%s user_id=%s course_id=%s len=%d",
             chat_id, user_id, course_id, len(text))

    # Mark any pending teacher alert as responded (fire-and-forget, non-blocking).
    asyncio.create_task(mark_alert_responded(chat_id, int(__import__("time").time())))

    history = _sessions[chat_id]

    # RAG retrieval
    docs = retrieve(text, course_id, top_k=5)
    context_text = ""
    if docs:
        context_text = "Contexto del curso:\n" + "\n---\n".join(
            f"[{d['source']}]: {d['content']}" for d in docs
        )

    # Build LangChain messages
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context_text))]
    for h in history[-MAX_HISTORY_TURNS:]:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=text))

    # LLM call
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        request_timeout=float(settings.ollama_timeout),
    )
    try:
        response = await llm.ainvoke(messages)
        reply = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        log.error("LLM error for Telegram chat_id=%s: %s", chat_id, exc)
        await client.send_message(
            chat_id,
            "⚠️ El asistente no está disponible en este momento. Intentá en unos minutos."
        )
        return

    # Update in-memory history (cap to avoid unbounded growth)
    history.append({"role": "user",      "content": text})
    history.append({"role": "assistant", "content": reply})
    if len(history) > MAX_HISTORY_TURNS * 2:
        _sessions[chat_id] = history[-(MAX_HISTORY_TURNS * 2):]

    log.info("Telegram reply: chat_id=%s len=%d sources=%d", chat_id, len(reply), len(docs))
    await client.send_message(chat_id, reply)
