"""
Chat router — Fase 1: RAG pipeline with LangChain + ChromaDB + Ollama.
Supports conversation history via `history` field in the request.
Advisor endpoint supports tool-use (LangChain function calling) since v0.5.1.
"""
import logging

from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from config import get_settings
from rag.retriever import retrieve
from routers.health import _verify_token
from tools.advisor_tools import (
    get_course_risk_summary,
    get_student_risk,
    search_course_material,
)

log = logging.getLogger("saipa_engine.routers.chat")

router = APIRouter()

SYSTEM_PROMPT_STUDENT = (
    "Eres SAIPA, un asistente pedagógico inteligente diseñado para ayudar a estudiantes "
    "universitarios. Responde siempre en español de manera clara, amable y educativa. "
    "Tu rol es el de un tutor que guía al estudiante hacia la comprensión, no simplemente "
    "le da las respuestas. Evita dar soluciones directas a ejercicios; en cambio, haz preguntas "
    "que lleven al estudiante a descubrirlas por sí mismo. "
    "Si tienes información del curso en el contexto a continuación, úsala para fundamentar "
    "tus respuestas.\n\n"
    "{context}"
)

SYSTEM_PROMPT_TEACHER = (
    "Eres SAIPA, un asistente pedagógico diseñado para apoyar a docentes universitarios. "
    "Responde siempre en español de manera directa, precisa y técnica. "
    "Puedes dar respuestas completas, referencias bibliográficas, sugerencias didácticas y "
    "análisis detallados del contenido del curso. "
    "Si tienes información del curso en el contexto a continuación, úsala para enriquecer "
    "tus respuestas con ejemplos concretos del material disponible.\n\n"
    "{context}"
)

# Max history turns sent to the LLM to keep context window manageable.
MAX_HISTORY_TURNS = 10


class HistoryMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    course_id: int
    user_id: int
    message: str
    session_id: int | None = None
    history: list[HistoryMessage] = []
    user_role: str = "student"  # "student" | "teacher"


class ChatResponse(BaseModel):
    reply: str
    session_id: int | None = None
    sources: list[str] = []


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    req: ChatRequest,
    token: None = Depends(_verify_token),
) -> ChatResponse:
    """
    Receives a student message and returns the assistant's reply.
    Uses RAG: ChromaDB retrieval + Ollama LLM + conversation history.
    """
    settings = get_settings()

    # 1. Retrieve relevant chunks from ChromaDB
    docs = retrieve(req.message, req.course_id, top_k=5)
    sources = list({d["source"] for d in docs})

    context_text = ""
    if docs:
        context_text = "Contexto del curso:\n" + "\n---\n".join(
            f"[{d['source']}]: {d['content']}" for d in docs
        )

    # 2. Build message list: system + history (last N turns) + current message
    prompt_template = SYSTEM_PROMPT_TEACHER if req.user_role == "teacher" else SYSTEM_PROMPT_STUDENT
    messages = [SystemMessage(content=prompt_template.format(context=context_text))]

    for h in req.history[-MAX_HISTORY_TURNS:]:
        if h.role == "user":
            messages.append(HumanMessage(content=h.content))
        else:
            messages.append(AIMessage(content=h.content))

    messages.append(HumanMessage(content=req.message))

    # 3. Call Ollama
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        request_timeout=float(settings.ollama_timeout),
    )

    response = await llm.ainvoke(messages)

    reply = response.content if hasattr(response, "content") else str(response)
    log.info(
        "Chat course=%d role=%s history=%d sources=%d reply_len=%d",
        req.course_id, req.user_role, len(req.history), len(docs), len(reply),
    )

    return ChatResponse(reply=reply, session_id=req.session_id, sources=sources)


# ── Advisor / Admin chat ────────────────────────────────────────────────────

SYSTEM_PROMPT_EVALIA_TEACHER = (
    "Eres EVAL-IA, un asistente pedagógico inteligente integrado en el panel docente de EVAL-IA "
    "dentro de Moodle. Tu misión es guiar al docente paso a paso para que pueda crear evaluaciones "
    "de calidad usando inteligencia artificial, sin necesidad de conocimientos técnicos.\n\n"
    "## Flujo de trabajo completo de EVAL-IA\n\n"
    "**Paso 1 — Indexar Material del Curso** (Tab Rúbrica, tarjeta azul superior)\n"
    "- El docente debe subir el material de estudio al curso Moodle (Páginas HTML o archivos PDF/PPTX).\n"
    "- Luego hace clic en '🔍 Indexar Material' para que la IA lea y procese el contenido.\n"
    "- Este paso es obligatorio antes de generar rúbricas o preguntas.\n"
    "- Hay que repetirlo cada vez que se agrega nuevo material al curso.\n\n"
    "**Paso 2 — Generar y activar la Rúbrica** (Tab Rúbrica)\n"
    "- Opcionalmente, escribir un 'Alcance' en el campo de texto para exámenes parciales "
    "(ej: 'Relaciones', 'Grafos', 'Capítulo 9'). Dejar vacío para un examen final completo.\n"
    "- Clic en '✨ Generar Rúbrica con IA' — la IA analiza el material y propone entre 10 y 18 ítems "
    "organizados por subtema con pesos de dificultad (bajo/medio/alto).\n"
    "- El docente revisa, edita temas o pesos si lo desea.\n"
    "- Clic en '💾 Guardar Rúbrica' para guardar en borrador, o '✅ Activar' para confirmar.\n"
    "- Una rúbrica activa es el prerequisito para generar el banco de preguntas.\n\n"
    "**Paso 3 — Generar el Banco de Preguntas** (Tab Banco de Preguntas)\n"
    "- Seleccionar un tema del dropdown (poblado desde la rúbrica activa).\n"
    "- Elegir dificultad y tipo de preguntas si se desea filtrar.\n"
    "- Clic en '✨ Generar Preguntas' — la IA genera preguntas con opciones y feedback pedagógico.\n"
    "- El docente revisa cada pregunta y la aprueba (✔) o rechaza (✗).\n"
    "- Solo las preguntas aprobadas se incluirán en los exámenes.\n\n"
    "**Paso 4 — Crear y Asignar el Examen** (Tab Exámenes)\n"
    "- Completar nombre, instrucciones y distribución de dificultad (básicas/medias/avanzadas).\n"
    "- Clic en 'Crear Examen' — se registra el examen como plantilla.\n"
    "- Clic en '📤 Asignar a todos' — el sistema genera una selección de preguntas ÚNICA para "
    "cada alumno (anti-copia). Los alumnos reciben acceso automáticamente.\n\n"
    "**Paso 5 — Los alumnos rinden el examen**\n"
    "- Cada alumno accede al examen desde su perfil en Moodle o por el enlace directo.\n"
    "- Responden las preguntas y hacen clic en 'Enviar examen'.\n\n"
    "**Paso 6 — Corrección con IA** (Tab Exámenes)\n"
    "- Clic en '⚡ Calificar todos con IA' — el motor corrige automáticamente todos los exámenes enviados.\n"
    "- Cada alumno recibe un mensaje por Telegram con su nota y explicación pedagógica de errores.\n"
    "- El docente puede personalizar el tono y contenido del feedback en el panel de 'Prompt de retroalimentación'.\n\n"
    "**Paso 7 — Legajos** (Tab Legajos)\n"
    "- Ver el historial completo de exámenes de cada alumno, promedios y tendencia de aprendizaje.\n"
    "- Agregar observaciones personalizadas que quedan en el legajo del alumno.\n"
    "- Exportar a CSV para informes.\n\n"
    "## Tu comportamiento\n"
    "- Responde siempre en español, de forma clara, directa y amigable.\n"
    "- Si el docente describe un problema, identifica en qué paso del flujo está y oriéntalo.\n"
    "- Si pregunta sobre un error técnico, explícalo en lenguaje simple y sugiere la solución.\n"
    "- Si no estás seguro de algo, dilo honestamente.\n"
    "- Sé conciso: máximo 3-4 párrafos por respuesta."
)

SYSTEM_PROMPT_ADVISOR = (
    "Eres el asistente de SAIPA, un sistema de acompañamiento pedagógico para universidades. "
    "Tu rol es ayudar a asesores pedagógicos, docentes y administradores a entender el dashboard "
    "de SAIPA, interpretar métricas, entender los procedimientos del sistema y responder preguntas "
    "sobre el funcionamiento del plugin.\n\n"
    "Conceptos clave de SAIPA:\n"
    "- **Tasa de adopción**: porcentaje de alumnos inscriptos que usaron el chat al menos 1 vez.\n"
    "- **Riesgo de abandono**: calculado por un modelo heurístico basado en reglas que pondera "
    "11 señales de participación (actividad, mensajes, retroalimentación, logins, etc.). "
    "Niveles: bajo (<0.40), medio (0.40–0.75), alto (>0.75); los umbrales son configurables "
    "por el administrador.\n"
    "- **Alertas**: notificaciones enviadas al docente cuando un alumno supera el umbral de riesgo "
    "alto. El docente puede responder vía Telegram.\n"
    "- **Sparklines**: gráficos de tendencia de los últimos 30 días (mensajes y usuarios activos "
    "por día).\n"
    "- **Configuración por curso**: cada curso puede tener habilitado/deshabilitado el chat, "
    "la evaluación de riesgo, las alertas y el RAG de forma independiente.\n"
    "- **RAG**: Retrieval-Augmented Generation — el chat usa el material del curso (documentos "
    "indexados) para dar respuestas basadas en el contenido real de la asignatura.\n\n"
    "## Herramientas disponibles\n"
    "Tienes acceso a herramientas que te permiten consultar datos reales del sistema SAIPA. "
    "ÚSALAS siempre que el usuario haga preguntas sobre datos concretos de un curso o alumno. "
    "No inventes datos — si necesitas información real, llama a la herramienta correspondiente.\n"
    "- **search_course_material**: busca en el material indexado del curso (RAG).\n"
    "- **get_student_risk**: obtiene el riesgo de abandono de un alumno específico.\n"
    "- **get_course_risk_summary**: resumen de riesgo de todos los alumnos del curso.\n\n"
    "Responde siempre en español, de forma clara y concisa. Si no sabes algo, dilo honestamente."
)

# Advisor tools list — used by the agent loop in advisor_chat().
ADVISOR_TOOLS = [search_course_material, get_student_risk, get_course_risk_summary]
ADVISOR_TOOLS_BY_NAME = {t.name: t for t in ADVISOR_TOOLS}
MAX_TOOL_ITERATIONS = 3


class AdvisorChatRequest(BaseModel):
    message: str
    context: str = "advisor"   # "advisor" | "evalia_teacher"
    history: list[HistoryMessage] = []
    course_id: int | None = None  # Optional: enables tool-use when provided


@router.post("/chat/advisor", tags=["chat"])
async def advisor_chat(
    req: AdvisorChatRequest,
    token: None = Depends(_verify_token),
) -> ChatResponse:
    """
    Advisor/admin contextual chat with optional tool-use.

    When ``course_id`` is provided and context is "advisor", the LLM can call
    tools to query real data (risk scores, course material, etc.).
    Falls back to plain chat when tools are unavailable or for evalia_teacher context.
    """
    settings = get_settings()

    system_prompt = (
        SYSTEM_PROMPT_EVALIA_TEACHER
        if req.context == "evalia_teacher"
        else SYSTEM_PROMPT_ADVISOR
    )
    messages = [SystemMessage(content=system_prompt)]
    for h in req.history[-MAX_HISTORY_TURNS:]:
        if h.role == "user":
            messages.append(HumanMessage(content=h.content))
        else:
            messages.append(AIMessage(content=h.content))
    messages.append(HumanMessage(content=req.message))

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.4,
        request_timeout=float(settings.ollama_timeout),
    )

    use_tools = req.context == "advisor" and req.course_id is not None
    sources: list[str] = []

    if use_tools:
        reply = await _advisor_agent_loop(llm, messages, req.course_id)
    else:
        response = await llm.ainvoke(messages)
        reply = response.content if hasattr(response, "content") else str(response)

    log.info(
        "AdvisorChat context=%s course_id=%s tools=%s reply_len=%d",
        req.context, req.course_id, use_tools, len(reply),
    )
    return ChatResponse(reply=reply, sources=sources)


async def _advisor_agent_loop(
    llm: ChatOllama,
    messages: list,
    course_id: int,
) -> str:
    """
    Run a tool-calling agent loop: the LLM may call tools up to
    MAX_TOOL_ITERATIONS times, then must produce a final text response.
    """
    llm_with_tools = llm.bind_tools(ADVISOR_TOOLS)

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = await llm_with_tools.ainvoke(messages)

        # If no tool calls, we have the final answer
        if not getattr(response, "tool_calls", None):
            return response.content if hasattr(response, "content") else str(response)

        # Process tool calls
        messages.append(response)
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc.get("id", f"call_{iteration}_{tool_name}")

            tool_fn = ADVISOR_TOOLS_BY_NAME.get(tool_name)
            if tool_fn is None:
                result_text = f"Error: herramienta '{tool_name}' no encontrada."
                log.warning("Unknown tool requested: %s", tool_name)
            else:
                log.info("Tool call: %s(%s)", tool_name, tool_args)
                try:
                    result = await tool_fn.ainvoke(tool_args)
                    result_text = str(result)
                except Exception as exc:
                    result_text = f"Error ejecutando {tool_name}: {exc}"
                    log.error("Tool %s failed: %s", tool_name, exc)

            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id)
            )

    # If we exhausted iterations, do a final call without tools
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)
