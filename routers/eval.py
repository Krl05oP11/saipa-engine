"""
EVAL-IA evaluation router.

Endpoints:
  POST /eval/rubric/generate    — RAG → LLM → structured rubric (topics + weights)
  POST /eval/questions/generate — RAG by topic → LLM few-shot → question bank entries
  POST /eval/exam/sample        — pure Python sampling (no LLM, anti-copying)
"""
import asyncio
import logging
import random
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import get_settings
from llm.factory import get_chat_model, get_structured_llm, invoke_structured, invoke_structured_sync
from rag.retriever import retrieve
from routers.health import _verify_token
from telegram.client import get_telegram_client

log = logging.getLogger("saipa_engine.routers.eval")
router = APIRouter(prefix="/eval", tags=["eval"])


# ─── /eval/rubric/generate ───────────────────────────────────────────────────

RUBRIC_SYSTEM = """Eres un experto en diseño curricular universitario.
Tu tarea es analizar el material de un curso y extraer una rúbrica de evaluación exhaustiva.

Responde ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{
  "name": "Rúbrica de <nombre del curso>",
  "description": "Breve descripción de la rúbrica (1-2 oraciones)",
  "items": [
    {
      "topic": "Nombre del tema/subtema específico",
      "description": "Qué debe demostrar el alumno en este tema",
      "difficulty_weight": "low|medium|high",
      "sortorder": 1
    }
  ]
}

Instrucciones:
- Genera EXACTAMENTE {item_count} ítems (ni más, ni menos).
- Cada ítem debe corresponder a un subtema concreto (no agrupes varios subtemas en uno solo).
- Ejemplo: no pongas "Grafos" como un solo ítem; pon ítems separados para
  "Representación con matrices de adyacencia", "Algoritmo de Dijkstra", "Teorema de Ore",
  "Grafos eulerianos", "Grafos hamiltonianos", "Árbol generador mínimo (Prim/Kruskal)", etc.
- Distribuye los pesos equilibradamente: aproximadamente 30% low, 45% medium, 25% high.
- Ordená los ítems agrupados por subtema.
- No incluyas texto fuera del JSON. No uses markdown."""


class RubricGenerateRequest(BaseModel):
    course_id: int
    scope: str = ""                # optional topic filter e.g. "Relaciones", "Grafos"
    item_count: int = 18           # number of rubric items to generate (5–40)
    source_filter: list[str] = []  # if non-empty, restrict RAG to these source IDs


class RubricItem(BaseModel):
    topic: str
    description: str
    difficulty_weight: Literal["low", "medium", "high"] = "medium"
    sortorder: int = 5


class RubricGenerateResponse(BaseModel):
    name: str
    description: str
    items: list[RubricItem]


class _FillPassItems(BaseModel):
    """Structured-output schema for generate_rubric's fill-pass — same
    item shape, no name/description (the fill-pass only adds items to
    an already-generated rubric)."""
    items: list[RubricItem]


@router.get("/sources/{course_id}")
async def list_sources(
    course_id: int,
    token: None = Depends(_verify_token),
) -> dict:
    """Returns the list of unique source identifiers indexed for a course."""
    import chromadb
    from config import get_settings
    settings = get_settings()
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection = client.get_collection(f"course_{course_id}")
        if collection.count() == 0:
            return {"sources": []}

        # ChromaDB's get() is bounded internally at ~100 items regardless of the
        # limit parameter.  Paginate with a safe batch size to collect all sources.
        all_sources: set[str] = set()
        batch_size = 500
        offset = 0
        while True:
            result = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
            metas = result.get("metadatas") or []
            for m in metas:
                src = m.get("source", "")
                if src:
                    all_sources.add(src)
            if len(metas) < batch_size:
                break
            offset += batch_size

        return {"sources": sorted(all_sources)}
    except Exception:
        return {"sources": []}


@router.post("/rubric/generate", response_model=RubricGenerateResponse)
async def generate_rubric(
    req: RubricGenerateRequest,
    token: None = Depends(_verify_token),
) -> RubricGenerateResponse:
    """
    Retrieves course content from ChromaDB and asks Ollama to produce
    a structured evaluation rubric with topics and difficulty weights.
    """
    # 1. Retrieve course content — scope-aware if a topic filter is provided.
    _src_filter = req.source_filter or None   # None → no filtering
    docs = []
    if req.scope:
        # Partial rubric: use many targeted queries to cover all subtopics
        # within the scope.  Deduplication is by content fingerprint only —
        # NOT by source — so multiple chunks from the same chapter can be
        # included (essential when all content lives in one indexed source).
        queries = [
            req.scope,
            f"{req.scope} conceptos definición",
            f"{req.scope} propiedades ejemplos aplicaciones",
            f"{req.scope} algoritmos teoremas",
            f"{req.scope} tipos clasificación",
        ]
        seen_fps: set[str] = set()
        for query in queries:
            for d in retrieve(query, req.course_id, top_k=15, allowed_sources=_src_filter):
                fp = d["content"][:120]
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    docs.append(d)
    else:
        # Full rubric: broad multi-query, deduplicate by source to ensure
        # coverage across all course materials (not just one chapter).
        seen_sources: set[str] = set()
        queries = [
            "temas principales del curso contenido temario",
            "objetivos aprendizaje evaluación competencias",
            "relaciones grafos árboles conjuntos lógica",
        ]
        for query in queries:
            for d in retrieve(query, req.course_id, top_k=10, allowed_sources=_src_filter):
                if d["source"] not in seen_sources:
                    seen_sources.add(d["source"])
                    docs.append(d)

    if not docs:
        # No content indexed — return minimal placeholder
        log.warning("generate_rubric: no indexed content for course %d", req.course_id)
        return RubricGenerateResponse(
            name=f"Rúbrica del curso {req.course_id}",
            description="No se encontró material indexado para este curso.",
            items=[
                RubricItem(topic="Sin contenido indexado",
                           description="Indexe el material del curso en SAIPA antes de generar la rúbrica.",
                           difficulty_weight="medium", sortorder=1)
            ],
        )

    # 2. Build context from retrieved chunks
    context = "\n---\n".join(
        f"[{d['source']}]:\n{d['content']}" for d in docs
    )

    # Resolve item_count early — used in scope_instruction and later in fill-pass
    item_count = max(5, min(40, req.item_count))

    # Build a list of OTHER topics to explicitly exclude (all scope-like words found
    # in source labels that are NOT the requested scope).
    other_sources = sorted({
        d["source"].split(":")[-1]
        for d in docs
        if req.scope.lower() not in d["source"].lower()
    })
    exclude_note = (
        f"- EXCLUIR COMPLETAMENTE los siguientes temas aunque aparezcan en el material: "
        f"{', '.join(other_sources)}.\n"
        if other_sources else ""
    )

    scope_instruction = (
        f"IMPORTANTE: Esta es una rúbrica para un EXAMEN PARCIAL sobre '{req.scope}'.\n"
        f"- Genera ítems ÚNICAMENTE sobre '{req.scope}' y sus subtemas específicos.\n"
        f"- No incluyas contenido de otros temas del curso.\n"
        f"{exclude_note}"
        f"- Debes generar EXACTAMENTE {item_count} ítems. "
        f"Desglosa cada subtema en ítems separados (algoritmos, teoremas, definiciones, "
        f"propiedades, representaciones, tipos, aplicaciones).\n\n"
    ) if req.scope else ""

    user_msg = (
        f"{scope_instruction}"
        f"Analiza el siguiente material del curso (course_id={req.course_id}) "
        f"y genera la rúbrica de evaluación exhaustiva cubriendo TODOS los subtemas:\n\n{context[:20000]}"
    )

    # 3. Call the LLM — with fill-pass if it returns fewer items than requested
    system_prompt = RUBRIC_SYSTEM.replace("{item_count}", str(item_count))
    llm = get_structured_llm(RubricGenerateResponse, temperature=0.2, model_purpose="grading")
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]

    try:
        parsed = await invoke_structured(llm, messages)
    except Exception as exc:
        log.error("generate_rubric: LLM/parse error for course %d: %s", req.course_id, exc)
        raise

    raw_items = list(parsed.items)

    # 4. Fill pass: if fewer items than requested, ask for the missing ones.
    # Keep the fill call small — no big context repetition, just the gap instruction.
    if len(raw_items) < item_count:
        missing = item_count - len(raw_items)
        existing_topics = "; ".join(it.topic for it in raw_items)
        topic_label = req.scope or "el curso"
        fill_system = (
            "Eres un experto en diseño curricular. Genera los ítems de rúbrica pedidos."
        )
        fill_user = (
            f"Necesito {missing} ítems adicionales de rúbrica sobre '{topic_label}'.\n"
            f"Ya tengo estos temas (NO repetir): {existing_topics}.\n"
            f"Genera exactamente {missing} ítems nuevos sobre subtemas aún no cubiertos "
            f"de '{topic_label}': algoritmos, teoremas, propiedades, representaciones, "
            f"tipos, matrices, caminos, aplicaciones — lo que falte según el material.\n"
            f"Contexto breve del material:\n{context[:4000]}"
        )
        fill_messages = [
            SystemMessage(content=fill_system),
            HumanMessage(content=fill_user),
        ]
        fill_llm = get_structured_llm(_FillPassItems, temperature=0.2, model_purpose="grading")
        try:
            fill_parsed = await invoke_structured(fill_llm, fill_messages)
            raw_items = raw_items + list(fill_parsed.items)
            log.info(
                "generate_rubric fill-pass: course=%d added=%d total=%d",
                req.course_id, len(fill_parsed.items), len(raw_items),
            )
        except Exception as exc:
            log.warning("generate_rubric: fill-pass failed (non-fatal): %s", exc)

    items = [
        RubricItem(
            topic=it.topic,
            description=it.description,
            difficulty_weight=it.difficulty_weight,
            sortorder=idx + 1,
        )
        for idx, it in enumerate(raw_items)
    ]

    log.info("generate_rubric: course=%d requested=%d final=%d", req.course_id, item_count, len(items))

    return RubricGenerateResponse(
        name=parsed.name,
        description=parsed.description,
        items=items,
    )


# ─── /eval/questions/generate ────────────────────────────────────────────────

QUESTIONS_SYSTEM = """Eres un experto en evaluación educativa universitaria.
Tu tarea es generar preguntas de evaluación a partir del material de un curso.

Responde ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{
  "questions": [
    {
      "stem": "Enunciado completo de la pregunta",
      "type": "multichoice|truefalse|numerical|shortanswer",
      "difficulty": "basic|medium|advanced",
      "topic": "Nombre del tema",
      "correct_answer": "Respuesta correcta (solo para shortanswer/numerical)",
      "tolerance": 0,
      "options": [
        {
          "text": "Texto de la opción",
          "correct": true,
          "feedback": "Retroalimentación breve para esta opción"
        }
      ]
    }
  ]
}

Reglas:
- Para "multichoice": genera exactamente 4 opciones, solo 1 correcta.
- Para "truefalse": genera exactamente 2 opciones ("Verdadero", "Falso").
- Para "numerical": deja "options" como array vacío [], pon el valor numérico en "correct_answer".
- Para "shortanswer": deja "options" como array vacío [], pon la respuesta en "correct_answer".
- Para "essay": deja "options" como array vacío []. En "correct_answer" escribe los ELEMENTOS CLAVE que debe mencionar una respuesta completa (máximo 5 puntos separados por punto y coma). El "stem" debe ser una consigna abierta que invite a reflexionar y desarrollar.
- Las preguntas deben basarse EXCLUSIVAMENTE en el material del curso proporcionado.
- No incluyas texto fuera del JSON. No uses markdown."""


class QuestionGenerateRequest(BaseModel):
    course_id: int
    topic: str
    description: str = ""
    difficulty: Literal["basic", "medium", "advanced"] = "medium"
    question_types: list[str] = ["multichoice"]
    count: int = 5


class QuestionOption(BaseModel):
    text: str
    correct: bool
    feedback: str = ""


class GeneratedQuestion(BaseModel):
    stem: str
    type: str
    difficulty: str
    topic: str
    correct_answer: str = ""
    tolerance: float = 0.0
    options: list[QuestionOption] = []


class QuestionGenerateResponse(BaseModel):
    questions: list[GeneratedQuestion]


@router.post("/questions/generate", response_model=QuestionGenerateResponse)
async def generate_questions(
    req: QuestionGenerateRequest,
    token: None = Depends(_verify_token),
) -> QuestionGenerateResponse:
    """
    Retrieves course content relevant to a specific rubric topic and generates
    evaluation questions using Ollama with a structured JSON prompt.
    """
    # 1. Retrieve topic-specific chunks
    query = f"{req.topic} {req.description}"
    docs = retrieve(query, req.course_id, top_k=8)

    context = ""
    if docs:
        context = "\n---\n".join(
            f"[{d['source']}]:\n{d['content']}" for d in docs
        )

    # 2. Build user message
    types_str = ", ".join(req.question_types)
    user_msg = (
        f"Genera {req.count} pregunta(s) de nivel '{req.difficulty}' "
        f"sobre el tema '{req.topic}' ({req.description}).\n"
        f"Tipos de pregunta a usar: {types_str}.\n"
        f"Distribuye los tipos de forma equilibrada si hay más de 1.\n\n"
    )
    if context:
        user_msg += f"Material del curso (usa SOLO este contenido):\n\n{context[:8000]}"
    else:
        user_msg += (
            "No hay material indexado para este curso. Genera preguntas conceptuales "
            "razonables sobre el tema indicado, de nivel universitario."
        )

    # 3. Call the LLM (low temperature for structured output)
    llm = get_structured_llm(QuestionGenerateResponse, temperature=0.1, model_purpose="grading")
    messages = [SystemMessage(content=QUESTIONS_SYSTEM), HumanMessage(content=user_msg)]

    try:
        parsed = await invoke_structured(llm, messages)
    except Exception as exc:
        log.error("generate_questions: LLM/parse error course=%d topic='%s': %s",
                  req.course_id, req.topic, exc)
        raise

    log.info("generate_questions: course=%d topic='%s' difficulty=%s generated=%d",
             req.course_id, req.topic, req.difficulty, len(parsed.questions))

    return parsed


# ─── /eval/exam/sample ───────────────────────────────────────────────────────

class QuestionMeta(BaseModel):
    id: int
    difficulty: Literal["basic", "medium", "advanced"]
    topic: str


class ExamSampleRequest(BaseModel):
    questions: list[QuestionMeta]    # full pool of approved questions with metadata
    basic_count: int = 3
    medium_count: int = 4
    advanced_count: int = 2
    topic_coverage: dict[str, int] = {}   # topic → minimum questions from this topic


class ExamSampleResponse(BaseModel):
    selected_ids: list[int]
    total_requested: int
    total_available: int
    coverage_satisfied: bool


@router.post("/exam/sample", response_model=ExamSampleResponse)
async def sample_exam(
    req: ExamSampleRequest,
    token: None = Depends(_verify_token),
) -> ExamSampleResponse:
    """
    Pure Python sampling — NO LLM involved.

    Algorithm:
    1. Separate pool by difficulty.
    2. Enforce topic_coverage minimums by pre-selecting required questions.
    3. Fill remaining slots randomly from each difficulty bucket.
    4. Shuffle the final list to prevent pattern detection.

    Returns unique question IDs for one student's exam.
    Caller must invoke this separately per student to ensure different selections.
    """
    difficulty_map: dict[str, list[QuestionMeta]] = {
        "basic": [],
        "medium": [],
        "advanced": [],
    }
    for q in req.questions:
        if q.difficulty in difficulty_map:
            difficulty_map[q.difficulty].append(q)

    # Shuffle each bucket for randomness
    for bucket in difficulty_map.values():
        random.shuffle(bucket)

    needed = {
        "basic":    req.basic_count,
        "medium":   req.medium_count,
        "advanced": req.advanced_count,
    }
    total_requested = sum(needed.values())
    total_available = len(req.questions)

    selected: list[int] = []
    used_ids: set[int] = set()

    # Phase 1 — satisfy topic_coverage minimums (pick from any difficulty)
    if req.topic_coverage:
        topic_pool: dict[str, list[QuestionMeta]] = {}
        for q in req.questions:
            topic_pool.setdefault(q.topic, []).append(q)
        for tp in topic_pool.values():
            random.shuffle(tp)

        for topic, min_count in req.topic_coverage.items():
            pool = [q for q in topic_pool.get(topic, []) if q.id not in used_ids]
            picks = pool[:min_count]
            for q in picks:
                used_ids.add(q.id)
                selected.append(q.id)
                # Deduct from the corresponding difficulty bucket counter
                if needed.get(q.difficulty, 0) > 0:
                    needed[q.difficulty] -= 1

    # Phase 2 — fill remaining slots by difficulty
    for diff, count in needed.items():
        if count <= 0:
            continue
        pool = [q for q in difficulty_map[diff] if q.id not in used_ids]
        picks = pool[:count]
        for q in picks:
            used_ids.add(q.id)
            selected.append(q.id)

    coverage_satisfied = (len(selected) == total_requested)

    # Final shuffle — randomise question order per student
    random.shuffle(selected)

    log.info(
        "exam/sample: requested=%d available=%d selected=%d coverage=%s",
        total_requested, total_available, len(selected), coverage_satisfied,
    )

    return ExamSampleResponse(
        selected_ids=selected,
        total_requested=total_requested,
        total_available=total_available,
        coverage_satisfied=coverage_satisfied,
    )


# ─── /eval/exam/grade ────────────────────────────────────────────────────────

_DIFFICULTY_WEIGHTS: dict[str, float] = {"basic": 1.0, "medium": 2.0, "advanced": 3.0}


ESSAY_GRADE_SYSTEM = """Eres un evaluador académico universitario experto en retroalimentación formativa.
Se te proporciona material del curso (contexto RAG), una pregunta de ensayo, los elementos
clave esperados en una respuesta completa, y la respuesta del alumno.

Evalúa la respuesta del 0.0 al 1.0 y proporciona retroalimentación breve y constructiva.

Responde ÚNICAMENTE con un objeto JSON válido:
{
  "score": 0.8,
  "feedback": "2-3 oraciones indicando qué estuvo bien y qué faltó o puede mejorar."
}

Criterios:
- 0.9–1.0: Completa, precisa y bien fundamentada en los conceptos del curso.
- 0.7–0.8: Conceptos principales correctos, algún detalle faltante.
- 0.5–0.6: Parcial — conceptos básicos presentes pero faltan aspectos importantes.
- 0.3–0.4: Débil — solo algunos elementos relevantes.
- 0.0–0.2: Irrelevante, incorrecta o muy incompleta.

No uses markdown. Solo el JSON."""


class GradeQuestionData(BaseModel):
    id: int
    type: str                       # multichoice|truefalse|numerical|shortanswer|essay
    difficulty: str = "medium"      # basic|medium|advanced — determines point weight
    correct_option_text: str = ""   # for multichoice / truefalse
    correct_answer: str = ""        # for numerical / shortanswer / essay (expected elements)
    tolerance: float = 0.0
    stem: str = ""                  # essay only — question text for RAG context
    topic: str = ""                 # essay only — topic for RAG retrieval


class ExamGradeRequest(BaseModel):
    questions: list[GradeQuestionData]
    answers: dict[str, str]             # str(question_id) → student_answer_text
    weights: dict[str, float] = {}      # optional weight overrides from Moodle config
    course_id: int = 0                  # required for essay RAG grading


class GradeDetail(BaseModel):
    question_id: int
    correct: bool
    score: float                    # weight if correct (objective) or weight*ai_score (essay)
    weight: float                   # point value of this question
    feedback: str = ""              # essay only — AI evaluation explanation


class ExamGradeResponse(BaseModel):
    total_score: float              # sum of weights for correct questions
    max_score: float                # sum of all question weights
    details: list[GradeDetail]


class EssayGradeResult(BaseModel):
    score: float
    feedback: str


def _grade_essay_sync(
    q: GradeQuestionData,
    student_answer: str,
    course_id: int,
) -> tuple[float, str]:
    """
    Synchronous essay grader — called via asyncio.to_thread().
    Retrieves RAG context for the question topic, then asks the LLM to score
    and explain the student's answer.  Returns (score 0.0–1.0, feedback_text).
    """
    docs = retrieve(f"{q.topic} {q.stem}", course_id, top_k=5)
    context = "\n---\n".join(d["content"] for d in docs) if docs else ""

    user_msg = (
        f"Pregunta: {q.stem}\n\n"
        f"Elementos esperados en una respuesta completa: {q.correct_answer}\n\n"
        f"Respuesta del alumno: {student_answer}\n\n"
    )
    if context:
        user_msg += f"Material del curso (referencia):\n{context[:4000]}"

    llm = get_structured_llm(EssayGradeResult, temperature=0.0, model_purpose="grading")
    messages = [SystemMessage(content=ESSAY_GRADE_SYSTEM), HumanMessage(content=user_msg)]

    try:
        result = invoke_structured_sync(llm, messages)
        return max(0.0, min(1.0, result.score)), result.feedback
    except Exception as exc:
        log.error("essay grading failed for question %d: %s", q.id, exc)

    return 0.5, "No se pudo evaluar la respuesta automáticamente."


@router.post("/exam/grade", response_model=ExamGradeResponse)
async def grade_exam(
    req: ExamGradeRequest,
    token: None = Depends(_verify_token),
) -> ExamGradeResponse:
    """
    Grades a student exam with difficulty-weighted scoring.
    basic=1pt, medium=2pts, advanced=3pts.
    score_10 = (total_score / max_score) * 10.
    """
    details: list[GradeDetail] = []

    # Merge default weights with any overrides from the Moodle config.
    effective_weights = {**_DIFFICULTY_WEIGHTS, **{k.lower(): v for k, v in req.weights.items()}}

    for q in req.questions:
        weight       = effective_weights.get(q.difficulty.lower(), 1.0)
        qid_str      = str(q.id)
        student_ans  = req.answers.get(qid_str, "").strip()
        correct      = False
        feedback_txt = ""

        if q.type == "essay":
            if not student_ans:
                details.append(GradeDetail(
                    question_id=q.id, correct=False, score=0.0,
                    weight=weight, feedback="Sin respuesta.",
                ))
                continue
            ai_score, feedback_txt = await asyncio.to_thread(
                _grade_essay_sync, q, student_ans, req.course_id
            )
            partial_score = round(weight * ai_score, 2)
            details.append(GradeDetail(
                question_id=q.id,
                correct=ai_score >= 0.6,
                score=partial_score,
                weight=weight,
                feedback=feedback_txt,
            ))
            continue

        if q.type in ("multichoice", "truefalse"):
            correct = student_ans.lower() == q.correct_option_text.lower().strip()
        elif q.type == "numerical":
            try:
                val     = float(student_ans)
                correct = abs(val - float(q.correct_answer)) <= q.tolerance
            except (ValueError, TypeError):
                correct = False
        elif q.type == "shortanswer":
            correct = student_ans.lower() == q.correct_answer.lower().strip()

        details.append(GradeDetail(
            question_id=q.id,
            correct=correct,
            score=weight if correct else 0.0,
            weight=weight,
        ))

    total_score = sum(d.score for d in details)
    max_score   = sum(d.weight for d in details)

    log.info(
        "exam/grade: questions=%d score=%.1f/%.1f (weighted)",
        len(req.questions), total_score, max_score,
    )

    return ExamGradeResponse(
        total_score=total_score,
        max_score=max_score,
        details=details,
    )


# /eval/portfolio/aggregate was removed 2026-09-16: dead code, confirmed by
# an exhaustive grep of local_evalia for engine_request('/portfolio/... --
# no PHP caller ever existed. Portfolio stats (avg_grade, total_exams) are
# computed directly in PHP (grade_exam.php, publish_grade.php), which is
# also where get_student_portfolio.php reads them from -- this endpoint
# had zero live coverage and nothing depended on it.


# ─── /eval/feedback/send ─────────────────────────────────────────────────────

FEEDBACK_SYSTEM = """Eres SAIPA, asistente pedagógico de acompañamiento universitario.
Acabas de conocer el resultado del examen de un alumno y tu misión es enviarle
un mensaje personal, cálido y educativo por Telegram.

El mensaje debe:
1. Saludar al alumno por su nombre de pila
2. Comunicar la nota de forma clara y honesta
3. Por cada pregunta INCORRECTA: explicar brevemente qué respondió el alumno,
   cuál era la respuesta correcta y POR QUÉ esa respuesta es la correcta
   (usa el contexto del material del curso si está disponible)
4. Si todas fueron correctas: felicitarlo genuinamente y destacar su logro
5. Indicar en qué temas conviene profundizar según los errores
6. Cerrar con una frase motivadora: los errores son oportunidades de aprendizaje,
   no fracasos. El alumno puede mejorar y SAIPA lo acompañará en ese camino

Reglas de formato:
- Usa HTML de Telegram: <b>negrita</b> para términos clave, <i>cursiva</i> para énfasis
- Sé conciso pero completo. Máximo ~600 palabras.
- Tono: cálido, directo, universitario. No paternalista ni condescendiente.
- NO uses markdown (**, ##, ---). SOLO HTML de Telegram.
- Responde ÚNICAMENTE con el mensaje, sin JSON ni comentarios adicionales."""


class QuestionFeedbackItem(BaseModel):
    question_id: int
    stem: str
    question_type: str              # multichoice|truefalse|numerical|shortanswer|essay
    student_answer: str
    correct_answer: str             # correct option text or correct_answer field
    correct: bool
    topic: str = ""
    ai_feedback: str = ""           # essay only — AI evaluation explanation


class FeedbackSendRequest(BaseModel):
    telegram_id: int                # Telegram chat_id (resolved by PHP before calling)
    userid: int                     # Moodle user ID (for logging only)
    examid: int                     # evalia_student_exams ID (for logging only)
    score: float                    # 0–10 scale
    max_score: float                # always 10.0
    course_name: str = ""
    course_id: int = 0              # for RAG context on wrong answers
    student_name: str = ""          # for personalised greeting
    questions: list[QuestionFeedbackItem] = []  # per-question breakdown
    feedback_prompt: str = ""       # teacher-customised system prompt (empty = use default)


class FeedbackSendResponse(BaseModel):
    sent: bool
    reason: str


async def _generate_and_send_feedback(req: FeedbackSendRequest) -> None:
    """
    Background task: generate a pedagogical feedback message via LLM + RAG
    and send it to the student via Telegram.
    Errors are logged but never propagated (non-fatal).
    """
    settings = get_settings()
    # Demo mode: if EVALIA_DEMO_TELEGRAM_ID is set, always use it.
    # Otherwise use the student's real telegram_id (may be 0 = no link → skip).
    target_telegram_id = (
        settings.evalia_demo_telegram_id
        if settings.evalia_demo_telegram_id > 0
        else req.telegram_id
    )
    if target_telegram_id <= 0:
        log.info("feedback: no telegram target for userid=%d — skipping", req.userid)
        return

    tg = get_telegram_client()
    try:
        percent       = round(req.score / req.max_score * 100) if req.max_score > 0 else 0
        wrong         = [q for q in req.questions if not q.correct]
        correct_count = len(req.questions) - len(wrong)
        name          = req.student_name.split()[0] if req.student_name else "alumno/a"

        # ── RAG: retrieve course material for incorrect question topics ──────
        rag_context = ""
        if wrong and req.course_id:
            unique_topics = list(dict.fromkeys(q.topic for q in wrong if q.topic))
            query = " ".join(unique_topics[:3])
            docs  = retrieve(query, req.course_id, top_k=5)
            if docs:
                rag_context = "\n---\n".join(
                    f"[{d['source']}]: {d['content'][:400]}" for d in docs
                )

        # ── Build LLM user message ────────────────────────────────────────────
        user_msg = (
            f"Alumno: {req.student_name or 'el alumno'}\n"
            f"Curso: {req.course_name}\n"
            f"Nota obtenida: {req.score:.1f} / {req.max_score:.1f} ({percent}%)\n"
            f"Respuestas correctas: {correct_count} de {len(req.questions)}\n"
        )

        if wrong:
            user_msg += "\nRespuestas INCORRECTAS o incompletas (explicar cada una):\n"
            for q in wrong[:5]:          # cap at 5 to keep prompt size manageable
                if q.question_type == "essay" and q.ai_feedback:
                    user_msg += (
                        f"\n• Tema: {q.topic}\n"
                        f"  Pregunta (ensayo): {q.stem}\n"
                        f"  Respondió: {q.student_answer or '(sin respuesta)'}\n"
                        f"  Evaluación IA: {q.ai_feedback}\n"
                    )
                else:
                    ca = q.correct_answer or "(ver opciones)"
                    user_msg += (
                        f"\n• Tema: {q.topic}\n"
                        f"  Pregunta: {q.stem}\n"
                        f"  Respondió: {q.student_answer or '(sin respuesta)'}\n"
                        f"  Correcto era: {ca}\n"
                    )
        else:
            user_msg += "\n¡Todas las respuestas fueron correctas!\n"

        if rag_context:
            user_msg += (
                f"\nContexto del material del curso "
                f"(úsalo para fundamentar las explicaciones):\n{rag_context[:3000]}"
            )

        # ── LLM call (plain text, not JSON) ──────────────────────────────────
        system_prompt = req.feedback_prompt.strip() if req.feedback_prompt.strip() else FEEDBACK_SYSTEM
        llm = get_chat_model(temperature=0.4, model_purpose="grading")
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]
        response = await llm.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)

        # Telegram hard limit: 4096 chars
        if len(text) > 3900:
            text = text[:3900] + "\n\n<i>(mensaje completo disponible en la plataforma)</i>"

        await tg.send_message(chat_id=target_telegram_id, text=text)
        log.info(
            "feedback/send: pedagogical message sent userid=%d examid=%d score=%.1f wrong=%d",
            req.userid, req.examid, req.score, len(wrong),
        )
    except Exception as exc:
        log.error(
            "feedback/send: background generation failed userid=%d examid=%d: %s",
            req.userid, req.examid, exc,
        )


class NotifyRequest(BaseModel):
    telegram_id: int
    message: str
    parse_mode: str = "HTML"


class NotifyResponse(BaseModel):
    sent: bool
    reason: str


@router.post("/notify", response_model=NotifyResponse, tags=["eval"])
async def notify(
    req: NotifyRequest,
    token: None = Depends(_verify_token),
) -> NotifyResponse:
    """
    Sends a plain Telegram message to a given chat_id.
    Used for teacher notifications (e.g. student submitted exam).
    Blocking but fast (no LLM involved).
    """
    tg = get_telegram_client()
    if not tg.enabled():
        return NotifyResponse(sent=False, reason="telegram_disabled")
    sent = await tg.send_message(req.telegram_id, req.message, req.parse_mode)
    log.info("notify: telegram_id=%d sent=%s", req.telegram_id, sent)
    return NotifyResponse(sent=sent, reason="ok" if sent else "send_failed")


@router.post("/feedback/send", response_model=FeedbackSendResponse)
async def send_feedback(
    req: FeedbackSendRequest,
    background_tasks: BackgroundTasks,
    token: None = Depends(_verify_token),
) -> FeedbackSendResponse:
    """
    Queues a pedagogical feedback message for async delivery via Telegram.
    Returns immediately (≤100ms) so PHP grading is never blocked.
    The LLM generation + Telegram send happen in the background.
    """
    tg = get_telegram_client()
    if not tg.enabled():
        log.warning("feedback/send: Telegram not configured — skipping notification")
        return FeedbackSendResponse(sent=False, reason="telegram_disabled")

    background_tasks.add_task(_generate_and_send_feedback, req)

    log.info(
        "feedback/send: queued background task userid=%d examid=%d score=%.1f",
        req.userid, req.examid, req.score,
    )
    return FeedbackSendResponse(sent=True, reason="queued")
