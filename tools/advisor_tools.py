"""
Advisor tools — LangChain tools for the SAIPA advisor chat agent.

These tools wrap existing engine capabilities (RAG retrieval, risk prediction)
so the LLM can query real data when answering advisor/teacher questions.
"""
import asyncio
import logging

import httpx
from langchain_core.tools import tool

from analytics.features import extract_features
from analytics.risk_model import predict
from config import get_settings
from rag.retriever import retrieve

log = logging.getLogger("saipa_engine.tools.advisor")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_enrolled_students(course_id: int) -> list[dict]:
    """Fetch enrolled students from Moodle via local_saipa_get_student_list WS.

    Returns list of dicts with 'userid' and 'fullname' keys.
    """
    settings = get_settings()
    url = f"{settings.moodle_url}/webservice/rest/server.php"
    params = {
        "wstoken": settings.moodle_token,
        "wsfunction": "local_saipa_get_student_list",
        "moodlewsrestformat": "json",
        "course_id": course_id,
    }
    headers = {"Host": settings.moodle_public_host, "X-Forwarded-Proto": "https"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, dict) and "exception" in data:
        raise RuntimeError(f"Moodle WS error: {data.get('message', data)}")

    return data.get("students", [])


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_course_material(query: str, course_id: int) -> str:
    """Busca en el material indexado del curso usando RAG (Retrieval-Augmented Generation).
    Usar cuando el usuario pregunta sobre contenido académico, temas específicos
    o material de estudio de un curso. Requiere course_id.

    Args:
        query: Texto de búsqueda sobre el tema de interés.
        course_id: ID del curso en Moodle.
    """
    try:
        docs = retrieve(query, course_id, top_k=5)
    except Exception as exc:
        log.warning("search_course_material failed: %s", exc)
        return f"Error al buscar material: {exc}"

    if not docs:
        return "No se encontró material indexado para este curso. El docente debe indexar el material primero."

    parts = []
    for d in docs:
        source = d.get("source", "desconocido")
        content = d.get("content", "")
        # Truncate long chunks to avoid flooding the context
        if len(content) > 500:
            content = content[:500] + "..."
        parts.append(f"[{source}]: {content}")

    return "\n---\n".join(parts)


@tool
async def get_student_risk(user_id: int, course_id: int) -> str:
    """Obtiene el riesgo de abandono de un alumno específico usando el modelo de riesgo heurístico.
    Usar cuando preguntan por el riesgo o situación de un alumno en particular.
    Retorna el score (0-1), nivel de riesgo y los factores principales.

    Args:
        user_id: ID del usuario (alumno) en Moodle.
        course_id: ID del curso en Moodle.
    """
    try:
        features = await extract_features(user_id, course_id)
        result = predict(features)
    except Exception as exc:
        log.warning("get_student_risk failed user=%d course=%d: %s", user_id, course_id, exc)
        return f"Error al obtener el riesgo del alumno {user_id}: {exc}"

    # Format the top factors for readability
    factors = result.get("factors", {})
    sorted_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    factor_lines = ", ".join(f"{name}={val:.2f}" for name, val in sorted_factors)

    return (
        f"Alumno ID {user_id} en curso {course_id}: "
        f"riesgo={result['risk_level']} (score={result['score']:.2f}). "
        f"Factores principales: {factor_lines}"
    )


@tool
async def get_course_risk_summary(course_id: int) -> str:
    """Obtiene un resumen del riesgo de abandono de TODOS los alumnos del curso.
    Muestra conteo por nivel de riesgo (bajo/medio/alto) y lista los alumnos en riesgo alto.
    Usar cuando preguntan por estadísticas generales del curso o cuántos alumnos están en riesgo.

    Args:
        course_id: ID del curso en Moodle.
    """
    try:
        students = await _get_enrolled_students(course_id)
    except Exception as exc:
        log.warning("get_course_risk_summary: failed to get enrolled users: %s", exc)
        return f"Error al obtener alumnos inscriptos: {exc}"

    if not students:
        return f"No se encontraron alumnos inscriptos en el curso {course_id}."

    # Build name lookup
    names = {s["userid"]: s.get("fullname", f"ID {s['userid']}") for s in students}
    student_ids = list(names.keys())

    # Batch assess all students (concurrently, with error handling)
    async def _safe_assess(uid: int) -> dict | None:
        try:
            features = await extract_features(uid, course_id)
            result = predict(features)
            return {"user_id": uid, "name": names.get(uid, str(uid)), **result}
        except Exception as exc:
            log.warning("risk assessment failed user=%d: %s", uid, exc)
            return None

    results = await asyncio.gather(*[_safe_assess(uid) for uid in student_ids])
    assessed = [r for r in results if r is not None]

    if not assessed:
        return f"No se pudo evaluar a ningún alumno del curso {course_id}."

    # Count by level
    counts = {"low": 0, "medium": 0, "high": 0}
    high_risk = []
    for r in assessed:
        level = r.get("risk_level", "low")
        counts[level] = counts.get(level, 0) + 1
        if level == "high":
            high_risk.append(r)

    total = len(assessed)
    errors = len(student_ids) - total

    summary = (
        f"Curso {course_id}: {total} alumnos evaluados"
        f"{f' ({errors} con error)' if errors else ''}.\n"
        f"  - Riesgo BAJO: {counts['low']} ({counts['low']*100//max(total,1)}%)\n"
        f"  - Riesgo MEDIO: {counts['medium']} ({counts['medium']*100//max(total,1)}%)\n"
        f"  - Riesgo ALTO: {counts['high']} ({counts['high']*100//max(total,1)}%)\n"
    )

    if high_risk:
        summary += "\nAlumnos en riesgo ALTO:\n"
        for r in sorted(high_risk, key=lambda x: x["score"], reverse=True):
            summary += f"  - {r['name']} (ID {r['user_id']}): score={r['score']:.2f}\n"

    return summary
