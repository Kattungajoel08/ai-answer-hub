from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import asyncio
import httpx
import random
import json
import re

load_dotenv()

app = FastAPI(title="AI Answer Hub")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

MODELS = ["chatgpt", "gemini", "perplexity", "claude"]

async def post_with_retry(client, url, *, headers, json_body, attempts=3):
    """Retry temporary provider failures with exponential backoff."""
    retryable_statuses = {408, 429, 500, 502, 503, 504}

    for attempt in range(attempts):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=json_body
            )

            if response.status_code not in retryable_statuses:
                response.raise_for_status()
                return response

            if attempt == attempts - 1:
                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            if (
                exc.response.status_code not in retryable_statuses
                or attempt == attempts - 1
            ):
                raise

        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == attempts - 1:
                raise

        delay = min(2 ** attempt, 8) + random.uniform(0, 0.5)
        await asyncio.sleep(delay)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    models: list[str] | None = None
    previous_answers: dict[str, str] | None = None

@app.get("/")
async def home():
    return FileResponse("app/static/index.html")

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "configured": {
            "chatgpt": bool(os.getenv("OPENAI_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
            "claude": bool(os.getenv("ANTHROPIC_API_KEY"))
        }
    }


async def stream_gemini_with_retry(client, model, key, question, delta_queue):
    """Stream Gemini text chunks as they arrive, retrying transient HTTP failures."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    retryable_statuses = {408, 429, 500, 502, 503, 504}

    for attempt in range(3):
        pieces = []
        try:
            async with client.stream(
                "POST",
                url,
                params={"alt": "sse"},
                headers={"x-goog-api-key": key},
                json={"contents": [{"parts": [{"text": question}]}]},
            ) as response:
                if response.status_code in retryable_statuses:
                    if attempt < 2:
                        await response.aread()
                        await asyncio.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))
                        continue
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        chunk = part.get("text", "")
                        if chunk:
                            pieces.append(chunk)
                            delta_queue.put_nowait({"type": "delta", "model": "gemini", "delta": chunk})

                return "".join(pieces)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in retryable_statuses or attempt == 2:
                raise
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 2:
                raise
        await asyncio.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))
    return "".join(pieces)


async def request_ai(name, question, delta_queue=None):
    key_names = {
        "chatgpt": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "claude": "ANTHROPIC_API_KEY"
    }

    key = os.getenv(key_names[name], "").strip()

    if not key:
        return {
            "model": name,
            "ok": False,
            "answer": "",
            "error": f"{key_names[name]} is missing in .env"
        }

    timeout = httpx.Timeout(180, connect=20)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:

            if name == "chatgpt":
                response = await post_with_retry(
                    client,
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {key}"},
                    json_body={
                        "model": os.getenv(
                            "OPENAI_MODEL", "gpt-4.1-mini"
                        ),
                        "input": question
                    }
                )
                data = response.json()

                answer = data.get("output_text", "")
                if not answer:
                    answer = "\n".join(
                        content.get("text", "")
                        for item in data.get("output", [])
                        for content in item.get("content", [])
                        if content.get("type") == "output_text"
                    )

            elif name == "gemini":
                model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                answer = await stream_gemini_with_retry(
                    client, model, key, question, delta_queue
                )

            elif name == "perplexity":
                response = await post_with_retry(
                    client,
                    "https://api.perplexity.ai/v1/responses",
                    headers={"Authorization": f"Bearer {key}"},
                    json_body={
                        "model": os.getenv(
                            "PERPLEXITY_MODEL", "sonar"
                        ),
                        "input": question
                    }
                )
                data = response.json()

                answer = data.get("output_text", "")
                if not answer:
                    answer = "\n".join(
                        content.get("text", "")
                        for item in data.get("output", [])
                        for content in item.get("content", [])
                        if content.get("type") == "output_text"
                    )

            elif name == "claude":
                response = await post_with_retry(
                    client,
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                    json_body={
                        "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": question}]
                    }
                )
                data = response.json()
                answer = "\n".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )

        return {
            "model": name,
            "ok": True,
            "answer": answer or "No text returned.",
            "error": None
        }

    except httpx.HTTPStatusError as exc:
        try:
            error_data = exc.response.json()
            detail = (
                error_data.get("error", {}).get("message")
                or error_data.get("message")
            )
        except Exception:
            detail = None

        return {
            "model": name,
            "ok": False,
            "answer": "",
            "error": detail or (
                f"Provider HTTP error {exc.response.status_code}"
            )
        }

    except Exception as exc:
        return {
            "model": name,
            "ok": False,
            "answer": "",
            "error": (
                f"{type(exc).__name__}: "
                f"{str(exc) or 'No details available'}"
            )
        }



def _tokens(text):
    return re.findall(r"\b[\w'-]+\b", (text or "").lower())


def evaluate_answers(question, answers_by_model):
    """Heuristic quality estimates only; factual claims are not independently verified."""
    question_terms = {w for w in _tokens(question) if len(w) > 3}
    results = {}
    for model, answer in answers_by_model.items():
        text = answer or ""
        words = _tokens(text)
        word_count = len(words)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        answer_terms = set(words)
        overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
        relevance = min(10.0, 4.0 + overlap * 8.0)

        completeness = 3.0
        if word_count >= 40: completeness += 1.5
        if word_count >= 100: completeness += 1.5
        if word_count >= 180: completeness += 1.0
        if re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text): completeness += 1.0
        if len(sentences) >= 3: completeness += 1.0
        completeness = min(10.0, completeness)

        reasoning_words = {"because", "therefore", "however", "since", "thus", "if", "then",
                           "first", "second", "finally", "consequently", "although", "means"}
        reasoning = min(10.0, 4.0 + len(answer_terms & reasoning_words) * 0.8
                        + (1.0 if len(sentences) >= 4 else 0.0))

        avg_sentence_words = word_count / max(1, len(sentences))
        clarity = 8.0
        if avg_sentence_words > 32: clarity -= 2.0
        elif avg_sentence_words > 24: clarity -= 1.0
        if word_count < 8: clarity -= 2.0
        if word_count > 60 and len(set(words)) / max(1, word_count) < 0.30: clarity -= 1.5
        clarity = max(1.0, min(10.0, clarity))

        # Citation-like formatting is only a signal, not proof that a claim is true.
        citations = len(re.findall(r"https?://|\[[0-9]+\]|\([^)]+,\s*\d{4}\)", text))
        factual_support = min(10.0, 4.5 + min(citations, 4) * 1.0)
        overall = round(0.25*factual_support + 0.20*relevance + 0.20*completeness
                        + 0.20*reasoning + 0.15*clarity, 1)
        results[model] = {"scores": {
            "factual_support_signals": round(factual_support, 1),
            "relevance": round(relevance, 1),
            "completeness": round(completeness, 1),
            "reasoning": round(reasoning, 1),
            "clarity": round(clarity, 1),
            "overall_estimated_quality": overall
        }}

    ranked = sorted(results, key=lambda m: results[m]["scores"]["overall_estimated_quality"], reverse=True)
    tied = len(ranked) > 1 and results[ranked[0]]["scores"]["overall_estimated_quality"] == results[ranked[1]]["scores"]["overall_estimated_quality"]
    return {
        "type": "evaluation", "scores": results,
        "recommended_model": None if not ranked or tied else ranked[0],
        "recommendation_note": "No clear winner: top estimated quality scores are tied." if tied else
            "Recommendation uses heuristic quality signals, not verified factual accuracy.",
        "verified_accuracy": False
    }


@app.post("/api/ask")
async def ask(body: Question):
    question = body.question.strip()
    requested_models = body.models if body.models is not None else MODELS
    selected_models = list(dict.fromkeys(name for name in requested_models if name in MODELS))
    if not selected_models:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No valid models selected")

    async def stream_results():
        delta_queue = asyncio.Queue()
        task_map = {
            asyncio.create_task(request_ai(name, question, delta_queue)): name
            for name in selected_models
        }
        successful_answers = {
            name: answer for name, answer in (body.previous_answers or {}).items()
            if name in MODELS and isinstance(answer, str) and answer.strip()
        }
        try:
            pending = set(task_map)
            while pending:
                # Wait briefly so streamed Gemini chunks can be forwarded immediately.
                done, pending = await asyncio.wait(
                    pending, timeout=0.08, return_when=asyncio.FIRST_COMPLETED
                )
                while not delta_queue.empty():
                    event = delta_queue.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                for task in done:
                    try:
                        result = task.result()
                    except asyncio.CancelledError:
                        continue
                    except Exception as exc:
                        result = {"model": task_map[task], "ok": False, "answer": "",
                                  "error": f"{type(exc).__name__}: {str(exc) or 'Request failed'}"}
                    if result.get("ok"):
                        successful_answers[result["model"]] = result.get("answer", "")
                    yield f"data: {json.dumps(result)}\n\n"
            while not delta_queue.empty():
                event = delta_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            if len(successful_answers) >= 2:
                yield f"data: {json.dumps(evaluate_answers(question, successful_answers))}\n\n"
        finally:
            for task in task_map:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*task_map, return_exceptions=True)

    return StreamingResponse(
        stream_results(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
