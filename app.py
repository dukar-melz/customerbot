import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rag_engine import RAGEngine
from bot import build_prompt
from analytics import log_query, set_feedback, get_stats, get_recent
from products import PRODUCTS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

engine = RAGEngine()
engine_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Indexing needs an embedding provider. If it is unavailable the app should
    # still boot so the static pages render; only /chat will fail.
    global engine_error
    try:
        await run_in_threadpool(engine.index)
    except Exception as exc:  # noqa: BLE001
        engine_error = str(exc)
        print(f"WARNING: knowledge base not indexed at startup: {exc}")
    yield


app = FastAPI(title="CustomerBot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "indexed": bool(engine.collection and engine.collection.count() > 0),
        "error": engine_error,
    }


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def index():
    return _page("home.html")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return _page("index.html")


@app.get("/products", response_class=HTMLResponse)
async def products_page():
    return _page("products.html")


@app.get("/payment", response_class=HTMLResponse)
async def payment_page():
    return _page("payment.html")


@app.get("/returns", response_class=HTMLResponse)
async def returns_page():
    return _page("returns.html")


@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return _page("about.html")


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page():
    return _page("analytics.html")


@app.get("/api/analytics")
async def analytics_api():
    return await run_in_threadpool(get_stats)


@app.get("/api/analytics/logs")
async def analytics_logs():
    return await run_in_threadpool(get_recent)


@app.get("/api/products")
async def products_api():
    return PRODUCTS


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    message = body["message"]
    history = body.get("history", [])
    session_id = body.get("session_id", "anon")

    context, sources = await run_in_threadpool(engine.query, message)
    prompt = build_prompt(message, context, history=history)
    response = await run_in_threadpool(engine.query_llm, prompt)

    await run_in_threadpool(log_query, session_id, message, response, sources)
    return {"response": response}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    message = body["message"]
    history = body.get("history", [])
    session_id = body.get("session_id", "anon")

    context, sources = await run_in_threadpool(engine.query, message)
    prompt = build_prompt(message, context, history=history)

    async def generate():
        full_response = ""
        try:
            async for token in engine.stream_llm(prompt):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        finally:
            await run_in_threadpool(
                log_query, session_id, message, full_response, sources
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/feedback")
async def feedback(request: Request):
    body = await request.json()
    await run_in_threadpool(set_feedback, body["log_id"], body["value"])
    return {"ok": True}