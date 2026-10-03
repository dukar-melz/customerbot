# CustomerBot — RAG Customer Support Agent

A retrieval-augmented customer support chatbot for **AcmeShop**, an online electronics and
accessories store. Visitors ask questions in natural language; the bot retrieves relevant
passages from a markdown knowledge base and answers using only that context.

Built with **FastAPI**, **ChromaDB**, and **OpenRouter**. Deployed on **Vercel**.

---

## How it works

```
question
   │
   ├─► embed the question            (OpenRouter · text-embedding-3-small)
   │
   ├─► cosine search over kb/*.md    (ChromaDB · in-memory collection)
   │        └─ top 3 chunks + their source documents
   │
   ├─► build a grounded prompt       (bot.py · system rules + context + history)
   │
   └─► stream the answer             (OpenRouter · llama-3.3-70b)
            └─ tokens streamed to the browser over SSE
```

The system prompt in `bot.py` instructs the model to answer **only** from the retrieved
context, to never invent tracking numbers, dates or policies, and to escalate to a human for
high-value payment disputes, legal threats, or repeated unresolved issues.

### Retrieval

`rag_engine.py` chunks every `kb/*.md` file on paragraph boundaries (~500 chars), embeds each
chunk, and stores them in a ChromaDB collection. The question is embedded with the same model
and compared against the collection; the closest 3 chunks become the context window.

The collection is held **in memory** and rebuilt from the markdown files on cold start. This is
deliberate: Vercel functions have a read-only filesystem, so a persistent on-disk vector store
is not an option there. The knowledge base is only 4 documents / ~35 chunks, so re-embedding it
at cold start costs one batched API call.

### Analytics

Every question, answer, and the source documents used to produce it are logged, along with
thumbs-up / thumbs-down feedback. The `/analytics` page visualises total queries, sentiment,
and which knowledge base documents are actually being used.

---

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI |
| Retrieval | ChromaDB (in-memory, cosine) |
| Embeddings | OpenRouter → `openai/text-embedding-3-small` |
| Generation | OpenRouter → `meta-llama/llama-3.3-70b-instruct:free` |
| Analytics | Supabase Postgres via PostgREST |
| Hosting | Vercel (Python functions) |

---

## Local setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # then fill in OPENROUTER_API_KEY
uv run uvicorn app:app --reload
```

Open http://localhost:8000.

### Running fully offline with Ollama

The original build used a local Ollama instance. That still works for offline development —
set both providers to `ollama` in `.env`:

```bash
EMBEDDING_PROVIDER=ollama
LLM_PROVIDER=ollama
```

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

This is how the app was tested end-to-end. Deployment uses OpenRouter, since Vercel cannot
reach a localhost Ollama server.

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Authenticates both chat and embedding calls |
| `LLM_MODEL` | no | Chat model id. Defaults to a free-tier Llama |
| `EMBED_MODEL` | no | Embedding model id. Defaults to `text-embedding-3-small` |
| `SUPABASE_URL` | no | Analytics database. Blank → local `analytics.db` |
| `SUPABASE_SERVICE_ROLE_KEY` | no | Analytics database auth |
| `EMBEDDING_PROVIDER` | no | `openrouter` (default) or `ollama` |
| `LLM_PROVIDER` | no | `openrouter` (default) or `ollama` |

Leaving the Supabase variables blank falls back to a local SQLite file, so the app runs with
no external database during development.

---

## Deploying to Vercel

1. Create a Supabase project and run `supabase/schema.sql` in the SQL editor.
2. Push this repo to GitHub and import it into Vercel (framework preset: **FastAPI**).
3. Add `OPENROUTER_API_KEY`, `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` as environment
   variables.
4. Deploy.

`vercel.json` routes every path to the single Python function in `api/index.py`, which exports
the FastAPI `app`. Vercel detects FastAPI and wraps it with an ASGI adapter automatically.

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/chat` | Chat interface |
| `GET` | `/products` | Product catalogue |
| `GET` | `/payment` | Payment page |
| `GET` | `/returns` | Returns policy |
| `GET` | `/about` | About page |
| `GET` | `/analytics` | Analytics dashboard |
| `POST` | `/chat` | Ask a question, get a full answer as JSON |
| `POST` | `/chat/stream` | Ask a question, receive tokens over SSE |
| `POST` | `/feedback` | Record thumbs up/down for a log entry |
| `GET` | `/api/products` | Product catalogue as JSON |
| `GET` | `/api/analytics` | Aggregate query and sentiment stats |
| `GET` | `/api/analytics/logs` | Recent query log |
| `GET` | `/docs` | Interactive OpenAPI docs |

`POST /chat` accepts:

```json
{
  "message": "How long do I have to return a product?",
  "session_id": "abc123",
  "history": ["previous message", "previous reply"]
}
```

---

## Project layout

```
app.py            FastAPI app: routes, HTML pages, SSE streaming
rag_engine.py     Chunking, embedding, Chroma retrieval, OpenRouter generation
bot.py            System prompt and prompt assembly
products.py       Product catalogue data
analytics.py      Query logging (Supabase in prod, SQLite locally)
kb/               Knowledge base — products, returns, shipping, account
static/           Seven self-contained HTML pages
api/index.py      Vercel function entrypoint
supabase/schema.sql
```