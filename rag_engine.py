import os
import glob
import json
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openrouter")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b")

CHUNK_SIZE = 500
COLLECTION_NAME = "customerbot_kb"
KB_DIR = Path(__file__).parent / "kb"


class OpenRouterEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        import httpx

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        texts = [t for t in input]
        resp = httpx.post(
            f"{OPENROUTER_BASE}/embeddings",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBED_MODEL, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        rows = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]

    @staticmethod
    def name() -> str:
        return "openrouter_embeddings"

    def get_config(self) -> dict:
        return {"model": EMBED_MODEL}

    @staticmethod
    def build_from_config(config: dict) -> "OpenRouterEmbeddingFunction":
        return OpenRouterEmbeddingFunction()


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        import httpx

        out = []
        for text in input:
            resp = httpx.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        return out

    @staticmethod
    def name() -> str:
        return "ollama_embeddings"

    def get_config(self) -> dict:
        return {"model": OLLAMA_EMBED_MODEL}

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction()


def _chunk_text(text: str, source: str) -> List[dict]:
    chunks = []
    paragraphs = text.split("\n\n")
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) < CHUNK_SIZE:
            if buffer:
                buffer += "\n\n"
            buffer += para
        else:
            if buffer:
                chunks.append({"text": buffer, "source": source})
            buffer = para
    if buffer:
        chunks.append({"text": buffer, "source": source})
    return chunks


def _load_kb_documents() -> List[dict]:
    chunks = []
    for md_file in glob.glob(str(KB_DIR / "*.md")):
        source = Path(md_file).stem
        content = Path(md_file).read_text(encoding="utf-8")
        chunks.extend(_chunk_text(content, source))
    return chunks


class RAGEngine:
    """Retrieval over kb/*.md. Uses an in-memory Chroma collection rebuilt per
    cold start, because Vercel functions have a read-only filesystem."""

    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.ef = (
            OllamaEmbeddingFunction()
            if EMBEDDING_PROVIDER == "ollama"
            else OpenRouterEmbeddingFunction()
        )
        self.collection = None

    def _get_or_create_collection(self):
        existing = [c.name for c in self.client.list_collections()]
        if COLLECTION_NAME in existing:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME, embedding_function=self.ef
            )
        else:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME, embedding_function=self.ef
            )
        return self.collection

    def index(self, force: bool = False):
        collection = self._get_or_create_collection()
        if not force and collection.count() > 0:
            return
        if force:
            self.client.delete_collection(COLLECTION_NAME)
            collection = self._get_or_create_collection()
        chunks = _load_kb_documents()
        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk["text"])
            metadatas.append({"source": chunk["source"]})
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        collection = self._get_or_create_collection()
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []

    def query(self, query: str, n_results: int = 3) -> tuple[str, List[str]]:
        context_docs = self.retrieve(query, n_results=n_results)
        context = "\n\n".join(context_docs) if context_docs else ""
        return context, context_docs

    def query_llm(self, prompt: str) -> str:
        if LLM_PROVIDER == "ollama":
            return self._query_ollama(prompt)
        return self._query_openrouter(prompt)

    def _query_ollama(self, prompt: str) -> str:
        import httpx

        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def _query_openrouter(self, prompt: str) -> str:
        import httpx

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        resp = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    async def stream_llm(self, prompt: str):
        """Yields text deltas. Falls back to one big chunk if the provider
        cannot stream."""
        import httpx

        if LLM_PROVIDER == "ollama":
            payload = {
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.3, "num_predict": 1024},
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/api/generate", json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        token = json.loads(line).get("response", "")
                        if token:
                            yield token
            return

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = parsed.get("choices") or []
                    if not choices:
                        continue
                    token = choices[0].get("delta", {}).get("content", "")
                    if token:
                        yield token