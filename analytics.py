import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
DB_PATH = Path(os.environ.get("ANALYTICS_DB_PATH", Path(__file__).parent / "analytics.db"))

TABLE = "logs"


def _sqlite_writable() -> bool:
    """SQLite is a local-dev convenience only. Serverless filesystems are
    read-only, so probing once here keeps chat working without Supabase."""
    if _use_supabase():
        return False
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DB_PATH, "a"):
            pass
        return True
    except OSError:
        return False


_sqlite_ok: bool | None = None


def _can_use_sqlite() -> bool:
    global _sqlite_ok
    if _sqlite_ok is None:
        _sqlite_ok = _sqlite_writable()
    return _sqlite_ok


def _use_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _headers(extra: dict | None = None) -> dict:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _rest(path: str) -> str:
    return f"{SUPABASE_URL.rstrip('/')}/rest/v1/{TABLE}{path}"


# --------------------------------------------------------------------------
# Supabase (PostgREST) backend — used in production
# --------------------------------------------------------------------------


def _pg_log_query(session_id: str, query: str, response: str, sources: list[str]):
    import httpx

    row = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "response": response,
        "sources": json.dumps(sources),
        "feedback": None,
    }
    resp = httpx.post(
        _rest(""),
        headers=_headers({"Prefer": "return=minimal"}),
        json=row,
        timeout=20,
    )
    resp.raise_for_status()


def _pg_set_feedback(log_id: str, feedback: int):
    import httpx

    resp = httpx.patch(
        _rest(f"?id=eq.{log_id}"),
        headers=_headers({"Prefer": "return=minimal"}),
        json={"feedback": feedback},
        timeout=20,
    )
    resp.raise_for_status()


def _pg_get_stats() -> dict:
    import httpx

    def rows(query: str) -> list[dict]:
        resp = httpx.get(
            _rest(query), headers=_headers(), timeout=20
        )
        resp.raise_for_status()
        return resp.json()

    total = rows("?select=id")
    with_feedback = rows("?select=feedback&feedback=not.is.null")
    sources_raw = rows("?select=sources&sources=not.is.null")

    up = sum(1 for r in with_feedback if r["feedback"] == 1)
    down = sum(1 for r in with_feedback if r["feedback"] == -1)

    source_counts: dict[str, int] = {}
    for row in sources_raw:
        try:
            for s in json.loads(row["sources"]):
                source_counts[s] = source_counts.get(s, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue

    return {
        "total_queries": len(total),
        "thumbs_up": up,
        "thumbs_down": down,
        "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:10],
    }


def _pg_get_recent(limit: int = 50) -> list[dict]:
    import httpx

    resp = httpx.get(
        _rest(
            "?select=id,session_id,timestamp,query,response,feedback"
            f"&order=timestamp.desc&limit={limit}"
        ),
        headers=_headers(),
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# SQLite backend — used for local development when Supabase is not configured
# --------------------------------------------------------------------------


def _get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute(
        """CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp TEXT,
            query TEXT,
            response TEXT,
            sources TEXT,
            feedback INTEGER
        )"""
    )
    db.commit()
    return db


def _sq_log_query(session_id: str, query: str, response: str, sources: list[str]):
    db = _get_db()
    db.execute(
        "INSERT INTO logs (id, session_id, timestamp, query, response, sources) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            session_id,
            datetime.now(timezone.utc).isoformat(),
            query,
            response,
            json.dumps(sources),
        ),
    )
    db.commit()
    db.close()


def _sq_set_feedback(log_id: str, feedback: int):
    db = _get_db()
    db.execute("UPDATE logs SET feedback = ? WHERE id = ?", (feedback, log_id))
    db.commit()
    db.close()


def _sq_get_stats() -> dict:
    db = _get_db()
    total = db.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    with_feedback = db.execute(
        "SELECT feedback, COUNT(*) FROM logs WHERE feedback IS NOT NULL GROUP BY feedback"
    ).fetchall()
    up = sum(r[1] for r in with_feedback if r[0] == 1)
    down = sum(r[1] for r in with_feedback if r[0] == -1)
    sources_raw = db.execute(
        "SELECT sources FROM logs WHERE sources IS NOT NULL"
    ).fetchall()
    source_counts = {}
    for row in sources_raw:
        for s in json.loads(row[0]):
            source_counts[s] = source_counts.get(s, 0) + 1
    db.close()
    return {
        "total_queries": total,
        "thumbs_up": up,
        "thumbs_down": down,
        "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:10],
    }


def _sq_get_recent(limit: int = 50) -> list[dict]:
    db = _get_db()
    rows = db.execute(
        "SELECT id, session_id, timestamp, query, response, feedback FROM logs ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Public API — picks a backend based on whether Supabase is configured
# --------------------------------------------------------------------------


def analytics_backend() -> str:
    if _use_supabase():
        return "supabase"
    if _can_use_sqlite():
        return "sqlite"
    return "disabled"


def _warn(op: str, exc: Exception):
    print(f"WARNING: analytics.{op} failed: {type(exc).__name__}: {exc}")


def log_query(session_id: str, query: str, response: str, sources: list[str]):
    try:
        if _use_supabase():
            _pg_log_query(session_id, query, response, sources)
        elif _can_use_sqlite():
            _sq_log_query(session_id, query, response, sources)
        else:
            _warn("log_query", RuntimeError("no analytics backend configured"))
    except Exception as exc:  # noqa: BLE001
        _warn("log_query", exc)


def set_feedback(log_id: str, feedback: int):
    try:
        if _use_supabase():
            _pg_set_feedback(log_id, feedback)
        elif _can_use_sqlite():
            _sq_set_feedback(log_id, feedback)
        else:
            return {"ok": False, "error": "analytics backend unavailable"}
    except Exception as exc:  # noqa: BLE001
        _warn("set_feedback", exc)
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def get_stats() -> dict:
    try:
        if _use_supabase():
            return _pg_get_stats()
        if _can_use_sqlite():
            return _sq_get_stats()
    except Exception as exc:  # noqa: BLE001
        _warn("get_stats", exc)
    return {
        "total_queries": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "top_sources": [],
        "backend": "unavailable",
    }


def get_recent(limit: int = 50) -> list[dict]:
    try:
        if _use_supabase():
            return _pg_get_recent(limit)
        if _can_use_sqlite():
            return _sq_get_recent(limit)
    except Exception as exc:  # noqa: BLE001
        _warn("get_recent", exc)
    return []