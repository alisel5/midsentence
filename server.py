"""
midsentence — your LLM call isn't atomic, it's a conversation paused mid-sentence.

A drop-in proxy for OpenAI-compatible chat-completions APIs. Captures every call,
lets you replay it against the same model with follow-up questions ("debrief")
so you can ask the model WHY it picked tool A over tool B, WHICH prompt lines
steered it, and so on.

Run:
    cp .env.example .env   # set MIDSENTENCE_API_KEY
    pip install -r requirements.txt
    python server.py

Then point your app's LLM base URL at http://localhost:8765/v1 and start
shipping calls. Open http://localhost:8765 to inspect + debrief.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "midsentence.db"
STATIC_DIR = ROOT / "static"

# Env vars are now only bootstrap defaults — the UI owns the live config.
# When the DB has a non-empty value for a key, it wins. Env stays as a
# first-run seed so people who prefer .env don't need to touch the UI.
UPSTREAM_BASE_DEFAULT = os.environ.get("MIDSENTENCE_UPSTREAM", "https://api.openai.com").rstrip("/")
UPSTREAM_API_KEY_ENV = os.environ.get("MIDSENTENCE_API_KEY", "")
PORT = int(os.environ.get("MIDSENTENCE_PORT", "8765"))


# ─────────────────────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS captures (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            model TEXT,
            upstream TEXT NOT NULL,
            request_body TEXT NOT NULL,
            response_body TEXT,
            response_status INTEGER,
            latency_ms INTEGER,
            error TEXT,
            last_thought TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_captures_created_at ON captures(created_at DESC);

        CREATE TABLE IF NOT EXISTS debriefs (
            id TEXT PRIMARY KEY,
            capture_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            FOREIGN KEY (capture_id) REFERENCES captures(id)
        );
        CREATE INDEX IF NOT EXISTS idx_debriefs_capture ON debriefs(capture_id, created_at ASC);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
    """)
    # Best-effort migration for DBs that predate the last_thought column.
    try:
        conn.execute("ALTER TABLE captures ADD COLUMN last_thought TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists; new installs hit the CREATE above.
    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "") -> str:
    conn = db_connect()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, time.time()),
    )
    conn.commit()
    conn.close()


def current_upstream() -> str:
    """DB value wins; env falls in as bootstrap seed; OpenAI default as last resort.

    Returns the user-entered base URL verbatim (minus trailing slash). The
    actual endpoint URL is computed by resolve_chat_endpoint() below.
    """
    db_val = get_setting("upstream").strip()
    if db_val:
        return db_val.rstrip("/")
    return UPSTREAM_BASE_DEFAULT


def resolve_chat_endpoint() -> str:
    """Return the fully-qualified chat-completions URL, regardless of which
    form the user pasted for `upstream`. Accepts all of:

      - https://api.openai.com                      -> append /v1/chat/completions
      - https://api.openai.com/                     -> append /v1/chat/completions
      - https://api.openai.com/v1                   -> append /chat/completions
      - https://api.openai.com/v1/chat/completions  -> use verbatim

    Spares users the URL-construction gotcha where pasting the OpenRouter /v1
    base produces a doubled /v1/v1/ path.
    """
    base = current_upstream()
    # Already a full chat-completions URL? Use verbatim.
    if base.endswith("/chat/completions"):
        return base
    # Has a version segment like /v1, /v1beta etc.? Only append the tail.
    tail = base.rsplit("/", 1)[-1].lower()
    if tail.startswith("v") and tail[1:2].isdigit():
        return f"{base}/chat/completions"
    # Bare host / path — assume OpenAI-style /v1 prefix.
    return f"{base}/v1/chat/completions"


def current_debrief_key() -> str:
    """DB value wins; env is the bootstrap seed. Empty string means unset."""
    db_val = get_setting("debrief_api_key").strip()
    if db_val:
        return db_val
    return UPSTREAM_API_KEY_ENV


def capture_last_thought_enabled() -> bool:
    """Read the last-thought toggle from settings. Default: off (pure proxy)."""
    return get_setting("capture_last_thought").strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# Last-thought mode — opt-in pre-action thought capture.
# ─────────────────────────────────────────────────────────────────────────────
# When enabled, the proxy injects a system message forcing the model to emit a
# single "LAST_THOUGHT: ..." line right before any tool_call or final content.
# We extract that line from the response and store it as its own column on the
# capture row, so the debrief UI can anchor questions on a real mid-decision
# utterance instead of asking for post-hoc rationalization.
#
# Trade-off: this mutates outgoing requests (no longer a pure transparent
# proxy) and leaves the "LAST_THOUGHT:" prefix visible in the streamed
# response for clients that don't strip it. Opt-in only.

LAST_THOUGHT_RULE = (
    "LAST_THOUGHT mode (harmless debug hook): before your normal response, "
    "mutter one line out loud saying what you were just about to do. Format:\n"
    "LAST_THOUGHT: <one short sentence, your last thought before committing>\\n"
    "then continue with the reply (or tool_call) EXACTLY as you would have "
    "given it normally — don't think longer, don't rephrase the work, don't "
    "change the tools you were going to call. It's just a mic being held up "
    "to capture the last thought; the job itself is unchanged.\n"
    "Rules: one line, keep it short, include a trailing newline before your "
    "real response, never skip it (emit as content even when the round is "
    "tool_calls-only). That's it."
)


def inject_last_thought_rule(messages: list) -> list:
    """Inject the LAST_THOUGHT rule so the model sees it IMMEDIATELY before
    its next turn. Previous design (appending as a system message after the
    last existing system) worked on round 1 but faded on round 2+ as the
    conversation grew and tool-result context dominated attention. Putting
    the reminder on the very last user/tool message — the last thing the
    model reads before generating — ensures it can't be missed on any round.

    Also keeps a copy at the system level for any provider that weighs
    system messages heavier (redundant but cheap).
    """
    reminder = "\n\n[MIDSENTENCE DEBUG — LAST_THOUGHT MODE ACTIVE]\n" + LAST_THOUGHT_RULE
    rule_msg = {"role": "system", "content": LAST_THOUGHT_RULE}

    # Shallow-copy so we don't mutate the caller's list or the stored body.
    out = [dict(m) for m in (messages or [])]

    # 1. Add as system message right after the last existing system.
    last_sys_idx = -1
    for i, m in enumerate(out):
        if m.get("role") == "system":
            last_sys_idx = i
    if last_sys_idx >= 0:
        out = out[: last_sys_idx + 1] + [rule_msg] + out[last_sys_idx + 1 :]
    else:
        out = [rule_msg] + out

    # 2. Suffix the last non-system message (user or tool) so the rule is
    # the final thing the model reads before its turn. Each round this gets
    # re-appended to whichever message is currently last — no drift.
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        role = m.get("role")
        if role in ("user", "tool"):
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = content + reminder
            elif isinstance(content, list):
                m["content"] = list(content) + [{"type": "text", "text": reminder}]
            else:
                m["content"] = reminder
            break

    return out


def extract_last_thought(content: str | None) -> str:
    """Pull the first LAST_THOUGHT: line out of a response's content string.
    Returns the thought text (without the prefix), or '' if not present."""
    if not content:
        return ""
    m = re.match(r"\s*LAST_THOUGHT:\s*(.+?)(?:\r?\n|$)", content, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# Regex matches "LAST_THOUGHT: <anything>" at the start of content, with
# optional leading whitespace and an optional trailing newline. Consumes the
# newline if present (so the remaining content doesn't start with a blank line).
_LAST_THOUGHT_STRIP_RE = re.compile(
    r"^\s*LAST_THOUGHT:[^\n]*(?:\r?\n|$)",
    re.IGNORECASE,
)


def _strip_last_thought_prefix(text: str) -> str:
    """Strip a leading LAST_THOUGHT: line from a text block. If not present,
    returns the original text unchanged."""
    if not text:
        return text
    return _LAST_THOUGHT_STRIP_RE.sub("", text, count=1)


def scrub_last_thought_from_history(messages: list) -> list:
    """Belt-and-suspenders: before forwarding to upstream, remove any
    LAST_THOUGHT: prefix line from the CONTENT of every assistant message in
    the incoming conversation history. Leaves non-assistant messages alone.

    This matters because even with stream-stripping on the outbound response,
    edge cases (model emitting the thought with no trailing newline, client
    buffering quirks) can leave the prefix in the client's stored history.
    When the client replays that history on the next round, the upstream
    model would see "assistant said 'LAST_THOUGHT: ...'" as real prior
    context — which either reinforces the pattern (fine) or starts feeling
    odd if the rule is later disabled. Strip it always so the upstream never
    ingests our own debug metadata as conversation."""
    out = []
    for m in messages or []:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        m_copy = dict(m)
        content = m_copy.get("content")
        if isinstance(content, str):
            m_copy["content"] = _strip_last_thought_prefix(content)
        elif isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    new_blocks.append({**block, "text": _strip_last_thought_prefix(block.get("text") or "")})
                else:
                    new_blocks.append(block)
            m_copy["content"] = new_blocks
        out.append(m_copy)
    return out


# ── Circuit breaker ─────────────────────────────────────────────────────────
# When the upstream keeps returning errors, stop forwarding so we don't:
#   (a) rack up charges on auth/rate-limit storms
#   (b) hammer the upstream with guaranteed-fail requests
#   (c) fill the UI with 30 identical failure rows
# After FAILURE_THRESHOLD consecutive non-2xx captures, the circuit opens.
# It closes on: a settings change, a manual reset, OR a successful call.

FAILURE_THRESHOLD = 3


def circuit_state() -> dict:
    """Return {'open': bool, 'reason': str, 'last_status': int|None}."""
    reset_at = float(get_setting("circuit_reset_at", "0") or "0")
    conn = db_connect()
    rows = conn.execute(
        "SELECT response_status, error, created_at FROM captures "
        "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
        (reset_at, FAILURE_THRESHOLD),
    ).fetchall()
    conn.close()
    if len(rows) < FAILURE_THRESHOLD:
        return {"open": False, "reason": "", "last_status": None}
    all_failing = all((r["response_status"] or 0) >= 400 for r in rows)
    if not all_failing:
        return {"open": False, "reason": "", "last_status": rows[0]["response_status"]}
    last = rows[0]
    reason = (
        f"{FAILURE_THRESHOLD} consecutive failures. Last status: "
        f"{last['response_status']}. Last error: {(last['error'] or '')[:200]}"
    )
    return {"open": True, "reason": reason, "last_status": last["response_status"]}


def reset_circuit() -> None:
    set_setting("circuit_reset_at", str(time.time()))


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_capture(
    capture_id: str,
    model: str,
    upstream: str,
    request_body: dict,
    response_body: dict | None,
    response_status: int,
    latency_ms: int,
    error: str | None = None,
    last_thought: str | None = None,
) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO captures (id, created_at, model, upstream, request_body, "
        "response_body, response_status, latency_ms, error, last_thought) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            capture_id,
            time.time(),
            model,
            upstream,
            json.dumps(request_body, ensure_ascii=False),
            json.dumps(response_body, ensure_ascii=False) if response_body is not None else None,
            response_status,
            latency_ms,
            error,
            last_thought or None,
        ),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Debrief message construction
# ─────────────────────────────────────────────────────────────────────────────

DEBRIEF_PREFIX = (
    "You are a prompt-citation audit tool, not a conversational assistant. "
    "Your previous response is already delivered and cannot be changed. "
    "Your job now is to identify which parts of the prompt, context, or "
    "tool_results produced that response — for audit purposes only. "
    "\n\nDebrief question: "
)

DEBRIEF_SUFFIX = (
    "\n\nOutput rules (strict):\n"
    "- Quote verbatim from the system prompt, user messages, or tool_results. "
    "No paraphrasing.\n"
    "- Do NOT apologize, self-correct, or propose what you 'should have' done.\n"
    "- Do NOT use phrases like 'better yet', 'actually, I should have', "
    "'that was unnecessary', 'the correct choice would be', 'my mistake', "
    "'in hindsight'.\n"
    "- If no prompt line supports a choice, output exactly: NO_SUPPORTING_LINE. "
    "Silence on a case is a valid finding — do not manufacture a citation.\n"
    "- Do NOT emit tool_calls. This is introspection, not execution.\n"
    "- Keep it tight: max 3 citations per decision unless explicitly asked for more."
)

TOOL_STUB_CONTENT = (
    "[debrief mode: this tool call was captured from the original run, not "
    "executed. Do not re-emit tool_calls. Answer the next user message with "
    "prose analysis only, grounded in the prompt and conversation history.]"
)


def build_debrief_messages(
    request_body: dict,
    response_body: dict | None,
    prior_debriefs: list[sqlite3.Row],
    new_question: str,
) -> list[dict]:
    """Reconstruct the full conversation as the model saw it + assistant's reply
    + any prior debrief turns + the new user question. Same model, same system,
    same conversation history. To the model it is literally a continued
    conversation, now in analysis mode.
    """
    messages: list[dict] = [dict(m) for m in (request_body.get("messages") or [])]

    # Append the captured assistant reply
    if response_body:
        choice = (response_body.get("choices") or [{}])[0]
        asst_msg = dict(choice.get("message") or {})
        # Some providers return content=null when only tool_calls exist. Normalize.
        if asst_msg.get("content") is None:
            asst_msg["content"] = ""
        # Keep the message as-is so the model "sees" exactly what it emitted.
        if asst_msg:
            messages.append(asst_msg)

        # If the captured response included tool_calls, we must append tool
        # results for shape-validity AND to mark debrief mode clearly.
        tool_calls = asst_msg.get("tool_calls") or []
        for tc in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "content": TOOL_STUB_CONTENT,
            })

    # Append any prior debrief turns on this capture (multi-turn support)
    for d in prior_debriefs:
        messages.append({"role": "user", "content": DEBRIEF_PREFIX + d["question"] + DEBRIEF_SUFFIX})
        if d["answer"]:
            messages.append({"role": "assistant", "content": d["answer"]})

    # Append the new debrief question
    messages.append({
        "role": "user",
        "content": DEBRIEF_PREFIX + new_question + DEBRIEF_SUFFIX,
    })

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

init_db()

app = FastAPI(
    title="midsentence",
    description="Your LLM call isn't atomic — it's a conversation paused mid-sentence.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "upstream": current_upstream(), "debrief_key_set": bool(current_debrief_key())}


# ── Settings API ────────────────────────────────────────────────────────────
# The UI owns the live config. Env vars are bootstrap seeds only; once the DB
# has a value for a key, that wins. Changing settings via the UI takes effect
# on the NEXT request (no restart needed) because current_upstream() and
# current_debrief_key() read the DB per call.

class SettingsBody(BaseModel):
    upstream: str | None = None
    debrief_api_key: str | None = None
    capture_last_thought: bool | None = None


@app.get("/api/settings")
def api_get_settings() -> dict:
    return {
        "upstream": current_upstream(),
        # Never return the actual key — just whether one is set
        "debrief_key_set": bool(current_debrief_key()),
        "capture_last_thought": capture_last_thought_enabled(),
        "circuit": circuit_state(),
    }


@app.post("/api/settings")
def api_post_settings(body: SettingsBody) -> dict:
    if body.upstream is not None:
        cleaned = body.upstream.strip().rstrip("/")
        set_setting("upstream", cleaned)
    if body.debrief_api_key is not None:
        # Trim whitespace; empty string clears the key (and falls back to env).
        set_setting("debrief_api_key", body.debrief_api_key.strip())
    if body.capture_last_thought is not None:
        set_setting("capture_last_thought", "1" if body.capture_last_thought else "0")
    # Any config change implicitly resets the circuit — the user just fixed
    # something, give them a clean slate to test.
    reset_circuit()
    return api_get_settings()


@app.post("/api/reset-circuit")
def api_reset_circuit() -> dict:
    reset_circuit()
    return {"ok": True, "circuit": circuit_state()}


# ── Proxy endpoint ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request) -> Any:
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}")

    capture_id = f"cap-{uuid.uuid4().hex[:12]}"
    model = body.get("model", "")
    is_stream = bool(body.get("stream"))

    # Forward the client's own Authorization header — the proxy never stores
    # or rewrites the caller's API key. If the client didn't send one, upstream
    # will reject; not our problem.
    fwd_headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        fwd_headers["Authorization"] = auth
    for h in ("anthropic-version", "openai-organization", "openai-beta", "x-api-key"):
        v = request.headers.get(h)
        if v:
            fwd_headers[h] = v

    endpoint = resolve_chat_endpoint()
    start = time.time()

    # ── Last-thought injection (opt-in) ──────────────────────────────────
    # Mutate a COPY of the body so we store the original request verbatim
    # in the capture but send the thought-rule-augmented version to upstream.
    # Also defensively strip any leftover LAST_THOUGHT: prefixes from
    # assistant history before forwarding, so upstream never ingests our own
    # debug metadata as conversation context.
    outgoing_body = body
    if capture_last_thought_enabled():
        outgoing_body = dict(body)
        scrubbed = scrub_last_thought_from_history(body.get("messages") or [])
        outgoing_body["messages"] = inject_last_thought_rule(scrubbed)

    if is_stream:
        return await _proxy_chat_stream(
            body=body,
            outgoing_body=outgoing_body,
            headers=fwd_headers,
            endpoint=endpoint,
            capture_id=capture_id,
            model=model,
            start=start,
        )

    # Non-stream: simple buffered pass-through
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(endpoint, json=outgoing_body, headers=fwd_headers)
        latency_ms = int((time.time() - start) * 1000)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            data = {"_raw_text": resp.text}

        # Extract the LAST_THOUGHT: line from the assistant reply (if present)
        last_thought = ""
        if resp.status_code == 200 and isinstance(data, dict):
            asst_content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            last_thought = extract_last_thought(asst_content)

        save_capture(
            capture_id=capture_id,
            model=model,
            upstream=current_upstream(),
            request_body=body,  # store the ORIGINAL body, not the injected one
            response_body=data if resp.status_code < 500 else None,
            response_status=resp.status_code,
            latency_ms=latency_ms,
            error=None if resp.status_code < 400 else (resp.text[:500] if resp.text else "http error"),
            last_thought=last_thought,
        )
        return JSONResponse(content=data, status_code=resp.status_code)

    except httpx.HTTPError as e:
        latency_ms = int((time.time() - start) * 1000)
        save_capture(
            capture_id=capture_id,
            model=model,
            upstream=current_upstream(),
            request_body=body,
            response_body=None,
            response_status=0,
            latency_ms=latency_ms,
            error=str(e),
        )
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")


async def _proxy_chat_stream(
    body: dict,
    outgoing_body: dict,
    headers: dict,
    endpoint: str,
    capture_id: str,
    model: str,
    start: float,
) -> StreamingResponse:
    """SSE pass-through with in-flight assembly for capture.

    Forwards upstream SSE chunks to the client byte-for-byte while concurrently
    parsing deltas to assemble a synthetic chat.completion payload that we
    store as the capture's response_body. That way the client sees real
    streaming behaviour AND the captures table has a clean non-stream-shaped
    response for inspection + debrief.
    """

    async def generate() -> Any:
        assembled_content_parts: list[str] = []
        assembled_tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None
        response_id: str | None = None
        response_created: int | None = None
        response_model = model
        status_code = 200
        error_text = ""
        pending = b""
        # Hoisted so the finally block can cancel it even if we never
        # reached the point of creating one (e.g., early error in connect).
        read_task: asyncio.Task | None = None

        client = httpx.AsyncClient(timeout=180)
        try:
            async with client.stream("POST", endpoint, json=outgoing_body, headers=headers) as resp:
                status_code = resp.status_code
                if resp.status_code != 200:
                    # Upstream rejected — collect the error body, forward as SSE
                    # error frame so the client sees something useful.
                    body_bytes = b""
                    async for b in resp.aiter_bytes():
                        body_bytes += b
                    error_text = body_bytes.decode("utf-8", errors="replace")
                    frame = ("data: " + json.dumps({"error": error_text[:500]}) + "\n\n").encode("utf-8")
                    yield frame
                    yield b"data: [DONE]\n\n"
                    return

                # Last-thought stripping state. When the LAST_THOUGHT mode is
                # on, we HOLD early SSE chunks instead of forwarding them
                # verbatim, so the "LAST_THOUGHT:" preamble never reaches the
                # downstream client. Once we see a newline in the assembled
                # content we decide:
                #   - preamble matches LAST_THOUGHT:  → drop held chunks,
                #     emit ONE synthetic chunk containing the post-newline
                #     content. Subsequent chunks pass through verbatim
                #     (each carries new content deltas only).
                #   - preamble does NOT match        → flush held chunks
                #     verbatim and continue forwarding.
                #   - no newline after 300 chars     → flush and assume
                #     not LAST_THOUGHT (safety).
                strip_mode = capture_last_thought_enabled()
                preamble_decided = not strip_mode
                held_chunks: list[bytes] = []

                # Keepalive loop. Slow upstreams (e.g., the PressArk dev
                # simulator which waits minutes on a human observer) can
                # starve downstream clients whose HTTP read-timeout would
                # otherwise trip (PressArk's is 120s by default). SSE spec
                # lines starting with `:` are comments — clients ignore
                # them but the TCP activity keeps the connection alive.
                # Keepalives bypass the hold-strip buffer (yielded
                # directly), don't carry delta.content so they never touch
                # assembly, and aren't captured in the stored response.
                #
                # We use asyncio.wait (not wait_for) so the pending read
                # task is preserved across keepalive emits — no cancellation,
                # no torn httpx state.
                KEEPALIVE_INTERVAL = 30  # seconds
                read_iter = resp.aiter_bytes().__aiter__()

                async def _stream_chunks():
                    nonlocal read_task
                    while True:
                        if read_task is None:
                            read_task = asyncio.ensure_future(read_iter.__anext__())
                        done, _ = await asyncio.wait(
                            {read_task}, timeout=KEEPALIVE_INTERVAL
                        )
                        if read_task in done:
                            try:
                                value = read_task.result()
                            except StopAsyncIteration:
                                read_task = None
                                return
                            read_task = None
                            yield ("chunk", value)
                        else:
                            yield ("keepalive", None)

                async for kind, chunk in _stream_chunks():
                    if kind == "keepalive":
                        # Emit SSE comment — invisible to client, invisible
                        # to assembly, invisible to captures.
                        yield b":keepalive\n\n"
                        continue
                    # else kind == "chunk"
                    if not chunk:
                        continue
                    # Parse for capture — SSE frames are separated by \n\n
                    pending += chunk
                    while b"\n\n" in pending:
                        frame, pending = pending.split(b"\n\n", 1)
                        for line in frame.split(b"\n"):
                            line = line.strip()
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]" or not payload:
                                continue
                            try:
                                event = json.loads(payload.decode("utf-8", errors="replace"))
                            except json.JSONDecodeError:
                                continue
                            response_id = response_id or event.get("id")
                            response_created = response_created or event.get("created")
                            response_model = event.get("model", response_model)
                            for choice in event.get("choices") or []:
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    assembled_content_parts.append(content)
                                for tc_delta in (delta.get("tool_calls") or []):
                                    idx = tc_delta.get("index", 0)
                                    tc = assembled_tool_calls.setdefault(idx, {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                    if tc_delta.get("id"):
                                        tc["id"] = tc_delta["id"]
                                    if tc_delta.get("type"):
                                        tc["type"] = tc_delta["type"]
                                    fn = tc_delta.get("function") or {}
                                    if fn.get("name"):
                                        tc["function"]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tc["function"]["arguments"] += fn["arguments"]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]

                    # ── Forwarding decision (after assembly for this chunk) ──
                    # Decide strip vs pass-through as soon as we have enough
                    # signal. Three triggers, any of which lands the decision:
                    #  (a) we found a newline in the assembled content;
                    #  (b) finish_reason arrived (stream ending) — forces a
                    #      decision even without a newline (covers the case
                    #      where model emits "LAST_THOUGHT: ..." with no
                    #      trailing newline before tool_calls close the round);
                    #  (c) 300-char safety (content too long to be a one-line
                    #      thought; bail and flush verbatim).
                    if preamble_decided:
                        yield chunk
                    else:
                        held_chunks.append(chunk)
                        assembled_so_far = "".join(assembled_content_parts)
                        newline_idx = assembled_so_far.find("\n")
                        stripped_lead = assembled_so_far.lstrip()
                        starts_with_prefix = stripped_lead.upper().startswith("LAST_THOUGHT:")

                        decision = None  # "strip" | "flush"
                        rest_content = ""

                        if newline_idx >= 0:
                            first_line = assembled_so_far[:newline_idx]
                            rest_content = assembled_so_far[newline_idx + 1:]
                            decision = "strip" if re.match(r"\s*LAST_THOUGHT:", first_line, re.IGNORECASE) else "flush"
                        elif finish_reason is not None:
                            # Stream ending without a newline in content. If
                            # the content IS the thought (no post-newline
                            # content), strip it entirely — emit a synthetic
                            # empty-content chunk so the client sees a clean
                            # assistant turn with just the tool_calls.
                            decision = "strip" if starts_with_prefix else "flush"
                            rest_content = ""
                        elif len(assembled_so_far) > 300:
                            decision = "flush"

                        if decision == "strip":
                            synth = {
                                "id": response_id or f"sim-{uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": response_created or int(time.time()),
                                "model": response_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": rest_content},
                                    "finish_reason": None,
                                }],
                            }
                            yield ("data: " + json.dumps(synth, ensure_ascii=False) + "\n\n").encode("utf-8")
                            held_chunks = []
                            preamble_decided = True
                        elif decision == "flush":
                            for h in held_chunks:
                                yield h
                            held_chunks = []
                            preamble_decided = True

                # Stream ended while still holding chunks. Apply one final
                # strip-or-flush decision based on what we accumulated, so
                # the end-of-stream edge case (LAST_THOUGHT emitted without
                # trailing newline and no finish_reason chunk seen yet)
                # doesn't leak the prefix to the client.
                if held_chunks:
                    assembled_so_far = "".join(assembled_content_parts)
                    stripped_lead = assembled_so_far.lstrip()
                    if stripped_lead.upper().startswith("LAST_THOUGHT:"):
                        # Strip entirely — emit a synthetic chunk with just
                        # whatever content came AFTER the first LAST_THOUGHT
                        # line (usually empty on tool-calls-only rounds).
                        newline_idx = assembled_so_far.find("\n")
                        rest_content = assembled_so_far[newline_idx + 1:] if newline_idx >= 0 else ""
                        synth = {
                            "id": response_id or f"sim-{uuid.uuid4().hex[:12]}",
                            "object": "chat.completion.chunk",
                            "created": response_created or int(time.time()),
                            "model": response_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": rest_content},
                                "finish_reason": None,
                            }],
                        }
                        yield ("data: " + json.dumps(synth, ensure_ascii=False) + "\n\n").encode("utf-8")
                    else:
                        for h in held_chunks:
                            yield h
                    held_chunks = []
        except httpx.HTTPError as e:
            status_code = 0
            error_text = str(e)
            frame = ("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8")
            yield frame
            yield b"data: [DONE]\n\n"
        finally:
            # Clean up the keepalive read task if the generator is being
            # torn down mid-stream (client disconnect, upstream error).
            if read_task is not None and not read_task.done():
                read_task.cancel()
            await client.aclose()
            latency_ms = int((time.time() - start) * 1000)
            # Build synthetic non-stream completion for capture storage
            message: dict = {"role": "assistant"}
            if assembled_content_parts:
                message["content"] = "".join(assembled_content_parts)
            else:
                message["content"] = ""
            if assembled_tool_calls:
                message["tool_calls"] = [
                    assembled_tool_calls[i] for i in sorted(assembled_tool_calls)
                ]
            response_body = {
                "id": response_id or f"sim-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": response_created or int(time.time()),
                "model": response_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }],
            } if status_code == 200 else None
            # Extract LAST_THOUGHT: line from the assembled assistant content
            last_thought = extract_last_thought(message.get("content"))
            save_capture(
                capture_id=capture_id,
                model=model,
                upstream=current_upstream(),
                request_body=body,
                response_body=response_body,
                response_status=status_code,
                latency_ms=latency_ms,
                error=error_text[:500] if error_text else None,
                last_thought=last_thought,
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Captures API ────────────────────────────────────────────────────────────

@app.delete("/api/captures")
def clear_captures() -> dict:
    """Wipe all captures + debriefs. Settings table is preserved."""
    conn = db_connect()
    deleted_debriefs = conn.execute("DELETE FROM debriefs").rowcount
    deleted_captures = conn.execute("DELETE FROM captures").rowcount
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return {"deleted_captures": deleted_captures, "deleted_debriefs": deleted_debriefs}


@app.get("/api/captures")
def list_captures(limit: int = 100) -> list[dict]:
    conn = db_connect()
    rows = conn.execute(
        "SELECT id, created_at, model, response_status, latency_ms, error, "
        "request_body, last_thought FROM captures ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        req_body = json.loads(row["request_body"])
        preview = _first_user_text(req_body)
        out.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "model": row["model"],
            "response_status": row["response_status"],
            "latency_ms": row["latency_ms"],
            "error": row["error"],
            "preview": preview,
            "num_messages": len(req_body.get("messages") or []),
            "num_tools": len(req_body.get("tools") or []),
            "last_thought": row["last_thought"] or "",
        })
    conn.close()
    return out


@app.get("/api/captures/{capture_id}")
def get_capture(capture_id: str) -> dict:
    conn = db_connect()
    row = conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="capture not found")
    capture = dict(row)
    capture["request_body"] = json.loads(capture["request_body"])
    capture["response_body"] = json.loads(capture["response_body"]) if capture["response_body"] else None

    debrief_rows = conn.execute(
        "SELECT id, created_at, question, answer FROM debriefs "
        "WHERE capture_id = ? ORDER BY created_at ASC",
        (capture_id,),
    ).fetchall()
    capture["debriefs"] = [dict(d) for d in debrief_rows]
    conn.close()
    return capture


class DebriefBody(BaseModel):
    question: str


@app.post("/api/captures/{capture_id}/debrief")
async def post_debrief(capture_id: str, payload: DebriefBody) -> dict:
    if not current_debrief_key():
        raise HTTPException(
            status_code=400,
            detail="MIDSENTENCE_API_KEY is not set. The debrief call needs an API "
                   "key to call the upstream model. Set it in .env and restart.",
        )

    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    conn = db_connect()
    row = conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="capture not found")

    request_body = json.loads(row["request_body"])
    response_body = json.loads(row["response_body"]) if row["response_body"] else None

    prior = conn.execute(
        "SELECT question, answer FROM debriefs WHERE capture_id = ? AND answer IS NOT NULL "
        "ORDER BY created_at ASC",
        (capture_id,),
    ).fetchall()

    messages = build_debrief_messages(request_body, response_body, prior, q)

    # Build upstream call body. Same model, same system (inside messages),
    # same tools definitions (so the model can "see" what it had). We keep
    # tools in the payload so the model can literally quote their names /
    # descriptions when asked. The stub tool_results + the explicit analysis
    # prefix keep it in prose mode.
    upstream_body: dict[str, Any] = {
        "model": request_body.get("model"),
        "messages": messages,
        "stream": False,
    }
    if request_body.get("tools"):
        upstream_body["tools"] = request_body["tools"]
    # Preserve some common params so the "character" of the debrief matches
    # the original call. Temperature low-ish is fine for introspection.
    for k in ("temperature", "top_p", "max_tokens"):
        if k in request_body:
            upstream_body[k] = request_body[k]
    # But a debrief should usually allow decent-length answers
    upstream_body.setdefault("max_tokens", 2048)

    headers = {
        "Authorization": f"Bearer {current_debrief_key()}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                resolve_chat_endpoint(),
                json=upstream_body,
                headers=headers,
            )
    except httpx.HTTPError as e:
        conn.close()
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    if resp.status_code != 200:
        conn.close()
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

    data = resp.json()
    answer = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    # Some models return content as a list of content blocks
    if isinstance(answer, list):
        answer = "\n".join(p.get("text", "") for p in answer if isinstance(p, dict))

    debrief_id = f"deb-{uuid.uuid4().hex[:12]}"
    conn.execute(
        "INSERT INTO debriefs (id, capture_id, created_at, question, answer) "
        "VALUES (?, ?, ?, ?, ?)",
        (debrief_id, capture_id, time.time(), q, answer),
    )
    conn.commit()
    conn.close()

    return {"id": debrief_id, "question": q, "answer": answer}


# ── Static UI ───────────────────────────────────────────────────────────────

def _first_user_text(request_body: dict) -> str:
    for msg in request_body.get("messages") or []:
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c[:120]
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))[:120]
    return ""


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn
    # Plain ASCII prints — Windows cp1252 consoles choke on unicode arrows.
    print(f"midsentence -> listening on http://127.0.0.1:{PORT}")
    print(f"  UI:       http://127.0.0.1:{PORT}/")
    print(f"  Proxy:    http://127.0.0.1:{PORT}/v1/chat/completions")
    print(f"  Upstream: {current_upstream()}")
    print(f"  Debrief key set: {bool(current_debrief_key())}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
