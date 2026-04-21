# midsentence

**Your LLM call isn't atomic — it's a conversation paused mid-sentence.**

A drop-in proxy for OpenAI-compatible chat/completions APIs. Captures every call. When the model does something weird, you don't grep prompts for hours — you just ask the model.

---

## why

I spent a month editing system prompts for a WordPress AI copilot before it hit me: **why am I treating this like a normal API call? The thing literally can talk.**

Every time the agent picked the wrong tool, hit the same guard three rounds in a row, or burned the round limit on what should have been a 2-step task — I'd spend hours guessing which sentence in a 38KB system prompt was misleading it. Grep-and-guess. Rinse and repeat.

Now when the model does something weird, I replay the exact request against a fresh call of the same model and just ask:

> *"Which lines in the prompt steered you to pick tool A over tool B?"*

The model answers. Specifically. With quotes. Often the fix is **subtracting** a misleading sentence, not piling on new rules.

`midsentence` is the smallest tool that makes this loop reproducible.

---

## how it works

```
your app  ──POST /v1/chat/completions──▶  midsentence  ──▶  OpenAI / Claude / OpenRouter / ...
                                              │
                                              ▼
                                          SQLite
                                              │
                                              ▼
                                     you open the UI,
                                     click a capture,
                                     type "why did you pick X?"
                                     the same model answers
```

Your app keeps its own API key. `midsentence` forwards `Authorization` through untouched. The only thing stored is the request/response envelope — no key rewriting, no token harvesting.

When you click **ask**, `midsentence` reconstructs the full conversation as the model originally saw it (same system prompt, same tools, same history), appends the model's own reply, appends your debrief question with an explicit "analysis only" prefix, and sends it back to the same model. The model experiences it as a continued conversation — now in analysis mode.

---

## install & run

```bash
git clone https://github.com/YOU/midsentence
cd midsentence
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — set MIDSENTENCE_API_KEY (needed for the debrief button)

python server.py
```

Open http://localhost:8765 — that's the UI.

---

## point your app at it

Anywhere you use an OpenAI-compatible client, replace the base URL:

```python
# OpenAI Python SDK
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8765/v1",
    api_key="sk-your-real-openai-key",
)

# Anthropic via OpenAI-compat endpoint
client = OpenAI(
    base_url="http://localhost:8765/v1",
    api_key="sk-ant-...",
)

# OpenRouter
client = OpenAI(
    base_url="http://localhost:8765/v1",
    api_key="sk-or-...",
)
```

Fire a call. Refresh the UI. Click the capture. Type a question.

---

## the loop

1. Your app makes an LLM call. `midsentence` captures request + response, forwards verbatim.
2. Something weird happens — unexpected tool pick, odd plan, prompt-rejection loop.
3. You open the UI, find the offending call.
4. Ask: *"which lines in the system prompt steered you to tool A over tool B?"*
5. The model answers with specifics, quoting actual prompt lines.
6. You edit the exact sentence it cited. Often by removing, not adding.
7. Re-run your chain. Problem fixed. Rinse for the next one.

Fix prompts by subtraction, not addition. The model tells you *where* to edit — you stop guessing.

## two signals, different questions — use the right one first

midsentence surfaces two distinct signal types from the same captured data. They answer different categories of bug. Using the wrong one wastes a debrief.

| Signal | What it catches | How to read it |
|---|---|---|
| **Round-to-round diff** | Harness-side bugs: synthetic user turns injected between rounds, replayed tool_results, state overrides, continuation messages your user never typed. | Compare `messages[]` between consecutive captures on the same chain. Anything in round N's prompt that wasn't in round N-1's reply is something YOUR HARNESS added. |
| **Debrief** | Model-side choices: why it picked tool A vs B, which prompt lines steered a behavior, whether it noticed a rule that should have applied. | Click a capture → type a follow-up in the debrief panel → the model replays against same inputs and cites the prompt lines that shaped its choice. |

**Correct debug sequence — always diff first:**

1. **Diff the rounds.** Look at the `messages[]` array of the round where the bug surfaced. Is its last user message something the real user typed? Or is it a synthetic retry / replay / continuation that the harness added? If synthetic, the bug is harness-side — go read the code that injected it. The model did nothing wrong.
2. **Only then debrief.** If the rounds look clean (every user message came from the real user, every tool_result maps to a real tool that ran) and the model still made a suspect choice — THEN ask the model why. It'll cite prompt lines, and those are your edit targets.

Skipping step 1 wastes time. I learned this debugging a PressArk chain where a simple "u here?" greeting triggered two extra rounds. The debrief told me the model's round-1 reasoning was fine ("no writes, no plan step needed"). It couldn't have told me what I actually wanted to know, because the bug was between rounds — a `plan_response_requires_user_input` regex in the harness only checked the first line of the model's reply, missed the trailing `?`, and injected a synthetic "read first" user turn for round 2. Model did nothing wrong; harness misread the reply. The round-diff showed it in ten seconds. The debrief would have looked forever.

**Rule of thumb:** if your suspicion is about a model *choice*, debrief. If your suspicion is about a transition *between* rounds, diff.

---

## limitations (this is v0)

- **Non-streaming only.** `"stream": true` returns a 400. Non-streaming captures work cleanly; streaming support is on the roadmap.
- **Chat completions endpoint only.** `/v1/chat/completions` is captured. Other endpoints (embeddings, images, responses API) aren't proxied at all yet.
- **Single-user, no auth.** Bind to `localhost` or a trusted network. Don't expose to the internet.
- **Debrief requires an API key.** `MIDSENTENCE_API_KEY` is used for debrief calls. Your app's key is forwarded for the original proxy pass — those are separate.
- **Debrief strips nothing smart yet.** Tools are preserved so the model can "see" what it had; stub tool_results mark debrief mode. Works well with Sonnet-class and GPT-4-class models; smaller models may need sharper prompting.
- **No request chaining, editing, or replay-with-modifications.** Coming.
- **Bodies stored in plain SQLite.** Don't point this at a prod key in a shared env. It's a debug tool.

---

## faq

**Does the model hallucinate its own reasoning?**
At the margins, yes. Larger models (Sonnet 4.5+, GPT-4 class) are reliable introspectors; smaller/older ones rationalize more. Treat the answer as a **strong signal to go check**, not gospel. The answer usually points at a specific sentence — reading that sentence yourself is always the last step.

**Is this observability?**
No. Observability logs everything. `midsentence` is forensic debug — you replay one specific call and interrogate the model about its choice. Closer to `gdb` than `datadog`.

**Is this evaluation?**
No. Evals compare runs across prompts or models. `midsentence` is one-call-at-a-time root-cause.

**Why "midsentence"?**
Because that's what an LLM call actually is — a conversation paused mid-sentence. You can resume it any time, from any point.

**Is this safe for production?**
No. Run it in a trusted dev environment. Request bodies are stored in a local SQLite file. Treat captures as sensitive.

---

## roadmap (if there's signal for it)

- Streaming proxy (SSE pass-through with capture)
- Multi-provider native (anthropic `/v1/messages`, google, etc.)
- Request editing & replay-with-modifications
- Chained follow-ups that fork into experiment branches
- Team mode (shared captures, annotations)
- Hosted SaaS

If you've used it and have opinions, open an issue.

---

## license

MIT.

---

*Built by someone who got tired of grep-and-guess.*
