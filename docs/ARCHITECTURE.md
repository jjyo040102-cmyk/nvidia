# Vigil architecture

## Thesis

Vigil is not a one-shot classifier. The central architectural decision is to make the language
model an **investigator that chooses what evidence to request next**. Perception and reasoning are
separate slots so the same agent loop can use deterministic fixtures, Nebius-hosted models, or a
local experimental physical-reasoning backend without changing the control logic.

## End-to-end flow

1. `VideoSource` turns one recording into overlapping temporal windows.
2. A perception backend surveys each requested window and returns typed scene evidence.
3. The reasoner receives the accumulated transcript and chooses exactly one typed action.
4. `ToolKit` executes the action against footage or the policy corpus.
5. The result is appended to the transcript and the reasoner chooses again.
6. Supported findings become `RiskAssessment` objects; rejected lines of inquiry are kept as
   explicit clears.
7. The final `IncidentReport` carries findings, timeline, citations, mitigations, model
   provenance, token counts, wall-clock time, and an `InvestigationTrace`.

## The action grammar

The agent can choose only:

- `survey(t0_s, t1_s)` — inspect another interval broadly.
- `relook(t0_s, t1_s, question)` — ask a narrow falsifiable question about a narrow interval.
- `policy_search(query, hazard_type)` — retrieve relevant rules.
- `flag(...)` — file a supported hazard with severity, likelihood, mechanism, lead time and
  mitigation.
- `clear(subject, reason)` — record something that was actively checked and rejected.
- `finish(summary)` — close the investigation.

All action models use Pydantic with `extra="forbid"`. This makes a malformed or hallucinated tool
call visible rather than silently dropping unsupported fields.

## Evidence grounding

### Footage

A finding's `clip_window` is resolved from footage the toolkit actually read. The model cannot
invent a window and have that invention silently become report metadata.

### Rules

The model asks for a policy search and names rule IDs when it flags a hazard, but the final report
contains only `RuleCitation` objects resolved by the local rulebook. This prevents an invented
citation from becoming a filed rule reference.

### Lead time

The project uses overlapping windows because anticipation depends on temporal boundaries. A
warning claim is scored against the authored event time in the evaluation harness. Over-claiming
lead time is treated as an integrity defect, not a better score.

## Backends

### `mock`

Deterministic scripted scenes and a scripted controller. Purpose: prove the product plumbing,
reporting, UI and failure semantics with zero credentials. It is never considered a measurement.

### `nebius`

Nebius Token Factory provides model discovery and chat-completions-compatible inference. The
client adds retries, a content-addressed response cache, token accounting, estimated spend, and a
pre-call budget guard. Automatic reasoning selection is restricted to Nemotron-family matches for
the hackathon path.

### `cosmos`

Experimental local `nvidia/Cosmos-Reason2-2B` path for a CUDA environment. It is intentionally
optional because the model is gated and GPU-memory-heavy. Failure messages distinguish missing
optional packages, missing CUDA, insufficient free VRAM, and model-license/load errors.

## Web/API layer

The browser surface is thin on purpose. `POST /api/investigate/{video_id}` calls the same
`investigate()` function as the CLI.

- `mode=demo` forces mock/mock.
- `mode=live` forces nebius/nebius and rejects a missing key.
- There is no silent fallback after LIVE is selected.

This prevents a polished dashboard from accidentally becoming a second implementation with
results that differ from the actual engine.

## Operational boundaries

- `max_iterations` bounds agent turns.
- `token_budget` bounds investigation token consumption.
- `usd_budget` rejects a call before its projected cost crosses the configured ceiling.
- stopped investigations are marked incomplete rather than presented as success.
- every test runs with credentials disabled or an in-process fake transport; the test suite is
  not allowed to spend real credit.
