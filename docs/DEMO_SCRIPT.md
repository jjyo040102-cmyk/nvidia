# Demo video script — target 2:40 to 2:55

The hackathon video must stay under three minutes, so this script is designed to show the product
working first and explain architecture only after the value is obvious.

## 0:00–0:12 — Cold open

**Screen:** Play the blind-corner clip in the Vigil dashboard. Do not begin with slides.

**Voice:**
"Most CCTV systems tell you what happened after an accident. Vigil investigates the few seconds
before it happens."

## 0:12–0:30 — The problem

**Screen:** Pause just before the conflict. Point at the worker/forklift routes.

**Voice:**
"A near miss is not just object detection. The system has to reason about motion, uncertainty and
when the evidence is strong enough to interrupt a human without becoming another alarm people
ignore."

## 0:30–1:18 — Run the product

**Screen:** Dashboard in LIVE mode if the Nebius account is available for recording; otherwise
record a separate verified LIVE run first and use that result. Click **Investigate**. Show the trace
as it fills or the completed trace if latency makes live capture awkward.

**Voice:**
"Vigil scans the recording, forms a hypothesis, and chooses its next action. Here it goes back to a
specific time window with a narrow question instead of trusting its first impression. It can search
the site rulebook, explicitly clear a rejected line of inquiry, or file a supported hazard."

## 1:18–1:42 — Human handoff

**Screen:** Risk score, warning lead time, immediate action and citation panel.

**Voice:**
"The output is not a chat answer. It is an auditable safety handoff: predicted event, risk score,
warning lead time, immediate mitigation, grounded rule citations and the full investigation trace."

## 1:42–2:08 — Nebius + NVIDIA

**Screen:** Provenance panel, then briefly show `vigil probe` in a terminal.

**Voice:**
"The investigator runs on NVIDIA Nemotron through Nebius Token Factory. Vigil discovers the models
our account can actually serve, keeps the default reasoning path on Nemotron, and records the exact
vision and reasoning models, token usage and latency in every report. We also built an optional
local Cosmos Reason path for edge experiments."

## 2:08–2:32 — Evaluation honesty

**Screen:** `docs/EVALUATION.md`, then terminal with the live `vigil eval` result if one has been
run. Never show the mock score as model performance.

**Voice:**
"We ship six hazard cases and two controls with ground-truth timing. The evaluator measures recall,
false positives, mechanism, severity and how much warning was actually captured. A scripted demo is
hard-coded to say NOT A MEASUREMENT, so we cannot accidentally turn our own fixtures into an
accuracy claim."

## 2:32–2:50 — Close

**Screen:** Return to dashboard headline and trace.

**Voice:**
"Vigil turns CCTV from a record of what went wrong into an investigator for what is about to go
wrong — while keeping a human in control of the decision."

## Recording checklist

- Keep total runtime below 3:00.
- Show the actual application running, not only slides.
- Say **Nebius Token Factory** and **NVIDIA Nemotron** aloud.
- Show LIVE model provenance if claiming a live run.
- Do not show API keys, `.env`, browser password managers, or private console data.
- Use only music/assets you have permission to publish; silence is safer than unlicensed music.
