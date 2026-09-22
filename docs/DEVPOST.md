# Devpost submission copy

## Project name

**Vigil — See the accident before it happens**

## Track

**Best Apps and Agents**

## One-line description

Vigil is an autonomous near-miss investigator that uses NVIDIA Nemotron on Nebius to examine the
seconds before an industrial accident, re-check its own hypothesis, ground the finding in site
rules, and produce an auditable action report for a human.

## Inspiration

Most CCTV analytics becomes useful after the bad moment: an event is detected, someone searches
for the clip, and the incident is reviewed. We wanted to move the useful part of that workflow a
few seconds earlier. A near miss is not just an object-detection problem; it requires temporal
reasoning about converging paths, uncertainty, a reason to go back and inspect one detail, and a
clear threshold for when a human should actually be interrupted.

## What it does

Vigil takes a fixed-camera recording and runs an evidence-first investigation. It first scans
covering windows across the footage. The reasoning agent then chooses one typed action at a time:
look at another interval, re-inspect a critical interval with a narrow question, search the policy
corpus, file a supported hazard, explicitly clear a rejected hypothesis, or finish. The resulting
report includes the predicted event, risk score, lead time, mitigations, resolved rule citations,
a chronological evidence trace, exact model provenance, token usage and latency.

The browser demo makes DEMO and LIVE visibly different. DEMO needs no key and proves the entire
workflow with deterministic fixtures. LIVE uses Nebius Token Factory for hosted perception and
reasoning and never silently falls back to scripted models.

## How we built it

The backend is Python/FastAPI with typed Pydantic models. The agent loop is provider-independent:
the reasoner emits a closed action grammar and the toolkit executes those actions against footage
or a local rule corpus. Hosted inference goes through a Nebius Token Factory client that discovers
the account's current model catalog, retries transient failures, caches identical requests, tracks
tokens and estimated spend, and rejects calls that would cross a configured budget.

For the default hackathon LIVE path, automatic reasoning selection is restricted to NVIDIA
Nemotron-family models. The exact model that actually ran is written into every incident report.
The vision slot prefers NVIDIA-capable models when exposed by the account. We also implemented an
experimental local NVIDIA Cosmos Reason path for CUDA-equipped environments.

The evaluation harness ships with six hazard scenarios and two controls plus ground-truth timing.
It scores recall, control false positives, lead-time capture, mechanism accuracy, severity,
citation resolution and integrity failures. Scripted runs intentionally exit non-zero with
`NOT A MEASUREMENT`, so a deterministic demo cannot be mistaken for a live benchmark.

## How Nebius and NVIDIA are essential

Nemotron is the investigator, not a cosmetic summarizer. It decides whether the current evidence
is sufficient, which temporal window to revisit, what specific question to ask of perception, when
to search policy, and when a finding crosses the reporting bar. Nebius Token Factory makes that
loop practical by providing hosted open-model inference, model discovery, and a single runtime
surface for the project. The application records every selected model and every investigation step
so the role of the NVIDIA/Nebius stack is visible to a judge.

## Challenges

The hardest problem was avoiding a demo that looked agentic while actually being a fixed pipeline.
We kept the control loop small and moved decisions into a typed action grammar, then made each
action produce evidence that the next turn can challenge. A second challenge was honest
measurement: because our credential-free scenarios are deterministic, allowing them to generate an
"accuracy" percentage would be misleading. The evaluator therefore refuses to call mock results a
measurement.

A third challenge was reliability under hackathon constraints. Token Factory model IDs and account
availability can change, so the client probes `/models` rather than betting the demo on one stale
hard-coded ID. Cost ceilings, caching and bounded iterations keep repeated demo runs from consuming
an open-ended amount of credit.

## Accomplishments

- Multi-step autonomous investigation instead of a one-call classifier.
- Falsifiable `relook` step that sends the agent back to the exact seconds it is uncertain about.
- Grounded rule citations resolved by the application rather than trusted from free-form model
  text.
- Full evidence trace and per-report model provenance.
- Explicit DEMO/LIVE separation with no hidden fallback.
- A live-evaluation harness that refuses unsupported performance claims.
- Browser-first judge experience plus the same workflow from the CLI.

## What we learned

Anticipatory safety is fundamentally temporal. If the system cannot say when evidence appeared,
what it checked next, and how much warning the footage actually allowed, "AI safety monitoring"
quickly collapses into a vague alert generator. We also learned that transparency improves the
product experience: the trace, model identity and incomplete-run state are useful debugging tools
and simultaneously make the result easier for a human to trust appropriately.

## What's next

The next step is a larger externally sourced near-miss dataset with camera/domain splits and a
blind evaluation protocol. For a real pilot, we would add authenticated customer uploads, privacy
and retention controls, site-specific policy ingestion, alert calibration by operating zone, and a
human-feedback loop that measures nuisance-alert rate as carefully as hazard recall.

## Significant update during the submission period

The submission version packages the project as a complete judge-facing product: a FastAPI browser
dashboard, explicit DEMO/LIVE execution modes, strict Nemotron default selection for hackathon
eligibility, a reproducible live-evaluation path, complete architecture/evaluation/safety/deployment
documentation, and an expanded offline regression suite around the submission surface. If Devpost
requires a pre-existing-project disclosure, pair this paragraph with the actual project start date
from your repository history.

## External links to add at submission time

- **Working demo URL:** add the deployed HTTPS URL after hosting
- **Public repository:** add the public GitHub/GitLab/Bitbucket URL after publishing
- **YouTube demo:** add the public <3 minute video URL after upload
