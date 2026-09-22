# Vigil

**See the accident before it happens.**

Vigil is an evidence-first near-miss investigator for industrial CCTV. Instead of stopping at
"something dangerous is visible," it runs an autonomous investigation: scan the footage, form a
hazard hypothesis, go back to the critical seconds with a narrow question, retrieve the relevant
site/OSHA rule, score the risk, recommend a human action, and expose the complete trace and model
provenance.

Built for the **Nebius × NVIDIA Global AI Hackathon — Best Apps and Agents** track.

## Why this exists

Conventional CCTV analytics is usually retrospective: detect an event, classify it, review it
later. Vigil is designed around the few seconds *before* a contact, fall, struck-by, slip, or other
near miss, when a warning can still be useful. The product is decision support for a trained human,
not an automated safety authority.

## 60-second judge path — no credentials required

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
vigil serve
```

Open **http://127.0.0.1:8000**. Choose a bundled camera event and click **Investigate the next few
seconds**.

The dashboard runs in **DEMO** mode without an API key. DEMO is a deterministic end-to-end proof of
the agent workflow; it is deliberately **not** presented as measured model accuracy.

CLI-only demo:

```bash
vigil demo
```

This writes JSON + Markdown incident reports to `artifacts/reports/`.

## Live Nebius + NVIDIA path

```bash
cp .env.example .env
# Set VIGIL_NEBIUS_API_KEY in .env
vigil probe
vigil serve
```

Then switch the dashboard to **LIVE**.

LIVE forces both hosted slots to Nebius Token Factory. Automatic reasoning selection is
**Nemotron-only**, so the default hackathon path cannot silently fall back to a non-NVIDIA
reasoner. Vigil also records the exact vision model, reasoning model, token counts, and wall-clock
latency in every report.

You can run the same path from the CLI:

```bash
vigil run \
  --video blind_corner_struck_by \
  --perception nebius \
  --reasoning nebius
```

## Agent loop

```text
CCTV recording
     │
     ▼
Overlapping temporal scan
     │
     ▼
Perception model ───────────────┐
     │                         │
     ▼                         │
Nemotron investigator          │
     │                         │
     ├── survey another window │
     ├── re-look with a narrow question ───────┘
     ├── search policy/rules
     ├── flag a supported hazard
     ├── clear a rejected hypothesis
     └── finish
     │
     ▼
Risk ledger + citations + mitigations
     │
     ▼
Auditable incident report + browser dashboard
```

The action grammar is closed and typed: `survey`, `relook`, `policy_search`, `flag`, `clear`, and
`finish`. Hallucinated tool arguments fail at the boundary instead of being silently ignored.
Rule citations are resolved by Vigil's policy layer rather than copied blindly from model text.

## Where Nebius and NVIDIA are used

| Layer | LIVE implementation | Why it matters |
| --- | --- | --- |
| Reasoning | NVIDIA Nemotron on Nebius Token Factory | Chooses the next investigative action, not just a final answer |
| Vision/perception | NVIDIA-capable VLM when exposed by the account; otherwise an explicitly configured hosted VLM | Reads temporal evidence from the recording |
| Optional edge path | `nvidia/Cosmos-Reason2-2B` locally | Experimental physical-reasoning path for a CUDA-equipped environment |
| Runtime/accounting | Nebius Token Factory client | Model discovery, retries, caching, token/cost guard, provenance |

The repository never hides which model actually ran. `vigil probe` lists what the account exposes
and which model would be selected for each slot.

## Evaluation integrity

Vigil ships eight authored clips: six hazard scenarios and two controls. Each has sidecar ground
truth so the live model path can be scored on more than "the demo looked convincing."

```bash
vigil eval --perception nebius --reasoning nebius
```

The scorecard includes:

- hazard recall
- false positives on controls
- lead-time capture and over-claim checks
- mechanism accuracy
- severity within ±1
- citation resolution
- wall-clock time, token usage, and estimated spend

A scripted run intentionally exits non-zero with **NOT A MEASUREMENT**. This prevents deterministic
fixtures from being pasted into a README as if they were live model performance. See
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## Judge-facing dashboard

`vigil serve` launches a FastAPI application with:

- bundled CCTV playback
- explicit DEMO/LIVE mode badge
- one-click investigation
- risk score and warning lead time
- immediate human action
- evidence timeline and full investigation trace
- resolved rule citations
- exact model provenance, tokens, and latency
- an evaluation-integrity disclosure in the UI itself

The API is also documented at `http://127.0.0.1:8000/docs`.

## Project structure

```text
src/vigil/
  agent/        autonomous loop, actions, reasoner, tools
  api/          FastAPI judge surface + dashboard
  eval/         ground truth, scoring, benchmark renderer
  models/       scene, risk, report, provenance and trace models
  perception/   mock, Nebius VLM, experimental local Cosmos backends
  policy/       grounded rule retrieval
  report/       Markdown handoff report
  video/        source windows, sampling and synthetic scenarios

data/
  clips/        bundled demo/evaluation footage + ground truth
  policy/       site/OSHA-style rule corpus

docs/           architecture, evaluation, deployment, safety and submission assets
tests/          offline test suite; no test is allowed to spend real API credit
```

## Safety posture

Vigil is **decision support only**. It does not autonomously stop machinery, discipline workers,
or replace emergency procedures. A report carries its evidence trace, confidence, citations, and
model provenance so a competent human can review the basis for the recommendation. See
[`docs/SAFETY.md`](docs/SAFETY.md).

## Cost and failure controls

- hard token budget per investigation
- hard estimated-USD budget per run
- bounded planning turns
- retry limits and timeouts
- content-addressed response cache
- explicit incomplete-run status
- no silent LIVE → mock fallback
- no hidden model substitution for the default Nemotron reasoning slot

## Deployment

A judge can run locally with `vigil serve`. For a public demo, deploy the repository with its
`data/` directory available and run:

```bash
vigil serve --host 0.0.0.0 --port 8000
```

Detailed production/demo-hosting notes are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Submission assets

- [`docs/DEVPOST.md`](docs/DEVPOST.md) — ready-to-paste project description
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — <3 minute demo-video script
- [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md) — final pre-submit checklist
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — technical walkthrough
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — benchmark protocol and claim policy

## License

Apache-2.0. See [`LICENSE`](LICENSE).
