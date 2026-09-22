# Evaluation and claim policy

## Principle

A deterministic controller scoring deterministic scenarios authored for that controller is a
pipeline test, not model evidence. Vigil encodes that distinction in code rather than relying on
someone remembering it during a deadline.

## Bundled set

`data/clips/` contains eight authored recordings:

- six hazard scenarios
- two hazard-free controls

Each clip has a ground-truth sidecar with the event mechanism and timing required by the scoring
harness.

## Live measurement command

```bash
vigil eval --perception nebius --reasoning nebius
```

A valid measured run writes:

```text
artifacts/eval/latest.json
artifacts/eval/latest.md
```

Do not copy a score into a submission until both files were produced by a live run and the command
returned exit code 0.

## Metrics

The scorecard tracks:

- `detected / truth_hazards`
- recall
- misses
- false positives on controls
- uncorroborated findings
- mean available lead time
- mean claimed lead time
- maximum over-claim
- captured lead-time ratio
- mean absolute timing error
- mechanism accuracy
- severity within ±1
- citations resolved / filed
- stop-work escalations
- per-clip matches and unmatched findings

## Integrity gates

A run is not considered a valid measurement when either perception or reasoning is scripted.
`vigil eval` still prints the table because it is useful for smoke testing, but exits non-zero and
prints `NOT A MEASUREMENT`.

The harness also rejects:

- claims of more warning time than the footage supports
- missing ground truth for requested evaluation clips
- stopped/incomplete investigations that cannot be fairly scored

## Reproducible live-report procedure

1. Copy `.env.example` to `.env` and set `VIGIL_NEBIUS_API_KEY`.
2. Run `vigil probe` and record the selected Nemotron reasoning model and vision model.
3. Keep `VIGIL_REASONING_MODEL` empty unless you intentionally pin a known NVIDIA model.
4. Run the full evaluation command above.
5. Preserve `artifacts/eval/latest.json` with the submitted commit/tag.
6. Report the exact model IDs, timestamp, token usage, latency and spend alongside any metric.

## What this repository currently claims

The credential-free demo proves the complete workflow: temporal scan, investigative re-look,
policy retrieval, risk filing, citations, mitigations, trace and provenance. It does **not** claim a
live model accuracy percentage. A live score belongs in the submission only after the command
above has been run with an actual Nebius account.
