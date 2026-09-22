# Safety and responsible-use notes

## Intended use

Vigil is decision-support software for reviewing industrial CCTV and surfacing possible near-term
physical hazards to a competent human operator or safety professional.

## Explicit non-goals

Vigil is not designed to:

- autonomously stop machinery or vehicles
- replace site emergency procedures
- determine employee fault or discipline
- establish regulatory compliance by itself
- make an employment, insurance, legal, or medical decision
- guarantee that an area is safe because no finding was reported

## Human confirmation

Every report carries a fixed disclaimer that the result is decision support only. Recommended
mitigations are written as actions for a human role, and the report exposes evidence, confidence,
rules and provenance so the recommendation can be checked.

## Failure modes addressed in code

### Silent incomplete analysis

An investigation that reaches its turn or token ceiling is marked incomplete. The CLI exits
non-zero rather than presenting the partial report as a clean success.

### Hallucinated policy citations

The reasoner cannot directly write the final citation object. It asks the policy tool for rules;
rule IDs in a final finding are resolved against the local corpus before they reach the report.

### Hidden scripted fallback

DEMO and LIVE are separate modes. LIVE without a Nebius key fails explicitly. It does not quietly
run a scripted model while the UI still says LIVE.

### Unbounded spending

Each investigation has bounded turns, a token ceiling and an estimated USD ceiling checked before
hosted calls.

### Overstated benchmark claims

Scripted backends are rejected as measurements by `vigil eval`, and warning-time overclaims are
listed as integrity failures.

## Remaining limitations

- The bundled footage is synthetic and is intended for reproducible demonstration/evaluation
  plumbing, not to represent the full visual diversity of industrial environments.
- A live VLM can miss hazards, infer incorrect motion, or be overconfident.
- Camera placement, frame rate, occlusion, lighting and compression can materially affect results.
- The local policy corpus is not a substitute for a site's current official procedures or legal
  advice.
- Real deployment requires privacy, retention, worker-notice, access-control and jurisdictional
  review appropriate to the installation.
