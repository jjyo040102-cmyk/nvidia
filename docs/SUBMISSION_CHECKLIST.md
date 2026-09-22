# Final submission checklist

Official rules should be re-opened on Devpost immediately before submission. This checklist is
based on the current hackathon requirements and the repository's actual implementation.

## Project eligibility

- [ ] Select **Best Apps and Agents**.
- [ ] Confirm the submitted build makes a runtime call to Nebius Token Factory or runs on Nebius AI
  Cloud.
- [ ] Confirm at least one NVIDIA open-source model actually runs in the submitted LIVE path.
- [ ] Run `vigil probe`; verify the reasoning pick is a Nemotron model.
- [ ] Do not pin a non-NVIDIA reasoning model for the judged LIVE demo.

## Repository

- [ ] Public GitHub/GitLab/Bitbucket repository.
- [ ] Apache-2.0 `LICENSE` visible at repository root.
- [ ] README renders correctly from the public repository.
- [ ] Fresh-clone quickstart tested.
- [ ] No `.env`, API key, credentials file, cache, coverage database, local virtual environment or
  unrelated project in the repository.
- [ ] `pytest`, Ruff and mypy checks pass on the submitted commit.

## Working demo

- [ ] Public HTTPS demo URL opens without private-network access.
- [ ] DEMO works with no credential visible to the user.
- [ ] LIVE mode is tested against the deployment's secret Nebius key.
- [ ] Provenance panel shows the actual model names.
- [ ] `/healthz` returns 200.
- [ ] `/docs` opens the API schema.

## Live evidence

- [ ] Save a successful `vigil probe` output with secrets redacted.
- [ ] Run `vigil eval --perception nebius --reasoning nebius` if you plan to quote performance.
- [ ] Quote only values from that live `artifacts/eval/latest.json`.
- [ ] Never describe `vigil demo` results as model accuracy.

## Demo video

- [ ] Public YouTube URL.
- [ ] Under 3:00.
- [ ] Shows the application functioning.
- [ ] Audio explicitly explains use of Nebius Token Factory and NVIDIA Nemotron/open models.
- [ ] No API key or secret appears on screen.
- [ ] No unlicensed third-party music/assets.

## Devpost form

- [ ] Project description pasted/reviewed from `docs/DEVPOST.md`.
- [ ] Working demo URL added.
- [ ] Public repository URL added.
- [ ] Public YouTube demo URL added.
- [ ] Nebius/NVIDIA feedback section completed.
- [ ] If applicable, disclose significant work completed during the submission period for any
  pre-existing project.
- [ ] If you attended an eligible in-person Builders & Brews event, select the correct city.

## Final five-minute smoke test

```bash
python -m pip install -e ".[dev]"
vigil demo
vigil serve --host 127.0.0.1 --port 8000
```

Then open the dashboard, run one hazard clip and one control clip, and confirm the trace,
citations/provenance, and clear/no-finding state all render correctly.
