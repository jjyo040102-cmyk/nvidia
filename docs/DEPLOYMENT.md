# Deployment

## Local judge demo

```bash
python -m venv .venv
. .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
vigil serve
```

Open `http://127.0.0.1:8000`.

## Public demo host

The application is a normal ASGI/FastAPI service. A host must preserve the repository `data/`
directory because the bundled judge footage and rule corpus live there.

Direct start command:

```bash
vigil serve --host 0.0.0.0 --port 8000
```

A minimal `Dockerfile` is included for hosts that accept container deployments:

```bash
docker build -t vigil .
docker run --rm -p 8000:8000 vigil
```

For LIVE, inject `VIGIL_NEBIUS_API_KEY` as a platform secret/environment variable rather than
baking it into the image.

Required for credential-free DEMO:

- Python 3.11+
- repository `data/` directory
- no secret

Additional environment for LIVE:

```text
VIGIL_NEBIUS_API_KEY=<secret>
VIGIL_NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
VIGIL_REASONING_MODEL=
VIGIL_VISION_MODEL=
```

Leaving the model IDs empty lets `vigil probe` / the client choose from the account's current
catalog. The default reasoning selector is Nemotron-only.

## Hosting checklist

1. Keep `.env` out of the repository.
2. Configure `VIGIL_NEBIUS_API_KEY` in the host's secret/environment UI.
3. Start from the repository root so `data/clips` and `data/policy` resolve correctly.
4. Verify `GET /healthz` returns `{"status":"ok",...}`.
5. Verify `GET /api/status` reports the expected LIVE availability.
6. Run one DEMO investigation in the browser.
7. Run one LIVE investigation and confirm the provenance panel names the actual models.
8. Put the resulting public HTTPS URL in Devpost.

## Nebius deployment option

The hackathon rules accept a runtime Token Factory call as the Nebius execution requirement; the
rest of the web application may be hosted separately. If deploying more of the stack on Nebius,
use an AI Cloud / serverless environment that can run the same ASGI start command and mount or
ship the `data/` directory with the application.

## Production hardening beyond the hackathon

The current web demo intentionally has no authentication because it exposes only bundled sample
clips. Before accepting arbitrary customer CCTV, add authentication, upload isolation, retention
controls, audit logging, access-policy enforcement, and a deployment-specific privacy review.
