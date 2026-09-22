FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["vigil", "serve", "--host", "0.0.0.0", "--port", "8000"]
