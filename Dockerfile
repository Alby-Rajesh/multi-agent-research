FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY configs ./configs
COPY corpus ./corpus
COPY data ./data

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"
CMD ["research", "serve", "--port", "8080"]
