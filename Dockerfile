FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY scripts/ scripts/

RUN mkdir -p data logs models

CMD ["python", "-m", "src.main"]
