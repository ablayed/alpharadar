FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY models/ models/
COPY scripts/ scripts/

RUN test -f /app/models/finbert-onnx/model.onnx
RUN mkdir -p data logs

CMD ["python", "-m", "src.main"]
