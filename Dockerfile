FROM python:3.12-slim AS model-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch
RUN python -c "from pathlib import Path; from optimum.onnxruntime import ORTModelForSequenceClassification; from transformers import AutoTokenizer; model_id = 'ProsusAI/finbert'; save_path = Path('/build/models/finbert-onnx'); save_path.parent.mkdir(parents=True, exist_ok=True); model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True); tokenizer = AutoTokenizer.from_pretrained(model_id); model.save_pretrained(save_path); tokenizer.save_pretrained(save_path); print(f'Model saved to {save_path}')"

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY --from=model-builder /build/models/ models/
COPY scripts/ scripts/

RUN mkdir -p data logs

CMD ["python", "-m", "src.main"]
