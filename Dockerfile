FROM python:3.12-slim

# No .pyc, unbuffered logs, no noisy telemetry
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (none needed beyond base for spaCy en_core_web_sm runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps first for layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m spacy download en_core_web_sm

# Copy data + remaining project files
COPY data ./data
COPY tests ./tests

# Bake a trained model into the image so `predict` works out of the box
RUN python -m polilean.cli train --dataset data/train.csv --axes-dataset data/axes.csv

# Non-root runtime
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: train (if model absent) then serve the FastAPI app.
# Override command for one-off predictions:
#   docker run --rm polilean predict "text here" --explain
CMD ["python", "-m", "uvicorn", "polilean.api:app", "--host", "0.0.0.0", "--port", "8000"]
