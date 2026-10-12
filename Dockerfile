FROM python:3.12-slim

# Container hygiene flags:
#   PYTHONDONTWRITEBYTECODE=1  - do not litter .pyc files inside the image
#   PYTHONUNBUFFERED=1         - print logs immediately (no delayed buffering)
#   PIP_NO_CACHE_DIR=1         - skip pip's download cache to keep layers small
#   PIP_DISABLE_PIP_VERSION_CHECK=1 - stop pip from phoning home to check versions
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# All following commands run inside /app (created automatically).
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m spacy download en_core_web_sm

# Copy the training data and tests 
COPY data ./data
COPY tests ./tests

# Bake a trained model into the image so `predict`, the API, and the GUI all work out of the box with zero setup.
RUN python -m polilean.cli train --dataset data/train.csv --axes-dataset data/axes.csv

# Create a plain (non-root) user and switch to it: if anything inside the
# container ever gets compromised, the attacker does NOT get root powers.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Document that the web server listens on port 8000 inside the container.
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "polilean.api:app", "--host", "0.0.0.0", "--port", "8000"]
