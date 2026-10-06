# =============================================================================
# PoliLean Docker image
# =============================================================================
# ELI5: a Dockerfile is a recipe a chef (Docker) follows to cook up a
# repeatable box containing PoliLean, its Python libraries, the spaCy
# language model, and a model already trained - so the container works
# the moment it starts.
#
# HOW TO USE THIS RECIPE (run these from the project folder in a terminal):
#
#   docker build -t polilean .            # 1. cook the image (takes a few min)
#   docker run --rm -p 8000:8000 polilean # 2. start a container -> GUI/API
#   open http://localhost:8000/          # 3. the dark web GUI appears
#
#   ...or let docker compose do both steps:  docker compose up --build
#
#   Other handy commands while it runs:
#     docker ps                          # see the running container
#     docker logs -f polilean-api        # watch its output (Ctrl+C to stop tailing)
#     docker exec -it polilean-api bash  # "open" the container: a shell inside it
#     docker stop polilean-api           # stop it (compose: docker compose down)
#
#   One-off prediction without the server (the model is baked into the image):
#     docker run --rm polilean python -m polilean.cli predict "some text" --explain
# =============================================================================

# The base: a slim Debian + Python 3.12 image from the official Docker Hub.
# "slim" = small footprint (no extra tools we do not need).
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

# ELI5: install curl only - it lets us add a real healthcheck later and is
# useful for poking the API from inside the container. apt-get clean-up on
# the same line keeps the layer from storing the package index (= smaller image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (split for layer caching) --------------------------
# ELI5: Docker caches each step. By copying pyproject.toml first and
# installing before copying our own source code, everyday code edits do NOT
# re-download the heavy libraries (spaCy, scikit-learn...) - only this rare
# step does. `pip install .` installs PoliLean itself from the copied src/;
# the second half downloads the small English spaCy model it needs to clean text.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && python -m spacy download en_core_web_sm

# Copy the training data and tests (data is needed by the next step; tests
# are included so `docker run ... pytest` works for a quick smoke check).
COPY data ./data
COPY tests ./tests

# Bake a trained model into the image so `predict`, the API, and the GUI all
# work out of the box with zero setup. It trains the lean classifier AND the
# five value axes from the bundled CSVs, then writes src/polilean/models/trained/logreg.pkl.
RUN python -m polilean.cli train --dataset data/train.csv --axes-dataset data/axes.csv

# --- Runtime hardening -------------------------------------------------------
# Create a plain (non-root) user and switch to it: if anything inside the
# container ever gets compromised, the attacker does NOT get root powers.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Document that the web server listens on port 8000 inside the container.
# (Publish it to your machine with `docker run -p 8000:8000` or compose's ports.)
EXPOSE 8000

# What to run when the container starts: uvicorn (a production ASGI server)
# serving the FastAPI app on 0.0.0.0:8000 - 0.0.0.0 means "reachable from
# outside the container", which is what makes the -p 8000:8000 mapping work.
# To override for a one-off job, put your command after the image name, e.g.:
#   docker run --rm polilean python -m polilean.cli predict "text" --explain
CMD ["python", "-m", "uvicorn", "polilean.api:app", "--host", "0.0.0.0", "--port", "8000"]
