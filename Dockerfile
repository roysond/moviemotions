# ─────────────────────────────────────────────────────────────────────────────
# MovieMotions — one image containing the API and the built front end.
#
# TWO STAGES, AND WHY
#   Building the React app needs Node, npm and ~200MB of node_modules. RUNNING it
#   needs none of that — only the handful of files Vite emits. So the build happens
#   in a throwaway stage and only its output is copied forward. The shipped image
#   has no Node in it at all.
#
# WHAT IS DELIBERATELY ABSENT
#   requirements.txt installs pandas, pyarrow, datasets, ragas and scipy for the
#   evals. None of it runs in production. The image installs requirements-runtime.txt
#   instead, and tests/test_runtime_deps.py fails if that file ever falls behind
#   what backend/ imports.
# ─────────────────────────────────────────────────────────────────────────────

# ── stage 1 · build the front end ────────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build
# package files first, on their own layer: dependencies only reinstall when THEY
# change, not every time a component is edited.
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci

COPY frontend/ ./frontend/
# vite.config.ts writes to ../static/app, so that folder has to exist first.
RUN mkdir -p static && cd frontend && npm run build


# ── stage 2 · the image that actually runs ───────────────────────────────────
FROM python:3.13-slim AS runtime

# PYTHONDONTWRITEBYTECODE — no .pyc files in a container that is thrown away.
# PYTHONUNBUFFERED     — print() reaches the log immediately instead of sitting
#                        in a buffer, which is the difference between seeing a
#                        crash and watching a container die silently.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY backend/ ./backend/
COPY static/index.html ./static/index.html
COPY --from=frontend /build/static/app ./static/app

# Not root. A web process should never be able to modify its own code, and the
# default in a container is root unless you say otherwise.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# --host 0.0.0.0 is mandatory in a container. The default binds to localhost,
# which inside a container means "reachable only from inside this container" —
# the request arrives, finds nothing listening, and the platform reports the
# service as unhealthy with no error in the log.
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
