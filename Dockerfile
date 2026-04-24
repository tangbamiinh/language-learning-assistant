# syntax=docker/dockerfile:1

# ─── Base image ────────────────────────────────────────────────────────────────
# Use UV Python 3.12 base image (fast package manager, matches project)
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# ─── Build stage ───────────────────────────────────────────────────────────────
FROM base AS build

# Install build dependencies for Python packages with native extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libsndfile1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Install all dependencies including livekit extras
# Note: We install [dev,livekit] to get pytest + livekit-agents with silero + turn-detector
RUN uv pip install --system -e ".[dev,livekit]"

# Copy all source code
COPY src/ ./src/

# ─── Production stage ─────────────────────────────────────────────────────────
FROM base

# Install runtime dependencies (needed for audio processing)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Create non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Copy the application and installed packages from build stage
COPY --from=build --chown=appuser:appuser /usr/local /usr/local
COPY --from=build --chown=appuser:appuser /app /app

WORKDIR /app
USER appuser

# Expose the health check port
EXPOSE 8081

# Run the voice agent in production mode
CMD ["python", "-m", "src.voice_agent", "start"]
