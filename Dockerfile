# Stage 9 — Production container for ScholarCite RAG
FROM python:3.13-slim

WORKDIR /app

# System deps needed by PyMuPDF, sentence-transformers, and general build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first (better Docker layer caching — deps only
# reinstall when these files change, not on every code edit).
# --no-install-project: at this point src/ doesn't exist in the build
# context yet, so we only install the project's DEPENDENCIES here, not
# the project itself as a package — avoids "Expected a Python module..."
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application
COPY . .

# Now that src/ actually exists, finish the sync so the project itself
# (if needed as an installed package) and any remaining deps are set up.
RUN uv sync --frozen --no-dev

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]