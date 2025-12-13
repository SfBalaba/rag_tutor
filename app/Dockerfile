# Combined Dockerfile for both backend and frontend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy pyproject.toml and application code
# Note: build context is root of project, so paths are relative to root
# README.md is excluded by .dockerignore, but it's optional for package build
COPY pyproject.toml .
COPY app/ ./app/

# Install the project as a package with app dependencies using uv
# This installs both the package and its dependencies
# The -e flag installs in editable mode, so changes to code are reflected
RUN uv pip install --system --no-cache -e ".[app]"

# Create symlinks or copy to make backend/ and frontend/ available at root level
# This maintains compatibility with existing code paths
RUN ln -s /app/app/backend /app/backend && \
    ln -s /app/app/frontend /app/frontend

# .env will be provided via docker-compose env_file

# Expose ports
EXPOSE 8000 8501

# Create startup script that runs both services
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting FastAPI backend..."\n\
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &\n\
sleep 3\n\
echo "Starting Streamlit frontend..."\n\
streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &\n\
echo "Both services started. Waiting..."\n\
wait' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

