FROM python:3.11-slim

# Install system dependencies if required
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv package manager globally
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy dependency definitions and the lockfile
COPY pyproject.toml uv.lock ./

# Install project dependencies into the container
RUN uv sync --frozen

# Copy the rest of the source code
COPY . .

# Ensure Python treats the root as the top-level module path
ENV PYTHONPATH=/app

# Default command to run the interactive CLI
CMD ["uv", "run", "python", "-m", "ml_framework_project.main"]
