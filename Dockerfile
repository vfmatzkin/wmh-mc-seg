FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set up a non-root user
RUN useradd -m appuser

# Create Python virtual environment with proper permissions
RUN python -m venv /opt/venv && \
    chown -R appuser:appuser /opt/venv

# Set proper path
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies at build time
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[dev]"

# Switch to appuser for the remaining operations
USER appuser

# Create a kernel for the virtual environment (jupyter installed via [dev] deps)
RUN python -m ipykernel install --user --name=wmh-mc-seg-env --display-name="Python (wmh-mc-seg)"

# Make sure the appuser owns the necessary directories
USER root
RUN chown -R appuser:appuser /app /home/appuser/.local
USER appuser

CMD ["bash"]