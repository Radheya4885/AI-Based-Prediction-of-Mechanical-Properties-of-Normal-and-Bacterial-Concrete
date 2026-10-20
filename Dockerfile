FROM python:3.10-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install PyTorch CPU first to keep build fast
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY --chown=user:user requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy application source, models, and frontend
COPY --chown=user:user src/ ./src/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user data/ ./data/
COPY --chown=user:user results/ ./results/

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI backend with mounted frontend
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
