FROM python:3.12-slim

# Torch is heavy; install system deps it needs
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer
COPY dashboard/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the whole project — the backend imports from feeder_env.py at the root
COPY . /app

# Render sets $PORT at runtime
ENV PORT=8000
EXPOSE 8000

# Run from the backend directory so relative model paths resolve
WORKDIR /app/dashboard/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]