FROM python:3.10-slim

# Prevent Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Bento config
COPY bentofile.yaml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r <(python - <<EOF
from yaml import safe_load
import sys
cfg = safe_load(open("bentofile.yaml"))
print("\n".join(cfg["python"]["packages"]))
EOF)


# Copy service code
COPY service.py .

# Expose BentoML port
EXPOSE 3000

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=https://dagshub.com/Aditya-Raj-Kaushik/complaint-priority.mlflow

# Start BentoML service
CMD ["bentoml", "serve", "service:svc", "--production"]
