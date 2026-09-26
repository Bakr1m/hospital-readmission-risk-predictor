FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/hospital-readmission-risk-predictor/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=5c81bbdcecab45f77286fa14a7fb5d6fc79632086e7323381c23887d0e075cc4
RUN mkdir -p models && \
    curl -fsSL -o models/readmission_best.joblib \
      "https://github.com/Bakr1m/hospital-readmission-risk-predictor/releases/download/${MODEL_TAG}/readmission_best.joblib" && \
    echo "${MODEL_SHA256}  models/readmission_best.joblib" | sha256sum -c - && \
    python -c "import joblib; m=joblib.load('models/readmission_best.joblib'); print('artifact OK:', type(m).__name__)"

EXPOSE 8000

CMD ["python", "api/main.py"]