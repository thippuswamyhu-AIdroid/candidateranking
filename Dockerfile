FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    pkg-config \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==1.13.0 \
    torchvision==0.14.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

RUN python -B download_hf_model_files.py

CMD ["python", "-B", "candidate_ranker.py"]