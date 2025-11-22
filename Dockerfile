# RunPod Dockerfile for doctown-builder
# Uses Rust nightly with edition 2024 support

# Stage 1: Build the binary
FROM rustlang/rust:nightly AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache buster - change this to force rebuild
ARG CACHE_BUST=v2

# Copy Cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to cache dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (this layer will be cached)
RUN cargo build --release || true

# Remove the dummy binary and source completely
RUN rm -rf src target/release/doctown-builder target/release/deps/doctown_builder*

# Copy actual source code
COPY src ./src

# Build the actual binary (force rebuild)
RUN cargo build --release

# Stage 2: Runtime image
# Use the same Debian base as the builder to ensure GLIBC compatibility
FROM debian:trixie-slim

# Install runtime dependencies including Python
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK
RUN pip install --no-cache-dir --break-system-packages runpod

# Create working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/doctown-builder /usr/local/bin/doctown-builder

# Copy the handler script
COPY handler.py /app/handler.py

# Copy the ONNX models for embeddings
COPY models /app/models

# Create output directory
RUN mkdir -p /app/output

# Environment variables that should be set by RunPod
# BUCKET_S3_ENDPOINT - R2 endpoint
# BUCKET_NAME - R2 bucket name
# BUCKET_ACCESS_KEY_ID - R2 access key
# BUCKET_SECRET_ACCESS_KEY - R2 secret key
# DOCTOWN_BUILDER_SHARED_SECRET - Webhook auth secret
# WEBHOOK_URL - URL to call on completion
# OPENAI_API_KEY - For LLM documentation generation

# RunPod serverless handler entry point
CMD ["python3", "-u", "/app/handler.py"]
