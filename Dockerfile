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

# Copy Cargo files for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to cache dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (this layer will be cached)
RUN cargo build --release || true

# Remove dummy source
RUN rm -rf src

# Copy actual source code
COPY src ./src

# Build the actual binary
RUN cargo build --release

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/doctown-builder /usr/local/bin/doctown-builder

# Copy ONNX model for embeddings (if needed)
# Note: The model is downloaded at runtime from HuggingFace, but we can pre-download for faster startup
# RUN mkdir -p /root/.cache/huggingface

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
# RunPod will set RUNPOD_INPUT with the job input JSON
ENTRYPOINT ["doctown-builder", "--serverless"]
