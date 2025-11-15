# Multi-stage build for RunPod serverless deployment
# Stage 1: Build the Rust binary
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies including git (for build metadata)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Stage 2: Runtime image with Python for RunPod handler
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary from builder
COPY --from=builder /build/target/release/doctown-builder /app/doctown-builder

# Copy Python handler and requirements
COPY handler.py /app/handler.py
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# RunPod expects the handler to be available
# The handler.py file should export a handler(job) function
CMD ["python", "-u", "handler.py"]
