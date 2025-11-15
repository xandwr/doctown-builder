# Doctown Builder

Complete pipeline for generating AI-documented code packages (.docpack files) from git repositories.

## Overview

Doctown Builder is a unified tool that:
1. **Extracts** source code from a git repository ZIP
2. **Parses** code into AST using tree-sitter
3. **Generates** AI documentation via vLLM (RunPod endpoint)
4. **Packages** everything into a .docpack file (compressed archive)

## Architecture

```
┌─────────────┐
│ Git Repo    │
│ (ZIP file)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ 1. Extract & Parse  │  <- tree-sitter AST extraction
│    (Builder)        │     (Rust, Python, JS, TS, Go)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 2. AI Generation    │  <- vLLM via RunPod API
│    (Generator)      │     (Qwen2.5-Coder-14B)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 3. Package Output   │  <- Creates .docpack (ZIP)
│    (Docpack)        │     manifest.json, tree.jsonl, content.jsonl
└─────────────────────┘
```

## Installation

```bash
cd doctown-builder
cargo build --release
```

## Configuration

Create a `.env` file:

```env
RUNPOD_API_KEY=
RUNPOD_VLLM_ENDPOINT_ID=
```

## Usage

```bash
./target/release/doctown-builder <zip-file> <repo-url> <commit-hash> <output-docpack>
```

### Example

```bash
./target/release/doctown-builder \
  repo.zip \
  https://github.com/user/repo \
  abc123def456 \
  output.docpack
```

### Parameters

- `<zip-file>` - Path to the git repository ZIP file
- `<repo-url>` - GitHub repository URL
- `<commit-hash>` - Git commit SHA that this docpack represents
- `<output-docpack>` - Output filename for the .docpack file

## Docpack Format

A `.docpack` file is a ZIP archive containing:

```
output.docpack
├── manifest.json      # Metadata about the pack
├── tree.jsonl         # File tree structure (one JSON per line)
├── content.jsonl      # AI-generated documentation (one JSON per line)
└── META/              # Optional metadata directory
    ├── MODEL_PROMPT.txt
    └── MODEL_PARAMS.json
```

### manifest.json

```json
{
  "docpack_format": 1,
  "name": "repo-name",
  "repo": "https://github.com/user/repo",
  "commit": "abc123def456",
  "generated_at": "2025-11-15T12:00:00Z",
  "version": "abc123def456",
  "generator": {
    "name": "doctown-generator",
    "version": "0.1.0",
    "model": "qwen2.5-coder-14b-instruct"
  },
  "stats": {
    "files_total": 123,
    "symbols_total": 842,
    "tokens_input": 50000,
    "tokens_output": 75000
  }
}
```

### tree.jsonl

One JSON object per line describing file structure:

```json
{"path":"src/main.rs","language":"rust","sha256":"abc123...","loc":150,"kind":"file"}
{"path":"src/lib.rs","language":"rust","sha256":"def456...","loc":200,"kind":"file"}
```

### content.jsonl

One JSON object per line with AI-generated documentation:

```json
{
  "path": "src/main.rs",
  "symbols": [
    {
      "id": "src/main.rs::main",
      "kind": "function",
      "signature": "fn main() -> Result<()>",
      "summary": "Main entry point for the application",
      "description": "Initializes the application, processes input...",
      "examples": ["let result = main();"],
      "complexity": "O(n)",
      "dependencies": ["src/lib.rs::init"],
      "notes": "Requires environment variables to be set"
    }
  ]
}
```

## Supported Languages

- Rust (.rs)
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Go (.go)

## Key Features

- **Incremental Processing**: Streams through files one at a time
- **Token Tracking**: Monitors input/output tokens for cost analysis
- **Error Handling**: Continues processing even if individual files fail
- **Size Limits**: Skips files larger than 512KB
- **Stable Symbol IDs**: Ensures consistent identification across versions

## Development

### Project Structure

```
doctown-builder/
├── src/
│   ├── main.rs       # Main pipeline orchestration
│   ├── vllm.rs       # RunPod vLLM API client
│   ├── generator.rs  # AI documentation generation
│   └── docpack.rs    # Docpack format & ZIP creation
├── Cargo.toml
└── .env
```

### Build & Test

```bash
# Development build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with example
cargo run -- example.zip https://github.com/example/repo abc123 output.docpack
```

## Performance

- **Parsing**: ~100-500 files/sec (depends on file size)
- **AI Generation**: Limited by vLLM endpoint (~1-2 files/sec)
- **Total Time**: Primarily bound by LLM inference time

## License

See repository license.
