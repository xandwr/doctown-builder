#!/usr/bin/env python3
"""
RunPod serverless handler for doctown-builder.
Downloads a GitHub repository ZIP and processes it with the Rust AST parser.
"""

import os
import sys
import subprocess
import tempfile
import requests
import runpod
from typing import Generator, Dict, Any


def download_github_zip(owner: str, repo: str, branch: str, token: str, output_path: str) -> None:
    """
    Download a GitHub repository as a ZIP file.

    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name
        token: GitHub OAuth token
        output_path: Where to save the ZIP file
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    response = requests.get(url, headers=headers, stream=True, timeout=300)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def process_repository(job_input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Main processing function that yields progress updates.

    Args:
        job_input: Job parameters from RunPod

    Yields:
        Status updates for streaming to frontend
    """
    # Extract job parameters
    owner = job_input.get("owner")
    repo = job_input.get("repo")
    branch = job_input.get("branch", "main")
    token = job_input.get("github_token")

    # Validate inputs
    if not owner or not repo:
        yield {
            "error": "Missing required parameters: owner and repo are required",
            "status": "failed"
        }
        return

    if not token:
        yield {
            "error": "Missing GitHub token",
            "status": "failed"
        }
        return

    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix="doctown_")
    zip_path = os.path.join(temp_dir, "repo.zip")

    try:
        # Step 1: Download repository ZIP
        yield {
            "status": "downloading",
            "message": f"Downloading {owner}/{repo} (branch: {branch})...",
            "progress": 10
        }

        download_github_zip(owner, repo, branch, token, zip_path)

        yield {
            "status": "downloaded",
            "message": f"Downloaded ZIP file ({os.path.getsize(zip_path)} bytes)",
            "progress": 30
        }

        # Step 2: Process with Rust binary
        yield {
            "status": "processing",
            "message": "Starting AST parsing...",
            "progress": 40
        }

        # Execute the Rust binary
        binary_path = "/app/doctown-builder"
        process = subprocess.Popen(
            [binary_path, zip_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        jsonl_lines = []
        file_count = 0

        # We need to read both stdout and stderr concurrently
        # Use threads to avoid blocking
        import threading
        import queue

        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()

        def read_stdout():
            if process.stdout:
                for line in process.stdout:
                    stdout_queue.put(('stdout', line.strip()))
                stdout_queue.put(('stdout', None))  # Signal EOF

        def read_stderr():
            if process.stderr:
                for line in process.stderr:
                    stderr_queue.put(('stderr', line.strip()))
                stderr_queue.put(('stderr', None))  # Signal EOF

        # Start reader threads
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Read from both streams
        stdout_done = False
        stderr_done = False

        while not (stdout_done and stderr_done):
            # Check stderr first for log messages
            try:
                stream_type, line = stderr_queue.get(timeout=0.1)
                if line is None:
                    stderr_done = True
                    continue

                # Parse log messages from stderr
                if line.startswith("[LOG]"):
                    log_msg = line[5:].strip()
                    yield {
                        "status": "processing",
                        "message": log_msg,
                        "progress": min(40 + file_count, 90)
                    }
                elif line.startswith("[ERROR]"):
                    error_msg = line[7:].strip()
                    yield {
                        "status": "processing",
                        "message": f"ERROR: {error_msg}",
                        "progress": min(40 + file_count, 90)
                    }
            except queue.Empty:
                pass

            # Check stdout for JSONL data
            try:
                stream_type, line = stdout_queue.get(timeout=0.1)
                if line is None:
                    stdout_done = True
                    continue

                if line:
                    # This is JSONL data
                    jsonl_lines.append(line)
                    file_count += 1

                    # Yield the data chunk for streaming
                    yield {
                        "status": "processing",
                        "data_chunk": line
                    }
            except queue.Empty:
                pass

        # Wait for threads and process to complete
        stdout_thread.join()
        stderr_thread.join()
        return_code = process.wait()

        # Check for errors
        if return_code != 0:
            yield {
                "error": f"Process failed with exit code {return_code}",
                "status": "failed"
            }
            return

        # Step 3: Complete
        yield {
            "status": "completed",
            "message": f"Successfully processed {file_count} files",
            "progress": 100,
            "file_count": file_count,
            "total_lines": len(jsonl_lines)
        }

    except requests.exceptions.RequestException as e:
        yield {
            "error": f"Failed to download repository: {str(e)}",
            "status": "failed"
        }
    except subprocess.SubprocessError as e:
        yield {
            "error": f"Failed to execute builder: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        yield {
            "error": f"Unexpected error: {str(e)}",
            "status": "failed"
        }
    finally:
        # Cleanup temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    RunPod serverless handler function.
    This function is called by RunPod for each job.

    Args:
        job: Job dictionary with 'id' and 'input' keys

    Yields:
        Status updates for streaming
    """
    job_input = job.get("input", {})

    # Log job start
    print(f"[RunPod] Starting job {job.get('id')}", file=sys.stderr)
    print(f"[RunPod] Input: {job_input}", file=sys.stderr)

    # Process and yield updates
    yield from process_repository(job_input)


if __name__ == "__main__":
    # Start the RunPod serverless worker
    print("[RunPod] Starting doctown-builder serverless worker...", file=sys.stderr)
    runpod.serverless.start({"handler": handler})
