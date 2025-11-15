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


def get_branch_commit_hash(owner: str, repo: str, branch: str, token: str) -> str:
    """
    Get the latest commit hash for a branch.

    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name
        token: GitHub OAuth token

    Returns:
        The commit SHA hash
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/branches/{branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data["commit"]["sha"]


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
        # Step 0: Verify binary exists
        binary_path = "/app/doctown-builder"
        if not os.path.exists(binary_path):
            yield {
                "error": f"Binary not found at {binary_path}",
                "status": "failed"
            }
            return

        print(f"[RunPod] Binary found at {binary_path}", file=sys.stderr)
        sys.stderr.flush()

        # Step 1: Get commit hash
        yield {
            "status": "fetching_metadata",
            "message": f"Fetching branch info for {owner}/{repo}...",
            "progress": 5
        }

        commit_hash = get_branch_commit_hash(owner, repo, branch, token)

        yield {
            "status": "metadata_fetched",
            "message": f"Commit: {commit_hash[:7]}",
            "progress": 10
        }

        # Step 2: Download repository ZIP
        yield {
            "status": "downloading",
            "message": f"Downloading {owner}/{repo} (branch: {branch})...",
            "progress": 15
        }

        download_github_zip(owner, repo, branch, token, zip_path)

        yield {
            "status": "downloaded",
            "message": f"Downloaded ZIP file ({os.path.getsize(zip_path)} bytes)",
            "progress": 30
        }

        # Step 3: Process with Rust binary
        yield {
            "status": "processing",
            "message": "Starting AST parsing and doc generation...",
            "progress": 40
        }

        # Construct repo URL
        repo_url = f"https://github.com/{owner}/{repo}"

        # Output path for docpack (in temp directory)
        output_docpack = os.path.join(temp_dir, f"{repo}.docpack")

        print(f"[RunPod] Executing: {binary_path} {zip_path} {repo_url} {commit_hash} {output_docpack}", file=sys.stderr)
        sys.stderr.flush()

        # Use subprocess.run with capture for simpler handling
        try:
            result = subprocess.run(
                [binary_path, zip_path, repo_url, commit_hash, output_docpack],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout (includes vLLM calls)
            )

            print(f"[RunPod] Process completed with code {result.returncode}", file=sys.stderr)
            sys.stderr.flush()

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"[RunPod] Process error: {error_msg}", file=sys.stderr)
                sys.stderr.flush()
                yield {
                    "error": f"Process failed: {error_msg}",
                    "status": "failed"
                }
                return

            # Parse stderr for log messages
            if result.stderr:
                for line in result.stderr.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("[PROGRESS]"):
                        # Parse progress messages like: "Processing file 3/10: src/main.rs"
                        progress_msg = line[10:].strip()

                        # Extract file count if present
                        progress = 50  # Default for processing stage
                        if "Processing file" in progress_msg:
                            try:
                                # Extract "X/Y" pattern
                                import re
                                match = re.search(r'(\d+)/(\d+)', progress_msg)
                                if match:
                                    current = int(match.group(1))
                                    total = int(match.group(2))
                                    # VLLM stage is 40-80% of total progress
                                    vllm_progress = (current / total) * 40
                                    progress = 40 + int(vllm_progress)
                            except:
                                pass

                        yield {
                            "status": "processing",
                            "message": progress_msg,
                            "progress": progress
                        }
                    elif line.startswith("[LOG]"):
                        log_msg = line[5:].strip()
                        # Map progress messages to percentages
                        progress = 50
                        if "Parsing" in log_msg or "Extracting" in log_msg:
                            progress = 35
                        elif "Processing" in log_msg and "files" in log_msg:
                            progress = 40
                        elif "Generating documentation" in log_msg or "Generating docs" in log_msg:
                            progress = 50
                        elif "Generated" in log_msg and "symbols" in log_msg:
                            progress = 60
                        elif "Building .docpack" in log_msg or "Building docpack" in log_msg:
                            progress = 80
                        elif "Uploading" in log_msg:
                            progress = 90
                        elif "Uploaded to S3" in log_msg:
                            progress = 95
                        elif "TOKENS" in log_msg:
                            # Token tracking messages - show but don't change progress
                            progress = None

                        if progress is not None:
                            yield {
                                "status": "processing",
                                "message": log_msg,
                                "progress": progress
                            }
                    elif line.startswith("[ERROR]"):
                        error_msg = line[7:].strip()
                        yield {
                            "status": "processing",
                            "message": f"ERROR: {error_msg}",
                            "progress": 50
                        }

            # Check that docpack was created
            if not os.path.exists(output_docpack):
                yield {
                    "error": "Docpack file was not created",
                    "status": "failed"
                }
                return

            # Parse stdout for S3 URL (last line should be the S3 key)
            s3_key = None
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if lines:
                    s3_key = lines[-1].strip()

            if not s3_key:
                yield {
                    "error": "S3 key not returned from builder",
                    "status": "failed"
                }
                return

            print(f"[RunPod] Docpack uploaded to: {s3_key}", file=sys.stderr)
            sys.stderr.flush()

        except subprocess.TimeoutExpired:
            yield {
                "error": "Process timed out after 10 minutes",
                "status": "failed"
            }
            return
        except Exception as e:
            print(f"[RunPod] Subprocess error: {e}", file=sys.stderr)
            sys.stderr.flush()
            yield {
                "error": f"Failed to execute builder: {str(e)}",
                "status": "failed"
            }
            return

        # Step 4: Complete
        yield {
            "status": "completed",
            "message": f"Docpack created successfully!",
            "progress": 100,
            "s3_key": s3_key,
            "docpack_url": f"https://commons.doctown.dev/{s3_key}" # R2 public domain on C.F
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


def handler(job: Dict[str, Any]):
    """
    RunPod serverless handler function.
    This function is called by RunPod for each job.

    Args:
        job: Job dictionary with 'id' and 'input' keys

    Returns:
        List of all status updates and outputs
    """
    job_input = job.get("input", {})
    outputs = []

    # Log job start
    print(f"[RunPod] Starting job {job.get('id')}", file=sys.stderr)
    print(f"[RunPod] Input: {job_input}", file=sys.stderr)
    sys.stderr.flush()

    # Add initial status
    outputs.append({
        "status": "initializing",
        "message": "Handler started",
        "progress": 0
    })

    try:
        # Process and collect all updates
        for update in process_repository(job_input):
            print(f"[RunPod] Collecting: {update}", file=sys.stderr)
            sys.stderr.flush()
            outputs.append(update)
    except Exception as e:
        print(f"[RunPod] Handler error: {e}", file=sys.stderr)
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        outputs.append({
            "error": f"Handler exception: {str(e)}",
            "status": "failed"
        })

    print(f"[RunPod] Returning {len(outputs)} outputs", file=sys.stderr)
    sys.stderr.flush()
    return outputs


if __name__ == "__main__":
    # Start the RunPod serverless worker
    print("[RunPod] Starting doctown-builder serverless worker...", file=sys.stderr)
    sys.stderr.flush()
    runpod.serverless.start({"handler": handler})
