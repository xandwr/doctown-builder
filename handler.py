#!/usr/bin/env python3
"""
RunPod serverless handler for doctown-builder.

This wrapper receives jobs from RunPod and invokes the Rust binary
with the job input passed via RUNPOD_INPUT environment variable.
"""

import os
import subprocess
import json
import runpod


def handler(job):
    """
    Handle a RunPod job by invoking the doctown-builder binary.

    Args:
        job: Dict containing 'id' and 'input' from RunPod

    Returns:
        Dict with job output or error
    """
    job_input = job.get("input", {})

    # Set the input as an environment variable for the builder
    env = os.environ.copy()
    env["RUNPOD_INPUT"] = json.dumps(job_input)

    try:
        # Run the builder binary
        result = subprocess.run(
            ["/usr/local/bin/doctown-builder", "--serverless"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Log stdout/stderr for debugging
        print(f"[handler] Exit code: {result.returncode}", flush=True)
        print(f"[handler] Stdout length: {len(result.stdout) if result.stdout else 0}", flush=True)
        print(f"[handler] Stderr length: {len(result.stderr) if result.stderr else 0}", flush=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, flush=True)

        # Check exit code
        if result.returncode != 0:
            return {
                "error": f"Builder exited with code {result.returncode}",
                "stderr": result.stderr[-2000:] if result.stderr else None
            }

        # Try to parse the last line as JSON output
        stdout_lines = result.stdout.strip().split('\n')
        for line in reversed(stdout_lines):
            try:
                output = json.loads(line)
                if "success" in output or "job_id" in output:
                    return output
            except json.JSONDecodeError:
                continue

        # No JSON output found, return raw output
        return {
            "success": True,
            "output": result.stdout[-2000:] if result.stdout else "No output"
        }

    except subprocess.TimeoutExpired:
        return {"error": "Build timed out after 10 minutes"}
    except Exception as e:
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})
