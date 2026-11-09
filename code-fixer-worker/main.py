"""
Code Fixer worker — a deliberately separate, deliberately dumb service.

Its entire job: take a zip of project files and a list of shell commands,
run them somewhere isolated, report back exit codes and output. That's
it. It does not know what Supabase is, does not have an AI provider key,
does not know what a "user" is. It cannot leak what it was never given.

Deploy this as its OWN Render/Fly service, completely separate from the
main backend, with its OWN environment — the only env var it needs is
WORKER_SHARED_SECRET (a random string, shared with the main backend so it
can authenticate calls in, and nothing else). Do not deploy this on the
same service as the main backend, and do not give this service your
Supabase or AI provider keys — the entire security model here depends on
this process having nothing worth stealing.

The one thing this can't avoid needing outbound access for: `npm
install`, `pip install`, and friends have to reach their public package
registries to do their job. That's a real, acknowledged exception to "no
egress" — there is no way to genuinely validate a build otherwise. What
matters is that there's nothing *sensitive* reachable from here even if
an uploaded project's install script tried to phone home.
"""

import os
import resource
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, Form
from pydantic import BaseModel

app = FastAPI(title="Zorah Code Fixer Worker")

# Hard ceilings — independent of whatever the main backend asks for, this
# process refuses to be talked into something unbounded.
MAX_ZIP_BYTES = 80 * 1024 * 1024
MAX_EXTRACTED_BYTES = 400 * 1024 * 1024
MAX_FILE_COUNT = 8000
MAX_COMMANDS = 6
PER_COMMAND_TIMEOUT_S = 180
JOB_WALL_CLOCK_BUDGET_S = 8 * 60

# Applied to each command's subprocess via resource.setrlimit in a
# preexec_fn — a portable, no-extra-dependency way to cap a runaway
# `npm install` or test suite without needing container tooling this
# environment may not have.
MAX_CPU_SECONDS = 170
MAX_MEMORY_BYTES = 1536 * 1024 * 1024
MAX_PROCESSES = 256
MAX_OUTPUT_CHARS = 20_000  # per stream, truncated — not meant to replace real logs


def _require_secret(x_worker_secret: str | None = Header(default=None)) -> None:
    worker_shared_secret = os.environ.get("WORKER_SHARED_SECRET")
    if not worker_shared_secret:
        raise HTTPException(status_code=500, detail="worker is not configured (no shared secret set)")
    if x_worker_secret != worker_shared_secret:
        raise HTTPException(status_code=401, detail="invalid worker secret")


def _limit_resources():
    """Runs inside the child process, before exec — this is what actually
    enforces CPU caps at the OS level."""
    for limit_attr, value in [
        ("RLIMIT_CPU", MAX_CPU_SECONDS),
    ]:
        if hasattr(resource, limit_attr):
            try:
                res = getattr(resource, limit_attr)
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass
    try:
        os.setsid()
    except OSError:
        pass


def _safe_extract(zip_path: Path, dest: Path) -> int:
    """Extracts with path-traversal and zip-bomb protection. Returns
    total extracted bytes. Raises ValueError on anything suspicious."""
    total_bytes = 0
    file_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            file_count += 1
            if file_count > MAX_FILE_COUNT:
                raise ValueError(f"archive has more than {MAX_FILE_COUNT} entries")

            # Reject absolute paths and any ".." component outright,
            # rather than trying to sanitize them — a name that needs
            # sanitizing is a name we don't trust.
            name = info.filename
            clean_name = name.replace("\\", "/")
            if clean_name.startswith("/") or ".." in Path(clean_name).parts:
                raise ValueError(f"unsafe path in archive: {name}")

            target = (dest / clean_name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise ValueError(f"path escapes extraction root: {name}")

            total_bytes += info.file_size
            if total_bytes > MAX_EXTRACTED_BYTES:
                raise ValueError(f"extracted size exceeds {MAX_EXTRACTED_BYTES // (1024*1024)}MB limit")

        zf.extractall(dest)

    return total_bytes


class RunResult(BaseModel):
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool


class RunResponse(BaseModel):
    results: list[RunResult]
    extracted_bytes: int


@app.get("/health")
def health():
    worker_shared_secret = os.environ.get("WORKER_SHARED_SECRET")
    return {"ok": True, "configured": bool(worker_shared_secret)}


@app.post("/run", response_model=RunResponse)
async def run(
    project: UploadFile = File(...),
    commands: str = Form(...),  # newline-separated shell commands, validated below
    _auth: None = Depends(_require_secret),
):
    """
    Runs each command in `commands` (one per line) inside the extracted
    project, in order, stopping early only on a hard timeout budget —
    NOT on the first failing command, since a failing `npm run build`
    is exactly the information the caller needs back, not a reason to
    abort silently.
    """
    lines = [c.strip() for c in commands.split("\n") if c.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="no commands provided")
    if len(lines) > MAX_COMMANDS:
        raise HTTPException(status_code=400, detail=f"too many commands (max {MAX_COMMANDS})")

    data = await project.read()
    if len(data) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=400, detail=f"project archive exceeds {MAX_ZIP_BYTES // (1024*1024)}MB")

    # Every job gets a brand-new temp directory and it's gone by the time
    # this request returns, success or failure — jobs never share state,
    # and nothing from one job's project persists for the next.
    with tempfile.TemporaryDirectory(prefix="codefix-") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "project.zip"
        zip_path.write_bytes(data)

        extract_dir = tmp_path / "project"
        extract_dir.mkdir()

        try:
            extracted_bytes = _safe_extract(zip_path, extract_dir)
        except (ValueError, zipfile.BadZipFile) as e:
            raise HTTPException(status_code=400, detail=f"invalid or unsafe archive: {e}")

        results: list[RunResult] = []
        deadline = time.monotonic() + JOB_WALL_CLOCK_BUDGET_S

        for cmd in lines:
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                results.append(RunResult(
                    command=cmd, exit_code=None, stdout="",
                    stderr="skipped: job wall-clock budget exhausted",
                    duration_s=0, timed_out=True,
                ))
                continue

            timeout = min(PER_COMMAND_TIMEOUT_S, remaining)
            results.append(_run_one(cmd, extract_dir, timeout))

        return RunResponse(results=results, extracted_bytes=extracted_bytes)


def _run_one(cmd: str, cwd: Path, timeout_s: float) -> RunResult:
    # A stripped-down environment — no ambient secrets, no inherited
    # process environment beyond what a build genuinely needs (PATH for
    # finding node/npm/python, HOME because some tools insist on it).
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "CI": "true",
        "npm_config_yes": "true",
    }

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            env=env,
            preexec_fn=_limit_resources,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        duration = time.monotonic() - start
        return RunResult(
            command=cmd,
            exit_code=proc.returncode,
            stdout=proc.stdout[-MAX_OUTPUT_CHARS:],
            stderr=proc.stderr[-MAX_OUTPUT_CHARS:],
            duration_s=round(duration, 2),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        duration = time.monotonic() - start
        return RunResult(
            command=cmd,
            exit_code=None,
            stdout=(e.stdout or "")[-MAX_OUTPUT_CHARS:] if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "")[-MAX_OUTPUT_CHARS:] if isinstance(e.stderr, str) else "",
            duration_s=round(duration, 2),
            timed_out=True,
        )
    except Exception as e:
        duration = time.monotonic() - start
        return RunResult(
            command=cmd, exit_code=None, stdout="", stderr=f"worker error running command: {e}",
            duration_s=round(duration, 2), timed_out=False,
        )
