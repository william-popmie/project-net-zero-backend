"""
FastAPI server — receives a project zip, runs the optimization pipeline,
and returns the modified project as a zip.
"""

from __future__ import annotations

import shutil
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.main import (
    CONFIG,
    find_python_project_root,
    install_input_repo_requirements,
    run_pipeline,
    write_python_files,
)

app = FastAPI(title="Project Net-Zero Backend")

BASE_DIR = Path(__file__).parent
INPUT_REPO = BASE_DIR / "input-repo"
OUTPUT_REPO = BASE_DIR / "output-repo"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/optimize")
async def optimize(
    project: UploadFile = File(...),
    entrypoint: str = Form(""),
):
    try:
        # ── Clear input-repo and extract uploaded zip into it ─────────────────
        if INPUT_REPO.exists():
            shutil.rmtree(INPUT_REPO)
        INPUT_REPO.mkdir(parents=True)

        contents = await project.read()
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            zf.extractall(INPUT_REPO)

        # ── Discover project root and install requirements ────────────────────
        project_root = find_python_project_root(INPUT_REPO)
        install_input_repo_requirements(project_root)

        # ── Run pipeline ──────────────────────────────────────────────────────
        run_pipeline(project_root)

        # ── Populate output-repo (copy + splice + results.json) ───────────────
        write_python_files(
            json_file=Path(CONFIG["optimizer_output"]),
            output_dir=OUTPUT_REPO,
            input_repo_dir=INPUT_REPO,
        )

        # ── Zip output-repo (project folder + results.json) ───────────────────
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in OUTPUT_REPO.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(OUTPUT_REPO))
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=optimized.zip"},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
