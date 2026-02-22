"""
FastAPI server — receives a project zip, runs the optimization pipeline,
and returns the modified project as a zip.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.main import run_pipeline

app = FastAPI(title="Project Net-Zero Backend")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/optimize")
async def optimize(
    project: UploadFile = File(...),
    entrypoint: str = Form(""),
):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # ── Extract uploaded zip ─────────────────────────────────────────
        contents = await project.read()
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            zf.extractall(tmp_dir)

        # If the zip contains a single top-level directory, use that as root
        top_level = [p for p in tmp_dir.iterdir() if p.name != "__MACOSX"]
        if len(top_level) == 1 and top_level[0].is_dir():
            project_root = top_level[0]
        else:
            project_root = tmp_dir

        # ── Run pipeline ─────────────────────────────────────────────────
        run_pipeline(project_root)

        # ── Zip the (now-modified) project back up ───────────────────────
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in project_root.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(project_root))
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=optimized.zip"},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
