"""
Local (no-API) build routes. Mount in server.py:

    from local_routes import router as local_router
    app.include_router(local_router)

Endpoints:
  POST /local/build          form fields: prompt (str, optional), photo (file, optional), grid (int, default 32)
  GET  /local/jobs/{id}      job record: steps, checks, render URLs, model
  GET  /local/renders/{id}/{name}   PNG for one step (name like step_03.png)
  GET  /local/builder        standalone page (local_builder.html) if present next to this file
"""
import io
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, UnidentifiedImageError

import local_pipeline as lp
import teach
from pydantic import BaseModel

router = APIRouter()

RENDER_DIR = Path(os.getenv("LOCAL_RENDER_DIR", "/tmp/verdant_local_renders"))
MAX_PHOTO_BYTES = 8 * 1024 * 1024
JOBS: Dict[str, Dict[str, Any]] = {}
MAX_JOBS = 200


def _remember(job: Dict[str, Any]) -> None:
    JOBS[job["id"]] = job
    while len(JOBS) > MAX_JOBS:
        JOBS.pop(next(iter(JOBS)))


@router.post("/local/build")
async def local_build(prompt: str = Form(""), grid: int = Form(32), photo: Optional[UploadFile] = File(None)):
    prompt = (prompt or "").strip()[:500]
    if not prompt and photo is None:
        raise HTTPException(400, "Give a prompt, a photo, or both.")

    job_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    log = []

    if photo is not None:
        raw = await photo.read(MAX_PHOTO_BYTES + 1)
        if len(raw) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "Photo is larger than 8 MB.")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Could not read that file as an image.")
        log.append(f"Photo loaded ({image.size[0]}x{image.size[1]}).")
        plan = lp.photo_to_blocks(image, grid_w=grid)
        if prompt:
            plan["notes"].append("Prompt was ignored for photo builds; the photo sets the model.")
    else:
        plan = lp.plan_from_prompt(prompt)
        log.append(f"Prompt matched: {plan['recipe'] or 'no recipe'}.")

    if not plan["matched"]:
        job = {"id": job_id, "status": "unmatched", "prompt": prompt, "steps": [], "notes": plan["notes"],
               "log": log, "created": t0}
        _remember(job)
        return job

    snaps = lp.cumulative_parts(plan["steps"])
    names = [s["name"] for s in plan["steps"]]
    titles = [f"Step {i + 1}/{len(names)}: {n}" for i, n in enumerate(names)]
    pngs = lp.render_all(snaps, titles=titles)

    job_dir = RENDER_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    steps_out = []
    for i, (s, snap, png) in enumerate(zip(plan["steps"], snaps, pngs), start=1):
        fname = f"step_{i:02d}.png"
        (job_dir / fname).write_bytes(png)
        steps_out.append({
            "index": i,
            "name": s["name"],
            "new_parts": len(s["parts"]),
            "total_parts": len(snap),
            "render_url": f"/local/renders/{job_id}/{fname}",
            "check": lp.check_model(snap),
        })
        log.append(f"Step {i}: {s['name']} ({len(s['parts'])} new parts).")

    final_parts = snaps[-1]
    final_check = lp.check_model(final_parts)
    job = {
        "id": job_id,
        "status": "done",
        "prompt": prompt,
        "recipe": plan["recipe"],
        "steps": steps_out,
        "check": final_check,
        "notes": plan["notes"],
        "log": log + [f"Finished in {time.time() - t0:.2f}s."],
        "model": {"parts": final_parts},
        "plan": plan["steps"],
        "created": t0,
    }
    _remember(job)
    return job


@router.get("/local/jobs/{job_id}")
async def local_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return {k: v for k, v in job.items() if k != "plan"}


class RateBody(BaseModel):
    job_id: str
    good: bool
    recipe_id: Optional[str] = None


class SaveBody(BaseModel):
    job_id: str
    prompt: Optional[str] = None


@router.post("/teach/save")
async def teach_save(body: SaveBody):
    """Save a finished build as a taught recipe. It is used for matching from now on."""
    job = JOBS.get(body.job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "No finished build with that id.")
    prompt = (body.prompt or job.get("prompt") or "").strip()
    try:
        rec = teach.save_taught(prompt, job["plan"], source_job=body.job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"saved": True, "recipe_id": rec["id"], "keywords": rec["keywords"]}


@router.post("/teach/rate")
async def teach_rate(body: RateBody):
    """Good or bad result. If the build came from a taught recipe, that recipe's rating changes."""
    job = JOBS.get(body.job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    recipe = job.get("recipe") or ""
    rid = body.recipe_id or (recipe.split(":", 1)[1] if recipe.startswith("taught:") else None)
    if rid:
        rec = teach.rate(rid, body.good)
        if rec is None:
            raise HTTPException(404, "Taught recipe not found.")
        return {"rated": True, "recipe_id": rid, "rating": rec["rating"]}
    return {"rated": False, "note": "Built-in recipes are not rated. Save a good build to teach it."}


@router.get("/teach/list")
async def teach_list():
    return {"recipes": teach.list_taught()}


@router.delete("/teach/{recipe_id}")
async def teach_delete(recipe_id: str):
    if not teach.delete_taught(recipe_id):
        raise HTTPException(404, "Not found.")
    return {"deleted": True}


@router.get("/local/renders/{job_id}/{name}")
async def local_render(job_id: str, name: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or not re.fullmatch(r"step_\d{2}\.png", name):
        raise HTTPException(400, "Bad path.")
    path = RENDER_DIR / job_id / name
    if not path.exists():
        raise HTTPException(404, "Render not found.")
    return FileResponse(path, media_type="image/png")


@router.get("/local/builder", response_class=HTMLResponse)
async def local_builder_page():
    page = Path(__file__).with_name("local_builder.html")
    if not page.exists():
        raise HTTPException(404, "local_builder.html not found next to local_routes.py.")
    return HTMLResponse(page.read_text(encoding="utf-8"))
