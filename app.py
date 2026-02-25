import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from image_generator import generate_images_batch
from pdf_builder import build_pdf

load_dotenv()

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

# In-memory job state
jobs: dict[str, dict] = {}


async def cleanup_old_jobs():
    """Remove job directories older than 1 hour."""
    while True:
        await asyncio.sleep(300)  # check every 5 min
        cutoff = datetime.now() - timedelta(hours=1)
        to_remove = []
        for job_id, info in jobs.items():
            if info.get("created_at") and info["created_at"] < cutoff:
                job_dir = JOBS_DIR / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                to_remove.append(job_id)
        for jid in to_remove:
            jobs.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_old_jobs())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


ACCESS_CODE = os.getenv("ACCESS_CODE", "edda2026")


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()

    # Validate access code
    code = data.get("code", "").strip()
    if code != ACCESS_CODE:
        return {"error": "Invalid access code"}

    words = [w.strip() for w in data.get("words", []) if w.strip()]
    grade = data.get("grade", "k")
    style = data.get("style", "cartoon")

    if not words:
        return {"error": "No words provided"}
    if len(words) > 200:
        return {"error": "Maximum 200 words allowed"}

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    jobs[job_id] = {
        "status": "generating",
        "progress": 0,
        "total": len(words),
        "current_word": "",
        "created_at": datetime.now(),
        "words": words,
        "grade": grade,
        "style": style,
        "error": None,
    }

    asyncio.create_task(_run_job(job_id, words, grade, style, job_dir))
    return {"job_id": job_id}


async def _run_job(job_id: str, words: list[str], grade: str, style: str, job_dir: Path):
    try:
        async def on_progress(done: int, total: int, word: str):
            jobs[job_id]["progress"] = done
            jobs[job_id]["current_word"] = word

        image_paths = await generate_images_batch(words, grade, style, job_dir, on_progress)

        pdf_path = job_dir / "vocab_cards.pdf"
        build_pdf(words, image_paths, grade, pdf_path)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["pdf_path"] = str(pdf_path)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/status/{job_id}")
async def status_sse(job_id: str):
    async def event_stream():
        if job_id not in jobs:
            yield f"data: {json.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
            return

        while True:
            info = jobs.get(job_id)
            if not info:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
                return

            payload = {
                "status": info["status"],
                "progress": info["progress"],
                "total": info["total"],
                "current_word": info.get("current_word", ""),
            }
            if info["status"] == "error":
                payload["error"] = info.get("error", "Unknown error")
            yield f"data: {json.dumps(payload)}\n\n"

            if info["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/download/{job_id}")
async def download(job_id: str):
    info = jobs.get(job_id)
    if not info or info["status"] != "done":
        return {"error": "PDF not ready"}
    pdf_path = Path(info["pdf_path"])
    if not pdf_path.exists():
        return {"error": "PDF file not found"}
    return FileResponse(pdf_path, filename="vocab_cards.pdf", media_type="application/pdf")
