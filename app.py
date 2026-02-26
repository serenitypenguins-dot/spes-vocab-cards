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

from image_generator import generate_cards, generate_image, parse_word_input
from pdf_builder import build_pdf

load_dotenv()

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


async def cleanup_old_jobs():
    while True:
        await asyncio.sleep(300)
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

    code = data.get("code", "").strip()
    if code != ACCESS_CODE:
        return {"error": "Invalid access code"}

    words = [w.strip() for w in data.get("words", []) if w.strip()]
    grade = data.get("grade", "k")
    style = data.get("style", "cartoon")
    multi_meanings = data.get("multiMeanings", False)

    if not words:
        return {"error": "No words provided"}
    if len(words) > 200:
        return {"error": "Maximum 200 words allowed"}

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    estimated_total = len(words) * (3 if multi_meanings else 1)

    jobs[job_id] = {
        "status": "generating",
        "progress": 0,
        "total": estimated_total,
        "current_word": "",
        "created_at": datetime.now(),
        "grade": grade,
        "style": style,
        "cards": [],  # list of {word, meaning, image_filename, accepted}
        "job_dir": str(job_dir),
        "error": None,
    }

    asyncio.create_task(_run_job(job_id, words, grade, style, multi_meanings, job_dir))
    return {"job_id": job_id}


async def _run_job(job_id: str, words: list[str], grade: str, style: str,
                   multi_meanings: bool, job_dir: Path):
    try:
        async def on_progress(done: int, total: int, word: str):
            jobs[job_id]["progress"] = done
            jobs[job_id]["total"] = total
            jobs[job_id]["current_word"] = word

        cards = await generate_cards(words, grade, style, job_dir, multi_meanings, on_progress)

        # Store cards as serializable data
        card_data = []
        for word, meaning, img_path in cards:
            card_data.append({
                "word": word,
                "meaning": meaning,
                "image_filename": img_path.name if img_path else None,
                "accepted": True,  # default: all accepted
            })

        jobs[job_id]["cards"] = card_data
        jobs[job_id]["status"] = "preview"  # NEW: goes to preview instead of done
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
            if info["status"] == "preview":
                payload["cards"] = info["cards"]
            yield f"data: {json.dumps(payload)}\n\n"

            if info["status"] in ("preview", "done", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/image/{job_id}/{filename}")
async def serve_image(job_id: str, filename: str):
    """Serve generated images for preview."""
    info = jobs.get(job_id)
    if not info:
        return {"error": "Job not found"}
    job_dir = Path(info["job_dir"])
    img_path = job_dir / filename
    if not img_path.exists():
        return {"error": "Image not found"}
    return FileResponse(img_path, media_type="image/png")


@app.post("/regenerate/{job_id}/{card_index}")
async def regenerate_card(job_id: str, card_index: int, request: Request):
    """Regenerate a single card's image with an optional custom prompt."""
    info = jobs.get(job_id)
    if not info or info["status"] != "preview":
        return {"error": "Job not in preview state"}

    data = await request.json()
    custom_prompt = data.get("prompt", "").strip()

    cards = info["cards"]
    if card_index < 0 or card_index >= len(cards):
        return {"error": "Invalid card index"}

    card = cards[card_index]
    job_dir = Path(info["job_dir"])
    grade = info["grade"]
    style = info["style"]

    # Use custom prompt as context if provided, otherwise use existing meaning
    context = custom_prompt if custom_prompt else card.get("meaning")

    try:
        new_path = await generate_image(card["word"], grade, style, job_dir, context)
        if new_path:
            card["image_filename"] = new_path.name
            if custom_prompt:
                card["meaning"] = custom_prompt
            return {"ok": True, "image_filename": new_path.name, "meaning": card["meaning"]}
        return {"error": "Failed to generate image"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/finalize/{job_id}")
async def finalize(job_id: str, request: Request):
    """Build PDF from accepted cards only."""
    info = jobs.get(job_id)
    if not info or info["status"] != "preview":
        return {"error": "Job not in preview state"}

    data = await request.json()
    accepted_indices = data.get("accepted", [])  # list of card indices to include

    job_dir = Path(info["job_dir"])
    cards_data = info["cards"]

    # Build card tuples for accepted cards only
    pdf_cards = []
    for i in accepted_indices:
        if 0 <= i < len(cards_data):
            c = cards_data[i]
            img_path = job_dir / c["image_filename"] if c["image_filename"] else None
            pdf_cards.append((c["word"], c["meaning"], img_path))

    if not pdf_cards:
        return {"error": "No cards selected"}

    pdf_path = job_dir / "vocab_cards.pdf"
    build_pdf(pdf_cards, info["grade"], pdf_path)

    info["status"] = "done"
    info["pdf_path"] = str(pdf_path)
    return {"ok": True}


@app.get("/download/{job_id}")
async def download(job_id: str):
    info = jobs.get(job_id)
    if not info or info["status"] != "done":
        return {"error": "PDF not ready"}
    pdf_path = Path(info["pdf_path"])
    if not pdf_path.exists():
        return {"error": "PDF file not found"}
    return FileResponse(pdf_path, filename="vocab_cards.pdf", media_type="application/pdf")
