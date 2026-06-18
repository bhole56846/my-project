"""
ccatai Web — Flask backend
==========================
Routes:
  GET  /                    → main UI
  POST /api/upload          → receive video, start job, return job_id
  GET  /api/progress/<id>   → SSE stream of progress events
  GET  /api/job/<id>        → job status + dashboard JSON
  GET  /api/download/<id>/<filename>  → serve rendered file
  GET  /api/jobs            → list recent jobs
  POST /api/cancel/<id>     → cancel running job
  GET  /health              → health check
"""

import os
import uuid
import json
import time
import queue
import threading
import logging
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, Response, stream_with_context
)
from werkzeug.utils import secure_filename

from pipeline import VideoProcessor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
LOG_DIR     = BASE_DIR / "logs"

for d in (UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("ccatai")

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024,  # 2 GB upload limit
    SECRET_KEY         = os.environ.get("SECRET_KEY", "ccatai-dev-key-change-in-prod"),
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis/DB for multi-worker prod)
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}          # job_id → job dict
progress_queues: dict[str, queue.Queue] = {}   # job_id → SSE queue


def new_job(job_id: str, video_path: str, options: dict) -> dict:
    return {
        "id":          job_id,
        "status":      "queued",       # queued | processing | done | error | cancelled
        "created_at":  datetime.utcnow().isoformat(),
        "updated_at":  datetime.utcnow().isoformat(),
        "video_path":  str(video_path),
        "options":     options,
        "progress":    0,
        "stage":       "Queued",
        "outputs":     [],             # list of output filenames
        "dashboard":   None,
        "error":       None,
    }


def update_job(job_id: str, **kwargs):
    if job_id not in jobs:
        return
    jobs[job_id].update(kwargs)
    jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()


def emit_progress(job_id: str, percent: int, stage: str, detail: str = ""):
    """Push an SSE event into the job's queue."""
    update_job(job_id, progress=percent, stage=stage)
    event = json.dumps({"percent": percent, "stage": stage, "detail": detail,
                         "status": jobs[job_id]["status"]})
    if job_id in progress_queues:
        try:
            progress_queues[job_id].put_nowait(event)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def run_job(job_id: str):
    job = jobs[job_id]
    options = job["options"]
    video_path = job["video_path"]
    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress_cb(percent: int, stage: str, detail: str = ""):
        if jobs.get(job_id, {}).get("status") == "cancelled":
            raise InterruptedError("Job cancelled by user")
        emit_progress(job_id, percent, stage, detail)

    try:
        update_job(job_id, status="processing")
        emit_progress(job_id, 0, "Starting", "Initialising pipeline…")

        processor = VideoProcessor(
            video_path   = video_path,
            output_dir   = str(out_dir),
            options      = options,
            progress_cb  = progress_cb,
        )
        result = processor.run()

        update_job(
            job_id,
            status    = "done",
            progress  = 100,
            stage     = "Complete",
            outputs   = result["outputs"],
            dashboard = result["dashboard"],
        )
        emit_progress(job_id, 100, "Complete", "All shorts ready for download")

    except InterruptedError:
        update_job(job_id, status="cancelled", stage="Cancelled")
        emit_progress(job_id, jobs[job_id]["progress"], "Cancelled", "Job was cancelled")
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        update_job(job_id, status="error", stage="Error", error=str(exc))
        emit_progress(job_id, jobs[job_id]["progress"], "Error", str(exc))
    finally:
        # Signal SSE stream to close after a short delay
        time.sleep(1)
        if job_id in progress_queues:
            progress_queues[job_id].put_nowait("__DONE__")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "jobs": len(jobs)})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id   = str(uuid.uuid4())
    safe_fn  = f"{job_id}{ext}"
    save_path = UPLOAD_DIR / safe_fn
    file.save(str(save_path))

    options = {
        "num_highlights":  int(request.form.get("num_highlights", 4)),
        "language":        request.form.get("language", "auto"),
        "multiple_shorts": request.form.get("multiple_shorts", "false").lower() == "true",
        "add_zoom":        request.form.get("add_zoom", "true").lower() == "true",
        "add_captions":    request.form.get("add_captions", "true").lower() == "true",
        "add_hook_card":   request.form.get("add_hook_card", "true").lower() == "true",
        "music_dir":       request.form.get("music_dir", ""),
        "groq_api_key":    request.form.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")),
        "whisper_size":    request.form.get("whisper_size", "small"),
    }

    if not options["groq_api_key"]:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Groq API key is required"}), 400

    jobs[job_id] = new_job(job_id, str(save_path), options)
    progress_queues[job_id] = queue.Queue(maxsize=200)

    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()

    log.info("Job %s started — %s, options=%s", job_id, file.filename, options)
    return jsonify({"job_id": job_id}), 202


@app.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = progress_queues.get(job_id)
        if q is None:
            yield "data: {\"error\": \"No queue\"}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                # heartbeat
                yield "data: {\"heartbeat\": true}\n\n"
                continue
            if msg == "__DONE__":
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {msg}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/job/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({k: v for k, v in job.items() if k != "video_path"})


@app.route("/api/jobs")
def list_jobs():
    result = []
    for j in sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)[:20]:
        result.append({
            "id":         j["id"],
            "status":     j["status"],
            "created_at": j["created_at"],
            "progress":   j["progress"],
            "stage":      j["stage"],
            "num_outputs": len(j["outputs"]),
        })
    return jsonify(result)


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] not in ("queued", "processing"):
        return jsonify({"error": f"Cannot cancel job in state '{job['status']}'"}), 400
    update_job(job_id, status="cancelled")
    return jsonify({"ok": True})


@app.route("/api/download/<job_id>/<filename>")
def download_file(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    safe_name = secure_filename(filename)
    out_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(out_dir), safe_name, as_attachment=True)


@app.route("/api/thumbnail/<job_id>/<filename>")
def thumbnail(job_id: str, filename: str):
    """Serve pre-generated thumbnail PNGs."""
    safe_name = secure_filename(filename)
    out_dir = OUTPUT_DIR / job_id
    return send_from_directory(str(out_dir), safe_name)


# ---------------------------------------------------------------------------
# Dev server entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("ccatai starting on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
