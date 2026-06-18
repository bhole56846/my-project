# ccatai Web — AI Short Video Generator

A fully local web application that turns long videos into viral short-form clips.
No cloud storage. No subscriptions. Runs entirely on your machine.

---

## Features

| Feature | Details |
|---|---|
| Subtitle auto-add | Whisper speech-to-text burned into video |
| Emoji captions | Groq assigns emojis per segment |
| TikTok animated captions | Word-by-word karaoke-style highlight |
| Viral hook detection | Groq finds strongest 3-4s opening |
| Auto zoom effect | Punch-in zoom on punchlines/reactions |
| Multiple shorts | Individual MP4 per highlight or merged |
| Nepali/Hindi/English | Whisper multilingual + Groq multilingual |
| Groq analysis dashboard | JSON + live UI with virality score |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

> **Note:** `faster-whisper` will download a speech model (~250MB for `small`) on first run.

### 2. Set your Groq API key

Get a free key at https://console.groq.com

```bash
# Linux / Mac
export GROQ_API_KEY="gsk_your_key_here"

# Windows (PowerShell)
$env:GROQ_API_KEY = "gsk_your_key_here"
```

You can also enter it directly in the web UI — it's saved in your browser's localStorage.

### 3. Run the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

1. **Drop a video** onto the upload zone (MP4, MOV, AVI, MKV, WEBM supported)
2. **Enter your Groq API key** (stored locally in browser, never sent anywhere except Groq)
3. **Configure settings:**
   - Number of shorts (2–8)
   - Language (auto-detect or force Nepali/Hindi/English/etc.)
   - Transcription quality (tiny=fast, medium=accurate)
   - Toggles: captions, zoom, hook card, separate files
4. **Click Generate Shorts**
5. Watch the live pipeline progress — transcription → AI analysis → rendering
6. **Download** individual clips or all at once

---

## File structure

```
ccatai_web/
├── app.py              Flask backend, routes, SSE progress streaming
├── pipeline.py         Full AI processing pipeline (all 8 features)
├── requirements.txt
├── templates/
│   └── index.html      Single-page app UI
├── static/
│   ├── css/app.css     Dark creator-tool design system
│   └── js/app.js       Upload, SSE, results rendering, history
├── uploads/            Incoming videos (auto-cleaned)
├── outputs/            Rendered shorts + thumbnails + dashboard.json
└── logs/               app.log
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `PORT` | `5000` | Server port |
| `SECRET_KEY` | dev key | Flask secret key (change in prod) |

---

## Performance tips

- Use `whisper_size=tiny` for videos under 5 minutes — fast and accurate enough
- Use `whisper_size=small` (default) for best speed/accuracy balance
- Use `whisper_size=medium` for long videos with heavy accents or multilingual content
- Rendering speed depends on video resolution — 1080p takes ~2–3× real-time on CPU
- GPU acceleration: set `device="cuda"` in `pipeline.py → _transcribe()` if you have CUDA

---

## Production deployment (optional)

For a more robust setup with gunicorn:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 3600 app:app
```

> Use `-w 1` (single worker) because video processing is CPU-bound and multi-worker would compete for resources. Use a task queue (Celery + Redis) for multi-user production deployments.

---

## Privacy

- Videos are stored locally in `uploads/` and `outputs/` only
- Only the **transcript text** is sent to Groq's API — never the video
- Your API key is stored in your browser's localStorage, never on the server
