"""
pipeline.py — Full AI video processing pipeline
================================================
Wraps every feature from auto_editor.py into a class that:
  • accepts a progress_cb(percent, stage, detail) callback
  • generates thumbnails for each output clip
  • saves a rich dashboard JSON
  • returns {"outputs": [...filenames], "dashboard": {...}}
"""

import os
import re
import json
import time
import random
import textwrap
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("ccatai.pipeline")

# ---------------------------------------------------------------------------
# Lazy imports (heavy deps — only loaded when a job actually runs)
# ---------------------------------------------------------------------------

def _import_moviepy():
    from moviepy.editor import (
        VideoFileClip, ImageClip, concatenate_videoclips,
        AudioFileClip, CompositeVideoClip, vfx, afx,
    )
    from moviepy.audio.AudioClip import CompositeAudioClip
    return (VideoFileClip, ImageClip, concatenate_videoclips,
            AudioFileClip, CompositeVideoClip, vfx, afx, CompositeAudioClip)


# ---------------------------------------------------------------------------
# Constants / style
# ---------------------------------------------------------------------------
TARGET_RATIO        = 9 / 16
MAX_CLIP_DURATION   = 12
TRANSITION_DURATION = 0.5
MUSIC_VOLUME        = 0.25
ZOOM_SCALE          = 1.12
ZOOM_DURATION_FRAC  = 0.35
CAPTION_FONT_SIZE   = 52
CAPTION_COLOR       = (255, 255, 255)
CAPTION_HIGHLIGHT   = (255, 220, 0)
CAPTION_BG          = (0, 0, 0, 160)
GROQ_MODEL          = "llama-3.3-70b-versatile"

LANG_NAMES = {"ne": "Nepali", "hi": "Hindi", "en": "English", "auto": "Auto-detect"}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ===========================================================================
class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, options: dict, progress_cb=None):
        self.video_path  = video_path
        self.output_dir  = Path(output_dir)
        self.options     = options
        self.cb          = progress_cb or (lambda p, s, d="": None)

        self.api_key        = options.get("groq_api_key", os.environ.get("GROQ_API_KEY", ""))
        self.num_highlights = int(options.get("num_highlights", 4))
        self.language       = options.get("language", "auto")
        self.multiple       = options.get("multiple_shorts", False)
        self.add_zoom       = options.get("add_zoom", True)
        self.add_captions   = options.get("add_captions", True)
        self.add_hook_card  = options.get("add_hook_card", True)
        self.music_dir      = options.get("music_dir", "")
        self.whisper_size   = options.get("whisper_size", "small")

    # -----------------------------------------------------------------------
    def run(self) -> dict:
        outputs    = []
        dashboard  = {}

        # 1. Transcribe
        self.cb(5, "Transcribing", f"Loading Whisper ({self.whisper_size})…")
        transcript, detected_lang = self._transcribe()
        self.cb(25, "Transcribed", f"Detected language: {detected_lang}")

        # 2. Groq analysis
        self.cb(30, "AI Analysis", "Sending to Groq LLM…")
        probe       = self._probe_duration()
        lang_hint   = LANG_NAMES.get(detected_lang, detected_lang)
        highlights, hook, dashboard = self._groq_analysis(transcript, probe, lang_hint)
        self.cb(45, "Analysis Complete", f"Virality score: {dashboard.get('overall_virality_score','?')}/10")

        # 3. Music
        self.cb(48, "Music", "Selecting background track…")
        music_path = self._pick_music()

        # 4. Render
        (VideoFileClip, ImageClip, concatenate_videoclips,
         AudioFileClip, CompositeVideoClip, vfx, afx, CompositeAudioClip) = _import_moviepy()
        self._moviepy = (VideoFileClip, ImageClip, concatenate_videoclips,
                         AudioFileClip, CompositeVideoClip, vfx, afx, CompositeAudioClip)

        source = VideoFileClip(self.video_path)
        w, h   = source.size

        if self.multiple:
            total = len(highlights)
            for i, hl in enumerate(highlights):
                pct_start = 50 + int(i / total * 45)
                pct_end   = 50 + int((i + 1) / total * 45)
                self.cb(pct_start, f"Rendering short {i+1}/{total}",
                        f"{hl['start']:.1f}s–{hl['end']:.1f}s  {hl.get('emoji','')}")
                clip = self._assemble_clip(source, hl, transcript, w, h)
                if clip is None:
                    continue
                clip = self._attach_music(clip, music_path, AudioFileClip, afx, CompositeAudioClip)
                fname = f"short_{i+1:02d}.mp4"
                out   = str(self.output_dir / fname)
                clip.write_videofile(out, codec="libx264", audio_codec="aac", fps=30,
                                     logger=None)
                outputs.append(fname)
                self._make_thumbnail(out, fname)
                self.cb(pct_end, f"Short {i+1} done", fname)
        else:
            self.cb(50, "Rendering", "Assembling clips…")
            clips = []

            # Hook card
            if self.add_hook_card and hook.get("text"):
                target_w = min(int(h * TARGET_RATIO), w)
                card = self._build_hook_card(target_w, h, hook["text"])
                clips.append(card)

            for i, hl in enumerate(highlights):
                pct = 50 + int(i / len(highlights) * 40)
                self.cb(pct, "Rendering", f"Clip {i+1}/{len(highlights)}  {hl.get('emoji','')}")
                clip = self._assemble_clip(source, hl, transcript, w, h)
                if clip:
                    clips.append(clip)

            if not clips:
                raise RuntimeError("No valid clips were generated from the highlights.")

            self.cb(90, "Merging", "Concatenating and encoding…")
            final = concatenate_videoclips(clips, method="compose",
                                           padding=-TRANSITION_DURATION)
            final = self._attach_music(final, music_path, AudioFileClip, afx, CompositeAudioClip)
            fname = "final_short.mp4"
            out   = str(self.output_dir / fname)
            final.write_videofile(out, codec="libx264", audio_codec="aac", fps=30, logger=None)
            outputs.append(fname)
            self._make_thumbnail(out, fname)

        source.close()

        # 5. Save dashboard
        dashboard["outputs"]     = outputs
        dashboard["output_dir"]  = str(self.output_dir)
        dash_path = str(self.output_dir / "dashboard.json")
        with open(dash_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)
        outputs.append("dashboard.json")

        self.cb(99, "Finalising", "Saving dashboard…")
        return {"outputs": outputs, "dashboard": dashboard}

    # -----------------------------------------------------------------------
    # TRANSCRIPTION
    # -----------------------------------------------------------------------
    def _transcribe(self):
        from faster_whisper import WhisperModel
        model = WhisperModel(self.whisper_size, device="cpu", compute_type="int8")
        kwargs = {"word_timestamps": True}
        if self.language != "auto":
            kwargs["language"] = self.language
        segments, info = model.transcribe(self.video_path, **kwargs)
        detected = getattr(info, "language", "en")
        transcript = []
        for seg in segments:
            words = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            transcript.append({
                "start": seg.start, "end": seg.end,
                "text": seg.text.strip(), "words": words,
            })
        log.info("Transcribed %d segments, lang=%s", len(transcript), detected)
        return transcript, detected

    # -----------------------------------------------------------------------
    # GROQ ANALYSIS
    # -----------------------------------------------------------------------
    def _probe_duration(self):
        from moviepy.editor import VideoFileClip as VFC
        v = VFC(self.video_path)
        d = v.duration
        v.close()
        return d

    def _groq_analysis(self, transcript, duration, lang_hint):
        from groq import Groq
        client = Groq(api_key=self.api_key)
        tx = "\n".join(f"[{t['start']:.1f}-{t['end']:.1f}] {t['text']}" for t in transcript)
        prompt = f"""You are an expert viral short-form video editor working with {lang_hint} content.
Timestamped transcript of a {duration:.0f}s video:

{tx}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "highlights": [
    {{"start": 12.5, "end": 22.0, "reason": "funny punchline", "emoji": "😂", "zoom": true}}
  ],
  "hook": {{"start": 0.0, "end": 4.0, "text": "opening hook line", "reason": "why it hooks"}},
  "language_detected": "{lang_hint}",
  "overall_virality_score": 7,
  "content_summary": "one-sentence summary",
  "recommended_caption_style": "energetic",
  "segment_analysis": [
    {{"start": 0, "end": 10, "energy": "low", "topic": "intro"}}
  ]
}}

Rules:
- Exactly {self.num_highlights} highlights, each max {MAX_CLIP_DURATION}s.
- emoji: 1–2 emojis per highlight capturing mood.
- zoom: true for punchlines, reactions, high-energy beats.
- overall_virality_score: integer 1–10.
- Preserve {lang_hint} text exactly in hook.text.
"""
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Groq JSON parse failed, using fallback")
            data = self._fallback_analysis(duration)

        highlights = data.get("highlights", [])
        hook       = data.get("hook", {"start": 0, "end": min(4, duration), "text": "", "reason": ""})
        dashboard  = {
            "model":                     GROQ_MODEL,
            "video_duration_s":          duration,
            "language":                  data.get("language_detected", lang_hint),
            "overall_virality_score":    data.get("overall_virality_score", "N/A"),
            "content_summary":           data.get("content_summary", ""),
            "recommended_caption_style": data.get("recommended_caption_style", "default"),
            "segment_analysis":          data.get("segment_analysis", []),
            "hook":                      hook,
            "highlights":                highlights,
            "num_highlights":            len(highlights),
            "options":                   self.options,
        }
        return highlights, hook, dashboard

    def _fallback_analysis(self, duration):
        step = duration / self.num_highlights
        return {
            "highlights": [
                {"start": i * step, "end": min(i * step + MAX_CLIP_DURATION, duration),
                 "reason": "fallback", "emoji": "🎬", "zoom": False}
                for i in range(self.num_highlights)
            ],
            "hook": {"start": 0, "end": min(4, duration), "text": "", "reason": "fallback"},
            "overall_virality_score": 5, "content_summary": "N/A",
            "recommended_caption_style": "default", "segment_analysis": [],
        }

    # -----------------------------------------------------------------------
    # CLIP ASSEMBLY
    # -----------------------------------------------------------------------
    def _assemble_clip(self, source, hl, transcript, src_w, src_h):
        (VideoFileClip, ImageClip, concatenate_videoclips,
         AudioFileClip, CompositeVideoClip, vfx, afx, CompositeAudioClip) = self._moviepy

        start = float(hl["start"])
        end   = min(float(hl["end"]), start + MAX_CLIP_DURATION, source.duration)
        if end <= start:
            return None

        sub = source.subclip(start, end)
        sub = self._smart_crop(sub)

        if self.add_zoom and hl.get("zoom"):
            sub = self._auto_zoom(sub)

        if self.add_captions:
            words = self._get_words(transcript, start, end)
            if words:
                cap = self._caption_overlay(sub, words, hl.get("emoji", ""),
                                            CompositeVideoClip)
                sub = cap

        sub = sub.fx(vfx.fadein, TRANSITION_DURATION).fx(vfx.fadeout, TRANSITION_DURATION)
        return sub

    def _smart_crop(self, clip):
        w, h = clip.size
        target_w = int(h * TARGET_RATIO)
        if target_w >= w:
            return clip
        frame  = clip.get_frame(clip.duration / 2)
        gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces  = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            best = faces[np.argmax([f[2] * f[3] for f in faces])]
            cx   = best[0] + best[2] / 2
        else:
            cx = w / 2
        x1 = int(max(0, min(cx - target_w / 2, w - target_w)))
        return clip.crop(x1=x1, y1=0, x2=x1 + target_w, y2=h)

    def _auto_zoom(self, clip):
        w, h = clip.size
        zoom_end = clip.duration * ZOOM_DURATION_FRAC

        def zoom_filter(get_frame, t):
            frame    = get_frame(t)
            progress = min(t / zoom_end, 1.0) if zoom_end > 0 else 1.0
            scale    = 1.0 + (ZOOM_SCALE - 1.0) * progress
            nw, nh   = int(w * scale), int(h * scale)
            resized  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x1, y1   = (nw - w) // 2, (nh - h) // 2
            return resized[y1:y1 + h, x1:x1 + w]

        return clip.fl(zoom_filter, apply_to=["mask"])

    def _get_words(self, transcript, start, end):
        words = []
        for seg in transcript:
            if seg["end"] < start or seg["start"] > end:
                continue
            for w in seg.get("words", []):
                if w["start"] >= start and w["end"] <= end:
                    words.append({"word": w["word"],
                                  "start": w["start"] - start,
                                  "end":   w["end"]   - start})
        if not words:
            for seg in transcript:
                if seg["end"] < start or seg["start"] > end:
                    continue
                sw = seg["text"].split()
                if not sw:
                    continue
                dur = (min(seg["end"], end) - max(seg["start"], start)) / max(len(sw), 1)
                base = max(seg["start"], start) - start
                for i, word in enumerate(sw):
                    words.append({"word": word, "start": base + i * dur,
                                  "end": base + (i + 1) * dur})
        return words

    def _caption_overlay(self, clip, words, emoji, CompositeVideoClip):
        w, h = clip.size
        fps  = 15
        times = np.arange(0, clip.duration, 1 / fps)
        frames = [self._render_caption_frame(w, h, words, t, emoji) for t in times]

        def make_frame(t):
            idx = min(int(t * fps), len(frames) - 1)
            return frames[idx][:, :, :3]

        def make_mask(t):
            idx = min(int(t * fps), len(frames) - 1)
            return frames[idx][:, :, 3] / 255.0

        from moviepy.editor import ImageClip as IC
        cap = (IC(frames[0][:, :, :3], duration=clip.duration)
               .fl(lambda gf, t: make_frame(t))
               .set_duration(clip.duration))
        mask = (IC(frames[0][:, :, 3] / 255.0, ismask=True, duration=clip.duration)
                .fl(lambda gf, t: make_mask(t)))
        cap = cap.set_mask(mask)
        return CompositeVideoClip([clip, cap])

    def _render_caption_frame(self, width, height, words, current_time, emoji=""):
        img  = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = self._get_font(CAPTION_FONT_SIZE)

        active = -1
        for i, w in enumerate(words):
            if w["start"] <= current_time <= w["end"]:
                active = i
                break
        if active == -1:
            for i, w in enumerate(words):
                if current_time > w["end"]:
                    active = i

        lines = textwrap.wrap(" ".join(w["word"] for w in words), width=22)
        line_h   = CAPTION_FONT_SIZE + 12
        total_h  = len(lines) * line_h + 20
        bg_y     = height - total_h - 80
        draw.rectangle([16, bg_y, width - 16, bg_y + total_h], fill=CAPTION_BG)

        word_idx = 0
        for li, line in enumerate(lines):
            lwords = line.split()
            lw_total = sum(draw.textlength(lw + " ", font=font) for lw in lwords)
            x = (width - lw_total) / 2
            y = bg_y + 10 + li * line_h
            for lw in lwords:
                color = CAPTION_HIGHLIGHT if word_idx == active else CAPTION_COLOR
                draw.text((x, y), lw, font=font, fill=color)
                x += draw.textlength(lw + " ", font=font)
                word_idx += 1

        if emoji:
            try:
                efont = self._get_font(56)
                draw.text((width - 80, 18), emoji, font=efont, fill=(255, 255, 255, 220))
            except Exception:
                pass

        return np.array(img)

    @staticmethod
    def _get_font(size):
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    # -----------------------------------------------------------------------
    # HOOK CARD
    # -----------------------------------------------------------------------
    def _build_hook_card(self, width, height, text, duration=2.5):
        from moviepy.editor import ImageClip, vfx
        img  = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = self._get_font(CAPTION_FONT_SIZE + 12)
        lines  = textwrap.wrap(text, width=20)
        line_h = CAPTION_FONT_SIZE + 22
        y = (height - len(lines) * line_h) / 2
        for line in lines:
            lw = draw.textlength(line, font=font)
            draw.text(((width - lw) / 2, y), line, font=font, fill=(255, 220, 0))
            y += line_h
        arr  = np.array(img)
        card = ImageClip(arr, duration=duration)
        return card.fx(vfx.fadein, 0.3).fx(vfx.fadeout, 0.3)

    # -----------------------------------------------------------------------
    # MUSIC
    # -----------------------------------------------------------------------
    def _pick_music(self):
        if not self.music_dir or not os.path.isdir(self.music_dir):
            return None
        tracks = [f for f in os.listdir(self.music_dir)
                  if f.lower().endswith((".mp3", ".wav", ".m4a"))]
        if not tracks:
            return None
        return os.path.join(self.music_dir, random.choice(tracks))

    def _attach_music(self, clip, music_path, AudioFileClip, afx, CompositeAudioClip):
        if not music_path:
            return clip
        bg = AudioFileClip(music_path)
        bg = bg.fx(afx.audio_loop, duration=clip.duration)
        bg = bg.fx(afx.volumex, MUSIC_VOLUME)
        combined = CompositeAudioClip([clip.audio, bg]) if clip.audio else bg
        return clip.set_audio(combined)

    # -----------------------------------------------------------------------
    # THUMBNAIL
    # -----------------------------------------------------------------------
    def _make_thumbnail(self, video_path: str, video_fname: str):
        try:
            cap  = cv2.VideoCapture(video_path)
            fps  = cap.get(cv2.CAP_PROP_FPS) or 30
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(fps * 1.5, total / 2))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            img.thumbnail((400, 711))   # 9:16 thumbnail
            thumb_name = Path(video_fname).stem + "_thumb.jpg"
            img.save(str(self.output_dir / thumb_name), "JPEG", quality=82)
        except Exception as e:
            log.warning("Thumbnail failed for %s: %s", video_fname, e)
