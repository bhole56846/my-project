"""
AI Auto Video Editor (CapCut-style) — Groq-powered  ✦ ENHANCED EDITION ✦
=========================================================================

NEW FEATURES IN THIS VERSION:
  ✅ Subtitle auto-add          — burned-in subtitles from Whisper transcription
  ✅ Emoji captions             — Groq assigns relevant emojis per segment
  ✅ TikTok-style animated captions — word-by-word highlight karaoke style
  ✅ Viral hook detection       — Groq finds the strongest opening hook
  ✅ Auto zoom effect           — slow punch-in zoom on key moments
  ✅ Multiple shorts generation — batch-render N individual short clips
  ✅ Nepali/Hindi/English support — Whisper multi-language + Groq multilingual
  ✅ Groq analysis dashboard    — saves a rich JSON report of every decision

ORIGINAL PIPELINE:
  1. Transcribe source video (faster-whisper, local/free)
  2. Groq LLM picks viral highlight segments + hook + emojis
  3. Cut segments from source video
  4. Smart vertical (9:16) crop with face detection
  5. Animated TikTok-style captions (word-by-word)
  6. Auto zoom on key moments
  7. Fade transitions + background music
  8. Render final short(s) + save Groq analysis dashboard JSON

------------------------------------------------------------------
INSTALL
------------------------------------------------------------------
    pip install opencv-python moviepy faster-whisper groq numpy pillow --break-system-packages

    # For burned-in subtitle font rendering (PIL-based, no ffmpeg subtitles needed)
    # Pillow is the only extra dep over the original.

------------------------------------------------------------------
SETUP
------------------------------------------------------------------
    export GROQ_API_KEY="your_key_here"      # Linux/Mac
    setx GROQ_API_KEY "your_key_here"        # Windows PowerShell

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
    # Single merged short (original behaviour + all new features):
    python auto_editor.py --input video.mp4 --music_dir ./music --output final.mp4

    # Generate 4 separate individual shorts:
    python auto_editor.py --input video.mp4 --output_dir ./shorts --multiple

    # Force Nepali transcription:
    python auto_editor.py --input video.mp4 --lang ne

    # Hindi:
    python auto_editor.py --input video.mp4 --lang hi

    # Save Groq dashboard JSON alongside output:
    python auto_editor.py --input video.mp4 --dashboard dashboard.json
"""

import os
import re
import json
import random
import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
    vfx,
    afx,
)
from moviepy.audio.AudioClip import CompositeAudioClip
from faster_whisper import WhisperModel
from groq import Groq

# ---------------------------------------------------------------------------
# CONFIG — tweak these
# ---------------------------------------------------------------------------
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY")
GROQ_MODEL          = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE  = "small"          # tiny / base / small / medium
TARGET_RATIO        = 9 / 16           # vertical (Reels/Shorts/TikTok)
MAX_CLIP_DURATION   = 12               # seconds per highlight
NUM_HIGHLIGHTS      = 4
TRANSITION_DURATION = 0.5              # crossfade seconds
MUSIC_VOLUME        = 0.25

# Caption style
CAPTION_FONT_SIZE   = 52               # px
CAPTION_COLOR       = (255, 255, 255)  # white text
CAPTION_HIGHLIGHT   = (255, 220, 0)    # yellow highlight for current word
CAPTION_BG          = (0, 0, 0, 160)  # semi-transparent black bg (RGBA)
CAPTION_EMOJI_SIZE  = 64              # px for emoji overlay

# Zoom config
ZOOM_SCALE          = 1.12             # how much to punch in (12%)
ZOOM_DURATION_FRAC  = 0.35             # fraction of clip duration to hold zoom

# Language map for display
LANG_NAMES = {"ne": "Nepali", "hi": "Hindi", "en": "English", "auto": "Auto-detect"}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ===========================================================================
# STEP 1: Transcribe (multi-language)
# ===========================================================================

def transcribe_video(video_path, language="auto"):
    """
    Transcribe with faster-whisper.
    language: "auto" lets Whisper detect, or pass "ne"/"hi"/"en" etc.
    Returns list of segment dicts with start/end/text + word-level timestamps.
    """
    lang_display = LANG_NAMES.get(language, language)
    print(f"[1/6] Transcribing audio ({lang_display}) via faster-whisper...")

    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    kwargs = {"word_timestamps": True}
    if language != "auto":
        kwargs["language"] = language

    segments, info = model.transcribe(video_path, **kwargs)

    detected = getattr(info, "language", "unknown")
    print(f"       Detected language: {detected} (confidence: {getattr(info, 'language_probability', '?'):.0%})")

    transcript = []
    for seg in segments:
        words = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})
        transcript.append({
            "start":  seg.start,
            "end":    seg.end,
            "text":   seg.text.strip(),
            "words":  words,
        })
    return transcript, detected


# ===========================================================================
# STEP 2: Groq full analysis — highlights + hook + emojis + dashboard
# ===========================================================================

def groq_full_analysis(transcript, video_duration, num_highlights, language_hint):
    """
    Single Groq call that returns:
      - highlights[]   : [{start, end, reason, emoji, zoom}]
      - hook           : {start, end, text, reason}
      - dashboard      : full metadata dict
    """
    print("[2/6] Groq AI analysis (highlights + hook + emojis + dashboard)...")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key")

    client = Groq(api_key=GROQ_API_KEY)
    transcript_text = "\n".join(
        f"[{t['start']:.1f}-{t['end']:.1f}] {t['text']}" for t in transcript
    )

    prompt = f"""You are an expert viral short-form video editor working with {language_hint} content.
Below is a timestamped transcript of a {video_duration:.0f}-second video.

{transcript_text}

Return ONLY a single JSON object (no markdown, no explanation) with this exact shape:
{{
  "highlights": [
    {{
      "start": 12.5,
      "end": 22.0,
      "reason": "funny punchline",
      "emoji": "😂",
      "zoom": true
    }}
  ],
  "hook": {{
    "start": 0.0,
    "end": 4.0,
    "text": "the attention-grabbing opening line",
    "reason": "why this hooks viewers in the first 3 seconds"
  }},
  "language_detected": "{language_hint}",
  "overall_virality_score": 7,
  "content_summary": "brief 1-sentence summary",
  "recommended_caption_style": "energetic"
}}

Rules:
- Pick exactly {num_highlights} highlights, each max {MAX_CLIP_DURATION}s.
- emoji: pick 1-2 emojis that best represent the mood/topic of EACH highlight.
- zoom: true if this moment is a punchline, reaction, or high-energy beat.
- hook: the strongest {min(4, int(video_duration * 0.1))}-second opening moment (may overlap a highlight).
- overall_virality_score: integer 1-10.
- Respect and preserve {language_hint} text exactly in 'hook.text'.
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
        print("  ⚠ Groq JSON parse failed — using fallback.")
        data = _fallback_analysis(video_duration, num_highlights)

    # Normalise
    highlights = data.get("highlights", [])
    hook       = data.get("hook", {"start": 0, "end": min(4, video_duration), "text": "", "reason": "fallback"})

    dashboard = {
        "model":                   GROQ_MODEL,
        "video_duration_s":        video_duration,
        "language":                data.get("language_detected", language_hint),
        "overall_virality_score":  data.get("overall_virality_score", "N/A"),
        "content_summary":         data.get("content_summary", ""),
        "recommended_caption_style": data.get("recommended_caption_style", "default"),
        "hook":                    hook,
        "highlights":              highlights,
        "num_highlights":          len(highlights),
        "raw_groq_response":       raw,
    }

    print(f"       Virality score: {dashboard['overall_virality_score']}/10")
    print(f"       Hook: {hook.get('text','')[:60]}...")
    return highlights, hook, dashboard


def _fallback_analysis(duration, num_highlights):
    step = duration / num_highlights
    highlights = [
        {"start": i * step, "end": min(i * step + MAX_CLIP_DURATION, duration),
         "reason": "fallback", "emoji": "🎬", "zoom": False}
        for i in range(num_highlights)
    ]
    return {"highlights": highlights,
            "hook": {"start": 0, "end": min(4, duration), "text": "", "reason": "fallback"},
            "overall_virality_score": 5, "content_summary": "", "recommended_caption_style": "default"}


# ===========================================================================
# STEP 3: Smart vertical crop
# ===========================================================================

def smart_vertical_crop(clip):
    w, h = clip.size
    target_w = int(h * TARGET_RATIO)
    if target_w >= w:
        return clip
    frame = clip.get_frame(clip.duration / 2)
    gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    center_x = (faces[np.argmax([f[2]*f[3] for f in faces])][0] + faces[np.argmax([f[2]*f[3] for f in faces])][2] / 2) if len(faces) > 0 else w / 2
    x1 = int(max(0, min(center_x - target_w / 2, w - target_w)))
    return clip.crop(x1=x1, y1=0, x2=x1 + target_w, y2=h)


# ===========================================================================
# STEP 4: Auto zoom effect
# ===========================================================================

def apply_auto_zoom(clip):
    """
    Slow punch-in zoom over the first ZOOM_DURATION_FRAC of the clip,
    then holds. Simulates a dramatic push-in effect.
    """
    w, h = clip.size
    zoom_end_t = clip.duration * ZOOM_DURATION_FRAC

    def zoom_filter(get_frame, t):
        frame = get_frame(t)
        progress = min(t / zoom_end_t, 1.0) if zoom_end_t > 0 else 1.0
        scale = 1.0 + (ZOOM_SCALE - 1.0) * progress
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        return resized[y1:y1+h, x1:x1+w]

    return clip.fl(zoom_filter, apply_to=["mask"])


# ===========================================================================
# STEP 5: TikTok-style animated captions (word-by-word karaoke)
# ===========================================================================

def _get_font(size):
    """Try to load a bold font; fall back to PIL default."""
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


def make_caption_frame(width, height, words_in_segment, current_time, emoji=""):
    """
    Render a single caption frame as a PIL Image (RGBA).
    Words already spoken are white; the current word is yellow-highlighted.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _get_font(CAPTION_FONT_SIZE)

    # Find current active word
    active_idx = -1
    for i, w in enumerate(words_in_segment):
        if w["start"] <= current_time <= w["end"]:
            active_idx = i
            break
    if active_idx == -1:
        for i, w in enumerate(words_in_segment):
            if current_time > w["end"]:
                active_idx = i  # last spoken word

    # Build display line (wrap at ~20 chars per line)
    line_words = [w["word"] for w in words_in_segment]
    full_text = " ".join(line_words)
    lines = textwrap.wrap(full_text, width=22)

    # Background pill
    line_h = CAPTION_FONT_SIZE + 10
    total_h = len(lines) * line_h + 20
    bg_y = height - total_h - 80
    draw.rectangle([20, bg_y, width - 20, bg_y + total_h], fill=CAPTION_BG)

    # Draw each word coloured appropriately
    word_idx = 0
    for li, line in enumerate(lines):
        lwords = line.split()
        # measure line width
        line_w = sum(draw.textlength(lw + " ", font=font) for lw in lwords)
        x = (width - line_w) / 2
        y = bg_y + 10 + li * line_h
        for lw in lwords:
            color = CAPTION_HIGHLIGHT if word_idx == active_idx else CAPTION_COLOR
            draw.text((x, y), lw, font=font, fill=color)
            x += draw.textlength(lw + " ", font=font)
            word_idx += 1

    # Emoji overlay (top-right corner)
    if emoji:
        try:
            efont = _get_font(CAPTION_EMOJI_SIZE)
            draw.text((width - CAPTION_EMOJI_SIZE * len(emoji) - 10, 20), emoji, font=efont, fill=(255, 255, 255, 220))
        except Exception:
            pass

    return img


def build_caption_overlay(clip, words, emoji=""):
    """
    Build an ImageClip overlay for TikTok-style animated captions.
    """
    w, h = clip.size
    fps = 15  # caption render fps (lower = faster encode)
    frames = []
    times = np.arange(0, clip.duration, 1 / fps)
    for t in times:
        pil_img = make_caption_frame(w, h, words, t, emoji)
        frames.append(np.array(pil_img))

    def make_frame(t):
        idx = min(int(t * fps), len(frames) - 1)
        return frames[idx][:, :, :3]   # RGB

    def make_mask(t):
        idx = min(int(t * fps), len(frames) - 1)
        return frames[idx][:, :, 3] / 255.0

    caption_clip = (
        ImageClip(frames[0][:, :, :3], duration=clip.duration)
        .fl(lambda gf, t: make_frame(t))
        .set_duration(clip.duration)
    )
    # attach alpha mask
    from moviepy.editor import ImageClip as IC
    mask_clip = IC(frames[0][:, :, 3] / 255.0, ismask=True, duration=clip.duration).fl(
        lambda gf, t: make_mask(t)
    )
    caption_clip = caption_clip.set_mask(mask_clip)
    return caption_clip


# ===========================================================================
# STEP 6: Viral hook intro card
# ===========================================================================

def build_hook_card(width, height, hook_text, duration=2.5):
    """
    Creates a short text card (black bg, bold white text) to prepend as a hook intro.
    """
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _get_font(CAPTION_FONT_SIZE + 10)

    lines = textwrap.wrap(hook_text, width=20)
    line_h = CAPTION_FONT_SIZE + 20
    total_h = len(lines) * line_h
    y = (height - total_h) / 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        draw.text(((width - lw) / 2, y), line, font=font, fill=(255, 220, 0))
        y += line_h

    arr = np.array(img)
    card = ImageClip(arr, duration=duration)
    card = card.fx(vfx.fadein, 0.3).fx(vfx.fadeout, 0.3)
    return card


# ===========================================================================
# STEP 7: Music
# ===========================================================================

def pick_background_track(music_dir):
    print("[5/6] Selecting background music...")
    if not music_dir or not os.path.isdir(music_dir):
        print("  No music_dir — skipping.")
        return None
    tracks = [f for f in os.listdir(music_dir) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not tracks:
        print("  No audio files found — skipping.")
        return None
    chosen = random.choice(tracks)
    print(f"  Track: {chosen}")
    return os.path.join(music_dir, chosen)


# ===========================================================================
# STEP 8: Assemble clip (shared for single + multiple)
# ===========================================================================

def get_words_for_clip(transcript, start, end):
    """Return word-level timestamps that fall within [start, end]."""
    words = []
    for seg in transcript:
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start and w["end"] <= end:
                words.append({
                    "word":  w["word"],
                    "start": w["start"] - start,  # relative to clip
                    "end":   w["end"] - start,
                })
    # Fallback: if no word timestamps, fake uniform distribution from segment text
    if not words:
        for seg in transcript:
            if seg["end"] < start or seg["start"] > end:
                continue
            seg_words = seg["text"].split()
            if not seg_words:
                continue
            dur = (min(seg["end"], end) - max(seg["start"], start)) / len(seg_words)
            for i, sw in enumerate(seg_words):
                ws = max(seg["start"], start) - start + i * dur
                words.append({"word": sw, "start": ws, "end": ws + dur})
    return words


def assemble_clip(source, h, transcript, add_zoom, add_captions):
    """Cut, crop, optionally zoom, optionally add captions. Returns MoviePy clip."""
    start, end = float(h["start"]), float(h["end"])
    end = min(end, start + MAX_CLIP_DURATION, source.duration)
    if end <= start:
        return None

    sub = source.subclip(start, end)
    sub = smart_vertical_crop(sub)

    if add_zoom and h.get("zoom", False):
        sub = apply_auto_zoom(sub)

    if add_captions:
        words = get_words_for_clip(transcript, start, end)
        if words:
            emoji = h.get("emoji", "")
            cap = build_caption_overlay(sub, words, emoji)
            sub = CompositeVideoClip([sub, cap])

    sub = sub.fx(vfx.fadein, TRANSITION_DURATION).fx(vfx.fadeout, TRANSITION_DURATION)
    return sub


def attach_music(final_clip, music_path):
    if not music_path:
        return final_clip
    bg = AudioFileClip(music_path)
    bg = bg.fx(afx.audio_loop, duration=final_clip.duration)
    bg = bg.fx(afx.volumex, MUSIC_VOLUME)
    combined = CompositeAudioClip([final_clip.audio, bg]) if final_clip.audio else bg
    return final_clip.set_audio(combined)


# ===========================================================================
# RENDER: single merged short
# ===========================================================================

def build_final_video(video_path, highlights, hook, transcript, music_path, output_path,
                      add_zoom=True, add_captions=True):
    print("[6/6] Cutting, cropping, captioning, and rendering merged short...")
    source = VideoFileClip(video_path)
    w, h = source.size
    target_w = int(h * TARGET_RATIO)
    out_w = min(target_w, w)

    clips = []

    # Prepend hook card if hook text is available
    if hook.get("text"):
        hook_card = build_hook_card(out_w if target_w <= w else w, h, hook["text"])
        clips.append(hook_card)

    for hl in highlights:
        clip = assemble_clip(source, hl, transcript, add_zoom, add_captions)
        if clip:
            clips.append(clip)

    if not clips:
        raise RuntimeError("No valid clips generated.")

    final = concatenate_videoclips(clips, method="compose", padding=-TRANSITION_DURATION)
    final = attach_music(final, music_path)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30)
    print(f"\n✅ Merged short saved: {output_path}")


# ===========================================================================
# RENDER: multiple individual shorts
# ===========================================================================

def build_multiple_shorts(video_path, highlights, hook, transcript, music_path, output_dir,
                           add_zoom=True, add_captions=True):
    print(f"[6/6] Rendering {len(highlights)} individual shorts to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    source = VideoFileClip(video_path)
    output_paths = []

    for i, hl in enumerate(highlights):
        out_path = os.path.join(output_dir, f"short_{i+1:02d}.mp4")
        print(f"  → short {i+1}/{len(highlights)}: {hl['start']:.1f}s–{hl['end']:.1f}s  {hl.get('emoji','')}  {hl.get('reason','')[:40]}")

        clip = assemble_clip(source, hl, transcript, add_zoom, add_captions)
        if clip is None:
            print(f"     Skipped (invalid range).")
            continue

        single = clip
        single = attach_music(single, music_path)
        single.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=30)
        output_paths.append(out_path)

    print(f"\n✅ {len(output_paths)} individual shorts saved in: {output_dir}")
    return output_paths


# ===========================================================================
# DASHBOARD — save JSON report
# ===========================================================================

def save_dashboard(dashboard, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    print(f"📊 Groq analysis dashboard saved: {path}")

    # Pretty-print summary to terminal
    print("\n" + "="*60)
    print("  GROQ ANALYSIS DASHBOARD")
    print("="*60)
    print(f"  Language        : {dashboard.get('language','?')}")
    print(f"  Duration        : {dashboard.get('video_duration_s',0):.0f}s")
    print(f"  Virality Score  : {dashboard.get('overall_virality_score','?')}/10")
    print(f"  Summary         : {dashboard.get('content_summary','')}")
    print(f"  Caption Style   : {dashboard.get('recommended_caption_style','?')}")
    print(f"\n  HOOK")
    hook = dashboard.get("hook", {})
    print(f"    [{hook.get('start',0):.1f}–{hook.get('end',0):.1f}s] {hook.get('text','')}")
    print(f"    Reason: {hook.get('reason','')}")
    print(f"\n  HIGHLIGHTS ({dashboard.get('num_highlights',0)} clips)")
    for i, hl in enumerate(dashboard.get("highlights", []), 1):
        zoom_tag = "🔍zoom" if hl.get("zoom") else "      "
        print(f"    {i}. [{hl.get('start',0):.1f}–{hl.get('end',0):.1f}s] {hl.get('emoji','')} {zoom_tag}  {hl.get('reason','')[:50]}")
    print("="*60 + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="AI Auto Video Editor — Enhanced Edition")
    parser.add_argument("--input",          required=True,  help="Source video path")
    parser.add_argument("--output",         default="final_short.mp4", help="Output path (merged mode)")
    parser.add_argument("--output_dir",     default=None,   help="Output folder (multiple shorts mode)")
    parser.add_argument("--multiple",       action="store_true", help="Generate separate shorts per highlight")
    parser.add_argument("--music_dir",      default=None,   help="Folder of YOUR background tracks")
    parser.add_argument("--num_highlights", type=int, default=NUM_HIGHLIGHTS)
    parser.add_argument("--lang",           default="auto", help="Language: auto / en / hi / ne / ...")
    parser.add_argument("--no_zoom",        action="store_true", help="Disable auto zoom effect")
    parser.add_argument("--no_captions",    action="store_true", help="Disable animated captions")
    parser.add_argument("--dashboard",      default=None,   help="Path to save JSON analysis dashboard")
    args = parser.parse_args()

    # Probe duration
    probe = VideoFileClip(args.input)
    duration = probe.duration
    probe.close()

    # 1. Transcribe
    transcript, detected_lang = transcribe_video(args.input, args.lang)

    # 2. Groq full analysis
    lang_hint = LANG_NAMES.get(detected_lang, detected_lang)
    highlights, hook, dashboard = groq_full_analysis(
        transcript, duration, args.num_highlights, lang_hint
    )

    # 3. Dashboard
    dashboard_path = args.dashboard or (
        Path(args.output_dir or args.output).with_suffix("").as_posix() + "_dashboard.json"
    )
    save_dashboard(dashboard, dashboard_path)

    # 4. Music
    music_path = pick_background_track(args.music_dir)

    add_zoom     = not args.no_zoom
    add_captions = not args.no_captions

    # 5. Render
    if args.multiple or args.output_dir:
        out_dir = args.output_dir or "shorts_output"
        build_multiple_shorts(
            args.input, highlights, hook, transcript, music_path, out_dir,
            add_zoom=add_zoom, add_captions=add_captions
        )
    else:
        build_final_video(
            args.input, highlights, hook, transcript, music_path, args.output,
            add_zoom=add_zoom, add_captions=add_captions
        )


if __name__ == "__main__":
    main()
