import os
import subprocess
import tempfile
import shlex
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import uuid

app = FastAPI(title="FFmpeg + Amharic Transcriber")

# Mount templates for web UI
templates = Jinja2Templates(directory="templates")

# Global Whisper model (load once at startup)
model = None

@app.on_event("startup")
def load_model():
    global model
    # Use tiny model for low memory; can be changed to 'base' if RAM permits
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

def run_ffmpeg(cmd: list, input_path: str, output_path: str):
    """Execute ffmpeg command with proper error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # Render's timeout is 60s
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr}")
        return output_path
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Processing timeout (max 60 seconds)")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------

@app.post("/api/convert")
async def convert_media(
    file: UploadFile = File(...),
    output_format: str = Form(...)
):
    """Convert media to another format (e.g., mp4 → avi, mp3 → wav)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    out_suffix = f".{output_format.strip('.')}"
    tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix).name

    cmd = ["ffmpeg", "-i", tmp_in_path, "-y", tmp_out_path]
    try:
        run_ffmpeg(cmd, tmp_in_path, tmp_out_path)
        return FileResponse(tmp_out_path, media_type="application/octet-stream", filename=f"converted{out_suffix}")
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

@app.post("/api/trim")
async def trim_media(
    file: UploadFile = File(...),
    start: str = Form(...),
    end: str = Form(...)
):
    """Trim media between start and end times (format: HH:MM:SS or seconds)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
    cmd = ["ffmpeg", "-i", tmp_in_path, "-ss", start, "-to", end, "-c", "copy", "-y", tmp_out_path]
    try:
        run_ffmpeg(cmd, tmp_in_path, tmp_out_path)
        return FileResponse(tmp_out_path, media_type="application/octet-stream", filename=f"trimmed_{file.filename}")
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

@app.post("/api/extract-audio")
async def extract_audio(
    file: UploadFile = File(...),
    audio_format: str = Form("mp3")
):
    """Extract audio stream and save as mp3/wav/aac."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    out_suffix = f".{audio_format.strip('.')}"
    tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix).name
    cmd = ["ffmpeg", "-i", tmp_in_path, "-vn", "-acodec", "libmp3lame" if audio_format=="mp3" else "copy", "-y", tmp_out_path]
    try:
        run_ffmpeg(cmd, tmp_in_path, tmp_out_path)
        return FileResponse(tmp_out_path, media_type="application/octet-stream", filename=f"audio{out_suffix}")
    finally:
        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

@app.post("/api/transcribe")
async def transcribe_media(
    file: UploadFile = File(...),
    language: str = Form("am"),          # Default to Amharic
    task: str = Form("transcribe")       # or "translate"
):
    """Transcribe audio/video and return SRT subtitles."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, try again in a few seconds")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name

    # If video, extract audio first to a temporary file
    audio_path = tmp_in_path
    if Path(file.filename).suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        extract_cmd = ["ffmpeg", "-i", tmp_in_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path]
        try:
            run_ffmpeg(extract_cmd, tmp_in_path, audio_path)
        except Exception as e:
            os.unlink(tmp_in_path)
            raise e

    try:
        # Run Whisper
        segments, info = model.transcribe(audio_path, language=language, task=task, beam_size=5)

        # Build SRT content
        srt_lines = []
        for i, seg in enumerate(segments, start=1):
            start_srt = f"{int(seg.start//3600):02d}:{int((seg.start%3600)//60):02d}:{seg.start%60:06.3f}".replace('.', ',')
            end_srt = f"{int(seg.end//3600):02d}:{int((seg.end%3600)//60):02d}:{seg.end%60:06.3f}".replace('.', ',')
            srt_lines.append(f"{i}\n{start_srt} --> {end_srt}\n{seg.text}\n")

        srt_content = "\n".join(srt_lines)

        # Return as downloadable file
        srt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".srt")
        srt_file.write(srt_content.encode('utf-8'))
        srt_file.close()

        return FileResponse(srt_file.name, media_type="text/plain", filename=f"subtitles_{file.filename}.srt")
    finally:
        os.unlink(tmp_in_path)
        if audio_path != tmp_in_path:
            os.unlink(audio_path)
        if 'srt_file' in locals():
            os.unlink(srt_file.name)

@app.post("/api/burn-subtitles")
async def burn_subtitles(
    file: UploadFile = File(...),
    subtitle_file: UploadFile = File(...)
):
    """Burn an SRT subtitle file into the video."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_video:
        video_content = await file.read()
        tmp_video.write(video_content)
        tmp_video_path = tmp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
        srt_content = await subtitle_file.read()
        tmp_srt.write(srt_content)
        tmp_srt_path = tmp_srt.name

    tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix).name
    # Burn subtitles using the subtitles filter
    cmd = ["ffmpeg", "-i", tmp_video_path, "-vf", f"subtitles={tmp_srt_path}", "-y", tmp_out_path]
    try:
        run_ffmpeg(cmd, tmp_video_path, tmp_out_path)
        return FileResponse(tmp_out_path, media_type="application/octet-stream", filename=f"burned_{file.filename}")
    finally:
        os.unlink(tmp_video_path)
        os.unlink(tmp_srt_path)
        os.unlink(tmp_out_path)

@app.get("/api/info")
async def media_info(url: str):
    """Get media information using ffprobe (accepts a URL or file path)."""
    # For simplicity, we assume the URL is publicly accessible.
    # In a real scenario, you might also accept uploaded files.
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Could not probe media")
        return JSONResponse(content=result.stdout)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Probe timeout")

# For n8n or other clients – simple status endpoint
@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
