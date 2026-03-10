import os
import subprocess
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Custom Caption Overlay")
templates = Jinja2Templates(directory="templates")

# Path to the font file (installed via fonts-freefont-ttf)
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"

def check_font():
    if not os.path.exists(FONT_PATH):
        logger.error(f"Font file not found at {FONT_PATH}")
        raise RuntimeError(f"Required font missing: {FONT_PATH}")
    logger.info(f"Font file found: {FONT_PATH}")

def add_caption_overlay(input_path: str, output_path: str, caption: str):
    """
    Run ffmpeg to overlay the given caption on the video.
    Text is yellow with black stroke and semi-transparent black background.
    """
    check_font()
    
    # Escape single quotes in caption for ffmpeg filter
    caption_escaped = caption.replace("'", r"'\''")
    
    # drawtext filter:
    # - text: user caption
    # - fontcolor=yellow
    # - borderw=3, bordercolor=black (black stroke)
    # - box=1, boxcolor=black@0.5 (semi-transparent background)
    # - boxborderw=10 (padding around text)
    # - centered: x=(w-text_w)/2, y=(h-text_h)/2
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"drawtext=text='{caption_escaped}':fontfile={FONT_PATH}:fontsize=48:fontcolor=yellow:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5:boxborderw=10:borderw=3:bordercolor=black",
        "-codec:a", "copy",
        "-y", output_path
    ]
    
    logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {result.stderr}")
        
        logger.info("FFmpeg completed successfully")
        
        # Verify output file exists and is not empty
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file was not created")
        if os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Output file is empty")
        
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg process timed out after 60 seconds")
        raise HTTPException(status_code=504, detail="Processing timeout (max 60 seconds)")
    except Exception as e:
        logger.exception("Unexpected error during ffmpeg execution")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the upload form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "healthy", "font_ok": os.path.exists(FONT_PATH)}

@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    caption: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a video and a caption, overlay the caption on the video,
    and return the result.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")
    
    # Determine file extension
    suffix = Path(file.filename).suffix
    if not suffix:
        suffix = ".mp4"  # default if no extension
    
    # Temporary input file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            content = await file.read()
            tmp_in.write(content)
            tmp_in_path = tmp_in.name
            logger.info(f"Saved uploaded file to {tmp_in_path}, size: {len(content)} bytes")
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Could not save uploaded file")
    
    # Temporary output file
    tmp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    
    try:
        # Process the video with caption
        add_caption_overlay(tmp_in_path, tmp_out_path, caption)
        
        # Schedule cleanup of temporary files after response is sent
        if background_tasks:
            background_tasks.add_task(os.unlink, tmp_in_path)
            background_tasks.add_task(os.unlink, tmp_out_path)
        else:
            # Fallback: delete input now (output will be cleaned later)
            os.unlink(tmp_in_path)
        
        # Return the processed file
        return FileResponse(
            tmp_out_path,
            media_type="video/mp4",
            filename=f"captioned_{file.filename}"
        )
    except HTTPException:
        # Clean up input file if it exists
        if 'tmp_in_path' in locals() and os.path.exists(tmp_in_path):
            try:
                os.unlink(tmp_in_path)
            except Exception as e:
                logger.warning(f"Could not delete input file {tmp_in_path}: {e}")
        # Re-raise the HTTP exception
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /process")
        # Clean up input file
        if 'tmp_in_path' in locals() and os.path.exists(tmp_in_path):
            try:
                os.unlink(tmp_in_path)
            except Exception as e:
                logger.warning(f"Could not delete input file {tmp_in_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
