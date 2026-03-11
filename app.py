import os
import subprocess
import tempfile
import shutil
import atexit
from flask import Flask, request, send_file, render_template, jsonify
from vosk import Model, KaldiRecognizer
import wave
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# -------------------------------------------------------------------
# Load Vosk model once at startup
# -------------------------------------------------------------------
MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', 'model')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {MODEL_PATH}. Please download it.")
model = Model(MODEL_PATH)

# -------------------------------------------------------------------
# Helper: clean up temporary files after response
# -------------------------------------------------------------------
_temp_files = []

def cleanup_temp_files():
    for f in _temp_files:
        try:
            os.remove(f)
        except:
            pass

atexit.register(cleanup_temp_files)

def register_temp_file(path):
    _temp_files.append(path)
    return path

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/separate', methods=['POST'])
def separate_audio():
    """Extract audio from uploaded video and return as MP3."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='_input_video') as tmp_in:
        video_file.save(tmp_in.name)
        input_path = register_temp_file(tmp_in.name)

    # Create a temporary output file for the audio
    output_fd, output_path = tempfile.mkstemp(suffix='.mp3')
    os.close(output_fd)
    register_temp_file(output_path)

    # Run ffmpeg to extract audio
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vn', '-acodec', 'libmp3lame',
        '-y', output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'ffmpeg failed: {e.stderr}'}), 500

    return send_file(output_path, as_attachment=True, download_name='audio.mp3')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio using Vosk and return text + word timestamps."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='_input_audio') as tmp_in:
        audio_file.save(tmp_in.name)
        input_path = register_temp_file(tmp_in.name)

    # Convert audio to 16kHz mono WAV using ffmpeg (required by Vosk)
    wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(wav_fd)
    register_temp_file(wav_path)

    convert_cmd = [
        'ffmpeg', '-i', input_path,
        '-ar', '16000', '-ac', '1',
        '-c:a', 'pcm_s16le',
        '-y', wav_path
    ]
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'ffmpeg conversion failed: {e.stderr}'}), 500

    # Open the WAV file with wave module
    wf = wave.open(wav_path, 'rb')
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
        return jsonify({'error': 'Audio file must be WAV format mono PCM.'}), 400

    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)  # Enable word timestamps

    results_text = []
    all_words = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            if 'text' in res:
                results_text.append(res['text'])
            if 'result' in res:
                all_words.extend(res['result'])

    final_res = json.loads(recognizer.FinalResult())
    if 'text' in final_res:
        results_text.append(final_res['text'])
    if 'result' in final_res:
        all_words.extend(final_res['result'])

    full_text = ' '.join(results_text).strip()

    return jsonify({
        'text': full_text,
        'words': all_words   # each word: {word, conf, start, end}
    })

@app.route('/merge', methods=['POST'])
def merge_video_audio():
    """Merge uploaded video and audio, adjusting volume."""
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Both video and audio files are required'}), 400
    video_file = request.files['video']
    audio_file = request.files['audio']
    if video_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'Empty file(s)'}), 400

    # Get volume factor (default 1.0)
    volume = request.form.get('volume', '1.0')
    try:
        volume = float(volume)
    except ValueError:
        return jsonify({'error': 'Volume must be a number'}), 400

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix='_video') as tmp_video:
        video_file.save(tmp_video.name)
        video_path = register_temp_file(tmp_video.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix='_audio') as tmp_audio:
        audio_file.save(tmp_audio.name)
        audio_path = register_temp_file(tmp_audio.name)

    # Output merged file
    out_fd, out_path = tempfile.mkstemp(suffix='_merged.mp4')
    os.close(out_fd)
    register_temp_file(out_path)

    # ffmpeg command: copy video stream, encode audio with volume filter
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-filter:a', f'volume={volume}',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y', out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'ffmpeg merge failed: {e.stderr}'}), 500

    return send_file(out_path, as_attachment=True, download_name='merged.mp4')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
