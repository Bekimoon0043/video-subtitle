FROM python:3.9-slim

# Install ffmpeg and wget
RUN apt-get update && apt-get install -y ffmpeg wget unzip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and extract Vosk small English model
RUN mkdir -p model && \
    wget -q https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O model.zip && \
    unzip model.zip -d model && \
    rm model.zip && \
    mv model/vosk-model-small-en-us-0.15/* model/ && \
    rmdir model/vosk-model-small-en-us-0.15

# Copy application code
COPY . .

# Set environment variable so app knows where the model is
ENV VOSK_MODEL_PATH=/app/model

# Render assigns a port via $PORT
EXPOSE 5000

CMD ["python", "app.py"]
