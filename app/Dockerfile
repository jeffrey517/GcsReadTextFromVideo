FROM python:3.11-slim

# system deps for OpenCV video I/O
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
# include yt-dlp and opencv-python-headless
# (plus google-cloud-vision, google-cloud-storage, flask, numpy, etc.)
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
