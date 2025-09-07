import tempfile
import os
import subprocess
import re
import json
import cv2
import requests
from flask import Flask, request, jsonify
from google.cloud import vision, storage

app = Flask(__name__)

# Parameters
FRAME_INTERVAL = 30
BOTTOM_IGNORE_RATIO = 0.25   # ignore bottom 25% (watermarks, captions)
TOP_IGNORE_RATIO = 0.15      # ignore top 15% (username/music text area)
UNWANTED = ["tiktok", "tik tok", "original sound", "music"]
MAX_VIDEO_SIZE = 512 * 1024 * 1024  # 512 MB

# Clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()


def is_unwanted(text: str) -> bool:
    text_norm = text.lower().replace(" ", "")
    if any(bad.replace(" ", "") in text_norm for bad in UNWANTED):
        return True
    if re.match(r"@[\w\d_]+", text_norm):  # usernames
        return True
    return False


def clean_text(response, frame_height):
    words = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                line_words = []
                for word in para.words:
                    word_text = "".join([s.text for s in word.symbols])
                    ymin = min(v.y for v in word.bounding_box.vertices)

                    # Ignore top/bottom overlays
                    if ymin < frame_height * TOP_IGNORE_RATIO:
                        continue
                    if ymin > frame_height * (1 - BOTTOM_IGNORE_RATIO):
                        continue

                    line_words.append(word_text)

                if line_words:
                    line_text = " ".join(line_words)
                    if not is_unwanted(line_text):
                        words.append(line_text)

    # Deduplicate
    seen = set()
    deduped = []
    for line in words:
        norm = line.lower().replace(" ", "")
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)

    return " ".join(deduped)


def download_gcs_to_tempfile(gcs_uri: str) -> str:
    """Download a GCS object into a temp file (with size check)."""
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.reload()
    if blob.size is None:
        raise ValueError("Could not determine object size.")
    if blob.size > MAX_VIDEO_SIZE:
        raise ValueError(f"Video too large: {blob.size/1024/1024:.2f} MB (limit 512 MB)")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    blob.download_to_filename(temp_file.name)
    return temp_file.name


def download_tiktok_to_tempfile(url: str) -> str:
    """Download TikTok video via yt-dlp + requests into a temp file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    try:
        # Get metadata + direct URL with headers
        result = subprocess.run(
            ["yt-dlp", "--dump-json", url],
            check=True, capture_output=True, text=True
        )

        if not result.stdout.strip():
            raise ValueError("yt-dlp returned no JSON output. Video may be unavailable.")

        info = json.loads(result.stdout)
        direct_url = info.get("url")
        headers = info.get("http_headers", {})

        if not direct_url:
            raise ValueError("Could not extract direct video URL from TikTok.")

        # Stream download
        import requests
        with requests.get(direct_url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(temp_file.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    except subprocess.CalledProcessError as e:
        raise ValueError(f"yt-dlp failed: {e.stderr.decode().strip()}")
    except Exception as e:
        raise ValueError(f"Failed to download TikTok video: {str(e)}")

    return temp_file.name


def process_video_local(video_path: str):
    """Run OCR frame-by-frame on a local video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV failed to open the video file.")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = 0
    last_text = None
    results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % FRAME_INTERVAL == 0:
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                image = vision.Image(content=encoded.tobytes())
                response = vision_client.text_detection(image=image)

                if response.full_text_annotation:
                    text = clean_text(response, height)
                    if text.strip() and text != last_text:
                        results.append(text)
                        last_text = text

            frame_num += 1
    finally:
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

    return results


@app.route("/", methods=["POST"])
def run_video_ocr():
    data = request.get_json(force=True)
    source = data.get("video_uri")

    if not source:
        return jsonify({"error": "Missing video_uri"}), 400

    try:
        if source.startswith("gs://"):
            video_path = download_gcs_to_tempfile(source)
        elif "tiktok.com" in source:
            video_path = download_tiktok_to_tempfile(source)
        else:
            return jsonify({"error": "Unsupported source"}), 400

        results = process_video_local(video_path)
        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
