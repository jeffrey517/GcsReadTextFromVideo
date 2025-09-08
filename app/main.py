import tempfile
import os
import subprocess
import re
import cv2
from flask import Flask, request, jsonify
from google.cloud import vision, storage
from yt_dlp import YoutubeDL

app = Flask(__name__)

# --- CONFIG ---
FRAME_INTERVAL = 30
BOTTOM_IGNORE_RATIO = 0.25
TOP_IGNORE_RATIO = 0.15
UNWANTED = ["tiktok", "tik tok", "original sound", "music"]
MAX_VIDEO_SIZE = 512 * 1024 * 1024  # 512 MB
# ----------------

vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()


def is_unwanted(text: str) -> bool:
    text_norm = text.lower().replace(" ", "")
    if any(bad.replace(" ", "") in text_norm for bad in UNWANTED):
        return True
    if re.match(r"@[\w\d_]+", text_norm):
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
                    if ymin < frame_height * TOP_IGNORE_RATIO:
                        continue
                    if ymin > frame_height * (1 - BOTTOM_IGNORE_RATIO):
                        continue
                    line_words.append(word_text)
                if line_words:
                    line_text = " ".join(line_words)
                    if not is_unwanted(line_text):
                        words.append(line_text)

    seen = set()
    deduped = []
    for line in words:
        norm = line.lower().replace(" ", "")
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)

    return " ".join(deduped)


def download_gcs_to_tempfile(gcs_uri: str) -> str:
    """Download video from GCS to a local temp file with size check."""
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
    """Download TikTok video directly to MP4 via yt-dlp."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    ydl_opts = {
        'format': 'mp4',
        'merge_output_format': 'mp4',
        'outtmpl': temp_file.name,
        'quiet': True,
        'noplaylist': True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise ValueError(f"TikTok download failed: {str(e)}")

    # Validate the file
    if not os.path.exists(temp_file.name) or os.path.getsize(temp_file.name) == 0:
        raise ValueError("TikTok download resulted in empty file.")

    if os.path.getsize(temp_file.name) > MAX_VIDEO_SIZE:
        os.remove(temp_file.name)
        raise ValueError(f"TikTok video too large (> {MAX_VIDEO_SIZE/1024/1024:.0f} MB)")

    return temp_file.name


def process_video_local(video_path: str):
    """Run OCR on a local video file."""
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
        else:
            video_path = download_tiktok_to_tempfile(source)

        results = process_video_local(video_path)
        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
