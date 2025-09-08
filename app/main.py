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
    """
    Download TikTok video to a local temp file using yt-dlp.
    Returns the path to the local temp file with sanitized filename.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    def sanitize_filename(filename):
        """Remove or replace problematic characters in filenames"""
        # Remove or replace special characters, keep only alphanumeric, hyphens, underscores
        sanitized = re.sub(r'[^\w\s-]', '', filename)  # Remove special chars except spaces, hyphens, underscores
        sanitized = re.sub(r'\s+', '_', sanitized)     # Replace spaces with underscores
        sanitized = re.sub(r'_+', '_', sanitized)      # Replace multiple underscores with single
        sanitized = sanitized.strip('_')               # Remove leading/trailing underscores
        return sanitized[:100]  # Limit length to 100 characters
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # prefer mp4, fallback to best available
        'outtmpl': os.path.join(temp_dir, 'tiktok_video.%(ext)s'),  # use simple filename first
        'quiet': True,
        'merge_output_format': 'mp4',    # merge fragments if needed
        'noplaylist': True
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        if not info:
            raise ValueError("Failed to download TikTok video: no info extracted.")
        
        # Get the actual downloaded file path
        original_filename = ydl.prepare_filename(info)
        if not os.path.exists(original_filename):
            # Try with different extension if the original doesn't exist
            base_name = os.path.splitext(original_filename)[0]
            for ext in ['.mp4', '.webm', '.mkv']:
                potential_file = base_name + ext
                if os.path.exists(potential_file):
                    original_filename = potential_file
                    break
        
        if not os.path.exists(original_filename) or os.path.getsize(original_filename) == 0:
            raise ValueError("Failed to download TikTok video: file not found or empty.")
        
        # Create sanitized filename
        title = info.get('title', 'tiktok_video')
        sanitized_title = sanitize_filename(title)
        file_extension = os.path.splitext(original_filename)[1]
        sanitized_filename = os.path.join(temp_dir, f"{sanitized_title}{file_extension}")
        
        # Rename the file to have a clean filename
        os.rename(original_filename, sanitized_filename)
        
        return sanitized_filename

    except Exception as e:
        # Clean up temp directory on error
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise ValueError(f"Failed to download TikTok video: {str(e)}")

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
