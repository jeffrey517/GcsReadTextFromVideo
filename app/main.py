import os
import re
import cv2
import tempfile
import subprocess
from flask import Flask, request, jsonify
from google.cloud import vision, storage

app = Flask(__name__)

# ---- OCR tuning (TikTok-friendly) ----
FRAME_INTERVAL = 30
BOTTOM_IGNORE_RATIO = 0.25   # ignore bottom 25% (watermark area)
TOP_IGNORE_RATIO = 0.15      # ignore top 15% (username/music overlays)
UNWANTED = ["tiktok", "tik tok", "original sound", "music"]
MAX_VIDEO_SIZE = 512 * 1024 * 1024  # 512 MB
# -------------------------------------

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
                    # skip top/bottom overlays common on TikTok
                    if ymin < frame_height * TOP_IGNORE_RATIO:
                        continue
                    if ymin > frame_height * (1 - BOTTOM_IGNORE_RATIO):
                        continue
                    line_words.append(word_text)

                if line_words:
                    line_text = " ".join(line_words)
                    if not is_unwanted(line_text):
                        words.append(line_text)

    # de-dupe by normalized content
    seen, deduped = set(), []
    for line in words:
        norm = line.lower().replace(" ", "")
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)
    return " ".join(deduped)


def download_gcs_to_tempfile(gcs_uri: str) -> str:
    """Download gs:// object to a unique temp file, enforcing 512MB limit."""
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    blob.reload()

    if blob.size is None:
        raise ValueError("Could not determine object size.")
    if blob.size > MAX_VIDEO_SIZE:
        raise ValueError(f"Video too large: {blob.size/1024/1024:.2f} MB (limit 512 MB)")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    blob.download_to_filename(tmp.name)
    return tmp.name


def download_tiktok_to_tempfile(url: str) -> str:
    """
    Download a TikTok video to a temp file using yt-dlp.
    Adds timeout, UA header, and max-filesize to avoid hangs/oversized files.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    cmd = [
        "python", "-m", "yt_dlp",  # module form is more reliable than 'yt-dlp' on PATH
        "-o", tmp.name,
        "--no-warnings", "-q", "--no-progress",
        "--max-filesize", "512M",
        "--socket-timeout", "15",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "--restrict-filenames",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        # yt-dlp not installed in the image
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise ValueError("yt-dlp not found. Add 'pip install yt-dlp' in your Dockerfile.")
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise ValueError("Timed out downloading TikTok (try a shorter video).")
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        # pass along stderr for easier debugging
        raise ValueError(f"TikTok download failed: {e.stderr.strip() or 'unknown error'}")
    return tmp.name


def process_video_local(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV failed to open the downloaded video.")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num, last_text = 0, None
    results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % FRAME_INTERVAL == 0:
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    frame_num += 1
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
    source = data.get("video_uri") or data.get("gcs_uri") or data.get("tiktok_url")
    if not source:
        return jsonify({"error": "Provide 'video_uri' (gs:// or TikTok URL)."}), 400

    try:
        if source.startswith("gs://"):
            path = download_gcs_to_tempfile(source)
        elif "tiktok.com" in source:
            path = download_tiktok_to_tempfile(source)
        else:
            return jsonify({"error": "Unsupported source. Use gs:// or TikTok URL."}), 400

        results = process_video_local(path)
        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # surface unexpected errors
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
