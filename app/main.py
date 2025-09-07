import os
import re
import cv2
import subprocess
from flask import Flask, request, jsonify
from google.cloud import vision, storage

app = Flask(__name__)

# ---- OCR tuning ----
FRAME_INTERVAL = 30
BOTTOM_IGNORE_RATIO = 0.25   # ignore bottom 25% (watermark)
TOP_IGNORE_RATIO = 0.15      # ignore top 15% (user/music overlays)
UNWANTED = ["tiktok", "tik tok", "original sound", "music"]
MAX_VIDEO_SIZE = 512 * 1024 * 1024  # 512 MB
# ---------------------

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

    # de-dupe
    seen, deduped = set(), []
    for line in words:
        norm = line.lower().replace(" ", "")
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)
    return " ".join(deduped)


def get_tiktok_direct_url(url: str) -> str:
    """Use yt-dlp to resolve TikTok into a playable direct video URL."""
    cmd = [
        "yt-dlp", "-g",
        "--no-warnings", "--quiet", "--no-progress",
        url,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        direct_url = result.stdout.strip().splitlines()[0]
        if not direct_url.startswith("http"):
            raise ValueError("Failed to extract direct video URL from TikTok.")
        return direct_url
    except subprocess.TimeoutExpired:
        raise ValueError("Timed out resolving TikTok URL.")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"yt-dlp failed: {e.stderr.strip() or 'unknown error'}")


def process_video_stream(video_url: str):
    """Read video frames directly from URL (TikTok or GCS signed URL)."""
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise ValueError("Failed to open video stream with OpenCV.")

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
    return results


@app.route("/", methods=["POST"])
def run_video_ocr():
    data = request.get_json(force=True)
    source = data.get("video_uri") or data.get("gcs_uri") or data.get("tiktok_url")
    if not source:
        return jsonify({"error": "Provide 'video_uri' (gs:// or TikTok URL)."}), 400

    try:
        if source.startswith("gs://"):
            # turn GCS into signed URL for streaming
            bucket_name, blob_name = source[5:].split("/", 1)
            blob = storage_client.bucket(bucket_name).blob(blob_name)
            blob.reload()
            if blob.size and blob.size > MAX_VIDEO_SIZE:
                return jsonify({"error": f"Video too large: {blob.size/1024/1024:.2f} MB"}), 400
            video_url = blob.generate_signed_url(version="v4", expiration=3600, method="GET")
        elif "tiktok.com" in source:
            video_url = get_tiktok_direct_url(source)
        else:
            return jsonify({"error": "Unsupported source. Use gs:// or TikTok URL."}), 400

        results = process_video_stream(video_url)
        return jsonify(results)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
