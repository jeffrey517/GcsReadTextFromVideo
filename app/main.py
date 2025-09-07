import io
import os
import cv2
import re
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import vision
from google.cloud import storage
import tempfile

# --- CONFIG ---
FRAME_INTERVAL = 30   # process every 30th frame
BOTTOM_IGNORE_RATIO = 0.25  # ignore bottom 25% of the frame (TikTok watermarks)
UNWANTED = ["tiktok", "tik tok"]  # keywords to filter out
# --------------

app = Flask(__name__)

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


def stream_video_from_gcs(gcs_uri: str):
    assert gcs_uri.startswith("gs://"), "Path must be a GCS URI (gs://bucket/file.mp4)"
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)

    cloud_run_service_account_email = os.environ.get("K_SERVICE_ACCOUNT", "30892104547-compute@developer.gserviceaccount.com")

    # Get signed URL
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(version="v4", expiration=3600, method="GET", service_account_email=cloud_run_service_account_email)

    # FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=10**8
    )
    return process

def download_gcs_to_tempfile(gcs_uri: str) -> str:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    blob.download_to_filename(temp.name)
    return temp.name

def process_video(gcs_uri: str):
    process = stream_video_from_gcs(gcs_uri)

    # Get frame dimensions
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    cloud_run_service_account_email = os.environ.get("K_SERVICE_ACCOUNT", "30892104547-compute@developer.gserviceaccount.com")
    url = blob.generate_signed_url(version="v4", expiration=3600, method="GET", service_account_email=cloud_run_service_account_email)
    cap = cv2.VideoCapture(url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_size = width * height * 3  # bgr24
    frame_num = 0
    last_text = None
    results = []

    while True:
        raw = process.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))

        if frame_num % FRAME_INTERVAL == 0:
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                continue
            image = vision.Image(content=encoded.tobytes())
            response = vision_client.text_detection(image=image)

            if response.full_text_annotation:
                text = clean_text(response, height)
                if text.strip() and text != last_text:
                    results.append({"frame": frame_num, "text": text})
                    last_text = text

        frame_num += 1

    process.terminate()
    return results


@app.route("/", methods=["POST"])
def run_video_ocr():
    data = request.get_json(force=True)
    gcs_uri = data.get("gcs_uri")
    if not gcs_uri:
        return jsonify({"error": "Missing gcs_uri"}), 400

    try:
        results = process_video(gcs_uri)
        return jsonify({"gcs_uri": gcs_uri, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)