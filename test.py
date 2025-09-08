import tempfile
import os
import re
from yt_dlp import YoutubeDL

def download_tiktok_video_local(url: str) -> str:
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
    
result = download_tiktok_video_local("https://www.tiktok.com/@johnzohrab/video/7537108523931553042")
print(f"Downloaded file: {result}")
if os.path.exists(result):
    print(f"File size: {os.path.getsize(result)} bytes")
else:
    print("File does not exist!")