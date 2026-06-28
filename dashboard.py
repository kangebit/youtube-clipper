import subprocess
import sys

# Memaksa sistem menginstal google-genai pada environment yang aktif digunakan Streamlit
try:
    import google_genai
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])

import os
import json
import pickle
import textwrap

import cv2
import numpy as np
import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel

# Library Tambahan untuk Google & YouTube API
from google import genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# =========================
# FOLDER
# =========================
DOWNLOAD_FOLDER = "downloads"
OUTPUT_FOLDER = "output"
SUBTITLE_FOLDER = "subtitles"

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)


# =========================
# PENGATURAN VIDEO
# =========================
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

VIDEO_CRF = "23"
VIDEO_PRESET = "veryfast"
AUDIO_BITRATE = "192k"


# =========================
# PENGATURAN DOWNLOAD
# =========================
COOKIE_FILE = "cookies.txt"


# =========================
# PENGATURAN DYNAMIC CROP
# =========================
DYNAMIC_DETECT_EVERY_FRAMES = 5
DYNAMIC_CROP_SMOOTHING = 0.12
DYNAMIC_DETECTION_WIDTH = 640
FACE_VERTICAL_POSITION = 0.35


# =========================
# PENGATURAN SUBTITLE
# =========================
SUBTITLE_FONT_NAME = "Montserrat ExtraBold"
SUBTITLE_FONT_SIZE = 60
SUBTITLE_FONT_COLOR = "#FFFFFF"
SUBTITLE_OUTLINE_COLOR = "#000000"
SUBTITLE_MARGIN_BOTTOM = 180
SUBTITLE_OUTLINE_SIZE = 4
SUBTITLE_SHADOW_SIZE = 1

SUBTITLE_MAX_CHARS_PER_LINE = 28
SUBTITLE_MAX_LINES = 2
SUBTITLE_MAX_SECONDS_PER_BLOCK = 2.6
SUBTITLE_MIN_SECONDS_PER_BLOCK = 0.6

WHISPER_LANGUAGE = "id"
BURN_SUBTITLE_TO_VIDEO = True


# =========================
# FUNGSI BANTUAN UMUM
# =========================
def run_command(cmd):
    subprocess.run(cmd, check=True)


def clean_old_downloads():
    for file in os.listdir(DOWNLOAD_FOLDER):
        path = os.path.join(DOWNLOAD_FOLDER, file)
        if os.path.isfile(path):
            os.remove(path)


def parse_timestamps(text):
    text = text.replace(".", ":")
    parts = [item.strip() for item in text.split(",") if item.strip()]
    clips = []
    for part in parts:
        if "-" not in part:
            raise ValueError(f"Format timestamp salah: {part}")
        start, end = part.split("-", 1)
        clips.append((start.strip(), end.strip()))
    return clips


def timestamp_to_seconds(timestamp):
    timestamp = timestamp.replace(".", ":")
    parts = timestamp.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    raise ValueError(f"Format timestamp tidak valid: {timestamp}")


def get_clip_duration(start, end):
    start_sec = timestamp_to_seconds(start)
    end_sec = timestamp_to_seconds(end)
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Durasi timestamp tidak valid: {start} - {end}")
    return str(duration)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


# =========================
# INTEGRASI AUTOMATION API
# =========================
def get_timestamps_from_gemini(video_url):
    """Meminta Gemini API menganalisis link video dan memberikan data timestamp + metadata."""
    try:
        # Mengambil API Key langsung dari sistem environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variable.")
        
        # Masukkan variabel api_key ke dalam Client()
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Gagal menginisialisasi Gemini Client: {e}")

    prompt = f"""
Analisis video YouTube ini dan buat klip Shorts berkualitas tinggi.
Link Video: {{video_url}}

WAJIB MEMBUAT TEPAT 3 KLIP TERBAIK!

PERSYARATAN PENTING:

1. JUMLAH KLIP: HARUS 3 KLIP (tidak kurang, tidak lebih)

2. TIMESTAMP HARUS PRESISI:
   - SELALU mulai dan akhiri timestamp pada KALIMAT LENGKAP, bukan tengah-tengah percakapan
   - Hindari memotong di saat seseorang sedang berbicara / tengah kata
   - Cari momen PUNCH LINE atau kesimpulan natural sebelum memotong
   - Contoh BAIK: "...dan itulah kenapa saya memilih ini." [POTONG DI SINI]
   - Contoh BURUK: "...dan itulah kenapa saya [POTONG DI TENGAH] memilih ini."

3. DURASI KLIP: 30-50 detik per klip

4. JUDUL HARUS SPESIFIK & MENARIK:
   - BUKAN: "Cerita Menarik" atau "Tips Bagus" atau "Klip Seru"
   - TAPI: Judul yang LANGSUNG MENJELASKAN konten klip dengan detail
   - Setiap klip harus punya judul BERBEDA dan UNIK
   - Tambahkan emoji dan hashtag yang relevan dengan topik klip

5. DESKRIPSI:
   - Jelaskan SECARA SPESIFIK apa yang dibahas di klip ini
   - Gunakan emoji yang sesuai dengan topik
   - Tambahkan CTA (Call To Action) untuk subscribe/follow
   - Setiap deskripsi HARUS BERBEDA, jangan copy-paste

6. PRIORITAS KLIP:
   - Klip 1: Momen surprise, plot twist, atau revelasi besar
   - Klip 2: Penjelasan yang sangat jelas dan mudah dipahami
   - Klip 3: Konten yang memicu emosi (tertawa, terkejut, atau tersentuh)

BERIKAN HASIL DALAM FORMAT JSON INI (TANPA MARKDOWN, LANGSUNG JSON):
{{
  "timestamps_string": "MM:SS-MM:SS,MM:SS-MM:SS,MM:SS-MM:SS",
  "clips_metadata": [
    {{
      "title": "[Judul SPESIFIK untuk Klip 1]",
      "description": "[Deskripsi unik untuk klip 1 + emoji + CTA]"
    }},
    {{
      "title": "[Judul SPESIFIK untuk Klip 2 - BERBEDA dari klip 1]",
      "description": "[Deskripsi unik untuk klip 2 + emoji + CTA]"
    }},
    {{
      "title": "[Judul SPESIFIK untuk Klip 3 - BERBEDA dari klip 1 & 2]",
      "description": "[Deskripsi unik untuk klip 3 + emoji + CTA]"
    }}
  ]
}}

CONTOH YANG BAIK:
{{
  "timestamps_string": "05:20-06:00,12:34-13:15,25:45-26:30",
  "clips_metadata": [
    {{
      "title": "Plot Twist yang Mengubah Segalanya 🤯",
      "description": "Siapa sangka cerita akan berbelok seperti ini? Lihat momen shockingnya! Subscribe untuk reaksi lainnya. #plottwist #surprise"
    }},
    {{
      "title": "Penjelasan Ilmu yang Bikin Sadar 🧠",
      "description": "Ini alasan kenapa kebanyakan orang salah paham tentang hal ini. Cek penjelasan lengkapnya di sini! #edukasi #ilmupengetahuan"
    }},
    {{
      "title": "Momen Lucu yang Bikin Ketawa 😂",
      "description": "Gak ada yang bisa menahan tawa di bagian ini. Lihat reaksi mereka! Jangan lupa subscribe untuk konten seru. #lucu #humor"
    }}
  ]
}}
"""
    
    # Format konfigurasi yang dijamin aman untuk SDK 2.10.0 agar terhindar dari 404 v1beta
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
        },
    )
    return json.loads(response.text)


def get_youtube_service():
    """Mengurus autentikasi OAuth2 dan mengembalikan objek service YouTube Data API v3."""
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    creds = None
    
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('client_secret.json'):
                raise FileNotFoundError("File 'client_secret.json' tidak ditemukan! Silakan unduh dari Google Cloud Console.")
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            
    return build('youtube', 'v3', credentials=creds)


def upload_clip_to_youtube(video_path, title, description):
    """Mengunggah file video hasil render ke akun YouTube."""
    try:
        youtube = get_youtube_service()
        
        body = {
            'snippet': {
                'title': title[:100],  # YouTube membatasi judul maksimal 100 karakter
                'description': description,
                'categoryId': '22'     # Kategori: People & Blogs
            },
            'status': {
                'privacyStatus': 'public'  # Bisa diubah ke 'private' atau 'unlisted'
            }
        }
        
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype='video/mp4')
        request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
        return response.get('id')
    except Exception as e:
        print(f"Peringatan proses upload YouTube gagal: {e}")
        return None


# =========================
# FUNGSI SUBTITLE ASS
# =========================
def ass_color(hex_color):
    hex_color = hex_color.replace("#", "")
    if len(hex_color) != 6:
        raise ValueError("Format warna harus HEX 6 digit, contoh: #FFFFFF")
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    return f"&H00{b}{g}{r}"


def format_ass_time(seconds):
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"


def clean_subtitle_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("{", "").replace("}", "").replace("\n", " ")
    return " ".join(text.split())


def escape_ass_text(text):
    text = clean_subtitle_text(text)
    return text.replace("{", "").replace("}", "")


def count_wrapped_lines(text):
    lines = textwrap.wrap(
        text,
        width=SUBTITLE_MAX_CHARS_PER_LINE,
        break_long_words=False,
        break_on_hyphens=False
    )
    return len(lines)


def wrap_subtitle_text(text):
    lines = textwrap.wrap(
        text,
        width=SUBTITLE_MAX_CHARS_PER_LINE,
        break_long_words=False,
        break_on_hyphens=False
    )
    lines = lines[:SUBTITLE_MAX_LINES]
    return "\\N".join(lines)


def split_text_into_caption_blocks(text, max_chars_per_line=28, max_lines=2):
    text = clean_subtitle_text(text)
    words = text.split()
    if not words:
        return []

    blocks = []
    current_words = []

    for word in words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        candidate_lines = textwrap.wrap(
            candidate_text,
            width=max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False
        )

        if len(candidate_lines) <= max_lines:
            current_words.append(word)
        else:
            if current_words:
                blocks.append(" ".join(current_words))
                current_words = [word]
            else:
                blocks.append(word)
                current_words = []

    if current_words:
        blocks.append(" ".join(current_words))
    return blocks


def write_ass_header(file_object):
    font_color = ass_color(SUBTITLE_FONT_COLOR)
    outline_color = ass_color(SUBTITLE_OUTLINE_COLOR)

    file_object.write("[Script Info]\n")
    file_object.write("ScriptType: v4.00+\n")
    file_object.write("PlayResX: 1080\n")
    file_object.write("PlayResY: 1920\n")
    file_object.write("WrapStyle: 0\n")
    file_object.write("ScaledBorderAndShadow: yes\n\n")

    file_object.write("[V4+ Styles]\n")
    file_object.write(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
    )

    style_line = (
        f"Style: Default,{SUBTITLE_FONT_NAME},{SUBTITLE_FONT_SIZE},{font_color},&H000000FF,"
        f"{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{SUBTITLE_OUTLINE_SIZE},{SUBTITLE_SHADOW_SIZE},2,70,70,{SUBTITLE_MARGIN_BOTTOM},1\n\n"
    )
    file_object.write(style_line)

    file_object.write("[Events]\n")
    file_object.write(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


def add_dialogue(file_object, start_time, end_time, text):
    text = clean_subtitle_text(text)
    if not text:
        return

    wrapped_text = wrap_subtitle_text(text)
    wrapped_text = escape_ass_text(wrapped_text)

    if end_time <= start_time:
        end_time = start_time + SUBTITLE_MIN_SECONDS_PER_BLOCK

    start = format_ass_time(start_time)
    end = format_ass_time(end_time)
    file_object.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{wrapped_text}\n")


def create_ass_subtitle(segments, ass_path):
    all_words = []
    for segment in segments:
        if hasattr(segment, "words") and segment.words:
            for word in segment.words:
                word_text = clean_subtitle_text(getattr(word, "word", ""))
                if not word_text:
                    continue

                start_time = getattr(word, "start", None)
                end_time = getattr(word, "end", None)

                if start_time is None:
                    start_time = getattr(segment, "start", 0)
                if end_time is None:
                    end_time = getattr(segment, "end", start_time + 0.5)

                all_words.append({
                    "text": word_text,
                    "start": float(start_time),
                    "end": float(end_time)
                })

    with open(ass_path, "w", encoding="utf-8") as f:
        write_ass_header(f)

        if all_words:
            current_words = []
            for word_data in all_words:
                candidate_words = current_words + [word_data]
                candidate_text = " ".join(item["text"] for item in candidate_words)

                block_start = candidate_words[0]["start"]
                block_end = candidate_words[-1]["end"]
                block_duration = block_end - block_start

                too_many_lines = count_wrapped_lines(candidate_text) > SUBTITLE_MAX_LINES
                too_long_duration = block_duration > SUBTITLE_MAX_SECONDS_PER_BLOCK

                if current_words and (too_many_lines or too_long_duration):
                    text = " ".join(item["text"] for item in current_words)
                    start_time = current_words[0]["start"]
                    end_time = current_words[-1]["end"]

                    if end_time - start_time < SUBTITLE_MIN_SECONDS_PER_BLOCK:
                        end_time = start_time + SUBTITLE_MIN_SECONDS_PER_BLOCK

                    add_dialogue(f, start_time, end_time, text)
                    current_words = [word_data]
                else:
                    current_words = candidate_words

            if current_words:
                text = " ".join(item["text"] for item in current_words)
                start_time = current_words[0]["start"]
                end_time = current_words[-1]["end"]

                if end_time - start_time < SUBTITLE_MIN_SECONDS_PER_BLOCK:
                    end_time = start_time + SUBTITLE_MIN_SECONDS_PER_BLOCK

                add_dialogue(f, start_time, end_time, text)
        else:
            for segment in segments:
                raw_text = clean_subtitle_text(getattr(segment, "text", ""))
                if not raw_text:
                    continue

                caption_blocks = split_text_into_caption_blocks(
                    raw_text,
                    max_chars_per_line=SUBTITLE_MAX_CHARS_PER_LINE,
                    max_lines=SUBTITLE_MAX_LINES
                )

                segment_start = float(getattr(segment, "start", 0))
                segment_end = float(getattr(segment, "end", segment_start + 1))
                segment_duration = max(0.01, segment_end - segment_start)

                total_words = sum(len(block.split()) for block in caption_blocks)
                if total_words <= 0:
                    continue

                current_time = segment_start
                cumulative_words = 0

                for index, block in enumerate(caption_blocks):
                    block_word_count = len(block.split())
                    cumulative_words += block_word_count

                    block_start = current_time
                    if index == len(caption_blocks) - 1:
                        block_end = segment_end
                    else:
                        block_end = segment_start + (
                            segment_duration * cumulative_words / total_words
                        )

                    if block_end - block_start < SUBTITLE_MIN_SECONDS_PER_BLOCK:
                        block_end = block_start + SUBTITLE_MIN_SECONDS_PER_BLOCK

                    add_dialogue(f, block_start, block_end, block)
                    current_time = block_end


def burn_subtitle(input_clip, ass_path, final_clip):
    safe_ass_path = ass_path.replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_clip,
        "-vf", f"ass='{safe_ass_path}'",
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        final_clip
    ]
    run_command(cmd)


# =========================
# FUNGSI CUT VIDEO
# =========================
def cut_original_clip(video_path, start, duration, output_clip):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", video_path,
        "-t", duration,
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        output_clip
    ]
    run_command(cmd)


def make_original_output(video_path, start, duration, output_clip):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", video_path,
        "-t", duration,
        "-c", "copy",
        output_clip
    ]
    run_command(cmd)


def make_blur_background_output(video_path, start, duration, output_clip):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", video_path,
        "-t", duration,
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:1[bg];"
        "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p",
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        output_clip
    ]
    run_command(cmd)


def make_center_crop_output(video_path, start, duration, output_clip):
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", video_path,
        "-t", duration,
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,format=yuv420p",
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        output_clip
    ]
    run_command(cmd)


# =========================
# FUNGSI DYNAMIC AUTO CROP
# =========================
def load_cascade(filename):
    cascade_path = cv2.data.haarcascades + filename
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Peringatan: cascade tidak berhasil dimuat: {filename}")
        return None
    return cascade


def safe_detect(detector, gray, **kwargs):
    if detector is None:
        return []
    try:
        return detector.detectMultiScale(gray, **kwargs)
    except Exception as e:
        print(f"Peringatan deteksi gagal: {e}")
        return []


def detect_main_subject(frame, face_detector, upper_body_detector, full_body_detector):
    if frame is None or frame.size == 0:
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    except Exception as e:
        print(f"Peringatan: gagal membuat grayscale untuk deteksi: {e}")
        return None

    faces = safe_detect(face_detector, gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return x + (w / 2), y + (h / 2), "face"

    upper_bodies = safe_detect(upper_body_detector, gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 80))
    if len(upper_bodies) > 0:
        x, y, w, h = max(upper_bodies, key=lambda box: box[2] * box[3])
        return x + (w / 2), y + (h * 0.35), "upper_body"

    full_bodies = safe_detect(full_body_detector, gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 120))
    if len(full_bodies) > 0:
        x, y, w, h = max(full_bodies, key=lambda box: box[2] * box[3])
        return x + (w / 2), y + (h * 0.25), "full_body"

    return None


def dynamic_auto_crop_person(input_clip, output_clip):
    temp_video = output_clip.replace(".mp4", "_noaudio.mp4")
    cap = cv2.VideoCapture(input_clip)

    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video untuk dynamic auto crop.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_w <= 0 or input_h <= 0:
        cap.release()
        raise RuntimeError("Ukuran video tidak valid.")

    scale_factor = max(TARGET_WIDTH / input_w, TARGET_HEIGHT / input_h)
    scaled_w = int(input_w * scale_factor)
    scaled_h = int(input_h * scale_factor)

    if scaled_w < TARGET_WIDTH: scaled_w = TARGET_WIDTH
    if scaled_h < TARGET_HEIGHT: scaled_h = TARGET_HEIGHT

    max_crop_x = max(0, scaled_w - TARGET_WIDTH)
    max_crop_y = max(0, scaled_h - TARGET_HEIGHT)

    current_crop_x = max_crop_x // 2
    current_crop_y = max_crop_y // 2
    target_crop_x, target_crop_y = current_crop_x, current_crop_y

    face_detector = load_cascade("haarcascade_frontalface_default.xml")
    upper_body_detector = load_cascade("haarcascade_upperbody.xml")
    full_body_detector = load_cascade("haarcascade_fullbody.xml")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Gagal membuat video writer untuk dynamic auto crop.")

    frame_index = 0
    detected_once = False
    last_detection_type = "center"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            resized = np.ascontiguousarray(resized, dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"Gagal resize frame ke-{frame_index}: {e}")

        if frame_index % DYNAMIC_DETECT_EVERY_FRAMES == 0:
            try:
                if scaled_w > DYNAMIC_DETECTION_WIDTH:
                    detection_scale = DYNAMIC_DETECTION_WIDTH / scaled_w
                    detection_w = DYNAMIC_DETECTION_WIDTH
                    detection_h = int(scaled_h * detection_scale)
                    if detection_h <= 0: detection_h = 1
                    detection_frame = cv2.resize(resized, (detection_w, detection_h), interpolation=cv2.INTER_LINEAR)
                else:
                    detection_scale = 1.0
                    detection_frame = resized

                detection_frame = np.ascontiguousarray(detection_frame, dtype=np.uint8)
                detection = detect_main_subject(detection_frame, face_detector, upper_body_detector, full_body_detector)

                if detection is not None:
                    detected_x, detected_y, detection_type = detection
                    subject_center_x = detected_x / detection_scale
                    subject_center_y = detected_y / detection_scale

                    target_crop_x = int(subject_center_x - (TARGET_WIDTH / 2))
                    if detection_type == "face":
                        target_crop_y = int(subject_center_y - (TARGET_HEIGHT * FACE_VERTICAL_POSITION))
                    else:
                        target_crop_y = int(subject_center_y - (TARGET_HEIGHT * 0.35))

                    target_crop_x = int(clamp(target_crop_x, 0, max_crop_x))
                    target_crop_y = int(clamp(target_crop_y, 0, max_crop_y))
                    detected_once = True
                    last_detection_type = detection_type
            except Exception as e:
                print(f"Peringatan: deteksi gagal pada frame ke-{frame_index}: {e}")

        current_crop_x = int(current_crop_x + (target_crop_x - current_crop_x) * DYNAMIC_CROP_SMOOTHING)
        current_crop_y = int(current_crop_y + (target_crop_y - current_crop_y) * DYNAMIC_CROP_SMOOTHING)

        current_crop_x = int(clamp(current_crop_x, 0, max_crop_x))
        current_crop_y = int(clamp(current_crop_y, 0, max_crop_y))

        x1, y1 = current_crop_x, current_crop_y
        x2, y2 = current_crop_x + TARGET_WIDTH, current_crop_y + TARGET_HEIGHT

        x1 = int(clamp(x1, 0, scaled_w - 1))
        y1 = int(clamp(y1, 0, scaled_h - 1))
        x2 = int(clamp(x2, x1 + 1, scaled_w))
        y2 = int(clamp(y2, y1 + 1, scaled_h))

        cropped = resized[y1:y2, x1:x2]
        if cropped is None or cropped.size == 0:
            raise RuntimeError(f"Hasil crop kosong pada frame ke-{frame_index}.")

        if cropped.shape[0] != TARGET_HEIGHT or cropped.shape[1] != TARGET_WIDTH:
            cropped = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        cropped = np.ascontiguousarray(cropped, dtype=np.uint8)
        writer.write(cropped)
        frame_index += 1

    cap.release()
    writer.release()

    if not detected_once:
        print("Peringatan: wajah/orang tidak terdeteksi. Hasil memakai crop tengah.")

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_clip,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", VIDEO_PRESET,
        "-crf", VIDEO_CRF,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-shortest",
        output_clip
    ]
    run_command(cmd)

    if os.path.exists(temp_video):
        os.remove(temp_video)


def make_dynamic_crop_output(video_path, start, duration, clip_index, output_clip):
    temp_original_clip = os.path.join(OUTPUT_FOLDER, f"clip_{clip_index:03d}_original_temp.mp4")
    cut_original_clip(video_path=video_path, start=start, duration=duration, output_clip=temp_original_clip)
    dynamic_auto_crop_person(input_clip=temp_original_clip, output_clip=output_clip)
    if os.path.exists(temp_original_clip):
        os.remove(temp_original_clip)


# =========================
# FUNGSI LIVE LOG DASHBOARD
# =========================
def update_live_log(log_box, message, status="running"):
    if status == "running":
        icon, color, bg, border = "⏳", "#111827", "#fff7ed", "#fed7aa"
    elif status == "success":
        icon, color, bg, border = "✅", "#065f46", "#ecfdf5", "#a7f3d0"
    elif status == "error":
        icon, color, bg, border = "❌", "#991b1b", "#fef2f2", "#fecaca"
    else:
        icon, color, bg, border = "ℹ️", "#111827", "#f8fafc", "#e5e7eb"

    log_box.markdown(
        f"""
        <div style="
            background:{bg};
            border:1px solid {border};
            color:{color};
            padding:16px 18px;
            border-radius:14px;
            font-weight:700;
            margin:12px 0 18px 0;
        ">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def create_yt_dlp_hook(log_box):
    def hook(d):
        status = d.get("status")
        if status == "downloading":
            percent = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            eta = d.get("_eta_str", "").strip()
            message = "Mengunduh video dari YouTube"
            if percent: message += f" | {percent}"
            if speed: message += f" | {speed}"
            if eta: message += f" | ETA {eta}"
            update_live_log(log_box, message)
        elif status == "finished":
            update_live_log(log_box, "Download selesai. Menyiapkan file video...", "success")
    return hook


# =========================
# DASHBOARD STREAMLIT
# =========================
st.set_page_config(
    page_title="YouTube Clipper Dashboard",
    page_icon="🎬",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp { background: #f8fafc !important; color: #111827 !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111827 !important; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 920px; }
    .hero-box { background: linear-gradient(135deg, #111827, #1f2937); padding: 30px; border-radius: 24px; color: white !important; margin-bottom: 24px; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18); }
    .hero-box h1, .hero-box p { color: white !important; }
    .hero-box p { color: #d1d5db !important; }
    div[data-testid="stTextInput"] input { background-color: white !important; color: #111827 !important; border: 1px solid #d1d5db !important; border-radius: 12px !important; }
    div[data-testid="stTextInput"] input::placeholder { color: #6b7280 !important; }
    div[data-testid="stRadio"] label, div[data-testid="stToggle"] label { color: #111827 !important; font-weight: 600 !important; }
    .stButton > button { height: 52px; border-radius: 14px; font-weight: 800; font-size: 16px; background: #ef4444 !important; color: white !important; border: none !important; }
    .stButton > button:hover { background: #dc2626 !important; }
    .stButton > button p, .stButton > button span { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
        <h1>🎬 AI YouTube Shorts Automator</h1>
        <p>Tempel tautan video, Gemini API akan mencari timestamp + judul terbaik, lalu otomatis render dan terbit ke YouTube Shorts.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### ⚙️ Setup API (WAJIB!)")
st.info("🔑 Setiap orang perlu input API Key mereka sendiri. Bukan pakai punya orang lain.")

with st.expander("📖 Cara Mendapatkan Gemini API Key (Klik Buka)", expanded=False):
    st.markdown("""
    **Langkah 1-5:**
    1. Buka: https://aistudio.google.com/app/apikey
    2. Klik **"Create API Key"**
    3. Pilih **"Create API key in new project"**
    4. Copy API Key yang muncul
    5. Paste di form bawah ini
    """)

gemini_api_key = st.text_input(
    "Masukkan Gemini API Key Kamu",
    type="password",
    placeholder="AIzaSy... (paste API Key mu di sini)"
)

if not gemini_api_key:
    st.warning("⚠️ Gemini API Key belum diisi. Aplikasi tidak bisa berjalan tanpa ini.")
    st.stop()

# Set ke environment variable untuk function get_timestamps_from_gemini
os.environ["GEMINI_API_KEY"] = gemini_api_key
st.success("✅ Gemini API Key sudah tersimpan di session ini.")
st.divider()

st.markdown("### Input Konten")
youtube_url = st.text_input("Link YouTube Sumber", placeholder="Tempel link YouTube panjang yang ingin diambil klipnya di sini")
st.caption("AI akan menganalisis konten video ini secara otomatis untuk mendeteksi momen paling seru.")
st.divider()

st.markdown("### Pengaturan Kreatif Output")
mode_label = st.radio("Mode Kedalaman Video (Crop)", ["Original", "Portrait Blur Background", "Portrait Crop Tengah", "Dynamic Crop Wajah/Orang"], horizontal=True)

mode_map = {
    "Original": "1",
    "Portrait Blur Background": "2",
    "Portrait Crop Tengah": "3",
    "Dynamic Crop Wajah/Orang": "4"
}
mode = mode_map[mode_label]

use_subtitle = st.toggle("Pakai subtitle dinamis (Whisper)", value=True)
if use_subtitle:
    whisper_model_size = st.radio("Skala Intelegensi Whisper", ["small", "medium", "large-v3"], horizontal=True, index=2)
else:
    whisper_model_size = None
    st.info("Subtitle tidak aktif. Waktu rendering video akan berjalan jauh lebih cepat.")

st.divider()
process_button = st.button("🚀 Jalankan Otomatisasi (Gemini -> Cut -> Upload YT)", type="primary", use_container_width=True)
live_log = st.empty()

if process_button:
    if not youtube_url.strip():
        st.error("Link YouTube sumber wajib diisi.")
        st.stop()

    result_files = []
    subtitle_files = []

    with st.status("Memulai pipeline otomatisasi...", expanded=True) as status:
        try:
            # --- LANGKAH 1: ANALISIS GEMINI API UNTUK MENDAPATKAN TIMESTAMP & METADATA ---
            st.write("🧠 Menghubungi Gemini API untuk menganalisis momen terbaik...")
            try:
                gemini_data = get_timestamps_from_gemini(youtube_url)
                timestamps_string = gemini_data["timestamps_string"]
                clips_metadata = gemini_data["clips_metadata"]
                st.success(f"Gemini merekomendasikan klip pada timestamp: {timestamps_string}")
                timestamps = parse_timestamps(timestamps_string)
            except Exception as e:
                st.error(f"Gagal mendapatkan data analitik dari Gemini: {e}")
                st.stop()

            # --- LANGKAH 2: PREPARASI & DOWNLOAD VIDEO UTAMA ---
            st.write("Clean-up sisa berkas lama...")
            clean_old_downloads()

            st.write("Mengunduh video utama dari YouTube...")
            ydl_opts = {
                "outtmpl": os.path.join(DOWNLOAD_FOLDER, "video.%(ext)s"),
                "format": "best[ext=mp4]/best",
                "merge_output_format": "mp4",
                "noplaylist": True,
                "retries": 10,
                "fragment_retries": 10,
                "progress_hooks": [create_yt_dlp_hook(live_log)],
                # === FIX HTTP 403 ERROR ===
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Referer": "https://www.youtube.com/"
                },
                "socket_timeout": 30,
                "skip_unavailable_fragments": True,
                "prefer_insecure": False,
                "youtube_include_dash_manifest": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["web"],
                        "player_skip_js": False
                    }
                }
            }

            if os.path.exists(COOKIE_FILE):
                ydl_opts["cookiefile"] = COOKIE_FILE
                st.write(f"Menggunakan cookie file: {COOKIE_FILE}")

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    st.error("❌ YouTube menolak download. Ini karena YouTube melindungi dari automated download dari server cloud.")
                    st.warning("💡 Solusi: Coba gunakan video dari platform lain atau hubungi support.")
                    st.stop()
                else:
                    raise

            video_files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.lower().endswith((".mp4", ".mkv", ".webm"))]
            if not video_files:
                st.error("Gagal mendeteksi berkas video pasca-unduh.")
                st.stop()

            video_path = os.path.join(DOWNLOAD_FOLDER, video_files[0])

            # Inisialisasi Whisper jika diaktifkan
            model = None
            if use_subtitle:
                st.write(f"Inisialisasi Whisper Model [{whisper_model_size}] ke System Memory...")
                try:
                    model = WhisperModel(whisper_model_size, device="cuda", compute_type="int8_float16")
                    st.write("Pemrosesan Bahasa: CUDA GPU diaktifkan.")
                except Exception as e:
                    st.write("Gagal memuat CUDA GPU. Berpindah ke CPU Engine.")
                    model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

            progress_bar = st.progress(0)
            total_clips = len(timestamps)

            # --- LANGKAH 3: LOOPING PROSES PEMOTONGAN KONTEN & SUBTITLING ---
            for i, (start, end) in enumerate(timestamps, start=1):
                st.write(f"Menjalankan konversi Klip ke-{i}: Rentang {start} s/d {end}")
                meta = clips_metadata[i-1] if i-1 < len(clips_metadata) else {"title": f"Clip Short {i} #Shorts", "description": ""}

                clip_name = f"clip_{i:03d}.mp4"
                ass_name = f"clip_{i:03d}.ass"
                final_name = f"clip_{i:03d}_subtitle.mp4"

                output_clip = os.path.join(OUTPUT_FOLDER, clip_name)
                ass_path = os.path.join(SUBTITLE_FOLDER, ass_name)
                final_clip = os.path.join(OUTPUT_FOLDER, final_name)
                duration = get_clip_duration(start, end)

                if mode == "1":
                    make_original_output(video_path, start, duration, output_clip)
                elif mode == "2":
                    make_blur_background_output(video_path, start, duration, output_clip)
                elif mode == "3":
                    make_center_crop_output(video_path, start, duration, output_clip)
                elif mode == "4":
                    make_dynamic_crop_output(video_path, start, duration, i, output_clip)

                if use_subtitle:
                    st.write(f"Whisper mulai mentranskrip Klip ke-{i}...")
                    segments, info = model.transcribe(
                        output_clip,
                        language=WHISPER_LANGUAGE,
                        task="transcribe",
                        beam_size=5,
                        best_of=5,
                        temperature=0,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 500},
                        word_timestamps=True,
                        condition_on_previous_text=False,
                        initial_prompt="Transkripsi rapi bahasa Indonesia."
                    )
                    create_ass_subtitle(list(segments), ass_path)
                    subtitle_files.append(ass_path)

                    if BURN_SUBTITLE_TO_VIDEO:
                        st.write("Menggabungkan hardsub via FFmpeg...")
                        burn_subtitle(output_clip, ass_path, final_clip)
                        video_siap_upload = final_clip
                    else:
                        video_siap_upload = output_clip
                else:
                    video_siap_upload = output_clip

                result_files.append(video_siap_upload)

                # --- LANGKAH 4: TRANSMISI OTOMATIS KE YOUTUBE DATA API ---
                st.write(f"📤 Mengunggah berkas '{os.path.basename(video_siap_upload)}' ke YouTube Shorts...")
                video_id = upload_clip_to_youtube(video_path=video_siap_upload, title=meta["title"], description=meta["description"])
                if video_id:
                    st.success(f"Klip {i} Sukses Terpublikasi! Video ID: {video_id}")
                else:
                    st.warning(f"Klip {i} selesai diproses lokal namun gagal dikirim ke YouTube API.")

                progress_bar.progress(i / total_clips)

            status.update(label="Seluruh rangkaian sistem Selesai!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline otomatisasi mengalami kendala.", state="error", expanded=True)
            st.error(f"Sistem Error: {e}")
            st.stop()

    st.markdown("## Pratinjau Klip Lokal")
    for result_file in result_files:
        st.markdown(f"### {os.path.basename(result_file)}")
        with open(result_file, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        st.download_button(label=f"⬇️ Ambil Dokumen Video ({os.path.basename(result_file)})", data=video_bytes, file_name=os.path.basename(result_file), mime="video/mp4", use_container_width=True)
