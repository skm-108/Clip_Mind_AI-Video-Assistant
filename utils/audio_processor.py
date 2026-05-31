import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


def _valid_ffmpeg_dir(path: str | None) -> str | None:
    if path and os.path.exists(os.path.join(path, "ffmpeg.exe")):
        return path
    return None


def _resolve_ffmpeg_bin() -> str | None:
    env_path = _valid_ffmpeg_dir(os.getenv("FFMPEG_BIN"))
    if env_path:
        return env_path

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return os.path.dirname(path_ffmpeg)

    winget_path = (
        r"C:\Users\ssssh\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-8.1.1-full_build\bin"
    )
    return _valid_ffmpeg_dir(winget_path)


FFMPEG_BIN = _resolve_ffmpeg_bin()
if FFMPEG_BIN:
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    AudioSegment = None

DOWNLOAD_DIR = "generated_audio"
CHUNK_DIR = os.path.join(DOWNLOAD_DIR, "chunks")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

def _configure_pydub_ffmpeg() -> None:
    if not AudioSegment or not FFMPEG_BIN:
        return

    AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")


_configure_pydub_ffmpeg()


def _require_audio_segment():
    if AudioSegment is None:
        raise ModuleNotFoundError(
            "pydub is required for audio processing. Install dependencies with 'pip install -r requirements.txt'."
        )
    return AudioSegment


def ensure_ffmpeg_available() -> None:
    if FFMPEG_BIN or shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "FFmpeg not found. Install FFmpeg or set FFMPEG_BIN in .env to the folder containing ffmpeg.exe."
    )


def _safe_stem(path: str, max_length: int = 48) -> str:
    stem = Path(path).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    stem = stem[:max_length].strip(" .") or "audio"
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{stem}_{digest}"


def _looks_like_pasted_error_text(source: str) -> bool:
    lowered = source.lower()
    return (
        "\n" in source
        or "traceback (most recent call last):" in lowered
        or "modulenotfounderror:" in lowered
        or "filenotfounderror:" in lowered
        or lowered.startswith("error:")
    )


def download_youtube_audio(url: str) -> str:
    try:
        import yt_dlp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yt-dlp is required for YouTube downloads. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    if FFMPEG_BIN:
        ydl_opts["ffmpeg_location"] = FFMPEG_BIN

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    ensure_ffmpeg_available()
    audio_segment = _require_audio_segment()
    output_path = os.path.join(DOWNLOAD_DIR, f"{_safe_stem(input_path)}_converted.wav")
    audio = audio_segment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio_segment = _require_audio_segment()
    audio = audio_segment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    run_id = uuid.uuid4().hex[:8]
    base_name = _safe_stem(wav_path)

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = os.path.join(CHUNK_DIR, f"{base_name}_chunk_{run_id}_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if not source or not source.strip():
        raise ValueError("Input source is empty. Provide a YouTube URL or a local file path.")

    source = source.strip().strip("\"'“”‘’")
    source = source.replace("\u200b", "").replace("\ufeff", "")

    if _looks_like_pasted_error_text(source):
        raise ValueError(
            "The input looks like pasted error text. Paste only a YouTube URL or a local file path."
        )

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        ensure_ffmpeg_available()
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        if source.startswith("file://"):
            source = unquote(urlparse(source).path).lstrip("/")
        source = os.path.expandvars(os.path.expanduser(source))

        try:
            exists = os.path.exists(source)
            is_file = os.path.isfile(source) if exists else False
        except OSError as exc:
            raise ValueError(
                f"Invalid local file path: {source}. Remove surrounding quotes or invalid characters."
            ) from exc

        if not exists:
            raise FileNotFoundError(f"Local file not found: {source}")
        if not is_file:
            raise FileNotFoundError(f"Path is not a file: {source}")
        if os.path.splitext(source)[1].lower() == ".wav":
            wav_path = source
        else:
            print("Converting to WAV...")
            wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
