import os
from pathlib import Path

from dotenv import load_dotenv


RUNTIME_KEYS = (
    "MISTRAL_API_KEY",
    "SARVAM_API_KEY",
    "FFMPEG_BIN",
    "WHISPER_MODEL",
    "SARVAM_STT_MODEL",
)


def configure_runtime_environment() -> None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

    try:
        import streamlit as st
    except Exception:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    for key in RUNTIME_KEYS:
        if os.getenv(key):
            continue
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)