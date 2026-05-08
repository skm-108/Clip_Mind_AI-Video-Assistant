# AI Video Assistant

Streamlit app for turning a YouTube video or local audio file into a transcript, summary, action items, decisions, open questions, and a RAG chat interface.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your own API keys.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Notes

- Local WAV files work without FFmpeg.
- YouTube URLs and non-WAV media require FFmpeg.
- Do not commit `.env`; it contains private API keys.
