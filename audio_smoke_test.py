"""Smoke test for local audio processing.

This script creates a short silent WAV file, reloads it through pydub,
exports a second WAV file, and prints a clear PASS/FAIL message.
"""

from pathlib import Path
import tempfile


def main() -> int:
    print("[audio-smoke-test] starting")

    try:
        from pydub import AudioSegment
    except ModuleNotFoundError as exc:
        print(f"[audio-smoke-test] FAIL: {exc}")
        return 1

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_wav = temp_path / "sample_input.wav"
            exported_wav = temp_path / "sample_export.wav"

            sample = AudioSegment.silent(duration=1000, frame_rate=16000).set_channels(1)
            sample.export(source_wav, format="wav")

            loaded = AudioSegment.from_wav(source_wav)
            loaded.export(exported_wav, format="wav")

            if not source_wav.exists() or not exported_wav.exists():
                print("[audio-smoke-test] FAIL: sample files were not created")
                return 1

            if len(loaded) < 900:
                print(f"[audio-smoke-test] FAIL: unexpected duration {len(loaded)} ms")
                return 1

            print(f"[audio-smoke-test] PASS: created and reloaded {source_wav.name} -> {exported_wav.name}")
            return 0
    except Exception as exc:
        print(f"[audio-smoke-test] FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())