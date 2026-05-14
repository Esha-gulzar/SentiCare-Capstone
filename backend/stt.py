# stt.py — FINAL CORRECTED (unchanged logic, added MarianMT install note)
#
# NOTE ON MARIAN MT:
# ─────────────────────────────────────────────────────────────────────────────
# If MarianMT is not installed, EmotionAnalyzer falls back to running Urdu
# text directly through the English classifier (Strategy B in v5).
# For best Urdu emotion accuracy, install the translation model:
#
#   pip install transformers sentencepiece sacremoses
#   python -c "from transformers import pipeline; \
#              pipeline('translation', model='Helsinki-NLP/opus-mt-ur-en')"
#
# This downloads ~300 MB once. After that, Urdu text is translated to English
# before emotion classification, giving much better accuracy.
#
# Also: change MODEL_SIZE from "medium" to "small" if you want faster STT
# at the cost of slightly lower Urdu accuracy.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import whisper

_SILENCE_TOKENS = {
    "\u2026",
    "...",
    ". . .",
    "[ Silence ]",
    "[silence]",
    "(silence)",
    "[BLANK_AUDIO]",
    "(BLANK_AUDIO)",
    "\u062e\u0627\u0645\u0648\u0634",   # خاموش
}

_MIN_CONFIDENCE = -0.7


class STT:
    _model = None

    @staticmethod
    def _get_model():
        if STT._model is None:
            MODEL_SIZE = "medium"
            print(f"[STT] Loading Whisper '{MODEL_SIZE}' model…", flush=True)
            STT._model = whisper.load_model(MODEL_SIZE)
            print("[STT] Whisper model ready.", flush=True)
        return STT._model

    get_model = _get_model

    def __init__(self):
        self.transcript:     str   = ""
        self.stt_confidence: float = 0.0

    def convert_to_text(self, audio_path: str, language: str = "en") -> dict:
        """
        Transcribe audio_path with Whisper.
        Always pass language explicitly — never None.
        """
        import os
        if not audio_path or not os.path.exists(audio_path):
            return {
                "transcript": "",
                "confidence": 0.0,
                "language":   language,
                "error":      f"Audio file not found: {audio_path}",
            }

        model   = self._get_model()
        options = {"fp16": False, "language": language}

        print(f"[STT] Transcribing with language='{language}'…", flush=True)

        try:
            result         = model.transcribe(audio_path, **options)
            raw_transcript = (result.get("text") or "").strip()

            if raw_transcript in _SILENCE_TOKENS:
                print(f"[STT] Silence placeholder '{raw_transcript}' → empty.", flush=True)
                raw_transcript = ""

            segments = result.get("segments", [])
            if segments:
                probs = [s.get("avg_logprob", 0.0) for s in segments]
                self.stt_confidence = sum(probs) / len(probs)
            else:
                self.stt_confidence = 0.0

            if self.stt_confidence < _MIN_CONFIDENCE and raw_transcript:
                print(
                    f"[STT] Low confidence ({self.stt_confidence:.3f}) — "
                    "clearing hallucinated transcript.",
                    flush=True,
                )
                raw_transcript = ""

            self.transcript = raw_transcript
            detected_lang   = result.get("language") or language

            print(
                f"[STT] Result → transcript='{self.transcript[:80]}'  "
                f"lang={detected_lang}  conf={self.stt_confidence:.3f}",
                flush=True,
            )

            return {
                "transcript": self.transcript,
                "confidence": self.stt_confidence,
                "language":   detected_lang,
            }

        except Exception as exc:
            logging.exception(f"[STT] Transcription failed: {exc}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "language":   language,
                "error":      str(exc),
            }