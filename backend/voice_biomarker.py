# voice_biomarker.py — FIXED v9
#
# BUGS FIXED vs v8:
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL — IndentationError in v8 crashed Flask on import.
#   v8 had broken indentation in the heuristic block (elif at 2 spaces,
#   orphaned comment mid-block). Python raised IndentationError before any
#   class loaded, so Flask silently kept running the old .pyc cache.
#   The old cache still had pitch>180/tone>0.015 thresholds → "aroused"
#   fired for all Urdu speech → always returned "excited".
#   FIX: Entire analyze_voice_emotion() rewritten with clean, consistent
#   4-space indentation.
#
# THRESHOLDS RAISED — "aroused" no longer fires for normal Urdu speech.
#   Urdu speakers naturally have higher prosodic energy than the old
#   thresholds assumed.
#     old: pitch > 180 AND tone > 0.015
#     new: pitch > 240 AND tone > 0.032
#   Only genuine high-activation states (panic, crying, shouting) reach
#   "aroused" now. Normal expressive Urdu speech falls through to
#   "stressed", "tense", or "sad" as appropriate.
#
# NEW "anxious" ACOUSTIC RULE — intermediate range between aroused and tense.
#   pitch 190–240 AND tone 0.018–0.032 → "anxious"
#   This was the zone where old code fired "aroused" for worried/nervous
#   Urdu speakers. Now correctly labelled.
#
# SHORT CLIP GUARD — clips < 0.5 s give unreliable pitches from piptrack.
#   If audio is shorter than MIN_SAMPLES_FOR_PITCH, pitch is forced to 0
#   and the tone-only path runs instead.
#
# DEPRESSION THRESHOLD tightened — mfcc_mean < -8.0 (was -5.0).
#   Old threshold caught normal quiet speech. -8.0 requires genuinely
#   flat, energy-less spectral content.
#
# ALL OTHER CODE IDENTICAL TO v8.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# Below this length, piptrack produces unreliable pitch — skip it.
MIN_SAMPLES_FOR_PITCH = 8000   # 0.5 s @ 16 kHz


class VoiceBiomarker:
    """
    Extracts acoustic biomarkers (MFCC, pitch, RMS energy) from audio,
    then maps them to a rough emotion estimate via rule-based heuristics.
    Language-agnostic — works for both English and Urdu speech.

    Possible emotion_from_voice values:
        "aroused" | "anxious" | "stressed" | "depressed" | "sad" | "tense" | "neutral"
    """

    SILENCE_TONE_THRESHOLD = 0.0001

    def __init__(self):
        self.mfcc_features:      object = None   # np.ndarray shape (n_mfcc,)
        self.pitch:              float  = 0.0
        self.tone:               float  = 0.0
        self.emotion_from_voice: str    = "neutral"

    # ── _load_audio ──────────────────────────────────────────────────────────
    @staticmethod
    def _load_audio(audio_path: str, sr: int = 16000):
        """
        Load audio → (numpy float32 array, sample_rate).
        Tries soundfile first (reliable on Windows PCM WAV), then librosa.
        """
        try:
            import soundfile as sf
            data, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if file_sr != sr:
                import librosa
                data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
            rms_val = float(np.sqrt(np.mean(data ** 2)))
            print(
                f"[VoiceBiomarker] loaded via soundfile: "
                f"{len(data)} samples @ {sr} Hz  rms={rms_val:.6f}",
                flush=True,
            )
            return data, sr
        except Exception as sf_err:
            print(
                f"[VoiceBiomarker] soundfile failed ({sf_err}) — trying librosa.",
                flush=True,
            )

        import librosa
        data, _ = librosa.load(audio_path, sr=sr, mono=True)
        rms_val = float(np.sqrt(np.mean(data ** 2)))
        print(
            f"[VoiceBiomarker] loaded via librosa: "
            f"{len(data)} samples @ {sr} Hz  rms={rms_val:.6f}",
            flush=True,
        )
        return data, sr

    # ── extract_mfcc ─────────────────────────────────────────────────────────
    def extract_mfcc(self, audio_path: str, n_mfcc: int = 40) -> np.ndarray:
        """
        Load audio → compute MFCCs, pitch (piptrack), RMS energy.
        Returns np.ndarray of shape (n_mfcc,) — mean MFCC across time.
        """
        import librosa

        y, sr = self._load_audio(audio_path)

        # ── MFCCs ─────────────────────────────────────────────────────────
        mfcc_matrix        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        self.mfcc_features = np.mean(mfcc_matrix, axis=1)

        # ── Pitch via piptrack ─────────────────────────────────────────────
        # Short-clip guard: skip piptrack if audio is too brief.
        if len(y) < MIN_SAMPLES_FOR_PITCH:
            self.pitch = 0.0
            print(
                f"[VoiceBiomarker] Clip too short ({len(y)} samples < "
                f"{MIN_SAMPLES_FOR_PITCH}) — skipping piptrack, pitch=0.",
                flush=True,
            )
        else:
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                nonzero_mag = magnitudes[magnitudes > 0]
                if len(nonzero_mag) > 0:
                    abs_threshold = np.percentile(nonzero_mag, 75)
                    mask          = magnitudes > abs_threshold
                    pitch_values  = pitches[mask]
                    # Filter to human voice range — covers English and Urdu
                    pitch_values  = pitch_values[
                        (pitch_values > 80) & (pitch_values < 500)
                    ]
                    self.pitch = (
                        float(np.median(pitch_values))
                        if len(pitch_values) > 0
                        else 0.0
                    )
                else:
                    self.pitch = 0.0
            except Exception as e:
                print(f"[VoiceBiomarker] piptrack failed ({e}), using pitch=0.", flush=True)
                self.pitch = 0.0

        # ── RMS energy ────────────────────────────────────────────────────
        rms       = librosa.feature.rms(y=y)
        self.tone = float(np.mean(rms))

        print(
            f"[VoiceBiomarker] Features → "
            f"pitch={self.pitch:.1f} Hz  "
            f"tone={self.tone:.6f}  "
            f"mfcc_mean={float(np.mean(self.mfcc_features)):.3f}",
            flush=True,
        )
        return self.mfcc_features

    # ── analyze_voice_emotion ─────────────────────────────────────────────────
    def analyze_voice_emotion(self, mfcc_features: np.ndarray = None) -> dict:
        """
        Rule-based biomarker → emotion mapping.
        Thresholds tuned for desk/laptop microphones including Urdu speakers.

        Rule order (highest priority first):
          1. Silence guard  — pitch=0 AND tone near-zero
          2. aroused        — pitch > 240 AND tone > 0.032  (panic / extreme activation)
          3. anxious        — pitch 190–240 AND tone 0.018–0.032  (worried / nervous)
          4. stressed       — tone > 0.018  (high energy, any pitch)
          5. depressed      — pitch < 110 AND tone < 0.006 AND mfcc_mean < -8.0
          6. sad            — 0 < pitch < 130 AND tone < 0.008
          7. tense          — tone > 0.012 OR any voiced pitch
          8. neutral fallback

        Returns dict: emotion_from_voice, pitch, tone, mfcc_mean
        """
        features = mfcc_features if mfcc_features is not None else self.mfcc_features
        if features is None:
            raise ValueError("Call extract_mfcc() first, or pass mfcc_features.")

        mfcc_mean = float(np.mean(features))

        print(
            f"[VoiceBiomarker] Thresholds → "
            f"pitch={self.pitch:.1f} Hz  tone={self.tone:.6f}  mfcc_mean={mfcc_mean:.3f}  |  "
            f"aroused: pitch>240 & tone>0.032  "
            f"anxious: 190<pitch<240 & 0.018<tone<0.032  "
            f"stressed: tone>0.018  "
            f"depressed: pitch<110 & tone<0.006 & mfcc_mean<-8  "
            f"sad: 0<pitch<130 & tone<0.008",
            flush=True,
        )

        # ── 1. Silence guard ──────────────────────────────────────────────
        if self.pitch == 0.0 and self.tone < self.SILENCE_TONE_THRESHOLD:
            self.emotion_from_voice = "neutral"
            print("[VoiceBiomarker] Truly silent audio → 'neutral'", flush=True)
            return {
                "emotion_from_voice": "neutral",
                "pitch":     0.0,
                "tone":      0.0,
                "mfcc_mean": mfcc_mean,
            }

        # ── 2. Aroused ────────────────────────────────────────────────────
        # Only genuine extreme activation: shouting, panic, crying.
        # Normal expressive Urdu speech does NOT reach these thresholds.
        if self.pitch > 240 and self.tone > 0.032:
            emotion = "aroused"

        # ── 3. Anxious ────────────────────────────────────────────────────
        # Intermediate zone: elevated pitch + moderate energy.
        # Covers worried / nervous speech in both English and Urdu.
        elif 190 <= self.pitch <= 240 and 0.018 <= self.tone <= 0.032:
            emotion = "anxious"

        # ── 4. Stressed ───────────────────────────────────────────────────
        # High energy regardless of pitch — loud, pressured speech.
        elif self.tone > 0.018:
            emotion = "stressed"

        # ── 5. Depressed ──────────────────────────────────────────────────
        # Very low pitch + very low energy + spectrally flat MFCCs.
        # All three conditions required to avoid catching normal quiet speech.
        elif self.pitch < 110 and self.tone < 0.006 and mfcc_mean < -8.0:
            emotion = "depressed"

        # ── 6. Sad ────────────────────────────────────────────────────────
        # Low pitch + low energy without full depression profile.
        elif (self.pitch > 0) and (self.pitch < 130) and (self.tone < 0.008):
            emotion = "sad"

        # ── 7. Tense ──────────────────────────────────────────────────────
        # Moderate energy or any voiced speech that didn't match above.
        elif self.tone > 0.012:
            emotion = "tense"

        elif self.pitch > 0:
            emotion = "tense"

        # ── 8. Neutral fallback ───────────────────────────────────────────
        else:
            if self.tone > self.SILENCE_TONE_THRESHOLD:
                # piptrack found no pitch but person was speaking.
                # Return "tense" (conservative) so EmotionAnalyzer gets
                # a real voice signal instead of defaulting to neutral.
                emotion = "tense"
                print(
                    f"[VoiceBiomarker] pitch=0 but tone={self.tone:.6f} > silence_thresh "
                    f"→ 'tense' (piptrack likely failed on short/quiet clip)",
                    flush=True,
                )
            else:
                emotion = "neutral"

        self.emotion_from_voice = emotion
        print(f"[VoiceBiomarker] emotion_from_voice='{emotion}'", flush=True)

        return {
            "emotion_from_voice": emotion,
            "pitch":     self.pitch,
            "tone":      self.tone,
            "mfcc_mean": mfcc_mean,
        }