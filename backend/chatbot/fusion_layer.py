# backend/chatbot/fusion_layer.py
#
# PIPELINE POSITION  (matches your diagram):
#
#   Voice check-in  (STT → NLU → EmotionAnalyzer)
#       ↓  voice_fusion scores + dominant_emotion stored in session
#   Screening + Feature Qs  (PHQ / GAD → condition, feature_answers)
#       ↓
#   ★ THIS FILE — Fusion Layer
#       INPUT  A : feature_answers  — structured scores from text questionnaire
#       INPUT  B : voice_fusion     — float dict from VoiceInputHandler
#                  keys: anxiety, stress, sadness, depression, joy
#       OUTPUT 1 : fused_features   → fed straight into ML Prediction
#       OUTPUT 2 : voice_dominant   → forwarded to CBT-Template Manager
#       OUTPUT 3 : primary_signal   → used by _apply_voice_fusion() to adjust level
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import copy
import math
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN INJECTION METADATA
#  (col_name, col_min, col_max, voice_weight)
#  voice_weight = fraction of full range a signal of 1.0 may shift the column
# ══════════════════════════════════════════════════════════════════════════════

_ANXIETY_COLS: list[tuple[str, float, float, float]] = [
    ("feeling_nervous",                                         0, 3,  0.55),
    ("restlessness",                                            0, 3,  0.45),
    ("uncontrollable_worry",                                    0, 3,  0.35),
    ("GAD_score",                                               0, 21, 0.50),
    ("gad_score",                                               0, 21, 0.50),
    ("total_score",                                             0, 21, 0.50),
]

_STRESS_COLS: list[tuple[str, float, float, float]] = [
    ("Have you recently experienced stress in your life?",      1, 5,  0.60),
    ("Have you been dealing with anxiety or tension recently?", 1, 5,  0.50),
    ("Do you feel overwhelmed with your academic workload?",    1, 5,  0.45),
    ("Do you get irritated easily?",                            1, 5,  0.40),
    ("Have you been feeling sadness or low mood?",              1, 5,  0.35),
    ("stress_level",                                            0, 3,  0.55),
    ("overwhelmed",                                             0, 3,  0.50),
    ("irritability",                                            0, 3,  0.40),
]

_DEPRESSION_COLS: list[tuple[str, float, float, float]] = [
    ("feeling_down",                                            0, 3,  0.60),
    ("loss_of_interest",                                        0, 3,  0.55),
    ("fatigue",                                                 0, 3,  0.40),
    ("feeling_hopeless",                                        0, 3,  0.50),
    ("worthlessness",                                           0, 3,  0.45),
    ("PHQ_score",                                               0, 27, 0.55),
    ("phq_score",                                               0, 27, 0.55),
    ("total_score",                                             0, 27, 0.55),
]

_INJECTION_MAP: dict[str, list[tuple[str, float, float, float]]] = {
    "anxiety":    _ANXIETY_COLS,
    "stress":     _STRESS_COLS,
    "depression": _DEPRESSION_COLS,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL → DELTA  (sigmoid, noise floor 0.30)
# ══════════════════════════════════════════════════════════════════════════════

def _voice_delta(signal: float, col_min: float, col_max: float,
                 voice_weight: float) -> float:
    """
    Convert a voice fusion score [0.0–1.0] to an additive delta on the
    column's native scale.

    - signal < 0.30  →  0.0   (noise floor: ignore weak voice evidence)
    - 0.30 – 1.0     →  sigmoid-shaped ramp up to voice_weight × range
      Sigmoid prevents strong signals from slamming features to ceiling.
    """
    NOISE_FLOOR   = 0.30
    SIGMOID_SCALE = 8.0

    if signal < NOISE_FLOOR:
        return 0.0

    normalised = (signal - NOISE_FLOOR) / (1.0 - NOISE_FLOOR)
    shaped     = 1.0 / (1.0 + math.exp(-SIGMOID_SCALE * (normalised - 0.5)))
    shaped     = (shaped - 0.5) / 0.5          # rescale to [0, 1]
    shaped     = max(0.0, min(1.0, shaped))

    return shaped * voice_weight * (col_max - col_min)


# ══════════════════════════════════════════════════════════════════════════════
#  DOMINANT VOICE EMOTION  (for CBT-Template Manager)
# ══════════════════════════════════════════════════════════════════════════════

def pick_voice_dominant(voice_fusion:        dict[str, float],
                        voice_emotion_label: str | None = None) -> str:
    """
    Selects the dominant distress emotion string for template selection.

    Strategy:
      1. If voice_emotion_label is supplied and not neutral/unknown → use it.
      2. Otherwise derive from highest scoring key in voice_fusion.
    """
    if voice_emotion_label and voice_emotion_label not in ("neutral", "unknown", ""):
        return voice_emotion_label

    THRESHOLD     = 0.10
    DISTRESS_KEYS = ["depression", "anxiety", "stress", "sadness"]
    best, hi      = "neutral", THRESHOLD

    for k in DISTRESS_KEYS:
        v = float(voice_fusion.get(k, 0.0))
        if v > hi:
            best, hi = k, v

    return best


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class FusionLayer:
    """
    Combines feature_answers (text questionnaire) with voice_fusion (audio)
    into one enhanced ML feature vector.

    Never mutates the original feature_answers or session dict.
    """

    @staticmethod
    def fuse(
        condition:           str,
        feature_answers:     dict[str, Any],
        voice_fusion:        dict[str, float] | None = None,
        voice_emotion_label: str | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        condition           : "anxiety" | "stress" | "depression"
        feature_answers     : dict collected by the chat screening flow
        voice_fusion        : dict from VoiceInputHandler.run_pipeline()
                              keys: anxiety, stress, sadness, depression, joy
                              Pass None or {} when no voice data is available.
        voice_emotion_label : dominant emotion from EmotionAnalyzer
                              e.g. "anxious", "stressed", "depressed", "sad"
                              Used for template selection, not for injection.

        Returns
        -------
        dict:
          "features"          : dict   ← pass to predictor.predict()
          "voice_dominant"    : str    ← pass to select_template()
          "primary_signal"    : float  ← echo for _apply_voice_fusion() / logging
          "injection_log"     : list of (col, before, delta, after)
          "voice_fusion_used" : dict   ← echo of voice_fusion input
          "condition"         : str    ← echo of condition input
        """

        # ── Guard: no voice data ──────────────────────────────────────────
        if not voice_fusion:
            print(
                f"[FusionLayer] No voice_fusion for condition='{condition}'. "
                f"Passing feature_answers through unchanged.",
                flush=True,
            )
            return {
                "features":          dict(feature_answers),
                "voice_dominant":    voice_emotion_label or "neutral",
                "primary_signal":    0.0,
                "injection_log":     [],
                "voice_fusion_used": {},
                "condition":         condition,
            }

        # ── Deep copy — never mutate session state ─────────────────────────
        features: dict[str, Any] = copy.deepcopy(feature_answers)

        # ── Dominant voice emotion for CBT-Template Manager ───────────────
        dominant = pick_voice_dominant(voice_fusion, voice_emotion_label)

        # ── Blend voice signals → single primary injection signal ─────────
        #    Each condition uses the voice dimensions that correlate with it.
        if condition == "anxiety":
            primary = float(voice_fusion.get("anxiety", 0.0))

        elif condition == "stress":
            sv      = float(voice_fusion.get("stress",  0.0))
            sad_v   = float(voice_fusion.get("sadness", 0.0))
            # sadness commonly co-presents in stressed users
            primary = min(0.70 * sv + 0.30 * sad_v, 1.0)

        elif condition == "depression":
            dep_v   = float(voice_fusion.get("depression", 0.0))
            sad_v   = float(voice_fusion.get("sadness",    0.0))
            primary = min(0.60 * dep_v + 0.40 * sad_v, 1.0)

        else:
            print(
                f"[FusionLayer] Unknown condition='{condition}' — "
                f"no injection performed.",
                flush=True,
            )
            return {
                "features":          features,
                "voice_dominant":    dominant,
                "primary_signal":    0.0,
                "injection_log":     [],
                "voice_fusion_used": voice_fusion,
                "condition":         condition,
            }

        print(
            f"[FusionLayer] condition='{condition}'  "
            f"primary_signal={primary:.3f}  "
            f"voice_dominant='{dominant}'  "
            f"voice_fusion={voice_fusion}",
            flush=True,
        )

        # ── Inject into eligible columns ──────────────────────────────────
        injection_log: list[tuple[str, Any, float, Any]] = []

        for col_name, col_min, col_max, vw in _INJECTION_MAP.get(condition, []):
            if col_name not in features:
                continue                        # column not collected — skip

            original = features[col_name]
            try:
                orig_f = float(original)
            except (TypeError, ValueError):
                continue                        # non-numeric — skip

            delta = _voice_delta(primary, col_min, col_max, vw)
            if delta == 0.0:
                continue                        # below noise floor

            clamped = max(float(col_min), min(float(col_max), orig_f + delta))

            # Preserve integer dtype when column is a short integer scale
            if isinstance(original, int) or (
                isinstance(original, float)
                and original == int(original)
                and (col_max - col_min) <= 30
            ):
                new_val: Any = int(round(clamped))
            else:
                new_val = round(clamped, 4)

            features[col_name] = new_val
            injection_log.append((col_name, original, round(delta, 4), new_val))

            print(
                f"[FusionLayer]   {col_name}: "
                f"{original} → {new_val}  (Δ={delta:+.4f})",
                flush=True,
            )

        n = len(injection_log)
        print(
            f"[FusionLayer] ✓ {n} column(s) injected for condition='{condition}'."
            if n else
            f"[FusionLayer] primary_signal={primary:.3f} — "
            f"no columns eligible (noise floor or absent).",
            flush=True,
        )

        return {
            "features":          features,
            "voice_dominant":    dominant,
            "primary_signal":    primary,
            "injection_log":     injection_log,
            "voice_fusion_used": voice_fusion,
            "condition":         condition,
        }