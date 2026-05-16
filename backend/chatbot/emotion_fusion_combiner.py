# backend/chatbot/emotion_fusion_combiner.py
#
# WHAT THIS FILE DOES (read this first):
# ─────────────────────────────────────────────────────────────────────────────
# Your current pipeline has TWO separate emotion signals:
#
#   SIGNAL A — Text/ML:
#     screening_answers → route_condition() → condition (str)
#     feature_answers   → predictor.predict() → level int (0/1/2)
#
#   SIGNAL B — Voice:
#     audio → STT → NLU → VoiceBiomarker → EmotionAnalyzer
#     → voice_fusion dict  {anxiety, stress, sadness, depression, joy}
#     → dominant_emotion   str
#
# RIGHT NOW these are combined in two broken ways:
#   1. FusionLayer injects voice scores into feature columns BEFORE prediction.
#      This only works if the ML model's columns exactly match the injected keys.
#      For most users those keys are ABSENT → injection silently does nothing.
#   2. _apply_voice_fusion() bumps the level AFTER prediction using thresholds.
#      This is a hard-coded rule, not a real fusion.
#
# THIS FILE replaces both with a proper LATE FUSION:
#
#   Text signal  → (condition_text, level_text, confidence_text)
#   Voice signal → (condition_voice, level_voice, confidence_voice)
#   ─────────────────────────────────────────────────────────────────
#   CombinedResult  → (condition_final, level_final, explanation)
#
# HOW TO INTEGRATE (3 changes in app.py):
#
#   1. Import:
#       from backend.chatbot.emotion_fusion_combiner import EmotionFusionCombiner
#
#   2. After ML prediction produces `prediction` (int 0/1/2) and
#      `level` (str), BEFORE calling build_cbt_message(), add:
#
#       combined = EmotionFusionCombiner.combine(
#           condition_text   = condition,
#           level_int_text   = prediction,          # raw int from predictor
#           voice_fusion     = voice_fusion,         # dict from session
#           voice_dominant   = voice_dominant,       # str from session
#           text_confidence  = 0.70,                 # trust ML at 70%
#       )
#       condition      = combined["condition"]
#       level          = combined["level"]
#       voice_dominant = combined["voice_dominant"]
#
#   3. Remove the _apply_voice_fusion() call below that — it is now redundant.
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# How voice_fusion scores map to a 0/1/2 severity level.
# These thresholds were tuned to match the ML model's output distribution.
_VOICE_LEVEL_THRESHOLDS = {
    "low":    (0.00, 0.35),   # [0.00, 0.35)
    "medium": (0.35, 0.65),   # [0.35, 0.65)
    "high":   (0.65, 1.01),   # [0.65, 1.00]
}

_LEVEL_TO_INT = {"low": 0, "medium": 1, "high": 2}
_INT_TO_LEVEL = {0: "low",  1: "medium", 2: "high"}

# Which voice_fusion key to read for each text-predicted condition.
# Depression uses a blend of depression + sadness because the EmotionAnalyzer
# depression score is often lower than the sadness score for mild cases.
_VOICE_KEY_FOR_CONDITION: dict[str, list[tuple[str, float]]] = {
    "anxiety":    [("anxiety",    1.00)],
    "stress":     [("stress",     0.70), ("sadness", 0.30)],
    "depression": [("depression", 0.60), ("sadness", 0.40)],
}

# Map dominant voice emotion string → condition string.
# Mirrors _VOICE_TO_CONDITION in template_selector.py — kept in sync manually.
_DOMINANT_TO_CONDITION: dict[str, str] = {
    "anxious":    "anxiety",
    "anxiety":    "anxiety",
    "stressed":   "stress",
    "stress":     "stress",
    "tense":      "stress",
    "depressed":  "depression",
    "depression": "depression",
    "sad":        "depression",
    "sadness":    "depression",
    "aroused":    "anxiety",
    "neutral":    "",
}

# Minimum voice signal to consider the voice channel reliable.
# Below this the voice signal is so weak it is treated as noise.
_VOICE_RELIABILITY_THRESHOLD = 0.20

# Minimum gap between two conditions' voice scores to call a clear voice winner.
# If they are within this margin, the text prediction wins the tie.
_VOICE_MARGIN = 0.12

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _voice_score_for_condition(condition: str,
                                voice_fusion: dict[str, float]) -> float:
    """
    Return a single blended voice score [0, 1] for the given condition.
    Uses the weighted blend defined in _VOICE_KEY_FOR_CONDITION.
    """
    weights = _VOICE_KEY_FOR_CONDITION.get(condition, [])
    if not weights:
        return 0.0
    total = 0.0
    for key, w in weights:
        total += w * float(voice_fusion.get(key, 0.0))
    return min(total, 1.0)


def _voice_score_to_level(score: float) -> str:
    """Map a [0, 1] voice score to low / medium / high."""
    for label, (lo, hi) in _VOICE_LEVEL_THRESHOLDS.items():
        if lo <= score < hi:
            return label
    return "high"   # score == 1.0 edge case


def _best_voice_condition(voice_fusion: dict[str, float]) -> tuple[str, float]:
    """
    Return (condition, score) for whichever condition has the highest
    blended voice score.  Returns ("", 0.0) if nothing clears the threshold.
    """
    best_cond  = ""
    best_score = _VOICE_RELIABILITY_THRESHOLD

    for cond in ("anxiety", "stress", "depression"):
        score = _voice_score_for_condition(cond, voice_fusion)
        if score > best_score:
            best_score = score
            best_cond  = cond

    return best_cond, best_score


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION FUSION COMBINER
# ══════════════════════════════════════════════════════════════════════════════

class EmotionFusionCombiner:
    """
    Late-fusion combiner that merges text ML prediction with voice emotion
    signals to produce a single (condition, level) pair for CBT template
    selection.

    Design principles:
    ──────────────────
    • Text ML is the PRIMARY signal — it saw the full structured questionnaire.
      Default weight 0.70.
    • Voice is the SECONDARY signal — it captures real-time affect but is
      noisy (mic quality, ambient noise, language).
      Default weight 0.30.
    • If voice_fusion is empty / all zeros → text prediction passes through
      unchanged (voice channel is absent, not wrong).
    • If voice and text AGREE on condition → level is a weighted blend.
    • If voice and text DISAGREE on condition:
        - Voice score must exceed a higher confidence bar (0.65) to override
          the text-predicted condition.
        - Otherwise text condition is kept but level may be adjusted upward.
    • Result includes a human-readable explanation for every decision
      so you can log it and understand what happened.
    """

    @staticmethod
    def combine(
        condition_text:  str,
        level_int_text:  int,                  # 0=low 1=medium 2=high from predictor
        voice_fusion:    dict[str, float] | None = None,
        voice_dominant:  str = "neutral",
        text_confidence: float = 0.70,         # weight for text ML channel
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        condition_text   : ML-routed condition  ("anxiety"|"stress"|"depression")
        level_int_text   : ML prediction integer (0, 1, or 2)
        voice_fusion     : dict from VoiceInputHandler  {anxiety, stress, sadness,
                           depression, joy} — pass None if no voice data
        voice_dominant   : dominant emotion label from EmotionAnalyzer
        text_confidence  : weight for text channel; voice gets (1 - text_confidence)

        Returns
        -------
        dict:
            condition      str    — final condition for CBT lookup
            level          str    — "low" | "medium" | "high"
            level_int      int    — 0 / 1 / 2
            voice_dominant str    — updated dominant for template selector
            text_signal    dict   — echoed text inputs
            voice_signal   dict   — echoed voice inputs
            explanation    str    — human-readable decision trace
            fusion_mode    str    — "text_only" | "agree" | "disagree_text_wins"
                                    | "disagree_voice_override" | "voice_adjusts_level"
        """
        voice_confidence = 1.0 - text_confidence

        # ── Build text signal dict ────────────────────────────────────────
        level_text = _INT_TO_LEVEL.get(level_int_text, "medium")
        text_signal = {
            "condition":  condition_text,
            "level":      level_text,
            "level_int":  level_int_text,
            "confidence": text_confidence,
        }

        # ── Guard: no voice data ──────────────────────────────────────────
        if not voice_fusion or all(v == 0.0 for v in voice_fusion.values()):
            explanation = (
                f"Voice channel absent or silent. "
                f"Text prediction passes through: "
                f"condition='{condition_text}' level='{level_text}'."
            )
            print(f"[EmotionFusionCombiner] TEXT_ONLY — {explanation}", flush=True)
            return {
                "condition":      condition_text,
                "level":          level_text,
                "level_int":      level_int_text,
                "voice_dominant": voice_dominant,
                "text_signal":    text_signal,
                "voice_signal":   {},
                "explanation":    explanation,
                "fusion_mode":    "text_only",
            }

        # ── Voice scores for EVERY condition ──────────────────────────────
        v_score_for_text = _voice_score_for_condition(condition_text, voice_fusion)
        v_level_for_text = _voice_score_to_level(v_score_for_text)
        v_int_for_text   = _LEVEL_TO_INT[v_level_for_text]

        voice_best_cond, voice_best_score = _best_voice_condition(voice_fusion)
        voice_best_level = _voice_score_to_level(voice_best_score) if voice_best_cond else level_text

        voice_signal = {
            "condition":       voice_best_cond or condition_text,
            "score_for_text":  round(v_score_for_text, 3),
            "level_for_text":  v_level_for_text,
            "best_condition":  voice_best_cond,
            "best_score":      round(voice_best_score, 3),
            "best_level":      voice_best_level,
            "dominant_label":  voice_dominant,
            "confidence":      round(voice_confidence, 2),
        }

        # ── Determine final voice_dominant ────────────────────────────────
        # If voice best condition differs from text, update the dominant label.
        if voice_best_cond and voice_best_cond != condition_text and voice_best_score >= 0.65:
            resolved_dominant = voice_dominant if _DOMINANT_TO_CONDITION.get(
                voice_dominant, "") == voice_best_cond else voice_best_cond
        else:
            resolved_dominant = voice_dominant

        # ══════════════════════════════════════════════════════════════════
        #  CASE 1: CONDITIONS AGREE
        # ══════════════════════════════════════════════════════════════════
        if not voice_best_cond or voice_best_cond == condition_text:
            # Voice agrees — blend the level scores.
            # Use voice score for the text condition (already computed above).

            if v_score_for_text < _VOICE_RELIABILITY_THRESHOLD:
                # Voice agrees on condition but score is too weak to influence level.
                blended_int   = level_int_text
                blended_level = level_text
                mode          = "agree"
                explanation   = (
                    f"Voice agrees: condition='{condition_text}'. "
                    f"Voice score={v_score_for_text:.2f} below reliability threshold "
                    f"({_VOICE_RELIABILITY_THRESHOLD}). "
                    f"Text level='{level_text}' kept."
                )
            else:
                # Weighted blend of text level_int and voice level_int.
                blended_float = (
                    text_confidence  * level_int_text
                    + voice_confidence * v_int_for_text
                )
                blended_int   = min(2, round(blended_float))
                blended_level = _INT_TO_LEVEL[blended_int]
                mode          = "agree"
                explanation   = (
                    f"Voice agrees: condition='{condition_text}'. "
                    f"Text level='{level_text}'({level_int_text}) × {text_confidence:.0%} + "
                    f"voice level='{v_level_for_text}'({v_int_for_text}) × {voice_confidence:.0%} "
                    f"→ blended={blended_float:.2f} → '{blended_level}'."
                )

            print(
                f"[EmotionFusionCombiner] AGREE — {explanation}",
                flush=True,
            )
            return {
                "condition":      condition_text,
                "level":          blended_level,
                "level_int":      blended_int,
                "voice_dominant": resolved_dominant,
                "text_signal":    text_signal,
                "voice_signal":   voice_signal,
                "explanation":    explanation,
                "fusion_mode":    mode,
            }

        # ══════════════════════════════════════════════════════════════════
        #  CASE 2: CONDITIONS DISAGREE
        # ══════════════════════════════════════════════════════════════════
        # voice_best_cond != condition_text AND voice_best_score > threshold

        voice_margin = voice_best_score - v_score_for_text

        # SUB-CASE 2A: Voice is confident AND clearly different from text.
        # Voice must clear 0.65 AND beat text condition by _VOICE_MARGIN.
        if voice_best_score >= 0.65 and voice_margin >= _VOICE_MARGIN:
            # Override condition; use voice level for the new condition.
            final_condition  = voice_best_cond
            final_level      = voice_best_level
            final_level_int  = _LEVEL_TO_INT[final_level]
            mode             = "disagree_voice_override"
            explanation      = (
                f"Voice OVERRIDES text. "
                f"Text: condition='{condition_text}' level='{level_text}'. "
                f"Voice: best_condition='{voice_best_cond}' "
                f"score={voice_best_score:.2f} (≥0.65) "
                f"margin={voice_margin:.2f} (≥{_VOICE_MARGIN}). "
                f"Final: condition='{final_condition}' level='{final_level}'."
            )
            print(
                f"[EmotionFusionCombiner] VOICE_OVERRIDE — {explanation}",
                flush=True,
            )
            return {
                "condition":      final_condition,
                "level":          final_level,
                "level_int":      final_level_int,
                "voice_dominant": resolved_dominant,
                "text_signal":    text_signal,
                "voice_signal":   voice_signal,
                "explanation":    explanation,
                "fusion_mode":    mode,
            }

        # SUB-CASE 2B: Voice disagrees but not confident enough to override.
        # Keep text condition. Adjust level upward if voice signal is elevated.
        # This replaces _apply_voice_fusion() in app.py.

        if v_score_for_text >= _VOICE_RELIABILITY_THRESHOLD:
            # Partial adjustment — shift text level toward voice level
            # using a conservative weight (half the voice_confidence).
            partial_weight = voice_confidence * 0.5
            blended_float  = (
                (1.0 - partial_weight) * level_int_text
                + partial_weight * v_int_for_text
            )
            adjusted_int   = min(2, round(blended_float))
            adjusted_level = _INT_TO_LEVEL[adjusted_int]
            mode           = "voice_adjusts_level"
            explanation    = (
                f"Voice DISAGREES with text but not confident enough to override. "
                f"Text: condition='{condition_text}' level='{level_text}'. "
                f"Voice best: '{voice_best_cond}' score={voice_best_score:.2f} "
                f"(margin={voice_margin:.2f} < {_VOICE_MARGIN} or score < 0.65). "
                f"Voice score for text condition={v_score_for_text:.2f}. "
                f"Level partial-adjusted: '{level_text}' → '{adjusted_level}'."
            )
            final_level     = adjusted_level
            final_level_int = adjusted_int
        else:
            # Voice too weak — pure text pass-through.
            final_level     = level_text
            final_level_int = level_int_text
            mode            = "disagree_text_wins"
            explanation     = (
                f"Voice DISAGREES with text and score too low to adjust. "
                f"Text: condition='{condition_text}' level='{level_text}'. "
                f"Voice best: '{voice_best_cond}' score={voice_best_score:.2f} "
                f"< {_VOICE_RELIABILITY_THRESHOLD}. "
                f"Text prediction kept unchanged."
            )

        print(
            f"[EmotionFusionCombiner] {mode.upper()} — {explanation}",
            flush=True,
        )
        return {
            "condition":      condition_text,
            "level":          final_level,
            "level_int":      final_level_int,
            "voice_dominant": resolved_dominant,
            "text_signal":    text_signal,
            "voice_signal":   voice_signal,
            "explanation":    explanation,
            "fusion_mode":    mode,
        }