# emotion_analyzer.py — FIXED v7
#
# CHANGES vs v6:
# ─────────────────────────────────────────────────────────────────────────────
# DEPRESSION FUSION SCORE (new):
#   v6 computed four fused scores: anxiety, stress, sadness, joy.
#   It upgraded the dominant label to "depressed" when voice_emotion was
#   "depressed" AND fused_sadness > threshold — but it never computed or
#   returned a dedicated depression score in the fusion dict.
#
#   Result: VoiceCheckIn.jsx always showed Depression 0% even when the user
#   explicitly said "I am very depressed", because fusion["depression"] was
#   missing from the API response.
#
#   Fix:
#     1. Compute fused_depression from voice biomarker + sadness_text signal.
#        Voice weight: 1.0 when voice_emotion=="depressed", else 0.
#        Text weight:  sadness_text score (sadness in text correlates with
#                      depression signal).
#     2. Add "depression" key to the returned fusion dict.
#     3. When NLU sadness_boost is applied and intent is distress/help_seeking,
#        also boost fused_depression proportionally.
#     4. sentiment_score for dominant=="depressed" now uses fused_depression.
#
# ALL OTHER LOGIC UNCHANGED from v6.
# ─────────────────────────────────────────────────────────────────────────────

from transformers import pipeline as hf_pipeline


class EmotionAnalyzer:
    """
    Classifies emotion from text + voice biomarkers + (optionally) NLU output.

    Language support:
      English — full text classification + voice fusion + NLU boost.
      Urdu    — Strategy A: translate → classify (best).
                Strategy B: run Urdu text directly through classifier (fallback).
                Strategy C: voice-only if classifier unavailable.

    Output labels:
        "anxious" | "stressed" | "depressed" | "sad" | "excited" | "neutral"
    """

    _classifier  = None
    _translator  = None
    _MODEL_NAME  = "j-hartmann/emotion-english-distilroberta-base"
    _TRANS_MODEL = "Helsinki-NLP/opus-mt-ur-en"

    _THRESHOLDS = {
        "anxious":   0.06,
        "stressed":  0.06,
        "sad":       0.07,
        "depressed": 0.05,
        "excited":   0.15,
    }
    _DEFAULT_THRESHOLD            = 0.08
    _DEPRESSION_SADNESS_THRESHOLD = 0.05

    # ── lazy loaders ─────────────────────────────────────────────────────────

    @classmethod
    def _get_classifier(cls):
        if cls._classifier is None:
            cls._classifier = hf_pipeline(
                "text-classification",
                model=cls._MODEL_NAME,
                top_k=None,
                truncation=True,
            )
        return cls._classifier

    @classmethod
    def _get_translator(cls):
        if cls._translator is None:
            try:
                cls._translator = hf_pipeline(
                    "translation",
                    model=cls._TRANS_MODEL,
                )
                print("[EmotionAnalyzer] Urdu→English translator loaded.", flush=True)
            except Exception as e:
                print(
                    f"[EmotionAnalyzer] Translator unavailable ({e}). "
                    "Will use direct Urdu classification.",
                    flush=True,
                )
                cls._translator = False
        return cls._translator if cls._translator else None

    def __init__(self):
        self.final_emotion_label: str   = "neutral"
        self.sentiment_score:     float = 0.0

    # ── translation helper ────────────────────────────────────────────────────

    @classmethod
    def _translate_urdu_to_english(cls, urdu_text: str) -> str:
        translator = cls._get_translator()
        if not translator:
            return ""
        try:
            result  = translator(urdu_text[:512])
            en_text = result[0].get("translation_text", "").strip()
            print(
                f"[EmotionAnalyzer] Urdu→English: "
                f"'{urdu_text[:60]}' → '{en_text[:60]}'",
                flush=True,
            )
            return en_text
        except Exception as e:
            print(f"[EmotionAnalyzer] Translation failed: {e}", flush=True)
            return ""

    # ── classify_emotion ──────────────────────────────────────────────────────

    def classify_emotion(
        self,
        text:             str,
        biomarker_result: dict = None,
        language:         str  = "en",
        nlu_result:       dict = None,
    ) -> dict:
        """
        Classify emotion from transcript + voice biomarkers + NLU signals.

        Parameters
        ----------
        text             : str       — transcript (may be empty)
        biomarker_result : dict|None — output of VoiceBiomarker.analyze_voice_emotion()
        language         : str       — "en" or "ur"
        nlu_result       : dict|None — output of NLU.analyze() [optional]

        Returns
        -------
        dict:
            final_emotion_label : str
            sentiment_score     : float
            text_scores         : dict
            fusion              : dict  — keys: anxiety, stress, sadness, depression, joy
        """
        if biomarker_result is None:
            biomarker_result = {
                "emotion_from_voice": "neutral",
                "pitch":     0.0,
                "tone":      0.0,
                "mfcc_mean": 0.0,
            }

        voice_emotion = biomarker_result.get("emotion_from_voice", "neutral")

        # ── Step 1: prepare text ──────────────────────────────────────────
        text_for_classification = text.strip()
        text_is_empty           = not text_for_classification
        text_reliability        = "reliable"

        if language == "ur" and text_for_classification:
            translated = self._translate_urdu_to_english(text_for_classification)
            if translated:
                text_for_classification = translated
                print("[EmotionAnalyzer] Strategy A: translated Urdu → English.", flush=True)
            else:
                text_reliability = "degraded"
                print(
                    "[EmotionAnalyzer] Strategy B: running Urdu text directly "
                    "through English classifier (degraded reliability).",
                    flush=True,
                )

        # ── Step 2: text classification ───────────────────────────────────
        if text_is_empty:
            text_scores = {
                "sadness": 0.0, "anger":  0.0, "fear":     0.0,
                "disgust": 0.0, "joy":    0.0, "surprise": 0.0,
                "neutral": 1.0,
            }
            print("[EmotionAnalyzer] Empty text — zero text scores.", flush=True)
        else:
            try:
                classifier  = self._get_classifier()
                raw         = classifier(text_for_classification[:512])
                text_scores = {
                    item["label"].lower(): item["score"]
                    for item in raw[0]
                }
                print(
                    f"[EmotionAnalyzer] Text scores "
                    f"(reliability={text_reliability}): {text_scores}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[EmotionAnalyzer] Classifier failed: {e} — zero scores.",
                    flush=True,
                )
                text_scores = {
                    "sadness": 0.0, "anger":  0.0, "fear":     0.0,
                    "disgust": 0.0, "joy":    0.0, "surprise": 0.0,
                    "neutral": 1.0,
                }
                text_is_empty = True

        # ── Step 3: map text scores → SentiCare emotion space ────────────
        anxiety_text = text_scores.get("fear",    0.0)
        stress_text  = (
            text_scores.get("anger",   0.0)
            + text_scores.get("disgust", 0.0)
        )
        sadness_text = text_scores.get("sadness", 0.0)
        joy_text     = (
            text_scores.get("joy",      0.0)
            + text_scores.get("surprise", 0.0) * 0.3
        )

        # ── Step 4: fusion weights ────────────────────────────────────────
        if text_is_empty:
            voice_weight = 1.0
            text_weight  = 0.0
        elif text_reliability == "degraded":
            voice_weight = 0.65
            text_weight  = 0.35
        elif voice_emotion != "neutral":
            voice_weight = 0.50
            text_weight  = 0.50
        else:
            voice_weight = 0.30
            text_weight  = 0.70

        # ── Step 5: voice map ─────────────────────────────────────────────
        _voice_map = {
            "aroused":   {"anxiety": 0.55, "stress": 0.25, "sadness": 0.0,  "depression": 0.0,  "joy": 0.05},
            "anxious":   {"anxiety": 1.0,  "stress": 0.0,  "sadness": 0.0,  "depression": 0.0,  "joy": 0.0 },
            "stressed":  {"anxiety": 0.0,  "stress": 1.0,  "sadness": 0.0,  "depression": 0.0,  "joy": 0.0 },
            "tense":     {"anxiety": 0.3,  "stress": 0.7,  "sadness": 0.0,  "depression": 0.0,  "joy": 0.0 },
            "sad":       {"anxiety": 0.0,  "stress": 0.0,  "sadness": 1.0,  "depression": 0.3,  "joy": 0.0 },
            # ── KEY FIX: "depressed" voice now drives the depression score ──
            "depressed": {"anxiety": 0.0,  "stress": 0.0,  "sadness": 0.6,  "depression": 1.0,  "joy": 0.0 },
            "neutral":   {"anxiety": 0.0,  "stress": 0.0,  "sadness": 0.0,  "depression": 0.0,  "joy": 0.0 },
        }
        vm = _voice_map.get(voice_emotion, _voice_map["neutral"])

        fused_anxiety    = min(text_weight * anxiety_text + voice_weight * vm["anxiety"],    1.0)
        fused_stress     = min(text_weight * stress_text  + voice_weight * vm["stress"],     1.0)
        fused_sadness    = min(text_weight * sadness_text + voice_weight * vm["sadness"],    1.0)
        fused_joy        = min(text_weight * joy_text     + voice_weight * vm["joy"],        1.0)
        # ── NEW: depression score fused from voice + sadness text signal ──
        # sadness_text contributes because textual sadness correlates with
        # depression; the main driver is the voice biomarker "depressed" label.
        fused_depression = min(
            text_weight * sadness_text * 0.5 + voice_weight * vm["depression"],
            1.0
        )

        # ── Step 6: NLU boost ─────────────────────────────────────────────
        if nlu_result:
            intent          = nlu_result.get("intent",        "neutral")
            anxiety_boost   = nlu_result.get("anxiety_boost", 0.0)
            stress_boost    = nlu_result.get("stress_boost",  0.0)
            sadness_boost   = nlu_result.get("sadness_boost", 0.0)
            negation_found  = nlu_result.get("negation_found", False)

            print(
                f"[EmotionAnalyzer] NLU boost → "
                f"intent={intent}  anxiety+={anxiety_boost}  "
                f"stress+={stress_boost}  sadness+={sadness_boost}  "
                f"negation={negation_found}",
                flush=True,
            )

            if intent == "denial":
                fused_anxiety    *= 0.5
                fused_stress     *= 0.5
                fused_sadness    *= 0.5
                fused_depression *= 0.5
                print(
                    "[EmotionAnalyzer] NLU denial intent → "
                    "all distress scores halved.",
                    flush=True,
                )

            elif intent in ("distress", "help_seeking"):
                fused_anxiety    = min(fused_anxiety + anxiety_boost, 1.0)
                fused_stress     = min(fused_stress  + stress_boost,  1.0)
                fused_sadness    = min(fused_sadness + sadness_boost,  1.0)
                # Sadness boost also lifts depression — they share the same
                # NLU signal (clinical sadness keywords).
                fused_depression = min(fused_depression + sadness_boost * 0.5, 1.0)
                fused_joy        = min(fused_joy, 0.05)

                if intent == "help_seeking":
                    fused_anxiety    = min(fused_anxiety + 0.05, 1.0)
                    fused_sadness    = min(fused_sadness + 0.05, 1.0)
                    fused_depression = min(fused_depression + 0.05, 1.0)

        fusion = {
            "anxiety":    round(fused_anxiety,    3),
            "stress":     round(fused_stress,     3),
            "sadness":    round(fused_sadness,    3),
            "depression": round(fused_depression, 3),   # ← NEW key
            "joy":        round(fused_joy,        3),
        }

        print(
            f"[EmotionAnalyzer] Fusion (after NLU) → "
            f"anxiety={fused_anxiety:.3f}  stress={fused_stress:.3f}  "
            f"sadness={fused_sadness:.3f}  depression={fused_depression:.3f}  "
            f"joy={fused_joy:.3f}  "
            f"(voice={voice_emotion}  lang={language}  "
            f"text_empty={text_is_empty}  text_rel={text_reliability}  "
            f"voice_w={voice_weight:.2f}  text_w={text_weight:.2f})",
            flush=True,
        )

        # ── Step 7: pick dominant emotion ─────────────────────────────────
        scores_map = {
            "anxious":   fused_anxiety,
            "stressed":  fused_stress,
            "sad":       fused_sadness,
            "depressed": fused_depression,   # ← now participates in dominant selection
            "excited":   fused_joy,
        }

        dominant  = max(scores_map, key=scores_map.get)
        top_score = scores_map[dominant]
        threshold = self._THRESHOLDS.get(dominant, self._DEFAULT_THRESHOLD)

        if top_score < threshold:
            dominant = "neutral"
        else:
            if (
                dominant == "anxious"
                and abs(fused_anxiety - fused_stress) < 0.02
                and fused_stress >= self._THRESHOLDS["stressed"]
            ):
                dominant = "stressed"
                print(
                    f"[EmotionAnalyzer] Tie-break: anxiety≈stress "
                    f"({fused_anxiety:.3f}≈{fused_stress:.3f}) → 'stressed'",
                    flush=True,
                )

        # ── Step 8: depression upgrade ────────────────────────────────────
        # Kept as safety net: if voice biomarker says "depressed" and there
        # is meaningful sadness/depression signal, always resolve to depressed.
        if (
            voice_emotion == "depressed"
            and fused_sadness > self._DEPRESSION_SADNESS_THRESHOLD
        ):
            dominant = "depressed"

        # ── Step 9: voice-only aroused safety net ─────────────────────────
        if text_is_empty and voice_emotion == "aroused" and dominant == "excited":
            dominant = "anxious"

        self.final_emotion_label = dominant
        self.sentiment_score     = scores_map.get(dominant, fused_sadness)

        print(
            f"[EmotionAnalyzer] dominant='{dominant}'  "
            f"score={self.sentiment_score:.3f}",
            flush=True,
        )

        return {
            "final_emotion_label": self.final_emotion_label,
            "sentiment_score":     self.sentiment_score,
            "text_scores":         text_scores,
            "fusion":              fusion,
        }