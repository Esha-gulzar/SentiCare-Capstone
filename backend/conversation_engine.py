# conversation_engine_voice_fusion_patch.py
#
# THIS IS A PATCH FILE — not a replacement for your full conversation_engine.py.
#
# ADD these two methods to your existing ConversationEngine class.
# They implement the "Fusion layer (new)" step in the required pipeline:
#
#   feature_answers + voice_fusion → combined ML feature vector → ML Prediction
#
# The class diagram shows:
#   ConversationManager.EmotionData ← send data ← EmotionAnalyzer
#   ConversationManager ← compare data → CBT-Template Manager
#
# In practice app.py already calls engine.run_prediction(condition, features).
# The fusion layer adds voice_fusion scores AS EXTRA FEATURES to the feature
# dict BEFORE it goes into the ML model.  This means the ML model sees both
# the user's typed/clicked answers AND the acoustic emotion signal together.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO USE:
#
#   In app.py (already done in app.py v3), replace:
#       prediction = engine.run_prediction(condition, features)
#   with:
#       fused = engine.build_fused_feature_vector(condition, features, voice_fusion)
#       prediction = engine.run_prediction(condition, fused)
#
# ─────────────────────────────────────────────────────────────────────────────

class ConversationEngineFusionMixin:
    """
    Mixin that adds voice-fusion capability to ConversationEngine.

    Mix into ConversationEngine by adding it to the class bases:
        class ConversationEngine(ConversationEngineFusionMixin, ...):
            ...

    Or simply paste the two methods directly into ConversationEngine.
    """

    def build_fused_feature_vector(
        self,
        condition:    str,
        features:     dict,
        voice_fusion: dict,
    ) -> dict:
        """
        Combine feature-question answers with voice biomarker scores
        to create a fused feature vector for ML prediction.

        This is the "Fusion layer (new)" in the required pipeline diagram:
            feature_answers + voice_fusion → combined into ML feature vector

        Parameters
        ----------
        condition    : "anxiety" or "stress"
        features     : dict of feature answers (keys = column names used by ML model)
        voice_fusion : dict with keys: anxiety, stress, sadness, joy (0.0–1.0 floats)
                       from VoiceInputHandler.run_pipeline()["voice_fusion_for_ml"]

        Returns
        -------
        dict — features dict with voice scores appended as extra columns.
               If voice_fusion is empty/None, returns features unchanged.

        Voice columns appended
        ----------------------
        Anxiety model:
            "voice_anxiety_score"  ← voice_fusion["anxiety"]  (0–1)
            "voice_sadness_score"  ← voice_fusion["sadness"]  (0–1)
        Stress model:
            "voice_stress_score"   ← voice_fusion["stress"]   (0–1)
            "voice_sadness_score"  ← voice_fusion["sadness"]  (0–1)

        NOTE: If your scikit-learn model was NOT trained with these extra
        columns, run_prediction() will raise a ValueError about unexpected
        feature names.  In that case use app.py's _adjust_level_with_voice()
        fallback instead (which adjusts the level AFTER prediction, without
        touching the feature vector).  The fallback is already in app.py v3.
        """
        if not voice_fusion:
            return features

        fused = dict(features)   # shallow copy — do not mutate caller's dict

        if condition == "anxiety":
            fused["voice_anxiety_score"] = float(voice_fusion.get("anxiety", 0.0))
            fused["voice_sadness_score"] = float(voice_fusion.get("sadness", 0.0))
        else:
            fused["voice_stress_score"]  = float(voice_fusion.get("stress",  0.0))
            fused["voice_sadness_score"] = float(voice_fusion.get("sadness", 0.0))

        print(
            f"[ConversationEngine] Fused feature vector built for condition={condition}. "
            f"Voice columns added: { {k: v for k, v in fused.items() if 'voice_' in k} }",
            flush=True,
        )
        return fused

    def map_prediction_to_level(self, prediction) -> str:
        """
        Map a raw ML model output (class label or numeric) to
        "low" | "medium" | "high".

        Override this method if your model returns different class labels.

        Returns empty string if prediction cannot be mapped (caller should
        then use _map_level() fallback in app.py).
        """
        if prediction is None:
            return ""

        # String class labels (most scikit-learn classifiers)
        if isinstance(prediction, str):
            pred_lower = prediction.strip().lower()
            if pred_lower in ("low",    "0", "mild"):      return "low"
            if pred_lower in ("medium", "1", "moderate"):  return "medium"
            if pred_lower in ("high",   "2", "severe"):    return "high"
            return ""

        # Numeric labels
        try:
            val = float(prediction)
            if val <= 0:    return "low"
            elif val <= 1:  return "medium"
            else:           return "high"
        except (TypeError, ValueError):
            return ""


