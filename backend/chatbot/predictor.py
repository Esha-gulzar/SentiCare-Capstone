# backend/chatbot/predictor.py
#
# FIXES — all three conditions:
# ─────────────────────────────────────────────────────────────────────────────
# ANXIETY:
#   anxiety_model.predict() was called with short-key dict (e.g. "phq_score").
#   Added _ANX_RENAME to map chat-form short keys → training column names.
#   Also added _ANX_DEFAULTS so missing columns are safely filled.
#
#   NEW FIX: anxiety model returns a regression float (e.g. 3.71), not 0/1/2.
#   Added _normalise_anxiety_raw() to map the float → 0 (low) / 1 (medium) / 2 (high).
#   Both predict paths (renamed keys + original keys fallback) now use this
#   normaliser instead of bare int(raw), so map_prediction_to_level() in
#   conversation_engine always receives a valid 0/1/2 and never hits the
#   "Unrecognised prediction" warning.
#
# STRESS:
#   Stress has no ML model — level is derived from averaging scale_cols.
#   BUT scale_cols used full question strings as dict keys, while features{}
#   arrives with short keys from STRESS_FEATURE_QUESTIONS["col"].
#   The average was ALWAYS empty → avg defaulted to 3.0 → always "medium".
#   Fixed by adding _STRESS_SHORT_TO_FULL rename so the average is computed
#   from the actual answers the user gave.
#   The stress predict() now returns 0/1/2 (not a level string) so
#   map_prediction_to_level() in conversation_engine works correctly.
#
# DEPRESSION:
#   All previous fixes preserved:
#   • Full rename of demo + PHQ columns
#   • any_symptom_count pre-computed
#   • Auto-built encoder int map at startup
#   • medium→1 (not 2) in fallback map
# ─────────────────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
from pathlib import Path
import __main__

from backend.chatbot.phq_feature_engineer import PHQFeatureEngineer

__main__.PHQFeatureEngineer = PHQFeatureEngineer

BASE_DIR = Path(__file__).resolve().parents[2]

# ── Load models ───────────────────────────────────────────────────────────────
stress_model = joblib.load(
    BASE_DIR / "artifacts/stress_classification/stress_bundle.joblib"
)
anxiety_model = joblib.load(
    BASE_DIR / "artifacts/anxiety_classification/anxiety_pipeline.joblib"
)
depression_bundle        = joblib.load(
    BASE_DIR / "artifacts/depression_classification/depression_bundle.joblib"
)
depression_pipeline      = depression_bundle["pipeline"]
depression_label_encoder = depression_bundle["label_encoder"]

_encoder_classes = list(depression_label_encoder.classes_)
print(
    f"[predictor] depression label_encoder.classes_ = {_encoder_classes}",
    flush=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  ANXIETY — column rename map
#  Maps chat-form short keys (from ANXIETY_FEATURE_QUESTIONS["col"])
#  → full column names the anxiety_pipeline was trained on.
#  Update these if your training data used different column names.
# ══════════════════════════════════════════════════════════════════════════════

_ANX_RENAME: dict[str, str] = {
    # Demographic
    "age":                  "Age",
    "gender":               "Gender",

    # PHQ-style anxiety items (GAD-7 equivalent)
    "feeling_nervous":      "feeling_nervous",        # often kept as-is
    "uncontrollable_worry": "uncontrollable_worry",
    "worrying_too_much":    "worrying_too_much",
    "trouble_relaxing":     "trouble_relaxing",
    "restlessness":         "restlessness",
    "easily_annoyed":       "easily_annoyed",
    "feeling_afraid":       "feeling_afraid",

    # Physiological / extra features
    "heart_rate":           "Heart Rate (bpm)",
    "sleep_hours":          "Sleep Hours",
    "breathing_difficulty": "Breathing Difficulty",
    "sweating":             "Sweating",
    "trembling":            "Trembling",
    "headache":             "Headache",
    "fatigue":              "Fatigue",
    "concentration":        "Concentration",

    # GAD total score (some models use pre-summed score)
    "gad_score":            "GAD Score",
    "phq_score":            "PHQ Score",
    "total_score":          "Total Score",
}

# Safe defaults for any column the anxiety model expects but wasn't collected
_ANX_DEFAULTS: dict[str, int] = {
    "Heart Rate (bpm)": 75,
}


# ══════════════════════════════════════════════════════════════════════════════
#  ANXIETY — raw prediction normaliser
#  The anxiety model returns a regression float (e.g. 3.71), not a 0/1/2 class.
#  This function maps it to 0/1/2 so map_prediction_to_level() works correctly.
#
#  Thresholds (adjust to match your training data distribution):
#    raw ≤ 2.0  →  0  (low)
#    raw ≤ 3.5  →  1  (medium)
#    raw >  3.5  →  2  (high)
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_anxiety_raw(raw) -> int:
    val = float(raw)
    if val < 1.5:
        level_int = 0   # low    (class 0 or 1)
    elif val < 3.5:
        level_int = 1   # medium (class 2)
    else:
        level_int = 2   # high   (class 4, or above)
    print(
        f"[predictor] anxiety _normalise_anxiety_raw: raw={val} → level_int={level_int}",
        flush=True,
    )
    return level_int

# ══════════════════════════════════════════════════════════════════════════════
#  STRESS — short key → full question string rename map
#  Maps chat-form short keys (from STRESS_FEATURE_QUESTIONS["col"])
#  → the exact strings used as scale_cols in app.py's stress averaging logic
#  AND as column names in the stress_bundle pipeline.
# ══════════════════════════════════════════════════════════════════════════════

_STRESS_SHORT_TO_FULL: dict[str, str] = {
    "stress_in_life":         "Have you recently experienced stress in your life?",
    "rapid_heartbeat":        "Have you noticed a rapid heartbeat or palpitations?",
    "anxiety_tension":        "Have you been dealing with anxiety or tension recently?",
    "sleep_problems":         "Do you face any sleep problems or difficulties falling asleep?",
    "headaches":              "Have you been getting headaches more often than usual?",
    "easily_irritated":       "Do you get irritated easily?",
    "trouble_concentrating":  "Do you have trouble concentrating on your academic tasks?",
    "feeling_sadness":        "Have you been feeling sadness or low mood?",
    "overwhelmed_workload":   "Do you feel overwhelmed with your academic workload?",
    "unpleasant_environment": "Is your working environment unpleasant or stressful?",

    # Demographic fields stress model may use
    "gender":                 "Gender",
    "age":                    "Age",
    "sleep_duration":         "Sleep Duration",
    "bmi":                    "BMI",
    "physical_activity":      "Physical Activity Level",
    "occupation":             "Occupation",
    "blood_pressure":         "Blood Pressure",
     "heart_rate":             "Heart Rate",
    "daily_steps":            "Daily Steps",
    "sleep_disorder":         "Sleep Disorder",
}

# The scale columns used for the average-based fallback (full strings)
_STRESS_SCALE_COLS_FULL = [
    "Have you recently experienced stress in your life?",
    "Have you noticed a rapid heartbeat or palpitations?",
    "Have you been dealing with anxiety or tension recently?",
    "Do you face any sleep problems or difficulties falling asleep?",
    "Have you been getting headaches more often than usual?",
    "Do you get irritated easily?",
    "Do you have trouble concentrating on your academic tasks?",
    "Have you been feeling sadness or low mood?",
    "Do you feel overwhelmed with your academic workload?",
    "Is your working environment unpleasant or stressful?",
]

_STRESS_DEFAULTS: dict[str, int] = {
    col: 3 for col in _STRESS_SCALE_COLS_FULL
}


# ══════════════════════════════════════════════════════════════════════════════
#  DEPRESSION — column rename maps
# ══════════════════════════════════════════════════════════════════════════════

_DEP_DEMO_RENAME: dict[str, str] = {
    "age":           "1. Age",
    "gender":        "2. Gender",
    "academic_year": "5. Academic Year",
    "cgpa":          "6. Current CGPA",
    "scholarship":   "7. Did you receive a waiver or scholarship at your university?",
}

_DEP_PHQ_RENAME: dict[str, str] = {
    "little_interest":    "1. In a semester, how often have you had little interest or pleasure in doing things?",
    "feeling_down":       "2. In a semester, how often have you been feeling down, depressed or hopeless?",
    "sleep_trouble":      "3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much? ",
    "feeling_tired":      "4. In a semester, how often have you been feeling tired or having little energy? ",
    "appetite":           "5. In a semester, how often have you had poor appetite or overeating? ",
    "feeling_bad":        "6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down? ",
    "concentration":      "7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television? ",
    "psychomotor":        "8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless? ",
    "self_harm_thoughts": "9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself? ",
}

_PHQ_COLS = list(_DEP_PHQ_RENAME.values())


# ══════════════════════════════════════════════════════════════════════════════
#  DEPRESSION — label normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _build_encoder_int_map(classes: list) -> dict[str, int]:
    LOW_KW    = {"low", "minimal", "none", "no depression", "no_depression",
                 "nodepression", "healthy", "normal", "0"}
    MEDIUM_KW = {"mild", "medium", "moderate", "slight", "1"}
    HIGH_KW   = {"high", "severe", "major", "2"}

    result, unmatched = {}, []
    for cls in classes:
        key = str(cls).strip().lower()
        if key in LOW_KW    or any(k in key for k in LOW_KW    if len(k) > 2):
            result[str(cls)] = 0
        elif key in MEDIUM_KW or any(k in key for k in MEDIUM_KW if len(k) > 2):
            result[str(cls)] = 1
        elif key in HIGH_KW   or any(k in key for k in HIGH_KW   if len(k) > 2):
            result[str(cls)] = 2
        else:
            unmatched.append(str(cls))

    used = set(result.values())
    remaining = sorted(set(range(3)) - used)
    for cls, slot in zip(sorted(unmatched), remaining):
        result[cls] = slot

    print(f"[predictor] depression int_map (auto-built) = {result}", flush=True)
    if unmatched:
        print(f"[predictor] ⚠️  Unmatched encoder classes: {unmatched}", flush=True)
    return result


_ENCODER_INT_MAP = _build_encoder_int_map(_encoder_classes)

_DEPRESSION_STRING_TO_INT: dict[str, int] = {
    # 0 — low / none
    "minimal": 0, "none": 0, "no depression": 0, "no_depression": 0,
    "nodepression": 0, "normal": 0, "healthy": 0, "low": 0, "0": 0,
    # 1 — mild / medium / moderate
    "mild": 1, "slight": 1, "medium": 1, "moderate": 1, "1": 1,
    # 2 — high / severe
    "high": 2, "severe": 2, "major": 2, "2": 2,
}


def _normalise_depression_label(raw_label) -> int:
    if raw_label is None:
        return 1
    raw_str = str(raw_label).strip()

    # 1. Auto-built map
    if raw_str in _ENCODER_INT_MAP:
        result = _ENCODER_INT_MAP[raw_str]
        print(f"[predictor] depression '{raw_str}' → auto_map → {result}", flush=True)
        return result

    # 2. Fallback exact
    lower = raw_str.lower()
    if lower in _DEPRESSION_STRING_TO_INT:
        result = _DEPRESSION_STRING_TO_INT[lower]
        print(f"[predictor] depression '{raw_str}' → fallback_exact → {result}", flush=True)
        return result

    # 3. Int cast
    try:
        val = int(float(raw_str))
        if val in (0, 1, 2):
            return val
    except (TypeError, ValueError):
        pass

    # 4. Substring scan
    for key in sorted(_DEPRESSION_STRING_TO_INT, key=len, reverse=True):
        if key in lower:
            result = _DEPRESSION_STRING_TO_INT[key]
            print(f"[predictor] depression '{raw_str}' → substring '{key}' → {result}", flush=True)
            return result

    print(f"[predictor] ⚠️  UNKNOWN depression label='{raw_str}' → defaulting to 1", flush=True)
    return 1


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC predict()
# ══════════════════════════════════════════════════════════════════════════════

def predict(condition: str, feature_answers: dict):

    feature_answers = dict(feature_answers)

    # ── ANXIETY ───────────────────────────────────────────────────────────────
    if condition == "anxiety":
        # Apply defaults first, then rename short keys → training column names
        for col, val in _ANX_DEFAULTS.items():
            feature_answers.setdefault(col, val)

        renamed = {_ANX_RENAME.get(k, k): v for k, v in feature_answers.items()}

        print(
            f"[predictor] anxiety renamed keys={list(renamed.keys())}",
            flush=True,
        )
        df = pd.DataFrame([renamed])
        try:
            raw = anxiety_model.predict(df)[0]
            print(f"[predictor] anxiety raw={raw!r}", flush=True)
            return _normalise_anxiety_raw(raw)          # ← FIXED: was int(raw)
        except Exception as exc:
            # If rename still doesn't match, fall back to direct short-key predict
            print(
                f"[predictor] ⚠️  anxiety renamed predict failed: {exc}. "
                f"Retrying with original keys.",
                flush=True,
            )
            df_orig = pd.DataFrame([feature_answers])
            raw = anxiety_model.predict(df_orig)[0]
            print(f"[predictor] anxiety raw (orig keys)={raw!r}", flush=True)
            return _normalise_anxiety_raw(raw)          # ← FIXED: was int(raw)

    # ── STRESS ────────────────────────────────────────────────────────────────
    if condition == "stress":
        # Step 1: rename short keys → full strings for both ML and scale average
        renamed = {_STRESS_SHORT_TO_FULL.get(k, k): v for k, v in feature_answers.items()}

        # Step 2: fill defaults for any missing scale columns
        for col, val in _STRESS_DEFAULTS.items():
            renamed.setdefault(col, val)

        print(
            f"[predictor] stress renamed keys={list(renamed.keys())}",
            flush=True,
        )

        # Step 3: try ML model first
        try:
            df = pd.DataFrame([renamed])
            raw = stress_model.predict(df)[0]
            print(f"[predictor] stress ML raw={raw!r}", flush=True)
            return int(raw)

        except Exception as exc:
            print(
                f"[predictor] ⚠️  stress ML failed: {exc}. "
                f"Falling back to scale average.",
                flush=True,
            )

        # Step 4: fallback — average of scale columns → 0/1/2
        sc = []
        for col in _STRESS_SCALE_COLS_FULL:
            val = renamed.get(col)
            if val is not None:
                try:
                    sc.append(float(val))
                except (TypeError, ValueError):
                    pass

        avg = sum(sc) / len(sc) if sc else 3.0
        # stress model: 2=high 1=medium 0=low (inverted from anxiety/depression)
        if avg <= 2.0:
            level_int = 0   # low stress
        elif avg > 3.5:
            level_int = 2   # high stress
        else:
            level_int = 1   # medium stress

        print(
            f"[predictor] stress scale_avg={avg:.2f} → level_int={level_int} "
            f"({len(sc)} cols found)",
            flush=True,
        )
        return level_int

    # ── DEPRESSION ────────────────────────────────────────────────────────────
    if condition == "depression":
        # Step 1: rename all columns
        full_rename = {**_DEP_DEMO_RENAME, **_DEP_PHQ_RENAME}
        renamed = {full_rename.get(k, k): v for k, v in feature_answers.items()}

        # Step 2: cast PHQ values to int
        for col in _PHQ_COLS:
            if col in renamed:
                try:
                    renamed[col] = int(float(str(renamed[col])))
                except (TypeError, ValueError):
                    renamed[col] = 0

        # Step 3: pre-compute any_symptom_count
        present_phq = [c for c in _PHQ_COLS if c in renamed]
        renamed["any_symptom_count"] = sum(
            1 for c in present_phq if int(renamed.get(c, 0)) >= 1
        )

        # Also add Heart Rate default
        renamed.setdefault("Heart Rate (bpm)", 75)

        print(
            f"[predictor] depression renamed keys={list(renamed.keys())}",
            flush=True,
        )

        df_dep       = pd.DataFrame([renamed])
        pred_encoded = depression_pipeline.predict(df_dep)
        raw_label    = depression_label_encoder.inverse_transform(pred_encoded)[0]

        print(f"[predictor] depression raw_label='{raw_label}'", flush=True)
        normalised = _normalise_depression_label(raw_label)
        print(f"[predictor] depression normalised={normalised}", flush=True)
        return normalised

    print(f"[predictor] ⚠️  Unknown condition='{condition}' → None", flush=True)
    return None