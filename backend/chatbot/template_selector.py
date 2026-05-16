# backend/chatbot/template_selector.py
#
# FIX: Rewrote to match the actual cbt_templates.json structure.
#
# ROOT CAUSE of "list indices must be integers or slices, not str":
#   Old code did _TEMPLATES[condition][level][lang] assuming a nested dict.
#   cbt_templates.json is a FLAT LIST of objects → _TEMPLATES["anxiety"]
#   threw TypeError immediately, caught silently, returned None → fallback.
#
# THIS VERSION:
#   1. At startup, converts the list into {emotion: {level: entry}} for O(1) lookup.
#   2. Reads bilingual content using _ur suffix fields (matching the JSON convention).
#   3. Applies voice_dominant overlay from "voice_variants" block if present.
#   4. Returns the exact dict shape conversation_engine.generate_cbt_response() expects.

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


# ── Load and index at startup ─────────────────────────────────────────────────

def _load_templates() -> dict:
    path = BASE_DIR / "templates" / "cbt_templates.json"
    with open(path, encoding="utf-8") as f:
        raw: list = json.load(f)

    indexed: dict = {}
    for entry in raw:
        emotion = entry.get("emotion", "").lower().strip()
        level   = entry.get("level",   "").lower().strip()
        if emotion and level:
            indexed.setdefault(emotion, {})[level] = entry

    print(
        f"[template_selector] Loaded {len(raw)} templates → "
        f"{ {k: list(v.keys()) for k, v in indexed.items()} }",
        flush=True,
    )
    return indexed


_TEMPLATES: dict = _load_templates()


# ── Voice dominant → variant key ──────────────────────────────────────────────

_VOICE_TO_VARIANT: dict[str, str] = {
    "anxious":    "anxiety",
    "anxiety":    "anxiety",
    "stressed":   "stress",
    "stress":     "stress",
    "tense":      "stress",
    "depressed":  "depression",
    "depression": "depression",
    "sad":        "sadness",
    "sadness":    "sadness",
    "aroused":    "anxiety",
    "neutral":    "",
    "joy":        "",
    "":           "",
}


# ── Bilingual field helper ────────────────────────────────────────────────────

def _pick(obj: dict, key_en: str, key_ur: str, lang: str):
    """Return Urdu value if lang=='ur' and the _ur key exists, else English."""
    if lang == "ur":
        val = obj.get(key_ur)
        if val:
            return val
    return obj.get(key_en)


# ── Public API ────────────────────────────────────────────────────────────────

def select_template(
    condition:      str,
    level:          str,
    lang:           str = "en",
    voice_dominant: str = "neutral",
) -> dict | None:
    """
    Parameters
    ----------
    condition      : "anxiety" | "stress" | "depression"
    level          : "low" | "medium" | "high"
    lang           : "en" | "ur"
    voice_dominant : dominant emotion from EmotionFusionCombiner

    Returns
    -------
    dict:
        therapy          dict  — validation, intervention_steps,
                                 steps_alt, grounding_statement
        guided_questions list[str]
        voice_dominant   str
        prefer_alt_steps bool
    or None if no template found.
    """
    condition = (condition or "").lower().strip()
    level     = (level     or "").lower().strip()
    lang      = (lang      or "en").lower().strip()

    # ── Base lookup ───────────────────────────────────────────────────────────
    entry = _TEMPLATES.get(condition, {}).get(level)
    if entry is None:
        print(
            f"[select_template] ⚠️  No template for "
            f"condition='{condition}' level='{level}'. "
            f"Available: { {k: list(v.keys()) for k, v in _TEMPLATES.items()} }",
            flush=True,
        )
        return None

    therapy_raw = entry.get("therapy", {})

    # ── Base bilingual fields ─────────────────────────────────────────────────
    validation  = _pick(therapy_raw, "validation",             "validation_ur",              lang)
    steps       = _pick(therapy_raw, "intervention_steps",     "intervention_steps_ur",      lang) or []
    steps_alt   = _pick(therapy_raw, "intervention_steps_alt", "intervention_steps_alt_ur",  lang)
    grounding   = _pick(therapy_raw, "grounding_statement",    "grounding_statement_ur",     lang)
    guided_qs   = _pick(entry,       "guided_questions",       "guided_questions_ur",        lang) or []
    prefer_alt  = False

    # ── Voice variant overlay ─────────────────────────────────────────────────
    variant_key    = _VOICE_TO_VARIANT.get(voice_dominant, "")
    voice_variants = therapy_raw.get("voice_variants", {})

    if variant_key and variant_key in voice_variants:
        v = voice_variants[variant_key]
        print(
            f"[select_template] ✓ Voice variant '{variant_key}' applied "
            f"(dominant='{voice_dominant}') | "
            f"condition='{condition}' level='{level}' lang='{lang}'",
            flush=True,
        )
        # Override only fields the variant specifies; inherit rest from base.
        v_validation = _pick(v, "validation",             "validation_ur",             lang)
        if v_validation:
            validation = v_validation

        v_grounding = _pick(v, "grounding_statement",    "grounding_statement_ur",    lang)
        if v_grounding:
            grounding = v_grounding

        v_steps_alt = _pick(v, "intervention_steps_alt", "intervention_steps_alt_ur", lang)
        if v_steps_alt:
            steps_alt = v_steps_alt

        v_steps = _pick(v, "intervention_steps", "intervention_steps_ur", lang)
        if v_steps:
            steps = v_steps

        prefer_alt = bool(v.get("prefer_alt_steps", False))

    else:
        print(
            f"[select_template] Base template used "
            f"(dominant='{voice_dominant}' → variant='{variant_key}' not found) | "
            f"condition='{condition}' level='{level}' lang='{lang}'",
            flush=True,
        )

    print(
        f"[select_template] ✓ Returning | condition='{condition}' level='{level}' "
        f"lang='{lang}' prefer_alt={prefer_alt}",
        flush=True,
    )

    return {
        "therapy": {
            "validation":          validation,
            "intervention_steps":  steps,
            "steps_alt":           steps_alt,
            "grounding_statement": grounding,
        },
        "guided_questions":  guided_qs,
        "voice_dominant":    voice_dominant,
        "prefer_alt_steps":  prefer_alt,
    }