# backend/chatbot/template_selector.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads cbt_templates.json and returns the right template for a given
# condition + level.  Supports bilingual output via the `lang` parameter.
#
# CHANGE from original:
#   Now also returns "steps_alt" (the alternate intervention steps) so that
#   app.py's build_cbt_message() can use them when policy_mode == "alternate".
#   Without this, the RL policy switch silently did nothing.
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = BASE_DIR / "templates" / "cbt_templates.json"

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    _templates = json.load(f)


def select_template(condition: str, level: str, lang: str = "en") -> dict | None:
    """
    Return a template dict for the given condition and level.

    The returned dict has these keys (language-appropriate):
        "validation"          – str
        "intervention_steps"  – list[str]   (default policy)
        "steps_alt"           – list[str]   (alternate policy — may be None)
        "grounding_statement" – str
        "guided_questions"    – list[str]

    Returns None if no matching template is found.
    """
    for t in _templates:
        if t["emotion"] == condition and t["level"] == level:
            therapy = t["therapy"]

            if lang == "ur":
                return {
                    "emotion": condition,
                    "level":   level,
                    "therapy": {
                        "validation": therapy.get(
                            "validation_ur", therapy["validation"]
                        ),
                        "intervention_steps": therapy.get(
                            "intervention_steps_ur",
                            therapy["intervention_steps"]
                        ),
                        # ── alternate steps for RL policy switch ──────────────
                        "steps_alt": therapy.get(
                            "intervention_steps_alt_ur",
                            therapy.get("intervention_steps_alt")
                        ),
                        # ─────────────────────────────────────────────────────
                        "grounding_statement": therapy.get(
                            "grounding_statement_ur",
                            therapy["grounding_statement"]
                        ),
                    },
                    "guided_questions": t.get(
                        "guided_questions_ur",
                        t.get("guided_questions", [])
                    ),
                }
            else:
                return {
                    "emotion": condition,
                    "level":   level,
                    "therapy": {
                        "validation":          therapy["validation"],
                        "intervention_steps":  therapy["intervention_steps"],
                        # ── alternate steps for RL policy switch ──────────────
                        "steps_alt":           therapy.get(
                            "intervention_steps_alt"
                        ),
                        # ─────────────────────────────────────────────────────
                        "grounding_statement": therapy["grounding_statement"],
                    },
                    "guided_questions": t.get("guided_questions", []),
                }

    return None