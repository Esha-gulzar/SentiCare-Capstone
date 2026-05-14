# backend/chatbot/conversation_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# CHANGE from original:
#   generate_cbt_response() now includes "steps_alt" in its return dict.
#   This is the only change — everything else is identical to your original.
# ─────────────────────────────────────────────────────────────────────────────

from backend.components.screening_manager import ScreeningManager
from backend.chatbot.router               import route_condition
from backend.chatbot.predictor            import predict
from backend.chatbot.template_selector    import select_template

from backend.chatbot.questions.anxiety_questions import ANXIETY_FEATURE_QUESTIONS
from backend.chatbot.questions.stress_questions  import STRESS_FEATURE_QUESTIONS


class ConversationEngine:

    def __init__(self):
        self.screening_manager = ScreeningManager()

    def calculate_screening_scores(self, screening_answers: dict) -> dict:
        return self.screening_manager.calculate_scores(screening_answers)

    def determine_condition(self, scores: dict) -> str:
        return route_condition(scores)

    def get_feature_questions(self, condition: str) -> list:
        if condition == "anxiety":
            return ANXIETY_FEATURE_QUESTIONS
        if condition == "stress":
            return STRESS_FEATURE_QUESTIONS
        return []

    def run_prediction(self, condition: str, feature_answers: dict):
        return predict(condition, feature_answers)

    def map_prediction_to_level(self, prediction) -> str | None:
        try:
            val = int(prediction)
        except (TypeError, ValueError):
            return None
        return {0: "low", 1: "medium", 2: "high"}.get(val)

    def generate_cbt_response(self, condition: str, level: str, lang: str = "en") -> dict | None:
        """
        Returns a dict with keys:
            validation   str
            steps        list[str]   — default intervention steps
            steps_alt    list[str]   — alternate steps (for RL policy switch)
                                       None if not defined in template
            grounding    str
            questions    list[str]
        or None if no template found.
        """
        template = select_template(condition, level, lang=lang)
        if template is None:
            return None

        therapy = template["therapy"]
        return {
            "validation": therapy["validation"],
            "steps":      therapy["intervention_steps"],
            "steps_alt":  therapy.get("steps_alt"),   # ← NEW: passes alt steps through
            "grounding":  therapy["grounding_statement"],
            "questions":  template["guided_questions"],
        }