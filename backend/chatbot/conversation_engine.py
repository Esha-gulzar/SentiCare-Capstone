from backend.components.screening_manager import ScreeningManager
from backend.chatbot.router import route_condition
from backend.chatbot.predictor import predict
from backend.chatbot.template_selector import select_template

from backend.chatbot.questions.anxiety_questions import ANXIETY_FEATURE_QUESTIONS
from backend.chatbot.questions.stress_questions import STRESS_FEATURE_QUESTIONS


class ConversationEngine:

    def __init__(self):
        self.screening_manager = ScreeningManager()

    # STEP 1
    def calculate_screening_scores(self, screening_answers: dict):
        return self.screening_manager.calculate_scores(screening_answers)

    # STEP 2
    def determine_condition(self, scores: dict):
        return route_condition(scores)

    # STEP 3
    def get_feature_questions(self, condition: str):

        if condition == "anxiety":
            return ANXIETY_FEATURE_QUESTIONS

        elif condition == "stress":
            return STRESS_FEATURE_QUESTIONS

        return {}

    # STEP 4
    def run_prediction(self, condition: str, feature_answers: dict):
        return predict(condition, feature_answers)

    # STEP 5
    def map_prediction_to_level(self, prediction):

        if prediction == 0:
            return "low"

        elif prediction == 1:
            return "medium"

        elif prediction == 2:
            return "high"

        return None

    # STEP 6
    def generate_cbt_response(self, condition: str, level: str):

        template = select_template(condition, level)

        if template is None:
            return None

        therapy = template["therapy"]

        response = {
            "validation": therapy["validation"],
            "steps": therapy["intervention_steps"],
            "grounding": therapy["grounding_statement"],
            "questions": template["guided_questions"]
        }

        return response