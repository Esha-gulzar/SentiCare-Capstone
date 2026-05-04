from backend.components.screening_manager import ScreeningManager
from backend.chatbot.router import route_condition
from backend.chatbot.predictor import predict
from backend.chatbot.template_selector import select_template


class MentalHealthChatbot:

    def __init__(self):
        self.screening_manager = ScreeningManager()

    def generate_response(self, screening_responses, feature_responses):

        scores = self.screening_manager.calculate_scores(screening_responses)

        condition = route_condition(scores)

        if condition == "neutral":
            return {
                "message": "You seem stable right now. I'm here whenever you want to talk."
            }

        prediction = predict(condition, feature_responses)

        label_map = {
            0: "low",
            1: "medium",
            2: "high"
        }

        level = label_map.get(prediction, prediction)

        template = select_template(condition, level)

        if not template:
            return {
                "message": "I'm here to support you. Let's take a slow breath together."
            }

        return {
            "condition": condition,
            "severity": level,
            "validation": template["therapy"]["validation"],
            "intervention_steps": template["therapy"]["intervention_steps"],
            "grounding_statement": template["therapy"]["grounding_statement"],
            "guided_questions": template["guided_questions"]
        }