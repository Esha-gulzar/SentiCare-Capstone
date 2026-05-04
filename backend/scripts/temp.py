import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.components.screening_manager import ScreeningManager

manager = ScreeningManager()

responses = {
    "feeling_nervous": 2,
    "uncontrollable_worry": 3,
    "restlessness": 1,
    "feeling_down": 0,
    "loss_of_interest": 1,
    "fatigue": 2,
    "overwhelmed": 3,
    "irritability": 2
}

scores = manager.calculate_scores(responses)

print(scores)
