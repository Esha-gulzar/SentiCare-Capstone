import json
import os


class ScreeningManager:

    def __init__(self):
        # Get backend directory
        backend_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # Correct path to JSON file
        file_path = os.path.join(
            backend_dir,
            "database",
            "data",
            "mental_health_screening.json"
        )

        # Debug print (optional - remove later)
        # print("Loading file from:", file_path)

        # Check if file exists (professional safety check)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Screening file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.questions = data.get("questions", [])

    def calculate_scores(self, responses):
        anxiety_score = 0
        depression_score = 0
        stress_score = 0

        for q in self.questions:
            q_id = q.get("id")
            domain = q.get("domain")
            value = responses.get(q_id, 0)

            if domain == "anxiety":
                anxiety_score += value
            elif domain == "depression":
                depression_score += value
            elif domain == "stress":
                stress_score += value

        return {
            "anxiety": anxiety_score,
            "depression": depression_score,
            "stress": stress_score
        }