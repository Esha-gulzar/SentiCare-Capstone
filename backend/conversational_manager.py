from db import cbt_collection
import random

def generate_response(stress_level, language="en"):

    templates = list(cbt_collection.find({"category": stress_level}))

    if not templates:
        return "I'm here to support you."

    template = random.choice(templates)

    steps = template.get("steps", [])

    if language == "ur":
        # Later you can translate
        return " ".join(steps)
    else:
        return " ".join(steps)
