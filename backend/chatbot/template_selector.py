import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

TEMPLATE_PATH = BASE_DIR / "templates" / "cbt_templates.json"

with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    templates = json.load(f)


def select_template(condition: str, level: str):

    for template in templates:

        if template["emotion"] == condition and template["level"] == level:
            return template

    return None