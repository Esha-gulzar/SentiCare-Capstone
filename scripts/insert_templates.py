from pymongo import MongoClient

client = MongoClient("mongodb://10.117.86.221:27017/")
db = client["senticare"]
collection = db["cbt_templates"]

templates = [
    {
        "template_id": "CBT_001",
        "trigger_emotion": "high_anxiety",
        "title": "5-4-3-2-1 Grounding",
        "response": "Take a deep breath. Name 5 things you can see, 4 you can feel, 3 you can hear, 2 you can smell, and 1 you can taste."
    },
    {
        "template_id": "CBT_002",
        "trigger_emotion": "panic",
        "title": "Breathing Control",
        "response": "Inhale for 4 seconds, hold for 4, exhale for 6. Repeat 5 times."
    },
    {
        "template_id": "CBT_003",
        "trigger_emotion": "overthinking",
        "title": "Thought Reframing",
        "response": "Ask yourself: Is this thought 100% true? What evidence do I have?"
    }
]

