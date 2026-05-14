# backend/chatbot/questions/stress_questions.py
# ─────────────────────────────────────────────────────────────────────────────
# Each question has:
#   col          → exact column name the ML pipeline expects
#   question_en  → English question text
#   question_ur  → Urdu question text
#   input_type   → "number" | "radio" | "select" | "slider" | "scale_5"
#   options_en   → list of {label, value} for button UI  (EN)
#   options_ur   → list of {label, value} for button UI  (UR)
#   min / max    → for number / slider validation
#
# NOTE: "scale_5" type = 1-5 frequency scale, rendered as buttons automatically
# ─────────────────────────────────────────────────────────────────────────────

STRESS_FEATURE_QUESTIONS = [
    {
        "col":         "Gender",
        "question_en": "What is your gender?",
        "question_ur": "آپ کی جنس کیا ہے؟",
        "input_type":  "stress_gender",
        "options_en": [
            {"label": "Male",   "value": "0"},
            {"label": "Female", "value": "1"},
        ],
        "options_ur": [
            {"label": "مرد",  "value": "0"},
            {"label": "عورت", "value": "1"},
        ],
    },
    {
        "col":         "Age",
        "question_en": "What is your age? (e.g. 22)",
        "question_ur": "آپ کی عمر کیا ہے؟ (مثلاً 22)",
        "input_type":  "number",
        "min": 10, "max": 100,
    },
    {
        "col":         "Have you recently experienced stress in your life?",
        "question_en": "How often have you recently experienced stress?",
        "question_ur": "آپ نے حال ہی میں کتنا دباؤ محسوس کیا؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Have you noticed a rapid heartbeat or palpitations?",
        "question_en": "How often do you notice a rapid heartbeat or palpitations?",
        "question_ur": "آپ کو کتنی بار دل کی تیز دھڑکن محسوس ہوئی؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Have you been dealing with anxiety or tension recently?",
        "question_en": "How often do you deal with anxiety or tension?",
        "question_ur": "آپ کو کتنی بار گھبراہٹ یا تناؤ محسوس ہوا؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Do you face any sleep problems or difficulties falling asleep?",
        "question_en": "How often do you face sleep problems?",
        "question_ur": "آپ کو کتنی بار نیند میں مشکل ہوئی؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Have you been getting headaches more often than usual?",
        "question_en": "How often do you get headaches more than usual?",
        "question_ur": "آپ کو کتنی بار معمول سے زیادہ سر درد ہوا؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Do you get irritated easily?",
        "question_en": "How often do you get irritated easily?",
        "question_ur": "آپ کتنی بار جلدی چڑچڑاتے ہیں؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Do you have trouble concentrating on your academic tasks?",
        "question_en": "How often do you have trouble concentrating?",
        "question_ur": "آپ کو کتنی بار توجہ مرکوز کرنے میں دشواری ہوئی؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Have you been feeling sadness or low mood?",
        "question_en": "How often do you feel sadness or low mood?",
        "question_ur": "آپ کتنی بار اداسی یا مایوسی محسوس کرتے ہیں؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Do you feel overwhelmed with your academic workload?",
        "question_en": "How often do you feel overwhelmed with your workload?",
        "question_ur": "آپ کام یا پڑھائی کے بوجھ سے کتنا دبا ہوا محسوس کرتے ہیں؟",
        "input_type":  "scale_5",
    },
    {
        "col":         "Is your working environment unpleasant or stressful?",
        "question_en": "How stressful is your working or study environment?",
        "question_ur": "آپ کا کام یا پڑھائی کا ماحول کتنا تناؤ والا ہے؟",
        "input_type":  "scale_5",
    },
]

# Columns the stress model needs that are NOT collected via questions.
# These are filled in with sensible defaults so the predictor never crashes.
STRESS_DEFAULTS = {
    "Have you been dealing with anxiety or tension recently?.1":  3,
    "Have you been experiencing any illness or health issues?":   2,
    "Do you often feel lonely or isolated?":                      2,
    "Are you in competition with your peers, and does it affect you?": 2,
    "Do you find that your relationship often causes you stress?": 2,
    "Are you facing any difficulties with your professors or instructors?": 2,
    "Do you struggle to find time for relaxation and leisure activities?": 3,
    "Is your hostel or home environment causing you difficulties?": 2,
    "Do you lack confidence in your academic performance?":        2,
    "Do you lack confidence in your choice of academic subjects?": 2,
    "Academic and extracurricular activities conflicting for you?": 2,
    "Do you attend classes regularly?":                            1,
    "Have you gained/lost weight?":                                2,
}