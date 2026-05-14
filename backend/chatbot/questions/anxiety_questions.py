# backend/chatbot/questions/anxiety_questions.py
# ─────────────────────────────────────────────────────────────────────────────
# Each question has:
#   col          → exact column name the ML pipeline expects
#   question_en  → English question text
#   question_ur  → Urdu question text
#   input_type   → "number" | "radio" | "select" | "slider"
#   options_en   → list of {label, value} for button UI  (EN)
#   options_ur   → list of {label, value} for button UI  (UR)
#   min / max    → for number / slider validation
# ─────────────────────────────────────────────────────────────────────────────

ANXIETY_FEATURE_QUESTIONS = [
    {
        "col":         "Age",
        "question_en": "What is your age? (e.g. 22)",
        "question_ur": "آپ کی عمر کیا ہے؟ (مثلاً 22)",
        "input_type":  "number",
        "min": 10, "max": 100,
    },
    {
        "col":         "Gender",
        "question_en": "What is your gender?",
        "question_ur": "آپ کی جنس کیا ہے؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Male",   "value": "Male"},
            {"label": "Female", "value": "Female"},
            {"label": "Other",  "value": "Other"},
        ],
        "options_ur": [
            {"label": "مرد",  "value": "Male"},
            {"label": "عورت", "value": "Female"},
            {"label": "دیگر", "value": "Other"},
        ],
    },
    {
        "col":         "Occupation",
        "question_en": "What is your current occupation?",
        "question_ur": "آپ کا پیشہ کیا ہے؟",
        "input_type":  "select",
        "options_en": [
            {"label": "Student",       "value": "Student"},
            {"label": "Employed",      "value": "Employed"},
            {"label": "Self-employed", "value": "Self-employed"},
            {"label": "Unemployed",    "value": "Unemployed"},
            {"label": "Other",         "value": "Other"},
        ],
        "options_ur": [
            {"label": "طالب علم",    "value": "Student"},
            {"label": "ملازم",       "value": "Employed"},
            {"label": "خود کاروبار", "value": "Self-employed"},
            {"label": "بے روزگار",   "value": "Unemployed"},
            {"label": "دیگر",        "value": "Other"},
        ],
    },
    {
        "col":         "Sleep Hours",
        "question_en": "How many hours do you sleep per night? (e.g. 6)",
        "question_ur": "آپ رات کو اوسطاً کتنے گھنٹے سوتے ہیں؟ (مثلاً 6)",
        "input_type":  "number",
        "min": 0, "max": 12,
    },
    {
        "col":         "Physical Activity (hrs/week)",
        "question_en": "How many hours per week do you exercise? (e.g. 3)",
        "question_ur": "آپ ہفتے میں کتنے گھنٹے ورزش کرتے ہیں؟ (مثلاً 3)",
        "input_type":  "number",
        "min": 0, "max": 20,
    },
    {
        "col":         "Caffeine Intake (mg/day)",
        "question_en": "How much caffeine do you consume per day in mg? (enter 0 if none)",
        "question_ur": "آپ روزانہ کتنی مقدار میں کیفین لیتے ہیں ملی گرام میں؟ (نہیں لیتے تو 0)",
        "input_type":  "number",
        "min": 0, "max": 600,
    },
    {
        "col":         "Alcohol Consumption (drinks/week)",
        "question_en": "How many alcoholic drinks per week? (enter 0 if none)",
        "question_ur": "آپ ہفتے میں کتنی الکوحل پیتے ہیں؟ (نہیں پیتے تو 0)",
        "input_type":  "number",
        "min": 0, "max": 30,
    },
    {
        "col":         "Smoking",
        "question_en": "Do you smoke?",
        "question_ur": "کیا آپ سگریٹ پیتے ہیں؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },
    {
        "col":         "Family History of Anxiety",
        "question_en": "Do you have a family history of anxiety?",
        "question_ur": "کیا آپ کے خاندان میں گھبراہٹ کی بیماری کی تاریخ ہے؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },
    {
        "col":         "Stress Level (1-10)",
        "question_en": "Rate your current stress level from 1 to 10",
        "question_ur": "اپنی موجودہ ذہنی پریشانی 1 سے 10 تک درجہ دیں",
        "input_type":  "number",
        "min": 1, "max": 10,
    },
    {
        "col":         "Heart Rate (bpm)",
        "question_en": "What is your resting heart rate in beats per minute? (e.g. 75)",
        "question_ur": "آپ کی دل کی دھڑکن فی منٹ کتنی ہے؟ (مثلاً 75)",
        "input_type":  "number",
        "min": 40, "max": 180,
    },
    {
        "col":         "Breathing Rate (breaths/min)",
        "question_en": "What is your breathing rate in breaths per minute? (e.g. 16)",
        "question_ur": "آپ کی سانس کی شرح فی منٹ کیا ہے؟ (مثلاً 16)",
        "input_type":  "number",
        "min": 8, "max": 40,
    },
    {
        "col":         "Sweating Level (1-5)",
        "question_en": "Rate your sweating level from 1 (none) to 5 (excessive)",
        "question_ur": "پسینے کی سطح 1 سے 5 تک درجہ دیں (1=بالکل نہیں، 5=بہت زیادہ)",
        "input_type":  "number",
        "min": 1, "max": 5,
    },
    {
        "col":         "Dizziness",
        "question_en": "Have you experienced dizziness recently?",
        "question_ur": "کیا آپ کو حال ہی میں چکر آئے؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },
    {
        "col":         "Medication",
        "question_en": "Are you currently taking any medication for anxiety?",
        "question_ur": "کیا آپ گھبراہٹ کی کوئی دوا لے رہے ہیں؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },
    {
        "col":         "Therapy Sessions (per month)",
        "question_en": "How many therapy sessions do you attend per month? (enter 0 if none)",
        "question_ur": "آپ ماہانہ کتنے تھراپی سیشن لیتے ہیں؟ (نہیں لیتے تو 0)",
        "input_type":  "number",
        "min": 0, "max": 20,
    },
    {
        "col":         "Recent Major Life Event",
        "question_en": "Have you experienced a major life event recently?",
        "question_ur": "کیا آپ کی زندگی میں حال ہی میں کوئی بڑا واقعہ ہوا؟",
        "input_type":  "radio",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },
    {
        "col":         "Diet Quality (1-10)",
        "question_en": "Rate your diet quality from 1 (very poor) to 10 (excellent)",
        "question_ur": "اپنی خوراک کا معیار 1 سے 10 تک درجہ دیں (1=بہت خراب، 10=بہترین)",
        "input_type":  "number",
        "min": 1, "max": 10,
    },
]