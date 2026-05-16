# backend/chatbot/questions/depression_questions.py
#
# FIXES:
# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — "age" question: was input_type="select" with options as plain strings
#   ["Below 18", "18-22", ...]. Frontend expects {label, value} objects.
#   Now uses input_type="number" so AgeInput component handles it correctly.
#   Backend validate_feature_input() already handles "number" type.
#
# FIX 2 — PHQ-9 slider questions: input_type="slider", options_en=None.
#   Frontend showed no buttons because _resolve_feature() returned null options
#   for slider type, and the slider UI doesn't exist in the frontend.
#   FIXED: changed all slider questions to input_type="radio" with 4-option
#   frequency scale {label, value} buttons (0=Not at all → 3=Nearly every day).
#   These match the PHQ-9 scoring rubric exactly.
#
# FIX 3 — "gender", "academic_year", "cgpa" options were plain strings.
#   Now all options are proper {label, value} objects.
#
# FIX 4 — "scholarship" options were plain strings ["Yes", "No"].
#   Now proper {label, value} objects.
# ─────────────────────────────────────────────────────────────────────────────

# PHQ-9 frequency options — reused across all 9 clinical questions
_PHQ_OPTS_EN = [
    {"label": "0 — Not at all",             "value": "0"},
    {"label": "1 — Several days",            "value": "1"},
    {"label": "2 — More than half the days", "value": "2"},
    {"label": "3 — Nearly every day",        "value": "3"},
]
_PHQ_OPTS_UR = [
    {"label": "0 — بالکل نہیں",       "value": "0"},
    {"label": "1 — کچھ دن",            "value": "1"},
    {"label": "2 — آدھے سے زیادہ دن", "value": "2"},
    {"label": "3 — تقریباً ہر روز",    "value": "3"},
]

DEPRESSION_FEATURE_QUESTIONS = [

    # ── DEMOGRAPHIC FEATURES ──────────────────────────────────────────────────

    {
        # FIX 1: was input_type="select" with plain string options.
        # Now input_type="number" so AgeInput component renders a number field.
        # app.py validate_feature_input() handles "number" correctly.
        "col":        "age",
        "input_type": "number",
        "min": 10,
        "max": 100,
        "question_en": "What is your age? (e.g. 22)",
        "question_ur": "آپ کی عمر کیا ہے؟ (مثلاً 22)",
        "options_en": None,
        "options_ur": None,
    },

    {
        # FIX 3: options were plain strings, now {label, value} objects.
        "col":        "gender",
        "input_type": "select",
        "question_en": "What is your gender?",
        "question_ur": "آپ کی جنس کیا ہے؟",
        "options_en": [
            {"label": "Female",             "value": "Female"},
            {"label": "Male",               "value": "Male"},
            {"label": "Prefer not to say",  "value": "Prefer not to say"},
        ],
        "options_ur": [
            {"label": "عورت",          "value": "Female"},
            {"label": "مرد",           "value": "Male"},
            {"label": "بتانا نہیں چاہتے", "value": "Prefer not to say"},
        ],
    },

    {
        # FIX 3: options were plain strings, now {label, value} objects.
        "col":        "academic_year",
        "input_type": "select",
        "question_en": "What is your current academic year?",
        "question_ur": "آپ کا موجودہ تعلیمی سال کیا ہے؟",
        "options_en": [
            {"label": "First Year or Equivalent",  "value": "First Year or Equivalent"},
            {"label": "Second Year or Equivalent", "value": "Second Year or Equivalent"},
            {"label": "Third Year or Equivalent",  "value": "Third Year or Equivalent"},
            {"label": "Fourth Year or Equivalent", "value": "Fourth Year or Equivalent"},
        ],
        "options_ur": [
            {"label": "پہلا سال یا مساوی",  "value": "First Year or Equivalent"},
            {"label": "دوسرا سال یا مساوی", "value": "Second Year or Equivalent"},
            {"label": "تیسرا سال یا مساوی", "value": "Third Year or Equivalent"},
            {"label": "چوتھا سال یا مساوی", "value": "Fourth Year or Equivalent"},
        ],
    },

    {
        # FIX 3: options were plain strings, now {label, value} objects.
        "col":        "cgpa",
        "input_type": "select",
        "question_en": "What is your current CGPA?",
        "question_ur": "آپ کا موجودہ سی جی پی اے کیا ہے؟",
        "options_en": [
            {"label": "Below 2.50",    "value": "Below 2.50"},
            {"label": "2.50 - 2.99",   "value": "2.50 - 2.99"},
            {"label": "3.00 - 3.39",   "value": "3.00 - 3.39"},
            {"label": "3.40 - 3.79",   "value": "3.40 - 3.79"},
            {"label": "3.80 - 4.00",   "value": "3.80 - 4.00"},
        ],
        "options_ur": [
            {"label": "2.50 سے کم",  "value": "Below 2.50"},
            {"label": "2.50 - 2.99", "value": "2.50 - 2.99"},
            {"label": "3.00 - 3.39", "value": "3.00 - 3.39"},
            {"label": "3.40 - 3.79", "value": "3.40 - 3.79"},
            {"label": "3.80 - 4.00", "value": "3.80 - 4.00"},
        ],
    },

    {
        # FIX 4: options were plain strings, now {label, value} objects.
        "col":        "scholarship",
        "input_type": "radio",
        "question_en": "Do you receive a scholarship or fee waiver at your university?",
        "question_ur": "کیا آپ کو یونیورسٹی میں وظیفہ یا فیس معافی ملتی ہے؟",
        "options_en": [
            {"label": "Yes", "value": "Yes"},
            {"label": "No",  "value": "No"},
        ],
        "options_ur": [
            {"label": "ہاں", "value": "Yes"},
            {"label": "نہیں", "value": "No"},
        ],
    },

    # ── PHQ-9 CLINICAL QUESTIONS ──────────────────────────────────────────────
    # FIX 2: All changed from input_type="slider" (no frontend UI exists)
    # to input_type="radio" with 4-option PHQ-9 frequency scale.
    # Values 0-3 match the ML model's expected input exactly.

    {
        "col":        "little_interest",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you had little interest or pleasure in doing things?",
        "question_ur": "آپ کتنی بار چیزوں میں دلچسپی یا خوشی محسوس نہیں کی؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "feeling_down",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you been feeling down, depressed, or hopeless?",
        "question_ur": "آپ کتنی بار اداسی، مایوسی یا ناامیدی محسوس کی؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "sleep_trouble",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you had trouble falling or staying asleep, or sleeping too much?",
        "question_ur": "آپ کتنی بار نیند نہ آنے یا بہت زیادہ سونے کی تکلیف رہی؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "feeling_tired",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you been feeling tired or having little energy?",
        "question_ur": "آپ کتنی بار تھکاوٹ یا توانائی کی کمی محسوس کی؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "appetite",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you had poor appetite or been overeating?",
        "question_ur": "آپ کتنی بار بھوک کم لگی یا ضرورت سے زیادہ کھایا؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "feeling_bad",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you felt bad about yourself, or that you are a failure?",
        "question_ur": "آپ کتنی بار خود کو برا یا ناکام سمجھا؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "concentration",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you had trouble concentrating on things such as studying?",
        "question_ur": "آپ کتنی بار پڑھائی میں توجہ مرکوز کرنا مشکل رہا؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "psychomotor",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you moved or spoken too slowly, or been unusually restless?",
        "question_ur": "آپ کتنی بار بہت آہستہ حرکت کی یا غیر معمولی بے چینی محسوس کی؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },

    {
        "col":        "self_harm_thoughts",
        "input_type": "radio",
        "min": 0,
        "max": 3,
        "question_en": "How often have you had thoughts that you would be better off dead, or of hurting yourself?",
        "question_ur": "آپ کتنی بار یہ خیال آیا کہ مر جانا بہتر ہے یا خود کو تکلیف دینے کا خیال آیا؟",
        "options_en": _PHQ_OPTS_EN,
        "options_ur": _PHQ_OPTS_UR,
    },
]