from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from backend.chatbot.conversation_engine import ConversationEngine
import edge_tts
import asyncio
import io
import re
import hashlib

app = Flask(__name__)
CORS(app)
engine = ConversationEngine()

sessions = {}

# ── IN-MEMORY TTS CACHE ────────────────────────────────────────────────────────
_tts_cache = {}

VOICE = "en-US-AriaNeural"

def _synthesize(text):
    async def run():
        communicate = edge_tts.Communicate(text=text, voice=VOICE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf.read()

    return asyncio.run(run())


def get_tts_bytes(text):
    key = hashlib.md5(text.encode()).hexdigest()

    if key not in _tts_cache:
        _tts_cache[key] = _synthesize(text)

    audio_bytes = _tts_cache[key]

    return io.BytesIO(audio_bytes if isinstance(audio_bytes, (bytes, bytearray)) else bytes(audio_bytes))

# ── SCREENING QUESTIONS (0-3 scale) ───────────────────────────────────────────
SCREENING_QUESTIONS = [
    {"id": "feeling_nervous",      "question": "Over the past two weeks, how often have you felt nervous, anxious, or on edge? (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)"},
    {"id": "uncontrollable_worry", "question": "How often have you been unable to stop or control worrying? (0-3)"},
    {"id": "restlessness",         "question": "How often have you felt restless or hard to relax? (0-3)"},
    {"id": "feeling_down",         "question": "How often have you felt down, depressed, or hopeless? (0-3)"},
    {"id": "loss_of_interest",     "question": "How often have you had little interest or pleasure in doing things? (0-3)"},
    {"id": "fatigue",              "question": "How often have you felt tired or low in energy? (0-3)"},
    {"id": "overwhelmed",          "question": "How often have you felt overwhelmed or unable to cope with daily responsibilities? (0-3)"},
    {"id": "irritability",         "question": "How often have you been easily irritated or frustrated? (0-3)"},
]

# ── ANXIETY FEATURE QUESTIONS ─────────────────────────────────────────────────
ANXIETY_QUESTIONS = [
    {"col": "Age",                              "question": "What is your age?",                                                          "type": "number"},
    {"col": "Gender",                           "question": "What is your gender? (Male / Female / Other)",                               "type": "gender"},
    {"col": "Occupation",                       "question": "What is your occupation? (Student / Employed / Self-employed / Unemployed / Other)", "type": "select", "options": ["student", "employed", "self-employed", "unemployed", "other"]},
    {"col": "Sleep Hours",                      "question": "How many hours do you sleep per night? (e.g. 6)",                            "type": "number"},
    {"col": "Physical Activity (hrs/week)",     "question": "How many hours per week do you exercise? (e.g. 3)",                          "type": "number"},
    {"col": "Caffeine Intake (mg/day)",         "question": "How much caffeine do you consume per day in mg? (e.g. 100)",                 "type": "number"},
    {"col": "Alcohol Consumption (drinks/week)","question": "How many alcoholic drinks per week? (e.g. 2, enter 0 if none)",             "type": "number"},
    {"col": "Smoking",                          "question": "Do you smoke? (Yes / No)",                                                   "type": "yesno"},
    {"col": "Family History of Anxiety",        "question": "Do you have a family history of anxiety? (Yes / No)",                       "type": "yesno"},
    {"col": "Stress Level (1-10)",              "question": "Rate your current stress level from 1 to 10",                                "type": "number"},
    {"col": "Heart Rate (bpm)",                 "question": "What is your resting heart rate in beats per minute? (e.g. 75)",             "type": "number"},
    {"col": "Breathing Rate (breaths/min)",     "question": "What is your breathing rate in breaths per minute? (e.g. 16)",               "type": "number"},
    {"col": "Sweating Level (1-5)",             "question": "Rate your sweating level from 1 (none) to 5 (excessive)",                    "type": "number"},
    {"col": "Dizziness",                        "question": "Have you experienced dizziness recently? (Yes / No)",                        "type": "yesno"},
    {"col": "Medication",                       "question": "Are you currently taking any medication for anxiety? (Yes / No)",            "type": "yesno"},
    {"col": "Therapy Sessions (per month)",     "question": "How many therapy sessions do you attend per month? (e.g. 2, enter 0 if none)","type": "number"},
    {"col": "Recent Major Life Event",          "question": "Have you experienced a major life event recently? (Yes / No)",               "type": "yesno"},
    {"col": "Diet Quality (1-10)",              "question": "Rate your diet quality from 1 (very poor) to 10 (excellent)",                "type": "number"},
]

# ── STRESS FEATURE QUESTIONS ──────────────────────────────────────────────────
STRESS_QUESTIONS = [
    {"col": "Gender",                                                    "question": "What is your gender? (Male / Female)",                                           "type": "stress_gender"},
    {"col": "Age",                                                       "question": "What is your age? (e.g. 22)",                                                    "type": "number"},
    {"col": "Have you recently experienced stress in your life?",        "question": "How often have you recently experienced stress? (1=Never, 5=Always)",            "type": "stress_scale"},
    {"col": "Have you noticed a rapid heartbeat or palpitations?",       "question": "How often do you notice a rapid heartbeat? (1=Never, 5=Always)",                 "type": "stress_scale"},
    {"col": "Have you been dealing with anxiety or tension recently?",   "question": "How often do you deal with anxiety or tension? (1=Never, 5=Always)",             "type": "stress_scale"},
    {"col": "Do you face any sleep problems or difficulties falling asleep?", "question": "How often do you face sleep problems? (1=Never, 5=Always)",                 "type": "stress_scale"},
    {"col": "Have you been getting headaches more often than usual?",    "question": "How often do you get headaches? (1=Never, 5=Always)",                            "type": "stress_scale"},
    {"col": "Do you get irritated easily?",                              "question": "How often do you get irritated easily? (1=Never, 5=Always)",                     "type": "stress_scale"},
    {"col": "Do you have trouble concentrating on your academic tasks?", "question": "How often do you have trouble concentrating? (1=Never, 5=Always)",               "type": "stress_scale"},
    {"col": "Have you been feeling sadness or low mood?",                "question": "How often do you feel sadness or low mood? (1=Never, 5=Always)",                 "type": "stress_scale"},
    {"col": "Do you feel overwhelmed with your academic workload?",      "question": "How often do you feel overwhelmed with your workload? (1=Never, 5=Always)",      "type": "stress_scale"},
    {"col": "Is your working environment unpleasant or stressful?",      "question": "How stressful is your working or study environment? (1=Not at all, 5=Extremely)","type": "stress_scale"},
]

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


def validate_input(user_input, field_type, options, question_text):
    """Returns (is_valid, cleaned_value, error_message)."""
    val = user_input.strip()

    if field_type == "number":
        try:
            float(val)
            return True, val, None
        except ValueError:
            return False, None, f"⚠️ Please enter a valid number.\n👉 {question_text}"

    elif field_type == "yesno":
        if val.lower() in ["yes", "no"]:
            return True, val.capitalize(), None
        return False, None, f"⚠️ Please type Yes or No.\n👉 {question_text}"

    elif field_type == "gender":
        if val.lower() in ["male", "female", "other"]:
            return True, val.capitalize(), None
        return False, None, f"⚠️ Please type Male, Female, or Other.\n👉 {question_text}"

    elif field_type == "select" and options:
        if val.lower() in options:
            return True, val.capitalize(), None
        opts = " / ".join([o.capitalize() for o in options])
        return False, None, f"⚠️ Please choose one of: {opts}\n👉 {question_text}"

    elif field_type == "stress_scale":
        try:
            val_int = int(val)
            if 1 <= val_int <= 5:
                return True, val_int, None
            return False, None, f"⚠️ Please enter a number between 1 and 5.\n👉 {question_text}"
        except ValueError:
            return False, None, f"⚠️ Please enter a number between 1 and 5.\n👉 {question_text}"

    elif field_type == "stress_gender":
        if val.lower() == "male":
            return True, 0, None
        elif val.lower() == "female":
            return True, 1, None
        return False, None, f"⚠️ Please type Male or Female.\n👉 {question_text}"

    return True, val, None


def map_to_level(prediction, condition):
    """Map model output to low / medium / high."""
    try:
        val = float(prediction)
    except (TypeError, ValueError):
        return "medium"

    if condition == "anxiety":
        if val <= 2.5:
            return "low"
        elif val <= 4:
            return "medium"
        else:
            return "high"
    else:
        # Stress classifier: 0=Distress(high), 1=Eustress(medium), 2=No Stress(low)
        if val == 0:
            return "high"
        elif val == 1:
            return "medium"
        else:
            return "low"


def clean_tts_text(text):
    """Strip emojis, markdown, and extra whitespace for TTS."""
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)
    text = re.sub(r'[*_#]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ════════════════════════════════════════════════════════════════
#  TTS ENDPOINT — streaming + cached, supports GET and POST
# ════════════════════════════════════════════════════════════════
@app.route("/tts", methods=["GET", "POST"])
def tts():
    # Accept text from either GET query param or POST JSON body
    if request.method == "GET":
        text = request.args.get("text", "").strip()
    else:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()

    text = clean_tts_text(text)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        mp3_fp = get_tts_bytes(text)   # cached synthesis

        def generate():
            chunk_size = 4096          # stream in 4 KB chunks
            while True:
                chunk = mp3_fp.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        return Response(
            stream_with_context(generate()),
            mimetype="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "Accept-Ranges": "bytes",
            }
        )
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════
#  CHAT ENDPOINT
# ════════════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_input = data.get("input", "").strip()

    if session_id not in sessions:
        sessions[session_id] = {
            "stage":             "greeting",
            "screening_answers": {},
            "screening_index":   0,
            "feature_answers":   {},
            "feature_index":     0,
            "condition":         None,
        }

    session = sessions[session_id]

    # ── GREETING ──────────────────────────────────────────────────
    if session["stage"] == "greeting":
        session["stage"] = "pre_screening"
        return jsonify({"message":
            "Hello! I am SentiCare, your mental health support assistant. "
            "Let me ask you a few questions to understand how you are feeling."
        })

    if session["stage"] == "pre_screening":
        session["stage"] = "screening"
        session["screening_index"] = 0
        return jsonify({"message": SCREENING_QUESTIONS[0]["question"]})

    # ── SCREENING ─────────────────────────────────────────────────
    if session["stage"] == "screening":
        idx = session["screening_index"]

        try:
            val = int(user_input)
            if val < 0 or val > 3:
                raise ValueError
        except (ValueError, TypeError):
            return jsonify({"message":
                "⚠️ Please enter a number between 0 and 3.\n👉 "
                + SCREENING_QUESTIONS[idx]["question"]
            })

        session["screening_answers"][SCREENING_QUESTIONS[idx]["id"]] = val
        idx += 1
        session["screening_index"] = idx

        if idx < len(SCREENING_QUESTIONS):
            return jsonify({"message": SCREENING_QUESTIONS[idx]["question"]})

        # Screening done — determine condition
        scores = engine.calculate_screening_scores(session["screening_answers"])
        condition = engine.determine_condition(scores)
        session["condition"] = condition
        session["stage"] = "features"
        session["feature_index"] = 0

        questions = ANXIETY_QUESTIONS if condition == "anxiety" else STRESS_QUESTIONS
        return jsonify({"message":
            f"Thank you. Based on your responses, it looks like you may be experiencing "
            f"some {condition}-related symptoms. I have a few more specific questions.\n\n"
            + questions[0]["question"]
        })

    # ── FEATURE QUESTIONS ─────────────────────────────────────────
    if session["stage"] == "features":
        condition = session["condition"]
        questions = ANXIETY_QUESTIONS if condition == "anxiety" else STRESS_QUESTIONS
        idx = session["feature_index"]
        current_q = questions[idx]

        is_valid, cleaned_value, error_msg = validate_input(
            user_input,
            current_q["type"],
            current_q.get("options", []),
            current_q["question"]
        )

        if not is_valid:
            return jsonify({"message": error_msg})

        session["feature_answers"][current_q["col"]] = cleaned_value
        idx += 1
        session["feature_index"] = idx

        if idx < len(questions):
            return jsonify({"message": questions[idx]["question"]})

        # All questions answered — predict
        session["stage"] = "done"
        try:
            features = dict(session["feature_answers"])

            # Fill stress defaults for columns we did not ask
            if condition == "stress":
                for col, default in STRESS_DEFAULTS.items():
                    if col not in features:
                        features[col] = default

            if condition == "stress":
                scale_cols = [
                    "Have you recently experienced stress in your life?",
                    "Have you noticed a rapid heartbeat or palpitations?",
                    "Have you been dealing with anxiety or tension recently?",
                    "Do you face any sleep problems or difficulties falling asleep?",
                    "Have you been getting headaches more often than usual?",
                    "Do you get irritated easily?",
                    "Do you have trouble concentrating on your academic tasks?",
                    "Have you been feeling sadness or low mood?",
                    "Do you feel overwhelmed with your academic workload?",
                    "Is your working environment unpleasant or stressful?",
                ]
                scores = []
                for col in scale_cols:
                    try:
                        scores.append(float(features.get(col, 3)))
                    except Exception:
                        scores.append(3)
                avg = sum(scores) / len(scores)
                print(f"DEBUG STRESS avg={avg}", flush=True)
                if avg <= 2:
                    level = "low"
                elif avg <= 3.5:
                    level = "medium"
                else:
                    level = "high"
            else:
                prediction = engine.run_prediction(condition, features)
                print(f"DEBUG FINAL >>> condition={condition}, prediction={prediction}", flush=True)
                level = map_to_level(prediction, condition)

            response = engine.generate_cbt_response(condition, level)

            if response:
                steps = " | ".join(response["steps"])
                return jsonify({
                "message": f"🔍 Result: {condition} — {level.upper()} level\n\n"
                + response["validation"] + "\n\n"
                + "💚 " + response["grounding"] + "\n\n"
                + "📋 Steps: " + " | ".join(response["steps"])
                })

        except Exception as e:
            print("Prediction error:", e)

        return jsonify({"message":
            "Thank you for sharing. Based on what you told me, I recommend focusing on "
            "self-care, rest, and speaking to a professional if needed. You are not alone."
        })

    # ── SESSION COMPLETE ──────────────────────────────────────────
    return jsonify({"message":
        "Thank you for using SentiCare. Please refresh to start a new session."
    })


# ════════════════════════════════════════════════════════════════
#  OTHER ENDPOINTS
# ════════════════════════════════════════════════════════════════
@app.route("/screening", methods=["POST"])
def screening():
    data = request.json
    answers = data.get("answers")
    scores = engine.calculate_screening_scores(answers)
    condition = engine.determine_condition(scores)
    print(f"DEBUG scores: {scores}")
    print(f"DEBUG condition: {condition}")
    questions = engine.get_feature_questions(condition)
    return jsonify({"condition": condition, "questions": questions})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    condition = data.get("condition")
    features = data.get("features")
    prediction = engine.run_prediction(condition, features)
    level = map_to_level(prediction, condition)
    print(f"DEBUG >>> condition={condition}, prediction={prediction}, level={level}")
    response = engine.generate_cbt_response(condition, level)
    return jsonify({"prediction": prediction, "level": level, "cbt_response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
