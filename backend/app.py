# app.py — FIXED v2
#
# FIX vs previous version:
# ─────────────────────────────────────────────────────────────────────────────
# VOICE-INTRO + DEBUG-VOICE SUFFIX FIX:
#
#   Previous code:
#     suffix = os.path.splitext(audio_file.filename or "")[-1] or ".webm"
#
#   Bug: os.path.splitext("recording.webm") returns ("recording", ".webm")
#        so [-1] gives ".webm". That's correct when filename is set.
#        But os.path.splitext("") returns ("", "") so [-1] gives "" and the
#        fallback ".webm" fires. HOWEVER — the real failure was different:
#
#        When Chrome sends the blob as "recording.webm", filename IS set and
#        suffix IS ".webm". The input file is saved with the correct extension.
#        Strategy 1 (_decode_ffmpeg_pipe) then writes the output WAV via
#        Python's wave module with raw s16le bytes — and Whisper's internal
#        ffmpeg rejects that WAV as "Invalid data found".
#
#   That bug is fixed in voice_input_handler.py (Strategy 1 now tells ffmpeg
#   to write the WAV directly instead of piping raw PCM through wave module).
#
#   This file: suffix extraction is made more explicit and defensive.
#   Both /voice-intro and /debug-voice now use the same helper.
#
# NO OTHER CHANGES to chat, TTS, feedback, or screening endpoints.
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from backend.chatbot.conversation_engine  import ConversationEngine
from backend.chatbot.questions.anxiety_questions import ANXIETY_FEATURE_QUESTIONS
from backend.chatbot.questions.stress_questions  import STRESS_FEATURE_QUESTIONS, STRESS_DEFAULTS

import edge_tts
import asyncio
import io
import re
import hashlib

import os
import tempfile
from backend.voice_input_handler import VoiceInputHandler
from backend.stt import STT


app = Flask(__name__)
CORS(app)

engine = ConversationEngine()
sessions: dict = {}

VOICE_EN = "en-US-AriaNeural"
VOICE_UR = "ur-PK-UzmaNeural"
_tts_cache: dict = {}

SUPPORTED_LANGUAGES = {"en", "ur"}

# Maps mime type fragments to file extensions.
# Used as a fallback when the uploaded filename has no extension.
_MIME_TO_EXT = {
    "ogg":  ".ogg",
    "mp4":  ".mp4",
    "mpeg": ".mp3",
    "wav":  ".wav",
    "webm": ".webm",   # default / most common from Chrome + Firefox
}


def _audio_suffix(audio_file) -> str:
    """
    Derive the correct file extension for the uploaded audio blob.

    Priority:
      1. Extension from audio_file.filename  (set by voiceApi.js)
      2. Extension inferred from content_type
      3. Hard fallback: ".webm"

    voiceApi.js always sends filename="recording.{ext}" where ext is derived
    from the blob's mimeType, so priority 1 almost always wins.
    """
    filename = audio_file.filename or ""
    ext = os.path.splitext(filename)[1]   # e.g. ".webm", ".ogg", ""
    if ext:
        return ext

    # Fallback: infer from Content-Type header
    ct = (audio_file.content_type or "").lower()
    for fragment, suffix in _MIME_TO_EXT.items():
        if fragment in ct:
            return suffix

    return ".webm"


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE INTRO ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/voice-intro", methods=["POST"])
def voice_intro():
    """
    Receives audio from VoiceCheckIn.jsx.
    Runs the full pipeline: preprocess → STT → VoiceBiomarker → EmotionAnalyzer.
    Returns JSON with transcript, dominant_emotion, fusion scores, and biomarkers.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    session_id = request.form.get("session_id", "anonymous")

    raw_lang = request.form.get("lang", "en").strip().lower()
    lang     = raw_lang if raw_lang in SUPPORTED_LANGUAGES else "en"

    if raw_lang not in SUPPORTED_LANGUAGES:
        print(f"[voice-intro] Unsupported lang='{raw_lang}' — defaulting to 'en'.", flush=True)

    suffix     = _audio_suffix(audio_file)
    tmp        = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    audio_path = tmp.name
    tmp.close()

    print(
        f"[voice-intro] session={session_id}  lang={lang}  "
        f"filename={audio_file.filename}  content_type={audio_file.content_type}  "
        f"suffix={suffix}",
        flush=True,
    )

    try:
        audio_file.save(audio_path)
        file_size = os.path.getsize(audio_path)
        print(f"[voice-intro] Saved: {audio_path}  ({file_size} bytes)", flush=True)

        if file_size < 100:
            print("[voice-intro] Upload too small — likely empty recording.", flush=True)
            return jsonify({
                "transcript":       "",
                "dominant_emotion": "neutral",
                "fusion":           {"anxiety": 0.0, "stress": 0.0, "sadness": 0.0},
                "biomarkers":       {"pitch": 0.0, "tone": 0.0, "mfcc_mean": 0.0},
                "warning":          "Audio upload was too small.",
            }), 200

        handler = VoiceInputHandler()
        result  = handler.run_pipeline(audio_path, lang=lang)

        print(f"[voice-intro] Pipeline result: {result}", flush=True)
        return jsonify(result), 200

    except Exception as exc:
        import traceback
        app.logger.error(f"[voice-intro] ERROR session={session_id}: {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ══════════════════════════════════════════════════════════════════════════════
#  TTS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _synthesize(text: str, voice: str) -> bytes:
    async def _run():
        communicate = edge_tts.Communicate(text=text, voice=voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf.read()
    return asyncio.run(_run())


def get_tts_bytes(text: str, voice: str) -> io.BytesIO:
    key = hashlib.md5((text + voice).encode()).hexdigest()
    if key not in _tts_cache:
        _tts_cache[key] = _synthesize(text, voice)
    return io.BytesIO(_tts_cache[key])


def clean_tts_text(text: str) -> str:
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)
    text = re.sub(r'[*_#]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_voice(lang: str) -> str:
    return VOICE_UR if lang == "ur" else VOICE_EN


# ══════════════════════════════════════════════════════════════════════════════
#  UI STRINGS
# ══════════════════════════════════════════════════════════════════════════════

_UI = {
    "en": {
        "greeting":          "Hello! I am SentiCare, your mental health support assistant. Let me ask you a few questions to understand how you are feeling.",
        "thank_you":         "Thank you. Based on your responses, it looks like you may be experiencing some {condition}-related symptoms. I have a few more specific questions.",
        "result":            "Result: {condition} — {level} level",
        "condition_anxiety": "Anxiety",
        "condition_stress":  "Stress",
        "level_low":         "LOW",
        "level_medium":      "MEDIUM",
        "level_high":        "HIGH",
        "fallback":          "Thank you for sharing. Based on what you told me, I recommend focusing on self-care, rest, and speaking to a professional if needed. You are not alone.",
        "session_done":      "Thank you for using SentiCare. Please start a new chat to continue.",
        "err_0_3":           "Please enter a number between 0 and 3.",
        "err_number":        "Please enter a valid number.",
        "err_range":         "Please enter a number between {lo} and {hi}.",
        "err_scale5":        "Please enter a number between 1 and 5.",
        "err_gender":        "Please select Male or Female.",
        "steps_label":       "Steps",
    },
    "ur": {
        "greeting":          "السلام علیکم! میں سینٹی کیئر ہوں، آپ کا ذہنی صحت کا معاون۔ آپ کی کیفیت سمجھنے کے لیے چند سوالات پوچھنا چاہتا ہوں۔",
        "thank_you":         "شکریہ۔ آپ کے جوابات کی بنیاد پر لگتا ہے آپ {condition} سے متعلق علامات محسوس کر رہے ہیں۔ چند اور مخصوص سوالات ہیں۔",
        "result":            "نتیجہ: {condition} — {level} سطح",
        "condition_anxiety": "گھبراہٹ",
        "condition_stress":  "ذہنی دباؤ",
        "level_low":         "کم",
        "level_medium":      "درمیانہ",
        "level_high":        "زیادہ",
        "fallback":          "آپ کی بات سن کر اچھا لگا۔ خود کی دیکھ بھال کریں، آرام کریں، اور ضرورت ہو تو کسی ماہر سے رابطہ کریں۔ آپ اکیلے نہیں ہیں۔",
        "session_done":      "سینٹی کیئر استعمال کرنے کا شکریہ۔ نیا چیٹ شروع کریں۔",
        "err_0_3":           "براہ کرم 0 سے 3 کے درمیان نمبر درج کریں۔",
        "err_number":        "براہ کرم درست نمبر درج کریں۔",
        "err_range":         "براہ کرم {lo} سے {hi} کے درمیان نمبر درج کریں۔",
        "err_scale5":        "براہ کرم 1 سے 5 کے درمیان نمبر درج کریں۔",
        "err_gender":        "براہ کرم مرد یا عورت منتخب کریں۔",
        "steps_label":       "اقدامات",
    },
}


def ui(key: str, lang: str, **kw) -> str:
    s = _UI.get(lang, _UI["en"]).get(key, _UI["en"].get(key, key))
    return s.format(**kw) if kw else s


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENING QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

_SCREENING_QS = [
    {"id": "feeling_nervous",
     "question_en": "Over the past two weeks, how often have you felt nervous, anxious, or on edge?",
     "question_ur": "گزشتہ دو ہفتوں میں، آپ کتنی بار گھبراہٹ، بے چینی یا پریشانی محسوس کرتے رہے؟"},
    {"id": "uncontrollable_worry",
     "question_en": "How often have you been unable to stop or control worrying?",
     "question_ur": "آپ کتنی بار فکروں کو روک یا کنٹرول نہیں کر پائے؟"},
    {"id": "restlessness",
     "question_en": "How often have you felt restless or hard to relax?",
     "question_ur": "آپ کتنی بار بے چین یا سکون میں رہنا مشکل محسوس ہوا؟"},
    {"id": "feeling_down",
     "question_en": "How often have you felt down, depressed, or hopeless?",
     "question_ur": "آپ کتنی بار اداسی، مایوسی یا ناامیدی محسوس کی؟"},
    {"id": "loss_of_interest",
     "question_en": "How often have you had little interest or pleasure in doing things?",
     "question_ur": "آپ کتنی بار کسی کام میں دلچسپی یا خوشی محسوس نہیں ہوئی؟"},
    {"id": "fatigue",
     "question_en": "How often have you felt tired or low in energy?",
     "question_ur": "آپ کتنی بار تھکاوٹ یا توانائی کی کمی محسوس کی؟"},
    {"id": "overwhelmed",
     "question_en": "How often have you felt overwhelmed or unable to cope with daily responsibilities?",
     "question_ur": "آپ کتنی بار اپنی روزمرہ ذمہ داریاں نبھانے میں خود کو ناکارہ محسوس کیا؟"},
    {"id": "irritability",
     "question_en": "How often have you been easily irritated or frustrated?",
     "question_ur": "آپ کتنی بار جلدی چڑچڑاپن یا غصہ محسوس کیا؟"},
]

_SCREENING_OPTS_EN = [
    {"label": "0 — Not at all",             "value": "0"},
    {"label": "1 — Several days",            "value": "1"},
    {"label": "2 — More than half the days", "value": "2"},
    {"label": "3 — Nearly every day",        "value": "3"},
]
_SCREENING_OPTS_UR = [
    {"label": "0 — بالکل نہیں",       "value": "0"},
    {"label": "1 — کچھ دن",            "value": "1"},
    {"label": "2 — آدھے سے زیادہ دن", "value": "2"},
    {"label": "3 — تقریباً ہر روز",    "value": "3"},
]

_SCALE5_EN = [
    {"label": "1 — Never",     "value": "1"},
    {"label": "2 — Rarely",    "value": "2"},
    {"label": "3 — Sometimes", "value": "3"},
    {"label": "4 — Often",     "value": "4"},
    {"label": "5 — Always",    "value": "5"},
]
_SCALE5_UR = [
    {"label": "1 — کبھی نہیں",  "value": "1"},
    {"label": "2 — کبھی کبھار", "value": "2"},
    {"label": "3 — کبھی کبھی",  "value": "3"},
    {"label": "4 — اکثر",       "value": "4"},
    {"label": "5 — ہمیشہ",      "value": "5"},
]


def _screening_q(idx: int, lang: str) -> dict:
    q = _SCREENING_QS[idx]
    return {
        "id":       q["id"],
        "question": q["question_ur"] if lang == "ur" else q["question_en"],
        "options":  _SCREENING_OPTS_UR if lang == "ur" else _SCREENING_OPTS_EN,
    }


def _feature_qs(condition: str) -> list:
    return ANXIETY_FEATURE_QUESTIONS if condition == "anxiety" else STRESS_FEATURE_QUESTIONS


def _resolve_feature(q: dict, lang: str) -> dict:
    text  = q["question_ur"] if lang == "ur" else q["question_en"]
    itype = q["input_type"]
    opts  = None
    if itype == "scale_5":
        opts = _SCALE5_UR if lang == "ur" else _SCALE5_EN
    elif itype in ("radio", "select", "stress_gender"):
        opts = q.get("options_ur" if lang == "ur" else "options_en")
    return {"question": text, "options": opts}


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_UR_MAP = {
    "ہاں": "Yes", "نہیں": "No", "نہيں": "No",
    "مرد": "Male", "عورت": "Female", "دیگر": "Other",
    "طالب علم": "Student", "ملازم": "Employed",
    "خود کاروبار": "Self-employed", "بے روزگار": "Unemployed",
}


def validate_feature_input(raw: str, q: dict, lang: str):
    val   = raw.strip()
    itype = q["input_type"]
    q_txt = q["question_ur"] if lang == "ur" else q["question_en"]

    if itype in ("number", "slider"):
        try:
            num = float(val)
        except (ValueError, TypeError):
            return False, None, f"⚠️ {ui('err_number', lang)}\n👉 {q_txt}"
        lo, hi = q.get("min"), q.get("max")
        if (lo is not None and num < lo) or (hi is not None and num > hi):
            return False, None, f"⚠️ {ui('err_range', lang, lo=lo, hi=hi)}\n👉 {q_txt}"
        return True, int(num) if num == int(num) else num, None

    if itype in ("radio", "select"):
        return True, _UR_MAP.get(val, val), None

    if itype == "stress_gender":
        if val in ("0", "1"):
            return True, int(val), None
        if val.lower() in ("male", "مرد"):     return True, 0, None
        if val.lower() in ("female", "عورت"):  return True, 1, None
        return False, None, f"⚠️ {ui('err_gender', lang)}\n👉 {q_txt}"

    if itype == "scale_5":
        try:
            n = int(val)
            if 1 <= n <= 5:
                return True, n, None
        except (ValueError, TypeError):
            pass
        return False, None, f"⚠️ {ui('err_scale5', lang)}\n👉 {q_txt}"

    return True, val, None


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL FALLBACK MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def _map_level(prediction, condition: str) -> str:
    try:
        val = float(prediction)
    except (TypeError, ValueError):
        return "medium"
    if condition == "anxiety":
        if val <= 0:   return "low"
        elif val <= 1: return "medium"
        else:          return "high"
    else:
        if val == 0:   return "high"
        elif val == 1: return "medium"
        else:          return "low"


# ══════════════════════════════════════════════════════════════════════════════
#  CBT RESPONSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_cbt_message(condition: str, level: str, lang: str,
                      policy_mode: str = "default") -> str:
    cond_label  = ui("condition_anxiety" if condition == "anxiety"
                     else "condition_stress", lang)
    level_label = ui(f"level_{level}", lang)
    result_line = ui("result", lang, condition=cond_label, level=level_label)
    steps_label = ui("steps_label", lang)

    r = engine.generate_cbt_response(condition, level, lang=lang)
    if not r:
        return ui("fallback", lang)

    if policy_mode == "alternate" and r.get("steps_alt"):
        steps_list = r["steps_alt"]
    else:
        steps_list = r["steps"]

    steps = " | ".join(steps_list)
    return (
        f"🔍 {result_line}\n\n"
        f"{r['validation']}\n\n"
        f"💚 {r['grounding']}\n\n"
        f"📋 {steps_label}: {steps}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TTS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/tts", methods=["GET", "POST"])
def tts():
    if request.method == "GET":
        text = request.args.get("text", "").strip()
        lang = request.args.get("lang", "en").strip()
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text", "").strip()
        lang = body.get("lang", "en").strip()

    text = clean_tts_text(text)
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        mp3 = get_tts_bytes(text, get_voice(lang))

        def _stream():
            while True:
                chunk = mp3.read(4096)
                if not chunk:
                    break
                yield chunk

        return Response(
            stream_with_context(_stream()),
            mimetype="audio/mpeg",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as exc:
        print(f"[TTS ERROR] {exc}")
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/chat", methods=["POST"])
def chat():
    body        = request.json or {}
    session_id  = body.get("session_id")
    user_input  = body.get("input", "").strip()
    lang        = body.get("lang", "en").strip()
    policy_mode = body.get("policy_mode", "default")

    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    if policy_mode not in ("default", "alternate"):
        policy_mode = "default"

    if session_id not in sessions:
        sessions[session_id] = {
            "stage":             "greeting",
            "lang":              lang,
            "screening_answers": {},
            "screening_index":   0,
            "feature_answers":   {},
            "feature_index":     0,
            "condition":         None,
        }

    sess = sessions[session_id]
    sess["lang"] = lang

    # ── GREETING ─────────────────────────────────────────────────────────────
    if sess["stage"] == "greeting":
        sess["stage"] = "pre_screening"
        return jsonify({"message": ui("greeting", lang), "stage": "pre_screening"})

    # ── PRE-SCREENING ─────────────────────────────────────────────────────────
    if sess["stage"] == "pre_screening":
        sess["stage"]           = "screening"
        sess["screening_index"] = 0
        q = _screening_q(0, lang)
        return jsonify({"message": q["question"], "options": q["options"]})

    # ── SCREENING ─────────────────────────────────────────────────────────────
    if sess["stage"] == "screening":
        idx = sess["screening_index"]
        try:
            val = int(user_input)
            if not (0 <= val <= 3):
                raise ValueError
        except (ValueError, TypeError):
            q = _screening_q(idx, lang)
            return jsonify({
                "message": f"⚠️ {ui('err_0_3', lang)}\n👉 {q['question']}",
                "options": q["options"],
            })

        sess["screening_answers"][_SCREENING_QS[idx]["id"]] = val
        idx += 1
        sess["screening_index"] = idx

        if idx < len(_SCREENING_QS):
            q = _screening_q(idx, lang)
            return jsonify({"message": q["question"], "options": q["options"]})

        scores    = engine.calculate_screening_scores(sess["screening_answers"])
        condition = engine.determine_condition(scores)

        if condition == "neutral":
            sess["stage"] = "done"
            return jsonify({"message": ui("fallback", lang)})

        sess["condition"]     = condition
        sess["stage"]         = "features"
        sess["feature_index"] = 0

        print(f"[DEBUG] scores={scores}  condition={condition}", flush=True)

        questions  = _feature_qs(condition)
        cond_label = ui(
            "condition_anxiety" if condition == "anxiety"
            else "condition_stress", lang
        )
        first   = _resolve_feature(questions[0], lang)
        payload = {
            "message": ui("thank_you", lang, condition=cond_label)
                       + "\n\n" + first["question"]
        }
        if first["options"]:
            payload["options"] = first["options"]
        return jsonify(payload)

    # ── FEATURE QUESTIONS ─────────────────────────────────────────────────────
    if sess["stage"] == "features":
        condition = sess["condition"]
        questions = _feature_qs(condition)
        idx       = sess["feature_index"]
        current_q = questions[idx]

        is_valid, cleaned, error_msg = validate_feature_input(
            user_input, current_q, lang
        )

        if not is_valid:
            info    = _resolve_feature(current_q, lang)
            payload = {"message": error_msg}
            if info["options"]:
                payload["options"] = info["options"]
            return jsonify(payload)

        sess["feature_answers"][current_q["col"]] = cleaned
        idx += 1
        sess["feature_index"] = idx

        if idx < len(questions):
            info    = _resolve_feature(questions[idx], lang)
            payload = {"message": info["question"]}
            if info["options"]:
                payload["options"] = info["options"]
            return jsonify(payload)

        # ── All features collected — run prediction ───────────────────────
        sess["stage"] = "done"
        try:
            features = dict(sess["feature_answers"])
            level    = "medium"

            if condition == "stress":
                for col, default in STRESS_DEFAULTS.items():
                    if col not in features:
                        features[col] = default

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
                sc    = [float(features[c]) for c in scale_cols if c in features]
                avg   = sum(sc) / len(sc) if sc else 3.0
                level = "low" if avg <= 2.0 else "high" if avg > 3.5 else "medium"
                print(f"[DEBUG] stress avg={avg:.2f}  level={level}", flush=True)

            else:
                prediction = engine.run_prediction(condition, features)
                level      = (engine.map_prediction_to_level(prediction)
                              or _map_level(prediction, condition))
                print(
                    f"[DEBUG] anxiety prediction={prediction}  level={level}",
                    flush=True,
                )

            cbt_text = build_cbt_message(
                condition, level, lang,
                policy_mode=policy_mode,
            )
            return jsonify({
                "message": cbt_text,
                "level":   level,
            })

        except Exception as exc:
            print(f"[ERROR] {exc}", flush=True)
            return jsonify({"message": ui("fallback", lang)})

    return jsonify({"message": ui("session_done", lang)})


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/screening", methods=["POST"])
def screening_route():
    body      = request.json or {}
    scores    = engine.calculate_screening_scores(body.get("answers", {}))
    condition = engine.determine_condition(scores)
    return jsonify({
        "condition": condition,
        "questions": engine.get_feature_questions(condition),
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    body      = request.json or {}
    condition = body.get("condition")
    features  = body.get("features", {})
    lang      = body.get("lang", "en")
    pred      = engine.run_prediction(condition, features)
    level     = engine.map_prediction_to_level(pred) or _map_level(pred, condition)
    return jsonify({
        "prediction":   str(pred),
        "level":        level,
        "cbt_response": engine.generate_cbt_response(condition, level, lang=lang),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 4 — PPO REWARD LOGGING
# ══════════════════════════════════════════════════════════════════════════════

_feedback_log: list = []


@app.route("/feedback", methods=["POST"])
def feedback_route():
    body = request.get_json(silent=True) or {}

    required = ("session_id", "type")
    if not all(k in body for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    if body.get("type") not in ("up", "down"):
        return jsonify({"error": "type must be 'up' or 'down'"}), 400

    entry = {
        "session_id": body.get("session_id"),
        "msg_idx":    body.get("msg_idx"),
        "type":       body.get("type"),
        "reward":     1 if body.get("type") == "up" else -1,
        "timestamp":  body.get("timestamp"),
    }
    _feedback_log.append(entry)
    print(f"[FEEDBACK] {entry}", flush=True)

    total      = len(_feedback_log)
    positives  = sum(1 for e in _feedback_log if e["type"] == "up")
    avg_reward = (positives * 1 + (total - positives) * -1) / total if total else 0

    return jsonify({
        "status":        "recorded",
        "total_signals": total,
        "avg_reward":    round(avg_reward, 3),
    })


@app.route("/feedback/summary", methods=["GET"])
def feedback_summary():
    total      = len(_feedback_log)
    positives  = sum(1 for e in _feedback_log if e["type"] == "up")
    negatives  = total - positives
    avg_reward = (positives - negatives) / total if total else 0
    return jsonify({
        "total":       total,
        "thumbs_up":   positives,
        "thumbs_down": negatives,
        "avg_reward":  round(avg_reward, 3),
        "policy_recommendation": (
            "switch_to_alternate" if avg_reward < -0.3
            else "maintain_default"
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  DEBUG ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/debug-voice", methods=["POST"])
def debug_voice():
    """
    Diagnostic endpoint.
    curl -X POST http://localhost:5000/debug-voice \
         -F "audio=@/path/to/recording.webm" \
         -F "lang=en"
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio field in request"}), 400

    audio_file = request.files["audio"]
    lang       = request.form.get("lang", "en").strip().lower()
    lang       = lang if lang in SUPPORTED_LANGUAGES else "en"

    suffix     = _audio_suffix(audio_file)
    tmp        = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    audio_path = tmp.name
    tmp.close()

    try:
        audio_file.save(audio_path)
        print(
            f"[debug-voice] {os.path.getsize(audio_path)} bytes  "
            f"suffix={suffix}  lang={lang}",
            flush=True,
        )
        handler = VoiceInputHandler()
        result  = handler.run_pipeline(audio_path, lang=lang)
        return jsonify(result), 200
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return "SentiCare backend is running!"


if __name__ == "__main__":
    app.run(debug=False)