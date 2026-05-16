# app.py — v14
#
# CHANGES vs v13:
# ─────────────────────────────────────────────────────────────────────────────
# ADDED: Flask now serves the React frontend build from ./static/
#
# The Dockerfile builds the React app and copies the output into ./static/.
# Flask serves index.html for any non-API route so React Router works.
# ALL OTHER LOGIC IDENTICAL TO v13.
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS

from backend.conversation_engine import ConversationEngine
from backend.chatbot.emotion_fusion_combiner import EmotionFusionCombiner
from backend.chatbot.questions.anxiety_questions    import ANXIETY_FEATURE_QUESTIONS
from backend.chatbot.questions.stress_questions     import STRESS_FEATURE_QUESTIONS
from backend.chatbot.questions.depression_questions import DEPRESSION_FEATURE_QUESTIONS
import edge_tts
import asyncio
import io
import re
import hashlib
import os
import tempfile
import traceback as _tb

from backend.voice_input_handler import VoiceInputHandler
from backend.stt import STT


# ── MongoDB helpers ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import (
    save_session,
    save_biomarkers,
    save_chat_message,
    save_prediction,
    save_feedback,
)


# ── Static folder points to the React build output ───────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

engine    = ConversationEngine()
sessions: dict = {}

VOICE_EN = "en-US-AriaNeural"
VOICE_UR = "ur-PK-UzmaNeural"
_tts_cache: dict = {}

SUPPORTED_LANGUAGES = {"en", "ur"}

_MIME_TO_EXT = {
    "ogg":  ".ogg",
    "mp4":  ".mp4",
    "mpeg": ".mp3",
    "wav":  ".wav",
    "webm": ".webm",
}


def _audio_suffix(audio_file) -> str:
    filename = audio_file.filename or ""
    ext = os.path.splitext(filename)[1]
    if ext:
        return ext
    ct = (audio_file.content_type or "").lower()
    for fragment, suffix in _MIME_TO_EXT.items():
        if fragment in ct:
            return suffix
    return ".webm"


# ══════════════════════════════════════════════════════════════════════════════
#  REACT FRONTEND SERVING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def serve_react():
    """Serve the React app's index.html."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """
    Serve React static assets (JS, CSS, images).
    For any unknown path (React routes), fall back to index.html
    so client-side routing works correctly.
    """
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(STATIC_DIR, path)
    # Unknown path → let React Router handle it
    return send_from_directory(STATIC_DIR, "index.html")


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE INTRO ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/voice-intro", methods=["POST"])
def voice_intro():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    session_id = request.form.get("session_id", "anonymous")
    raw_lang   = request.form.get("lang", "en").strip().lower()
    lang       = raw_lang if raw_lang in SUPPORTED_LANGUAGES else "en"
    suffix     = _audio_suffix(audio_file)

    tmp        = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    audio_path = tmp.name
    tmp.close()

    print(
        f"[voice-intro] session={session_id}  lang={lang}  "
        f"filename={audio_file.filename}  suffix={suffix}",
        flush=True,
    )

    try:
        audio_file.save(audio_path)
        file_size = os.path.getsize(audio_path)
        print(f"[voice-intro] Saved: {audio_path}  ({file_size} bytes)", flush=True)

        if file_size < 100:
            return jsonify({
                "transcript":       "",
                "dominant_emotion": "neutral",
                "fusion":           {"anxiety": 0.0, "stress": 0.0,
                                     "sadness": 0.0, "depression": 0.0, "joy": 0.0},
                "biomarkers":       {"pitch": 0.0, "tone": 0.0, "mfcc_mean": 0.0},
                "warning":          "Audio upload was too small.",
            }), 200

        handler = VoiceInputHandler()
        result  = handler.run_pipeline(audio_path, lang=lang)
        print(f"[voice-intro] Pipeline result: {result}", flush=True)

        sess = sessions.setdefault(session_id, {
            "stage":             "greeting",
            "lang":              lang,
            "screening_answers": {},
            "screening_index":   0,
            "feature_answers":   {},
            "feature_index":     0,
            "condition":         None,
            "voice_fusion":      {},
            "voice_dominant":    "neutral",
        })
        sess["voice_fusion"]   = result.get("voice_fusion_for_ml", {})
        sess["voice_dominant"] = result.get("dominant_emotion", "neutral")

        # ── MongoDB: save session + biomarkers ────────────────────────────
        try:
            save_session(
                session_id,
                lang,
                sess["voice_dominant"],
                sess["voice_fusion"],
            )
            save_biomarkers(
                session_id,
                result.get("biomarkers", {}),
                result.get("fusion", {}),
            )
            print(f"[voice-intro] ✅ MongoDB saved session + biomarkers", flush=True)
        except Exception as db_exc:
            print(f"[voice-intro] ⚠️  MongoDB save failed: {db_exc}", flush=True)
        # ─────────────────────────────────────────────────────────────────

        print(
            f"[voice-intro] Stored → "
            f"voice_fusion={sess['voice_fusion']}  "
            f"voice_dominant='{sess['voice_dominant']}'",
            flush=True,
        )

        return jsonify(result), 200

    except Exception as exc:
        app.logger.error(f"[voice-intro] ERROR: {exc}")
        _tb.print_exc()
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
        "greeting":             "Hello! I am SentiCare, your mental health support assistant. Let me ask you a few questions to understand how you are feeling.",
        "thank_you":            "Thank you. Based on your responses, it looks like you may be experiencing some {condition}-related symptoms. I have a few more specific questions for you.",
        "result":               "Result: {condition} — {level} level",
        "condition_anxiety":    "Anxiety",
        "condition_stress":     "Stress",
        "condition_depression": "Depression",
        "level_low":            "LOW",
        "level_medium":         "MEDIUM",
        "level_high":           "HIGH",
        "fallback":             "Thank you for sharing. Based on what you told me, I recommend focusing on self-care, rest, and speaking to a professional if needed. You are not alone.",
        "session_done":         "Thank you for using SentiCare. Please start a new chat to continue.",
        "err_0_3":              "Please enter a number between 0 and 3.",
        "err_number":           "Please enter a valid number.",
        "err_range":            "Please enter a number between {lo} and {hi}.",
        "err_scale5":           "Please enter a number between 1 and 5.",
        "err_gender":           "Please select an option.",
        "err_required":         "Please answer this question before continuing.",
        "steps_label":          "Steps",
    },
    "ur": {
        "greeting":             "السلام علیکم! میں سینٹی کیئر ہوں، آپ کا ذہنی صحت کا معاون۔ آپ کی کیفیت سمجھنے کے لیے چند سوالات پوچھنا چاہتا ہوں۔",
        "thank_you":            "شکریہ۔ آپ کے جوابات کی بنیاد پر لگتا ہے آپ {condition} سے متعلق علامات محسوس کر رہے ہیں۔ چند اور مخصوص سوالات ہیں۔",
        "result":               "نتیجہ: {condition} — {level} سطح",
        "condition_anxiety":    "گھبراہٹ",
        "condition_stress":     "ذہنی دباؤ",
        "condition_depression": "ڈپریشن",
        "level_low":            "کم",
        "level_medium":         "درمیانہ",
        "level_high":           "زیادہ",
        "fallback":             "آپ کی بات سن کر اچھا لگا۔ خود کی دیکھ بھال کریں، آرام کریں، اور ضرورت ہو تو کسی ماہر سے رابطہ کریں۔ آپ اکیلے نہیں ہیں۔",
        "session_done":         "سینٹی کیئر استعمال کرنے کا شکریہ۔ نیا چیٹ شروع کریں۔",
        "err_0_3":              "براہ کرم 0 سے 3 کے درمیان نمبر درج کریں۔",
        "err_number":           "براہ کرم درست نمبر درج کریں۔",
        "err_range":            "براہ کرم {lo} سے {hi} کے درمیان نمبر درج کریں۔",
        "err_scale5":           "براہ کرم 1 سے 5 کے درمیان نمبر درج کریں۔",
        "err_gender":           "براہ کرم ایک آپشن منتخب کریں۔",
        "err_required":         "براہ کرم جاری رکھنے سے پہلے اس سوال کا جواب دیں۔",
        "steps_label":          "اقدامات",
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


def _screening_q(idx: int, lang: str) -> dict:
    q = _SCREENING_QS[idx]
    return {
        "id":       q["id"],
        "question": q["question_ur"] if lang == "ur" else q["question_en"],
        "options":  _SCREENING_OPTS_UR if lang == "ur" else _SCREENING_OPTS_EN,
    }


def _feature_qs(condition: str) -> list:
    if condition == "anxiety":    return ANXIETY_FEATURE_QUESTIONS
    if condition == "stress":     return STRESS_FEATURE_QUESTIONS
    if condition == "depression": return DEPRESSION_FEATURE_QUESTIONS
    return []


def _resolve_feature(q: dict, lang: str) -> dict:
    text  = q["question_ur"] if lang == "ur" else q["question_en"]
    itype = q["input_type"]

    if itype == "number":
        return {"question": text, "options": None}
    if itype == "scale_5":
        return {"question": text, "options": _SCALE5_UR if lang == "ur" else _SCALE5_EN}
    if itype in ("radio", "select", "stress_gender"):
        opts = q.get("options_ur" if lang == "ur" else "options_en")
        return {"question": text, "options": opts}
    if itype == "slider":
        raw_opts = q.get("options_ur" if lang == "ur" else "options_en")
        if raw_opts:
            return {"question": text, "options": raw_opts}
        lo = q.get("min", 0)
        hi = q.get("max", 3)
        if lo == 0 and hi == 3:
            return {"question": text, "options": _PHQ_OPTS_UR if lang == "ur" else _PHQ_OPTS_EN}
        return {"question": text, "options": None}
    return {"question": text, "options": None}


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_UR_MAP = {
    "ہاں": "Yes", "نہیں": "No", "نہيں": "No",
    "مرد": "Male", "عورت": "Female", "دیگر": "Other",
    "طالب علم": "Student", "ملازم": "Employed",
    "خود کاروبار": "Self-employed", "بے روزگار": "Unemployed",
    "پہلا سال یا مساوی":  "First Year or Equivalent",
    "دوسرا سال یا مساوی": "Second Year or Equivalent",
    "تیسرا سال یا مساوی": "Third Year or Equivalent",
    "چوتھا سال یا مساوی": "Fourth Year or Equivalent",
    "2.50 سے کم":  "Below 2.50",
    "بتانا نہیں چاہتے": "Prefer not to say",
}


def validate_feature_input(raw: str, q: dict, lang: str):
    val   = raw.strip()
    itype = q["input_type"]
    q_txt = q["question_ur"] if lang == "ur" else q["question_en"]

    if not val:
        return False, None, f"⚠️ {ui('err_required', lang)}\n👉 {q_txt}"

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
        opts_key = "options_ur" if lang == "ur" else "options_en"
        options  = q.get(opts_key) or []
        for opt in options:
            if isinstance(opt, dict) and opt.get("value") == val:
                return True, val, None
            if isinstance(opt, dict) and opt.get("label") == val:
                return True, opt.get("value", val), None
        mapped = _UR_MAP.get(val, val)
        return True, mapped, None

    if itype == "stress_gender":
        if val in ("0", "1"):
            return True, int(val), None
        if val.lower() in ("male", "مرد"):    return True, 0, None
        if val.lower() in ("female", "عورت"): return True, 1, None
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
#  LEVEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

def _map_level(prediction, condition: str) -> str:
    try:
        val = int(float(str(prediction)))
        if condition == "stress":
            return {0: "high", 1: "medium", 2: "low"}.get(val, "medium")
        return {0: "low", 1: "medium", 2: "high"}.get(val, "medium")
    except (TypeError, ValueError):
        pass
    s = str(prediction).strip().lower()
    if any(k in s for k in ("minimal", "none", "low", "no depression", "normal", "healthy")):
        return "low"
    if any(k in s for k in ("moderate", "medium", "mild")):
        return "medium"
    if any(k in s for k in ("severe", "high", "major")):
        return "high"
    print(
        f"[_map_level] ⚠️  Unrecognised prediction={prediction!r} for "
        f"condition='{condition}' — defaulting to medium.",
        flush=True,
    )
    return "medium"


# ══════════════════════════════════════════════════════════════════════════════
#  LEVEL LOOKUP DICTS
# ══════════════════════════════════════════════════════════════════════════════

_LEVEL_ORDER  = {"low": 0, "medium": 1, "high": 2}
_LEVEL_NAME   = {0: "low", 1: "medium", 2: "high"}
_LEVEL_TO_INT = {"low": 0, "medium": 1, "high": 2}


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITION LABEL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _condition_label(condition: str, lang: str) -> str:
    key_map = {
        "anxiety":    "condition_anxiety",
        "stress":     "condition_stress",
        "depression": "condition_depression",
    }
    return ui(key_map.get(condition, "condition_anxiety"), lang)


# ══════════════════════════════════════════════════════════════════════════════
#  CBT RESPONSE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_cbt_message(
    condition:      str,
    level:          str,
    lang:           str,
    policy_mode:    str = "default",
    voice_dominant: str = "neutral",
) -> str:
    cond_label  = _condition_label(condition, lang)
    level_label = ui(f"level_{level}", lang)
    result_line = ui("result", lang, condition=cond_label, level=level_label)
    steps_label = ui("steps_label", lang)

    print(
        f"[build_cbt] condition='{condition}'  level='{level}'  "
        f"lang='{lang}'  policy='{policy_mode}'  "
        f"voice_dominant='{voice_dominant}'",
        flush=True,
    )

    r = engine.generate_cbt_response(
        condition,
        level,
        lang=lang,
        voice_dominant=voice_dominant,
    )

    if not r:
        print(
            f"[build_cbt] ⚠️  No CBT response for condition='{condition}' "
            f"level='{level}'. Returning fallback.",
            flush=True,
        )
        return ui("fallback", lang)

    use_alt = (
        (policy_mode == "alternate" or r.get("prefer_alt_steps", False))
        and r.get("steps_alt")
    )

    steps_list = r["steps_alt"] if use_alt else r["steps"]
    steps      = " | ".join(steps_list)

    print(
        f"[build_cbt] steps_variant={'alt' if use_alt else 'default'}  "
        f"prefer_alt={r.get('prefer_alt_steps', False)}  "
        f"policy_mode={policy_mode}",
        flush=True,
    )

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
    body         = request.json or {}
    session_id   = body.get("session_id")
    user_input   = body.get("input", "").strip()
    lang         = body.get("lang", "en").strip()
    policy_mode  = body.get("policy_mode", "default")

    body_voice_fusion = body.get("voice_fusion") or {}

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
            "voice_fusion":      {},
            "voice_dominant":    "neutral",
        }

    sess = sessions[session_id]
    sess["lang"] = lang

    voice_fusion   = sess.get("voice_fusion") or body_voice_fusion or {}
    voice_dominant = sess.get("voice_dominant", "neutral")

    print(
        f"[chat] session={session_id}  stage={sess['stage']}  "
        f"voice_fusion={voice_fusion}  voice_dominant='{voice_dominant}'",
        flush=True,
    )

    # ── GREETING ──────────────────────────────────────────────────────────────
    if sess["stage"] == "greeting":
        sess["stage"] = "pre_screening"
        msg = ui("greeting", lang)
        try:
            save_chat_message(session_id, "greeting", "assistant", msg)
        except Exception as db_exc:
            print(f"[chat/greeting] ⚠️  MongoDB save failed: {db_exc}", flush=True)
        return jsonify({"message": msg, "stage": "pre_screening"})

    # ── PRE-SCREENING ─────────────────────────────────────────────────────────
    if sess["stage"] == "pre_screening":
        sess["stage"]           = "screening"
        sess["screening_index"] = 0
        q = _screening_q(0, lang)
        try:
            save_chat_message(session_id, "pre_screening", "assistant", q["question"])
        except Exception as db_exc:
            print(f"[chat/pre_screening] ⚠️  MongoDB save failed: {db_exc}", flush=True)
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

        try:
            save_chat_message(session_id, "screening", "user", user_input)
        except Exception as db_exc:
            print(f"[chat/screening] ⚠️  MongoDB save failed: {db_exc}", flush=True)

        sess["screening_answers"][_SCREENING_QS[idx]["id"]] = val
        idx += 1
        sess["screening_index"] = idx

        if idx < len(_SCREENING_QS):
            q = _screening_q(idx, lang)
            try:
                save_chat_message(session_id, "screening", "assistant", q["question"])
            except Exception as db_exc:
                print(f"[chat/screening] ⚠️  MongoDB save failed: {db_exc}", flush=True)
            return jsonify({"message": q["question"], "options": q["options"]})

        scores    = engine.calculate_screening_scores(sess["screening_answers"])
        condition = engine.determine_condition(scores)
        print(f"[chat/screening] scores={scores}  condition={condition}", flush=True)

        if condition == "neutral":
            sess["stage"] = "done"
            msg = ui("fallback", lang)
            try:
                save_chat_message(session_id, "screening", "assistant", msg)
            except Exception as db_exc:
                print(f"[chat/screening] ⚠️  MongoDB save failed: {db_exc}", flush=True)
            return jsonify({"message": msg})

        sess["condition"] = condition
        sess["stage"]     = "thankyou"
        cond_label        = _condition_label(condition, lang)
        msg = ui("thank_you", lang, condition=cond_label)
        try:
            save_chat_message(session_id, "thankyou", "assistant", msg)
        except Exception as db_exc:
            print(f"[chat/screening] ⚠️  MongoDB save failed: {db_exc}", flush=True)
        return jsonify({"message": msg, "stage": "thankyou"})

    # ── THANKYOU → first feature question ────────────────────────────────────
    if sess["stage"] == "thankyou":
        sess["stage"]         = "features"
        sess["feature_index"] = 0
        condition = sess["condition"]
        questions = _feature_qs(condition)
        info      = _resolve_feature(questions[0], lang)
        try:
            save_chat_message(session_id, "features", "assistant", info["question"])
        except Exception as db_exc:
            print(f"[chat/thankyou] ⚠️  MongoDB save failed: {db_exc}", flush=True)
        payload = {"message": info["question"]}
        if info["options"]:
            payload["options"] = info["options"]
        return jsonify(payload)

    # ── FEATURE QUESTIONS ─────────────────────────────────────────────────────
    if sess["stage"] == "features":
        condition = sess["condition"]
        questions = _feature_qs(condition)
        idx       = sess["feature_index"]
        current_q = questions[idx]

        if not user_input:
            info    = _resolve_feature(current_q, lang)
            payload = {"message": info["question"]}
            if info["options"]:
                payload["options"] = info["options"]
            return jsonify(payload)

        is_valid, cleaned, error_msg = validate_feature_input(
            user_input, current_q, lang
        )
        if not is_valid:
            info    = _resolve_feature(current_q, lang)
            payload = {"message": error_msg}
            if info["options"]:
                payload["options"] = info["options"]
            return jsonify(payload)

        try:
            save_chat_message(session_id, "features", "user", user_input)
        except Exception as db_exc:
            print(f"[chat/features] ⚠️  MongoDB save failed: {db_exc}", flush=True)

        sess["feature_answers"][current_q["col"]] = cleaned
        idx += 1
        sess["feature_index"] = idx

        if idx < len(questions):
            info    = _resolve_feature(questions[idx], lang)
            try:
                save_chat_message(session_id, "features", "assistant", info["question"])
            except Exception as db_exc:
                print(f"[chat/features] ⚠️  MongoDB save failed: {db_exc}", flush=True)
            payload = {"message": info["question"]}
            if info["options"]:
                payload["options"] = info["options"]
            return jsonify(payload)

        sess["stage"] = "done"
        level         = "medium"
        prediction    = None
        features      = dict(sess["feature_answers"])
        combined      = {}

        try:
            prediction = engine.run_prediction(condition, features)
        except Exception as exc:
            print(f"[chat/stageA] ❌ ML FAILED: {exc}", flush=True)
            _tb.print_exc()

        if prediction is not None:
            try:
                mapped = engine.map_prediction_to_level(prediction)
                level  = mapped if mapped else _map_level(prediction, condition)
            except Exception as exc:
                level = _map_level(prediction, condition)

        try:
            if prediction is not None:
                try:
                    level_int_for_fusion = int(prediction)
                except (TypeError, ValueError):
                    level_int_for_fusion = _LEVEL_TO_INT.get(level, 1)
            else:
                level_int_for_fusion = _LEVEL_TO_INT.get(level, 1)

            combined = EmotionFusionCombiner.combine(
                condition_text  = condition,
                level_int_text  = level_int_for_fusion,
                voice_fusion    = voice_fusion,
                voice_dominant  = voice_dominant,
                text_confidence = 0.70,
            )
            condition      = combined["condition"]
            level          = combined["level"]
            voice_dominant = combined["voice_dominant"]
        except Exception as exc:
            print(f"[chat/stageB] ⚠️  Fusion error: {exc}", flush=True)
            _tb.print_exc()

        try:
            cbt_text = build_cbt_message(
                condition, level, lang,
                policy_mode=policy_mode,
                voice_dominant=voice_dominant,
            )
        except Exception as exc:
            print(f"[chat/stageC] ⚠️  build_cbt_message error: {exc}", flush=True)
            cbt_text = ui("fallback", lang)

        try:
            screening_scores = engine.calculate_screening_scores(sess["screening_answers"])
            save_prediction(
                session_id       = session_id,
                condition        = condition,
                level            = level,
                screening_scores = screening_scores,
                features         = features,
                fusion_mode      = combined.get("fusion_mode", "unknown"),
                voice_dominant   = voice_dominant,
            )
            save_chat_message(session_id, "result", "assistant", cbt_text)
        except Exception as db_exc:
            print(f"[chat/result] ⚠️  MongoDB save failed: {db_exc}", flush=True)

        return jsonify({
            "message":        cbt_text,
            "level":          level,
            "condition":      condition,
            "voice_dominant": voice_dominant,
        })

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


@app.route("/debug-depression", methods=["POST"])
def debug_depression():
    body     = request.json or {}
    features = body.get("features", {})
    lang     = body.get("lang", "en")
    result   = {
        "features_received":   features,
        "feature_keys":        list(features.keys()),
        "raw_prediction":      None,
        "raw_prediction_type": None,
        "mapped_level":        None,
        "template_found":      False,
        "cbt_response":        None,
        "error":               None,
    }
    try:
        pred = engine.run_prediction("depression", features)
        result["raw_prediction"]      = str(pred)
        result["raw_prediction_type"] = type(pred).__name__
    except Exception as exc:
        result["error"] = f"PREDICTION ERROR: {exc}"
        _tb.print_exc()
        return jsonify(result), 200

    try:
        level = engine.map_prediction_to_level(pred) or _map_level(pred, "depression")
        result["mapped_level"] = level
    except Exception as exc:
        result["error"] = f"LEVEL MAPPING ERROR: {exc}"
        return jsonify(result), 200

    try:
        r = engine.generate_cbt_response("depression", level, lang=lang)
        result["template_found"] = r is not None
        result["cbt_response"]   = r
    except Exception as exc:
        result["error"] = f"CBT RESPONSE ERROR: {exc}"

    return jsonify(result), 200


@app.route("/debug-session/<session_id>", methods=["GET"])
def debug_session(session_id):
    sess = sessions.get(session_id)
    if not sess:
        return jsonify({"error": f"Session '{session_id}' not found"}), 404
    return jsonify(dict(sess)), 200


_feedback_log: list = []


@app.route("/feedback", methods=["POST"])
def feedback_route():
    body     = request.get_json(silent=True) or {}
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

    try:
        save_feedback(
            session_id    = entry["session_id"],
            msg_idx       = entry["msg_idx"],
            feedback_type = entry["type"],
            reward        = entry["reward"],
        )
    except Exception as db_exc:
        print(f"[feedback] ⚠️  MongoDB save failed: {db_exc}", flush=True)

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


@app.route("/debug-voice", methods=["POST"])
def debug_voice():
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
        handler = VoiceInputHandler()
        result  = handler.run_pipeline(audio_path, lang=lang)
        return jsonify(result), 200
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)