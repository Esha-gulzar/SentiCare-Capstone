# db.py — MongoDB integration for SentiCare
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

client = MongoClient(os.getenv("MONGO_URI"))
db     = client[os.getenv("MONGO_DB", "senticare")]

# ── Collections ───────────────────────────────────────────────────────────────
sessions_col     = db["sessions"]
biomarkers_col   = db["biomarkers"]
chat_history_col = db["chat_history"]
predictions_col  = db["predictions"]
feedback_col     = db["feedback"]

# ── TTL index: auto-delete sessions older than 90 days ───────────────────────
sessions_col.create_index("created_at", expireAfterSeconds=90 * 24 * 60 * 60)


# ─────────────────────────────────────────────────────────────────────────────
def save_session(session_id: str, lang: str, voice_dominant: str, voice_fusion: dict):
    """Called once after /voice-intro completes."""
    sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {
            "session_id":     session_id,
            "lang":           lang,
            "voice_dominant": voice_dominant,
            "voice_fusion":   voice_fusion,
            "created_at":     datetime.utcnow(),
        }},
        upsert=True,
    )


def save_biomarkers(session_id: str, biomarkers: dict, emotion_scores: dict):
    """Store raw voice biomarkers + fused emotion scores."""
    biomarkers_col.insert_one({
        "session_id":     session_id,
        "pitch":          biomarkers.get("pitch"),
        "tone":           biomarkers.get("tone"),
        "mfcc_mean":      biomarkers.get("mfcc_mean"),
        "emotion_scores": emotion_scores,
        "recorded_at":    datetime.utcnow(),
    })


def save_chat_message(session_id: str, stage: str, role: str, message: str):
    """Store every chat turn (user or assistant)."""
    chat_history_col.insert_one({
        "session_id": session_id,
        "stage":      stage,
        "role":       role,       # 'user' or 'assistant'
        "message":    message,
        "timestamp":  datetime.utcnow(),
    })


def save_prediction(
    session_id:       str,
    condition:        str,
    level:            str,
    screening_scores: dict,
    features:         dict,
    fusion_mode:      str,
    voice_dominant:   str,
):
    """Store final ML prediction + all inputs used."""
    predictions_col.insert_one({
        "session_id":       session_id,
        "condition":        condition,
        "level":            level,
        "screening_scores": screening_scores,
        "features":         features,
        "fusion_mode":      fusion_mode,
        "voice_dominant":   voice_dominant,
        "predicted_at":     datetime.utcnow(),
    })


def save_feedback(session_id: str, msg_idx, feedback_type: str, reward: int):
    """Store thumbs-up / thumbs-down feedback."""
    feedback_col.insert_one({
        "session_id": session_id,
        "msg_idx":    msg_idx,
        "type":       feedback_type,
        "reward":     reward,
        "timestamp":  datetime.utcnow(),
    })