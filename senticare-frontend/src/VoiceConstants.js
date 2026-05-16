// voiceConstants.js
// All UI strings for the VoiceCheckIn screen (English and Urdu).
// Import in VoiceCheckIn.jsx: import { VOICE_UI } from "./voiceConstants";
//
// IMPORTANT — how to render emotion labels in the UI:
//
//   The backend returns an English key such as "anxious", "excited", etc.
//   NEVER render this raw key. Always look it up:
//
//     const lang = "ur";  // or "en"
//     const label = result.dominant_emotion;          // e.g. "anxious"
//     const display = VOICE_UI[lang].emotions[label]  // e.g. "آپ کی آواز میں گھبراہٹ"
//                  ?? VOICE_UI[lang].emotions.unknown;
//
//   This is why the UI was showing "Excited" in English even in Urdu mode —
//   the component was rendering `label` directly instead of `emotions[label]`.

export const VOICE_UI = {
  en: {
    checkin_title:   "How are you feeling today?",
    checkin_sub:     "Speak freely for a few seconds. Tell us in your own words what's on your mind. We'll understand your emotional state from your voice.",
    skip:            "Skip — go straight to chat",
    start:           "Start speaking",
    stop:            "Done speaking",
    recording:       "Listening… speak freely",
    processing:      "Analysing your voice…",
    result_title:    "I heard you",
    result_said:     "You said:",
    result_emotion:  "Voice emotion detected:",
    continue_btn:    "Continue to assessment →",
    retry:           "Try again",
    unavailable:     "Voice analysis unavailable. Continuing with text assessment.",
    continue_anyway: "Continue anyway",
    emotions: {
      anxious:   "Anxiety detected in your voice",
      stressed:  "Stress detected in your voice",
      sad:       "Sadness detected in your voice",
      depressed: "Signs of depression detected in your voice",
      tense:     "Tension detected in your voice",
      excited:   "Excitement detected in your voice",
      neutral:   "Your voice sounds calm",
      unknown:   "Voice noted",
    },
  },

  ur: {
    checkin_title:   "آج آپ کیسا محسوس کر رہے ہیں؟",
    checkin_sub:     "چند سیکنڈ کے لیے آزادانہ بولیں۔ اپنے الفاظ میں بتائیں کہ آپ کے ذہن میں کیا ہے۔ ہم آپ کی آواز سے آپ کی کیفیت سمجھیں گے۔",
    skip:            "چھوڑیں — سیدھا چیٹ پر جائیں",
    start:           "بولنا شروع کریں",
    stop:            "بولنا بند کریں",
    recording:       "سن رہے ہیں… آزادانہ بولیں",
    processing:      "آپ کی آواز تجزیہ ہو رہی ہے…",
    result_title:    "میں نے سنا",
    result_said:     "آپ نے کہا:",
    result_emotion:  "آواز سے محسوس ہوا:",
    continue_btn:    "جائزہ جاری رکھیں ←",
    retry:           "دوبارہ کوشش کریں",
    unavailable:     "آواز کا تجزیہ دستیاب نہیں۔ متنی جائزے پر جاری ہے۔",
    continue_anyway: "جاری رکھیں",
    emotions: {
      anxious:   "آپ کی آواز میں گھبراہٹ محسوس ہوئی",
      stressed:  "آپ کی آواز میں دباؤ محسوس ہوا",
      sad:       "آپ کی آواز میں اداسی محسوس ہوئی",
      depressed: "آپ کی آواز میں ڈپریشن کے آثار محسوس ہوئے",
      tense:     "آپ کی آواز میں تناؤ محسوس ہوا",
      excited:   "آپ کی آواز میں جوش محسوس ہوا",
      neutral:   "آپ کی آواز پرسکون لگ رہی ہے",
      unknown:   "آواز نوٹ ہو گئی",
    },
  },
};

export const VOICE_INTRO_URL = "http://localhost:5000/voice-intro";

// ── Helper: get translated emotion display string ─────────────────────────
// Use this everywhere you render an emotion label from the backend.
//
// Usage:
//   import { VOICE_UI, getEmotionDisplay } from "./voiceConstants";
//   const display = getEmotionDisplay("anxious", "ur");
//   // → "آپ کی آواز میں گھبراہٹ محسوس ہوئی"
//
export function getEmotionDisplay(emotionKey, lang = "en") {
  const ui = VOICE_UI[lang] ?? VOICE_UI["en"];
  return ui.emotions[emotionKey] ?? ui.emotions["unknown"];
}