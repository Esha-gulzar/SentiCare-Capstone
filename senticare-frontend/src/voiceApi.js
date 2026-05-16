// voiceApi.js — uses relative URL so it works on HuggingFace Spaces
// (no hardcoded localhost)

const VOICE_INTRO_URL = "/voice-intro";

/**
 * Send a recorded audio blob to the /voice-intro endpoint.
 *
 * @param {Blob}   blob       - The recorded audio blob
 * @param {string} ext        - File extension hint, e.g. "webm", "ogg", "mp4"
 * @param {string} sessionId  - Current session ID
 * @param {string} lang       - "en" or "ur"
 * @returns {Promise<object>} - Parsed JSON response from the backend
 */
export async function sendVoiceIntro(blob, ext, sessionId, lang) {
  const formData = new FormData();
  formData.append("audio", blob, `recording.${ext}`);
  formData.append("session_id", sessionId);
  formData.append("lang", lang);

  const response = await fetch(VOICE_INTRO_URL, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Voice intro failed (${response.status}): ${errorText}`);
  }

  return response.json();
}






// ##original
// // voiceApi.js — FIXED v6
// //
// // CHANGES vs v5:
// // ─────────────────────────────────────────────────────────────────────────────
// // • Added "depression" awareness note: the backend may now return
// //   dominant_emotion = "depressed". No API change needed here — the response
// //   shape is unchanged. VoiceCheckIn.jsx and voiceConstants.js handle the
// //   new label on the UI side.
// //
// // ⚠️ CRITICAL: NEVER set Content-Type manually for FormData requests.
// //    The browser must set it automatically so it includes the multipart
// //    boundary. Setting it manually breaks the upload.
// // ─────────────────────────────────────────────────────────────────────────────

// const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

// /**
//  * Send a raw audio blob to /voice-intro pipeline.
//  *
//  * @param {Blob}   audioBlob  - audio blob from MediaRecorder (webm/ogg/mp4)
//  * @param {string} ext        - file extension matching the blob container: "webm" | "ogg" | "mp4"
//  * @param {string} sessionId
//  * @param {string} lang       - "en" | "ur"
//  * @returns {Promise<{ transcript, dominant_emotion, fusion, biomarkers }>}
//  *
//  * dominant_emotion may be one of:
//  *   "anxious" | "stressed" | "sad" | "depressed" | "excited" | "neutral"
//  */
// export async function sendVoiceIntro(audioBlob, ext = "webm", sessionId, lang = "en") {
//   const formData = new FormData();

//   // filename must match the actual container format.
//   // "recording.wav" for a WebM blob confuses Flask's suffix detection
//   // and can make ffmpeg reject the file.
//   formData.append("audio",      audioBlob, `recording.${ext}`);
//   formData.append("session_id", sessionId || "");
//   formData.append("lang",       lang      || "en");

//   // ⚠️ NO manual Content-Type header — browser sets it with correct boundary
//   const response = await fetch(`${BASE_URL}/voice-intro`, {
//     method: "POST",
//     body:   formData,
//   });

//   if (!response.ok) {
//     let detail = `HTTP ${response.status}`;
//     try {
//       const err = await response.json();
//       detail = err.error || JSON.stringify(err);
//     } catch (_) { /* ignore */ }
//     throw new Error(`Voice intro failed: ${detail}`);
//   }

//   const data = await response.json();
//   if (data.error) throw new Error(data.error);
//   return data;
// }

// /**
//  * Send a chat message to /chat.
//  */
// export async function sendChatMessage({ sessionId, input, lang, policyMode = "default" }) {
//   const response = await fetch(`${BASE_URL}/chat`, {
//     method:  "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({
//       session_id:  sessionId,
//       input,
//       lang,
//       policy_mode: policyMode,
//     }),
//   });

//   if (!response.ok) throw new Error(`Chat request failed: HTTP ${response.status}`);
//   return response.json();
// }