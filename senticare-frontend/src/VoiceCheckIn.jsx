// VoiceCheckIn.jsx — FIXED v8
//
// BUGS FIXED vs v7:
// ─────────────────────────────────────────────────────────────────────────────
// BUG 7 FIX — Emotion labels always showed English regardless of language.
//   EMOTION_META had hardcoded English strings ("Anxious", "Excited", etc).
//   FIX: Replaced static EMOTION_META object with getEmotionMeta(lang)
//   function that returns Urdu labels when lang="ur".
//
// BUG 8 FIX — autoGainControl + noiseSuppression + echoCancellation were ON.
//   Browser AGC was boosting mic gain before audio reached Python pipeline.
//   Combined with volume=8.0 in ffmpeg → double amplification → pitch always
//   ~280 Hz, tone always "Very loud" → always classified as "aroused"/"excited".
//   FIX: All three set to false so raw audio reaches the backend unmodified.
//
// ALL OTHER CODE IDENTICAL TO v7.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useRef, useCallback } from "react";
import { sendVoiceIntro } from "./voiceApi";
import "./VoiceCheckIn.css";

const MAX_RECORD_MS = 30_000;
const MIN_RECORD_MS = 2_000;

// BUG 7 FIX: language-aware emotion labels
const getEmotionMeta = (lang) => ({
  anxious:   { emoji: "😰", cls: "vc-em-anxious",   label: lang === "ur" ? "گھبراہٹ"     : "Anxious"   },
  stressed:  { emoji: "😤", cls: "vc-em-stressed",  label: lang === "ur" ? "دباؤ"        : "Stressed"  },
  sad:       { emoji: "😢", cls: "vc-em-sad",       label: lang === "ur" ? "اداسی"       : "Sad"       },
  depressed: { emoji: "😞", cls: "vc-em-depressed", label: lang === "ur" ? "ڈپریشن"      : "Depressed" },
  tense:     { emoji: "😬", cls: "vc-em-tense",     label: lang === "ur" ? "تناؤ"        : "Tense"     },
  neutral:   { emoji: "😐", cls: "vc-em-neutral",   label: lang === "ur" ? "معمول"       : "Neutral"   },
  excited:   { emoji: "😄", cls: "vc-em-excited",   label: lang === "ur" ? "جوش و خروش" : "Excited"   },
  unknown:   { emoji: "❓", cls: "vc-em-unknown",   label: lang === "ur" ? "نامعلوم"     : "Unknown"   },
});

const fmtPitch = (hz) => (!hz || hz === 0) ? "Not detected" : `${Math.round(hz)} Hz`;
const fmtTone  = (v) => {
  if (!v || v === 0) return "Silent";
  if (v < 0.01) return "Very soft";
  if (v < 0.03) return "Soft";
  if (v < 0.06) return "Moderate";
  if (v < 0.10) return "Loud";
  return "Very loud";
};
const pitchPct = (hz) => Math.min(100, Math.max(0, ((hz - 80) / 320) * 100));
const tonePct  = (v)  => Math.min(100, Math.max(0, (v / 0.12) * 100));

// Pick the best supported MIME type for this browser
function getBestMimeType() {
  const types = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function mimeToExt(mimeType) {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "mp4";
  return "webm";
}

export default function VoiceCheckIn({
  sessionId,
  lang      = "en",
  isRTL     = false,
  onComplete,
  onSkip,
}) {
  const [status,   setStatus]   = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [result,   setResult]   = useState(null);
  const [seconds,  setSeconds]  = useState(0);
  const [canStop,  setCanStop]  = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const streamRef        = useRef(null);
  const clockRef         = useRef(null);
  const minTimerRef      = useRef(null);
  const maxTimerRef      = useRef(null);

  const isUrdu = lang === "ur";

  // BUG 7 FIX: build EMOTION_META from lang so labels are in correct language
  const EMOTION_META = getEmotionMeta(lang);

  const L = {
    title:          isUrdu ? "آواز کی جانچ"                                      : "Voice Check-In",
    subtitle:       isUrdu ? "مائیکروفون بٹن دبائیں اور اپنی کیفیت بیان کریں۔" : "Press Start, speak for a few seconds about how you feel today.",
    recording:      isUrdu ? "ریکارڈنگ جاری ہے…"                                : "Recording…",
    processing:     isUrdu ? "آواز کا تجزیہ ہو رہا ہے…"                          : "Analysing your voice…",
    requesting:     isUrdu ? "مائیکروفون کی اجازت مانگی جا رہی ہے…"             : "Requesting microphone access…",
    stop:           isUrdu ? "⏹ روکیں"                                           : "⏹ Stop",
    start:          isUrdu ? "🎤 بولیں"                                           : "🎤 Start",
    again:          isUrdu ? "🔄 دوبارہ"                                          : "🔄 Try again",
    skip:           isUrdu ? "چھوڑیں"                                            : "Skip",
    continue_btn:   isUrdu ? "چیٹ جاری رکھیں ←"                                : "Continue to chat →",
    result_title:   isUrdu ? "آواز کا تجزیہ"                                     : "Voice analysis result",
    transcript:     isUrdu ? "آپ نے کہا:"                                        : "What we heard:",
    emotion:        isUrdu ? "غالب جذبہ"                                         : "Dominant emotion",
    pitch_label:    isUrdu ? "آواز کی بلندی (Pitch)"                             : "Voice pitch",
    tone_label:     isUrdu ? "آواز کی توانائی (Tone)"                            : "Voice energy",
    fusion_title:   isUrdu ? "جذباتی اشارے"                                      : "Emotional signals",
    anxiety_lbl:    isUrdu ? "گھبراہٹ"                                           : "Anxiety",
    stress_lbl:     isUrdu ? "دباؤ"                                              : "Stress",
    sadness_lbl:    isUrdu ? "اداسی"                                             : "Sadness",
    depression_lbl: isUrdu ? "ڈپریشن"                                            : "Depression",
    joy_lbl:        isUrdu ? "خوشی"                                              : "Joy",
    nothing:        isUrdu ? "(کچھ نہیں سنا گیا)"                               : "(nothing detected — please speak louder)",
    err_label:      isUrdu ? "خرابی:"                                            : "Error:",
    err_silent:     isUrdu
      ? "آواز بہت خاموش تھی۔ مائیکروفون کے قریب بولیں اور دوبارہ کوشش کریں۔"
      : "No audio detected. Please speak louder and try again.",
    keep_speaking:  isUrdu ? "بولتے رہیں…" : "Keep speaking…",
  };

  const _cleanup = useCallback(() => {
    if (clockRef.current)    { clearInterval(clockRef.current);  clockRef.current    = null; }
    if (minTimerRef.current) { clearTimeout(minTimerRef.current); minTimerRef.current = null; }
    if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null; }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    setErrorMsg("");
    setResult(null);
    setSeconds(0);
    setCanStop(false);
    chunksRef.current = [];
    setStatus("requesting");

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // BUG 8 FIX: all three disabled so browser does NOT pre-process
          // the audio before it reaches the Python pipeline.
          // autoGainControl=true was doubling the ffmpeg volume amplification
          // → pitch always ~280 Hz → tone always "Very loud" → always "aroused".
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl:  false,
          channelCount:     1,
        },
      });
    } catch (err) {
      const msg =
        err.name === "NotAllowedError"
          ? (isUrdu ? "مائیکروفون کی اجازت نہیں ملی۔ براہ کرم اجازت دیں۔"
                    : "Microphone permission denied. Please allow access and try again.")
          : err.name === "NotFoundError"
          ? (isUrdu ? "کوئی مائیکروفون نہیں ملا۔"
                    : "No microphone found. Please connect one.")
          : `Microphone error: ${err.message}`;
      setErrorMsg(msg);
      setStatus("error");
      return;
    }

    streamRef.current = stream;

    const mimeType = getBestMimeType();
    console.log(`[VoiceCheckIn] Using MediaRecorder mimeType: "${mimeType || '(browser default)'}"`);

    const recorderOptions = mimeType ? { mimeType } : {};
    let mr;
    try {
      mr = new MediaRecorder(stream, recorderOptions);
    } catch (e) {
      console.warn("[VoiceCheckIn] MediaRecorder with mimeType failed, using default:", e);
      mr = new MediaRecorder(stream);
    }

    mediaRecorderRef.current = mr;

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunksRef.current.push(e.data);
        console.log(`[VoiceCheckIn] chunk: ${e.data.size} bytes`);
      }
    };

    mr.onstop = async () => {
      _cleanup();
      setStatus("processing");

      const allChunks = chunksRef.current;
      const totalSize = allChunks.reduce((s, c) => s + c.size, 0);
      console.log(`[VoiceCheckIn] MediaRecorder stopped. Total: ${totalSize} bytes, chunks: ${allChunks.length}`);

      if (totalSize < 1000) {
        setErrorMsg(L.err_silent);
        setStatus("error");
        return;
      }

      const actualMime = mr.mimeType || mimeType || "audio/webm";
      const ext        = mimeToExt(actualMime);
      const blob       = new Blob(allChunks, { type: actualMime });

      console.log(`[VoiceCheckIn] Sending blob: ${blob.size} bytes, type: ${actualMime}, ext: ${ext}`);

      try {
        const data = await sendVoiceIntro(blob, ext, sessionId, lang);
        console.log("[VoiceCheckIn] API response:", data);
        setResult(data);
        setStatus("done");
      } catch (err) {
        console.error("[VoiceCheckIn] pipeline error:", err);
        setErrorMsg(err.message || "Voice analysis failed. Please try again.");
        setStatus("error");
      }
    };

    mr.start(250);
    setStatus("recording");

    clockRef.current    = setInterval(() => setSeconds(s => s + 1), 1000);
    minTimerRef.current = setTimeout(() => setCanStop(true), MIN_RECORD_MS);
    maxTimerRef.current = setTimeout(() => stopRecording(), MAX_RECORD_MS);

  }, [isUrdu, _cleanup, sessionId, lang, L.err_silent]);

  const stopRecording = useCallback(() => {
    if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null; }
    const mr = mediaRecorderRef.current;
    if (mr && mr.state === "recording") {
      mr.stop();
    }
  }, []);

  const reset = () => {
    _cleanup();
    setStatus("idle");
    setResult(null);
    setErrorMsg("");
    setSeconds(0);
    setCanStop(false);
    chunksRef.current = [];
  };

  const emo           = EMOTION_META[result?.dominant_emotion] || EMOTION_META.unknown;
  const fusion        = result?.fusion     || {};
  const bio           = result?.biomarkers || {};
  const anxietyPct    = Math.round((fusion.anxiety    || 0) * 100);
  const stressPct     = Math.round((fusion.stress     || 0) * 100);
  const sadnessPct    = Math.round((fusion.sadness    || 0) * 100);
  const depressionPct = Math.round((fusion.depression || 0) * 100);
  const joyPct        = Math.round((fusion.joy        || 0) * 100);

  return (
    <div className={`vc-root ${isRTL ? "vc-rtl" : ""}`}>
      <div className="vc-card">

        <div className="vc-header-row">
          <div className="vc-icon-ring">
            {status === "recording"  && <span className="vc-mic-anim">🎙️</span>}
            {status === "processing" && <span className="vc-spin-icon">⚙️</span>}
            {status === "done"       && <span>{emo.emoji}</span>}
            {status === "error"      && <span>⚠️</span>}
            {(status === "idle" || status === "requesting") && <span>🎤</span>}
          </div>
          <div>
            <h2 className="vc-title">{L.title}</h2>
            {status !== "done" && <p className="vc-subtitle">{L.subtitle}</p>}
            {status === "done" && <p className="vc-subtitle">{L.result_title}</p>}
          </div>
        </div>

        {status === "recording" && (
          <div className="vc-status-strip vc-recording">
            <div className="vc-wave">
              <div className="vc-bar"/><div className="vc-bar"/>
              <div className="vc-bar"/><div className="vc-bar"/><div className="vc-bar"/>
            </div>
            <span>{canStop ? L.recording : L.keep_speaking}</span>
            <span className="vc-clock">{seconds}s</span>
          </div>
        )}

        {(status === "requesting" || status === "processing") && (
          <div className="vc-status-strip vc-processing">
            <div className="vc-spinner"/>
            <span>{status === "requesting" ? L.requesting : L.processing}</span>
          </div>
        )}

        {status === "error" && (
          <div className="vc-error-box" role="alert">
            <strong>{L.err_label}</strong> {errorMsg}
          </div>
        )}

        {status === "done" && result && (
          <div className="vc-result-panel">

            <div className="vc-result-section">
              <div className="vc-result-label">{L.transcript}</div>
              <div className="vc-transcript-box">
                {result.transcript
                  ? `"${result.transcript}"`
                  : <span className="vc-muted">{L.nothing}</span>}
              </div>
            </div>

            <div className="vc-result-section">
              <div className="vc-result-label">{L.emotion}</div>
              <div className={`vc-emotion-big ${emo.cls}`}>
                <span className="vc-emotion-emoji">{emo.emoji}</span>
                <span className="vc-emotion-name">{emo.label}</span>
              </div>
            </div>

            <div className="vc-result-section vc-two-col">
              <div className="vc-biomarker-box">
                <div className="vc-result-label">{L.pitch_label}</div>
                <div className="vc-bar-track">
                  <div className="vc-bar-fill vc-fill-purple"
                       style={{ width: `${pitchPct(bio.pitch || 0)}%` }}/>
                </div>
                <div className="vc-bar-value">{fmtPitch(bio.pitch)}</div>
              </div>
              <div className="vc-biomarker-box">
                <div className="vc-result-label">{L.tone_label}</div>
                <div className="vc-bar-track">
                  <div className="vc-bar-fill vc-fill-teal"
                       style={{ width: `${tonePct(bio.tone || 0)}%` }}/>
                </div>
                <div className="vc-bar-value">{fmtTone(bio.tone)}</div>
              </div>
            </div>

            <div className="vc-result-section">
              <div className="vc-result-label">{L.fusion_title}</div>
              {[
                { lbl: L.anxiety_lbl,    pct: anxietyPct,    cls: "vc-fill-amber"  },
                { lbl: L.stress_lbl,     pct: stressPct,     cls: "vc-fill-coral"  },
                { lbl: L.sadness_lbl,    pct: sadnessPct,    cls: "vc-fill-blue"   },
                { lbl: L.depression_lbl, pct: depressionPct, cls: "vc-fill-indigo" },
                { lbl: L.joy_lbl,        pct: joyPct,        cls: "vc-fill-green"  },
              ].map(({ lbl, pct, cls }) => (
                <div className="vc-fusion-row" key={lbl}>
                  <span className="vc-fusion-label">{lbl}</span>
                  <div className="vc-bar-track vc-fusion-track">
                    <div className={`vc-bar-fill ${cls}`} style={{ width: `${pct}%` }}/>
                  </div>
                  <span className="vc-fusion-pct">{pct}%</span>
                </div>
              ))}
            </div>

          </div>
        )}

        <div className="vc-btn-row">
          {(status === "idle" || status === "error") && (
            <button className="vc-btn vc-btn-primary" onClick={startRecording}>
              {L.start}
            </button>
          )}
          {status === "recording" && (
            <button className="vc-btn vc-btn-danger" onClick={stopRecording} disabled={!canStop}>
              {canStop ? L.stop : `⏱ ${Math.max(0, 2 - seconds)}s…`}
            </button>
          )}
          {status === "done" && (
            <>
              <button className="vc-btn vc-btn-secondary" onClick={reset}>
                {L.again}
              </button>
              <button
                className="vc-btn vc-btn-primary vc-btn-continue"
                onClick={() => onComplete && onComplete(result)}
              >
                {L.continue_btn}
              </button>
            </>
          )}
          {status !== "done" && (
            <button
              className="vc-btn vc-btn-ghost"
              onClick={onSkip}
              disabled={status === "processing" || status === "requesting"}
            >
              {L.skip}
            </button>
          )}
        </div>

      </div>
    </div>
  );
}