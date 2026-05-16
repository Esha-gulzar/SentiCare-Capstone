// App.jsx — v10
//
// CHANGES vs v9:
// ─────────────────────────────────────────────────────────────────────────────
// FIX: "Thank you. Based on your responses..." message was SILENT.
//
// ROOT CAUSE:
//   In the `thankyou` stage, `tyIdx` is assigned INSIDE the setMessages
//   updater callback, but speakMessage(data.message, tyIdx) was called
//   IMMEDIATELY after in the same synchronous tick — before React flushed
//   the state update. So tyIdx was still `undefined` when speakMessage ran,
//   causing it to silently fail or speak the wrong message.
//
//   Same issue existed in `pre_screening` for the greeting message.
//
// FIX:
//   Wrapped both speakMessage calls in setTimeout(..., 50) so React has
//   time to flush the state update before TTS tries to use the index.
//
// ALL OTHER LOGIC IDENTICAL TO v9.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
} from "recharts";
import "./App.css";
import VoiceCheckIn from "./VoiceCheckIn";

const API_URL = "http://localhost:5000/chat";
const TTS_URL = "http://localhost:5000/tts";
const MAX_INPUT_CHARS = 500;

// ════════════════════════════════════════════════════════════════
//  STORAGE HELPERS
// ════════════════════════════════════════════════════════════════
const local = {
  get: (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
  del: (k) => localStorage.removeItem(k),
};

const session = {
  get: (k) => { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => sessionStorage.setItem(k, JSON.stringify(v)),
  del: (k) => sessionStorage.removeItem(k),
};

// ── Therapy detection ───────────────────────────────────────────
const THERAPY_TRIGGERS = [
  "steps:", "exercise:", "technique:", "try this:", "practice:",
  "breathing", "grounding", "progressive muscle", "cognitive", "reframe",
  "cbt", "mindfulness", "relaxation", "coping", "strategy:", "tip:",
  "here is a", "here's a", "let's try", "i recommend",
  "اقدامات:", "سانس", "قدم",
];
const isTherapyMessage = (text) =>
  THERAPY_TRIGGERS.some(t => text.toLowerCase().includes(t)) && text.length > 120;

const parseTherapyCard = (text) => {
  const lines = text.split(/\n+/).filter(Boolean);
  const title = lines[0]?.length < 80 ? lines[0].replace(/[*_#🔍]/g, "").trim() : "Therapy Exercise";
  const steps = lines.slice(1).filter(l => l.trim().length > 0);
  return { title, steps };
};

// ── Input validation ────────────────────────────────────────────
const validateInput = (text, lang) => {
  const trimmed = text.trim();
  if (!trimmed) return lang === "ur" ? "پیغام خالی نہیں ہو سکتا۔" : "Message cannot be empty.";
  if (trimmed.length > MAX_INPUT_CHARS)
    return lang === "ur"
      ? `پیغام ${MAX_INPUT_CHARS} حروف سے زیادہ نہیں ہو سکتا۔`
      : `Message must be under ${MAX_INPUT_CHARS} characters.`;
  if (lang === "ur") {
    const urduRange = /[\u0600-\u06FF]/;
    const numericOnly = /^[\d\s]+$/;
    if (!urduRange.test(trimmed) && !numericOnly.test(trimmed)) {
      return "براہ کرم اردو میں لکھیں یا نمبر درج کریں۔";
    }
  }
  return "";
};

// ── Audio engine ─────────────────────────────────────────────────
const sharedAudio = new Audio();
sharedAudio.preload = "none";

const stopAudio = () => {
  sharedAudio.pause();
  sharedAudio.src = "";
  sharedAudio.oncanplay = null;
  sharedAudio.onended = null;
  sharedAudio.onerror = null;
};

const playTTS = (text, rate, lang, onStart, onEnd, onError) => {
  stopAudio();
  const clean = text
    .replace(/[\u{1F300}-\u{1FFFF}]/gu, "")
    .replace(/[🔍💚📋✅🌱🟠🔵🟢]/g, "")
    .replace(/\*+/g, " ")
    .replace(/\n+/g, ". ")
    .trim();
  if (!clean) { onError(); return; }
  const params = new URLSearchParams({ text: clean, lang });
  sharedAudio.src = `${TTS_URL}?${params.toString()}`;
  sharedAudio.playbackRate = Math.min(Math.max(rate, 0.5), 2);
  sharedAudio.oncanplay = () => onStart();
  sharedAudio.onended = onEnd;
  sharedAudio.onerror = () => { console.error("TTS error"); onError(); };
  sharedAudio.play().catch(err => { console.error("TTS play() error:", err); onError(); });
};

// ════════════════════════════════════════════════════════════════
//  UI STRINGS — bilingual
// ════════════════════════════════════════════════════════════════
const UI = {
  en: {
    appName:        "SentiCare",
    tagline:        "Your personal mental health companion",
    nav_chat:       "Mental Health Chat",
    nav_sessions:   "My Sessions",
    nav_resources:  "Resources",
    nav_signout:    "Sign Out",
    voice_on:       "Voice ON",
    voice_off:      "Voice OFF",
    voice_settings: "🔊 Voice Settings",
    speed:          "Speed",
    tip_title:      "💡 Tip of the day",
    lang_toggle:    "اردو",
    topbar_title:   "Mental Health Assistant",
    topbar_sub:     "AI-powered screening & CBT-based support",
    speaking:       "Speaking...",
    loading_voice:  "Loading voice…",
    available:      "Available",
    greeting_name:  (n) => `Hello ${n}, I am here for you`,
    empty_sub:      (voiceOn) => `SentiCare guides you through a short mental health screening and provides personalised CBT-based support. ${voiceOn ? "Voice is ON — I will speak to you." : "Turn on voice in the sidebar."}`,
    quick1:         "Hi, I need support",
    quick2:         "I feel anxious",
    quick3:         "I am feeling stressed",
    senticare:      "SentiCare",
    replay:         "🔊 Replay",
    stop:           "⏹ Stop",
    view_card:      "🃏 View Card",
    placeholder:    "Share how you are feeling today…",
    input_hint:     "Press Enter to send · Shift+Enter for new line",
    sessions_title: "My Sessions",
    sessions_sub:   "Your past mental health conversations",
    sessions_head:  "Session History",
    sessions_count: (n) => `${n} saved session${n !== 1 ? "s" : ""}`,
    new_chat:       "+ New Chat",
    clear_all:      "🗑 Clear All",
    no_sessions:    "No sessions yet.",
    no_sessions_sub:"Complete a chat to see it saved here.",
    start_first:    "Start your first session →",
    session_on:     "Session —",
    messages:       "messages",
    anxiety_tag:    "🟠 Anxiety",
    stress_tag:     "🔵 Stress",
    general_tag:    "🟢 General",
    delete:         "🗑 Delete",
    resources_title:"Mental Health Resources",
    resources_sub:  "Practical tools and evidence-based strategies",
    self_help:      "Self-Help Tools",
    self_help_sub:  "Things you can try right now to feel better",
    checklist:      "✅ Daily Wellness Checklist",
    hotline_title:  "Need immediate help?",
    hotline_text:   <span>In Pakistan, the <strong>Umang helpline</strong> is available at <strong>0317-4288665</strong>. You deserve support and you are not alone.</span>,
    therapy_tag:    "🌿 Therapy Exercise",
    listen:         "🔊 Listen to This",
    stop_speaking:  "Stop Speaking",
    got_it:         "✓ Got It",
    loading_dots:   "⏳ Loading…",
    confirm_all_title: "Clear All Sessions?",
    confirm_all_text:  "This will permanently delete all your saved sessions. This cannot be undone.",
    confirm_one_title: "Delete This Session?",
    confirm_one_text:  "This session will be permanently deleted. Are you sure?",
    confirm_cancel: "Cancel",
    confirm_delete: "Delete",
    login_title:    "Welcome back 👋",
    signup_title:   "Create your account",
    name_label:     "Full Name",
    email_label:    "Email Address",
    pass_label:     "Password",
    confirm_label:  "Confirm Password",
    signin_btn:     "Sign In",
    signup_btn:     "Create Account",
    wait_btn:       "Please wait…",
    have_account:   "Already have an account?",
    no_account:     "Don't have an account?",
    sign_in_link:   "Sign In",
    sign_up_link:   "Sign Up",
    info_box:       <span>Your account and session history are <strong>always saved</strong>. Sign in to continue where you left off.</span>,
    new_here:       "NEW HERE?",
    have_acct:      "HAVE AN ACCOUNT?",
    char_count:     (n, max) => `${n} / ${max}`,
    toast_session:  "Session saved ✓",
    toast_deleted:  "Session deleted",
    toast_all_del:  "All sessions cleared",
    menu:           "Menu",
    fb_helpful:     "👍 Helpful",
    fb_not_helpful: "👎 Not helpful",
    fb_thanks_up:   "✓ Thanks! Response style saved.",
    fb_thanks_down: "✓ Got it. We'll adapt next time.",
    ppo_switching:  "⚡ Adapting response strategy for you…",
    trend_title:    "📈 Emotional Trend",
    escalate_msg:   "⚠️ Your last 3 sessions show high severity. Please consider reaching out to the Umang helpline: 0317-4288665. You are not alone.",
    trend_low:      "Low",
    trend_med:      "Med",
    trend_high:     "High",
  },
  ur: {
    appName:        "سینٹی کیئر",
    tagline:        "آپ کا ذہنی صحت کا ساتھی",
    nav_chat:       "ذہنی صحت چیٹ",
    nav_sessions:   "میرے سیشن",
    nav_resources:  "وسائل",
    nav_signout:    "سائن آؤٹ",
    voice_on:       "آواز چالو",
    voice_off:      "آواز بند",
    voice_settings: "🔊 آواز کی ترتیبات",
    speed:          "رفتار",
    tip_title:      "💡 آج کی نصیحت",
    lang_toggle:    "English",
    topbar_title:   "ذہنی صحت معاون",
    topbar_sub:     "اے آئی پر مبنی اسکریننگ اور سی بی ٹی معاونت",
    speaking:       "بول رہا ہوں...",
    loading_voice:  "آواز لوڈ ہو رہی ہے…",
    available:      "دستیاب",
    greeting_name:  (n) => `السلام علیکم ${n}، میں آپ کے ساتھ ہوں`,
    empty_sub:      (voiceOn) => `سینٹی کیئر آپ کی ذہنی صحت کی اسکریننگ میں مدد کرتا ہے۔ ${voiceOn ? "آواز چالو ہے۔" : "سائڈ بار میں آواز چالو کریں۔"}`,
    quick1:         "السلام علیکم، مجھے مدد چاہیے",
    quick2:         "مجھے گھبراہٹ محسوس ہو رہی ہے",
    quick3:         "میں دباؤ میں ہوں",
    senticare:      "سینٹی کیئر",
    replay:         "🔊 دوبارہ سنیں",
    stop:           "⏹ روکیں",
    view_card:      "🃏 کارڈ دیکھیں",
    placeholder:    "آج آپ کیسا محسوس کر رہے ہیں؟",
    input_hint:     "بھیجنے کے لیے Enter دبائیں",
    sessions_title: "میرے سیشن",
    sessions_sub:   "آپ کی گزشتہ ذہنی صحت گفتگو",
    sessions_head:  "سیشن کی تاریخ",
    sessions_count: (n) => `${n} محفوظ سیشن`,
    new_chat:       "+ نئی گفتگو",
    clear_all:      "🗑 سب حذف کریں",
    no_sessions:    "ابھی کوئی سیشن نہیں۔",
    no_sessions_sub:"گفتگو مکمل کریں تو یہاں محفوظ ہو گا۔",
    start_first:    "پہلا سیشن شروع کریں ←",
    session_on:     "سیشن —",
    messages:       "پیغامات",
    anxiety_tag:    "🟠 گھبراہٹ",
    stress_tag:     "🔵 دباؤ",
    general_tag:    "🟢 عمومی",
    delete:         "🗑 حذف",
    resources_title:"ذہنی صحت کے وسائل",
    resources_sub:  "عملی آلات اور ثبوت پر مبنی حکمت عملیاں",
    self_help:      "خود مدد کے آلات",
    self_help_sub:  "ابھی بہتر محسوس کرنے کے لیے کچھ آزمائیں",
    checklist:      "✅ روزانہ تندرستی کی فہرست",
    hotline_title:  "فوری مدد چاہیے؟",
    hotline_text:   <span>پاکستان میں <strong>امنگ ہیلپ لائن</strong>: <strong>0317-4288665</strong>۔ آپ اکیلے نہیں ہیں۔</span>,
    therapy_tag:    "🌿 علاجی مشق",
    listen:         "🔊 سنیں",
    stop_speaking:  "روکیں",
    got_it:         "✓ سمجھ گیا",
    loading_dots:   "⏳ لوڈ ہو رہا ہے…",
    confirm_all_title: "سب سیشن حذف کریں؟",
    confirm_all_text:  "یہ آپ کے تمام محفوظ سیشن مستقل طور پر حذف کر دے گا۔",
    confirm_one_title: "یہ سیشن حذف کریں؟",
    confirm_one_text:  "یہ سیشن مستقل طور پر حذف ہو جائے گا۔ کیا آپ مطمئن ہیں؟",
    confirm_cancel: "منسوخ",
    confirm_delete: "حذف کریں",
    login_title:    "خوش آمدید 👋",
    signup_title:   "اکاؤنٹ بنائیں",
    name_label:     "پورا نام",
    email_label:    "ای میل",
    pass_label:     "پاس ورڈ",
    confirm_label:  "پاس ورڈ کی تصدیق",
    signin_btn:     "سائن ان",
    signup_btn:     "اکاؤنٹ بنائیں",
    wait_btn:       "انتظار کریں…",
    have_account:   "پہلے سے اکاؤنٹ ہے؟",
    no_account:     "اکاؤنٹ نہیں ہے؟",
    sign_in_link:   "سائن ان",
    sign_up_link:   "سائن اپ",
    info_box:       <span>آپ کا اکاؤنٹ <strong>ہمیشہ محفوظ رہتا ہے۔</strong> جاری رکھنے کے لیے سائن ان کریں۔</span>,
    new_here:       "نئے ہیں؟",
    have_acct:      "اکاؤنٹ ہے؟",
    char_count:     (n, max) => `${n} / ${max}`,
    toast_session:  "سیشن محفوظ ہو گیا ✓",
    toast_deleted:  "سیشن حذف ہو گیا",
    toast_all_del:  "تمام سیشن حذف ہو گئے",
    menu:           "مینو",
    fb_helpful:     "👍 مددگار",
    fb_not_helpful: "👎 مددگار نہیں",
    fb_thanks_up:   "✓ شکریہ! جواب کا انداز محفوظ ہو گیا۔",
    fb_thanks_down: "✓ سمجھ گئے۔ اگلی بار بہتر کریں گے۔",
    ppo_switching:  "⚡ آپ کے لیے جواب کی حکمت عملی بدل رہی ہے…",
    trend_title:    "📈 جذباتی رجحان",
    escalate_msg:   "⚠️ آپ کے آخری 3 سیشن میں شدت زیادہ رہی ہے۔ براہ کرم امنگ ہیلپ لائن سے رابطہ کریں: 0317-4288665۔",
    trend_low:      "کم",
    trend_med:      "درمیانہ",
    trend_high:     "زیادہ",
  },
};

const RESOURCES_EN = [
  { icon: "🧘", title: "Breathing Exercises", desc: "4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Reduces anxiety within minutes and calms your nervous system." },
  { icon: "📓", title: "Journaling", desc: "Writing 3 things you are grateful for daily reduces stress by up to 25% according to research." },
  { icon: "🚶", title: "Physical Activity", desc: "Even a 20-minute walk releases endorphins that improve mood and significantly reduce anxiety." },
  { icon: "😴", title: "Sleep Hygiene", desc: "Keep a consistent sleep schedule. Avoid screens 1 hour before bed for deeper, more restful sleep." },
  { icon: "🌿", title: "Mindfulness", desc: "5 minutes of mindful breathing each morning sets a calm, focused tone for the entire day." },
  { icon: "👥", title: "Social Support", desc: "Talking to a trusted friend or family member is one of the most effective stress relievers available." },
];

const RESOURCES_UR = [
  { icon: "🧘", title: "سانس کی مشقیں", desc: "4-7-8 سانس: 4 گنتی میں سانس لیں، 7 گنتی روکیں، 8 گنتی میں چھوڑیں۔ گھبراہٹ فوری کم ہوتی ہے۔" },
  { icon: "📓", title: "ڈائری لکھنا", desc: "روزانہ 3 شکرگزاری کی باتیں لکھنا تناؤ 25 فیصد تک کم کر سکتا ہے۔" },
  { icon: "🚶", title: "جسمانی سرگرمی", desc: "20 منٹ کی چہل قدمی بھی موڈ بہتر کرنے والے ہارمون پیدا کرتی ہے۔" },
  { icon: "😴", title: "نیند کی عادات", desc: "سونے کا وقت مقرر رکھیں۔ سونے سے ایک گھنٹہ پہلے اسکرین سے دور رہیں۔" },
  { icon: "🌿", title: "ذہن سازی", desc: "صبح 5 منٹ آہستہ سانس لینا پورے دن کو پرسکون بنا سکتا ہے۔" },
  { icon: "👥", title: "سماجی معاونت", desc: "کسی قابل اعتماد دوست یا خاندانی فرد سے بات کرنا تناؤ کم کرنے کا بہترین طریقہ ہے۔" },
];

const DAILY_TIPS_EN = [
  "Drink at least 8 glasses of water — dehydration makes anxiety worse.",
  "Limit caffeine to 1-2 cups per day, especially if you feel anxious.",
  "Take a 5-minute break every hour when studying or working.",
  "Practice the 5-4-3-2-1 grounding technique when feeling overwhelmed.",
  "Set one small, achievable goal for today and celebrate completing it.",
  "Reach out to someone you trust if you are struggling — you are not alone.",
];

const DAILY_TIPS_UR = [
  "روزانہ کم از کم 8 گلاس پانی پیئں۔",
  "کیفین کو روزانہ 1-2 کپ تک محدود رکھیں۔",
  "پڑھائی کے دوران ہر گھنٹے میں 5 منٹ کا وقفہ لیں۔",
  "جب دباؤ ہو تو 5-4-3-2-1 کی زمینی تکنیک آزمائیں۔",
  "آج کے لیے ایک چھوٹا ہدف بنائیں۔",
  "اگر مشکل ہو تو کسی قابل اعتماد شخص سے بات کریں۔",
];

const levelToNum = (level) =>
  level === "high" ? 3 : level === "medium" ? 2 : 1;

// ════════════════════════════════════════════════════════════════
//  TOAST SYSTEM
// ════════════════════════════════════════════════════════════════
function ToastContainer({ toasts }) {
  return (
    <div className="toast-container">
      {toasts.map(t => <div key={t.id} className="toast">{t.msg}</div>)}
    </div>
  );
}

function useToast() {
  const [toasts, setToasts] = useState([]);
  const show = useCallback((msg) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, msg }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 2500);
  }, []);
  return { toasts, show };
}

// ════════════════════════════════════════════════════════════════
//  AUTH SCREEN
// ════════════════════════════════════════════════════════════════
function AuthScreen({ onLogin, lang }) {
  const t = UI[lang];
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const setField = (k, v) => { setForm(f => ({ ...f, [k]: v })); setErrors(e => ({ ...e, [k]: "" })); setGlobalError(""); };

  const validate = () => {
    const e = {};
    if (mode === "signup" && !form.name.trim()) e.name = "Name is required.";
    if (!form.email.trim()) e.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) e.email = "Enter a valid email.";
    if (!form.password) e.password = "Password is required.";
    else if (form.password.length < 6) e.password = "At least 6 characters.";
    if (mode === "signup" && form.password !== form.confirm) e.confirm = "Passwords do not match.";
    return e;
  };

  const handleSubmit = () => {
    const e = validate();
    if (Object.keys(e).length) { setErrors(e); return; }
    setLoading(true);
    setTimeout(() => {
      const users = local.get("sc_users") || {};
      const email = form.email.toLowerCase().trim();
      if (mode === "signup") {
        if (users[email]) { setErrors({ email: "This email is already registered." }); setLoading(false); return; }
        const user = { name: form.name.trim(), email, password: form.password, createdAt: Date.now() };
        users[email] = user;
        local.set("sc_users", users);
        session.set("sc_current_user", user);
        onLogin(user);
      } else {
        const user = users[email];
        if (!user) { setGlobalError("No account found with this email."); setLoading(false); return; }
        if (user.password !== form.password) { setErrors({ password: "Incorrect password." }); setLoading(false); return; }
        session.set("sc_current_user", user);
        onLogin(user);
      }
      setLoading(false);
    }, 400);
  };

  const handleKey = (e) => { if (e.key === "Enter") handleSubmit(); };
  const switchMode = (m) => { setMode(m); setErrors({}); setGlobalError(""); setForm({ name:"",email:"",password:"",confirm:"" }); };

  return (
    <div className="auth-bg">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">🌿</div>
          <div className="auth-brand">{t.appName}</div>
          <div className="auth-tagline">{t.tagline}</div>
        </div>
        <div className="auth-body">
          <div className="auth-info-box"><span className="auth-info-icon">ℹ️</span><span>{t.info_box}</span></div>
          <div className="auth-title">{mode === "login" ? t.login_title : t.signup_title}</div>
          {mode === "signup" && (
            <div className="auth-field">
              <label className="auth-label">{t.name_label}</label>
              <input className={`auth-input ${errors.name ? "error" : ""}`} placeholder="Your name"
                value={form.name} onChange={e => setField("name", e.target.value)} onKeyDown={handleKey} />
              {errors.name && <div className="auth-error">{errors.name}</div>}
            </div>
          )}
          <div className="auth-field">
            <label className="auth-label">{t.email_label}</label>
            <input className={`auth-input ${errors.email ? "error" : ""}`} type="email" placeholder="you@example.com"
              value={form.email} onChange={e => setField("email", e.target.value)} onKeyDown={handleKey} />
            {errors.email && <div className="auth-error">{errors.email}</div>}
          </div>
          <div className="auth-field">
            <label className="auth-label">{t.pass_label}</label>
            <input className={`auth-input ${errors.password ? "error" : ""}`} type="password" placeholder="••••••••"
              value={form.password} onChange={e => setField("password", e.target.value)} onKeyDown={handleKey} />
            {errors.password && <div className="auth-error">{errors.password}</div>}
          </div>
          {mode === "signup" && (
            <div className="auth-field">
              <label className="auth-label">{t.confirm_label}</label>
              <input className={`auth-input ${errors.confirm ? "error" : ""}`} type="password" placeholder="••••••••"
                value={form.confirm} onChange={e => setField("confirm", e.target.value)} onKeyDown={handleKey} />
              {errors.confirm && <div className="auth-error">{errors.confirm}</div>}
            </div>
          )}
          {globalError && <div className="auth-error" style={{ marginBottom: 12 }}>{globalError}</div>}
          <button className="auth-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? t.wait_btn : mode === "login" ? t.signin_btn : t.signup_btn}
          </button>
          <div className="auth-divider">
            <div className="auth-divider-line" />
            <div className="auth-divider-text">{mode === "login" ? t.new_here : t.have_acct}</div>
            <div className="auth-divider-line" />
          </div>
          <div className="auth-switch">
            {mode === "login"
              ? <>{t.no_account} <span className="auth-switch-link" onClick={() => switchMode("signup")}>{t.sign_up_link}</span></>
              : <>{t.have_account} <span className="auth-switch-link" onClick={() => switchMode("login")}>{t.sign_in_link}</span></>
            }
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
//  CONFIRM DIALOG
// ════════════════════════════════════════════════════════════════
function ConfirmDialog({ icon, title, text, onConfirm, onCancel, lang }) {
  const t = UI[lang];
  return (
    <div className="confirm-overlay" onClick={e => { if (e.target === e.currentTarget) onCancel(); }}>
      <div className="confirm-card">
        <div className="confirm-icon">{icon}</div>
        <div className="confirm-title">{title}</div>
        <div className="confirm-text">{text}</div>
        <div className="confirm-actions">
          <button className="confirm-cancel" onClick={onCancel}>{t.confirm_cancel}</button>
          <button className="confirm-ok" onClick={onConfirm}>{t.confirm_delete}</button>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
//  SIDEBAR CONTENT
// ════════════════════════════════════════════════════════════════
function SidebarContent({ user, page, setPage, sessions, lang, voiceOn, setVoiceOn, speechRate, setSpeechRate, onLogout, todayTip, onClose }) {
  const t = UI[lang];
  const initials = user.name.split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2);
  const nav = (p) => { setPage(p); onClose && onClose(); };

  return (
    <>
      <div className="logo-area">
        <div className="logo-icon">🌿</div>
        <div className="logo-text">SentiCare</div>
      </div>
      <div className="user-badge">
        <div className="user-avatar">{initials}</div>
        <div className="user-info">
          <div className="user-name">{user.name}</div>
          <div className="user-email">{user.email}</div>
        </div>
      </div>
      <div className="sidebar-label">Navigation</div>
      <button className={`sidebar-item ${page === "chat" ? "active" : ""}`} onClick={() => nav("chat")}>
        <span className="sidebar-item-icon">💬</span> {t.nav_chat}
      </button>
      <button className={`sidebar-item ${page === "sessions" ? "active" : ""}`} onClick={() => nav("sessions")}>
        <span className="sidebar-item-icon">🗂️</span> {t.nav_sessions}
        {sessions.length > 0 && <span className="sidebar-item-badge">{sessions.length}</span>}
      </button>
      <button className={`sidebar-item ${page === "resources" ? "active" : ""}`} onClick={() => nav("resources")}>
        <span className="sidebar-item-icon">📚</span> {t.nav_resources}
      </button>
      <button className="sidebar-logout" onClick={onLogout}>
        <span style={{ fontSize: 15 }}>🚪</span> {t.nav_signout}
      </button>
      <div className="voice-box" style={{ marginTop: 14 }}>
        <div className="voice-box-title">{t.voice_settings}</div>
        <div className="voice-row">
          <span className="voice-label">{voiceOn ? t.voice_on : t.voice_off}</span>
          <button className={`toggle ${voiceOn ? "on" : "off"}`} onClick={() => { stopAudio(); setVoiceOn(v => !v); }}>
            <div className="toggle-knob" />
          </button>
        </div>
        {voiceOn && (
          <div className="speed-row">
            <div className="speed-top">
              <span className="voice-label">{t.speed}</span>
              <span className="speed-val">{speechRate.toFixed(1)}x</span>
            </div>
            <input type="range" min="0.5" max="2" step="0.1" value={speechRate}
              onChange={e => { const r = parseFloat(e.target.value); setSpeechRate(r); sharedAudio.playbackRate = r; }} />
          </div>
        )}
      </div>
      <div className="sidebar-footer">
        <div className="sidebar-footer-title">{t.tip_title}</div>
        <div className="sidebar-footer-text">{todayTip}</div>
      </div>
    </>
  );
}

// ════════════════════════════════════════════════════════════════
//  EMOTION TREND CHART
// ════════════════════════════════════════════════════════════════
function EmotionTrendChart({ sessions, lang }) {
  const t = UI[lang];
  const chartData = [...sessions].reverse().map(s => ({
    date: s.date,
    level: levelToNum(s.level || "medium"),
    condition: s.condition,
    levelLabel: s.level || "medium",
  }));
  const shouldEscalate =
    sessions.length >= 3 &&
    sessions.slice(0, 3).every(s => (s.level || "medium") === "high");

  const CustomDot = (props) => {
    const { cx, cy, payload } = props;
    const color = payload.levelLabel === "high" ? "#dc2626" : payload.levelLabel === "medium" ? "#f5a623" : "#4ade80";
    return <circle cx={cx} cy={cy} r={6} fill={color} stroke="#fff" strokeWidth={2} />;
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    const levelColors = { high: "#dc2626", medium: "#f5a623", low: "#4ade80" };
    const levelLabels = { high: t.trend_high, medium: t.trend_med, low: t.trend_low };
    return (
      <div style={{ background: "#fff", border: "1.5px solid #c8e8d4", borderRadius: 10, padding: "8px 14px", fontSize: "0.8rem" }}>
        <div style={{ fontWeight: 700, color: "#0e4d2a", marginBottom: 3 }}>{d.date}</div>
        <div style={{ color: "#4a8a62" }}>{d.condition}</div>
        <div style={{ color: levelColors[d.levelLabel], fontWeight: 700, marginTop: 3 }}>{levelLabels[d.levelLabel]}</div>
      </div>
    );
  };

  if (sessions.length < 2) return null;

  return (
    <div className="trend-card">
      <div className="trend-card-title">{t.trend_title}</div>
      <div className="trend-legend">
        <div className="trend-legend-item"><div className="legend-dot" style={{ background: "#4ade80" }} />{t.trend_low}</div>
        <div className="trend-legend-item"><div className="legend-dot" style={{ background: "#f5a623" }} />{t.trend_med}</div>
        <div className="trend-legend-item"><div className="legend-dot" style={{ background: "#dc2626" }} />{t.trend_high}</div>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: -20, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e8f5ee" />
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#4a8a62" }} tickLine={false} />
          <YAxis domain={[0.5, 3.5]} ticks={[1, 2, 3]}
            tickFormatter={v => ({ 1: t.trend_low, 2: t.trend_med, 3: t.trend_high }[v] || "")}
            tick={{ fontSize: 11, fill: "#4a8a62" }} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={3} stroke="#dc2626" strokeDasharray="5 3" strokeOpacity={0.5} />
          <Line type="monotone" dataKey="level" stroke="#1a7a4a" strokeWidth={2.5} dot={<CustomDot />} activeDot={{ r: 8, fill: "#1a7a4a" }} />
        </LineChart>
      </ResponsiveContainer>
      {shouldEscalate && (
        <div className="escalate-banner">
          <span style={{ fontSize: "1.1rem", flexShrink: 0 }}>⚠️</span>
          <span>{t.escalate_msg}</span>
        </div>
      )}
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
//  ROOT
// ════════════════════════════════════════════════════════════════
export default function App() {
  const [currentUser, setCurrentUser] = useState(() => session.get("sc_current_user") || null);
  const [lang, setLang] = useState(local.get("sc_lang") || "en");

  useEffect(() => { return () => stopAudio(); }, []);

  const toggleLang = () => {
    const next = lang === "en" ? "ur" : "en";
    setLang(next);
    local.set("sc_lang", next);
  };

  const handleLogin = (user) => setCurrentUser(user);
  const handleLogout = () => {
    stopAudio();
    session.del("sc_current_user");
    setCurrentUser(null);
  };

  if (!currentUser) return <AuthScreen onLogin={handleLogin} lang={lang} />;
  return <MainApp user={currentUser} onLogout={handleLogout} lang={lang} toggleLang={toggleLang} />;
}

// ════════════════════════════════════════════════════════════════
//  MAIN APP
// ════════════════════════════════════════════════════════════════
function MainApp({ user, onLogout, lang, toggleLang }) {
  const t = UI[lang];
  const isRTL = lang === "ur";
  const RESOURCES = lang === "ur" ? RESOURCES_UR : RESOURCES_EN;
  const DAILY_TIPS = lang === "ur" ? DAILY_TIPS_UR : DAILY_TIPS_EN;

  const [page, setPage] = useState("chat");
  const [sessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [inputError, setInputError] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [voiceOn, setVoiceOn] = useState(true);
  const [speechRate, setSpeechRate] = useState(1.0);
  const [speakingIdx, setSpeakingIdx] = useState(null);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [popup, setPopup] = useState(null);
  const [popupSpeaking, setPopupSpeaking] = useState(false);
  const [popupTtsLoading, setPopupTtsLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [confirm, setConfirm] = useState(null);
  const [currentOptions, setCurrentOptions] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { toasts, show: showToast } = useToast();

  const [feedback, setFeedback] = useState({});
  const [thumbsDownCount, setThumbsDownCount] = useState(0);
  const [policyMode, setPolicyMode] = useState("default");
  const [voiceCheckInDone, setVoiceCheckInDone] = useState(false);
  const [voiceFusion, setVoiceFusion] = useState(null);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const sessionSaved = useRef(false);

  const sessionsKey = `sc_sessions_${user.email}`;
  const charCount = input.length;
  const charOver = charCount > MAX_INPUT_CHARS;
  const charWarn = charCount > MAX_INPUT_CHARS * 0.85;

  useEffect(() => { setSessions(local.get(sessionsKey) || []); return () => stopAudio(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isTyping]);
  useEffect(() => { if (inputError) setInputError(""); }, [input]);

  const handleVoiceCheckInComplete = useCallback((result) => {
    if (result && result.voice_fusion_for_ml) setVoiceFusion(result.voice_fusion_for_ml);
    setVoiceCheckInDone(true);
  }, []);

  const handleFeedback = useCallback((msgIdx, type) => {
    if (feedback[msgIdx]) return;
    setFeedback(prev => ({ ...prev, [msgIdx]: type }));
    const rewardsKey = `sc_rewards_${user.email}`;
    const existing = local.get(rewardsKey) || [];
    existing.push({ sessionId, msgIdx, type, reward: type === "up" ? 1 : -1, timestamp: Date.now() });
    local.set(rewardsKey, existing);
    if (type === "down") {
      const newCount = thumbsDownCount + 1;
      setThumbsDownCount(newCount);
      if (newCount >= 3) { setPolicyMode("alternate"); showToast(t.ppo_switching); }
    }
  }, [feedback, thumbsDownCount, sessionId, user.email, t]);

  // Auto-save session
  useEffect(() => {
    if (sessionSaved.current) return;
    const lastBot = [...messages].reverse().find(m => m.sender === "bot");
    if (!lastBot) return;
    const txt = lastBot.text.toLowerCase();
    const done = ["steps:", "اقدامات:", "you are not alone", "آپ اکیلے نہیں"].some(k => txt.includes(k));
    if (!done || messages.length < 4) return;
    const alreadySaved = local.get(sessionsKey) || [];
    if (alreadySaved.find(s => s.id === sessionId)) return;
    sessionSaved.current = true;
    const condition = txt.includes("anxiet") || txt.includes("گھبراہٹ") ? "anxiety"
      : txt.includes("stress") || txt.includes("دباؤ") ? "stress" : "general";
    let level = "medium";
    if (txt.includes("high") || txt.includes("زیادہ")) level = "high";
    else if (txt.includes("low") || txt.includes("کم")) level = "low";
    const newS = {
      id: sessionId,
      date: new Date().toLocaleDateString("en-PK", { day: "numeric", month: "short", year: "numeric" }),
      time: new Date().toLocaleTimeString("en-PK", { hour: "2-digit", minute: "2-digit" }),
      preview: lastBot.text.slice(0, 120) + "…",
      condition, level, messageCount: messages.length,
    };
    const updated = [newS, ...alreadySaved];
    local.set(sessionsKey, updated);
    setSessions(updated);
    showToast(t.toast_session);
  }, [messages]);

  const speakMessage = useCallback((text, idx) => {
    setTtsLoading(true);
    setSpeakingIdx(null);
    playTTS(text, speechRate, lang,
      () => { setTtsLoading(false); setSpeakingIdx(idx); },
      () => { setSpeakingIdx(null); setTtsLoading(false); },
      () => { setSpeakingIdx(null); setTtsLoading(false); },
    );
  }, [speechRate, lang]);

  const handleReplay = (text, idx) => {
    if (speakingIdx === idx) { stopAudio(); setSpeakingIdx(null); setTtsLoading(false); }
    else speakMessage(text, idx);
  };

  const handlePopupSpeak = (text) => {
    if (popupSpeaking) { stopAudio(); setPopupSpeaking(false); setPopupTtsLoading(false); }
    else {
      setPopupTtsLoading(true);
      playTTS(text, speechRate, lang,
        () => { setPopupTtsLoading(false); setPopupSpeaking(true); },
        () => { setPopupSpeaking(false); setPopupTtsLoading(false); },
        () => { setPopupSpeaking(false); setPopupTtsLoading(false); },
      );
    }
  };

  const openPopup = (text) => setPopup({ text, card: parseTherapyCard(text) });
  const closePopup = () => { stopAudio(); setPopupSpeaking(false); setPopupTtsLoading(false); setPopup(null); };

  // ── Core send ─────────────────────────────────────────────────
  const sendMessage = async (text) => {
    setIsTyping(true);
    setCurrentOptions(null);
    stopAudio(); setSpeakingIdx(null); setTtsLoading(false);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id:   sessionId,
          input:        text,
          lang,
          policy_mode:  policyMode,
          voice_fusion: voiceFusion ?? null,
        }),
      });
      const data = await res.json();
      const firstOptions = data.options ?? null;

      // ── THANKYOU stage ────────────────────────────────────────
      if (data.stage === "thankyou") {
        // Step 1: show thankyou message immediately
        const tyText = data.message;
        let tyIdx;
        setMessages(prev => {
          tyIdx = prev.length;
          return [...prev, { sender: "bot", text: tyText, options: null }];
        });

        // Step 2: fetch q1 in background while thankyou speaks
        setIsTyping(true);
        let q1Data = null;
        try {
          const res2 = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id:   sessionId,
              input:        "",
              lang,
              policy_mode:  policyMode,
              voice_fusion: voiceFusion ?? null,
            }),
          });
          q1Data = await res2.json();
        } catch { /* silent */ }

        // Step 3: show q1 message in UI
        let q1Idx;
        if (q1Data) {
          setMessages(prev => {
            q1Idx = prev.length;
            return [...prev, { sender: "bot", text: q1Data.message, options: q1Data.options ?? null }];
          });
          setCurrentOptions(q1Data.options ?? null);
        }
        setIsTyping(false);

        // Step 4: chain TTS — speak thankyou, then on end speak q1
        if (voiceOn) {
          // Use playTTS directly so we can chain via onEnd callback
          const cleanTy = tyText
            .replace(/[\u{1F300}-\u{1FFFF}]/gu, "")
            .replace(/[🔍💚📋✅🌱🟠🔵🟢]/g, "")
            .replace(/\*+/g, " ")
            .replace(/\n+/g, ". ")
            .trim();

          if (cleanTy) {
            const params = new URLSearchParams({ text: cleanTy, lang });
            stopAudio();
            sharedAudio.src = `${TTS_URL}?${params.toString()}`;
            sharedAudio.playbackRate = Math.min(Math.max(speechRate, 0.5), 2);
            sharedAudio.oncanplay = () => {
              setTtsLoading(false);
              setSpeakingIdx(tyIdx);
            };
            setTtsLoading(true);
            sharedAudio.onended = () => {
              // thankyou finished → now speak q1
              setSpeakingIdx(null);
              sharedAudio.onended = null;
              sharedAudio.onerror = null;
              if (q1Data && q1Idx !== undefined) {
                setTimeout(() => speakMessage(q1Data.message, q1Idx), 100);
              }
            };
            sharedAudio.onerror = () => {
              setSpeakingIdx(null);
              setTtsLoading(false);
              if (q1Data && q1Idx !== undefined) {
                setTimeout(() => speakMessage(q1Data.message, q1Idx), 100);
              }
            };
            sharedAudio.play().catch(() => {
              setSpeakingIdx(null);
              setTtsLoading(false);
              if (q1Data && q1Idx !== undefined) {
                setTimeout(() => speakMessage(q1Data.message, q1Idx), 100);
              }
            });
          } else if (q1Data && q1Idx !== undefined) {
            // tyText empty, just speak q1 directly
            setTimeout(() => speakMessage(q1Data.message, q1Idx), 50);
          }
        }
        return;
      }

      // ── PRE_SCREENING stage (greeting → first screening Q) ────
      if (data.stage === "pre_screening") {
        let greetingIdx;
        setMessages(prev => {
          greetingIdx = prev.length;
          return [...prev, { sender: "bot", text: data.message, options: null }];
        });

        const greetingText = data.message;

        let questionSpeakFn = null;
        const fetchFirstQ = async () => {
          try {
            const res2 = await fetch(API_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                session_id:   sessionId,
                input:        "",
                lang,
                policy_mode:  policyMode,
                voice_fusion: voiceFusion ?? null,
              }),
            });
            const data2 = await res2.json();
            const q1opts = data2.options ?? null;
            const q1Text = data2.message;
            let q1Idx;
            setMessages(prev => {
              q1Idx = prev.length;
              // ✅ FIX v10: store speak fn, called after greeting finishes
              questionSpeakFn = () => {
                if (voiceOn) {
                  setTimeout(() => speakMessage(q1Text, q1Idx), 50);
                }
              };
              return [...prev, { sender: "bot", text: q1Text, options: q1opts }];
            });
            setCurrentOptions(q1opts);
          } catch { /* silent */ }
        };

        if (voiceOn) {
          const questionPromise = fetchFirstQ();
          setTtsLoading(true);
          const clean = greetingText
            .replace(/[\u{1F300}-\u{1FFFF}]/gu, "")
            .replace(/[🔍💚📋✅🌱🟠🔵🟢]/g, "")
            .replace(/\*+/g, " ")
            .replace(/\n+/g, ". ")
            .trim();
          if (clean) {
            const params = new URLSearchParams({ text: clean, lang });
            sharedAudio.src = `${TTS_URL}?${params.toString()}`;
            sharedAudio.playbackRate = Math.min(Math.max(speechRate, 0.5), 2);
            sharedAudio.oncanplay = () => {
              setTtsLoading(false);
              // ✅ FIX v10: defer greetingIdx usage by 50ms
              setTimeout(() => setSpeakingIdx(greetingIdx), 50);
            };
            sharedAudio.onended = async () => {
              setSpeakingIdx(null); setTtsLoading(false);
              sharedAudio.onended = null; sharedAudio.onerror = null;
              await questionPromise;
              if (questionSpeakFn) questionSpeakFn();
            };
            sharedAudio.onerror = async () => {
              setSpeakingIdx(null); setTtsLoading(false);
              await questionPromise;
              if (questionSpeakFn) questionSpeakFn();
            };
            sharedAudio.play().catch(async () => {
              setSpeakingIdx(null); setTtsLoading(false);
              await questionPromise;
              if (questionSpeakFn) questionSpeakFn();
            });
          } else {
            await fetchFirstQ();
          }
        } else {
          await fetchFirstQ();
        }
        return;
      }

      // ── Normal question/answer flow ───────────────────────────
      let newMsgIdx;
      setMessages(prev => {
        newMsgIdx = prev.length;
        return [...prev, { sender: "bot", text: data.message, options: firstOptions }];
      });
      setCurrentOptions(firstOptions);

      // ✅ FIX v10: defer speak for normal flow messages too
      if (voiceOn) {
        const msgToSpeak = data.message;
        setTimeout(() => {
          speakMessage(msgToSpeak, newMsgIdx);
        }, 50);
      }

      if (isTherapyMessage(data.message)) setTimeout(() => openPopup(data.message), 400);

    } catch (err) {
      console.error("[sendMessage] error:", err);
      setMessages(prev => [...prev, { sender: "bot", text: "Unable to connect. Please make sure the backend is running on port 5000." }]);
      setCurrentOptions(null);
    } finally {
      setIsTyping(false);
    }
  };

  const handleOptionClick = (strValue) => {
    setMessages(prev => [...prev, { sender: "user", text: strValue }]);
    setCurrentOptions(null);
    sendMessage(strValue);
  };

  const handleSubmit = async () => {
    if (isTyping) return;
    const err = validateInput(input, lang);
    if (err) { setInputError(err); return; }
    const text = input.trim();
    setMessages(prev => [...prev, { sender: "user", text }]);
    setInput("");
    setInputError("");
    setCurrentOptions(null);
    await sendMessage(text);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } };

  const handleQuick = (text) => {
    setMessages(prev => [...prev, { sender: "user", text }]);
    setCurrentOptions(null);
    sendMessage(text);
  };

  const deleteSession = (id) => {
    setSessions(prev => { const u = prev.filter(s => s.id !== id); local.set(sessionsKey, u); return u; });
    setConfirm(null);
    showToast(t.toast_deleted);
  };
  const deleteAllSessions = () => {
    setSessions([]); local.set(sessionsKey, []);
    setConfirm(null);
    showToast(t.toast_all_del);
  };

  const startNewChat = () => {
    stopAudio();
    setMessages([]);
    setCurrentOptions(null);
    setFeedback({});
    setThumbsDownCount(0);
    setPolicyMode("default");
    setVoiceCheckInDone(false);
    setVoiceFusion(null);
    sessionSaved.current = false;
    setPage("chat");
  };

  const todayTip = DAILY_TIPS[new Date().getDay() % DAILY_TIPS.length];
  const isAnySpeaking = speakingIdx !== null;
  const initials = user.name.split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2);
  const firstName = user.name.split(" ")[0];

  const hasOptions = !isTyping && Array.isArray(currentOptions) && currentOptions.length > 0;

  const sidebarProps = {
    user, page, setPage, sessions, lang, voiceOn, setVoiceOn, speechRate, setSpeechRate,
    onLogout: () => { stopAudio(); onLogout(); },
    todayTip,
  };

  return (
    <div className="app">
      <div className="sidebar">
        <SidebarContent {...sidebarProps} />
      </div>
      {drawerOpen && (
        <>
          <div className="drawer-overlay" onClick={() => setDrawerOpen(false)} />
          <div className="drawer">
            <SidebarContent {...sidebarProps} onClose={() => setDrawerOpen(false)} />
          </div>
        </>
      )}

      <div className={`main ${isRTL ? "rtl" : ""}`}>

        {page === "chat" && (
          <>
            <div className="topbar">
              <button className="hamburger" onClick={() => setDrawerOpen(true)} aria-label={t.menu}><span /></button>
              <div className="topbar-left">
                <div className="topbar-title">{t.topbar_title}</div>
                <div className="topbar-sub">{t.topbar_sub}</div>
              </div>
              <div className="topbar-right">
                <button className="lang-pill" onClick={toggleLang}>🌐 {t.lang_toggle}</button>
                {isAnySpeaking ? (
                  <div className="speaking-badge">
                    <div className="wave"><div className="wave-bar"/><div className="wave-bar"/><div className="wave-bar"/><div className="wave-bar"/></div>
                    {t.speaking}
                  </div>
                ) : ttsLoading ? (
                  <div className="status-badge"><div className="status-dot" style={{ background: "#ffd700" }} /> {t.loading_voice}</div>
                ) : (
                  <div className="status-badge"><div className="status-dot" /> {t.available}</div>
                )}
              </div>
            </div>

            <div className="messages-area">
              {messages.length === 0 && !voiceCheckInDone && (
                <VoiceCheckIn
                  sessionId={sessionId}
                  lang={lang}
                  isRTL={isRTL}
                  onComplete={handleVoiceCheckInComplete}
                  onSkip={() => setVoiceCheckInDone(true)}
                />
              )}

              {messages.length === 0 && voiceCheckInDone && (
                <div className="empty-state">
                  <div className="empty-icon-wrap">🌱</div>
                  <div className="empty-title">{t.greeting_name(firstName)}</div>
                  <div className="empty-sub">{t.empty_sub(voiceOn)}</div>
                  <div className="quick-btns">
                    <button className="quick-btn" onClick={() => handleQuick(t.quick1)}>{t.quick1}</button>
                    <button className="quick-btn" onClick={() => handleQuick(t.quick2)}>{t.quick2}</button>
                    <button className="quick-btn" onClick={() => handleQuick(t.quick3)}>{t.quick3}</button>
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.sender}`}>
                  <div className={`avatar ${msg.sender}`}>{msg.sender === "bot" ? "SC" : initials}</div>
                  <div className="bubble-wrap">
                    <div className="sender-name">{msg.sender === "bot" ? t.senticare : firstName}</div>
                    <div className={`bubble ${msg.sender}`}>{msg.text}</div>
                    {msg.sender === "bot" && (
                      <div className="msg-actions">
                        {voiceOn && (
                          <button
                            className={`replay-btn ${speakingIdx === i ? "playing" : ""}`}
                            onClick={() => handleReplay(msg.text, i)}
                            disabled={ttsLoading && speakingIdx !== i}
                          >
                            {speakingIdx === i ? t.stop : ttsLoading && speakingIdx === null ? "⏳" : t.replay}
                          </button>
                        )}
                        {isTherapyMessage(msg.text) && (
                          <button className="view-card-btn" onClick={() => openPopup(msg.text)}>{t.view_card}</button>
                        )}
                        {isTherapyMessage(msg.text) && (
                          <>
                            {!feedback[i] ? (
                              <>
                                <button className="fb-btn up" onClick={() => handleFeedback(i, "up")}>{t.fb_helpful}</button>
                                <button className="fb-btn down" onClick={() => handleFeedback(i, "down")}>{t.fb_not_helpful}</button>
                              </>
                            ) : (
                              <span className="fb-thanks">{feedback[i] === "up" ? t.fb_thanks_up : t.fb_thanks_down}</span>
                            )}
                          </>
                        )}
                      </div>
                    )}
                    {msg.sender === "bot" && isTherapyMessage(msg.text) && feedback[i] && policyMode === "alternate" &&
                      i === messages.findLastIndex(m => m.sender === "bot" && isTherapyMessage(m.text)) && (
                      <div className="ppo-note switch">
                        ⚡ {lang === "ur" ? "PPO پالیسی: جواب کی حکمت عملی بدل دی گئی ہے" : "PPO policy: response strategy switched to alternate template"}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="message-row bot">
                  <div className="avatar bot">SC</div>
                  <div className="bubble-wrap">
                    <div className="sender-name">{t.senticare}</div>
                    <div className="bubble bot">
                      <div className="typing-indicator">
                        <div className="typing-dot"/><div className="typing-dot"/><div className="typing-dot"/>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Option pills — shown when backend sends multiple-choice options */}
            {hasOptions && (
              <div className={`option-btns ${isRTL ? "rtl" : ""}`} style={{ padding: "8px 0 0" }}>
                {currentOptions.map((opt, i) => {
                  const label = typeof opt === "object" ? (opt.label || opt.value || String(opt)) : String(opt);
                  const value = typeof opt === "object" ? (opt.value ?? opt.label ?? String(opt)) : String(opt);
                  return (
                    <button key={i} className="option-btn" onClick={() => handleOptionClick(value)}>
                      {label}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Regular chat input — always visible for text and numeric answers */}
            <div className="input-section">
              <div className={`input-box ${inputError ? "input-error-border" : ""}`}>
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={t.placeholder}
                  rows={1}
                  style={isRTL ? { direction: "rtl", textAlign: "right", fontFamily: "'Noto Nastaliq Urdu', sans-serif" } : {}}
                />
                <button className="send-btn" onClick={handleSubmit} disabled={isTyping || charOver}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13" stroke="white" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" fill="white" />
                  </svg>
                </button>
              </div>
              {inputError && <div className={`input-error ${isRTL ? "rtl" : ""}`}>{inputError}</div>}
              <div className="input-footer">
                <div className="input-hint">{t.input_hint}</div>
                <div className={`char-counter ${charOver ? "over" : charWarn ? "warn" : "ok"}`}>
                  {t.char_count(charCount, MAX_INPUT_CHARS)}
                </div>
              </div>
            </div>
          </>
        )}

        {page === "sessions" && (
          <>
            <div className="topbar">
              <button className="hamburger" onClick={() => setDrawerOpen(true)} aria-label={t.menu}><span /></button>
              <div className="topbar-left">
                <div className="topbar-title">{t.sessions_title}</div>
                <div className="topbar-sub">{t.sessions_sub}</div>
              </div>
              <div className="topbar-right">
                <button className="lang-pill" onClick={toggleLang}>🌐 {t.lang_toggle}</button>
              </div>
            </div>
            <div className="page-content">
              <div className="page-toolbar">
                <div>
                  <div className="page-heading">{t.sessions_head}</div>
                  <div className="page-sub">{t.sessions_count(sessions.length)}</div>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="new-session-btn" onClick={startNewChat}>{t.new_chat}</button>
                  {sessions.length > 0 && (
                    <button className="clear-all-btn" onClick={() => setConfirm({ type: "all" })}>{t.clear_all}</button>
                  )}
                </div>
              </div>
              <EmotionTrendChart sessions={sessions} lang={lang} />
              {sessions.length === 0 ? (
                <div className="empty-sessions">
                  <div className="empty-sessions-icon">🗂️</div>
                  <div className="empty-sessions-text">{t.no_sessions}<br />{t.no_sessions_sub}</div>
                  <button className="quick-btn" style={{ marginTop: 20 }} onClick={startNewChat}>{t.start_first}</button>
                </div>
              ) : (
                sessions.map(s => (
                  <div key={s.id} className="session-card">
                    <div className="session-card-top">
                      <div className="session-card-title">{t.session_on} {s.date}</div>
                      <div className="session-card-meta">{s.time} · {s.messageCount} {t.messages}</div>
                    </div>
                    <div className="session-card-preview">{s.preview}</div>
                    <div className="session-card-footer">
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <span className={`session-tag tag-${s.condition}`}>
                          {s.condition === "anxiety" ? t.anxiety_tag : s.condition === "stress" ? t.stress_tag : t.general_tag}
                        </span>
                        {s.level && (
                          <span style={{
                            padding: "3px 10px", borderRadius: 20, fontSize: "0.72rem", fontWeight: 700,
                            background: s.level === "high" ? "#fff0f0" : s.level === "medium" ? "#fff8e1" : "#e8f5ee",
                            color: s.level === "high" ? "#c53030" : s.level === "medium" ? "#7c4f00" : "#0e4d2a",
                            border: `1.5px solid ${s.level === "high" ? "#fca5a5" : s.level === "medium" ? "#f5d78a" : "#4aaa72"}`,
                          }}>
                            {s.level === "high" ? t.trend_high : s.level === "medium" ? t.trend_med : t.trend_low}
                          </span>
                        )}
                      </div>
                      <button className="session-del-btn" onClick={() => setConfirm({ type: "one", id: s.id })}>{t.delete}</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}

        {page === "resources" && (
          <>
            <div className="topbar">
              <button className="hamburger" onClick={() => setDrawerOpen(true)} aria-label={t.menu}><span /></button>
              <div className="topbar-left">
                <div className="topbar-title">{t.resources_title}</div>
                <div className="topbar-sub">{t.resources_sub}</div>
              </div>
              <div className="topbar-right">
                <button className="lang-pill" onClick={toggleLang}>🌐 {t.lang_toggle}</button>
              </div>
            </div>
            <div className="page-content">
              <div className="page-heading">{t.self_help}</div>
              <div className="page-sub">{t.self_help_sub}</div>
              <div className="resources-grid">
                {RESOURCES.map((r, i) => (
                  <div key={i} className="resource-card">
                    <div className="resource-icon">{r.icon}</div>
                    <div className="resource-title">{r.title}</div>
                    <div className="resource-desc">{r.desc}</div>
                  </div>
                ))}
              </div>
              <div className="tip-section">
                <div className="tip-section-title">{t.checklist}</div>
                {DAILY_TIPS.map((tip, i) => (
                  <div key={i} className="tip-item">
                    <div className="tip-num">{i + 1}</div>
                    <div className="tip-text">{tip}</div>
                  </div>
                ))}
              </div>
              <div className="hotline-card">
                <div className="hotline-icon">📞</div>
                <div>
                  <div className="hotline-title">{t.hotline_title}</div>
                  <div className="hotline-text">{t.hotline_text}</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {popup && (
        <div className="popup-overlay" onClick={e => { if (e.target === e.currentTarget) closePopup(); }}>
          <div className="popup-card">
            <div className="popup-header">
              <div className="popup-tag">{t.therapy_tag}</div>
              <div className="popup-title">{popup.card.title}</div>
              <button className="popup-close" onClick={closePopup}>✕</button>
            </div>
            <div className="popup-body">
              {popup.card.steps.length > 1 ? (
                <div className="popup-steps">
                  {popup.card.steps.map((step, i) => (
                    <div key={i} className="popup-step">
                      <div className="popup-step-num">{i + 1}</div>
                      <div className="popup-step-text">{step.replace(/^[-•*\d.]+\s*/, "")}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="popup-full-text">{popup.text}</div>
              )}
              <div className="popup-actions">
                <button
                  className={`popup-btn-speak ${popupSpeaking ? "speaking" : ""}`}
                  onClick={() => handlePopupSpeak(popup.text)}
                  disabled={popupTtsLoading}
                >
                  {popupTtsLoading ? t.loading_dots : popupSpeaking ? (
                    <><div className="wave" style={{ gap: "2px" }}>
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                    </div> {t.stop_speaking}</>
                  ) : t.listen}
                </button>
                <button className="popup-btn-close" onClick={closePopup}>{t.got_it}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {confirm && (
        <ConfirmDialog
          icon={confirm.type === "all" ? "🗑️" : "⚠️"}
          title={confirm.type === "all" ? t.confirm_all_title : t.confirm_one_title}
          text={confirm.type === "all" ? t.confirm_all_text : t.confirm_one_text}
          onConfirm={() => confirm.type === "all" ? deleteAllSessions() : deleteSession(confirm.id)}
          onCancel={() => setConfirm(null)}
          lang={lang}
        />
      )}

      <ToastContainer toasts={toasts} />
    </div>
  );
}