import React, { useState, useEffect, useRef, useCallback } from "react";

const API_URL = "http://localhost:5000/chat";
const TTS_URL = "http://localhost:5000/tts";

// ════════════════════════════════════════════════════════════════
//  STORAGE HELPERS
//  - sessionStorage → current login (lost on browser close, kept on refresh)
//  - localStorage   → user accounts + chat sessions (permanent)
// ════════════════════════════════════════════════════════════════
const session = {
  get: (k) => { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => sessionStorage.setItem(k, JSON.stringify(v)),
  del: (k) => sessionStorage.removeItem(k),
};
const local = {
  get: (k) => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
  del: (k) => localStorage.removeItem(k),
};

// ── Therapy detection ───────────────────────────────────────────
const THERAPY_TRIGGERS = [
  "steps:", "exercise:", "technique:", "try this:", "practice:",
  "breathing", "grounding", "progressive muscle", "cognitive", "reframe",
  "cbt", "mindfulness", "relaxation", "coping", "strategy:", "tip:",
  "here is a", "here's a", "let's try", "i recommend",
];
const isTherapyMessage = (text) =>
  THERAPY_TRIGGERS.some(t => text.toLowerCase().includes(t)) && text.length > 120;

const parseTherapyCard = (text) => {
  const lines = text.split(/\n+/).filter(Boolean);
  const title = lines[0]?.length < 80 ? lines[0].replace(/[*_#]/g, "").trim() : "Therapy Exercise";
  const steps = lines.slice(1).filter(l => l.trim().length > 0);
  return { title, steps };
};

// ── Audio engine (streaming gTTS) ───────────────────────────────
//
//  KEY FIX: Instead of fetch() → blob → objectURL (which waits for the
//  FULL mp3 to download before playing), we set audio.src directly to
//  the TTS endpoint URL with text as a query param.  The browser then
//  streams the response and fires `canplay` as soon as the first chunk
//  arrives (~0.3-0.8 s), so the user hears audio almost immediately.
//
const sharedAudio = new Audio();
sharedAudio.preload = "none";

const stopAudio = () => {
  sharedAudio.pause();
  sharedAudio.src = "";
  // clear all event handlers to avoid stale callbacks firing
  sharedAudio.oncanplay = null;
  sharedAudio.onended = null;
  sharedAudio.onerror = null;
};

const playTTS = (text, rate, onStart, onEnd, onError) => {
  stopAudio();

  const clean = text
    .replace(/[\u{1F300}-\u{1FFFF}]/gu, "")
    .replace(/[🔍💚📋✅🌱🟠🔵🟢]/g, "")
    .replace(/\*+/g, " ")
    .replace(/\n+/g, ". ")
    .trim();

  if (!clean) { onError(); return; }

  // Build GET URL — browser streams it, no blob needed
  const params = new URLSearchParams({ text: clean });
  sharedAudio.src = `${TTS_URL}?${params.toString()}`;
  sharedAudio.playbackRate = Math.min(Math.max(rate, 0.5), 2);

  // `canplay` fires as soon as the browser has enough data to start
  sharedAudio.oncanplay = () => {
    onStart();                     // ← user sees "Speaking..." almost instantly
  };
  sharedAudio.onended = onEnd;
  sharedAudio.onerror = () => {
    console.error("TTS audio error");
    onError();
  };

  // play() returns a promise; errors are handled by onerror above
  sharedAudio.play().catch((err) => {
    console.error("TTS play() error:", err);
    onError();
  });
};

// ════════════════════════════════════════════════════════════════
//  STYLES
// ════════════════════════════════════════════════════════════════
const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #e8f5ee; min-height: 100vh; font-family: 'Nunito', sans-serif; }

  /* ── AUTH ── */
  .auth-bg {
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #0e4d2a 0%, #1a7a4a 50%, #2aaa6a 100%); padding: 24px;
  }
  .auth-card {
    background: #ffffff; border-radius: 28px; width: 100%; max-width: 420px;
    box-shadow: 0 32px 80px rgba(10,40,20,0.35); overflow: hidden;
    animation: authIn 0.35s cubic-bezier(0.34,1.4,0.64,1);
  }
  @keyframes authIn { from{opacity:0;transform:scale(0.9) translateY(24px)} to{opacity:1;transform:scale(1) translateY(0)} }
  .auth-header { background: linear-gradient(135deg,#1a7a4a 0%,#0e4d2a 100%); padding: 36px 36px 28px; text-align: center; }
  .auth-logo { width:60px;height:60px;background:#ffffff;border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:26px;margin:0 auto 16px;box-shadow:0 4px 16px rgba(0,0,0,0.15); }
  .auth-brand { font-family:'Playfair Display',serif;font-size:1.8rem;color:#ffffff;font-weight:600; }
  .auth-tagline { font-size:0.82rem;color:#a8d8b8;margin-top:6px;font-weight:500; }
  .auth-body { padding: 32px 36px 36px; }
  .auth-title { font-size:1.15rem;font-weight:700;color:#0e4d2a;margin-bottom:22px; }
  .auth-field { margin-bottom: 16px; }
  .auth-label { font-size:0.78rem;font-weight:700;color:#3a6a4a;margin-bottom:6px;letter-spacing:0.5px;text-transform:uppercase;display:block; }
  .auth-input {
    width:100%;padding:13px 16px;border:2px solid #c8e8d4;border-radius:12px;
    font-family:'Nunito',sans-serif;font-size:0.92rem;color:#0e4d2a;outline:none;
    transition:border-color 0.2s,box-shadow 0.2s;background:#f8fdfb;
  }
  .auth-input:focus { border-color:#1a7a4a;box-shadow:0 0 0 3px rgba(26,122,74,0.12);background:#fff; }
  .auth-input.error { border-color:#e53e3e;box-shadow:0 0 0 3px rgba(229,62,62,0.1); }
  .auth-error { font-size:0.78rem;color:#e53e3e;margin-top:5px;font-weight:600; }
  .auth-btn {
    width:100%;padding:14px;background:#1a7a4a;border:none;border-radius:14px;
    font-family:'Nunito',sans-serif;font-size:0.95rem;font-weight:700;color:#ffffff;
    cursor:pointer;transition:background 0.15s,transform 0.1s;margin-top:8px;
    box-shadow:0 4px 16px rgba(26,122,74,0.4);
  }
  .auth-btn:hover{background:#15623c;transform:translateY(-1px)}
  .auth-btn:active{transform:scale(0.98)}
  .auth-btn:disabled{background:#a8d8b8;cursor:not-allowed;box-shadow:none;transform:none}
  .auth-switch { text-align:center;margin-top:20px;font-size:0.85rem;color:#4a8a62;font-weight:500; }
  .auth-switch-link { color:#1a7a4a;font-weight:700;cursor:pointer;text-decoration:underline; }
  .auth-switch-link:hover{color:#0e4d2a}
  .auth-divider { display:flex;align-items:center;gap:12px;margin:20px 0; }
  .auth-divider-line { flex:1;height:1px;background:#c8e8d4; }
  .auth-divider-text { font-size:0.75rem;color:#7daa90;font-weight:600; }
  .auth-info-box {
    background:#f0faf4;border:1.5px solid #c8e8d4;border-radius:12px;
    padding:12px 16px;margin-bottom:18px;font-size:0.8rem;color:#2a6a3a;
    font-weight:500;line-height:1.6;display:flex;gap:10px;align-items:flex-start;
  }
  .auth-info-icon { font-size:16px;flex-shrink:0;margin-top:1px; }

  /* ── APP LAYOUT ── */
  .app { height:100vh;display:flex;background:#e8f5ee;overflow:hidden; }

  /* ── SIDEBAR ── */
  .sidebar {
    width:260px;background:#1a7a4a;border-right:3px solid #15623c;
    display:flex;flex-direction:column;padding:18px 14px;flex-shrink:0;height:100vh;overflow:hidden;
  }
  .logo-area { display:flex;align-items:center;gap:12px;margin-bottom:18px; }
  .logo-icon { width:42px;height:42px;background:#ffffff;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0; }
  .logo-text { font-family:'Playfair Display',serif;font-size:1.75rem;color:#ffffff;font-weight:500; }
  .user-badge {
    background:rgba(0,0,0,0.2);border-radius:14px;padding:6px 8px;
    margin-bottom:14px;display:flex;align-items:center;gap:10px;
    border:1px solid rgba(255,255,255,0.15);
  }
  .user-avatar { width:36px;height:36px;background:#4ade80;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#0e4d2a;flex-shrink:0; }
  .user-info { flex:1;min-width:0; }
  .user-name { font-size:0.85rem;font-weight:700;color:#ffffff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
  .user-email { font-size:0.72rem;color:#a8d8b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis; }
  .sidebar-label { font-size:0.68rem;color:#7dcda4;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px; }
  .sidebar-item {
    display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:12px;
    font-size:0.87rem;color:#c8edd8;margin-bottom:2px;cursor:pointer;
    transition:background 0.15s;font-weight:600;border:none;background:transparent;
    width:100%;text-align:left;font-family:'Nunito',sans-serif;
  }
  .sidebar-item.active{background:rgba(255,255,255,0.2);color:#ffffff}
  .sidebar-item:hover{background:rgba(255,255,255,0.12);color:#ffffff}
  .sidebar-item-icon{font-size:15px;width:20px;text-align:center}
  .sidebar-item-badge{margin-left:auto;background:#4ade80;color:#0a3d1f;border-radius:10px;padding:1px 7px;font-size:0.7rem;font-weight:700}
  .sidebar-logout {
    display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:12px;
    font-size:0.84rem;color:#fca5a5;cursor:pointer;transition:background 0.15s;
    font-weight:600;border:none;background:transparent;width:100%;
    text-align:left;font-family:'Nunito',sans-serif;margin-top:6px;
  }
  .sidebar-logout:hover{background:rgba(220,38,38,0.2);color:#fecaca}
  .voice-box{margin-top:10px;padding:10px;background:rgba(0,0,0,0.18);border-radius:12px;border:1px solid rgba(255,255,255,0.15)}
  .voice-box-title{font-size:0.76rem;font-weight:700;color:#ffffff;margin-bottom:10px}
  .voice-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
  .voice-label{font-size:0.76rem;color:#a8d8b8;font-weight:500}
  .toggle{width:44px;height:24px;border-radius:12px;border:none;cursor:pointer;position:relative;transition:background 0.2s;flex-shrink:0;padding:0}
  .toggle.on{background:#4ade80}
  .toggle.off{background:rgba(255,255,255,0.25)}
  .toggle-knob{width:18px;height:18px;background:white;border-radius:50%;position:absolute;top:3px;transition:left 0.2s}
  .toggle.on .toggle-knob{left:23px}
  .toggle.off .toggle-knob{left:3px}
  .speed-row{display:flex;flex-direction:column;gap:5px}
  .speed-top{display:flex;justify-content:space-between}
  .speed-val{font-size:0.74rem;color:#4ade80;font-weight:700}
  input[type=range]{width:100%;accent-color:#4ade80;cursor:pointer}
  .sidebar-footer{margin-top:auto;padding:14px;background:rgba(0,0,0,0.15);border-radius:12px;border:1px solid rgba(255,255,255,0.15)}
  .sidebar-footer-title{font-size:0.76rem;font-weight:700;color:#ffffff;margin-bottom:5px}
  .sidebar-footer-text{font-size:0.73rem;color:#a8d8b8;line-height:1.6}

  /* ── MAIN ── */
  .main{flex:1;display:flex;flex-direction:column;max-width:740px;margin:0 auto;width:100%;padding:0 28px;height:100vh;overflow:hidden}
  .topbar{padding:20px 0 16px;border-bottom:2px solid #b8ddc8;flex-shrink:0;display:flex;align-items:center;justify-content:space-between}
  .topbar-title{font-size:1.05rem;font-weight:700;color:#0e4d2a}
  .topbar-sub{font-size:0.8rem;color:#4a8a62;margin-top:2px;font-weight:500}
  .status-badge{display:flex;align-items:center;gap:7px;background:#1a7a4a;border:2px solid #15623c;padding:7px 14px;border-radius:20px;font-size:0.78rem;color:#ffffff;font-weight:600}
  .status-dot{width:8px;height:8px;background:#7dff9e;border-radius:50%;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .speaking-badge{display:flex;align-items:center;gap:8px;background:#0a3d1f;border:2px solid #4ade80;padding:7px 14px;border-radius:20px;font-size:0.78rem;color:#4ade80;font-weight:700}
  .wave{display:flex;gap:3px;align-items:center}
  .wave-bar{width:3px;background:#4ade80;border-radius:2px;animation:waveBounce 0.7s infinite ease-in-out}
  .wave-bar:nth-child(1){height:7px;animation-delay:0s}
  .wave-bar:nth-child(2){height:13px;animation-delay:0.1s}
  .wave-bar:nth-child(3){height:9px;animation-delay:0.2s}
  .wave-bar:nth-child(4){height:15px;animation-delay:0.3s}
  @keyframes waveBounce{0%,100%{transform:scaleY(0.5)}50%{transform:scaleY(1.3)}}

  /* ── MESSAGES ── */
  .messages-area{flex:1;overflow-y:auto;padding:20px 0;display:flex;flex-direction:column;gap:16px;scrollbar-width:thin;scrollbar-color:#a8d8b8 #e8f5ee}
  .messages-area::-webkit-scrollbar{width:5px}
  .messages-area::-webkit-scrollbar-thumb{background:#a8d8b8;border-radius:3px}
  .empty-state{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:40px 0}
  .empty-icon-wrap{width:72px;height:72px;background:#1a7a4a;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;box-shadow:0 4px 16px rgba(26,122,74,0.3)}
  .empty-title{font-size:1.1rem;font-weight:700;color:#0e4d2a}
  .empty-sub{font-size:0.85rem;color:#4a8a62;text-align:center;max-width:280px;line-height:1.7;font-weight:500}
  .quick-btns{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:6px}
  .quick-btn{padding:10px 18px;background:#1a7a4a;border:none;border-radius:22px;font-size:0.83rem;color:#ffffff;cursor:pointer;font-family:'Nunito',sans-serif;font-weight:600;transition:background 0.15s,transform 0.1s;box-shadow:0 2px 8px rgba(26,122,74,0.25)}
  .quick-btn:hover{background:#15623c;transform:translateY(-1px)}
  .message-row{display:flex;align-items:flex-end;gap:12px;animation:fadeUp 0.25s ease forwards}
  @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .message-row.user{flex-direction:row-reverse}
  .avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;font-weight:700}
  .avatar.bot{background:#1a7a4a;color:white;box-shadow:0 2px 8px rgba(26,122,74,0.3)}
  .avatar.user{background:#ffffff;border:2px solid #1a7a4a;color:#1a7a4a}
  .bubble-wrap{display:flex;flex-direction:column;max-width:70%}
  .message-row.user .bubble-wrap{align-items:flex-end}
  .sender-name{font-size:0.72rem;font-weight:700;color:#4a8a62;margin-bottom:5px;letter-spacing:0.5px;text-transform:uppercase}
  .bubble{padding:13px 17px;border-radius:18px;font-size:0.92rem;line-height:1.7;font-weight:500}
  .bubble.bot{background:#ffffff;color:#1a3a28;border:2px solid #c8e8d4;border-bottom-left-radius:4px;box-shadow:0 2px 10px rgba(26,122,74,0.08)}
  .bubble.user{background:#1a7a4a;color:#ffffff;border-bottom-right-radius:4px;box-shadow:0 2px 10px rgba(26,122,74,0.3)}
  .msg-actions{display:flex;gap:8px;margin-top:6px;flex-wrap:wrap}
  .replay-btn{background:#e8f5ee;border:1.5px solid #4aaa72;border-radius:20px;padding:4px 12px;font-size:0.72rem;color:#1a7a4a;cursor:pointer;font-family:'Nunito',sans-serif;font-weight:700;transition:background 0.15s,transform 0.1s;display:flex;align-items:center;gap:5px}
  .replay-btn:hover{background:#c8e8d4;transform:scale(1.03)}
  .replay-btn.playing{background:#1a7a4a;color:#ffffff;border-color:#1a7a4a}
  .replay-btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
  .view-card-btn{background:#fff8e1;border:1.5px solid #f5a623;border-radius:20px;padding:4px 12px;font-size:0.72rem;color:#bf6000;cursor:pointer;font-family:'Nunito',sans-serif;font-weight:700;transition:background 0.15s,transform 0.1s;display:flex;align-items:center;gap:5px}
  .view-card-btn:hover{background:#ffe8a0;transform:scale(1.03)}
  .typing-indicator{display:flex;gap:5px;align-items:center;padding:4px 0}
  .typing-dot{width:8px;height:8px;background:#1a7a4a;border-radius:50%;animation:typingBounce 1.2s infinite}
  .typing-dot:nth-child(2){animation-delay:0.18s}
  .typing-dot:nth-child(3){animation-delay:0.36s}
  @keyframes typingBounce{0%,60%,100%{transform:translateY(0);opacity:0.4}30%{transform:translateY(-6px);opacity:1}}

  /* ── INPUT ── */
  .input-section{flex-shrink:0;padding:14px 0 20px;border-top:2px solid #b8ddc8}
  .input-box{background:#ffffff;border:2.5px solid #4aaa72;border-radius:18px;display:flex;align-items:flex-end;gap:10px;padding:11px 11px 11px 18px;transition:border-color 0.2s,box-shadow 0.2s;box-shadow:0 2px 12px rgba(26,122,74,0.12)}
  .input-box:focus-within{border-color:#1a7a4a;box-shadow:0 0 0 4px rgba(26,122,74,0.15)}
  .chat-input{flex:1;background:transparent;border:none;outline:none;font-family:'Nunito',sans-serif;font-size:0.95rem;color:#0e4d2a;resize:none;padding:4px 0;min-height:36px;max-height:120px;line-height:1.5;font-weight:500}
  .chat-input::placeholder{color:#7daa90;font-weight:400}
  .send-btn{width:44px;height:44px;border-radius:14px;background:#1a7a4a;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background 0.15s,transform 0.1s;box-shadow:0 2px 8px rgba(26,122,74,0.4)}
  .send-btn:hover{background:#15623c;transform:scale(1.05)}
  .send-btn:active{transform:scale(0.95)}
  .send-btn:disabled{background:#a8d8b8;cursor:not-allowed;box-shadow:none}
  .input-hint{font-size:0.72rem;color:#7daa90;margin-top:7px;text-align:center;font-weight:500}

  /* ── SESSIONS PAGE ── */
  .page-content{flex:1;overflow-y:auto;padding:20px 0;scrollbar-width:thin;scrollbar-color:#a8d8b8 #e8f5ee}
  .page-content::-webkit-scrollbar{width:5px}
  .page-content::-webkit-scrollbar-thumb{background:#a8d8b8;border-radius:3px}
  .page-heading{font-family:'Playfair Display',serif;font-size:1.45rem;color:#0e4d2a;margin-bottom:5px}
  .page-sub{font-size:0.84rem;color:#4a8a62;margin-bottom:22px;font-weight:500}
  .page-toolbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px}
  .new-session-btn{background:#1a7a4a;color:#ffffff;border:none;border-radius:12px;padding:9px 18px;font-size:0.84rem;font-family:'Nunito',sans-serif;font-weight:700;cursor:pointer;transition:background 0.15s;box-shadow:0 2px 8px rgba(26,122,74,0.3)}
  .new-session-btn:hover{background:#15623c}
  .clear-all-btn{background:#fff0f0;color:#c53030;border:1.5px solid #fca5a5;border-radius:12px;padding:9px 16px;font-size:0.82rem;font-family:'Nunito',sans-serif;font-weight:700;cursor:pointer;transition:background 0.15s}
  .clear-all-btn:hover{background:#ffe0e0}
  .session-card{background:#ffffff;border:2px solid #c8e8d4;border-radius:16px;padding:18px 20px;margin-bottom:12px;transition:box-shadow 0.15s,border-color 0.15s,transform 0.1s;box-shadow:0 2px 8px rgba(26,122,74,0.07)}
  .session-card:hover{box-shadow:0 4px 16px rgba(26,122,74,0.15);border-color:#1a7a4a;transform:translateY(-1px)}
  .session-card-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px}
  .session-card-title{font-size:0.92rem;font-weight:700;color:#0e4d2a}
  .session-card-meta{font-size:0.74rem;color:#7daa90;font-weight:600;white-space:nowrap;margin-left:12px}
  .session-card-preview{font-size:0.83rem;color:#3a6a4a;line-height:1.6;font-weight:500}
  .session-card-footer{display:flex;align-items:center;justify-content:space-between;margin-top:10px}
  .session-tag{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:700}
  .tag-anxiety{background:#fff3e0;color:#bf4800;border:1.5px solid #ffaa60}
  .tag-stress{background:#e3f2fd;color:#0d47a1;border:1.5px solid #90caf9}
  .tag-general{background:#e8f5ee;color:#0e4d2a;border:1.5px solid #4aaa72}
  .session-del-btn{background:transparent;border:1.5px solid #fca5a5;border-radius:10px;padding:4px 10px;font-size:0.72rem;color:#c53030;cursor:pointer;font-family:'Nunito',sans-serif;font-weight:700;transition:background 0.15s}
  .session-del-btn:hover{background:#fff0f0}
  .empty-sessions{text-align:center;padding:60px 0;color:#4a8a62}
  .empty-sessions-icon{font-size:48px;margin-bottom:16px}
  .empty-sessions-text{font-size:0.9rem;line-height:1.7;font-weight:500}

  /* ── RESOURCES ── */
  .resources-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
  .resource-card{background:#ffffff;border:2px solid #c8e8d4;border-radius:16px;padding:20px;transition:box-shadow 0.15s,transform 0.1s;box-shadow:0 2px 8px rgba(26,122,74,0.07)}
  .resource-card:hover{box-shadow:0 4px 16px rgba(26,122,74,0.15);transform:translateY(-2px)}
  .resource-icon{font-size:26px;margin-bottom:10px}
  .resource-title{font-size:0.92rem;font-weight:700;color:#0e4d2a;margin-bottom:7px}
  .resource-desc{font-size:0.81rem;color:#3a6a4a;line-height:1.7;font-weight:500}
  .tip-section{background:#ffffff;border:2px solid #c8e8d4;border-radius:16px;padding:20px;margin-bottom:14px;box-shadow:0 2px 8px rgba(26,122,74,0.07)}
  .tip-section-title{font-size:0.92rem;font-weight:700;color:#0e4d2a;margin-bottom:14px}
  .tip-item{display:flex;gap:13px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #e8f5ee}
  .tip-item:last-child{border-bottom:none;padding-bottom:0}
  .tip-num{width:26px;height:26px;background:#1a7a4a;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;color:#ffffff;flex-shrink:0}
  .tip-text{font-size:0.84rem;color:#1a3a28;line-height:1.7;font-weight:500;padding-top:3px}
  .hotline-card{background:#fff8e1;border:2px solid #ffcc02;border-radius:16px;padding:18px 20px;display:flex;gap:14px;align-items:flex-start;box-shadow:0 2px 8px rgba(255,160,0,0.1)}
  .hotline-icon{font-size:24px;flex-shrink:0;margin-top:2px}
  .hotline-title{font-size:0.92rem;font-weight:700;color:#bf4800;margin-bottom:5px}
  .hotline-text{font-size:0.82rem;color:#5a3a00;line-height:1.7;font-weight:500}

  /* ── THERAPY POPUP ── */
  .popup-overlay{position:fixed;inset:0;background:rgba(10,40,20,0.55);backdrop-filter:blur(4px);z-index:1000;display:flex;align-items:center;justify-content:center;padding:24px;animation:overlayIn 0.2s ease}
  @keyframes overlayIn{from{opacity:0}to{opacity:1}}
  .popup-card{background:#ffffff;border-radius:24px;width:100%;max-width:520px;max-height:85vh;overflow-y:auto;box-shadow:0 24px 60px rgba(10,40,20,0.3);border:2px solid #c8e8d4;animation:cardIn 0.28s cubic-bezier(0.34,1.56,0.64,1);scrollbar-width:thin;scrollbar-color:#a8d8b8 #f0faf4}
  @keyframes cardIn{from{opacity:0;transform:scale(0.88) translateY(20px)}to{opacity:1;transform:scale(1) translateY(0)}}
  .popup-header{padding:26px 26px 18px;background:linear-gradient(135deg,#1a7a4a 0%,#0e4d2a 100%);border-radius:22px 22px 0 0;position:relative}
  .popup-tag{font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7dffad;margin-bottom:7px}
  .popup-title{font-family:'Playfair Display',serif;font-size:1.25rem;color:#ffffff;line-height:1.4;padding-right:38px}
  .popup-close{position:absolute;top:16px;right:16px;width:34px;height:34px;background:rgba(255,255,255,0.15);border:none;border-radius:50%;color:#ffffff;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.15s;font-family:'Nunito',sans-serif;font-weight:700}
  .popup-close:hover{background:rgba(255,255,255,0.3)}
  .popup-body{padding:22px 26px 26px}
  .popup-steps{display:flex;flex-direction:column;gap:11px;margin-bottom:22px}
  .popup-step{display:flex;gap:13px;align-items:flex-start;padding:13px 15px;background:#f0faf4;border-radius:13px;border:1.5px solid #d4eed8;transition:border-color 0.15s,background 0.15s}
  .popup-step:hover{border-color:#1a7a4a;background:#e8f5ee}
  .popup-step-num{width:26px;height:26px;background:#1a7a4a;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;color:#ffffff;flex-shrink:0}
  .popup-step-text{font-size:0.87rem;color:#1a3a28;line-height:1.7;font-weight:500;padding-top:2px}
  .popup-full-text{font-size:0.87rem;color:#1a3a28;line-height:1.8;font-weight:500;margin-bottom:22px;white-space:pre-wrap}
  .popup-actions{display:flex;gap:11px;flex-wrap:wrap}
  .popup-btn-speak{flex:1;min-width:140px;padding:12px 18px;background:#1a7a4a;border:none;border-radius:13px;font-size:0.87rem;font-family:'Nunito',sans-serif;font-weight:700;color:#ffffff;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:background 0.15s,transform 0.1s;box-shadow:0 3px 12px rgba(26,122,74,0.35)}
  .popup-btn-speak:hover{background:#15623c;transform:translateY(-1px)}
  .popup-btn-speak.speaking{background:#0a3d1f;border:2px solid #4ade80;color:#4ade80}
  .popup-btn-speak:disabled{background:#a8d8b8;cursor:not-allowed;box-shadow:none;transform:none}
  .popup-btn-close{flex:1;min-width:110px;padding:12px 18px;background:#f0faf4;border:2px solid #c8e8d4;border-radius:13px;font-size:0.87rem;font-family:'Nunito',sans-serif;font-weight:700;color:#1a7a4a;cursor:pointer;transition:background 0.15s,transform 0.1s}
  .popup-btn-close:hover{background:#e0f2e8;transform:translateY(-1px)}

  /* ── CONFIRM DIALOG ── */
  .confirm-overlay{position:fixed;inset:0;background:rgba(10,40,20,0.5);backdrop-filter:blur(3px);z-index:2000;display:flex;align-items:center;justify-content:center;padding:24px;animation:overlayIn 0.15s ease}
  .confirm-card{background:#ffffff;border-radius:20px;width:100%;max-width:360px;padding:28px;box-shadow:0 20px 50px rgba(10,40,20,0.25);animation:cardIn 0.2s cubic-bezier(0.34,1.4,0.64,1)}
  .confirm-icon{font-size:36px;text-align:center;margin-bottom:14px}
  .confirm-title{font-size:1.05rem;font-weight:700;color:#0e4d2a;text-align:center;margin-bottom:8px}
  .confirm-text{font-size:0.85rem;color:#4a6a52;text-align:center;line-height:1.6;font-weight:500;margin-bottom:22px}
  .confirm-actions{display:flex;gap:10px}
  .confirm-cancel{flex:1;padding:11px;background:#f0faf4;border:2px solid #c8e8d4;border-radius:12px;font-size:0.88rem;font-family:'Nunito',sans-serif;font-weight:700;color:#1a7a4a;cursor:pointer;transition:background 0.15s}
  .confirm-cancel:hover{background:#e0f2e8}
  .confirm-ok{flex:1;padding:11px;background:#dc2626;border:none;border-radius:12px;font-size:0.88rem;font-family:'Nunito',sans-serif;font-weight:700;color:#ffffff;cursor:pointer;transition:background 0.15s}
  .confirm-ok:hover{background:#b91c1c}

  @media(max-width:640px){
    .sidebar{display:none}
    .main{padding:0 16px}
    .resources-grid{grid-template-columns:1fr}
    .popup-card{max-height:90vh}
  }
`;

const RESOURCES = [
  { icon: "🧘", title: "Breathing Exercises", desc: "4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Reduces anxiety within minutes and calms your nervous system." },
  { icon: "📓", title: "Journaling", desc: "Writing 3 things you are grateful for daily reduces stress by up to 25% according to research." },
  { icon: "🚶", title: "Physical Activity", desc: "Even a 20-minute walk releases endorphins that improve mood and significantly reduce anxiety." },
  { icon: "😴", title: "Sleep Hygiene", desc: "Keep a consistent sleep schedule. Avoid screens 1 hour before bed for deeper, more restful sleep." },
  { icon: "🌿", title: "Mindfulness", desc: "5 minutes of mindful breathing each morning sets a calm, focused tone for the entire day." },
  { icon: "👥", title: "Social Support", desc: "Talking to a trusted friend or family member is one of the most effective stress relievers available." },
];
const DAILY_TIPS = [
  "Drink at least 8 glasses of water — dehydration makes anxiety worse.",
  "Limit caffeine to 1-2 cups per day, especially if you feel anxious.",
  "Take a 5-minute break every hour when studying or working.",
  "Practice the 5-4-3-2-1 grounding technique when feeling overwhelmed.",
  "Set one small, achievable goal for today and celebrate completing it.",
  "Reach out to someone you trust if you are struggling — you are not alone.",
];

// ════════════════════════════════════════════════════════════════
//  AUTH SCREEN
// ════════════════════════════════════════════════════════════════
function AuthScreen({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const setField = (k, v) => {
    setForm(f => ({ ...f, [k]: v }));
    setErrors(e => ({ ...e, [k]: "" }));
    setGlobalError("");
  };

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
        session.set("sc_login", email);
        onLogin(user);
      } else {
        const user = users[email];
        if (!user) { setGlobalError("No account found with this email."); setLoading(false); return; }
        if (user.password !== form.password) { setErrors({ password: "Incorrect password." }); setLoading(false); return; }
        session.set("sc_login", email);
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
          <div className="auth-brand">SentiCare</div>
          <div className="auth-tagline">Your personal mental health companion</div>
        </div>
        <div className="auth-body">
          <div className="auth-info-box">
            <span className="auth-info-icon">ℹ️</span>
            <span>You will need to <strong>sign in each time</strong> you open a new browser session. Your account and session history are always saved.</span>
          </div>
          <div className="auth-title">{mode === "login" ? "Welcome back 👋" : "Create your account"}</div>
          {mode === "signup" && (
            <div className="auth-field">
              <label className="auth-label">Full Name</label>
              <input className={`auth-input ${errors.name ? "error" : ""}`} placeholder="Wajeeha Ijaz"
                value={form.name} onChange={e => setField("name", e.target.value)} onKeyDown={handleKey} />
              {errors.name && <div className="auth-error">{errors.name}</div>}
            </div>
          )}
          <div className="auth-field">
            <label className="auth-label">Email Address</label>
            <input className={`auth-input ${errors.email ? "error" : ""}`} type="email" placeholder="you@example.com"
              value={form.email} onChange={e => setField("email", e.target.value)} onKeyDown={handleKey} />
            {errors.email && <div className="auth-error">{errors.email}</div>}
          </div>
          <div className="auth-field">
            <label className="auth-label">Password</label>
            <input className={`auth-input ${errors.password ? "error" : ""}`} type="password" placeholder="••••••••"
              value={form.password} onChange={e => setField("password", e.target.value)} onKeyDown={handleKey} />
            {errors.password && <div className="auth-error">{errors.password}</div>}
          </div>
          {mode === "signup" && (
            <div className="auth-field">
              <label className="auth-label">Confirm Password</label>
              <input className={`auth-input ${errors.confirm ? "error" : ""}`} type="password" placeholder="••••••••"
                value={form.confirm} onChange={e => setField("confirm", e.target.value)} onKeyDown={handleKey} />
              {errors.confirm && <div className="auth-error">{errors.confirm}</div>}
            </div>
          )}
          {globalError && <div className="auth-error" style={{ marginBottom: 12 }}>{globalError}</div>}
          <button className="auth-btn" onClick={handleSubmit} disabled={loading}>
            {loading ? "Please wait…" : mode === "login" ? "Sign In" : "Create Account"}
          </button>
          <div className="auth-divider">
            <div className="auth-divider-line" />
            <div className="auth-divider-text">{mode === "login" ? "NEW HERE?" : "HAVE AN ACCOUNT?"}</div>
            <div className="auth-divider-line" />
          </div>
          <div className="auth-switch">
            {mode === "login"
              ? <>Don't have an account? <span className="auth-switch-link" onClick={() => switchMode("signup")}>Sign Up</span></>
              : <>Already have an account? <span className="auth-switch-link" onClick={() => switchMode("login")}>Sign In</span></>
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
function ConfirmDialog({ icon, title, text, onConfirm, onCancel }) {
  return (
    <div className="confirm-overlay" onClick={e => { if (e.target === e.currentTarget) onCancel(); }}>
      <div className="confirm-card">
        <div className="confirm-icon">{icon}</div>
        <div className="confirm-title">{title}</div>
        <div className="confirm-text">{text}</div>
        <div className="confirm-actions">
          <button className="confirm-cancel" onClick={onCancel}>Cancel</button>
          <button className="confirm-ok" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════
//  ROOT
// ════════════════════════════════════════════════════════════════
export default function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    const email = session.get("sc_login");
    if (email) {
      const users = local.get("sc_users") || {};
      if (users[email]) setCurrentUser(users[email]);
    }
    setAuthChecked(true);
    return () => stopAudio();
  }, []);

  const handleLogin = (user) => setCurrentUser(user);
  const handleLogout = () => {
    stopAudio();
    session.del("sc_login");
    setCurrentUser(null);
  };

  if (!authChecked) return null;
  if (!currentUser) return <><style>{styles}</style><AuthScreen onLogin={handleLogin} /></>;
  return <><style>{styles}</style><MainApp user={currentUser} onLogout={handleLogout} /></>;
}

// ════════════════════════════════════════════════════════════════
//  MAIN APP (authenticated)
// ════════════════════════════════════════════════════════════════
function MainApp({ user, onLogout }) {
  const [page, setPage] = useState("chat");
  const [sessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
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
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const sessionSaved = useRef(false);

  const sessionsKey = `sc_sessions_${user.email}`;

  useEffect(() => {
    setSessions(local.get(sessionsKey) || []);
    return () => stopAudio();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // ── Auto-save session ───────────────────────────────────────────
  useEffect(() => {
    if (sessionSaved.current) return;
    const lastBot = [...messages].reverse().find(m => m.sender === "bot");
    if (!lastBot) return;
    const t = lastBot.text.toLowerCase();
    const hasTherapy = ["steps:", "exercise:", "technique:", "breathing", "grounding", "cbt", "you are not alone"].some(k => t.includes(k));
    if (!hasTherapy || messages.length < 4) return;
    sessionSaved.current = true;
    const condition = t.includes("anxiet") ? "anxiety" : t.includes("stress") ? "stress" : "general";
    const newSession = {
      id: sessionId,
      date: new Date().toLocaleDateString("en-PK", { day: "numeric", month: "short", year: "numeric" }),
      time: new Date().toLocaleTimeString("en-PK", { hour: "2-digit", minute: "2-digit" }),
      preview: lastBot.text.slice(0, 120) + "…",
      condition,
      messageCount: messages.length,
    };
    setSessions(prev => {
      if (prev.find(s => s.id === sessionId)) return prev;
      const updated = [newSession, ...prev];
      local.set(sessionsKey, updated);
      return updated;
    });
  }, [messages]);

  // ── TTS helpers ─────────────────────────────────────────────────
  const speakMessage = useCallback((text, idx) => {
    // Show loading state immediately so the user gets feedback right away
    setTtsLoading(true);
    setSpeakingIdx(null);

    playTTS(
      text,
      speechRate,
      () => {
        // onStart — fires on canplay, audio has begun
        setTtsLoading(false);
        setSpeakingIdx(idx);
      },
      () => {
        // onEnd
        setSpeakingIdx(null);
        setTtsLoading(false);
      },
      () => {
        // onError
        setSpeakingIdx(null);
        setTtsLoading(false);
      },
    );
  }, [speechRate]);

  const handleReplay = (text, idx) => {
    if (speakingIdx === idx) {
      stopAudio();
      setSpeakingIdx(null);
      setTtsLoading(false);
    } else {
      speakMessage(text, idx);
    }
  };

  const handlePopupSpeak = (text) => {
    if (popupSpeaking) {
      stopAudio();
      setPopupSpeaking(false);
      setPopupTtsLoading(false);
    } else {
      setPopupTtsLoading(true);
      setPopupSpeaking(false);
      playTTS(
        text,
        speechRate,
        () => { setPopupTtsLoading(false); setPopupSpeaking(true); },
        () => { setPopupSpeaking(false); setPopupTtsLoading(false); },
        () => { setPopupSpeaking(false); setPopupTtsLoading(false); },
      );
    }
  };

  const openPopup = (text) => setPopup({ text, card: parseTherapyCard(text) });
  const closePopup = () => { stopAudio(); setPopupSpeaking(false); setPopupTtsLoading(false); setPopup(null); };

  // ── Send message ────────────────────────────────────────────────
  const sendMessage = async (text) => {
    setIsTyping(true);
    stopAudio();
    setSpeakingIdx(null);
    setTtsLoading(false);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, input: text }),
      });
      const data = await res.json();
      setMessages(prev => {
        const newIdx = prev.length;
        // Start TTS immediately — no more waiting for a blob
        if (voiceOn) speakMessage(data.message, newIdx);
        if (isTherapyMessage(data.message)) setTimeout(() => openPopup(data.message), 400);
        return [...prev, { sender: "bot", text: data.message }];
      });
      if (data.stage === "pre_screening") {
        const res2 = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, input: "" }),
        });
        const data2 = await res2.json();
        setMessages(prev => {
          const idx2 = prev.length;
          if (voiceOn) setTimeout(() => speakMessage(data2.message, idx2), 600);
          if (isTherapyMessage(data2.message)) setTimeout(() => openPopup(data2.message), 1000);
          return [...prev, { sender: "bot", text: data2.message }];
        });
      }
    } catch {
      setMessages(prev => [...prev, { sender: "bot", text: "Unable to connect. Please make sure the backend is running on port 5000." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSubmit = async () => {
    const text = input.trim();
    if (!text || isTyping) return;
    setMessages(prev => [...prev, { sender: "user", text }]);
    setInput("");
    await sendMessage(text);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); } };
  const handleQuick = (text) => { setMessages(prev => [...prev, { sender: "user", text }]); sendMessage(text); };

  const deleteSession = (id) => {
    setSessions(prev => { const u = prev.filter(s => s.id !== id); local.set(sessionsKey, u); return u; });
    setConfirm(null);
  };
  const deleteAllSessions = () => { setSessions([]); local.set(sessionsKey, []); setConfirm(null); };
  const startNewChat = () => { stopAudio(); setMessages([]); sessionSaved.current = false; setPage("chat"); };

  const todayTip = DAILY_TIPS[new Date().getDay() % DAILY_TIPS.length];
  const isAnySpeaking = speakingIdx !== null;
  const initials = user.name.split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2);

  return (
    <div className="app">

      {/* ── SIDEBAR ── */}
      <div className="sidebar">
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
        <button className={`sidebar-item ${page === "chat" ? "active" : ""}`} onClick={() => setPage("chat")}>
          <span className="sidebar-item-icon">💬</span> Mental Health Chat
        </button>
        <button className={`sidebar-item ${page === "sessions" ? "active" : ""}`} onClick={() => setPage("sessions")}>
          <span className="sidebar-item-icon">🗂️</span> My Sessions
          {sessions.length > 0 && <span className="sidebar-item-badge">{sessions.length}</span>}
        </button>
        <button className={`sidebar-item ${page === "resources" ? "active" : ""}`} onClick={() => setPage("resources")}>
          <span className="sidebar-item-icon">📚</span> Resources
        </button>
        <button className="sidebar-logout" onClick={onLogout}>
          <span style={{ fontSize: 15 }}>🚪</span> Sign Out
        </button>
        <div className="voice-box">
          <div className="voice-box-title">🔊 Voice Settings</div>
          <div className="voice-row">
            <span className="voice-label">{voiceOn ? "Voice ON" : "Voice OFF"}</span>
            <button className={`toggle ${voiceOn ? "on" : "off"}`} onClick={() => {
              stopAudio();
              setSpeakingIdx(null);
              setTtsLoading(false);
              setPopupSpeaking(false);
              setVoiceOn(v => !v);
            }}>
              <div className="toggle-knob" />
            </button>
          </div>
          {voiceOn && (
            <div className="speed-row">
              <div className="speed-top">
                <span className="voice-label">Speed</span>
                <span className="speed-val">{speechRate.toFixed(1)}x</span>
              </div>
              <input type="range" min="0.5" max="2" step="0.1" value={speechRate}
                onChange={e => {
                  const r = parseFloat(e.target.value);
                  setSpeechRate(r);
                  sharedAudio.playbackRate = r;
                }} />
            </div>
          )}
        </div>
        <div className="sidebar-footer">
          <div className="sidebar-footer-title">💡 Tip of the day</div>
          <div className="sidebar-footer-text">{todayTip}</div>
        </div>
      </div>

      {/* ── MAIN ── */}
      <div className="main">

        {/* CHAT PAGE */}
        {page === "chat" && (
          <>
            <div className="topbar">
              <div>
                <div className="topbar-title">Mental Health Assistant</div>
                <div className="topbar-sub">AI-powered screening & CBT-based support</div>
              </div>
              {isAnySpeaking ? (
                <div className="speaking-badge">
                  <div className="wave">
                    <div className="wave-bar"/><div className="wave-bar"/>
                    <div className="wave-bar"/><div className="wave-bar"/>
                  </div>
                  Speaking...
                </div>
              ) : ttsLoading ? (
                <div className="status-badge">
                  <div className="status-dot" style={{ background: "#ffd700" }} /> Loading voice…
                </div>
              ) : (
                <div className="status-badge"><div className="status-dot" /> Available</div>
              )}
            </div>

            <div className="messages-area">
              {messages.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon-wrap">🌱</div>
                  <div className="empty-title">Hello {user.name.split(" ")[0]}, I am here for you</div>
                  <div className="empty-sub">
                    SentiCare guides you through a short mental health screening and provides personalised CBT-based support.
                    {voiceOn ? " Voice is ON — I will speak to you." : " Turn on voice in the sidebar."}
                  </div>
                  <div className="quick-btns">
                    <button className="quick-btn" onClick={() => handleQuick("Hi, I need support")}>Hi, I need support</button>
                    <button className="quick-btn" onClick={() => handleQuick("I feel anxious")}>I feel anxious</button>
                    <button className="quick-btn" onClick={() => handleQuick("I am feeling stressed")}>I am feeling stressed</button>
                  </div>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.sender}`}>
                  <div className={`avatar ${msg.sender}`}>{msg.sender === "bot" ? "SC" : initials}</div>
                  <div className="bubble-wrap">
                    <div className="sender-name">{msg.sender === "bot" ? "SentiCare" : user.name.split(" ")[0]}</div>
                    <div className={`bubble ${msg.sender}`}>{msg.text}</div>
                    {msg.sender === "bot" && (
                      <div className="msg-actions">
                        {voiceOn && (
                          <button
                            className={`replay-btn ${speakingIdx === i ? "playing" : ""}`}
                            onClick={() => handleReplay(msg.text, i)}
                            disabled={ttsLoading && speakingIdx !== i}
                          >
                            {speakingIdx === i ? "⏹ Stop" : ttsLoading && speakingIdx === null ? "⏳" : "🔊 Replay"}
                          </button>
                        )}
                        {isTherapyMessage(msg.text) && (
                          <button className="view-card-btn" onClick={() => openPopup(msg.text)}>🃏 View Card</button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="message-row bot">
                  <div className="avatar bot">SC</div>
                  <div className="bubble-wrap">
                    <div className="sender-name">SentiCare</div>
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

            <div className="input-section">
              <div className="input-box">
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Share how you are feeling today…"
                  rows={1}
                />
                <button className="send-btn" onClick={handleSubmit} disabled={isTyping || !input.trim()}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13" stroke="white" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" fill="white" />
                  </svg>
                </button>
              </div>
              <div className="input-hint">Press Enter to send · Shift+Enter for new line</div>
            </div>
          </>
        )}

        {/* SESSIONS PAGE */}
        {page === "sessions" && (
          <>
            <div className="topbar">
              <div>
                <div className="topbar-title">My Sessions</div>
                <div className="topbar-sub">Your past mental health conversations</div>
              </div>
            </div>
            <div className="page-content">
              <div className="page-toolbar">
                <div>
                  <div className="page-heading">Session History</div>
                  <div className="page-sub">{sessions.length} saved session{sessions.length !== 1 ? "s" : ""}</div>
                </div>
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="new-session-btn" onClick={startNewChat}>+ New Chat</button>
                  {sessions.length > 0 && (
                    <button className="clear-all-btn" onClick={() => setConfirm({ type: "all" })}>🗑 Clear All</button>
                  )}
                </div>
              </div>
              {sessions.length === 0 ? (
                <div className="empty-sessions">
                  <div className="empty-sessions-icon">🗂️</div>
                  <div className="empty-sessions-text">No sessions yet.<br />Complete a chat to see it saved here.</div>
                  <button className="quick-btn" style={{ marginTop: 20 }} onClick={startNewChat}>Start your first session →</button>
                </div>
              ) : (
                sessions.map(s => (
                  <div key={s.id} className="session-card">
                    <div className="session-card-top">
                      <div className="session-card-title">Session — {s.date}</div>
                      <div className="session-card-meta">{s.time} · {s.messageCount} messages</div>
                    </div>
                    <div className="session-card-preview">{s.preview}</div>
                    <div className="session-card-footer">
                      <span className={`session-tag tag-${s.condition}`}>
                        {s.condition === "anxiety" ? "🟠 Anxiety" : s.condition === "stress" ? "🔵 Stress" : "🟢 General"}
                      </span>
                      <button className="session-del-btn" onClick={() => setConfirm({ type: "one", id: s.id })}>🗑 Delete</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}

        {/* RESOURCES PAGE */}
        {page === "resources" && (
          <>
            <div className="topbar">
              <div>
                <div className="topbar-title">Mental Health Resources</div>
                <div className="topbar-sub">Practical tools and evidence-based strategies</div>
              </div>
            </div>
            <div className="page-content">
              <div className="page-heading">Self-Help Tools</div>
              <div className="page-sub">Things you can try right now to feel better</div>
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
                <div className="tip-section-title">✅ Daily Wellness Checklist</div>
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
                  <div className="hotline-title">Need immediate help?</div>
                  <div className="hotline-text">
                    In Pakistan, the <strong>Umang helpline</strong> is available at <strong>0317-4288665</strong>.
                    You deserve support and you are not alone.
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* THERAPY POPUP */}
      {popup && (
        <div className="popup-overlay" onClick={e => { if (e.target === e.currentTarget) closePopup(); }}>
          <div className="popup-card">
            <div className="popup-header">
              <div className="popup-tag">🌿 Therapy Exercise</div>
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
                  {popupTtsLoading ? <>⏳ Loading…</> : popupSpeaking ? (
                    <><div className="wave" style={{ gap: "2px" }}>
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                      <div className="wave-bar" style={{ background: "#4ade80" }} />
                    </div> Stop Speaking</>
                  ) : <>🔊 Listen to This</>}
                </button>
                <button className="popup-btn-close" onClick={closePopup}>✓ Got It</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* CONFIRM DIALOG */}
      {confirm && (
        <ConfirmDialog
          icon={confirm.type === "all" ? "🗑️" : "⚠️"}
          title={confirm.type === "all" ? "Clear All Sessions?" : "Delete This Session?"}
          text={confirm.type === "all"
            ? "This will permanently delete all your saved sessions. This cannot be undone."
            : "This session will be permanently deleted. Are you sure?"}
          onConfirm={() => confirm.type === "all" ? deleteAllSessions() : deleteSession(confirm.id)}
          onCancel={() => setConfirm(null)}
        />
      )}
    </div>
  );
}