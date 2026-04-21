"use client";

import VocalisOrb from "@/components/VocalisOrb";
import Starfield from "@/components/Starfield";
import Sidebar, { ChatSession } from "@/components/Sidebar";
import { useVoice } from "@/hooks/useVoice";
import { Mic, MicOff, Settings2, ArrowLeft, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect, useCallback, useSyncExternalStore } from "react";
import VoiceLibrary from "@/components/VoiceLibrary";
import LogViewer from "@/components/LogViewer";
import Dialer from "@/components/Dialer";
import CallHistory from "@/components/CallHistory";
import ModelSettings from "@/components/ModelSettings";
import SystemMonitor from "@/components/SystemMonitor";
import SttChip from "@/components/SttChip";
import { Phone, History, Activity } from "lucide-react";

interface VoiceMeta { id: string; name: string }

// ── Session persistence ──────────────────────────────────────────────────────

const SESSIONS_KEY = "vaani_chat_sessions";

function loadSessions(): ChatSession[] {
  if (typeof window === "undefined") return [];
  try { return JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]"); }
  catch { return []; }
}

// Returns true after the component has hydrated on the client. During SSR and
// the first client render (pre-hydration) it returns false. Using
// useSyncExternalStore here is the React-recommended way to derive a
// client-only boolean without triggering setState-in-effect lint or causing
// a hydration mismatch.
const _emptySubscribe = () => () => {};
function useIsHydrated(): boolean {
  return useSyncExternalStore(_emptySubscribe, () => true, () => false);
}

function persistSessions(s: ChatSession[]) {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(s.slice(0, 80))); }
  catch { /* storage full */ }
}

function makeTitle(messages: ChatSession["messages"]) {
  const first = messages.find(m => m.role === "user");
  if (!first) return "New conversation";
  const t = first.text.trim().replace(/\s+/g, " ");
  return t.length > 52 ? t.slice(0, 50) + "…" : t;
}

// Ambient glow colours per state
const AMBIENT: Record<string, string> = {
  idle:      "radial-gradient(ellipse 80% 70% at 50% 55%, rgba(59,130,246,0.12) 0%, transparent 75%)",
  listening: "radial-gradient(ellipse 80% 70% at 50% 55%, rgba(52,211,153,0.15) 0%, transparent 75%)",
  thinking:  "radial-gradient(ellipse 80% 70% at 50% 55%, rgba(139,92,246,0.15) 0%, transparent 75%)",
  speaking:  "radial-gradient(ellipse 80% 70% at 50% 55%, rgba(251,146,60,0.15) 0%, transparent 75%)",
  error:     "radial-gradient(ellipse 80% 70% at 50% 55%, rgba(239,68,68,0.12) 0%, transparent 75%)",
};

// ── Component ────────────────────────────────────────────────────────────────

export default function VocalisDashboard() {
  const [activeVoiceId, setActiveVoiceId] = useState("ravi");
  const { state, messages, isLive, toggleLive, switchVoice, switchModel, resetChat } =
    useVoice(activeVoiceId);

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isVoiceLibraryOpen, setIsVoiceLibraryOpen]   = useState(false);
  const [isDialerOpen,       setIsDialerOpen]          = useState(false);
  const [isHistoryOpen,      setIsHistoryOpen]          = useState(false);
  const [isModelSettingsOpen,setIsModelSettingsOpen]   = useState(false);
  const [isSystemMonitorOpen,setIsSystemMonitorOpen]   = useState(false);
  const [activeModel, setActiveModel] = useState({ id: "gemma3:4b", name: "Gemma 3 (4B)" });
  const [voices, setVoices] = useState<VoiceMeta[]>([]);

  // Session state.
  // [HYDRATION] State initializers run on BOTH server (returns []/empty id) and
  // first client render (returns localStorage value / fresh UUID). That mismatch
  // is what triggered the React hydration error in the sidebar. Instead of
  // setting state in an effect (which the project's lint rule forbids), we keep
  // the natural client-only initializers and gate any *rendered* output behind
  // `isHydrated`, which is false during SSR + first client render and flips to
  // true only after hydration commits — by that point both trees match.
  const isHydrated = useIsHydrated();
  const [sessions, setSessions]           = useState<ChatSession[]>(loadSessions);
  const [viewingSessionId, setViewingSessionId] = useState<string | null>(null);
  const [liveSessionId, setLiveSessionId] = useState(() =>
    typeof window === "undefined" ? "" : crypto.randomUUID()
  );
  const liveSessionIdRef = useRef(liveSessionId);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // What we actually pass to the sidebar / message list. Server + first-paint
  // see []; post-hydration the real persisted sessions appear without a render
  // mismatch.
  const visibleSessions = isHydrated ? sessions : [];

  // Auto-save live messages
  useEffect(() => {
    if (messages.length === 0) return;
    const id  = liveSessionIdRef.current;
    const now = Date.now();
    const title = makeTitle(messages);
    setSessions(prev => {
      const idx = prev.findIndex(s => s.id === id);
      const next = idx !== -1
        ? prev.map((s, i) => i === idx ? { ...s, title, messages, updatedAt: now } : s)
        : [{ id, title, messages, createdAt: now, updatedAt: now }, ...prev];
      persistSessions(next);
      return next;
    });
  }, [messages]);

  const handleNewChat = useCallback(() => {
    const newId = crypto.randomUUID();
    liveSessionIdRef.current = newId;
    setLiveSessionId(newId);
    setViewingSessionId(null);
    resetChat();
  }, [resetChat]);

  const handleSelectSession = useCallback((id: string) => {
    setViewingSessionId(id === liveSessionIdRef.current ? null : id);
  }, []);

  const handleDeleteSession = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions(prev => { const n = prev.filter(s => s.id !== id); persistSessions(n); return n; });
    if (viewingSessionId === id) setViewingSessionId(null);
  }, [viewingSessionId]);

  const viewingSession   = viewingSessionId ? sessions.find(s => s.id === viewingSessionId) ?? null : null;
  const displayedMessages = viewingSession ? viewingSession.messages : messages;
  const sidebarActiveId  = viewingSessionId ?? (messages.length > 0 ? liveSessionId : null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [displayedMessages]);

  // Hydrate activeModel from the backend on mount so a page refresh doesn't
  // flash the hardcoded default (gemma3:4b) when the user's persisted choice
  // is actually something else (e.g. sarvam-m). /system/status tells us the
  // active model id; /models/llm is queried in parallel to get the pretty
  // label so the header shows "Sarvam M (24B)" not the raw "sarvam-m".
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch("http://127.0.0.1:8000/system/status").then(r => r.ok ? r.json() : null).catch(() => null),
      fetch("http://127.0.0.1:8000/models/llm").then(r => r.ok ? r.json() : []).catch(() => []),
    ]).then(([status, listing]) => {
      if (cancelled) return;
      const id = status?.models?.llm?.name;
      if (!id) return;
      const pretty = Array.isArray(listing) ? listing.find((m: { id: string; name: string }) => m.id === id) : null;
      setActiveModel({ id, name: pretty?.name || id });
    });
    return () => { cancelled = true; };
  }, []);

  // Fetch voice names
  useEffect(() => {
    let cancelled = false;
    fetch("http://127.0.0.1:8000/voices")
      .then(r => r.ok ? r.json() : [])
      .then(d => { if (!cancelled) setVoices(d); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [isVoiceLibraryOpen]);

  const activeVoiceName =
    voices.find(v => v.id === activeVoiceId)?.name ??
    (activeVoiceId === "ravi" ? "Ravi Sir" : activeVoiceId);

  const handleDial = async (number: string) => {
    try {
      const r = await fetch("http://127.0.0.1:8000/calls/dial", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: number, voice_id: activeVoiceId }),
      });
      if (r.ok) setIsHistoryOpen(true);
    } catch { /* silent */ }
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <main className="flex h-screen bg-black text-white overflow-hidden font-sans selection:bg-emerald-500/30">
      <Starfield />

      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(p => !p)}
        onNewChat={handleNewChat}
        onSettingsClick={() => setIsModelSettingsOpen(true)}
        sessions={visibleSessions}
        activeSessionId={sidebarActiveId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* ── Two-column main area ── */}
      <motion.div
        animate={{ paddingLeft: isSidebarCollapsed ? 80 : 300 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="flex-1 flex h-full overflow-hidden"
      >
        {/* ═══════════════════════════════════════
            LEFT — Orb Panel
        ═══════════════════════════════════════ */}
        <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden">

          {/* Ambient background glow */}
          <motion.div
            key={state}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.2 }}
            className="absolute inset-0 pointer-events-none"
            style={{ background: AMBIENT[state] ?? AMBIENT.idle }}
          />

          {/* Orb */}
          <div className="relative z-10 w-[34rem] h-[34rem] flex items-center justify-center">
            <VocalisOrb state={state} />

            {/* State badge — floats just below the orb centre */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
              <AnimatePresence mode="wait">
                <motion.div
                  key={state}
                  initial={{ opacity: 0, y: 6, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0,  scale: 1 }}
                  exit={{ opacity: 0, y: -4, scale: 0.9 }}
                  transition={{ duration: 0.2 }}
                  className={`px-4 py-1.5 rounded-full text-[9px] font-extrabold tracking-[0.45em] uppercase border backdrop-blur-xl whitespace-nowrap ${
                    state === "listening" ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
                    state === "speaking"  ? "bg-orange-500/10  border-orange-500/30  text-orange-400"  :
                    state === "thinking"  ? "bg-indigo-500/10  border-indigo-500/30  text-indigo-400"  :
                    state === "error"     ? "bg-red-500/10     border-red-500/30     text-red-400"     :
                                           "bg-white/5        border-white/10       text-slate-500"
                  }`}
                >
                  {state}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

          {/* Controls row */}
          <div className="absolute bottom-8 flex items-center gap-4 z-20">
            {/* Voice pill */}
            <button
              onClick={() => setIsVoiceLibraryOpen(true)}
              className="flex items-center gap-2 pl-3 pr-4 h-11 rounded-full bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10 hover:text-white transition-all"
              title={`Voice: ${activeVoiceName}`}
            >
              <Settings2 className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-[10px] font-bold uppercase tracking-wider max-w-[90px] truncate">{activeVoiceName}</span>
            </button>

            {/* STT chip — quick toggle between Sarvam (cloud) and local
                Whisper. Whisper loads onto GPU only when selected. */}
            <SttChip />

            {/* Model pill */}
            <button
              onClick={() => setIsModelSettingsOpen(true)}
              className="flex items-center gap-2 pl-3 pr-4 h-11 rounded-full bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10 hover:text-white transition-all"
              title={`Model: ${activeModel.name}`}
            >
              <div className="p-0.5 bg-indigo-500/20 rounded-md">
                <Sparkles className="w-3 h-3 text-indigo-400" />
              </div>
              <span className="text-[10px] font-bold uppercase tracking-wider max-w-[90px] truncate">{activeModel.name}</span>
            </button>

            {/* ── Mic button (hero) ── */}
            <button
              onClick={() => {
                if (viewingSession) setViewingSessionId(null);
                toggleLive();
              }}
              className={`relative w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-500 ${
                isLive
                  ? "bg-red-500/20 text-red-400 border-red-500/40 shadow-[0_0_50px_rgba(239,68,68,0.25)]"
                  : "bg-white text-black border-transparent shadow-2xl shadow-blue-500/10 hover:scale-105"
              }`}
            >
              {isLive ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
              {isLive && (
                <span className="absolute inset-0 rounded-full animate-ping bg-red-500/20 pointer-events-none" />
              )}
            </button>

            {/* Phone */}
            <button
              onClick={() => setIsDialerOpen(true)}
              className="w-11 h-11 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-500 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all"
            >
              <Phone className="w-4 h-4" />
            </button>

            {/* Call History */}
            <button
              onClick={() => setIsHistoryOpen(true)}
              className="w-11 h-11 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-500 hover:bg-indigo-500/10 hover:text-indigo-400 transition-all"
            >
              <History className="w-4 h-4" />
            </button>

            {/* System Monitor */}
            <button
              onClick={() => setIsSystemMonitorOpen(v => !v)}
              className={`w-11 h-11 rounded-full border flex items-center justify-center transition-all ${
                isSystemMonitorOpen
                  ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-400"
                  : "bg-white/5 border-white/10 text-slate-500 hover:bg-emerald-500/10 hover:text-emerald-400"
              }`}
              title="System Monitor"
            >
              <Activity className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* ═══════════════════════════════════════
            RIGHT — Chat Panel
        ═══════════════════════════════════════ */}
        <div className="w-[420px] flex flex-col border-l border-white/[0.04] bg-white/[0.02] backdrop-blur-2xl relative overflow-hidden">

          {/* Subtle top fade */}
          <div className="absolute top-0 inset-x-0 h-20 bg-gradient-to-b from-black/30 to-transparent pointer-events-none z-10" />

          {/* Header */}
          <div className="shrink-0 px-5 pt-5 pb-4 z-20">
            <AnimatePresence mode="wait">
              {viewingSession ? (
                <motion.div
                  key="history"
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  className="flex items-center gap-2"
                >
                  <button
                    onClick={() => setViewingSessionId(null)}
                    className="p-1.5 rounded-lg hover:bg-white/10 text-slate-500 hover:text-white transition-colors"
                  >
                    <ArrowLeft className="w-3.5 h-3.5" />
                  </button>
                  <span className="text-xs font-semibold text-slate-300 truncate flex-1">
                    {viewingSession.title}
                  </span>
                </motion.div>
              ) : (
                <motion.div
                  key="live"
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  className="flex items-center gap-2"
                >
                  {isLive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                  )}
                  <span className="text-xs font-semibold text-slate-500 truncate">
                    {isLive ? `Talking with ${activeVoiceName}` : "Conversation"}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="mt-3 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-5 pb-6 z-10 scrollbar-hide">
            <AnimatePresence initial={false}>
              {displayedMessages.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center gap-3 text-slate-700 py-20"
                >
                  <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-center">
                    <Mic className="w-5 h-5" />
                  </div>
                  <p className="text-xs font-medium text-center leading-relaxed">
                    Press mic and<br />start speaking
                  </p>
                </motion.div>
              ) : (
                <div className="flex flex-col gap-2 pt-2">
                  {displayedMessages.map((msg, i) => (
                    <motion.div
                      key={`${viewingSessionId ?? "live"}-${i}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.18, delay: Math.min(i * 0.02, 0.3) }}
                      className={`flex flex-col gap-1 ${msg.role === "user" ? "items-end" : "items-start"}`}
                    >
                      <span className={`text-[9px] font-bold uppercase tracking-[0.2em] px-1 ${
                        msg.role === "user" ? "text-emerald-500/50" : "text-slate-600"
                      }`}>
                        {msg.role === "user" ? "You" : activeVoiceName}
                      </span>
                      <div className={`max-w-[88%] px-4 py-3 rounded-2xl text-[13px] leading-relaxed ${
                        msg.role === "user"
                          ? "bg-emerald-500/12 border border-emerald-500/20 text-emerald-50 rounded-br-md"
                          : "bg-white/6 border border-white/8 text-slate-200 rounded-bl-md"
                      }`}>
                        {msg.text}
                      </div>
                    </motion.div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              )}
            </AnimatePresence>
          </div>

          {/* Bottom subtle fade */}
          <div className="absolute bottom-0 inset-x-0 h-8 bg-gradient-to-t from-black/30 to-transparent pointer-events-none" />
        </div>
      </motion.div>

      {/* ── Modals ── */}
      <Dialer isOpen={isDialerOpen} onClose={() => setIsDialerOpen(false)} onDial={handleDial} />
      <CallHistory isOpen={isHistoryOpen} onClose={() => setIsHistoryOpen(false)} activeVoiceId={activeVoiceId} />
      <VoiceLibrary
        isOpen={isVoiceLibraryOpen}
        onClose={() => setIsVoiceLibraryOpen(false)}
        activeVoiceId={activeVoiceId}
        onSelect={(id) => { setActiveVoiceId(id); switchVoice(id); setTimeout(() => setIsVoiceLibraryOpen(false), 800); }}
      />
      <ModelSettings
        isOpen={isModelSettingsOpen}
        onClose={() => setIsModelSettingsOpen(false)}
        activeModelId={activeModel.id}
        onSelect={(id, name) => { setActiveModel({ id, name }); switchModel(id); setTimeout(() => setIsModelSettingsOpen(false), 600); }}
      />
      <LogViewer />
      <SystemMonitor isOpen={isSystemMonitorOpen} onClose={() => setIsSystemMonitorOpen(false)} />
    </main>
  );
}
