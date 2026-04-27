"use client";

import VocalisOrb from "@/components/VocalisOrb";
import Starfield from "@/components/Starfield";
import Sidebar, { ChatSession } from "@/components/Sidebar";
import { useVoice } from "@/hooks/useVoice";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, useMemo } from "react";
import VoiceLibrary from "@/components/VoiceLibrary";
import ModelSettings from "@/components/ModelSettings";
import TelephonyBento from "@/components/TelephonyBento";
import TelemetryBento from "@/components/TelemetryBento";
import SttChip from "@/components/SttChip";
import { Mic, MicOff, Settings2, ArrowLeft, Sparkles, Send, Layers } from "lucide-react";
import CampaignManager from "@/components/CampaignManager";

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
  const { state, messages, isLive, isMicMuted, toggleLive, toggleMic, switchVoice, switchModel, resetChat, sendText } =
    useVoice(activeVoiceId);
  const [chatInput, setChatInput] = useState("");
  // Phone transcript: real-time streaming with karaoke support
  const [phoneMessages, setPhoneMessages] = useState<{role:"user"|"ai"; text:string; partial?:boolean; spokenIndex?:number}[]>([]);
  // Track which word in the AI message is currently being spoken (karaoke)
  const ttsTimerRef = useRef<NodeJS.Timeout | null>(null);

  const handlePhoneMessage = useCallback((msg: {role:"user"|"ai"; text:string}) => {
    // Final messages: mark the last partial AI as complete, or add new user/ai
    if (msg.role === "ai") {
      setPhoneMessages(prev => {
        // Find the last partial AI message and finalize it
        const lastIdx = prev.findLastIndex(m => m.role === "ai" && m.partial);
        if (lastIdx >= 0) {
          const updated = [...prev];
          updated[lastIdx] = { role: "ai", text: msg.text, partial: false };
          return updated;
        }
        return [...prev, { role: "ai", text: msg.text }];
      });
    } else {
      setPhoneMessages(prev => [...prev, { role: "user", text: msg.text }]);
    }
  }, []);

  // Real-time partial AI text streaming (updates as LLM generates)
  const handlePhoneAiPartial = useCallback((text: string) => {
    setPhoneMessages(prev => {
      const lastIdx = prev.findLastIndex(m => m.role === "ai" && m.partial);
      if (lastIdx >= 0) {
        const updated = [...prev];
        updated[lastIdx] = { ...updated[lastIdx], text };
        return updated;
      }
      return [...prev, { role: "ai", text, partial: true, spokenIndex: -1 }];
    });
  }, []);

  // Karaoke: when TTS starts speaking a chunk, progressively highlight words
  const handlePhoneTtsSpeaking = useCallback((chunk: string) => {
    const words = chunk.split(/\s+/).filter(Boolean);
    if (words.length === 0) return;
    // Estimate ~200ms per word for Hindi speech
    const msPerWord = 200;
    let wordIdx = 0;
    // Clear any previous karaoke timer
    if (ttsTimerRef.current) clearInterval(ttsTimerRef.current);
    // Find the chunk's position in the full response text
    setPhoneMessages(prev => {
      const lastAiIdx = prev.findLastIndex(m => m.role === "ai");
      if (lastAiIdx < 0) return prev;
      const fullText = prev[lastAiIdx].text;
      const chunkStart = fullText.indexOf(chunk.trim());
      if (chunkStart < 0) return prev;
      // Count words before the chunk to get the starting word index
      const textBeforeChunk = fullText.slice(0, chunkStart);
      const wordsBeforeCount = textBeforeChunk.split(/\s+/).filter(Boolean).length;
      const updated = [...prev];
      updated[lastAiIdx] = { ...updated[lastAiIdx], spokenIndex: wordsBeforeCount };
      // Start progressive highlighting
      ttsTimerRef.current = setInterval(() => {
        wordIdx++;
        if (wordIdx >= words.length) {
          if (ttsTimerRef.current) clearInterval(ttsTimerRef.current);
          return;
        }
        setPhoneMessages(p => {
          const idx = p.findLastIndex(m => m.role === "ai");
          if (idx < 0) return p;
          const u = [...p];
          u[idx] = { ...u[idx], spokenIndex: wordsBeforeCount + wordIdx };
          return u;
        });
      }, msPerWord);
      return updated;
    });
  }, []);

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isVoiceLibraryOpen, setIsVoiceLibraryOpen]   = useState(false);
  const [isModelSettingsOpen,setIsModelSettingsOpen]   = useState(false);
  const [isCampaignManagerOpen, setIsCampaignManagerOpen] = useState(false);
  const [activeCampaign, setActiveCampaign] = useState("default");
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

  const viewingSession = useMemo(() => 
    viewingSessionId ? sessions.find(s => s.id === viewingSessionId) ?? null : null,
    [viewingSessionId, sessions]
  );
  const displayedMessages = useMemo(() => 
    viewingSession ? viewingSession.messages : [...messages, ...phoneMessages],
    [viewingSession, messages, phoneMessages]
  );
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
      fetch("http://127.0.0.1:8000/models/llm").then(r => r.ok ? r.json() : []),
      fetch("http://127.0.0.1:8000/campaign").then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([status, listing, campaign]) => {
      if (cancelled) return;
      const id = status?.models?.llm?.name;
      if (id) {
        const pretty = Array.isArray(listing) ? listing.find((m: { id: string; name: string }) => m.id === id) : null;
        setActiveModel({ id, name: pretty?.name || id });
      }
      if (campaign?.voice_id) {
        setActiveVoiceId(campaign.voice_id);
      }
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

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <main className="flex h-screen bg-black text-white overflow-hidden font-sans selection:bg-emerald-500/30">
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(p => !p)}
        onNewChat={handleNewChat}
        onSettingsClick={() => setIsModelSettingsOpen(true)}
        onCampaignClick={() => setIsCampaignManagerOpen(true)}
        sessions={visibleSessions}
        activeSessionId={sidebarActiveId}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* ── BENTO DASHBOARD ── */}
      <motion.div
        animate={{ paddingLeft: isSidebarCollapsed ? 80 : 300 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="flex-1 w-full h-full p-4 overflow-hidden"
      >
        <div className="w-full h-full flex flex-col lg:flex-row gap-4">
          
          {/* ═══════════════════════════════════════
              LEFT BENTO — AI Canvas (Orb)
          ═══════════════════════════════════════ */}
          <div className="flex-[0.4] relative rounded-3xl bg-white/[0.02] border border-white/[0.05] overflow-hidden flex flex-col backdrop-blur-2xl">
            <Starfield />

            {/* Ambient background glow */}
            <motion.div
              key={state}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1.2 }}
              className="absolute inset-0 pointer-events-none z-0"
              style={{ background: AMBIENT[state] ?? AMBIENT.idle }}
            />

            {/* Top Controls */}
            <div className="absolute top-4 inset-x-4 flex items-start justify-between z-20">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsVoiceLibraryOpen(true)}
                  className="flex items-center gap-2 px-3 h-10 rounded-xl bg-black/40 border border-white/10 text-slate-300 hover:bg-black/60 hover:text-white transition-all backdrop-blur-md shadow-lg font-mono"
                >
                  <Settings2 className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-[10px] uppercase tracking-wider">{activeVoiceName}</span>
                </button>
                <SttChip />
              </div>
              <button
                onClick={() => setIsModelSettingsOpen(true)}
                className="flex items-center gap-2 px-3 h-10 rounded-xl bg-black/40 border border-white/10 text-slate-300 hover:bg-black/60 hover:text-white transition-all backdrop-blur-md shadow-lg font-mono"
              >
                <div className="p-0.5 bg-indigo-500/20 rounded-md">
                  <Sparkles className="w-3 h-3 text-indigo-400" />
                </div>
                <span className="text-[10px] uppercase tracking-wider">{activeModel.name}</span>
              </button>
            </div>

            {/* AI Orb */}
            <div className="flex-1 flex items-center justify-center relative z-10">
              <div className="w-[85%] h-[85%] flex items-center justify-center relative">
                <VocalisOrb state={state} />
                <div className="absolute bottom-16 left-1/2 -translate-x-1/2">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={state}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -4 }}
                      transition={{ duration: 0.2 }}
                      className={`px-3 py-1 rounded-md text-[9px] font-extrabold tracking-[0.45em] uppercase border backdrop-blur-xl whitespace-nowrap shadow-xl ${
                        state === "listening" ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
                        state === "speaking"  ? "bg-orange-500/10 border-orange-500/30 text-orange-400"  :
                        state === "thinking"  ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400"  :
                        state === "error"     ? "bg-red-500/10 border-red-500/30 text-red-400"     :
                        "bg-white/5 border-white/10 text-slate-500"
                      }`}
                    >
                      {state}
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </div>

            {/* Bottom Mic Row */}
            <div className="absolute bottom-6 inset-x-0 flex justify-center items-center gap-6 z-20">
              <button
                onClick={() => {
                  if (viewingSession) setViewingSessionId(null);
                  toggleLive();
                }}
                className={`px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all backdrop-blur-md shadow-lg border ${
                  isLive
                    ? "bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/20"
                    : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20"
                }`}
              >
                {isLive ? "End Session" : "Start Live"}
              </button>

              <button
                onClick={() => {
                  if (viewingSession) setViewingSessionId(null);
                  if (isLive) toggleMic();
                }}
                disabled={!isLive}
                className={`relative w-14 h-14 rounded-full flex items-center justify-center border-2 transition-all duration-500 backdrop-blur-md shadow-[0_0_30px_rgba(0,0,0,0.5)] ${
                  !isLive
                    ? "opacity-50 cursor-not-allowed bg-white/5 border-white/10 text-white/50"
                    : isMicMuted
                    ? "bg-red-500/20 text-red-500 border-red-500/40"
                    : "bg-white text-black border-transparent shadow-[0_0_30px_rgba(59,130,246,0.15)] hover:scale-105"
                }`}
              >
                {!isLive || isMicMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                {isLive && !isMicMuted && <span className="absolute inset-0 rounded-full animate-ping bg-white/30 pointer-events-none" />}
              </button>
            </div>
            
            {/* Inline Settings overlays inside the AI Cell */}
            <div className="absolute inset-0 z-50 pointer-events-none">
              <VoiceLibrary isOpen={isVoiceLibraryOpen} onClose={() => setIsVoiceLibraryOpen(false)} activeVoiceId={activeVoiceId} onSelect={(id) => { setActiveVoiceId(id); switchVoice(id); setTimeout(() => setIsVoiceLibraryOpen(false), 800); }} />
              <ModelSettings isOpen={isModelSettingsOpen} onClose={() => setIsModelSettingsOpen(false)} activeModelId={activeModel.id} onSelect={(id, name) => { setActiveModel({ id, name }); switchModel(id); setTimeout(() => setIsModelSettingsOpen(false), 600); }} />
              <CampaignManager 
                isOpen={isCampaignManagerOpen} 
                onClose={() => setIsCampaignManagerOpen(false)} 
                activeCampaignName={activeCampaign} 
                onSelect={(name, campaign) => { 
                  setActiveCampaign(name); 
                  if (campaign.voice_id) {
                    setActiveVoiceId(campaign.voice_id);
                    switchVoice(campaign.voice_id);
                  }
                  setTimeout(() => setIsCampaignManagerOpen(false), 600); 
                }} 
              />
            </div>

          </div>

          {/* ═══════════════════════════════════════
              MIDDLE BENTO — Conversation
          ═══════════════════════════════════════ */}
          <div className="flex-[0.3] rounded-3xl bg-[#0c0c0e] border border-white/[0.05] overflow-hidden shadow-2xl flex flex-col backdrop-blur-2xl relative">
            {/* Header */}
            <div className="shrink-0 px-5 pt-4 pb-3 z-20 border-b border-white/5 bg-white/[0.02]">
              <AnimatePresence mode="wait">
                {viewingSession ? (
                  <motion.div key="history" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-2">
                    <button onClick={() => setViewingSessionId(null)} className="p-1.5 rounded-lg hover:bg-white/10 text-slate-500 hover:text-white transition-colors">
                      <ArrowLeft className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-xs font-semibold text-slate-300 truncate flex-1">{viewingSession.title}</span>
                  </motion.div>
                ) : (
                  <motion.div key="live" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-2">
                    {isLive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />}
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 truncate">{isLive ? `Talking with ${activeVoiceName}` : "Conversation"}</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-5 pb-5 pt-4 z-10 custom-scrollbar">
              <AnimatePresence initial={false}>
                {displayedMessages.length === 0 ? (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col items-center justify-center gap-3 text-slate-700">
                    <div className="w-10 h-10 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-center"><Mic className="w-4 h-4 text-zinc-600" /></div>
                    <p className="text-[9px] font-bold text-center uppercase tracking-widest text-zinc-600">Conversation Empty</p>
                  </motion.div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {displayedMessages.map((msg, i) => {
                      const isPhoneAi = msg.role === "ai" && 'spokenIndex' in msg && typeof (msg as Record<string, unknown>).spokenIndex === "number";
                      const spokenIdx = isPhoneAi ? ((msg as Record<string, unknown>).spokenIndex as number) : -1;
                      const isPartial = 'partial' in msg && (msg as Record<string, unknown>).partial;
                      const words = msg.text.split(/\s+/).filter(Boolean);
                      return (
                      <motion.div key={`${viewingSessionId ?? "live"}-${i}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.15 }} className={`flex flex-col gap-0.5 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                        <span className={`text-[8px] font-bold uppercase tracking-[0.2em] px-1 ${msg.role === "user" ? "text-emerald-500/50" : "text-slate-600"}`}>
                          {msg.role === "user" ? "You" : activeVoiceName}
                          {isPartial && <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse ml-1.5 align-middle" />}
                        </span>
                        <div className={`max-w-[90%] px-3 py-2 rounded-2xl text-[11px] leading-relaxed ${msg.role === "user" ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-100 rounded-tr-sm" : "bg-white/5 border border-white/10 text-slate-200 rounded-tl-sm"}`}>
                          {isPhoneAi && spokenIdx >= 0 ? (
                            <span>
                              {words.map((word, wi) => {
                                const isSpoken = wi <= spokenIdx;
                                const isCurrent = wi === spokenIdx;
                                return (
                                  <span
                                    key={wi}
                                    className={`transition-all duration-200 ${
                                      isCurrent
                                        ? "text-amber-300 font-semibold"
                                        : isSpoken
                                        ? "text-slate-100"
                                        : "text-slate-500"
                                    }`}
                                    style={isCurrent ? { textShadow: "0 0 8px rgba(251,191,36,0.5)" } : undefined}
                                  >
                                    {word}{wi < words.length - 1 ? " " : ""}
                                  </span>
                                );
                              })}
                            </span>
                          ) : (
                            msg.text
                          )}
                        </div>
                      </motion.div>
                      );
                    })}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </AnimatePresence>
            </div>

            {/* Chat Input (Gemini style) */}
            <div className="shrink-0 p-4 border-t border-white/5 bg-[#0c0c0e] z-20">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!chatInput.trim()) return;
                  if (viewingSession) setViewingSessionId(null);
                  sendText(chatInput.trim());
                  setChatInput("");
                }}
                className="relative flex items-center bg-white/[0.03] border border-white/10 rounded-2xl px-4 py-2 hover:bg-white/[0.05] transition-colors focus-within:bg-white/[0.05] focus-within:border-white/20"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 bg-transparent text-[11px] text-slate-200 placeholder:text-zinc-600 focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={!chatInput.trim()}
                  className="p-1 ml-2 text-slate-500 hover:text-emerald-400 disabled:opacity-50 disabled:hover:text-slate-500 transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </div>

          {/* ═══════════════════════════════════════
              RIGHT BENTO — Split Panels
          ═══════════════════════════════════════ */}
          <div className="flex-[0.3] flex flex-col gap-4 overflow-hidden h-full">
            
            {/* Row 1: Telephony Box */}
            <div className="flex-[0.55] rounded-3xl bg-[#0c0c0e] border border-white/[0.05] overflow-hidden shadow-2xl flex flex-col">
               <TelephonyBento activeVoiceId={activeVoiceId} />
            </div>

            {/* Row 2: Telemetry / Logs */}
            <div className="flex-[0.45] rounded-3xl bg-[#0c0c0e] border border-white/[0.05] overflow-hidden shadow-2xl flex flex-col">
               <TelemetryBento onPhoneMessage={handlePhoneMessage} onPhoneAiPartial={handlePhoneAiPartial} onPhoneTtsSpeaking={handlePhoneTtsSpeaking} />
            </div>

          </div>
        </div>
      </motion.div>
    </main>
  );
}
