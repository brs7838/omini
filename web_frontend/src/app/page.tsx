"use client";

import VocalisOrb from "@/components/VocalisOrb";
import Starfield from "@/components/Starfield";
import Sidebar from "@/components/Sidebar";
import { useVoice } from "@/hooks/useVoice";
import { Mic, MicOff, Settings2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import VoiceLibrary from "@/components/VoiceLibrary";
import LogViewer from "@/components/LogViewer";
import Dialer from "@/components/Dialer";
import CallHistory from "@/components/CallHistory";
import { Phone, History } from "lucide-react";

export default function VocalisDashboard() {
  const [activeVoiceId, setActiveVoiceId] = useState("ravi");
  const { state, messages, isLive, toggleLive, switchVoice } = useVoice(activeVoiceId);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isVoiceLibraryOpen, setIsVoiceLibraryOpen] = useState(false);
  const [isDialerOpen, setIsDialerOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const handleDial = async (number: string) => {
    try {
      const resp = await fetch("http://127.0.0.1:8000/calls/dial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: number })
      });
      if (!resp.ok) console.error("Dial failed");
      else setIsHistoryOpen(true); // Open history to show dialing status
    } catch (e) {
      console.error("Dial request error", e);
    }
  };

  // Auto-scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <main className="flex h-screen bg-black text-white relative overflow-hidden font-sans selection:bg-emerald-500/30">
      <Starfield />

      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />

      {/* Main Area */}
      <motion.div
        animate={{ paddingLeft: isSidebarCollapsed ? 80 : 300 }}
        className="flex-1 relative flex flex-col items-center transition-all duration-300 ease-in-out"
      >

        {/* Orb + State Badge */}
        <section className="relative w-[18rem] h-[18rem] flex items-center justify-center mt-6 z-10 shrink-0">
            <VocalisOrb state={state} />
            <div className="absolute -bottom-2">
                 <AnimatePresence mode="wait">
                    <motion.div
                        key={state}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className={`px-3 py-1 rounded-full text-[8px] font-extrabold tracking-[0.4em] uppercase border backdrop-blur-xl ${
                            state === 'listening' ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400' :
                            state === 'speaking' ? 'bg-orange-500/10 border-orange-500/40 text-orange-400' :
                            state === 'thinking' ? 'bg-indigo-500/10 border-indigo-500/40 text-indigo-400' :
                            'bg-slate-500/10 border-slate-500/40 text-slate-400'
                        }`}
                    >
                        {state}
                    </motion.div>
                 </AnimatePresence>
            </div>
        </section>

        {/* Chat Messages — restricted to area below orb */}
        <div className="flex-1 w-full max-w-xl overflow-y-auto px-6 py-12 z-20 chat-fade-mask">
          <div className="flex flex-col gap-3">
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[75%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-emerald-500/15 border border-emerald-500/20 text-emerald-100 rounded-br-sm'
                    : 'bg-white/8 border border-white/10 text-slate-200 rounded-bl-sm'
                }`}>
                  <span className={`text-[9px] uppercase tracking-widest font-bold block mb-1 ${
                    msg.role === 'user' ? 'text-emerald-400/60' : 'text-slate-500'
                  }`}>
                    {msg.role === 'user' ? 'You' : 'Vaani'}
                  </span>
                  {msg.text}
                </div>
              </motion.div>
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Mic Button — bottom */}
        <footer className="pb-8 pt-4 z-40 shrink-0 flex items-center gap-6">
            <button
                onClick={() => setIsVoiceLibraryOpen(true)}
                className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-400 hover:bg-white/10 hover:text-white transition-all"
                title="Voice Library"
            >
                <Settings2 className="w-5 h-5" />
            </button>

            <button
                onClick={toggleLive}
                className={`group relative w-16 h-16 rounded-full transition-all duration-500 flex items-center justify-center border-2 ${
                    isLive
                    ? 'bg-red-500/20 text-red-500 border-red-500/40 shadow-[0_0_60px_rgba(239,68,68,0.2)]'
                    : 'bg-white text-black border-transparent hover:scale-105 shadow-2xl shadow-blue-500/10'
                }`}
            >
                {isLive ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
            </button>
            
            <div className="flex items-center gap-4">
              <button
                  onClick={() => setIsDialerOpen(true)}
                  className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-400 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all"
                  title="Phone Dialer"
              >
                  <Phone className="w-5 h-5" />
              </button>

              <button
                  onClick={() => setIsHistoryOpen(true)}
                  className="w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-400 hover:bg-indigo-500/10 hover:text-indigo-400 transition-all"
                  title="Call History"
              >
                  <History className="w-5 h-5" />
              </button>
            </div>
        </footer>

        <Dialer 
          isOpen={isDialerOpen} 
          onClose={() => setIsDialerOpen(false)} 
          onDial={handleDial} 
        />

        <CallHistory 
          isOpen={isHistoryOpen} 
          onClose={() => setIsHistoryOpen(false)} 
        />

        <VoiceLibrary 
          isOpen={isVoiceLibraryOpen} 
          onClose={() => setIsVoiceLibraryOpen(false)}
          activeVoiceId={activeVoiceId}
          onSelect={(id) => {
             setActiveVoiceId(id);
             switchVoice(id);
             // Keep open for a second to show selection then close
             setTimeout(() => setIsVoiceLibraryOpen(false), 800);
          }}
        />

        <LogViewer />
      </motion.div>
    </main>
  );
}
