"use client";

import { motion, AnimatePresence } from "framer-motion";
import { PhoneOutgoing, PhoneIncoming, Clock, CheckCircle2, XCircle, AlertCircle, PhoneOff } from "lucide-react";
import { useEffect, useState } from "react";

interface CallRecord {
  id: string;
  phone: string;
  timestamp: string;
  status: string;
  type: string;
}

interface CallHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  activeVoiceId: string;
}

export default function CallHistory({ isOpen, onClose, activeVoiceId }: CallHistoryProps) {
  const [history, setHistory] = useState<CallRecord[]>([]);
  const [dialingPhone, setDialingPhone] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      const resp = await fetch("http://127.0.0.1:8000/calls/history");
      if (resp.ok) {
        const data = await resp.json();
        setHistory(data);
      }
    } catch (e) {
      console.error("Failed to fetch call history", e);
    }
  };

  const redial = async (phone: string) => {
    if (dialingPhone) return;
    setDialingPhone(phone);
    try {
      const resp = await fetch("http://127.0.0.1:8000/calls/dial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Use the currently selected voice for redials (was hardcoded "ravi").
        body: JSON.stringify({ phone_number: phone, voice_id: activeVoiceId }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Dial failed" }));
        alert(err.detail || "Dial failed");
      }
      await fetchHistory();
    } catch (e) {
      console.error("Redial failed", e);
      alert("Could not reach backend");
    } finally {
      setDialingPhone(null);
    }
  };


  useEffect(() => {
    if (isOpen) {
      // Defer initial fetch to avoid cascading render warning
      const timeout = setTimeout(() => {
        void fetchHistory();
      }, 0);

      const interval = setInterval(() => {
        void fetchHistory();
      }, 5000);

      return () => {
        clearTimeout(timeout);
        clearInterval(interval);
      };
    }
  }, [isOpen]);


  const getStatusIcon = (status: string) => {
    switch (status) {
      case "answered": return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case "dialing": return <motion.div animate={{ opacity: [1, 0.4, 1] }} transition={{ repeat: Infinity, duration: 1.5 }}><PhoneOutgoing className="w-4 h-4 text-blue-400" /></motion.div>;
      case "failed": return <XCircle className="w-4 h-4 text-red-400" />;
      default: return <AlertCircle className="w-4 h-4 text-zinc-500" />;
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            className="w-full max-w-2xl max-h-[85vh] bg-[#0c0c0e] border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden backdrop-blur-xl"
          >
          <div className="p-8 flex justify-between items-center border-b border-white/5">
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">Call History</h2>
              <p className="text-[10px] uppercase tracking-widest text-zinc-500 mt-1 font-bold">Recent activity</p>
            </div>
            <button onClick={onClose} className="p-2 -mr-2 text-zinc-500 hover:text-white transition-colors">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
            <div className="flex flex-col gap-2">
              {history.length === 0 ? (
                <div className="py-20 text-center">
                  <div className="w-12 h-12 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-4">
                     <Clock className="w-6 h-6 text-zinc-700" />
                  </div>
                  <p className="text-zinc-500 text-sm italic">No recent calls found</p>
                </div>
              ) : (
                history.map((call) => {
                  const isLive = call.status === 'dialing' || call.status === 'active';
                  const isRedialing = dialingPhone === call.phone;
                  return (
                  <motion.div
                    key={call.id}
                    layoutId={call.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    onClick={() => { if (!isLive && !dialingPhone) void redial(call.phone); }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if ((e.key === 'Enter' || e.key === ' ') && !isLive && !dialingPhone) {
                        e.preventDefault();
                        void redial(call.phone);
                      }
                    }}
                    className={`p-4 rounded-2xl bg-white/5 border border-white/5 transition-all group ${
                      isLive ? 'cursor-default' : 'cursor-pointer hover:bg-white/[0.08] hover:border-white/10'
                    } ${isRedialing ? 'opacity-70' : ''}`}
                    title={isLive ? undefined : `Click to call ${call.phone}`}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <div className="flex items-center gap-3">
                         <div className="w-10 h-10 rounded-full bg-zinc-900 flex items-center justify-center text-zinc-400 group-hover:bg-zinc-800 transition-colors">
                            {call.type === 'outbound' ? <PhoneOutgoing className="w-4 h-4" /> : <PhoneIncoming className="w-4 h-4" />}
                         </div>
                         <div className="flex flex-col">
                            <span className="text-white font-medium tracking-tight text-sm">{call.phone}</span>
                            <span className="text-[10px] text-zinc-500 font-medium">{call.timestamp}</span>
                         </div>
                      </div>
                      <div className="flex items-center gap-2 pr-2">
                         <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">{isRedialing ? 'dialing...' : call.status}</span>
                         {getStatusIcon(isRedialing ? 'dialing' : call.status)}

                         {/* Hangup Button for active/dialing calls */}
                         {isLive && (
                           <button
                             onClick={async (e) => {
                               e.stopPropagation();
                               try {
                                 await fetch("http://127.0.0.1:8000/calls/hangup", { method: "POST" });
                                 void fetchHistory(); // Refresh
                               } catch (err) { console.error("Hangup failed", err); }
                             }}
                             className="ml-2 p-1.5 rounded-full bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white transition-all shadow-[0_0_15px_rgba(239,68,68,0.1)] active:scale-90"
                             title="End Call"
                           >
                             <PhoneOff className="w-3 h-3" />
                           </button>
                         )}
                      </div>
                    </div>
                  </motion.div>
                  );
                })
              )}
            </div>
          </div>

          <div className="p-8 border-t border-white/5 text-[9px] uppercase tracking-[0.3em] font-black text-center text-zinc-700 bg-zinc-900/30">
             Vaani Telephony Service
          </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
