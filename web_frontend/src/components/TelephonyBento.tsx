"use client";

import { useState, useEffect } from "react";
import { Phone, Users, History, PhoneOutgoing, PhoneIncoming, Clock, PhoneOff, Trash2, UserPlus, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface CallRecord {
  id: string;
  phone: string;
  timestamp: string;
  status: string;
  type: string;
}

interface Contact {
  id: string;
  name: string;
  phone: string;
}

interface TelephonyBentoProps {
  activeVoiceId: string;
}

export default function TelephonyBento({ activeVoiceId }: TelephonyBentoProps) {
  const [activeTab, setActiveTab] = useState<"dialer" | "contacts" | "recent">("dialer");
  
  // Dialer State
  const [number, setNumber] = useState("");
  const [isDialing, setIsDialing] = useState(false);
  
  // Call History State
  const [history, setHistory] = useState<CallRecord[]>([]);
  const [dialingPhone, setDialingPhone] = useState<string | null>(null);

  // Contacts State
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [isAddingContact, setIsAddingContact] = useState(false);
  const [newContactName, setNewContactName] = useState("");
  const [newContactPhone, setNewContactPhone] = useState("");

  // Load Contacts from localStorage
  useEffect(() => {
    const t = setTimeout(() => {
      try {
        const stored = localStorage.getItem("vaani_contacts");
        if (stored) setContacts(JSON.parse(stored));
      } catch {}
    }, 0);
    return () => clearTimeout(t);
  }, []);

  const saveContacts = (c: Contact[]) => {
    setContacts(c);
    localStorage.setItem("vaani_contacts", JSON.stringify(c));
  };

  const handleCreateContact = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newContactName.trim() || !newContactPhone.trim()) return;
    const newContact = {
      id: crypto.randomUUID(),
      name: newContactName.trim(),
      phone: newContactPhone.replace(/[^0-9+]/g, "")
    };
    saveContacts([...contacts, newContact]);
    setIsAddingContact(false);
    setNewContactName("");
    setNewContactPhone("");
  };

  const deleteContact = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    saveContacts(contacts.filter(c => c.id !== id));
  };

  // Fetch History
  const fetchHistory = async () => {
    try {
      const resp = await fetch("http://127.0.0.1:8000/calls/history");
      if (resp.ok) {
        const data = await resp.json();
        setHistory(data);
      }
    } catch {}
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activeTab === "recent") {
      const t = setTimeout(() => void fetchHistory(), 0);
      interval = setInterval(() => void fetchHistory(), 5000);
      return () => {
        clearTimeout(t);
        if (interval) clearInterval(interval);
      };
    }
    return () => { if (interval) clearInterval(interval); };
  }, [activeTab]);

  const placeCall = async (phoneToDial: string) => {
    if (!phoneToDial || dialingPhone || isDialing) return;
    setDialingPhone(phoneToDial);
    setIsDialing(true);
    try {
      const resp = await fetch("http://127.0.0.1:8000/calls/dial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone_number: phoneToDial, voice_id: activeVoiceId }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Dial failed" }));
        alert(err.detail || "Dial failed");
      } else {
        setActiveTab("recent"); // Switch to recent tab to show the status
      }
      void fetchHistory();
    } catch (e) {
      console.error("Call failed", e);
      alert("Could not reach backend");
    } finally {
      setIsDialing(false);
      setDialingPhone(null);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0c0c0e]">
      {/* Bento Header / Tabs */}
      <div className="flex border-b border-white/5 bg-white/[0.02]">
        <button
          onClick={() => setActiveTab("dialer")}
          className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2 ${
            activeTab === "dialer" ? "text-emerald-400 bg-white/[0.05]" : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Phone className="w-3.5 h-3.5" /> Dial
        </button>
        <button
          onClick={() => setActiveTab("contacts")}
          className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest border-x border-white/5 transition-colors flex items-center justify-center gap-2 ${
            activeTab === "contacts" ? "text-blue-400 bg-white/[0.05]" : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Users className="w-3.5 h-3.5" /> Contacts
        </button>
        <button
          onClick={() => setActiveTab("recent")}
          className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2 ${
            activeTab === "recent" ? "text-purple-400 bg-white/[0.05]" : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <History className="w-3.5 h-3.5" /> Recent
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden relative">
        <AnimatePresence mode="wait">
          
          {/* DIALER TAB */}
          {activeTab === "dialer" && (
            <motion.div
              key="dialer"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 flex flex-col items-center justify-between p-3 overflow-hidden"
            >
              <div className="w-full text-center flex items-center justify-center shrink-0 h-10 mb-2">
                <span className="text-xl font-light tracking-widest text-white truncate px-4">
                  {number || <span className="text-zinc-700">Enter number</span>}
                </span>
              </div>
              <div className="w-full max-w-[220px] flex-1 grid grid-cols-3 gap-3 min-h-0 mb-4 px-2">
                {["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"].map(btn => (
                  <button
                    key={btn}
                    onClick={() => { if (number.length < 15) setNumber(p => p + btn); }}
                    className="aspect-square rounded-full bg-white/[0.03] border border-white/5 hover:bg-white/10 active:scale-90 transition-all text-sm font-bold text-zinc-300 flex items-center justify-center shadow-lg hover:shadow-emerald-500/5 hover:border-emerald-500/20"
                  >
                    {btn}
                  </button>
                ))}
              </div>
              <div className="flex gap-4 items-center shrink-0 h-12">
                <button
                  onClick={() => setNumber(p => p.slice(0, -1))}
                  className="w-10 h-10 rounded-full flex items-center justify-center text-zinc-600 hover:text-zinc-400 disabled:opacity-30 transition-colors bg-white/5"
                  disabled={!number}
                >
                  <X className="w-4 h-4" />
                </button>
                <button
                  onClick={() => { placeCall(number); setNumber(""); }}
                  disabled={!number || isDialing}
                  className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                    number && !isDialing
                      ? 'bg-emerald-500 text-black shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:scale-105 active:scale-95' 
                      : 'bg-zinc-800 text-zinc-600 grayscale cursor-not-allowed'
                  }`}
                >
                  <Phone className="w-4 h-4 fill-current" />
                </button>
              </div>
            </motion.div>
          )}

          {/* CONTACTS TAB */}
          {activeTab === "contacts" && (
            <motion.div
              key="contacts"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 flex flex-col p-4 overflow-hidden"
            >
              {isAddingContact ? (
                <div className="flex-1 flex flex-col justify-center animate-in fade-in zoom-in-95">
                  <h4 className="text-sm font-bold text-white mb-4">New Contact</h4>
                  <form onSubmit={handleCreateContact} className="flex flex-col gap-3">
                    <input 
                      type="text" 
                      placeholder="Name (e.g. Rahul)" 
                      value={newContactName}
                      onChange={e => setNewContactName(e.target.value)}
                      className="bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-blue-500"
                      autoFocus
                    />
                    <input 
                      type="tel" 
                      placeholder="Phone Number" 
                      value={newContactPhone}
                      onChange={e => setNewContactPhone(e.target.value)}
                      className="bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-blue-500"
                    />
                    <div className="flex gap-2 mt-2">
                       <button type="button" onClick={() => setIsAddingContact(false)} className="flex-1 py-3 rounded-lg text-sm font-medium text-zinc-400 bg-white/5 hover:bg-white/10">Cancel</button>
                       <button type="submit" className="flex-1 py-3 rounded-lg text-sm font-medium text-black bg-blue-500 hover:bg-blue-400 shadow-lg shadow-blue-500/20">Save</button>
                    </div>
                  </form>
                </div>
              ) : (
                <>
                  <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 flex flex-col gap-2">
                    {contacts.length === 0 ? (
                      <div className="flex-1 flex flex-col items-center justify-center text-center">
                        <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center mb-3">
                          <Users className="w-5 h-5 text-blue-400" />
                        </div>
                        <p className="text-zinc-500 text-xs">No contacts saved</p>
                      </div>
                    ) : (
                      contacts.map((c) => (
                        <div key={c.id} className="flex items-center justify-between p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 transition-all group">
                          <div className="relative flex-1 cursor-pointer" onClick={() => placeCall(c.phone)}>
                            <div className="font-medium text-sm text-white">{c.name}</div>
                            <div className="text-xs text-zinc-500 tracking-wider font-mono mt-0.5">{c.phone}</div>
                          </div>
                          <button onClick={(e) => deleteContact(c.id, e)} className="p-2 text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all rounded-lg hover:bg-red-500/10">
                             <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <button 
                    onClick={() => setIsAddingContact(true)}
                    className="w-full mt-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white text-sm font-medium flex items-center justify-center gap-2 hover:bg-white/10 transition-colors"
                  >
                    <UserPlus className="w-4 h-4 text-blue-400" /> Add Contact
                  </button>
                </>
              )}
            </motion.div>
          )}

          {/* RECENT TAB */}
          {activeTab === "recent" && (
            <motion.div
              key="recent"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 flex flex-col p-4 overflow-y-auto custom-scrollbar pr-2"
            >
              <div className="flex flex-col gap-2">
                {history.length === 0 ? (
                  <div className="py-12 text-center flex flex-col items-center">
                    <div className="w-10 h-10 bg-white/5 rounded-full flex items-center justify-center mb-3">
                       <Clock className="w-4 h-4 text-zinc-600" />
                    </div>
                    <p className="text-zinc-500 text-xs">No recent calls</p>
                  </div>
                ) : (
                  history.map((call) => {
                    const isLive = call.status === 'dialing' || call.status === 'active';
                    const isRedialing = dialingPhone === call.phone;
                    return (
                      <div
                        key={call.id}
                        onClick={() => { if (!isLive && !dialingPhone) void placeCall(call.phone); }}
                        className={`p-3 rounded-xl bg-white/5 border border-white/5 transition-all group flex items-center justify-between ${
                          isLive ? 'cursor-default border-emerald-500/20 bg-emerald-500/5' : 'cursor-pointer hover:bg-white/[0.08]'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                           <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
                              call.type === 'outbound' ? 'bg-zinc-800 text-zinc-400 group-hover:bg-zinc-700' : 'bg-zinc-800 text-purple-400'
                           }`}>
                              {call.type === 'outbound' ? <PhoneOutgoing className="w-3 h-3" /> : <PhoneIncoming className="w-3 h-3" />}
                           </div>
                           <div className="flex flex-col">
                              <span className="text-white font-medium text-xs tracking-tight">{call.phone}</span>
                              <span className="text-[9px] text-zinc-500 mt-0.5">{call.timestamp.split(" ")[1] ?? call.timestamp}</span>
                           </div>
                        </div>
                        <div className="flex items-center gap-2">
                           {isLive && (
                             <button
                               onClick={async (e) => {
                                 e.stopPropagation();
                                 try {
                                    await fetch("http://127.0.0.1:8000/calls/hangup", { method: "POST" });
                                    setTimeout(fetchHistory, 500);
                                  } catch {}
                               }}
                               className="p-1.5 rounded-full bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white transition-all active:scale-90"
                               title="End Call"
                             >
                               <PhoneOff className="w-3 h-3" />
                             </button>
                           )}
                           <span className="text-[9px] uppercase font-bold tracking-wider text-zinc-500">
                             {isRedialing ? '...' : call.status}
                           </span>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
