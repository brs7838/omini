"use client";

import { useState, useEffect, useRef } from "react";
import { Activity, Terminal, Cpu, HardDrive, Copy, Trash2, Download } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface LogEntry {
  type: string;
  data: string | number | boolean | Record<string, unknown> | null;
  timestamp: number;
}

interface SysStatus {
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpus: Array<{
    index: number;
    name: string;
    util_pct: number;
    mem_used_mb: number;
    mem_total_mb: number;
    processes: Array<{
      pid: number;
      name: string;
      used_memory_mb: number;
    }>;
  }>;
}

interface TelemetryBentoProps {
  onPhoneMessage?: (msg: { role: "user" | "ai"; text: string }) => void;
}

export default function TelemetryBento({ onPhoneMessage }: TelemetryBentoProps) {
  const [activeTab, setActiveTab] = useState<"logs" | "sys">("sys");

  // System State
  const [sysStatus, setSysStatus] = useState<SysStatus | null>(null);

  // Logs State
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activeTab === "sys") {
      const fetchSys = () => {
        fetch("http://127.0.0.1:8000/system/status")
          .then(r => r.json())
          .then(setSysStatus)
          .catch(() => {});
      };
      fetchSys();
      interval = setInterval(fetchSys, 2500);
    }
    return () => { if (interval) clearInterval(interval); };
  }, [activeTab]);

  useEffect(() => {
    const connect = () => {
      const host = window.location.hostname || "localhost";
      const ws = new WebSocket(`ws://${host}:8000/ws/logs`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setLogs((prev) => [...prev.slice(-199), msg]);
        if (onPhoneMessage && typeof msg.data === "string") {
          if (msg.type === "phone_user") onPhoneMessage({ role: "user", text: msg.data });
          else if (msg.type === "phone_ai") onPhoneMessage({ role: "ai", text: msg.data });
        }
      };
      ws.onclose = () => { setTimeout(connect, 3000); };
      wsRef.current = ws;
    };
    connect();
    return () => wsRef.current?.close();
  }, [onPhoneMessage]);

  useEffect(() => {
    if (activeTab === "logs" && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, activeTab]);

  const getTime = (ts: number) => {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case "status": return "text-blue-400";
      case "log": return "text-emerald-400";
      case "transcript": return "text-purple-400";
      case "llm_chunk": return "text-zinc-500 text-[9px]";
      case "system": return "text-amber-400 font-bold";
      default: return "text-zinc-400";
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0c0c0e]">
      <div className="flex border-b border-white/5 bg-white/[0.02]">
        <button
          onClick={() => setActiveTab("sys")}
          className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2 ${
            activeTab === "sys" ? "text-emerald-400 bg-white/[0.05]" : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Activity className="w-3.5 h-3.5" /> Monitor
        </button>
        <button
          onClick={() => setActiveTab("logs")}
          className={`flex-1 py-3 text-xs font-bold uppercase tracking-widest border-l border-white/5 transition-colors flex items-center justify-center gap-2 ${
            activeTab === "logs" ? "text-blue-400 bg-white/[0.05]" : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Terminal className="w-3.5 h-3.5" /> Logs
        </button>
      </div>

      <div className="flex-1 overflow-hidden relative">
        <AnimatePresence mode="wait">
          
          {/* SYSTEM TAB */}
          {activeTab === "sys" && (
            <motion.div
              key="sys"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 p-4 overflow-y-auto custom-scrollbar flex flex-col gap-4 text-xs"
            >
              {!sysStatus ? (
                <div className="flex-1 flex items-center justify-center text-zinc-600">Gathering metrics...</div>
              ) : (
                <>
                  <div className="bg-white/5 border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                    <div className="flex items-center gap-2 mb-1"><Cpu className="w-4 h-4 text-amber-500"/> <span className="font-bold text-white tracking-widest uppercase">System</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">CPU Usage</span> <span className="text-white">{sysStatus.cpu_percent}%</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">RAM</span> <span className="text-white">{sysStatus.ram_used_gb} / {sysStatus.ram_total_gb} GB</span></div>
                  </div>

                  {sysStatus.gpus?.map((g, i) => (
                    <div key={i} className="bg-white/5 border border-white/5 p-3 rounded-lg flex flex-col gap-1">
                      <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-2"><HardDrive className="w-4 h-4 text-indigo-500"/> <span className="font-bold text-white tracking-widest uppercase truncate max-w-[150px]">{g.name}</span></div>
                        <span className="text-[9px] text-zinc-500 bg-black px-1.5 py-0.5 rounded">GPU {g.index}</span>
                      </div>
                      <div className="flex justify-between mt-2"><span className="text-zinc-500">Usage</span> <span className="text-white">{g.util_pct}%</span></div>
                      <div className="flex justify-between"><span className="text-zinc-500">VRAM</span> <span className="text-amber-400">{g.mem_used_mb}MB / {g.mem_total_mb}MB</span></div>
                      
                      {g.processes?.length > 0 && (
                        <div className="mt-2 border-t border-white/5 pt-2 flex flex-col gap-1">
                          {g.processes.map((p, idx) => (
                            <div key={idx} className="flex justify-between text-[10px]">
                              <span className="text-zinc-500 truncate max-w-[120px]">{p.name} (PID: {p.pid})</span>
                              <span className="text-emerald-400">{p.used_memory_mb} MB</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </>
              )}
            </motion.div>
          )}

          {/* LOGS TAB */}
          {activeTab === "logs" && (
            <motion.div
              key="logs"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 flex flex-col overflow-hidden"
            >
              {/* Log Actions Toolbar */}
              <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/5 bg-white/[0.01]">
                <div className="text-[9px] font-bold uppercase tracking-[0.2em] text-zinc-500">Live Stream</div>
                <div className="flex items-center gap-1">
                  <button
                    title="Copy to clipboard"
                    onClick={() => {
                      const text = logs.map(l => `[${getTime(l.timestamp)}] [${l.type}] ${typeof l.data === 'string' ? l.data : JSON.stringify(l.data)}`).join('\n');
                      navigator.clipboard.writeText(text);
                    }}
                    className="p-1 text-zinc-500 hover:text-emerald-400 hover:bg-white/5 rounded transition-all"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                  <button
                    title="Download logs"
                    onClick={() => {
                      const text = logs.map(l => `[${getTime(l.timestamp)}] [${l.type}] ${typeof l.data === 'string' ? l.data : JSON.stringify(l.data)}`).join('\n');
                      const blob = new Blob([text], { type: 'text/plain' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `vaani_logs_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.txt`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="p-1 text-zinc-500 hover:text-blue-400 hover:bg-white/5 rounded transition-all"
                  >
                    <Download className="w-3 h-3" />
                  </button>
                  <button
                    title="Clear logs"
                    onClick={() => setLogs([])}
                    className="p-1 text-zinc-500 hover:text-red-400 hover:bg-white/5 rounded transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>

              <div 
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-4 font-mono text-[9px] space-y-2 selection:bg-blue-500/30 break-words leading-relaxed custom-scrollbar"
              >
                {logs.length === 0 && (
                  <div className="text-white/20 italic mt-8 text-center">Waiting for stream...</div>
                )}
                {logs.map((log, i) => (
                  <div key={i} className="flex gap-2 border-l border-white/5 pl-2 hover:bg-white/5 transition-colors">
                    <span className="text-white/20 shrink-0">{getTime(log.timestamp)}</span>
                    <span className={`flex-1 break-words ${getLogColor(log.type)}`}>
                      {log.type !== 'llm_chunk' && <span className="opacity-50 text-[8px] uppercase mr-1.5 shrink-0">[{log.type}]</span>}
                      {typeof log.data === 'string' ? log.data : JSON.stringify(log.data)}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  );
}
