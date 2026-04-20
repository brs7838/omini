"use client";

import React, { useEffect, useState, useRef } from "react";
import { Terminal, ScrollText, XCircle, Copy, Check } from "lucide-react";

interface LogEntry {
  type: string;
  data: string | number | boolean | Record<string, unknown> | null;
  timestamp: number;
}

export default function LogViewer() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const handleCopy = () => {
    const logText = logs.map(log => {
      const time = new Date(log.timestamp * 1000).toLocaleTimeString();
      const data = typeof log.data === 'string' ? log.data : JSON.stringify(log.data);
      return `[${time}] [${log.type}] ${data}`;
    }).join('\n');

    navigator.clipboard.writeText(logText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    const connect = () => {
      const host = window.location.hostname || "localhost";
      const ws = new WebSocket(`ws://${host}:8000/ws/logs`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        setLogs((prev) => [...prev.slice(-199), msg]); // Keep last 200 logs
      };
      ws.onclose = () => {
        setTimeout(connect, 3000); // Reconnect
      };
      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const getTime = (ts: number) => {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case "status": return "text-blue-400";
      case "log": return "text-green-400";
      case "transcript": return "text-purple-400";
      case "llm_chunk": return "text-gray-500 text-xs";
      case "system": return "text-yellow-400 font-bold";
      default: return "text-gray-300";
    }
  };

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 p-3 bg-zinc-900/80 border border-white/10 rounded-full hover:bg-zinc-800 transition-all shadow-xl group z-50"
        title="Open Live Logs"
      >
        <Terminal className="w-5 h-5 text-green-400 group-hover:scale-110 transition-transform" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 w-96 h-[450px] bg-black/90 border border-white/10 rounded-xl shadow-2xl flex flex-col overflow-hidden z-50 backdrop-blur-xl animate-in slide-in-from-bottom-5 duration-300">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-zinc-900/50">
        <div className="flex items-center gap-2">
          <ScrollText className="w-4 h-4 text-green-400" />
          <span className="text-xs font-mono font-bold tracking-wider text-white/80 uppercase">Live Debug Console</span>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={handleCopy}
            className={`p-1.5 rounded-md transition-all ${copied ? 'text-emerald-400 bg-emerald-500/10' : 'text-white/40 hover:text-white hover:bg-white/10'}`}
            title="Copy Logs"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          <button onClick={() => setIsOpen(false)} className="hover:text-red-400 text-white/40 transition-colors">
            <XCircle className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 font-mono text-[10px] space-y-2 selection:bg-green-500/30"
      >
        {logs.length === 0 && (
          <div className="text-white/20 italic text-center mt-20">Waiting for events...</div>
        )}
        {logs.map((log, i) => (
          <div key={i} className="flex gap-2 border-l border-white/5 pl-2 hover:bg-white/5 transition-colors group">
            <span className="text-white/20 select-none">{getTime(log.timestamp)}</span>
            <span className={`flex-1 break-words ${getLogColor(log.type)}`}>
              <span className="opacity-50 text-[8px] uppercase mr-2">[{log.type}]</span>
              {typeof log.data === 'string' ? log.data : JSON.stringify(log.data)}
            </span>
          </div>
        ))}
      </div>
      
      <div className="px-4 py-2 border-t border-white/10 bg-zinc-900/30 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-[9px] font-mono text-white/40 uppercase tracking-tight">Stream Active</span>
        </div>
        <button 
          onClick={() => setLogs([])}
          className="text-[9px] uppercase font-bold text-white/30 hover:text-white transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
