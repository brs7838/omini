"use client";

import { History, Plus, Menu, Settings, Zap, Trash2, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useMemo } from "react";

export interface ChatSession {
  id: string;
  title: string;
  messages: { role: "user" | "ai"; text: string }[];
  createdAt: number;
  updatedAt: number;
}

interface SidebarProps {
  isCollapsed: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onSettingsClick: () => void;
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
}

function groupByDate(sessions: ChatSession[]) {
  const now = Date.now();
  const DAY = 86400000;
  const today: ChatSession[] = [];
  const yesterday: ChatSession[] = [];
  const week: ChatSession[] = [];
  const older: ChatSession[] = [];

  for (const s of sessions) {
    const age = now - s.updatedAt;
    if (age < DAY) today.push(s);
    else if (age < 2 * DAY) yesterday.push(s);
    else if (age < 7 * DAY) week.push(s);
    else older.push(s);
  }
  return [
    { label: "Today", items: today },
    { label: "Yesterday", items: yesterday },
    { label: "Previous 7 days", items: week },
    { label: "Older", items: older },
  ].filter(g => g.items.length > 0);
}

export default function Sidebar({
  isCollapsed,
  onToggle,
  onNewChat,
  onSettingsClick,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
}: SidebarProps) {
  const grouped = useMemo(() => groupByDate(sessions), [sessions]);

  return (
    <motion.aside
      initial={false}
      animate={{ width: isCollapsed ? 80 : 300 }}
      className="fixed left-0 top-0 h-full bg-slate-950/60 backdrop-blur-3xl border-r border-white/5 z-50 flex flex-col shadow-2xl"
    >
      {/* Header */}
      <div className="p-4 flex flex-col items-center">
        <div className={`w-full flex ${isCollapsed ? "justify-center" : "justify-between"} items-center mb-10 mt-2 px-2`}>
          {!isCollapsed && (
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-emerald-400 fill-emerald-400" />
              <span className="font-bold text-lg tracking-tight">Vaani</span>
            </div>
          )}
          <button
            onClick={onToggle}
            className="p-2.5 hover:bg-white/10 rounded-xl transition-colors text-slate-400"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>

        <button
          onClick={onNewChat}
          className={`flex items-center gap-3 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-2xl p-4 transition-all group overflow-hidden ${
            isCollapsed ? "justify-center w-12 h-12 p-0 shadow-lg shadow-emerald-500/10" : "w-full shadow-md"
          }`}
        >
          <Plus className="w-5 h-5 text-emerald-400 shrink-0" />
          {!isCollapsed && (
            <span className="text-sm font-semibold text-emerald-100 whitespace-nowrap">New Chat</span>
          )}
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto px-3 scrollbar-hide">
        {!isCollapsed && sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-slate-600">
            <MessageSquare className="w-8 h-8" />
            <p className="text-xs font-medium text-center leading-relaxed px-4">
              Your conversations<br />will appear here
            </p>
          </div>
        )}

        {!isCollapsed && grouped.map(group => (
          <div key={group.label} className="mb-4">
            <div className="text-slate-600 text-[10px] font-bold uppercase tracking-[0.25em] mb-2 ml-2 mt-2">
              {group.label}
            </div>
            <div className="space-y-0.5">
              <AnimatePresence initial={false}>
                {group.items.map(s => {
                  const isActive = s.id === activeSessionId;
                  return (
                    <motion.div
                      key={s.id}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -16, transition: { duration: 0.15 } }}
                      className="group relative"
                    >
                      <button
                        onClick={() => onSelectSession(s.id)}
                        className={`w-full text-left px-3 py-2.5 rounded-xl transition-colors flex items-center gap-3 pr-9 ${
                          isActive
                            ? "bg-white/10 text-white"
                            : "hover:bg-white/5 text-slate-400 hover:text-white"
                        }`}
                      >
                        <History className={`w-3.5 h-3.5 shrink-0 ${isActive ? "text-emerald-400" : "text-slate-600 group-hover:text-slate-400"}`} />
                        <span className="text-sm truncate font-medium leading-snug">
                          {s.title}
                        </span>
                      </button>
                      <button
                        onClick={(e) => onDeleteSession(s.id, e)}
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1.5 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all rounded-lg hover:bg-red-500/10"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </div>
        ))}

        {/* Collapsed: just icons */}
        {isCollapsed && sessions.slice(0, 8).map(s => (
          <button
            key={s.id}
            onClick={() => onSelectSession(s.id)}
            className={`w-full flex justify-center p-3 rounded-xl transition-colors mb-1 ${
              s.id === activeSessionId
                ? "bg-white/10 text-emerald-400"
                : "text-slate-600 hover:text-emerald-400 hover:bg-white/5"
            }`}
            title={s.title}
          >
            <History className="w-4 h-4" />
          </button>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 mt-auto border-t border-white/5 flex flex-col gap-2 pb-4">
        <button
          onClick={onSettingsClick}
          className={`w-full flex items-center gap-4 p-2.5 rounded-xl hover:bg-white/5 transition-colors group ${isCollapsed ? "justify-center" : ""}`}
        >
          <Settings className="w-4 h-4 text-slate-500 group-hover:text-slate-300 shrink-0" />
          {!isCollapsed && <span className="text-sm text-slate-400 font-medium">Model Settings</span>}
        </button>

      </div>
    </motion.aside>
  );
}
