"use client";

import { History, Plus, Menu, Settings, Zap } from "lucide-react";
import { motion } from "framer-motion";

interface SidebarProps {
  isCollapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ isCollapsed, onToggle }: SidebarProps) {
  const sessions = [
    { id: 1, title: "Natural Hinglish Chat", date: "Today" },
    { id: 2, title: "Haryanvi Discussion", date: "Yesterday" },
  ];

  return (
    <motion.aside 
      initial={false}
      animate={{ width: isCollapsed ? 80 : 300 }}
      className="fixed left-0 top-0 h-full bg-slate-950/60 backdrop-blur-3xl border-r border-white/5 z-50 flex flex-col transition-all duration-300 ease-in-out shadow-2xl"
    >
      {/* Upper Section & Toggle */}
      <div className="p-4 flex flex-col items-center">
        <div className={`w-full flex ${isCollapsed ? 'justify-center' : 'justify-between'} items-center mb-10 mt-2 px-2`}>
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

        <button className={`flex items-center gap-3 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-2xl p-4 transition-all group overflow-hidden ${isCollapsed ? 'justify-center w-12 h-12 p-0 shadow-lg shadow-emerald-500/10' : 'w-full shadow-md'}`}>
          <Plus className="w-5 h-5 text-emerald-400 shrink-0" />
          {!isCollapsed && <span className="text-sm font-semibold text-emerald-100 whitespace-nowrap">New Chat</span>}
        </button>
      </div>

      {/* Navigation / History */}
      <div className="flex-1 overflow-y-auto px-4 mt-8 scrollbar-hide">
        {!isCollapsed && (
          <div className="flex items-center gap-2 text-slate-500 text-[10px] font-bold uppercase tracking-[0.3em] mb-6 ml-3">
            Recents
          </div>
        )}
        
        <div className="space-y-3">
          {sessions.map((s) => (
            <button key={s.id} className={`w-full text-left p-3 rounded-xl hover:bg-white/5 transition-colors group flex items-center gap-4 ${isCollapsed ? 'justify-center' : ''}`}>
              <History className={`w-4 h-4 shrink-0 transition-colors ${isCollapsed ? 'text-slate-500 group-hover:text-emerald-400' : 'text-slate-600'}`} />
              {!isCollapsed && (
                <div className="min-w-0">
                  <div className="text-sm text-slate-400 group-hover:text-white transition-colors truncate font-medium">
                    {s.title}
                  </div>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Lower Actions */}
      <div className="p-4 mt-auto border-t border-white/5 flex flex-col gap-2 pb-4">
        <button className={`w-full flex items-center gap-4 p-2.5 rounded-xl hover:bg-white/5 transition-colors group ${isCollapsed ? 'justify-center' : ''}`}>
           <Settings className="w-4 h-4 text-slate-500 group-hover:text-slate-300 shrink-0" />
           {!isCollapsed && <span className="text-sm text-slate-400 font-medium">Model Settings</span>}
        </button>

        <div className={`flex items-center gap-3 rounded-2xl transition-all ${isCollapsed ? 'justify-center p-2 mx-auto' : 'p-2.5 bg-white/5 border border-white/5'}`}>
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-emerald-500 to-blue-600 shrink-0 shadow-md shadow-black/20 flex items-center justify-center text-[9px] font-bold">GU</div>
          {!isCollapsed && (
            <div className="min-w-0">
              <div className="text-xs font-bold text-slate-200 truncate">Guest User</div>
              <div className="text-[9px] uppercase tracking-widest text-emerald-500 font-bold">Pro Account</div>
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  );
}
