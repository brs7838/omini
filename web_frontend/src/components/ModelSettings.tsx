"use client";

import { X, Cpu, Check, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";

interface Model {
  id: string;
  name: string;
}

interface ModelSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (id: string, name: string) => void;
  activeModelId: string;
}

export default function ModelSettings({ isOpen, onClose, onSelect, activeModelId }: ModelSettingsProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetch("http://127.0.0.1:8000/models/llm")
        .then(r => r.ok ? r.json() : [])
        .then(data => {
            setModels(data);
            setLoading(false);
        })
        .catch(() => {
            setModels([]);
            setLoading(false);
        });
    }
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="relative w-full max-w-md bg-slate-900/90 border border-white/10 rounded-3xl shadow-2xl p-6 overflow-hidden"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-500/10 rounded-xl text-indigo-400">
                  <Cpu className="w-5 h-5" />
                </div>
                <h2 className="text-xl font-bold text-white">Model Settings</h2>
              </div>
              <button 
                onClick={onClose}
                className="p-2 hover:bg-white/5 rounded-full text-slate-400 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
              {loading ? (
                <div className="py-12 flex flex-col items-center justify-center gap-4 text-slate-500">
                  <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
                  <span className="text-xs font-medium uppercase tracking-widest">Fetching Models...</span>
                </div>
              ) : models.length > 0 ? (
                models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => onSelect(model.id, model.name)}
                    className={`w-full flex items-center justify-between p-4 rounded-2xl border transition-all group ${
                      activeModelId === model.id
                        ? 'bg-indigo-500/10 border-indigo-500/40 text-white'
                        : 'bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/20'
                    }`}
                  >
                    <div className="flex flex-col items-start">
                      <span className="font-bold text-sm tracking-tight">{model.name}</span>
                      <span className="text-[10px] text-slate-500 group-hover:text-slate-400 uppercase font-bold tracking-widest mt-0.5">{model.id}</span>
                    </div>
                    {activeModelId === model.id && (
                      <div className="w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center text-white shadow-lg shadow-indigo-500/20">
                        <Check className="w-4 h-4" />
                      </div>
                    )}
                  </button>
                ))
              ) : (
                <div className="py-8 text-center text-slate-500 text-sm italic">
                  No models found on Ollama server.
                </div>
              )}
            </div>

            <div className="mt-8 p-4 bg-white/5 rounded-2xl border border-white/5">
                <p className="text-[10px] text-slate-500 leading-relaxed">
                   <span className="text-indigo-400 font-bold">INFO:</span> Changing models will notify the backend instantly. Ollama may take a few seconds to swap models in VRAM.
                </p>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
