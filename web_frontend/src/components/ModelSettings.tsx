"use client";

import { X, Cpu, Check, Loader2, Server, Cloud, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useCallback } from "react";

interface Model {
  id: string;
  name: string;
}

type ProviderName = "ollama" | "sarvam" | "minimax";

interface ProviderState {
  provider: ProviderName;
  model: string;
}

interface ModelSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (id: string, name: string) => void;
  activeModelId: string;
}

export default function ModelSettings({ isOpen, onClose, onSelect, activeModelId }: ModelSettingsProps) {
  // `providerState.provider` is the *actual* backend-active provider (from
  // /system/status). `viewedProvider` is which tab the user is browsing in
  // this panel — decoupled so clicking a tab doesn't flip the backend or
  // evict VRAM. The switch only happens when the user explicitly picks a
  // model row, which POSTs {provider, model} as a single atomic call.
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [providerState, setProviderState] = useState<ProviderState | null>(null);
  const [viewedProvider, setViewedProvider] = useState<ProviderName | null>(null);
  const [switching, setSwitching] = useState<string | null>(null); // model id being switched
  const [switchError, setSwitchError] = useState<string | null>(null);
  const [lastSwitchInfo, setLastSwitchInfo] = useState<string | null>(null);

  const fetchModelsFor = useCallback(async (provider: ProviderName): Promise<Model[]> => {
    // /models/llm returns the list for whichever provider the backend is
    // currently on. To preview the *other* provider's list without actually
    // switching, we hardcode both rosters here — they're small and stable.
    if (provider === "sarvam") {
      return [
        { id: "sarvam-m",    name: "Sarvam M (24B) - Average" },
        { id: "sarvam-30b",  name: "Sarvam 30B - High IQ" },
        { id: "sarvam-105b", name: "Sarvam 105B - Flagship" },
      ];
    }
    if (provider === "minimax") {
      return [
        { id: "MiniMax-M2.5",             name: "MiniMax M2.5 - Thinking" },
        { id: "MiniMax-M2.7",             name: "MiniMax M2.7 - Thinking" },
        { id: "MiniMax-M2.5-non-thinking", name: "MiniMax M2.5 - Non-Thinking" },
        { id: "MiniMax-M2.7-non-thinking", name: "MiniMax M2.7 - Non-Thinking" },
      ];
    }
    return [
      { id: "gemma3:4b",                  name: "Gemma 3 (4B) - Default" },
      { id: "qwen2:7b",                   name: "Qwen 2 (7B) - High IQ" },
      { id: "gemma4:e4b",                 name: "Gemma 4 (Large)" },
      { id: "mashriram/sarvam-m:latest",  name: "Sarvam (M-Hinglish)" },
    ];
  }, []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      const sRes = await fetch("http://127.0.0.1:8000/system/status");
      let activeProv: ProviderName = "ollama";
      let activeModel = "";
      if (sRes.ok) {
        const sData = await sRes.json();
        if (sData?.models?.llm) {
          activeProv = (sData.models.llm.provider || "ollama") as ProviderName;
          activeModel = sData.models.llm.name;
          setProviderState({ provider: activeProv, model: activeModel });
        }
      }
      const viewTarget = viewedProvider ?? activeProv;
      setViewedProvider(viewTarget);
      setModels(await fetchModelsFor(viewTarget));
    } catch {
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, [viewedProvider, fetchModelsFor]);

  useEffect(() => {
    if (isOpen) {
      const t = setTimeout(() => refreshAll(), 0);
      return () => clearTimeout(t);
    }
  }, [isOpen, refreshAll]);

  // Tab click = filter-only; no backend call, no switch, no VRAM reload.
  const handleViewProvider = async (target: ProviderName) => {
    if (viewedProvider === target) return;
    setViewedProvider(target);
    setSwitchError(null);
    setLastSwitchInfo(null);
    setModels(await fetchModelsFor(target));
  };

  // Explicit switch: user picked a specific model from the list. Sends both
  // provider AND model so the backend doesn't default to gemma3:4b on a bare
  // ollama request.
  const handleSwitchToModel = async (modelId: string, modelName: string) => {
    if (!viewedProvider || switching) return;
    setSwitching(modelId);
    setSwitchError(null);
    setLastSwitchInfo(null);
    try {
      const r = await fetch("http://127.0.0.1:8000/provider/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: viewedProvider, model: modelId }),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(detail.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      const bits: string[] = [];
      if (data.noop) bits.push("already active");
      if (data.unloaded?.length) bits.push(`freed ${data.unloaded.length} Ollama model${data.unloaded.length > 1 ? "s" : ""}`);
      if (data.warmed === true) bits.push(`warmed ${data.model}`);
      if (data.warmed === false) bits.push("warmup failed");
      setLastSwitchInfo(bits.join(" · ") || "switched");
      // Now tell the parent + useVoice hook; this stays local to the UI,
      // backend already switched atomically above.
      onSelect(modelId, modelName);
      await refreshAll();
    } catch (e) {
      setSwitchError((e as Error).message || "switch failed");
    } finally {
      setSwitching(null);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center p-4 pointer-events-auto">
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
            className="relative w-full max-w-lg bg-slate-900/90 border border-white/10 rounded-3xl shadow-2xl p-6 overflow-hidden max-h-full flex flex-col"
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

            {/* Provider tabs — filter-only. Switching tabs shows the other
                provider's model list but does NOT change the backend. The
                switch fires only when a specific model row is clicked below. */}
            <div className="mb-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Provider</span>
                {providerState && (
                  <span className="text-[9px] uppercase tracking-widest text-emerald-400/80">
                    Active: {providerState.provider} · {providerState.model}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2 p-1 rounded-2xl bg-white/[0.03] border border-white/5">
                <ProviderPill
                  active={viewedProvider === "ollama"}
                  isBackendActive={providerState?.provider === "ollama"}
                  loading={false}
                  onClick={() => handleViewProvider("ollama")}
                  icon={<Server className="w-3.5 h-3.5" />}
                  label="Ollama"
                  sub="Local · RTX 3060"
                  accent="emerald"
                />
                <ProviderPill
                  active={viewedProvider === "sarvam"}
                  isBackendActive={providerState?.provider === "sarvam"}
                  loading={false}
                  onClick={() => handleViewProvider("sarvam")}
                  icon={<Cloud className="w-3.5 h-3.5" />}
                  label="Sarvam"
                  sub="Cloud API"
                  accent="indigo"
                />
                <ProviderPill
                  active={viewedProvider === "minimax"}
                  isBackendActive={providerState?.provider === "minimax"}
                  loading={false}
                  onClick={() => handleViewProvider("minimax")}
                  icon={<Zap className="w-3.5 h-3.5" />}
                  label="MiniMax"
                  sub="High Speed"
                  accent="indigo"
                />
              </div>

              {(lastSwitchInfo || switchError) && (
                <div className={`mt-2 text-[10px] flex items-center gap-1.5 ${switchError ? "text-red-400" : "text-emerald-400/80"}`}>
                  <Zap className="w-3 h-3" />
                  <span>{switchError || lastSwitchInfo}</span>
                </div>
              )}
            </div>

            {/* Model list */}
            <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 mb-2">
              Models {viewedProvider && <span className="text-slate-600 font-normal normal-case tracking-normal">· {viewedProvider}</span>}
              <span className="text-slate-600 font-normal normal-case tracking-normal ml-1">· click to switch</span>
            </div>
            <div className="space-y-2 max-h-[45vh] overflow-y-auto pr-2 custom-scrollbar">
              {loading ? (
                <div className="py-10 flex flex-col items-center justify-center gap-3 text-slate-500">
                  <Loader2 className="w-6 h-6 animate-spin text-indigo-400" />
                  <span className="text-[10px] font-medium uppercase tracking-widest">Fetching…</span>
                </div>
              ) : models.length > 0 ? (
                models.map((model) => (
                  <button
                    key={model.id}
                    disabled={switching !== null}
                    onClick={() => handleSwitchToModel(model.id, model.name)}
                    className={`w-full flex items-center justify-between p-3.5 rounded-2xl border transition-all group ${
                      activeModelId === model.id && providerState?.provider === viewedProvider
                        ? "bg-indigo-500/10 border-indigo-500/40 text-white"
                        : "bg-white/5 border-white/5 text-slate-400 hover:bg-white/10 hover:border-white/20"
                    } ${switching !== null && switching !== model.id ? "opacity-50" : ""}`}
                  >
                    <div className="flex flex-col items-start min-w-0">
                      <span className="font-bold text-sm tracking-tight truncate">{model.name}</span>
                      <span className="text-[10px] text-slate-500 group-hover:text-slate-400 uppercase font-bold tracking-widest mt-0.5 truncate">{model.id}</span>
                    </div>
                    {switching === model.id ? (
                      <Loader2 className="w-5 h-5 animate-spin text-indigo-400 shrink-0 ml-2" />
                    ) : activeModelId === model.id && providerState?.provider === viewedProvider ? (
                      <div className="w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center text-white shadow-lg shadow-indigo-500/20 shrink-0 ml-2">
                        <Check className="w-4 h-4" />
                      </div>
                    ) : null}
                  </button>
                ))
              ) : (
                <div className="py-8 text-center text-slate-500 text-sm italic">
                  No models available for this provider.
                </div>
              )}
            </div>

            <div className="mt-5 p-3.5 bg-white/5 rounded-2xl border border-white/5">
              <p className="text-[10px] text-slate-500 leading-relaxed">
                <span className="text-indigo-400 font-bold">INFO:</span> Switching to Sarvam evicts Ollama from GPU (frees VRAM). Switching back warms the selected model before your next turn.
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

function ProviderPill({
  active, isBackendActive, loading, onClick, icon, label, sub, accent,
}: {
  active: boolean;                  // this tab is currently being viewed
  isBackendActive: boolean;         // backend is actually running this provider
  loading: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  sub: string;
  accent: "emerald" | "indigo";
}) {
  const activeCls =
    accent === "emerald"
      ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
      : "bg-indigo-500/15 border-indigo-500/30 text-indigo-300";
  return (
    <button
      disabled={loading}
      onClick={onClick}
      className={`relative flex items-center gap-2.5 px-3 py-2.5 rounded-xl border transition-all ${
        active
          ? activeCls
          : "bg-transparent border-transparent text-slate-400 hover:bg-white/5 hover:text-white"
      } ${loading ? "opacity-70 cursor-wait" : ""}`}
    >
      <div className="p-1.5 rounded-lg bg-white/5">
        {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : icon}
      </div>
      <div className="flex flex-col items-start min-w-0">
        <span className="text-xs font-bold tracking-tight">{label}</span>
        <span className="text-[9px] uppercase tracking-wider text-slate-500 truncate">{sub}</span>
      </div>
      {isBackendActive && (
        <span
          title="Backend is running this provider"
          className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]"
        />
      )}
    </button>
  );
}
