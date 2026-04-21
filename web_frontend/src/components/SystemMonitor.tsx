"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  Cpu,
  X,
  Mic,
  Volume2,
  Sparkles,
  Thermometer,
  Zap,
  PhoneCall,
  Trash2,
  Loader2,
  Server,
  Cloud,
  AlertTriangle,
  HardDrive,
} from "lucide-react";

interface GpuInfo {
  index: number;
  name: string;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
  power_w: number;
  power_limit_w: number;
}

interface ResidentModel { name: string; size_mb: number; vram_mb: number; }

interface ModelInfo {
  name: string;
  device: string;
  loaded: boolean;
  role: string;
  provider?: string;
  url?: string;
  resident_models?: ResidentModel[];
}

interface SystemStatus {
  ts: number;
  gpus: GpuInfo[];
  models: { stt: ModelInfo; tts: ModelInfo; llm: ModelInfo };
  active_calls: number;
}

interface SttStatus {
  provider: "sarvam" | "whisper" | null;
  model: string | null;
  whisper_loaded: boolean;
}

interface SystemMonitorProps {
  isOpen: boolean;
  onClose: () => void;
}

const POLL_MS = 1500;

function utilColor(pct: number): string {
  if (pct >= 85) return "bg-red-500/80";
  if (pct >= 60) return "bg-orange-500/80";
  if (pct >= 25) return "bg-emerald-500/80";
  return "bg-blue-500/60";
}

function memBar(used: number, total: number): { pct: number; color: string } {
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
  const color =
    pct >= 90 ? "bg-red-500/80" :
    pct >= 70 ? "bg-orange-500/80" :
    "bg-indigo-500/70";
  return { pct, color };
}

function tempColor(c: number): string {
  if (c >= 80) return "text-red-400";
  if (c >= 70) return "text-orange-400";
  return "text-slate-400";
}

function fmtMb(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb} MB`;
}

export default function SystemMonitor({ isOpen, onClose }: SystemMonitorProps) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unloading, setUnloading] = useState(false);
  const [unloadMsg, setUnloadMsg] = useState<string | null>(null);
  const [stt, setStt] = useState<SttStatus | null>(null);
  const [sttSwitching, setSttSwitching] = useState<"sarvam" | "whisper" | null>(null);
  const [sttErr, setSttErr] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Provider switching previously lived here but was removed — pills at the
  // top of this panel are now read-only indicators. Actual switching happens
  // in ModelSettings (pick a model row) which sends {provider, model} as one
  // atomic POST so we can't accidentally revert to the default model.

  const fetchStt = async () => {
    try {
      const r = await fetch("http://127.0.0.1:8000/stt/status", { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: SttStatus = await r.json();
      setStt(data);
    } catch {
      // Non-fatal: panel just won't render the toggle until it comes back.
    }
  };

  const switchStt = async (provider: "sarvam" | "whisper") => {
    if (sttSwitching) return;
    if (stt?.provider === provider) return; // no-op guard
    setSttSwitching(provider);
    setSttErr(null);
    try {
      const r = await fetch("http://127.0.0.1:8000/stt/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => "");
        throw new Error(txt || `HTTP ${r.status}`);
      }
      await fetchStt();
    } catch (e) {
      setSttErr((e as Error).message || "switch failed");
      setTimeout(() => setSttErr(null), 4000);
    } finally {
      setSttSwitching(null);
    }
  };

  const forceUnloadOllama = async () => {
    if (unloading) return;
    setUnloading(true);
    setUnloadMsg(null);
    try {
      const r = await fetch("http://127.0.0.1:8000/ollama/unload", { method: "POST" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const n = (data.unloaded || []).length;
      setUnloadMsg(n > 0 ? `freed ${n} model${n > 1 ? "s" : ""}` : "nothing to free");
      setTimeout(() => setUnloadMsg(null), 3500);
    } catch (e) {
      setUnloadMsg((e as Error).message || "failed");
      setTimeout(() => setUnloadMsg(null), 3500);
    } finally {
      setUnloading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }

    let cancelled = false;
    const tick = async () => {
      try {
        const r = await fetch("http://127.0.0.1:8000/system/status", { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (!cancelled) { setStatus(data); setError(null); }
      } catch (e) {
        if (!cancelled) setError((e as Error).message || "offline");
      }
    };
    const sttTick = async () => { if (!cancelled) await fetchStt(); };
    tick();
    sttTick();
    timerRef.current = setInterval(() => { tick(); sttTick(); }, POLL_MS);

    return () => {
      cancelled = true;
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 40 }}
          transition={{ type: "spring", damping: 24, stiffness: 260 }}
          className="fixed top-6 right-6 w-[400px] max-h-[calc(100vh-3rem)] bg-slate-950/90 border border-white/8 rounded-2xl shadow-2xl backdrop-blur-2xl z-[80] overflow-hidden flex flex-col"
        >
          {/* Header */}
          <div className="shrink-0 flex items-center justify-between px-5 py-4 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="p-1.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <div className="text-sm font-semibold text-white tracking-tight">System Monitor</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${error ? "bg-red-500" : "bg-emerald-500 animate-pulse"}`} />
                  {error ? "offline" : "live"}
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-slate-500 hover:bg-white/10 hover:text-white transition-all"
              title="Close"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Body */}
          <div className="overflow-y-auto px-5 py-4 space-y-5">
            {/* Active provider — read-only display. Previously this had click-
                to-switch pills, which were getting triggered accidentally and
                wiping the persisted model. Switching is now done only from
                Model Settings → pick a model row. */}
            {status && (
              <section>
                <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 mb-2">Active Provider</div>
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-white/[0.03] border border-white/5">
                  <InlineProviderPill
                    active={status.models.llm.provider === "ollama"}
                    loading={false}
                    onClick={() => { /* read-only */ }}
                    icon={<Server className="w-3 h-3" />}
                    label="Ollama"
                    accent="emerald"
                  />
                  <InlineProviderPill
                    active={status.models.llm.provider === "sarvam"}
                    loading={false}
                    onClick={() => { /* read-only */ }}
                    icon={<Cloud className="w-3 h-3" />}
                    label="Sarvam"
                    accent="indigo"
                  />
                </div>
                <div className="mt-1.5 text-[9px] uppercase tracking-widest text-slate-600">
                  Change in Model Settings · pick a model
                </div>
              </section>
            )}

            {/* STT backend — clickable toggle. Sarvam keeps the GPU free for
                OmniVoice (the pitch-drift workaround); Whisper loads locally
                only on demand and unloads when you switch back. */}
            <section>
              <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">STT Backend</div>
                {!stt ? (
                  <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-slate-500/15 text-slate-500">
                    loading…
                  </span>
                ) : stt.provider === "whisper" ? (
                    <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-amber-500/15 text-amber-300 flex items-center gap-1">
                      <HardDrive className="w-2.5 h-2.5" /> GPU
                    </span>
                  ) : stt.whisper_loaded ? (
                    <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-slate-500/15 text-slate-400">
                      whisper still resident
                    </span>
                ) : (
                  <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-emerald-500/10 text-emerald-400/80">
                    gpu free
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-white/[0.03] border border-white/5">
                <SttPill
                  active={stt?.provider === "sarvam"}
                  loading={sttSwitching === "sarvam"}
                  disabled={sttSwitching !== null || !stt}
                  onClick={() => switchStt("sarvam")}
                  icon={<Cloud className="w-3 h-3" />}
                  label="Sarvam"
                  accent="indigo"
                />
                <SttPill
                  active={stt?.provider === "whisper"}
                  loading={sttSwitching === "whisper"}
                  disabled={sttSwitching !== null || !stt}
                  onClick={() => switchStt("whisper")}
                  icon={<HardDrive className="w-3 h-3" />}
                  label="Whisper"
                  accent="amber"
                />
              </div>
              <div className="mt-1.5 text-[9px] uppercase tracking-widest text-slate-600">
                {stt?.model ? <>model · <span className="text-slate-500 font-mono normal-case tracking-normal">{stt.model}</span></> : null}
              </div>
              {sttErr && (
                <div className="mt-1.5 text-[10px] text-red-400/80 flex items-center gap-1">
                  <AlertTriangle className="w-2.5 h-2.5" /> {sttErr}
                </div>
              )}
            </section>

            {/* Models */}
            <section>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 mb-2">Models</div>
              <div className="space-y-2">
                {status && (
                  <>
                    <ModelRow icon={<Mic className="w-3.5 h-3.5" />} accent="emerald" data={status.models.stt} />
                    <ModelRow icon={<Volume2 className="w-3.5 h-3.5" />} accent="orange" data={status.models.tts} />
                    <ModelRow icon={<Sparkles className="w-3.5 h-3.5" />} accent="indigo" data={status.models.llm} />
                  </>
                )}
                {!status && !error && (
                  <div className="text-xs text-slate-600 italic px-1">loading…</div>
                )}
                {error && !status && (
                  <div className="text-xs text-red-400/80 px-1">backend unreachable</div>
                )}
              </div>
            </section>

            {/* Ollama resident */}
            {status?.models.llm.resident_models && status.models.llm.resident_models.length > 0 && (
              <section>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Ollama Resident</div>
                  <button
                    onClick={forceUnloadOllama}
                    disabled={unloading}
                    className="flex items-center gap-1 px-2 py-1 rounded-md bg-red-500/10 border border-red-500/20 text-red-400 text-[9px] font-bold uppercase tracking-wider hover:bg-red-500/15 hover:border-red-500/30 transition-all disabled:opacity-60 disabled:cursor-wait"
                    title="Force-evict Ollama models from GPU"
                  >
                    {unloading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                    <span>{unloading ? "freeing…" : "free vram"}</span>
                  </button>
                </div>
                {status.models.llm.provider === "ollama" && (
                  <div className="mb-2 flex items-start gap-1.5 px-2.5 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/20 text-orange-300 text-[10px] leading-relaxed">
                    <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                    <span>Ollama is still the active provider — your next turn will reload this model. Switch to Sarvam above to keep VRAM free.</span>
                  </div>
                )}
                <div className="space-y-1.5">
                  {status.models.llm.resident_models.map((m) => (
                    <div key={m.name} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-white/5">
                      <span className="text-xs font-mono text-slate-300 truncate">{m.name}</span>
                      <span className="text-[10px] text-slate-500 shrink-0 ml-2">{fmtMb(m.vram_mb)} vram</span>
                    </div>
                  ))}
                </div>
                {unloadMsg && (
                  <div className="mt-1.5 text-[10px] text-emerald-400/80 flex items-center gap-1">
                    <Zap className="w-2.5 h-2.5" /> {unloadMsg}
                  </div>
                )}
              </section>
            )}

            {/* Force-unload shown even when Ollama reports nothing resident,
                 so the user has a recourse if the provider switch didn't
                 evict for some reason. */}
            {status && (!status.models.llm.resident_models || status.models.llm.resident_models.length === 0) && (
              <section>
                <button
                  onClick={forceUnloadOllama}
                  disabled={unloading}
                  className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/5 text-slate-500 text-[10px] font-bold uppercase tracking-wider hover:bg-red-500/10 hover:border-red-500/20 hover:text-red-400 transition-all disabled:opacity-60 disabled:cursor-wait"
                  title="Force-evict any Ollama models still in VRAM"
                >
                  {unloading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
                  <span>{unloading ? "freeing vram…" : "force unload ollama"}</span>
                </button>
                {unloadMsg && (
                  <div className="mt-1.5 text-[10px] text-center text-emerald-400/80">{unloadMsg}</div>
                )}
              </section>
            )}

            {/* GPUs */}
            <section>
              <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500 mb-2">
                GPUs {status && status.gpus.length > 0 && <span className="text-slate-600 font-normal normal-case tracking-normal">· {status.gpus.length}</span>}
              </div>
              <div className="space-y-3">
                {status && status.gpus.length > 0 ? (
                  status.gpus.map((g) => <GpuCard key={g.index} gpu={g} />)
                ) : status && status.gpus.length === 0 ? (
                  <div className="text-xs text-slate-600 italic px-1">no NVIDIA GPU detected</div>
                ) : null}
              </div>
            </section>

            {/* Footer */}
            {status && (
              <section className="pt-2 border-t border-white/5 flex items-center justify-between text-[10px] text-slate-600">
                <span className="flex items-center gap-1.5">
                  <PhoneCall className="w-3 h-3" />
                  {status.active_calls} active call{status.active_calls === 1 ? "" : "s"}
                </span>
                <span>updated {new Date(status.ts * 1000).toLocaleTimeString()}</span>
              </section>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function InlineProviderPill({
  active, loading, icon, label, accent,
}: {
  active: boolean;
  loading: boolean;
  onClick: () => void; // accepted for call-site compatibility; ignored (read-only display)
  icon: React.ReactNode;
  label: string;
  accent: "emerald" | "indigo";
}) {
  const activeCls =
    accent === "emerald"
      ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
      : "bg-indigo-500/15 border-indigo-500/30 text-indigo-300";
  return (
    <div
      className={`relative flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-all text-[11px] font-bold cursor-default ${
        active
          ? activeCls
          : "bg-transparent border-transparent text-slate-500"
      }`}
    >
      {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : icon}
      <span>{label}</span>
      {active && <span className="w-1 h-1 rounded-full bg-current ml-0.5" />}
    </div>
  );
}

function SttPill({
  active, loading, disabled, onClick, icon, label, accent,
}: {
  active: boolean;
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  accent: "indigo" | "amber";
}) {
  const activeCls =
    accent === "indigo"
      ? "bg-indigo-500/15 border-indigo-500/30 text-indigo-300"
      : "bg-amber-500/15 border-amber-500/30 text-amber-300";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`relative flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-all text-[11px] font-bold ${
        active
          ? activeCls
          : "bg-transparent border-transparent text-slate-500 hover:bg-white/5 hover:text-slate-300"
      } ${disabled && !loading ? "opacity-40 cursor-not-allowed" : ""} ${loading ? "cursor-wait" : "cursor-pointer"}`}
    >
      {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : icon}
      <span>{label}</span>
      {active && <span className="w-1 h-1 rounded-full bg-current ml-0.5" />}
    </button>
  );
}

function ModelRow({
  icon,
  accent,
  data,
}: {
  icon: React.ReactNode;
  accent: "emerald" | "orange" | "indigo";
  data: ModelInfo;
}) {
  const accentMap = {
    emerald: "bg-emerald-500/10 text-emerald-400",
    orange:  "bg-orange-500/10 text-orange-400",
    indigo:  "bg-indigo-500/10 text-indigo-400",
  };
  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/[0.03] border border-white/5">
      <div className={`p-1.5 rounded-lg shrink-0 ${accentMap[accent]}`}>{icon}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-white truncate">{data.name}</span>
          <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-md shrink-0 ${
            data.loaded
              ? "bg-emerald-500/15 text-emerald-400"
              : "bg-slate-500/15 text-slate-500"
          }`}>
            {data.loaded ? "loaded" : "idle"}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-0.5 text-[10px] text-slate-500">
          <span>{data.role}</span>
          <span className="text-slate-700">·</span>
          <span className="font-mono">{data.device}</span>
          {data.provider && <>
            <span className="text-slate-700">·</span>
            <span className="uppercase tracking-wider">{data.provider}</span>
          </>}
        </div>
      </div>
    </div>
  );
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const mem = memBar(gpu.mem_used_mb, gpu.mem_total_mb);
  return (
    <div className="px-3 py-3 rounded-xl bg-white/[0.03] border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <Cpu className="w-3.5 h-3.5 text-slate-500 shrink-0" />
          <span className="text-xs font-semibold text-white truncate">{gpu.name}</span>
        </div>
        <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500 shrink-0 ml-2">cuda:{gpu.index}</span>
      </div>

      {/* Util */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-[10px] mb-1">
          <span className="text-slate-500 uppercase tracking-wider font-semibold">Util</span>
          <span className="text-slate-300 font-mono tabular-nums">{gpu.util_pct}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${utilColor(gpu.util_pct)}`}
            animate={{ width: `${gpu.util_pct}%` }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* VRAM */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-[10px] mb-1">
          <span className="text-slate-500 uppercase tracking-wider font-semibold">VRAM</span>
          <span className="text-slate-300 font-mono tabular-nums">
            {fmtMb(gpu.mem_used_mb)} / {fmtMb(gpu.mem_total_mb)}
          </span>
        </div>
        <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${mem.color}`}
            animate={{ width: `${mem.pct}%` }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Temp + Power */}
      <div className="flex items-center justify-between text-[10px] pt-1.5 border-t border-white/5">
        <span className={`flex items-center gap-1 ${tempColor(gpu.temp_c)}`}>
          <Thermometer className="w-3 h-3" />
          <span className="font-mono tabular-nums">{gpu.temp_c}°C</span>
        </span>
        {gpu.power_w > 0 && (
          <span className="flex items-center gap-1 text-slate-500">
            <Zap className="w-3 h-3" />
            <span className="font-mono tabular-nums">
              {Math.round(gpu.power_w)}W
              {gpu.power_limit_w > 0 && <span className="text-slate-700"> / {Math.round(gpu.power_limit_w)}W</span>}
            </span>
          </span>
        )}
      </div>
    </div>
  );
}
