"use client";

import { useEffect, useRef, useState } from "react";
import { Cloud, HardDrive, Loader2, ChevronDown, Check } from "lucide-react";

interface SttStatus {
  provider: "sarvam" | "whisper" | null;
  model: string | null;
  whisper_loaded: boolean;
}

const POLL_MS = 3000;

export default function SttChip() {
  const [stt, setStt] = useState<SttStatus | null>(null);
  const [switching, setSwitching] = useState<"sarvam" | "whisper" | null>(null);
  const [open, setOpen] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const fetchStt = async () => {
    try {
      const r = await fetch("http://127.0.0.1:8000/stt/status", { cache: "no-store" });
      if (!r.ok) return;
      setStt(await r.json());
    } catch {
      // Non-fatal.
    }
  };

  useEffect(() => {
    fetchStt();
    timerRef.current = setInterval(fetchStt, POLL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = async (provider: "sarvam" | "whisper") => {
    if (switching) return;
    if (stt?.provider === provider) { setOpen(false); return; }
    setSwitching(provider);
    try {
      const r = await fetch("http://127.0.0.1:8000/stt/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });
      if (r.ok) await fetchStt();
    } catch {
      // Silent; System Monitor surfaces error detail if needed.
    } finally {
      setSwitching(null);
      setOpen(false);
    }
  };

  if (!stt?.provider) {
    return (
      <button
        disabled
        className="flex items-center gap-2 pl-3 pr-4 h-11 rounded-full bg-white/5 border border-white/10 text-slate-600 opacity-60 cursor-wait"
        title="STT status…"
      >
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        <span className="text-[10px] font-bold uppercase tracking-wider">STT</span>
      </button>
    );
  }

  const isWhisper = stt.provider === "whisper";
  const activeIcon = isWhisper ? HardDrive : Cloud;
  const ActiveIcon = activeIcon;
  const accent = isWhisper ? "text-amber-400" : "text-indigo-400";
  const accentBg = isWhisper ? "bg-amber-500/20" : "bg-indigo-500/20";
  const label = isWhisper ? "Whisper" : "Sarvam";
  const busy = switching !== null;

  return (
    <div ref={rootRef} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        disabled={busy}
        title={`STT: ${label}${isWhisper ? " (on GPU)" : " (cloud)"}`}
        className={`flex items-center gap-2 pl-3 pr-3 h-11 rounded-full border transition-all ${
          open
            ? "bg-white/10 border-white/20 text-white"
            : "bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-white"
        } disabled:opacity-60 disabled:cursor-wait`}
      >
        <div className={`p-0.5 rounded-md ${accentBg}`}>
          {busy ? (
            <Loader2 className={`w-3 h-3 animate-spin ${accent}`} />
          ) : (
            <ActiveIcon className={`w-3 h-3 ${accent}`} />
          )}
        </div>
        <span className="text-[10px] font-bold uppercase tracking-wider">{label}</span>
        {isWhisper && (
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400" title="GPU resident" />
        )}
        <ChevronDown className={`w-3 h-3 text-slate-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="absolute bottom-full mb-2 left-0 min-w-[220px] bg-slate-950/95 border border-white/10 rounded-xl shadow-2xl backdrop-blur-xl p-1.5 z-50">
          <div className="px-2.5 py-1.5 text-[9px] font-bold uppercase tracking-[0.18em] text-slate-500">
            STT Backend
          </div>
          <SttOption
            icon={<Cloud className="w-3.5 h-3.5 text-indigo-400" />}
            label="Sarvam"
            sublabel="Cloud · keeps GPU free"
            active={stt.provider === "sarvam"}
            loading={switching === "sarvam"}
            onClick={() => pick("sarvam")}
          />
          <SttOption
            icon={<HardDrive className="w-3.5 h-3.5 text-amber-400" />}
            label="Whisper"
            sublabel="Local · loads onto GPU"
            active={stt.provider === "whisper"}
            loading={switching === "whisper"}
            onClick={() => pick("whisper")}
          />
          {stt.model && (
            <div className="mt-1 px-2.5 py-1.5 text-[9px] uppercase tracking-widest text-slate-600 border-t border-white/5">
              model · <span className="text-slate-500 font-mono normal-case tracking-normal">{stt.model}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SttOption({
  icon, label, sublabel, active, loading, onClick,
}: {
  icon: React.ReactNode;
  label: string;
  sublabel: string;
  active: boolean;
  loading: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-all text-left ${
        active
          ? "bg-white/5 text-white"
          : "text-slate-400 hover:bg-white/5 hover:text-white"
      } disabled:cursor-wait`}
    >
      <div className="shrink-0">{loading ? <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" /> : icon}</div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-semibold">{label}</div>
        <div className="text-[9px] text-slate-500 uppercase tracking-wider">{sublabel}</div>
      </div>
      {active && !loading && <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
    </button>
  );
}
