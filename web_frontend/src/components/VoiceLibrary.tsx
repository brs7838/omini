"use client";

import { useState, useEffect, useRef } from "react";
import { Upload, X, Check, Trash2, Mic, Pencil, Plus, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Voice {
  id: string;
  name: string;
  gender: string;
  style: string;
  age?: string;
  about?: string;
  catchphrases?: string;
  file_path: string;
}

interface VoiceLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  activeVoiceId: string;
  onSelect: (voiceId: string) => void;
}

// Gradient palette — cycles through voices
const AVATAR_GRADIENTS = [
  "from-blue-500 to-indigo-700",
  "from-violet-500 to-purple-700",
  "from-pink-500 to-rose-600",
  "from-emerald-500 to-teal-700",
  "from-orange-500 to-amber-600",
  "from-cyan-500 to-blue-600",
  "from-fuchsia-500 to-pink-700",
];

function avatarGradient(index: number, gender: string) {
  if (gender === "female") {
    return ["from-pink-500 to-rose-600", "from-fuchsia-500 to-pink-700", "from-violet-500 to-purple-700"][index % 3];
  }
  return AVATAR_GRADIENTS[index % AVATAR_GRADIENTS.length];
}

// Tiny animated waveform bars shown on active/hover
function Waveform({ active }: { active: boolean }) {
  return (
    <div className={`flex items-end gap-[2px] h-3 transition-opacity duration-300 ${active ? "opacity-100" : "opacity-0"}`}>
      {[0.6, 1, 0.75, 1, 0.5].map((h, i) => (
        <span
          key={i}
          className="w-[2px] rounded-full bg-current"
          style={{
            height: `${h * 100}%`,
            animation: active ? `waveBar 0.8s ease-in-out ${i * 0.12}s infinite alternate` : "none",
          }}
        />
      ))}
    </div>
  );
}

// Individual voice card
function VoiceCard({
  voice,
  index,
  isActive,
  onClick,
  onEdit,
  onDelete,
}: {
  voice: Voice;
  index: number;
  isActive: boolean;
  onClick: () => void;
  onEdit: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const initials = voice.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();

  return (
    <motion.div
      whileHover={{ scale: 1.04, y: -2 }}
      whileTap={{ scale: 0.97 }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => setHovered(false)}
      onClick={onClick}
      className={`relative flex flex-col items-center gap-2.5 p-4 rounded-2xl cursor-pointer border transition-all group select-none ${
        isActive
          ? "border-emerald-500/50 bg-emerald-500/8 shadow-[0_0_20px_rgba(52,211,153,0.12)]"
          : "border-white/6 bg-white/3 hover:border-white/12 hover:bg-white/6"
      }`}
    >
      {/* Edit / Delete — top-right on hover */}
      {voice.id !== "ravi" && (
        <div className={`absolute top-2 right-2 flex gap-1 transition-all duration-200 ${hovered ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
          <button
            onClick={(e) => { e.stopPropagation(); onEdit(); }}
            className="w-6 h-6 flex items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 text-slate-400 hover:text-white transition-colors"
          >
            <Pencil className="w-3 h-3" />
          </button>
          <button
            onClick={onDelete}
            className="w-6 h-6 flex items-center justify-center rounded-lg bg-white/10 hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Avatar */}
      <div className={`relative w-[68px] h-[68px] rounded-full bg-gradient-to-br ${avatarGradient(index, voice.gender)} flex items-center justify-center text-xl font-extrabold text-white shadow-lg`}>
        {initials}
        {isActive && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-emerald-500 border-2 border-black flex items-center justify-center shadow-lg"
          >
            <Check className="w-3 h-3 text-black" strokeWidth={3} />
          </motion.div>
        )}
      </div>

      {/* Name */}
      <div className="text-center w-full">
        <div className={`text-sm font-bold truncate ${isActive ? "text-white" : "text-slate-200"}`}>
          {voice.name}
        </div>
        <div className={`text-[9px] uppercase tracking-[0.15em] font-bold mt-0.5 ${isActive ? "text-emerald-400" : "text-slate-600"}`}>
          {voice.gender}{voice.style && voice.style !== "default" ? ` · ${voice.style}` : ""}
        </div>
      </div>

      {/* Waveform */}
      <div className={isActive ? "text-emerald-400" : "text-slate-500"}>
        <Waveform active={hovered || isActive} />
      </div>

      {/* Catchphrase tooltip */}
      {voice.catchphrases && hovered && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute -bottom-10 left-1/2 -translate-x-1/2 w-max max-w-[180px] px-2.5 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-[10px] text-emerald-400/80 italic text-center z-50 pointer-events-none shadow-xl"
        >
          &quot;{voice.catchphrases.slice(0, 60)}{voice.catchphrases.length > 60 ? "&hellip;" : ""}&quot;
        </motion.div>
      )}
    </motion.div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

const EMPTY_FORM = { name: "", gender: "male", age: "", about: "", catchphrases: "" };

export default function VoiceLibrary({ isOpen, onClose, activeVoiceId, onSelect }: VoiceLibraryProps) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [editVoiceId, setEditVoiceId] = useState<string | null>(null);
  const [formData, setFormData] = useState(EMPTY_FORM);
  const [isGenerating, setIsGenerating] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchVoices = async () => {
    try {
      const res = await fetch("http://localhost:8000/voices");
      setVoices(await res.json());
    } catch { /* silent */ }
  };

  useEffect(() => {
    if (isOpen) {
      const t = setTimeout(() => fetchVoices(), 0);
      return () => clearTimeout(t);
    }
  }, [isOpen]);

  const closeForm = () => {
    setShowForm(false);
    setEditVoiceId(null);
    setFormData(EMPTY_FORM);
    setPendingFile(null);
  };

  const handleFileDrop = (file: File) => {
    setPendingFile(file);
    setShowForm(true);
  };

  const handleUpload = async () => {
    if (!formData.name || !pendingFile) return;
    const fd = new FormData();
    fd.append("file", pendingFile);
    fd.append("name", formData.name);
    fd.append("gender", formData.gender);
    fd.append("age", formData.age);
    fd.append("about", formData.about);
    fd.append("catchphrases", formData.catchphrases);
    fd.append("style", "cloned");

    setIsUploading(true);
    try {
      await fetch("http://localhost:8000/voices/upload", { method: "POST", body: fd });
      closeForm();
      fetchVoices();
    } catch { /* silent */ }
    finally { setIsUploading(false); }
  };

  const handleUpdate = async () => {
    if (!formData.name || !editVoiceId) return;
    setIsUploading(true);
    try {
      await fetch(`http://localhost:8000/voices/${editVoiceId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: formData.name, gender: formData.gender, age: formData.age, about: formData.about, catchphrases: formData.catchphrases }),
      });
      closeForm();
      fetchVoices();
    } catch { /* silent */ }
    finally { setIsUploading(false); }
  };

  const deleteVoice = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`http://localhost:8000/voices/${id}`, { method: "DELETE" });
      if (editVoiceId === id) closeForm();
      fetchVoices();
    } catch { /* silent */ }
  };

  const autoFill = async () => {
    if (!formData.name) return;
    setIsGenerating(true);
    try {
      const res = await fetch("http://localhost:8000/voices/generate-persona", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: formData.name, gender: formData.gender }),
      });
      if (res.ok) {
        const d = await res.json();
        setFormData(prev => ({ ...prev, age: d.age || prev.age, about: d.about || prev.about, catchphrases: d.catchphrases || prev.catchphrases }));
      }
    } catch { /* silent */ }
    finally { setIsGenerating(false); }
  };

  const set = (key: keyof typeof EMPTY_FORM) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setFormData(prev => ({ ...prev, [key]: e.target.value }));

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="absolute inset-0 z-[100] flex items-center justify-center p-4 pointer-events-auto">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/70 backdrop-blur-md z-[60]"
          />

          {/* Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.94, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 12 }}
            transition={{ type: "spring", stiffness: 380, damping: 30 }}
            className="absolute inset-0 z-[100] flex items-center justify-center p-4"
          >
            <div
              className="w-[95%] h-[95%] max-w-[540px] max-h-[85vh] bg-slate-950/90 backdrop-blur-2xl border border-white/8 rounded-3xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] flex flex-col overflow-hidden pointer-events-auto"
              onClick={e => e.stopPropagation()}
            >
              {/* ── Header ── */}
              <div className="flex items-center justify-between px-6 pt-5 pb-4 shrink-0">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-xl bg-emerald-500/15 border border-emerald-500/20 flex items-center justify-center">
                    <Mic className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div>
                    <h2 className="text-sm font-bold text-white leading-none">Voice Library</h2>
                    <p className="text-[10px] text-slate-500 mt-0.5">{voices.length} voice{voices.length !== 1 ? "s" : ""} available</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={() => { setShowForm(p => !p); setEditVoiceId(null); setFormData(EMPTY_FORM); }}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] font-bold transition-all border ${
                      showForm && !editVoiceId
                        ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-300"
                        : "bg-white/5 border-white/10 text-slate-400 hover:text-white hover:bg-white/10"
                    }`}
                  >
                    <Plus className="w-3.5 h-3.5" />
                    Add Voice
                  </motion.button>
                  <button onClick={onClose} className="w-8 h-8 flex items-center justify-center rounded-xl hover:bg-white/8 text-slate-500 hover:text-white transition-colors">
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="h-px bg-white/5 mx-6 shrink-0" />

              {/* ── Voice Grid ── */}
              <div className="flex-1 overflow-y-auto px-6 py-5 scrollbar-hide">
                <div className="grid grid-cols-3 gap-3 mb-5">
                  {voices.map((v, i) => (
                    <VoiceCard
                      key={v.id}
                      voice={v}
                      index={i}
                      isActive={activeVoiceId === v.id}
                      onClick={() => { onSelect(v.id); onClose(); }}
                      onEdit={() => {
                        setEditVoiceId(v.id);
                        setFormData({ name: v.name, gender: v.gender, age: v.age || "", about: v.about || "", catchphrases: v.catchphrases || "" });
                        setShowForm(true);
                      }}
                      onDelete={(e) => deleteVoice(v.id, e)}
                    />
                  ))}
                </div>

                {/* ── Add / Edit Form ── */}
                <AnimatePresence>
                  {showForm && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ type: "spring", stiffness: 300, damping: 28 }}
                      className="overflow-hidden"
                    >
                      <div className="h-px bg-white/5 mb-5" />

                      <div className="text-[10px] font-bold uppercase tracking-[0.25em] text-slate-500 mb-4">
                        {editVoiceId ? "Edit Voice" : "Add New Voice"}
                      </div>

                      {/* Drop zone (only for new upload) */}
                      {!editVoiceId && (
                        <div
                          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                          onDragLeave={() => setDragActive(false)}
                          onDrop={(e) => { e.preventDefault(); setDragActive(false); const f = e.dataTransfer.files[0]; if (f) handleFileDrop(f); }}
                          onClick={() => fileInputRef.current?.click()}
                          className={`flex items-center gap-3 px-4 py-3 rounded-2xl border-2 border-dashed cursor-pointer transition-all mb-4 ${
                            dragActive ? "border-emerald-500 bg-emerald-500/8" :
                            pendingFile ? "border-emerald-500/40 bg-emerald-500/5" :
                            "border-white/10 bg-white/3 hover:border-white/20"
                          }`}
                        >
                          <div className="w-9 h-9 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
                            {pendingFile
                              ? <Check className="w-4 h-4 text-emerald-400" />
                              : <Upload className="w-4 h-4 text-slate-500" />
                            }
                          </div>
                          <div>
                            <div className="text-sm font-semibold text-slate-300">
                              {pendingFile ? pendingFile.name : "Drop a voice clip here"}
                            </div>
                            <div className="text-[10px] text-slate-600 mt-0.5">
                              {pendingFile ? "Ready to upload" : "or click to browse — MP3, WAV, M4A"}
                            </div>
                          </div>
                        </div>
                      )}
                      <input ref={fileInputRef} type="file" accept="audio/*" className="hidden"
                        onChange={(e) => { const f = e.target.files?.[0]; if (f) setPendingFile(f); }} />

                      {/* Form fields */}
                      <div className="space-y-3">
                        <div className="flex gap-2">
                          <input
                            placeholder="Name (e.g. Deepika)"
                            value={formData.name}
                            onChange={set("name")}
                            className="flex-1 bg-black/40 border border-white/8 rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-white placeholder:text-slate-600 transition-colors"
                          />
                          <input
                            placeholder="Age"
                            value={formData.age}
                            onChange={set("age")}
                            className="w-20 bg-black/40 border border-white/8 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-white placeholder:text-slate-600 text-center transition-colors"
                          />
                          <select
                            value={formData.gender}
                            onChange={set("gender")}
                            className="w-28 bg-black/40 border border-white/8 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-white transition-colors"
                          >
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                          </select>
                        </div>

                        <textarea
                          placeholder="About them (e.g. Famous Bollywood actor known for intense roles)"
                          value={formData.about}
                          onChange={set("about")}
                          rows={2}
                          className="w-full bg-black/40 border border-white/8 rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-white placeholder:text-slate-600 resize-none transition-colors"
                        />

                        <div className="relative">
                          <textarea
                            placeholder="Famous dialogues / catchphrases (optional)"
                            value={formData.catchphrases}
                            onChange={set("catchphrases")}
                            rows={2}
                            className="w-full bg-black/40 border border-white/8 rounded-xl px-3.5 py-2.5 pb-9 text-sm focus:outline-none focus:border-emerald-500/50 text-white placeholder:text-slate-600 resize-none transition-colors"
                          />
                          <button
                            onClick={autoFill}
                            disabled={isGenerating || !formData.name}
                            className={`absolute bottom-2.5 right-2.5 flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide transition-all ${
                              isGenerating || !formData.name
                                ? "opacity-40 cursor-not-allowed bg-white/5 text-slate-500"
                                : "bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25"
                            }`}
                          >
                            {isGenerating
                              ? <div className="w-3 h-3 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
                              : <Sparkles className="w-3 h-3" />
                            }
                            {isGenerating ? "Generating…" : "Auto Fill"}
                          </button>
                        </div>

                        <div className="flex gap-2 pt-1">
                          <button
                            onClick={closeForm}
                            className="flex-1 py-2.5 rounded-xl border border-white/8 text-slate-400 text-xs font-bold hover:bg-white/5 hover:text-white transition-all"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={editVoiceId ? handleUpdate : handleUpload}
                            disabled={isUploading || !formData.name || (!editVoiceId && !pendingFile)}
                            className={`flex-2 flex-1 py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 ${
                              isUploading || !formData.name || (!editVoiceId && !pendingFile)
                                ? "bg-white/5 text-slate-600 cursor-not-allowed"
                                : "bg-emerald-500 text-black hover:bg-emerald-400 shadow-lg shadow-emerald-500/20"
                            }`}
                          >
                            {isUploading
                              ? <><div className="w-3.5 h-3.5 border-2 border-black/40 border-t-transparent rounded-full animate-spin" /> Cloning…</>
                              : editVoiceId ? "Save Changes" : "Clone Voice"
                            }
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>

          {/* Waveform keyframe */}
          <style>{`
            @keyframes waveBar {
              from { transform: scaleY(0.4); }
              to   { transform: scaleY(1); }
            }
          `}</style>
        </div>
      )}
    </AnimatePresence>
  );
}
