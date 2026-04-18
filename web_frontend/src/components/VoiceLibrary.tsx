"use client";

import { useState, useEffect, useCallback } from "react";
import { Upload, X, Check, Trash2, Mic, Settings2, Pencil, Wand2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Voice {
  id: string;
  name: string;
  gender: string;
  style: string;
  age?: string;
  about?: string;
  catchphrases: string;
  file_path: string;
}

interface VoiceLibraryProps {
  isOpen: boolean;
  onClose: () => void;
  activeVoiceId: string;
  onSelect: (voiceId: string) => void;
}

export default function VoiceLibrary({ isOpen, onClose, activeVoiceId, onSelect }: VoiceLibraryProps) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSwitching, setIsSwitching] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadData, setUploadData] = useState({ name: "", gender: "male", age: "", about: "", catchphrases: "" });
  const [editVoiceId, setEditVoiceId] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const fetchVoices = async () => {
    try {
      const res = await fetch("http://localhost:8000/voices");
      const data = await res.json();
      setVoices(data);
    } catch (e) {
      console.error("Failed to fetch voices", e);
    }
  };

  useEffect(() => {
    if (isOpen) fetchVoices();
  }, [isOpen]);

  const handleUpload = async (file: File) => {
    if (!uploadData.name) {
      alert("Please enter a name for the voice");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", uploadData.name);
    formData.append("gender", uploadData.gender);
    formData.append("age", uploadData.age);
    formData.append("about", uploadData.about);
    formData.append("catchphrases", uploadData.catchphrases);
    formData.append("style", "cloned");

    setIsUploading(true);
    try {
      await fetch("http://localhost:8000/voices/upload", {
        method: "POST",
        body: formData,
      });
      setUploadData({ name: "", gender: "male", catchphrases: "" });
      fetchVoices();
    } catch (e) {
      console.error("Upload failed", e);
    } finally {
      setIsUploading(false);
    }
  };

  const deleteVoice = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`http://localhost:8000/voices/${id}`, { method: "DELETE" });
      if (editVoiceId === id) {
        setEditVoiceId(null);
        setUploadData({ name: "", gender: "male", age: "", about: "", catchphrases: "" });
      }
      fetchVoices();
    } catch (e) {
        console.error("Delete failed", e);
    }
  };

  const handleUpdate = async () => {
    if (!uploadData.name || !editVoiceId) return;
    setIsUploading(true);
    try {
      await fetch(`http://localhost:8000/voices/${editVoiceId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: uploadData.name,
          gender: uploadData.gender,
          age: uploadData.age,
          about: uploadData.about,
          catchphrases: uploadData.catchphrases,
        }),
      });
      setEditVoiceId(null);
      setUploadData({ name: "", gender: "male", age: "", about: "", catchphrases: "" });
      fetchVoices();
    } catch (e) {
      console.error("Update failed", e);
    } finally {
      setIsUploading(false);
    }
  };

  const autoGeneratePersona = async () => {
    if (!uploadData.name) {
      alert("Please enter a Voice Name first.");
      return;
    }
    setIsGenerating(true);
    try {
      const res = await fetch("http://localhost:8000/voices/generate-persona", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: uploadData.name, gender: uploadData.gender }),
      });
      if (res.ok) {
        const data = await res.json();
        setUploadData(prev => ({ 
          ...prev, 
          age: data.age || prev.age,
          about: data.about || prev.about,
          catchphrases: data.catchphrases || prev.catchphrases 
        }));
      }
    } catch (e) {
      console.error("Auto generate failed", e);
    } finally {
      setIsGenerating(false);
    }
  };

  return (

    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]"
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            className="fixed right-0 top-0 h-full w-[400px] bg-slate-950 border-l border-white/10 z-[70] shadow-2xl flex flex-col p-6 overflow-hidden"
          >
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-emerald-400" />
                <h2 className="text-xl font-bold tracking-tight">Voice Library</h2>
              </div>
              <button 
                onClick={onClose}
                className="p-2 hover:bg-white/5 rounded-full text-slate-400 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Upload Area */}
            <div 
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                const file = e.dataTransfer.files[0];
                if (file) handleUpload(file);
              }}
              className={`relative border-2 border-dashed rounded-3xl p-8 mb-8 transition-all flex flex-col items-center justify-center gap-4 ${
                dragActive ? 'border-emerald-500 bg-emerald-500/10' : 'border-white/10 bg-white/5 hover:bg-white/8'
              }`}
            >
              {editVoiceId ? (
                <div className="text-center w-full">
                  <h3 className="text-sm font-bold text-emerald-400 mb-1">Editing Voice Template</h3>
                  <p className="text-[11px] text-slate-500 uppercase tracking-widest font-semibold mb-2">Update properties below</p>
                </div>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-slate-900 border border-white/10 flex items-center justify-center shadow-lg">
                    <Upload className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div className="text-center">
                    <button className="text-sm font-bold text-white mb-1 hover:text-emerald-400 transition-colors">
                      Upload a voice clip
                    </button>
                    <p className="text-[11px] text-slate-500 uppercase tracking-widest font-semibold">
                      or drag and drop here
                    </p>
                  </div>
                </>
              )}
              
              <div className="w-full mt-4 space-y-3">
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    placeholder="Name (e.g. Deepika)"
                    value={uploadData.name}
                    onChange={(e) => setUploadData({...uploadData, name: e.target.value})}
                    className="w-2/3 bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/50 transition-all text-white placeholder:text-slate-600"
                  />
                  <input 
                    type="text" 
                    placeholder="Age"
                    value={uploadData.age}
                    onChange={(e) => setUploadData({...uploadData, age: e.target.value})}
                    className="w-1/3 bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/50 transition-all text-white placeholder:text-slate-600 text-center"
                  />
                </div>
                <select 
                   value={uploadData.gender}
                   onChange={(e) => setUploadData({...uploadData, gender: e.target.value})}
                   className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/50 transition-all text-white"
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
                <textarea 
                  placeholder="About them (e.g. She is a famous Bollywood actress known for intense roles)"
                  value={uploadData.about}
                  onChange={(e) => setUploadData({...uploadData, about: e.target.value})}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-emerald-500/50 transition-all text-white h-16 resize-none text-[12px] placeholder:text-slate-600"
                />
                <div className="relative w-full">
                  <textarea 
                    placeholder="Famous Dialogues (optional). e.g. Rishte mein toh hum tumhare baap lagte hain"
                    value={uploadData.catchphrases}
                    onChange={(e) => setUploadData({...uploadData, catchphrases: e.target.value})}
                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 pb-8 text-sm focus:outline-none focus:border-emerald-500/50 transition-all text-white h-24 resize-none text-[12px] placeholder:text-slate-600"
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); autoGeneratePersona(); }}
                    disabled={isGenerating || !uploadData.name}
                    className={`absolute bottom-2 right-2 flex items-center gap-1 px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-lg text-[10px] font-bold tracking-wider uppercase transition-colors ${
                      isGenerating || !uploadData.name ? 'opacity-50 cursor-not-allowed' : 'hover:bg-emerald-500/30'
                    }`}
                  >
                    {isGenerating ? <div className="w-3 h-3 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" /> : <Wand2 className="w-3 h-3" />}
                    <span>{isGenerating ? "Generating..." : "Auto Fill"}</span>
                  </button>
                </div>

                {editVoiceId && (
                  <div className="flex gap-2 w-full pt-2">
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditVoiceId(null);
                        setUploadData({ name: "", gender: "male", age: "", about: "", catchphrases: "" });
                      }}
                      className="flex-1 py-2 rounded-xl border border-white/10 text-white text-xs font-bold hover:bg-white/5 transition-colors"
                    >
                      Cancel
                    </button>
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleUpdate(); }}
                      className="flex-1 py-2 rounded-xl bg-emerald-500 text-black text-xs font-bold hover:bg-emerald-400 transition-colors"
                    >
                      Save Changes
                    </button>
                  </div>
                )}
              </div>

              {isUploading && (
                <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm rounded-3xl flex items-center justify-center">
                   <div className="flex flex-col items-center gap-3">
                      <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                      <span className="text-xs font-bold text-emerald-500 uppercase tracking-widest">Cloning Voice...</span>
                   </div>
                </div>
              )}
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto pr-2 space-y-3 scrollbar-hide">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em] mb-4 ml-2">
                Saved Templates
              </div>
              {voices.map((v) => (
                <div
                  key={v.id}
                  onClick={() => {
                    if (isSwitching) return;
                    setIsSwitching(true);
                    onSelect(v.id);
                    setTimeout(() => setIsSwitching(false), 2000);
                  }}
                  className={`w-full group relative flex items-center gap-4 p-4 rounded-3xl transition-all border ${
                    isSwitching ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  } ${
                    activeVoiceId === v.id 
                      ? 'bg-emerald-500/10 border-emerald-500/40 shadow-lg shadow-emerald-500/5' 
                      : 'bg-white/5 border-white/5 hover:border-white/10'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                    activeVoiceId === v.id ? 'bg-emerald-500 text-black' : 'bg-slate-900 border border-white/10 text-slate-400'
                  }`}>
                    {activeVoiceId === v.id ? <Check className="w-5 h-5" /> : <Mic className="w-4 h-4" />}
                  </div>
                  <div className="flex-1 text-left min-w-0">
                    <div className="font-bold text-sm text-white truncate">{v.name}</div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-bold mb-1">{v.gender} • {v.style}</div>
                    {v.catchphrases && (
                       <div className="text-[10px] text-emerald-500/60 truncate italic">"{v.catchphrases}"</div>
                    )}
                  </div>
                  {v.id !== 'ravi' && (
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all z-10">
                      <button 
                         onClick={(e) => {
                             e.stopPropagation();
                             setEditVoiceId(v.id);
                             setUploadData({ name: v.name, gender: v.gender, age: v.age || "", about: v.about || "", catchphrases: v.catchphrases || "" });
                         }}
                         className="p-2 hover:text-emerald-500 transition-colors"
                      >
                        <Pencil className="w-4 h-4" />
                      </button>
                      <button 
                         onClick={(e) => deleteVoice(v.id, e)}
                         className="p-2 hover:text-red-500 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
