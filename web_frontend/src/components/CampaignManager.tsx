"use client";

import { useState, useEffect } from "react";
import { X, Check, Trash2, Plus, MessageSquare, User, Volume2, BookOpen, Save, Layers, Sparkles, Wand2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Campaign {
  label: string;
  agent_name: string;
  candidate_name: string;
  candidate_party: string;
  party_symbol: string;
  constituency: string;
  election_date: string;
  booth_lookup_number: string;
  human_callback_number: string;
  call_goal: string;
  call_goal_example: string;
  closing_cta: string;
  system_instruction?: string;
  voice_id?: string;
  objections: { trigger: string; reply: string }[];
  max_sentences_per_turn: number;
}

interface CampaignManagerProps {
  isOpen: boolean;
  onClose: () => void;
  activeCampaignName: string;
  onSelect: (name: string, campaign: Campaign) => void;
}

interface Voice {
  id: string;
  name: string;
}

export default function CampaignManager({ isOpen, onClose, activeCampaignName, onSelect }: CampaignManagerProps) {
  const [campaigns, setCampaigns] = useState<Record<string, Campaign>>({});
  const [activeName, setActiveName] = useState(activeCampaignName);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingName, setEditingName] = useState<string | null>(null);
  const [formData, setFormData] = useState<Campaign | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const cRes = await fetch("http://localhost:8000/campaigns");
      const cData = await cRes.json();
      setCampaigns(cData.campaigns || {});
      const active = cData.active || "default";
      setActiveName(active);

      // Auto-load the active campaign into the editor if we're not already editing something
      if (!editingName && cData.campaigns?.[active]) {
        handleEdit(active, cData.campaigns[active]);
      }

      const vRes = await fetch("http://localhost:8000/voices");
      const vData = await vRes.json();
      setVoices(vData || []);
    } catch { /* silent */ }
    finally { setLoading(false); }
  };

  useEffect(() => {
    if (isOpen) {
      fetchAll();
    }
  }, [isOpen]);

  const handleEdit = (name: string, campaign: Campaign) => {
    setEditingName(name);
    setFormData({ ...campaign });
  };

  const handleCreateNew = () => {
    const name = `campaign_${Date.now()}`;
    const newCampaign: Campaign = {
      label: "New Template",
      agent_name: "Vaani",
      candidate_name: "",
      candidate_party: "",
      party_symbol: "",
      constituency: "",
      election_date: "",
      booth_lookup_number: "",
      human_callback_number: "",
      call_goal: "",
      call_goal_example: "",
      closing_cta: "",
      system_instruction: "",
      voice_id: "ravi",
      objections: [],
      max_sentences_per_turn: 2
    };
    setEditingName(name);
    setFormData(newCampaign);
  };

  const handleLoadDefaultPrompt = async () => {
    try {
      const res = await fetch("http://localhost:8000/campaigns/default-prompt");
      const data = await res.json();
      if (data.prompt) updateField("system_instruction", data.prompt);
    } catch { /* silent */ }
  };

  const handleApplyPreset = (type: "political" | "personal" | "behalf" | "kcr") => {
    if (!formData) return;
    let instruction = "";
    if (type === "political") {
      instruction = `आप {agent_name} हैं — {candidate_party} के एक कार्यकर्ता। आप {constituency} के मतदाताओं से फ़ोन पर बात कर रहे हैं। 
आपका उद्देश्य {candidate_name} जी के लिए समर्थन माँगना है। 
1. केवल हिंदी में बात करें। 
2. अत्यंत छोटे जवाब दें (1-2 वाक्य)। 
3. अंत में कहें: "{closing_cta}"`;
    } else if (type === "personal") {
      instruction = `आप {agent_name} हैं — {candidate_name} के एक व्यक्तिगत डिजिटल सहायक (Personal AI Assistant)। 
आपका लहजा बहुत ही विनम्र, अनौपचारिक और मददगार होना चाहिए। 
1. आप {candidate_name} के निजी कामों और शेड्यूल में मदद करते हैं। 
2. बातचीत बहुत ही छोटी और टू-द-पॉइंट रखें। 
3. "जी", "भैया", "दीदी" जैसे शब्दों का प्रयोग करें।`;
    } else if (type === "behalf") {
      instruction = `आप {agent_name} हैं और आप {candidate_name} जी की ओर से (on behalf of) कॉल कर रहे हैं। 
आपका काम केवल एक संदेश पहुँचाना है: "{call_goal}"। 
1. स्पष्ट करें कि आप उनके प्रतिनिधि के तौर पर बात कर रहे हैं। 
2. यदि मतदाता सवाल पूछे, तो उन्हें {human_callback_number} पर संपर्क करने को कहें। 
3. बहुत ही पेशेवर (Professional) व्यवहार रखें।`;
    } else if (type === "kcr") {
      instruction = `आप एक बड़े नेता (Principal) हैं। आप अपने P.S. 'अमर' से बात कर रहे हैं।

1. शुरुआत: हमेशा 'हां अमर कैसे हो?' से करें।
2. शिष्टाचार: यदि अमर नमस्ते करे, तो नमस्ते कहें।
3. पहचान: यदि अमर पूछे 'कौन बोल रहा है?' या 'आप कौन?', तो कहें 'मैं बोल रहा हूँ।' अपना नाम कभी न लें।
4. लहजा: आप एक 'Humble Gentleman' हैं। आप रूड (rude) नहीं हैं, पर आपकी बातों में एक बड़े व्यक्तित्व की सहज अथॉरिटी (Authority) है।
5. बातचीत का क्रम:
   - पहले दिल्ली ऑफिस का हाल पूछें: 'दिल्ली ऑफिस में क्या चल रहा है?'
   - उसके बाद रवि की बात करें: 'रवि क्या कर रहा है आजकल? सुना है बहुत बढ़िया ऐप बनाया है उसने, काफी काम की लग रही है।'
   - अंत में रवि को बुलाने का कहें: 'बुलाओ उसे, बुलाओ... बात करेंगे उससे। देखेंगे आगे इसको कैसे लेके जाना है।'

6. अत्यंत संक्षिप्त (Concise) रहें। एक बार में 1-2 छोटे वाक्य ही बोलें।`;
    }
    updateField("system_instruction", instruction);
  };

  const handleSave = async () => {
    if (!editingName || !formData) return;
    setIsSaving(true);
    try {
      await fetch(`http://localhost:8000/campaign?name=${editingName}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      await fetchAll();
      setEditingName(null);
      setFormData(null);
    } catch { /* silent */ }
    finally { setIsSaving(false); }
  };

  const handleSwitch = async (name: string) => {
    try {
      const res = await fetch(`http://localhost:8000/campaigns/${name}/switch`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        onSelect(name, data.campaign);
        setActiveName(name);
      }
    } catch { /* silent */ }
  };

  const handleDelete = async (name: string) => {
    if (name === "default") return;
    // Logic to delete from campaigns.json would go here, but for now we just don't show it locally or use a DELETE endpoint if available
    // For now, let's assume we just want to hide it or we can add a delete endpoint later.
  };

  const updateField = (field: keyof Campaign, value: any) => {
    if (!formData) return;
    setFormData({ ...formData, [field]: value });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="absolute inset-0 z-[110] flex items-center justify-center p-4 pointer-events-auto">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/80 backdrop-blur-md z-[60]"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-5xl max-h-[92vh] bg-slate-950/95 border border-white/10 rounded-3xl shadow-[0_0_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden z-[100]"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/5 bg-white/[0.02]">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-indigo-500/10 rounded-2xl text-indigo-400 border border-indigo-500/20">
                  <Layers className="w-6 h-6" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white tracking-tight">AI Persona & Templates</h2>
                  <p className="text-[11px] text-slate-500 mt-0.5 uppercase tracking-wider font-semibold">Master Behavior Control</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <button
                  onClick={handleCreateNew}
                  className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-bold rounded-xl transition-all shadow-lg shadow-indigo-600/20 active:scale-95"
                >
                  <Plus className="w-4 h-4" />
                  New Template
                </button>
                <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full text-slate-500 hover:text-white transition-all">
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            <div className="flex-1 flex overflow-hidden">
              {/* Sidebar: Template List */}
              <div className="w-64 border-r border-white/5 overflow-y-auto p-4 space-y-2 bg-black/40 custom-scrollbar">
                <div className="text-[10px] font-bold uppercase tracking-[0.25em] text-slate-600 mb-4 px-3">Library</div>
                {Object.entries(campaigns).map(([name, c]) => (
                  <div
                    key={name}
                    onClick={() => handleEdit(name, c)}
                    className={`w-full text-left px-4 py-3.5 rounded-2xl border transition-all flex flex-col gap-1.5 group cursor-pointer ${
                      editingName === name
                        ? "bg-indigo-500/15 border-indigo-500/40 shadow-lg shadow-indigo-500/5"
                        : activeName === name
                        ? "bg-white/5 border-emerald-500/30"
                        : "bg-transparent border-transparent hover:bg-white/5"
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-sm font-bold truncate ${editingName === name ? "text-indigo-300" : "text-slate-200 group-hover:text-white"}`}>
                        {c.label || name}
                      </span>
                      {activeName === name && (
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]" />
                      )}
                    </div>
                    <span className="text-[10px] text-slate-600 group-hover:text-slate-500 truncate font-medium">{c.candidate_name || "Generic AI"}</span>
                    {activeName !== name && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleSwitch(name); }}
                        className="mt-1 text-[9px] font-bold text-emerald-500/80 hover:text-emerald-400 uppercase tracking-widest text-left transition-colors relative z-10"
                      >
                        Activate
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Main Content: Editor */}
              <div className="flex-1 overflow-y-auto p-8 custom-scrollbar bg-black/10">
                {editingName && formData ? (
                  <div className="max-w-3xl mx-auto space-y-10 pb-12">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      <div className="space-y-2.5">
                        <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500 ml-1">Template Label</label>
                        <input
                          value={formData.label || ""}
                          onChange={(e) => updateField("label", e.target.value)}
                          placeholder="e.g. Political Campaign v1"
                          className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-5 py-3.5 text-sm focus:outline-none focus:border-indigo-500/40 text-white placeholder:text-slate-700 transition-all"
                        />
                      </div>
                      <div className="space-y-2.5">
                        <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500 ml-1">Voice Persona</label>
                        <div className="relative">
                          <select
                            value={formData.voice_id || "ravi"}
                            onChange={(e) => updateField("voice_id", e.target.value)}
                            className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-5 py-3.5 text-sm focus:outline-none focus:border-indigo-500/40 text-white appearance-none cursor-pointer transition-all"
                          >
                            {voices.map(v => (
                              <option key={v.id} value={v.id} className="bg-slate-900">{v.name}</option>
                            ))}
                          </select>
                          <Volume2 className="absolute right-5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600 pointer-events-none" />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/5 pb-4">
                        <div className="flex items-center gap-3 text-indigo-400">
                          <BookOpen className="w-5 h-5" />
                          <h3 className="text-sm font-bold uppercase tracking-[0.15em]">Instructions & Mindset</h3>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={() => handleApplyPreset("political")}
                            className="text-[9px] px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 hover:bg-indigo-500/20 text-indigo-400 uppercase font-bold transition-all"
                          >
                            Simple Political
                          </button>
                          <button
                            onClick={handleLoadDefaultPrompt}
                            className="text-[9px] px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 hover:bg-blue-500/20 text-blue-400 uppercase font-bold transition-all"
                          >
                            Full Baseline
                          </button>
                          <button
                            onClick={() => handleApplyPreset("personal")}
                            className="text-[9px] px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 text-emerald-400 uppercase font-bold transition-all"
                          >
                            Personal Assistant
                          </button>
                          <button
                            onClick={() => handleApplyPreset("behalf")}
                            className="text-[9px] px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 hover:bg-amber-500/20 text-amber-400 uppercase font-bold transition-all"
                          >
                            Representative
                          </button>
                          <button
                            onClick={() => handleApplyPreset("kcr")}
                            className="text-[9px] px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 text-red-400 uppercase font-bold transition-all shadow-[0_0_15px_rgba(239,68,68,0.1)]"
                          >
                            KCR Style
                          </button>
                        </div>
                      </div>
                      <p className="text-xs text-slate-500 leading-relaxed max-w-2xl">
                        Define how the AI thinks and responds. Use a <strong>Preset</strong> above to auto-fill, or write your own guidelines. Supports dynamic placeholders.
                      </p>
                      <textarea
                        value={formData.system_instruction || ""}
                        onChange={(e) => updateField("system_instruction", e.target.value)}
                        placeholder="Define AI behavior rules here..."
                        rows={10}
                        className="w-full bg-white/[0.02] border border-white/10 rounded-3xl px-6 py-5 text-[13px] focus:outline-none focus:border-indigo-500/40 text-slate-200 font-mono leading-relaxed placeholder:text-slate-800 transition-all custom-scrollbar"
                      />
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-indigo-400 mb-1">
                        <User className="w-4 h-4" />
                        <h3 className="text-sm font-bold uppercase tracking-wider">Identity & Context</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <label className="text-[10px] text-slate-500 ml-1">Principal / Person Name</label>
                          <input
                            value={formData.candidate_name || ""}
                            onChange={(e) => updateField("candidate_name", e.target.value)}
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white"
                          />
                        </div>
                        <div className="space-y-2">
                          <label className="text-[10px] text-slate-500 ml-1">Affiliation / Role</label>
                          <input
                            value={formData.candidate_party || ""}
                            onChange={(e) => updateField("candidate_party", e.target.value)}
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4 pt-4 border-t border-white/5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-indigo-400">
                          <MessageSquare className="w-4 h-4" />
                          <h3 className="text-sm font-bold uppercase tracking-wider">Conversation Rules</h3>
                        </div>
                        <button
                          onClick={() => updateField("objections", [...(formData.objections || []), { trigger: "", reply: "" }])}
                          className="text-[10px] font-bold uppercase text-indigo-400 hover:text-indigo-300"
                        >
                          + Add Rule
                        </button>
                      </div>
                      <div className="space-y-3">
                        {formData.objections?.map((obj, i) => (
                          <div key={i} className="flex gap-3 items-start group">
                            <div className="flex-1 space-y-2">
                              <input
                                placeholder="If user says..."
                                value={obj.trigger || ""}
                                onChange={(e) => {
                                  const next = [...formData.objections];
                                  next[i].trigger = e.target.value;
                                  updateField("objections", next);
                                }}
                                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-white"
                              />
                              <textarea
                                placeholder="AI should say..."
                                value={obj.reply || ""}
                                onChange={(e) => {
                                  const next = [...formData.objections];
                                  next[i].reply = e.target.value;
                                  updateField("objections", next);
                                }}
                                rows={2}
                                className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-white resize-none"
                              />
                            </div>
                            <button
                              onClick={() => updateField("objections", formData.objections.filter((_, idx) => idx !== i))}
                              className="p-2 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center justify-end gap-3 pt-6 border-t border-white/5">
                      <button
                        onClick={() => { setEditingName(null); setFormData(null); }}
                        className="px-6 py-2.5 rounded-xl border border-white/10 text-slate-400 text-sm font-bold hover:bg-white/5 transition-all"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="flex items-center gap-2 px-6 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm font-bold rounded-xl transition-all"
                      >
                        {isSaving ? <Sparkles className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                        Save
                      </button>
                      <button
                        onClick={async () => {
                          await handleSave();
                          if (editingName) handleSwitch(editingName);
                        }}
                        disabled={isSaving}
                        className="flex items-center gap-2 px-8 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-bold rounded-xl transition-all shadow-lg shadow-indigo-600/20 active:scale-95"
                      >
                        <Check className="w-4 h-4" />
                        Save & Activate
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-600 space-y-4">
                    <div className="p-4 rounded-3xl bg-white/5 border border-white/5">
                      <Wand2 className="w-12 h-12 text-slate-700" />
                    </div>
                    <div className="text-center">
                      <h3 className="text-lg font-bold text-slate-400">Template Editor</h3>
                      <p className="text-sm max-w-xs mx-auto leading-relaxed">
                        Select a template from the sidebar or create a new one to start editing your AI's behavior.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
