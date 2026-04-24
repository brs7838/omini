"""Campaign config loader + prompt builder for the political-call agent.

Why a separate module: the LLM system prompt is the single biggest lever on
call quality, and we need to edit it per campaign (different candidate,
constituency, goal, objection playbook) without restarting the backend or
editing `engine.py`. `campaigns.json` at project root is the source of truth;
this module reads it, validates minimally, and renders the final system
prompt string that `engine.llm_worker` splices into each request.

The prompt follows the 6-section structure researched for Indian political
outbound calls:
    1. IDENTITY & DISCLOSURE
    2. CAMPAIGN CONTEXT
    3. CALL PURPOSE (one goal)
    4. CONVERSATION STYLE
    5. OBJECTION PLAYBOOK
    6. HARD CONSTRAINTS

Placeholders in user-supplied strings (e.g. "{candidate_name}" inside a
closing CTA) are resolved against the campaign's own fields so the user can
write natural Hindi without concatenation boilerplate.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


# Project-root `campaigns.json`. Matches the `calls_history.json` convention.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "campaigns.json",
)

DEFAULT_POLITICAL_PROMPT = """## 1. पहचान (IDENTITY)
आप {agent_name} हैं — {candidate_party} के एक कार्यकर्ता। आप {constituency} के मतदाताओं से फ़ोन पर बात कर रहे हैं। 
{disclose_ai_line}

## 2. अभियान सन्दर्भ (CONTEXT)
- प्रत्याशी: {candidate_name} ({candidate_party})
- चुनाव चिह्न: {party_symbol}
- मतदान तिथि: {election_date}
- बूथ/सूची हेल्पलाइन: {booth_lookup_number}
- मानव callback: {human_callback_number}

## 3. इस कॉल का उद्देश्य (केवल एक)
{call_goal}
{call_goal_example_block}

## 4. बातचीत की शैली (STYLE)
- केवल देवनागरी हिंदी में बोलें। अंग्रेज़ी केवल तब जब मतदाता अंग्रेज़ी में बोले।
- अत्यंत छोटा जवाब दें: हर जवाब 1 या अधिकतम {max_sentences_per_turn} छोटे वाक्यों का ही होना चाहिए। यह एक फ़ोन कॉल है, लंबा भाषण या मोनोलॉग (monologue) बिल्कुल मना है।
- इंसानों जैसी लय: कभी-कभी छोटा filler लगाएँ — "अच्छा", "जी", "हाँ भैया", "एक सेकंड"। हर जवाब में नहीं, सिर्फ़ जब स्वाभाविक लगे।
- सम्मानजनक सम्बोधन: "आप", "भैया", "दीदी", "जी"। "तुम" कभी नहीं।
- विषय का प्रवाह (Context Flow): यदि मतदाता केवल 'हाँ', 'हूँ', 'ठीक है' या 'हम्म' जैसी छोटी सहमति दे, तो बातचीत को बीच में न रोकें (धन्यवाद कहकर कॉल न काटें), बल्कि अपनी पिछली बात को स्वाभाविक रूप से आगे बढ़ाते हुए पूरा करें।
- सवाल का जवाब (Direct Answering): यदि मतदाता बीच में अपना कोई नया या बिल्कुल अलग सवाल पूछे, तो अपनी पुरानी बात तुरंत भूलकर सीधे और स्पष्ट रूप से उनके नए सवाल का जवाब दें।
- मतदाता की भाषा से मिलान करें — वह हरियाणवी बोले तो पूरी हरियाणवी में जवाब दें ("के हाल सै?", "थाने", "म्हारे", "कद", "किसा"); भोजपुरी/अवधी बोले तो उसमें मिलाएँ।
- हर बार एक ही शुरुआत न करें ("हाँ बिलकुल" / "जी बिलकुल" बार-बार नहीं)।
- रोबोट जैसी शुरुआतें ("मुझे खुशी होगी...", "आपके प्रश्न का उत्तर...") कभी नहीं।

## 5. विरोध/शंका कैसे सम्भालें (OBJECTIONS)
{objections_block}

## 6. कठोर नियम (HARD RULES)
1. कभी झूठ न बोलें। ऐसा कोई वादा न करें जो प्रत्याशी ने सार्वजनिक रूप से न किया हो।
2. स्थानीय जानकारी (बूथ नं., मतदाता सूची) न पता हो तो स्वीकार करें और {human_callback_number} पर callback ऑफ़र करें।
3. कॉल का अंत हमेशा एक स्पष्ट आग्रह से करें: "{closing_cta}"
4. कभी बहस न करें। मतदाता की शिकायत पहले स्वीकारें, फिर विषय वापस लाएँ।
5. दूसरी पार्टी / व्यक्तियों की आलोचना न करें — केवल अपने प्रत्याशी के काम बताएँ।

## ACOUSTIC TAGS
केवल ये चार: [laughter], [sigh], [sniff], [dissatisfaction-hnn]. 
और कोई bracket-tag न लिखें।"""


def _resolve_placeholders(text: str, fields: Dict[str, Any]) -> str:
    """Very intentionally NOT str.format — user text often contains literal
    braces, math, or curly punctuation that would crash format(). We do a
    simple {key} → value swap for known keys only."""
    if not isinstance(text, str):
        return text
    out = text
    for k, v in fields.items():
        if isinstance(v, (str, int, float)):
            out = out.replace("{" + k + "}", str(v))
    return out


def load_campaign(path: str = DEFAULT_PATH, name: Optional[str] = None) -> Dict[str, Any]:
    """Read campaigns.json and return the campaign dict for `name` (or the
    file's `active` pointer). Returns an empty dict on any I/O or parse error
    so the engine falls back to a safe generic prompt rather than crashing a
    live call."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[campaigns] load error: {e}", flush=True)
        return {}
    campaigns = data.get("campaigns", {})
    key = name or data.get("active", "default")
    campaign = campaigns.get(key) or campaigns.get("default") or {}
    if not campaign:
        return {}
    return dict(campaign)


def save_campaign(campaign: Dict[str, Any], name: str = "default",
                  path: str = DEFAULT_PATH) -> None:
    """Persist an edit. Preserves other campaigns in the file and updates
    `active` only if this is the first write."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"active": name, "campaigns": {}}
    
    if not isinstance(data, dict):
        data = {"active": name, "campaigns": {}}
    
    if "campaigns" not in data or not isinstance(data["campaigns"], dict):
        data["campaigns"] = {}
        
    data["campaigns"][name] = campaign
    data.setdefault("active", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_system_prompt(
    campaign: Dict[str, Any],
    voice_name: str = "AI",
    voice_gender: str = "male",
    voice_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the 6-section political-call-agent system prompt.

    Falls back to a simple generic assistant prompt if `campaign` is empty,
    so a broken/missing campaigns.json never leaves the model with no
    guidance at all."""
    # --- VOICE METADATA CONTEXT ---
    v_meta = voice_metadata or {}
    voice_context = {
        "voice_name": voice_name,
        "voice_gender": voice_gender,
        "voice_age": v_meta.get("age") or "30",
        "voice_about": v_meta.get("about") or "एक मिलनसार और बुद्धिमान सहायक।",
        "voice_catchphrases": v_meta.get("catchphrases") or "जी बताइए!",
    }

    if not campaign:
        base = (
            f"आप {voice_name} हैं — एक गर्मजोशी से भरे, असल इंसान जैसे बोलने वाले हिंदी सहायक।\n"
            f"व्यक्तित्व: {voice_context['voice_about']}\n"
            "यह एक वॉइस कॉल है — बोल-चाल की भाषा में बात करें, किताबी नहीं।\n"
            "\n"
            "## STYLE\n"
            "- मुख्यतः देवनागरी हिंदी। अंग्रेज़ी केवल तब जब यूज़र अंग्रेज़ी में बोले।\n"
            "- बोली अनुकूलन: यदि यूज़र हरियाणवी बोले या हरियाणवी में बोलने को कहे, तो पूरी तरह हरियाणवी बोली में जवाब दें "
            "(जैसे: \"के हाल सै?\", \"थाने\", \"म्हारे\", \"कद\", \"किसा\", \"बता भाई\")। "
            "यदि भोजपुरी/अवधी बोले तो उसमें मिलाएँ।\n"
            "- जवाब अधिकतम 1-2 छोटे वाक्य। लम्बा मोनोलॉग कभी नहीं।\n"
            "- शुरुआत में कभी-कभी छोटा filler लगाएँ: \"अच्छा\", \"हाँ\", \"ठीक है\", \"एक सेकंड\", \"हम्म\"। "
            "हर बार नहीं — सिर्फ़ जब स्वाभाविक लगे।\n"
            "- संवाद वाली लय रखें: \"...है ना?\", \"...तो बताइए\", \"सच में?\"। सुनने वाले से रिश्ता बनता है।\n"
            "- \"आप\" से बात करें। \"तुम\" नहीं।\n"
            "- रोबोट जैसी शुरुआतें (\"मुझे खुशी होगी...\", \"आपके प्रश्न के उत्तर में...\") कभी नहीं।\n"
            "\n"
            "## ACOUSTIC TAGS (emotion hints for TTS)\n"
            "कभी-कभी — ज़रूरत पड़ने पर — ये चार tags उचित जगह लगा सकते हैं:\n"
            "[laughter] — हल्की हँसी, [sigh] — गहरी साँस, [sniff] — हल्की सी सिसकी, [dissatisfaction-hnn] — थोड़ी नाख़ुशी।\n"
            "इनके अलावा कोई भी bracket-tag कभी n लिखें। हर जवाब में tag ज़रूरी नहीं।"
        )
        return _resolve_placeholders(base, voice_context)

    # Resolve basic fields
    c = {**voice_context}
    for k, v in campaign.items():
        if isinstance(v, str):
            c[k] = _resolve_placeholders(v, c)
        else:
            c[k] = v

    # --- CUSTOM SYSTEM INSTRUCTION OVERRIDE ---
    custom_instruction = campaign.get("system_instruction")
    if custom_instruction:
        return _resolve_placeholders(custom_instruction, c)

    # --- DEFAULT POLITICAL PROMPT ---
    agent_name       = c.get("agent_name") or voice_name
    candidate_name   = c.get("candidate_name", "प्रत्याशी जी")
    disclose_ai      = c.get("disclose_ai_on_ask", True)
    
    disclose_ai_line = (
        "अगर मतदाता पूछे \"क्या आप इंसान हैं?\" — सच बताएँ: "
        f"\"मैं एक AI सहायक हूँ, पर {candidate_name} जी के अभियान की ओर से आपसे बात कर रहा हूँ।\""
        if disclose_ai else
        "AI होने का स्वतः खुलासा न करें; केवल सीधे पूछने पर सच बताएँ।"
    )

    objections = c.get("objections", [])
    objection_lines = []
    for o in objections:
        if isinstance(o, dict):
            t = _resolve_placeholders(o.get("trigger", ""), c)
            r = _resolve_placeholders(o.get("reply", ""), c)
            if t and r:
                objection_lines.append(f"- जब मतदाता: {t}\n  तब कहें: \"{r}\"")
    objections_block = "\n".join(objection_lines) if objection_lines else "- (कोई विशेष objection script नहीं — सामान्य शिष्टता से जवाब दें।)"

    call_goal_ex = c.get("call_goal_example", "")
    call_goal_example_block = f"उदाहरण शुरुआत: \"{call_goal_ex}\"" if call_goal_ex else ""

    return _resolve_placeholders(DEFAULT_POLITICAL_PROMPT, {
        **c,
        "agent_name": agent_name,
        "disclose_ai_line": disclose_ai_line,
        "objections_block": objections_block,
        "call_goal_example_block": call_goal_example_block,
    })
