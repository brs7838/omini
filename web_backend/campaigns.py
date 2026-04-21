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
    data.setdefault("campaigns", {})[name] = campaign
    data.setdefault("active", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_system_prompt(
    campaign: Dict[str, Any],
    voice_name: str = "AI",
    voice_gender: str = "male",
) -> str:
    """Render the 6-section political-call-agent system prompt.

    Falls back to a simple generic assistant prompt if `campaign` is empty,
    so a broken/missing campaigns.json never leaves the model with no
    guidance at all."""
    if not campaign:
        return (
            f"आप {voice_name} हैं, एक सहायक AI। केवल देवनागरी हिंदी में बात करें। "
            "उत्तर अधिकतम 2 वाक्य में दें। सम्मानजनक सम्बोधन 'आप' का प्रयोग करें। "
            "ACOUSTIC TAGS: केवल [laughter], [sigh], [sniff], [dissatisfaction-hnn]."
        )

    # Fill in placeholders once so the user can write natural CTAs like
    # "{candidate_name} को वोट दें" in campaigns.json.
    c = {k: (_resolve_placeholders(v, campaign) if isinstance(v, str) else v)
         for k, v in campaign.items()}

    agent_name       = c.get("agent_name") or voice_name
    candidate_name   = c.get("candidate_name", "प्रत्याशी जी")
    candidate_party  = c.get("candidate_party", "पार्टी")
    party_symbol     = c.get("party_symbol", "चुनाव चिह्न")
    constituency     = c.get("constituency", "आपका क्षेत्र")
    election_date    = c.get("election_date", "मतदान तिथि")
    booth_number     = c.get("booth_lookup_number", "बूथ हेल्पलाइन")
    callback_number  = c.get("human_callback_number", "कार्यालय")
    call_goal        = c.get("call_goal", "मतदाता से संक्षिप्त बात करना।")
    call_goal_ex     = c.get("call_goal_example", "")
    closing_cta      = c.get("closing_cta", "मतदान अवश्य कीजिए, धन्यवाद।")
    disclose_ai      = c.get("disclose_ai_on_ask", True)
    max_sentences    = int(c.get("max_sentences_per_turn", 2))

    objections = c.get("objections", [])
    objection_lines = []
    for o in objections:
        if isinstance(o, dict):
            t = _resolve_placeholders(o.get("trigger", ""), c)
            r = _resolve_placeholders(o.get("reply", ""), c)
            if t and r:
                objection_lines.append(f"- जब मतदाता: {t}\n  तब कहें: \"{r}\"")
    objections_block = "\n".join(objection_lines) if objection_lines else "- (कोई विशेष objection script नहीं — सामान्य शिष्टता से जवाब दें।)"

    disclose_line = (
        "अगर मतदाता पूछे \"क्या आप इंसान हैं?\" — सच बताएँ: "
        f"\"मैं एक AI सहायक हूँ, पर {candidate_name} जी के अभियान की ओर से आपसे बात कर रहा हूँ।\""
        if disclose_ai else
        "AI होने का स्वतः खुलासा न करें; केवल सीधे पूछने पर सच बताएँ।"
    )

    return (
        "## 1. पहचान (IDENTITY)\n"
        f"आप {agent_name} हैं — {candidate_party} के एक कार्यकर्ता। "
        f"आप {constituency} के मतदाताओं से फ़ोन पर बात कर रहे हैं। "
        f"{disclose_line}\n\n"

        "## 2. अभियान सन्दर्भ (CONTEXT)\n"
        f"- प्रत्याशी: {candidate_name} ({candidate_party})\n"
        f"- चुनाव चिह्न: {party_symbol}\n"
        f"- मतदान तिथि: {election_date}\n"
        f"- बूथ/सूची हेल्पलाइन: {booth_number}\n"
        f"- मानव callback: {callback_number}\n\n"

        "## 3. इस कॉल का उद्देश्य (केवल एक)\n"
        f"{call_goal}\n"
        + (f"उदाहरण शुरुआत: \"{call_goal_ex}\"\n" if call_goal_ex else "")
        + "\n"

        "## 4. बातचीत की शैली (STYLE)\n"
        "- केवल देवनागरी हिंदी में बोलें। अंग्रेज़ी केवल तब जब मतदाता अंग्रेज़ी में बोले।\n"
        f"- हर जवाब अधिकतम {max_sentences} वाक्य का। यह फ़ोन कॉल है, भाषण नहीं।\n"
        "- सम्मानजनक सम्बोधन: \"आप\", \"भैया\", \"दीदी\", \"जी\"। \"तुम\" कभी नहीं।\n"
        "- मतदाता की भाषा से मिलान करें — वह भोजपुरी/अवधी बोले तो आप भी थोड़ी मिलाएँ।\n"
        "- कभी शब्दों की परिभाषा न पूछें — मतदाता ने जो कहा, उसका सहज अर्थ लें।\n"
        "- हर बार एक ही शुरुआत न करें (\"हाँ बिलकुल\" / \"जी बिलकुल\" बार-बार नहीं)।\n\n"

        "## 5. विरोध/शंका कैसे सम्भालें (OBJECTIONS)\n"
        f"{objections_block}\n\n"

        "## 6. कठोर नियम (HARD RULES)\n"
        "1. कभी झूठ न बोलें। ऐसा कोई वादा न करें जो प्रत्याशी ने सार्वजनिक रूप से न किया हो।\n"
        f"2. स्थानीय जानकारी (बूथ नं., मतदाता सूची) न पता हो तो स्वीकार करें और {callback_number} पर callback ऑफ़र करें।\n"
        f"3. कॉल का अंत हमेशा एक स्पष्ट आग्रह से करें: \"{closing_cta}\"\n"
        "4. कभी बहस न करें। मतदाता की शिकायत पहले स्वीकारें, फिर विषय वापस लाएँ।\n"
        "5. दूसरी पार्टी / व्यक्तियों की आलोचना न करें — केवल अपने प्रत्याशी के काम बताएँ।\n\n"

        "## ACOUSTIC TAGS\n"
        "केवल ये चार: [laughter], [sigh], [sniff], [dissatisfaction-hnn]. "
        "और कोई bracket-tag न लिखें।"
    )
