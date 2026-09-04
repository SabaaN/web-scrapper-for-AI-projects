"""
core/intent.py
==============
Extracts structured intent from a user's natural language message
using Gemini. This is the first place in the system where the LLM
is actually used.

The LLM's only job here is to read a message like:
    "find me RAG chatbot projects under $5000 on Freelancer"
and turn it into a clean structured dict:
    {
        "action":  "SEARCH",
        "query":   "RAG chatbot",
        "filters": { "budget_max": 5000, "platform": "Freelancer.com" }
    }

Nothing is scraped or stored here. This is purely NLU
(Natural Language Understanding) — reading intent, nothing else.

Four action types:
    SEARCH   → user wants fresh/cached project results
    FILTER   → user wants to narrow the current results
    RECALL   → user asks about past results ("what did we find last week?")
    CHITCHAT → greeting, question about the bot, off-topic
"""

import os
import json
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL")

# Initialize Gemini client once at module level
_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Intent Schema ────────────────────────────────────────────────────────────

VALID_ACTIONS = {"SEARCH", "FILTER", "RECALL", "CHITCHAT"}

VALID_PLATFORMS = {
    "Freelancer.com",
    "Remotive",
    "Himalayas",
    "RemoteOK",
    "Arbeitnow",
}

# ─── System Prompt ────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """
You are an intent extraction engine for an AI project leads search tool.

Your ONLY job is to read the user's message and return a JSON object.
Do NOT respond with anything other than valid JSON.
Do NOT include markdown code fences, backticks, or any explanation.
Return ONLY the raw JSON object, nothing else.

The JSON must follow this exact schema:
{
    "action":      string,   // one of: SEARCH, FILTER, RECALL, CHITCHAT
    "query":       string,   // the core AI search term (empty string if not applicable)
    "filters":     object,   // optional filters (empty object if none)
    "force_fresh": boolean,  // true if user explicitly wants fresh results
    "explanation": string    // one sentence: why you chose this action
}

Action definitions:
    SEARCH   — user wants to find AI projects (new search or repeat search)
    FILTER   — user wants to narrow/sort existing results without a new search
    RECALL   — user asks about previous searches or historical data
    CHITCHAT — greeting, thanks, question about the bot, anything off-topic

Filter fields (all optional, only include what the user specified):
    budget_min  : number  (minimum budget in USD)
    budget_max  : number  (maximum budget in USD)
    bid_max     : number  (maximum number of bids, for competition filtering)
    platform    : string  (one of: Freelancer.com, Remotive, Himalayas, RemoteOK, Arbeitnow)
    sort_by     : string  (one of: relevance_score, budget_max, bid_count)

force_fresh = true when user says things like:
    "search again", "refresh", "get fresh results", "re-run", "update results"

Examples:

User: "find me computer vision projects"
{
    "action": "SEARCH",
    "query": "computer vision",
    "filters": {},
    "force_fresh": false,
    "explanation": "User wants to search for computer vision projects."
}

User: "show me only the ones under $3000"
{
    "action": "FILTER",
    "query": "",
    "filters": { "budget_max": 3000 },
    "force_fresh": false,
    "explanation": "User wants to filter current results by budget."
}

User: "what did we find last Tuesday?"
{
    "action": "RECALL",
    "query": "",
    "filters": {},
    "force_fresh": false,
    "explanation": "User is asking about historical search results."
}

User: "hi, how are you?"
{
    "action": "CHITCHAT",
    "query": "",
    "filters": {},
    "force_fresh": false,
    "explanation": "User is greeting the bot."
}

User: "find RAG pipeline projects on Freelancer under $5000 with low competition"
{
    "action": "SEARCH",
    "query": "RAG pipeline",
    "filters": { "platform": "Freelancer.com", "budget_max": 5000, "bid_max": 10 },
    "force_fresh": false,
    "explanation": "User wants RAG pipeline projects filtered by platform, budget, and competition."
}

User: "search again for machine learning projects"
{
    "action": "SEARCH",
    "query": "machine learning",
    "filters": {},
    "force_fresh": true,
    "explanation": "User explicitly wants a fresh search, bypassing cache."
}
""".strip()


# ─── Intent Dataclass ─────────────────────────────────────────────────────────

class Intent:
    """
    Structured result of intent extraction.

    Attributes:
        action      : SEARCH | FILTER | RECALL | CHITCHAT
        query       : the core search term extracted from the message
        filters     : dict of filters (budget_min, budget_max, bid_max, platform, sort_by)
        force_fresh : whether to bypass the cache
        explanation : LLM's reason for this classification
        raw         : the raw JSON string from the LLM (for debugging)
    """

    def __init__(
        self,
        action:      str,
        query:       str,
        filters:     dict,
        force_fresh: bool,
        explanation: str,
        raw:         str = "",
    ):
        self.action      = action.upper()
        self.query       = query.strip()
        self.filters     = filters
        self.force_fresh = force_fresh
        self.explanation = explanation
        self.raw         = raw

    def is_search(self)   -> bool: return self.action == "SEARCH"
    def is_filter(self)   -> bool: return self.action == "FILTER"
    def is_recall(self)   -> bool: return self.action == "RECALL"
    def is_chitchat(self) -> bool: return self.action == "CHITCHAT"

    def __repr__(self) -> str:
        return (
            f"Intent(action={self.action}, query='{self.query}', "
            f"filters={self.filters}, force_fresh={self.force_fresh})"
        )


# ─── Fallback Parser ──────────────────────────────────────────────────────────

def _rule_based_fallback(message: str) -> Intent:
    """
    Simple keyword-based fallback if Gemini fails or returns invalid JSON.
    Not as smart as the LLM but always returns something valid.
    """
    msg = message.lower().strip()

    # Force fresh signals
    force_fresh = any(phrase in msg for phrase in [
        "search again", "refresh", "fresh results",
        "re-run", "update results", "new search",
    ])

    # FILTER signals — narrowing without a new topic
    filter_words = ["only", "filter", "show me only", "narrow", "sort by", "under $", "above $"]
    if any(w in msg for w in filter_words) and not any(
        ai in msg for ai in ["machine learning", "computer vision", "llm", "rag", "nlp", "ai", "deep learning"]
    ):
        return Intent(
            action="FILTER", query="", filters={},
            force_fresh=False,
            explanation="Fallback: detected filter keywords without new AI topic",
        )

    # RECALL signals
    recall_words = ["last time", "previous", "history", "what did we", "last week", "yesterday", "before"]
    if any(w in msg for w in recall_words):
        return Intent(
            action="RECALL", query="", filters={},
            force_fresh=False,
            explanation="Fallback: detected recall/history keywords",
        )

    # CHITCHAT signals
    chitchat_words = ["hello", "hi", "hey", "thanks", "thank you", "who are you", "what are you", "help"]
    if any(w in msg for w in chitchat_words) and len(msg.split()) < 6:
        return Intent(
            action="CHITCHAT", query="", filters={},
            force_fresh=False,
            explanation="Fallback: detected greeting or short chitchat",
        )

    # Default to SEARCH — use the original message as the query
    return Intent(
        action="SEARCH",
        query=message.strip(),
        filters={},
        force_fresh=force_fresh,
        explanation="Fallback: defaulting to SEARCH with raw message as query",
    )


# ─── JSON Cleaner ─────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    """
    Strip markdown fences and whitespace from LLM output.
    Gemini sometimes wraps JSON in ```json ... ``` even when told not to.
    """
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ─── Core Extraction ──────────────────────────────────────────────────────────

def extract_intent(
    message: str,
    chat_history: list[dict] | None = None,
) -> Intent:
    """
    Extract structured intent from a user message using Gemini.

    Args:
        message:      The user's raw message
        chat_history: Optional list of previous turns for context.
                      Format: [{"role": "user"|"assistant", "content": "..."}]
                      Passed so the LLM understands follow-up messages like
                      "show me only the cheap ones" (what cheap ones? the ones
                      from the previous search)

    Returns:
        Intent object with action, query, filters, force_fresh, explanation
    """

    if not message or not message.strip():
        return Intent(
            action="CHITCHAT", query="", filters={},
            force_fresh=False,
            explanation="Empty message received",
        )

    # Build context from recent chat history (last 4 turns max)
    context = ""
    if chat_history:
        recent = chat_history[-4:]
        lines  = []
        for turn in recent:
            role    = turn.get("role", "")
            content = turn.get("content", "")[:200]   # truncate long turns
            lines.append(f"{role.capitalize()}: {content}")
        if lines:
            context = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    prompt = f"{context}User message: {message}"

    try:
        from google.genai import types

        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=INTENT_SYSTEM_PROMPT,
                temperature=0.1,         # low temp → consistent, predictable JSON
                max_output_tokens=300,
            ),
        )

        raw_text = response.text
        cleaned  = _clean_json(raw_text)
        parsed   = json.loads(cleaned)

        # Validate action
        action = parsed.get("action", "SEARCH").upper()
        if action not in VALID_ACTIONS:
            action = "SEARCH"

        # Validate platform if provided
        filters = parsed.get("filters", {})
        if "platform" in filters and filters["platform"] not in VALID_PLATFORMS:
            del filters["platform"]

        return Intent(
            action      = action,
            query       = parsed.get("query", ""),
            filters     = filters,
            force_fresh = bool(parsed.get("force_fresh", False)),
            explanation = parsed.get("explanation", ""),
            raw         = raw_text,
        )

    except json.JSONDecodeError as e:
        print(f"  [Intent] JSON parse failed: {e} — using fallback")
        return _rule_based_fallback(message)

    except Exception as e:
        print(f"  [Intent] Gemini call failed: {e} — using fallback")
        return _rule_based_fallback(message)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_messages = [
        # SEARCH cases
        "find me computer vision projects",
        "I need RAG pipeline projects under $5000 on Freelancer",
        "show me machine learning projects with low competition",
        "search again for NLP chatbot projects",

        # FILTER cases
        "show me only the ones under $3000",
        "filter by Freelancer only",
        "sort by budget",
        "only show low competition ones",

        # RECALL cases
        "what did we find last time?",
        "show me previous results",
        "what was in yesterday's search?",

        # CHITCHAT cases
        "hi there",
        "thanks!",
        "what can you do?",
    ]

    print("Testing core/intent.py...\n")
    print(f"  Model: {LLM_MODEL}")
    print(f"  {'─' * 60}\n")

    for msg in test_messages:
        intent = extract_intent(msg)
        print(f"  Message:  \"{msg}\"")
        print(f"  → {intent}")
        print(f"     Why: {intent.explanation}")
        print() 