"""
core/llm.py
===========
Gemini-powered conversation layer.

Two responsibilities:
  1. Format raw scraper results into clean, conversational responses
  2. Handle non-search turns (FILTER, RECALL, CHITCHAT) conversationally

This is the only file that talks to Gemini for conversation purposes.
intent.py uses Gemini too but only for classification, not conversation.

The main class is ConversationManager — one instance per user session.
It holds the chat history and knows the current search context
(what was last searched, what results are currently shown).
"""

import os
from dataclasses import asdict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
LLM_MODEL       = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", 2000))

_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── System Prompt ────────────────────────────────────────────────────────────

CONVERSATION_SYSTEM_PROMPT = """
You are an AI project leads assistant for a software development agency.

Your job is to help the team find AI/ML client projects to work on.
You have access to a scraper that searches platforms like Freelancer.com,
Remotive, Himalayas, RemoteOK, and Arbeitnow for AI project opportunities.

Your personality:
- Professional but friendly
- Concise — no unnecessary padding or filler phrases
- Direct — lead with the answer, explain after
- Honest — if results are thin, say so

What you can do:
- Search for AI project leads across multiple platforms
- Filter results by budget, platform, competition level
- Recall and discuss previous search results
- Answer questions about the projects found
- Give recommendations on which projects to prioritize

What you cannot do:
- Access any website directly (the scraper does that)
- Guarantee project availability (listings change fast)
- Apply for projects on the user's behalf

When presenting project results:
- Lead with the count and a one-line summary
- Highlight the top 2-3 projects with key details
- Always mention bid count when available (it signals competition)
- Group by platform if results span multiple sources
- Keep individual project summaries to 3-4 lines max

When the user asks to filter or sort:
- Apply the filter to what's already shown
- Confirm what you filtered and how many remain
- Don't re-scrape unless explicitly asked

Tone: think of yourself as a sharp research analyst
briefing a team before a client pitch meeting.
""".strip()


# ─── Response Formatter ───────────────────────────────────────────────────────

def _listings_to_context(listings: list[dict], max_listings: int = 20) -> str:
    """
    Convert raw MongoDB listing dicts into a compact text block
    that Gemini can read and summarise.

    We cap at max_listings to stay within token limits.
    The most relevant ones (highest relevance_score) come first
    since MongoDB returns them sorted that way.
    """
    if not listings:
        return "No project listings available."

    lines = [f"Total results: {len(listings)}\n"]

    for i, p in enumerate(listings[:max_listings], 1):
        budget = "Not specified"
        b_min  = p.get("budget_min")
        b_max  = p.get("budget_max")
        curr   = p.get("currency", "USD") or "USD"
        if b_min and b_max:
            budget = (
                f"{curr} {b_min:,.0f}"
                if b_min == b_max
                else f"{curr} {b_min:,.0f} – {b_max:,.0f}"
            )

        bid_count = p.get("bid_count")
        competition = ""
        if bid_count is not None:
            level = "Low" if bid_count < 10 else ("Medium" if bid_count < 25 else "High")
            competition = f" | Bids: {bid_count} ({level} competition)"

        skills = ", ".join(p.get("skills", [])[:6]) or "Not listed"
        desc   = (p.get("description") or "")[:200]
        score  = p.get("relevance_score", 0)
        pref   = p.get("preference_rank")
        star   = "⭐ " if pref and pref <= 3 else ""

        lines.append(
            f"{star}#{i} {p.get('title', 'Untitled')}\n"
            f"  Platform: {p.get('platform')} | Budget: {budget}{competition}\n"
            f"  Skills: {skills}\n"
            f"  Score: {score} | Posted: {p.get('posted_date', 'Unknown')}\n"
            f"  Description: {desc}\n"
            f"  Link: {p.get('url', '')}\n"
        )

    if len(listings) > max_listings:
        lines.append(f"... and {len(listings) - max_listings} more results not shown.")

    return "\n".join(lines)


# ─── Conversation Manager ─────────────────────────────────────────────────────

class ConversationManager:
    """
    Manages one user session — chat history, current results context,
    and all Gemini calls for conversation.

    One instance is created when the Streamlit app starts
    and lives for the duration of the session (stored in st.session_state).

    Attributes:
        history         : list of {"role": "user"|"assistant", "content": "..."}
        current_query   : the last search query run
        current_listings: the listings currently in context
        _chat           : the Gemini chat session (handles history internally)
    """

    def __init__(self):
        self.history:          list[dict] = []
        self.current_query:    str        = ""
        self.current_listings: list[dict] = []
        self._chat = _client.chats.create(
            model=LLM_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=CONVERSATION_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

    # ── Public methods ────────────────────────────────────────────────────────

    def respond_to_search(
        self,
        query:        str,
        listings:     list[dict],
        from_cache:   bool = False,
        cache_status: dict | None = None,
    ) -> str:
        """
        Called after a SEARCH action — formats scraper results
        into a conversational response.

        Args:
            query:        The search query that was run
            listings:     The project listing dicts to present
            from_cache:   Whether results came from cache or fresh scrape
            cache_status: Cache metadata (age, expiry) to mention if relevant

        Returns:
            Formatted assistant response string
        """
        self.current_query    = query
        self.current_listings = listings

        source_note = ""
        if from_cache and cache_status:
            age = cache_status.get("age_minutes", 0)
            exp = cache_status.get("expires_in_minutes", 0)
            source_note = f"(Results from cache — {age} min old, refreshes in {exp} min)\n\n"

        listings_text = _listings_to_context(listings)

        prompt = (
            f"The user searched for: \"{query}\"\n\n"
            f"{source_note}"
            f"Here are the scraped project listings:\n\n"
            f"{listings_text}\n\n"
            f"Present these results to the user in a clear, conversational way. "
            f"Highlight the best opportunities. Mention bid counts where available. "
            f"If results seem thin or off-topic, say so honestly."
        )

        response = self._send(prompt, user_facing_message=f"Find me {query} projects")
        return response

    def respond_to_filter(
        self,
        user_message: str,
        filters:      dict,
    ) -> str:
        """
        Called after a FILTER action — applies filters to current listings
        in memory and asks Gemini to present the narrowed results.

        Args:
            user_message: The user's original filter request
            filters:      The extracted filter dict from intent.py

        Returns:
            Formatted assistant response string
        """
        if not self.current_listings:
            return (
                "I don't have any results loaded right now. "
                "Try searching for something first — for example: "
                "\"find me RAG chatbot projects\"."
            )

        filtered = self._apply_filters(self.current_listings, filters)
        listings_text = _listings_to_context(filtered)

        filter_desc = self._describe_filters(filters)

        prompt = (
            f"The user wants to filter the current results: \"{user_message}\"\n\n"
            f"Applied filters: {filter_desc}\n"
            f"Results after filtering ({len(filtered)} of {len(self.current_listings)}):\n\n"
            f"{listings_text}\n\n"
            f"Present the filtered results conversationally. "
            f"Confirm what was filtered and how many remain."
        )

        response = self._send(prompt, user_facing_message=user_message)
        return response

    def respond_to_recall(
        self,
        user_message:  str,
        past_listings: list[dict] | None = None,
        past_query:    str = "",
    ) -> str:
        """
        Called after a RECALL action — discusses historical results.

        Args:
            user_message:  The user's recall request
            past_listings: Historical listings from MongoDB (optional)
            past_query:    The query those listings came from

        Returns:
            Formatted assistant response string
        """
        if not past_listings:
            prompt = (
                f"The user asked: \"{user_message}\"\n\n"
                f"There are no historical search results in the database yet. "
                f"Tell them this and suggest they run a search first."
            )
        else:
            listings_text = _listings_to_context(past_listings)
            prompt = (
                f"The user asked about past results: \"{user_message}\"\n\n"
                f"These were found in a previous search for \"{past_query}\":\n\n"
                f"{listings_text}\n\n"
                f"Summarise these historical results in context of what the user asked."
            )

        response = self._send(prompt, user_facing_message=user_message)
        return response

    def respond_to_chitchat(self, user_message: str) -> str:
        """
        Called after a CHITCHAT action — handles greetings, questions
        about the bot, thanks, and anything off-topic.

        Args:
            user_message: The user's message

        Returns:
            Conversational assistant response
        """
        response = self._send(user_message, user_facing_message=user_message)
        return response

    def get_history(self) -> list[dict]:
        """Returns the conversation history for intent.py context."""
        return self.history

    def reset(self) -> None:
        """
        Resets the conversation — clears history, current results,
        and creates a fresh Gemini chat session.
        Called when the user clicks 'New Conversation' in the UI.
        """
        self.history          = []
        self.current_query    = ""
        self.current_listings = []
        self._chat = _client.chats.create(
            model=LLM_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=CONVERSATION_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _send(self, prompt: str, user_facing_message: str = "") -> str:
        """
        Sends a message to Gemini and updates chat history.

        Args:
            prompt:              The full prompt including context and data.
                                 This is what Gemini actually receives.
            user_facing_message: The shorter user message to store in history
                                 (what the user actually typed, not the full prompt)

        Returns:
            Gemini's response text
        """
        try:
            response      = self._chat.send_message(prompt)
            response_text = response.text.strip()

            # Store the human-readable version in history, not the full data prompt
            display_msg = user_facing_message if user_facing_message else prompt[:100]
            self.history.append({"role": "user",      "content": display_msg})
            self.history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            error_msg = (
                f"I ran into an issue generating a response: {e}\n"
                f"The data was fetched successfully — "
                f"try rephrasing your message."
            )
            print(f"  [LLM] Gemini error: {e}")
            return error_msg

    def _apply_filters(
        self,
        listings: list[dict],
        filters:  dict,
    ) -> list[dict]:
        """
        Apply filter dict to a list of listing dicts in memory.
        No database call — works on whatever is currently loaded.
        """
        result = listings

        if "budget_min" in filters:
            result = [
                p for p in result
                if p.get("budget_min") is not None
                and p["budget_min"] >= filters["budget_min"]
            ]

        if "budget_max" in filters:
            result = [
                p for p in result
                if p.get("budget_max") is not None
                and p["budget_max"] <= filters["budget_max"]
            ]

        if "bid_max" in filters:
            result = [
                p for p in result
                if p.get("bid_count") is not None
                and p["bid_count"] <= filters["bid_max"]
            ]

        if "platform" in filters:
            result = [
                p for p in result
                if p.get("platform") == filters["platform"]
            ]

        if "sort_by" in filters:
            sort_field = filters["sort_by"]
            reverse    = sort_field != "bid_count"
            result = sorted(
                result,
                key=lambda x: x.get(sort_field) or (0 if reverse else 999),
                reverse=reverse,
            )

        return result

    @staticmethod
    def _describe_filters(filters: dict) -> str:
        """Turn a filter dict into a human-readable string for the prompt."""
        if not filters:
            return "none"
        parts = []
        if "budget_min" in filters:
            parts.append(f"budget ≥ ${filters['budget_min']:,}")
        if "budget_max" in filters:
            parts.append(f"budget ≤ ${filters['budget_max']:,}")
        if "bid_max" in filters:
            parts.append(f"bids ≤ {filters['bid_max']}")
        if "platform" in filters:
            parts.append(f"platform = {filters['platform']}")
        if "sort_by" in filters:
            parts.append(f"sorted by {filters['sort_by']}")
        return ", ".join(parts)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Testing core/llm.py...\n")
    print(f"  Model: {LLM_MODEL}\n")

    manager = ConversationManager()

    # ── Test 1: chitchat ──────────────────────────────────────────────────────
    print("Test 1 — chitchat:")
    reply = manager.respond_to_chitchat("Hi! What can you help me with?")
    print(f"  {reply[:200]}\n")

    # ── Test 2: search response with dummy listings ───────────────────────────
    print("Test 2 — search response:")
    dummy_listings = [
        {
            "title":           "AI Chatbot with RAG for Customer Support",
            "platform":        "Freelancer.com",
            "description":     "Build a RAG-based chatbot using LangChain and GPT-4 for a SaaS company.",
            "budget_min":      3000,
            "budget_max":      6000,
            "currency":        "USD",
            "skills":          ["Python", "LangChain", "RAG", "OpenAI"],
            "bid_count":       4,
            "relevance_score": 42.0,
            "preference_rank": 1,
            "posted_date":     "2026-09-01",
            "url":             "https://freelancer.com/projects/test-001",
        },
        {
            "title":           "NLP Pipeline for Document Classification",
            "platform":        "Remotive",
            "description":     "Build an NLP classification pipeline for legal documents using BERT.",
            "budget_min":      5000,
            "budget_max":      8000,
            "currency":        "USD",
            "skills":          ["Python", "BERT", "NLP", "HuggingFace"],
            "bid_count":       None,
            "relevance_score": 38.0,
            "preference_rank": 2,
            "posted_date":     "2026-09-01",
            "url":             "https://remotive.com/jobs/test-002",
        },
        {
            "title":           "LLM Fine-tuning for Medical QA",
            "platform":        "Himalayas",
            "description":     "Fine-tune an open-source LLM on medical Q&A datasets.",
            "budget_min":      8000,
            "budget_max":      15000,
            "currency":        "USD",
            "skills":          ["Python", "PyTorch", "LLM", "Fine-tuning"],
            "bid_count":       2,
            "relevance_score": 35.0,
            "preference_rank": 3,
            "posted_date":     "2026-09-01",
            "url":             "https://himalayas.app/jobs/test-003",
        },
    ]

    reply = manager.respond_to_search(
        query="RAG chatbot",
        listings=dummy_listings,
        from_cache=False,
    )
    print(f"  {reply[:400]}\n")

    # ── Test 3: filter response ───────────────────────────────────────────────
    print("Test 3 — filter response:")
    reply = manager.respond_to_filter(
        user_message="show me only the ones under $7000",
        filters={"budget_max": 7000},
    )
    print(f"  {reply[:300]}\n")

    # ── Test 4: follow-up chitchat (tests history) ────────────────────────────
    print("Test 4 — follow-up (tests conversation history):")
    reply = manager.respond_to_chitchat("Which of those projects had the lowest competition?")
    print(f"  {reply[:300]}\n")

    # ── Test 5: recall with no history ───────────────────────────────────────
    print("Test 5 — recall with no past data:")
    reply = manager.respond_to_recall(
        user_message="what did we find last week?",
        past_listings=None,
    )
    print(f"  {reply[:200]}\n")

    print("✓ All LLM tests passed")