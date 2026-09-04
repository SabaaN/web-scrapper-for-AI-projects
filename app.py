"""
app.py
======
Streamlit chat interface.

Search queries  → scraper runs (or cache hit) → download JSON + Excel
Everything else → Gemini handles conversationally

Run with:
  streamlit run app.py
"""

import os
import json
import streamlit as st
from dataclasses import asdict

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Project Leads",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports ──────────────────────────────────────────────────────────────────

from db.connection import get_db
from db.models import create_indexes
from db.queries import get_recent_scrape_runs, get_all_listings, get_platform_stats

from core.intent import extract_intent
from core.cache import get_or_invalidate, store_results
from core.scraper import scrape
from core.llm import ConversationManager

# ─── DB Init ──────────────────────────────────────────────────────────────────

@st.cache_resource
def init_db():
    get_db()
    create_indexes()

init_db()

# ─── Session State ────────────────────────────────────────────────────────────

if "manager"      not in st.session_state:
    st.session_state.manager = ConversationManager()
if "messages"     not in st.session_state:
    st.session_state.messages = []
if "last_query"   not in st.session_state:
    st.session_state.last_query = ""
if "last_count"   not in st.session_state:
    st.session_state.last_count = 0
if "is_searching" not in st.session_state:
    st.session_state.is_searching = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_file(directory: str, ext: str) -> str | None:
    """Return the most recently modified file with given extension."""
    if not os.path.isdir(directory):
        return None
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(ext)
    ]
    return max(files, key=os.path.getmtime) if files else None


def render_message(msg: dict) -> None:
    """Render one chat message with optional download buttons."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("json_path") or msg.get("excel_path"):
            st.success(
                f"✅ **{msg.get('result_count', 0)} projects fetched** — "
                f"download your results:"
            )
            col1, col2 = st.columns(2)

            json_path  = msg.get("json_path")
            excel_path = msg.get("excel_path")

            if json_path and os.path.exists(json_path):
                with open(json_path, "rb") as f:
                    col1.download_button(
                        label="⬇️ Download JSON",
                        data=f,
                        file_name=os.path.basename(json_path),
                        mime="application/json",
                        use_container_width=True,
                        key=f"json_{msg.get('_id', json_path)}",
                    )

            if excel_path and os.path.exists(excel_path):
                with open(excel_path, "rb") as f:
                    col2.download_button(
                        label="⬇️ Download Excel",
                        data=f,
                        file_name=os.path.basename(excel_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key=f"excel_{msg.get('_id', excel_path)}",
                    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("Cache")
    st.slider("Cache expiry (hours)", 1, 24, 4,
              help="How long before cached results expire")

    st.subheader("Scraper")
    st.slider("Min relevance score", 0, 30, 3,
              help="Lower = more results, higher = stricter filtering")
    st.multiselect(
        "Platforms",
        ["Freelancer.com", "Remotive", "Himalayas", "RemoteOK", "Arbeitnow"],
        default=["Freelancer.com", "Remotive", "Himalayas", "RemoteOK", "Arbeitnow"],
    )

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.manager    = ConversationManager()
        st.session_state.messages   = []
        st.session_state.last_query = ""
        st.session_state.last_count = 0
        st.rerun()

    st.divider()

    st.subheader("📋 Recent Searches")
    try:
        runs = get_recent_scrape_runs(limit=5)
        if runs:
            for run in runs:
                ran_at = run.get("ran_at")
                ts     = ran_at.strftime("%b %d %H:%M") if ran_at else "—"
                st.caption(
                    f"**{run.get('query', '?')}**  \n"
                    f"{run.get('total_results', 0)} results · {ts}"
                )
        else:
            st.caption("No searches yet.")
    except Exception:
        st.caption("Database not connected.")

    st.divider()

    st.subheader("📊 Platform Stats")
    try:
        stats = get_platform_stats()
        for s in stats:
            st.caption(f"**{s['_id']}**: {s['count']} listings")
        if not stats:
            st.caption("No data yet.")
    except Exception:
        st.caption("No data yet.")

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🤖 AI Project Leads")
st.caption(
    "Scrapes Freelancer · Remotive · Himalayas · RemoteOK · Arbeitnow "
    "for AI/ML client projects"
)

if st.session_state.last_query:
    st.info(
        f"📌 Last search: **{st.session_state.last_query}** "
        f"— {st.session_state.last_count} results"
    )

st.divider()

# ─── Render existing messages ─────────────────────────────────────────────────

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I find AI/ML project leads for your dev team.\n\n"
            "**To search**, just describe what you need:\n"
            "- *\"find RAG chatbot projects\"*\n"
            "- *\"computer vision projects\"*\n"
            "- *\"machine learning pipeline projects\"*\n\n"
            "Results are saved as JSON and Excel — ready to download.\n\n"
            "You can also ask me anything else and I'll answer conversationally."
        )

for msg in st.session_state.messages:
    render_message(msg)

# ─── Chat Input ───────────────────────────────────────────────────────────────

user_input = st.chat_input(
    "Search for AI projects or ask anything...",
    disabled=st.session_state.is_searching,
)

if not user_input:
    st.stop()

# ── Show user message immediately ─────────────────────────────────────────────
import time as _time
msg_id = str(int(_time.time() * 1000))   # unique key for download buttons

st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

# ── Extract intent ─────────────────────────────────────────────────────────────
with st.spinner("Thinking..."):
    intent = extract_intent(
        message=user_input,
        chat_history=st.session_state.manager.get_history(),
    )

# ── Build assistant message ────────────────────────────────────────────────────
assistant_msg = {
    "role":         "assistant",
    "content":      "",
    "json_path":    None,
    "excel_path":   None,
    "result_count": 0,
    "_id":          msg_id,
}

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH — scraper runs, LLM is NOT used for results
# ─────────────────────────────────────────────────────────────────────────────
if intent.is_search():

    query = intent.query if intent.query else user_input

    # ── Check cache ───────────────────────────────────────────────────────────
    with st.spinner("Checking cache..."):
        cache_result = get_or_invalidate(
            query=query,
            filters=intent.filters or None,
            force_fresh=intent.force_fresh,
        )

    if cache_result.hit:
        # ── Cache hit — no scraping needed ────────────────────────────────────
        n   = len(cache_result.listings)
        age = cache_result.status.get("age_minutes", 0)
        exp = cache_result.status.get("expires_in_minutes", 0)

        assistant_msg["content"] = (
            f"📦 **Loaded from cache** — these results are **{age} min old** "
            f"(refreshes in {exp} min).\n\n"
            f"Query: `{query}` · **{n} projects** found across "
            f"Freelancer, Remotive, Himalayas, RemoteOK and Arbeitnow."
        )
        assistant_msg["result_count"] = n
        assistant_msg["json_path"]    = find_latest_file("results", ".json")
        assistant_msg["excel_path"]   = find_latest_file("excel",   ".xlsx")

        st.session_state.last_query = query
        st.session_state.last_count = n

    else:
        # ── Cache miss — run scraper ──────────────────────────────────────────
        st.session_state.is_searching = True

        with st.spinner(
            f"Please wait. "
            f"this takes 30–60 seconds..."
        ):
            try:
                listings = scrape(
                    query=query,
                    save_json=True,
                    save_excel=True,
                    silent=True,
                )

                store_results(
                    query=query,
                    listings=listings,
                    filters_applied=intent.filters,
                )

                n          = len(listings)
                json_path  = find_latest_file("results", ".json")
                excel_path = find_latest_file("excel",   ".xlsx")

                # Count per platform for the summary
                from collections import Counter
                platform_counts = Counter(p.platform for p in listings)
                platform_lines  = "  \n".join(
                    f"• **{k}**: {v}" for k, v in platform_counts.most_common()
                )

                assistant_msg["content"] = (
                    f"✅ **Scrape complete** for `{query}`\n\n"
                    f"**{n} AI/ML projects** found:\n\n"
                    f"{platform_lines}\n\n"
                    f"Results saved to `{os.path.basename(json_path or '')}` "
                    f"and `{os.path.basename(excel_path or '')}`. "
                    f"Download below 👇"
                )
                assistant_msg["result_count"] = n
                assistant_msg["json_path"]    = json_path
                assistant_msg["excel_path"]   = excel_path

                st.session_state.last_query = query
                st.session_state.last_count = n

            except Exception as e:
                assistant_msg["content"] = (
                    f"⚠️ Something went wrong while scraping: `{e}`\n\n"
                    f"This is usually a network issue or a platform blocking "
                    f"the request. Try again in a moment."
                )
            finally:
                st.session_state.is_searching = False

# ─────────────────────────────────────────────────────────────────────────────
# FILTER — narrow results already in context (LLM responds)
# ─────────────────────────────────────────────────────────────────────────────
elif intent.is_filter():
    with st.spinner("Filtering..."):
        response = st.session_state.manager.respond_to_filter(
            user_message=user_input,
            filters=intent.filters,
        )
    assistant_msg["content"] = response

# ─────────────────────────────────────────────────────────────────────────────
# RECALL — ask about past results (LLM responds)
# ─────────────────────────────────────────────────────────────────────────────
elif intent.is_recall():
    with st.spinner("Checking history..."):
        try:
            past     = get_all_listings(limit=10)
            last_run = get_recent_scrape_runs(limit=1)
            pq       = last_run[0].get("query", "") if last_run else ""
        except Exception:
            past, pq = [], ""

        response = st.session_state.manager.respond_to_recall(
            user_message=user_input,
            past_listings=past,
            past_query=pq,
        )
    assistant_msg["content"] = response

# ─────────────────────────────────────────────────────────────────────────────
# CHITCHAT — anything else (LLM responds)
# ─────────────────────────────────────────────────────────────────────────────
else:
    with st.spinner("..."):
        response = st.session_state.manager.respond_to_chitchat(user_input)
    assistant_msg["content"] = response

# ── Append and render ─────────────────────────────────────────────────────────
st.session_state.messages.append(assistant_msg)
render_message(assistant_msg)