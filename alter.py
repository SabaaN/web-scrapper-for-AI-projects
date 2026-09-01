"""
AI Project Leads Scraper
========================
Scrapes AI/ML project opportunities from:
  - Freelancer.com  (public JSON endpoint, no auth)
  - PeoplePerHour   (static HTML)
  - Toptal          (static HTML)
  - Guru.com        (static HTML — client-posted projects)

Usage:
  python main.py
  python main.py --query "computer vision object detection"
  python main.py --query "NLP chatbot LLM"
"""

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime

import httpx
from dotenv import load_dotenv
from selectolax.parser import HTMLParser

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

REQUEST_DELAY    = float(os.getenv("REQUEST_DELAY", 2.5))
REQUEST_TIMEOUT  = int(os.getenv("REQUEST_TIMEOUT", 15))
FREELANCER_LIMIT = int(os.getenv("FREELANCER_LIMIT", 20))
PPH_BASE_URL     = os.getenv("PPH_BASE_URL",    "https://www.peopleperhour.com")
TOPTAL_BASE_URL  = os.getenv("TOPTAL_BASE_URL", "https://www.toptal.com")
GURU_BASE_URL    = os.getenv("GURU_BASE_URL",   "https://www.guru.com")
RESULTS_DIR      = os.getenv("RESULTS_DIR",     "results")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── KEYWORD MAP ──────────────────────────────────────────────────────────────
# User query words → specific search strings sent to each platform.
# Kept short and explicit — vague terms like "model training" caused junk results.

KEYWORD_MAP: dict[str, list[str]] = {
"computer vision": ["computer vision", "object detection", "image recognition", "image classification", "YOLO", "image segmentation", "semantic segmentation", "instance segmentation", "face recognition", "facial recognition", "pose estimation", "visual inspection", "video analytics", "video intelligence", "OCR"],
    "nlp": ["NLP", "natural language processing", "text classification", "sentiment analysis", "named entity recognition", "NER", "text extraction", "text summarization", "question answering", "language model", "document understanding"],
    "llm": ["LLM", "large language model", "ChatGPT API", "GPT", "Claude", "Gemini", "RAG", "RAG pipeline", "retrieval augmented generation", "fine tuning LLM", "LLM fine-tuning", "prompt engineering", "LLM application", "LLM integration", "AI chatbot"],
    "agent": ["AI agent", "AI agents", "agentic AI", "agentic systems", "autonomous agent", "autonomous AI", "multi-agent system", "multi-agent", "AI automation agent", "agentic workflow", "AI assistant"],
    "ml": ["machine learning", "ML", "deep learning", "AI model", "artificial intelligence", "predictive modeling", "classification model", "regression model", "neural network", "supervised learning", "unsupervised learning", "reinforcement learning"],
    "mlops": ["MLOps", "ML Ops", "model deployment", "ML pipeline", "machine learning pipeline", "model monitoring", "model serving", "model inference", "model management", "AI infrastructure"],
    "ocr_document_ai": ["OCR", "optical character recognition", "document extraction AI", "document AI", "intelligent document processing", "IDP", "document processing", "document understanding", "invoice extraction", "receipt extraction", "PDF extraction", "form extraction"],
    "recommendation": ["recommendation system", "recommendation engine", "personalization engine", "personalized recommendations", "recommender system", "product recommendation", "content recommendation"],
    "generative_ai": ["generative AI", "GenAI", "generative artificial intelligence", "image generation AI", "text generation", "AI content generation", "stable diffusion", "DALL-E", "Midjourney", "diffusion model", "multimodal AI"],
    "speech": ["speech recognition", "automatic speech recognition", "ASR", "text to speech", "TTS", "voice AI", "voice assistant", "speech-to-text", "voice bot", "conversational AI", "voice automation"],
    "data_science": ["data science", "data analysis", "AI data analysis", "predictive analytics", "business intelligence", "BI", "data modeling", "forecasting", "time series forecasting"],
    "multimodal_ai": ["multimodal AI", "multimodal model", "vision language model", "VLM", "image understanding", "video understanding", "audio AI", "vision-language model"],
    "rag": ["RAG", "retrieval augmented generation", "retrieval-augmented generation", "vector search", "semantic search", "knowledge base AI", "AI knowledge base", "enterprise search", "document Q&A"],
    "chatbot": ["AI chatbot", "chatbot development", "intelligent chatbot", "conversational AI", "customer service chatbot", "WhatsApp chatbot", "website chatbot", "AI virtual assistant"],
    "ai_automation": ["AI automation", "AI workflow automation", "intelligent automation", "business process automation", "AI workflow", "workflow automation", "automated decision making", "process automation"],
    "predictive_ai": ["predictive AI", "predictive analytics", "demand forecasting", "sales forecasting", "price prediction", "risk prediction", "fraud prediction", "predictive maintenance", "churn prediction"],
}

# ─── AI RELEVANCE FILTER ──────────────────────────────────────────────────────
# Two-stage filter:
#   1. Title must contain at least one STRONG AI term (tight check)
#   2. OR title + description together contain enough signal
#
# This prevents Freelancer's loose search from returning cleaning jobs,
# aviation consulting, etc. that happen to share a word with our query.

# Strong terms — if ANY of these appear in the title alone, it's relevant
TITLE_MUST_MATCH: set[str] = {
    "machine learning", "deep learning", "neural network", "artificial intelligence",
    "computer vision", "object detection", "image recognition", "image classification",
    "yolo", "opencv", "face detection", "facial recognition", "pose estimation",
    "nlp", "natural language", "language model", "llm", "gpt", "chatgpt", "bert",
    "transformer", "text classification", "sentiment", "rag", "embeddings",
    "ai agent", "agentic", "multi-agent", "langchain", "autogen",
    "stable diffusion", "generative ai", "image generation", "diffusion",
    "speech recognition", "text to speech", "whisper",
    "ocr", "document ai", "document extraction",
    "recommendation system", "collaborative filtering",
    "tensorflow", "pytorch", "scikit", "hugging face",
    "mlops", "model deployment", "model training", "fine tuning", "fine-tuning",
    "data science", "predictive model", "ai model", "ml model",
}

# Broader terms allowed only when they appear in both title AND description
BODY_TERMS: set[str] = {
    "machine learning", "deep learning", "neural", "artificial intelligence",
    "computer vision", "object detection", "nlp", "natural language",
    "language model", "llm", "gpt", "chatbot ai", "ai chatbot",
    "bert", "transformer model", "text classification", "sentiment analysis",
    "ai agent", "langchain", "vector database", "embeddings", "rag pipeline",
    "stable diffusion", "generative ai", "image generation",
    "speech recognition", "voice ai", "whisper",
    "ocr", "document ai",
    "recommendation system", "tensorflow", "pytorch", "scikit-learn",
    "hugging face", "mlops", "model deployment", "fine tuning",
    "data science project", "predictive analytics", "ai model", "ml model",
}

# Hard-reject: if the title contains any of these and has no AI term, drop it
NEGATIVE_TITLE_TERMS: set[str] = {
    "sales", "clean", "cleaning", "aviation", "sculpture", "casting",
    "transcrib", "copy", "translate", "translation", "seo", "logo",
    "accounting", "bookkeeping", "legal", "immigration", "visa",
    "recruitment", "hr ", "human resource", "closer ", "marketing rep",
    "business development rep", "cold call", "lead generation",
    "social media post", "content writ",
}


def is_ai_relevant(title: str, description: str) -> bool:
    """
    Two-stage relevance check:
      Stage 1 — fast: does the title alone contain a strong AI term?
      Stage 2 — slower: does the combined title+description contain body terms?
    Either stage passing = relevant. Any negative title term = rejected.
    """
    t = title.lower()
    d = description.lower()

    # Hard reject on obviously non-AI titles
    if any(neg in t for neg in NEGATIVE_TITLE_TERMS):
        # Only reject if no strong AI term in title overrides it
        if not any(ai in t for ai in TITLE_MUST_MATCH):
            return False

    # Stage 1: strong AI term in title → accept immediately
    if any(term in t for term in TITLE_MUST_MATCH):
        return True

    # Stage 2: body terms must appear in BOTH title and description
    td = t + " " + d
    matches = [term for term in BODY_TERMS if term in td]
    # Require at least 2 distinct body matches to reduce false positives
    return len(matches) >= 2


# ─── DATA MODEL ───────────────────────────────────────────────────────────────

@dataclass
class ProjectListing:
    title:        str
    platform:     str
    description:  str
    budget_min:   float | None
    budget_max:   float | None
    currency:     str | None
    skills:       list[str]
    bid_count:    int | None    # Freelancer only — key competition signal
    posted_date:  str | None
    url:          str
    source_type:  str           # "freelance" | "project_board"
    keyword:      str = ""      # which keyword surfaced this result


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def extract_keywords(query: str) -> list[str]:
    """
    Map a plain-English user query to a deduplicated list of search keywords.
    Falls back to the raw query if nothing in the map matches.
    """
    query_lower = query.lower()
    matched: list[str] = []

    for category, keywords in KEYWORD_MAP.items():
        if category in query_lower or any(kw.lower() in query_lower for kw in keywords):
            matched.extend(keywords)

    seen: set[str] = set()
    unique = []
    for kw in matched:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique if unique else [query]


def deduplicate(listings: list[ProjectListing]) -> list[ProjectListing]:
    """Remove duplicate listings by URL."""
    seen: set[str] = set()
    unique = []
    for listing in listings:
        if listing.url not in seen:
            seen.add(listing.url)
            unique.append(listing)
    return unique


def sort_listings(listings: list[ProjectListing]) -> list[ProjectListing]:
    """
    Sort by bid_count ascending (fewer bids = better opportunity),
    then budget_max descending. Non-Freelancer listings (no bid count) go last.
    """
    return sorted(
        listings,
        key=lambda x: (
            x.bid_count if x.bid_count is not None else 999,
            -(x.budget_max or 0),
        ),
    )


def parse_budget_string(text: str) -> tuple[float | None, float | None, str | None]:
    """Parse budget strings like '£50', '$100-$200', '€500+' into (min, max, currency)."""
    import re
    if not text:
        return None, None, None

    symbol_map = {"£": "GBP", "$": "USD", "€": "EUR"}
    currency = None
    for sym, code in symbol_map.items():
        if sym in text:
            currency = code
            break

    nums = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+", text) if n]
    if not nums:
        return None, None, currency
    if len(nums) == 1:
        return nums[0], nums[0], currency
    return min(nums), max(nums), currency


def http_get(client: httpx.Client, url: str, **kwargs) -> httpx.Response | None:
    """Shared GET with timeout and graceful error handling."""
    try:
        resp = client.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as e:
        print(f"    [HTTP {e.response.status_code}] {url}")
        return None
    except httpx.RequestError as e:
        print(f"    [Request error] {url} — {e}")
        return None


# ─── SCRAPER: FREELANCER.COM ──────────────────────────────────────────────────

def scrape_freelancer(client: httpx.Client, keyword: str) -> list[ProjectListing]:
    """
    Calls Freelancer's public active-project search endpoint (no auth required).
    Applies strict two-stage AI relevance filter after fetching.
    """
    url = "https://www.freelancer.com/api/projects/0.1/projects/active/"
    params = {
        "query":            keyword,
        "job_details":      "true",
        "full_description": "true",
        "offset":           0,
        "limit":            FREELANCER_LIMIT,
        "sort_field":       "time_updated",
    }
    headers = {**HEADERS, "Accept": "application/json", "Freelancer-OAuth-V1": ""}

    resp = http_get(client, url, params=params, headers=headers)
    if not resp:
        return []

    try:
        data = resp.json()
    except Exception:
        print("    [Freelancer] Failed to parse JSON")
        return []

    projects = data.get("result", {}).get("projects", [])
    results: list[ProjectListing] = []

    for p in projects:
        title  = (p.get("title") or "").strip()
        desc   = (p.get("description") or "")[:500].strip()

        # Two-stage AI relevance check — rejects cleaning jobs, aviation, sales, etc.
        if not is_ai_relevant(title, desc):
            continue

        budget    = p.get("budget", {}) or {}
        currency  = (p.get("currency") or {}).get("code")
        skills    = [j.get("name", "") for j in (p.get("jobs") or []) if j.get("name")]
        seo_url   = p.get("seo_url", "")
        bid_count = (p.get("bid_stats") or {}).get("bid_count")
        ts        = p.get("time_submitted")
        posted    = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else None

        results.append(ProjectListing(
            title       = title,
            platform    = "Freelancer.com",
            description = desc,
            budget_min  = budget.get("minimum"),
            budget_max  = budget.get("maximum"),
            currency    = currency,
            skills      = skills,
            bid_count   = bid_count,
            posted_date = posted,
            url         = f"https://www.freelancer.com/projects/{seo_url}" if seo_url else "https://www.freelancer.com",
            source_type = "freelance",
            keyword     = keyword,
        ))

    return results


# ─── SCRAPER: PEOPLEPERHOUR ───────────────────────────────────────────────────

def scrape_peopleperhour(client: httpx.Client, keyword: str) -> list[ProjectListing]:
    """
    Scrapes PeoplePerHour project listings (static HTML).
    Applies AI relevance filter.
    """
    url    = f"{PPH_BASE_URL}/freelance-jobs"
    params = {"keyword": keyword, "type": "projects"}

    resp = http_get(client, url, params=params, headers=HEADERS)
    if not resp:
        return []

    tree    = HTMLParser(resp.text)
    results: list[ProjectListing] = []

    cards = tree.css("li.feed-item") or tree.css("[data-test='job-item']")

    for card in cards:
        title_node = (
            card.css_first("h2 a") or
            card.css_first(".feed-item-title a") or
            card.css_first("a.job-title")
        )
        if not title_node:
            continue
        title = title_node.text(strip=True)
        href  = title_node.attrs.get("href", "")
        link  = href if href.startswith("http") else f"{PPH_BASE_URL}{href}"

        desc_node = (
            card.css_first(".feed-item-description") or
            card.css_first(".job-description") or
            card.css_first("p")
        )
        description = desc_node.text(strip=True)[:400] if desc_node else ""

        if not is_ai_relevant(title, description):
            continue

        budget_node = (
            card.css_first(".budget") or
            card.css_first("[data-test='budget']") or
            card.css_first(".price")
        )
        budget_text = budget_node.text(strip=True) if budget_node else ""
        budget_min, budget_max, currency = parse_budget_string(budget_text)

        skill_nodes = card.css(".skill-tag") or card.css(".tag") or card.css(".job-tag")
        skills = [s.text(strip=True) for s in skill_nodes if s.text(strip=True)]

        date_node = card.css_first("time") or card.css_first(".posted-date")
        posted    = (date_node.attrs.get("datetime") or date_node.text(strip=True)) if date_node else None

        results.append(ProjectListing(
            title       = title,
            platform    = "PeoplePerHour",
            description = description,
            budget_min  = budget_min,
            budget_max  = budget_max,
            currency    = currency,
            skills      = skills,
            bid_count   = None,
            posted_date = posted,
            url         = link,
            source_type = "freelance",
            keyword     = keyword,
        ))

    return results


# ─── SCRAPER: TOPTAL ──────────────────────────────────────────────────────────

def scrape_toptal(client: httpx.Client, keyword: str) -> list[ProjectListing]:
    """
    Scrapes Toptal's public job/project listings (static HTML).
    Toptal carries larger, higher-value engagements from vetted companies.
    """
    url    = f"{TOPTAL_BASE_URL}/jobs"
    params = {"q": keyword}

    resp = http_get(client, url, params=params, headers=HEADERS)
    if not resp:
        return []

    tree    = HTMLParser(resp.text)
    results: list[ProjectListing] = []

    cards = (
        tree.css("li.job-listing") or
        tree.css("[data-test='job-card']") or
        tree.css(".job-item") or
        tree.css("article")
    )

    for card in cards:
        title_node = (
            card.css_first("h2 a") or
            card.css_first("h3 a") or
            card.css_first(".job-title a") or
            card.css_first("a")
        )
        if not title_node:
            continue
        title = title_node.text(strip=True)
        if not title:
            continue

        href = title_node.attrs.get("href", "")
        link = href if href.startswith("http") else f"{TOPTAL_BASE_URL}{href}"

        desc_node   = card.css_first(".job-description") or card.css_first("p") or card.css_first(".description")
        description = desc_node.text(strip=True)[:400] if desc_node else ""

        if not is_ai_relevant(title, description):
            continue

        comp_node  = card.css_first(".compensation") or card.css_first(".salary") or card.css_first("[data-test='compensation']")
        comp_text  = comp_node.text(strip=True) if comp_node else ""
        budget_min, budget_max, currency = parse_budget_string(comp_text)

        skill_nodes = card.css(".skill") or card.css(".tag") or card.css(".badge")
        skills      = [s.text(strip=True) for s in skill_nodes if s.text(strip=True)]

        date_node = card.css_first("time") or card.css_first(".date")
        posted    = (date_node.attrs.get("datetime") or date_node.text(strip=True)) if date_node else None

        results.append(ProjectListing(
            title       = title,
            platform    = "Toptal",
            description = description,
            budget_min  = budget_min,
            budget_max  = budget_max,
            currency    = currency or "USD",
            skills      = skills,
            bid_count   = None,
            posted_date = posted,
            url         = link,
            source_type = "freelance",
            keyword     = keyword,
        ))

    return results


# ─── SCRAPER: GURU.COM ────────────────────────────────────────────────────────

def scrape_guru(client: httpx.Client, keyword: str) -> list[ProjectListing]:
    """
    Scrapes Guru.com's public job/project listings (static HTML).
    Guru is a freelance marketplace with genuine client-posted projects,
    budgets, and skill tags — good complement to Freelancer.com.
    URL pattern: https://www.guru.com/work/search/?q=<keyword>
    """
    url    = f"{GURU_BASE_URL}/work/search/"
    params = {"q": keyword}

    resp = http_get(client, url, params=params, headers=HEADERS)
    if not resp:
        return []

    tree    = HTMLParser(resp.text)
    results: list[ProjectListing] = []

    # Guru project cards sit in <div class="serviceItem"> or similar
    cards = (
        tree.css(".serviceItem") or
        tree.css(".job-item") or
        tree.css("[class*='service-item']") or
        tree.css("li.listed-job")
    )

    for card in cards:
        title_node = (
            card.css_first("h2 a") or
            card.css_first("h3 a") or
            card.css_first(".jobTitle a") or
            card.css_first("a.jobTitle") or
            card.css_first("a")
        )
        if not title_node:
            continue
        title = title_node.text(strip=True)
        if not title:
            continue

        href = title_node.attrs.get("href", "")
        link = href if href.startswith("http") else f"{GURU_BASE_URL}{href}"

        desc_node   = card.css_first(".jobDescription") or card.css_first("p") or card.css_first(".description")
        description = desc_node.text(strip=True)[:400] if desc_node else ""

        if not is_ai_relevant(title, description):
            continue

        budget_node = (
            card.css_first(".jobBudget") or
            card.css_first(".budget") or
            card.css_first("[class*='budget']")
        )
        budget_text = budget_node.text(strip=True) if budget_node else ""
        budget_min, budget_max, currency = parse_budget_string(budget_text)

        skill_nodes = card.css(".skill") or card.css(".tag") or card.css("[class*='skill']")
        skills      = [s.text(strip=True) for s in skill_nodes if s.text(strip=True)]

        date_node = card.css_first("time") or card.css_first(".jobPosted") or card.css_first(".date")
        posted    = (date_node.attrs.get("datetime") or date_node.text(strip=True)) if date_node else None

        results.append(ProjectListing(
            title       = title,
            platform    = "Guru.com",
            description = description,
            budget_min  = budget_min,
            budget_max  = budget_max,
            currency    = currency or "USD",
            skills      = skills,
            bid_count   = None,
            posted_date = posted,
            url         = link,
            source_type = "project_board",
            keyword     = keyword,
        ))

    return results


# ─── OUTPUT ───────────────────────────────────────────────────────────────────

def print_listing(i: int, p: ProjectListing) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  #{i}  {p.title}")
    print(f"{sep}")
    print(f"  Platform:    {p.platform}  [{p.source_type}]")

    if p.budget_min and p.budget_max:
        curr = p.currency or ""
        budget_str = (
            f"{curr} {p.budget_min:,.0f}"
            if p.budget_min == p.budget_max
            else f"{curr} {p.budget_min:,.0f} – {p.budget_max:,.0f}"
        )
    else:
        budget_str = "Not specified"
    print(f"  Budget:      {budget_str}")

    if p.bid_count is not None:
        signal = (
            "🟢 Low competition"  if p.bid_count < 10 else
            "🟡 Medium"           if p.bid_count < 25 else
            "🔴 High competition"
        )
        print(f"  Bids so far: {p.bid_count}  {signal}")

    if p.skills:
        skills_str = ", ".join(p.skills[:7])
        if len(p.skills) > 7:
            skills_str += f" (+{len(p.skills) - 7} more)"
        print(f"  Skills:      {skills_str}")

    if p.description:
        desc = p.description[:280] + ("..." if len(p.description) > 280 else "")
        print(f"  Description: {desc}")

    print(f"  Posted:      {p.posted_date or 'Unknown'}")
    print(f"  Link:        {p.url}")


def print_results(listings: list[ProjectListing]) -> None:
    if not listings:
        print("\n  No AI/ML projects found. Try a more specific query.")
        return

    counts = Counter(p.platform for p in listings)
    print("\n" + "─" * 65)
    print(f"  Found {len(listings)} AI/ML projects:")
    for platform, count in counts.most_common():
        print(f"    • {platform}: {count}")
    print("─" * 65)

    for i, listing in enumerate(listings, 1):
        print_listing(i, listing)

    print(f"\n{'─' * 65}")
    print(f"  Total: {len(listings)} projects | {datetime.now().strftime('%H:%M:%S')}")
    print("─" * 65)


ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
]


def next_filename(results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    existing = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    n = len(existing)
    if n < len(ORDINALS):
        return os.path.join(results_dir, f"{ORDINALS[n]}.json")
    return os.path.join(results_dir, f"run_{n + 1}.json")


def save_results(listings: list[ProjectListing], query: str) -> str:
    path = next_filename(RESULTS_DIR)
    data = {
        "scraped_at":    datetime.utcnow().isoformat(),
        "query":         query,
        "total_results": len(listings),
        "results":       [asdict(p) for p in listings],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved {len(listings)} results → {path}")
    return path


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(query: str) -> list[ProjectListing]:
    keywords = extract_keywords(query)

    print(f"\n  Query:    \"{query}\"")
    print(f"  Keywords: {', '.join(keywords[:5])}" + (f" (+{len(keywords)-5} more)" if len(keywords) > 5 else ""))
    print(f"  Sources:  Freelancer.com, PeoplePerHour, Toptal, Guru.com\n")

    all_listings: list[ProjectListing] = []

    with httpx.Client(follow_redirects=True) as client:

        # ── Freelancer.com ──────────────────────────────────────────────────
        print("  [1/4] Freelancer.com ...")
        for kw in keywords[:4]:
            results = scrape_freelancer(client, kw)
            print(f"        → '{kw}': {len(results)} AI projects")
            all_listings.extend(results)
            time.sleep(REQUEST_DELAY)

        # ── PeoplePerHour ───────────────────────────────────────────────────
        print("\n  [2/4] PeoplePerHour ...")
        for kw in keywords[:3]:
            results = scrape_peopleperhour(client, kw)
            print(f"        → '{kw}': {len(results)} AI projects")
            all_listings.extend(results)
            time.sleep(REQUEST_DELAY)

        # ── Toptal ──────────────────────────────────────────────────────────
        print("\n  [3/4] Toptal ...")
        for kw in keywords[:3]:
            results = scrape_toptal(client, kw)
            print(f"        → '{kw}': {len(results)} AI projects")
            all_listings.extend(results)
            time.sleep(REQUEST_DELAY)

        # ── Guru.com ────────────────────────────────────────────────────────
        print("\n  [4/4] Guru.com ...")
        for kw in keywords[:4]:
            results = scrape_guru(client, kw)
            print(f"        → '{kw}': {len(results)} AI projects")
            all_listings.extend(results)
            time.sleep(REQUEST_DELAY)

    unique  = deduplicate(all_listings)
    sorted_ = sort_listings(unique)

    print_results(sorted_)
    save_results(sorted_, query)

    return sorted_


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape AI project leads from Freelancer, PeoplePerHour, Toptal, and Guru.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "computer vision"
  python main.py --query "LLM RAG pipeline"
  python main.py --query "machine learning"

Results are auto-saved to results/first.json, second.json, ...
        """
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query. Prompted interactively if not provided.",
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        print("\n  AI Project Leads Scraper")
        print("  ─────────────────────────────────────────────────────────────")
        print("  Sources: Freelancer.com · PeoplePerHour · Toptal · Guru.com")
        print("  Examples: 'computer vision', 'NLP chatbot', 'machine learning'")
        query = input("\n  Enter your search query: ").strip()
        if not query:
            print("  No query entered. Exiting.")
            raise SystemExit(0)

    run(query=query)
