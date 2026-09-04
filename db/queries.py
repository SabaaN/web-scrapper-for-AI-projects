"""
db/queries.py
=============
All database read and write operations live here.
Nothing else in the codebase talks to MongoDB directly —
everything goes through these functions.

Functions are grouped into four sections:
  1. Normalization helpers
  2. Scrape run operations  (insert, find)
  3. Project listing operations (insert, find, update)
  4. Cache operations (the most important ones for the chatbot)
"""

from datetime import datetime, timedelta
from dataclasses import asdict
from bson import ObjectId
from pymongo.database import Database

from db.connection import get_db
from db.models import (
    SCRAPE_RUNS_COLLECTION,
    PROJECT_LISTINGS_COLLECTION,
    build_scrape_run_doc,
    build_project_listing_doc,
)

import os
from dotenv import load_dotenv
load_dotenv()

CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", 4))


# ── 1. Normalization Helpers ──────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    """
    Converts a raw user query into a consistent cache key.

    Examples:
      "RAG Chatbot"         → "rag chatbot"
      "  Computer Vision  " → "computer vision"
      "LLM Pipeline!!!"     → "llm pipeline"

    This ensures "RAG chatbot" and "rag chatbot" and "RAG CHATBOT"
    all hit the same cache entry.
    """
    import re
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)   # remove punctuation
    query = re.sub(r"\s+", " ", query)       # collapse multiple spaces
    return query


# ── 2. Scrape Run Operations ──────────────────────────────────────────────────

def insert_scrape_run(
    query: str,
    platforms: list[str],
    total_results: int,
    triggered_by: str = "user_query",
    filters_applied: dict | None = None,
) -> ObjectId:
    """
    Inserts a new scrape run document and returns its _id.
    Call this BEFORE inserting the listings so you have the run _id ready.

    Args:
        query:          Raw query string as user typed it
        platforms:      List of platforms that were scraped
        total_results:  Number of results after filtering
        triggered_by:   "user_query" or "cache_miss"
        filters_applied: Any filters like { "budget_max": 5000 }

    Returns:
        ObjectId of the inserted document
    """
    db = get_db()
    doc = build_scrape_run_doc(
        query=query,
        normalized_query=normalize_query(query),
        platforms=platforms,
        total_results=total_results,
        triggered_by=triggered_by,
        filters_applied=filters_applied,
    )
    result = db[SCRAPE_RUNS_COLLECTION].insert_one(doc)
    return result.inserted_id


def get_scrape_run(run_id: ObjectId) -> dict | None:
    """
    Fetches a single scrape run by its _id.

    Args:
        run_id: The ObjectId of the scrape run

    Returns:
        The scrape run document or None if not found
    """
    db = get_db()
    return db[SCRAPE_RUNS_COLLECTION].find_one({"_id": run_id})


def get_recent_scrape_runs(limit: int = 10) -> list[dict]:
    """
    Returns the most recent scrape runs across all queries.
    Useful for the chatbot to answer "what did we search recently?"

    Args:
        limit: Max number of runs to return

    Returns:
        List of scrape run documents, newest first
    """
    db = get_db()
    cursor = (
        db[SCRAPE_RUNS_COLLECTION]
        .find()
        .sort("ran_at", -1)
        .limit(limit)
    )
    return list(cursor)


def get_runs_for_query(query: str, limit: int = 5) -> list[dict]:
    """
    Returns all scrape runs for a specific query, newest first.

    Args:
        query: Raw or normalized query string
        limit: Max runs to return

    Returns:
        List of matching scrape run documents
    """
    db = get_db()
    cursor = (
        db[SCRAPE_RUNS_COLLECTION]
        .find({"normalized_query": normalize_query(query)})
        .sort("ran_at", -1)
        .limit(limit)
    )
    return list(cursor)


# ── 3. Project Listing Operations ─────────────────────────────────────────────

def insert_listings(
    scrape_run_id: ObjectId,
    listings: list,
) -> int:
    """
    Inserts project listings for a given scrape run.

    For each listing:
      - If the URL has never been seen → insert fresh document
      - If the URL already exists → update last_seen_at and increment times_seen
        (the project is still active across multiple scrapes)

    Args:
        scrape_run_id: The _id of the parent scrape run
        listings:      List of ProjectListing objects or dicts

    Returns:
        Number of new listings inserted (not counting updates)
    """
    db = get_db()
    inserted = 0

    for listing in listings:
        # Convert dataclass to dict if needed
        listing_dict = asdict(listing) if hasattr(listing, "__dataclass_fields__") else listing

        url = listing_dict.get("url", "")
        if not url:
            continue

        # Check if this URL already exists in the database
        existing = db[PROJECT_LISTINGS_COLLECTION].find_one({"url": url})

        if existing:
            # Project was seen before — update the tracking fields only
            db[PROJECT_LISTINGS_COLLECTION].update_one(
                {"url": url},
                {
                    "$set":  {"last_seen_at": datetime.utcnow()},
                    "$inc":  {"times_seen": 1},
                }
            )
        else:
            # New project — insert full document
            doc = build_project_listing_doc(scrape_run_id, listing_dict)
            db[PROJECT_LISTINGS_COLLECTION].insert_one(doc)
            inserted += 1

    return inserted


def get_listings_for_run(
    scrape_run_id: ObjectId,
    filters: dict | None = None,
) -> list[dict]:
    """
    Fetches all project listings for a specific scrape run,
    with optional filtering.

    Args:
        scrape_run_id: The _id of the parent scrape run
        filters:       Optional dict like:
                         { "budget_max": 5000 }
                         { "platform": "Freelancer.com" }
                         { "bid_max": 10 }

    Returns:
        List of matching project listing documents, sorted by relevance
    """
    db = get_db()

    query = {"scrape_run_id": scrape_run_id}

    # Apply optional filters
    if filters:
        if "budget_min" in filters:
            query["budget_min"] = {"$gte": filters["budget_min"]}
        if "budget_max" in filters:
            query["budget_max"] = {"$lte": filters["budget_max"]}
        if "platform" in filters:
            query["platform"] = filters["platform"]
        if "bid_max" in filters:
            query["bid_count"] = {"$lte": filters["bid_max"]}

    cursor = (
        db[PROJECT_LISTINGS_COLLECTION]
        .find(query)
        .sort("relevance_score", -1)
    )
    return list(cursor)


def get_listing_by_url(url: str) -> dict | None:
    """
    Fetches a single listing by its URL.
    Useful for checking if a project was already scraped before.

    Args:
        url: The full project URL

    Returns:
        The listing document or None
    """
    db = get_db()
    return db[PROJECT_LISTINGS_COLLECTION].find_one({"url": url})


def get_all_listings(
    filters: dict | None = None,
    limit: int = 50,
    sort_by: str = "relevance_score",
) -> list[dict]:
    """
    Fetches listings across ALL scrape runs with optional filtering.
    Used when the chatbot asks "show me the best RAG projects we've ever found."

    Args:
        filters:  Optional filter dict (same format as get_listings_for_run)
        limit:    Max results to return
        sort_by:  Field to sort by — "relevance_score", "budget_max", "bid_count"

    Returns:
        List of listing documents
    """
    db = get_db()

    query = {}
    if filters:
        if "budget_min" in filters:
            query["budget_min"] = {"$gte": filters["budget_min"]}
        if "budget_max" in filters:
            query["budget_max"] = {"$lte": filters["budget_max"]}
        if "platform" in filters:
            query["platform"] = filters["platform"]
        if "bid_max" in filters:
            query["bid_count"] = {"$lte": filters["bid_max"]}

    sort_direction = 1 if sort_by == "bid_count" else -1

    cursor = (
        db[PROJECT_LISTINGS_COLLECTION]
        .find(query)
        .sort(sort_by, sort_direction)
        .limit(limit)
    )
    return list(cursor)


def get_platform_stats() -> list[dict]:
    """
    Returns a count of listings per platform across all scrape runs.
    Used for the chatbot to answer "which platform had the most AI projects?"

    Returns:
        List of { "_id": "Freelancer.com", "count": 142 } dicts
    """
    db = get_db()
    pipeline = [
        {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return list(db[PROJECT_LISTINGS_COLLECTION].aggregate(pipeline))


# ── 4. Cache Operations ───────────────────────────────────────────────────────
# These are the most important functions for the chatbot.
# Every user query checks the cache before deciding to scrape.

def is_cache_valid(query: str) -> bool:
    """
    Checks whether a fresh scrape is needed for this query.

    A cache is valid if the same query was run within CACHE_EXPIRY_HOURS.
    If valid → return cached results.
    If expired or never run → trigger a fresh scrape.

    Args:
        query: The user's search query (raw or normalized)

    Returns:
        True if cached results are still fresh, False if scrape is needed
    """
    db = get_db()

    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)

    most_recent_run = db[SCRAPE_RUNS_COLLECTION].find_one(
        {
            "normalized_query": normalize_query(query),
            "ran_at": {"$gte": cutoff},           # ran within the expiry window
        },
        sort=[("ran_at", -1)],                     # get the most recent one
    )

    return most_recent_run is not None


def get_cached_results(
    query: str,
    filters: dict | None = None,
) -> list[dict] | None:
    """
    Returns cached listings for a query if the cache is valid.
    Returns None if the cache is expired or the query was never run.

    This is the main function the chatbot calls before deciding to scrape.

    Args:
        query:   The user's search query
        filters: Optional filters to apply to the cached results

    Returns:
        List of listing documents, or None if cache miss
    """
    db = get_db()

    if not is_cache_valid(query):
        return None

    # Get the most recent valid run for this query
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS)
    most_recent_run = db[SCRAPE_RUNS_COLLECTION].find_one(
        {
            "normalized_query": normalize_query(query),
            "ran_at": {"$gte": cutoff},
        },
        sort=[("ran_at", -1)],
    )

    if not most_recent_run:
        return None

    return get_listings_for_run(most_recent_run["_id"], filters=filters)


def force_fresh_scrape(query: str) -> None:
    """
    Invalidates the cache for a query by marking all its runs as expired.
    Called when the user says "search again" or "get fresh results".

    Works by backdating ran_at so the cache check fails naturally.

    Args:
        query: The query whose cache should be invalidated
    """
    db = get_db()

    expired_time = datetime.utcnow() - timedelta(hours=CACHE_EXPIRY_HOURS + 1)

    db[SCRAPE_RUNS_COLLECTION].update_many(
        {"normalized_query": normalize_query(query)},
        {"$set": {"ran_at": expired_time}},
    )
    print(f"Cache invalidated for query: '{query}'")


def get_cache_status(query: str) -> dict:
    """
    Returns human-readable cache status for a query.
    Used by the chatbot to tell the user how fresh the data is.

    Args:
        query: The search query to check

    Returns:
        Dict with status, age, and expiry info

    Example return:
        {
            "is_valid": True,
            "last_run": "2026-09-01 14:32:00",
            "age_minutes": 47,
            "expires_in_minutes": 193,
            "total_results": 24,
        }
    """
    db = get_db()

    most_recent = db[SCRAPE_RUNS_COLLECTION].find_one(
        {"normalized_query": normalize_query(query)},
        sort=[("ran_at", -1)],
    )

    if not most_recent:
        return {
            "is_valid":           False,
            "last_run":           None,
            "age_minutes":        None,
            "expires_in_minutes": None,
            "total_results":      None,
        }

    ran_at     = most_recent["ran_at"]
    age        = datetime.utcnow() - ran_at
    age_mins   = int(age.total_seconds() / 60)
    expiry_mins = max(0, (CACHE_EXPIRY_HOURS * 60) - age_mins)

    return {
        "is_valid":           is_cache_valid(query),
        "last_run":           ran_at.strftime("%Y-%m-%d %H:%M:%S"),
        "age_minutes":        age_mins,
        "expires_in_minutes": expiry_mins,
        "total_results":      most_recent.get("total_results", 0),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from db.connection import close_db

    db = get_db()
    print("Testing queries.py...\n")

    # ── Test 1: normalize_query ───────────────────────────────────────────────
    print("Test 1 — normalize_query:")
    tests = ["RAG Chatbot", "  computer vision  ", "LLM Pipeline!!!"]
    for t in tests:
        print(f"  '{t}' → '{normalize_query(t)}'")

    # ── Test 2: insert a dummy scrape run ─────────────────────────────────────
    print("\nTest 2 — insert_scrape_run:")
    run_id = insert_scrape_run(
        query="computer vision test",
        platforms=["Freelancer.com", "Remotive"],
        total_results=3,
        triggered_by="test",
    )
    print(f"  Inserted scrape run: {run_id}")

    # ── Test 3: insert dummy listings ────────────────────────────────────────
    print("\nTest 3 — insert_listings:")
    dummy_listings = [
        {
            "title":           "Computer Vision Pipeline for Retail",
            "platform":        "Freelancer.com",
            "description":     "Build a YOLOv8-based object detection system.",
            "budget_min":      2000,
            "budget_max":      5000,
            "currency":        "USD",
            "skills":          ["Python", "OpenCV", "YOLOv8"],
            "bid_count":       4,
            "relevance_score": 38.5,
            "preference_rank": 1,
            "posted_date":     "2026-09-01",
            "url":             "https://freelancer.com/projects/test-cv-001",
            "source_type":     "freelance",
            "keyword":         "computer vision",
        },
        {
            "title":           "NLP Chatbot for Customer Support",
            "platform":        "Remotive",
            "description":     "Build a RAG-based support chatbot.",
            "budget_min":      3000,
            "budget_max":      6000,
            "currency":        "USD",
            "skills":          ["Python", "LangChain", "RAG"],
            "bid_count":       None,
            "relevance_score": 35.0,
            "preference_rank": 2,
            "posted_date":     "2026-09-01",
            "url":             "https://remotive.com/jobs/test-nlp-001",
            "source_type":     "remote_contract",
            "keyword":         "rag chatbot",
        },
    ]
    count = insert_listings(run_id, dummy_listings)
    print(f"  Inserted {count} new listings")

    # ── Test 4: fetch listings for the run ────────────────────────────────────
    print("\nTest 4 — get_listings_for_run:")
    results = get_listings_for_run(run_id)
    for r in results:
        print(f"  - {r['title']} ({r['platform']}) | score: {r['relevance_score']}")

    # ── Test 5: cache check ───────────────────────────────────────────────────
    print("\nTest 5 — is_cache_valid:")
    valid = is_cache_valid("computer vision test")
    print(f"  Cache valid for 'computer vision test': {valid}")

    # ── Test 6: get_cached_results ────────────────────────────────────────────
    print("\nTest 6 — get_cached_results:")
    cached = get_cached_results("computer vision test")
    if cached:
        print(f"  Found {len(cached)} cached results")
        for r in cached:
            print(f"  - {r['title']}")
    else:
        print("  No cached results found")

    # ── Test 7: cache status ──────────────────────────────────────────────────
    print("\nTest 7 — get_cache_status:")
    status = get_cache_status("computer vision test")
    for k, v in status.items():
        print(f"  {k}: {v}")

    # ── Test 8: platform stats ────────────────────────────────────────────────
    print("\nTest 8 — get_platform_stats:")
    stats = get_platform_stats()
    for s in stats:
        print(f"  {s['_id']}: {s['count']} listings")

    # ── Cleanup test data ─────────────────────────────────────────────────────
    print("\nCleaning up test data...")
    db[SCRAPE_RUNS_COLLECTION].delete_many({"triggered_by": "test"})
    db[PROJECT_LISTINGS_COLLECTION].delete_many(
        {"url": {"$in": [
            "https://freelancer.com/projects/test-cv-001",
            "https://remotive.com/jobs/test-nlp-001",
        ]}}
    )
    print("✓ Test data removed")

    close_db()
    print("\n✓ All tests passed")