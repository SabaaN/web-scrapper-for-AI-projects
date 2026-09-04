"""
core/cache.py
=============
Cache layer between the chatbot and the scraper.

Every time the user sends a search query, this module decides:
  - Is there a fresh result in MongoDB for this query?  → return it
  - Is the cache stale or missing?                      → tell caller to scrape

All the actual database operations live in db/queries.py.
This file is purely the decision logic — thin by design.

The one public function the chatbot calls is:
    get_or_invalidate(query, filters, force_fresh)
"""

import os
from dataclasses import asdict
from dotenv import load_dotenv

from db.queries import (
    is_cache_valid,
    get_cached_results,
    get_cache_status,
    force_fresh_scrape,
    normalize_query,
)

load_dotenv()

CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", 4))


# ─── Cache Result Container ───────────────────────────────────────────────────

class CacheResult:
    """
    Returned by get_or_invalidate().
    Tells the caller exactly what happened and what to do next.

    Attributes:
        hit       : True  → fresh data found, use `listings`
                    False → cache miss, caller must run the scraper
        listings  : list of project listing dicts from MongoDB
                    (empty list if hit=False)
        status    : human-readable dict describing cache age and freshness
        query     : the normalized query that was checked
        from_cache: always mirrors `hit` — exists for readable log messages
    """

    def __init__(
        self,
        hit: bool,
        listings: list[dict],
        status: dict,
        query: str,
    ):
        self.hit        = hit
        self.listings   = listings
        self.status     = status
        self.query      = query
        self.from_cache = hit

    def summary(self) -> str:
        """
        One-line human readable summary — used in chatbot responses.

        Examples:
            "Cache hit — 24 results (47 min old, refreshes in 193 min)"
            "Cache miss — no data for 'rag chatbot', scraping now..."
            "Cache invalidated — forcing fresh scrape for 'computer vision'"
        """
        if not self.hit:
            return f"Cache miss — no fresh data for '{self.query}', scraping now..."

        age  = self.status.get("age_minutes", 0)
        exp  = self.status.get("expires_in_minutes", 0)
        n    = len(self.listings)
        return (
            f"Cache hit — {n} results "
            f"({age} min old, refreshes in {exp} min)"
        )

    def __repr__(self) -> str:
        return (
            f"CacheResult(hit={self.hit}, "
            f"listings={len(self.listings)}, "
            f"query='{self.query}')"
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def get_or_invalidate(
    query: str,
    filters: dict | None = None,
    force_fresh: bool = False,
) -> CacheResult:
    """
    The single function the chatbot calls before deciding to scrape.

    Flow:
        1. If force_fresh=True  → invalidate cache, return miss
        2. If cache is valid    → return cached listings
        3. If cache is stale    → return miss (caller runs scraper)

    Args:
        query:       Raw user query string ("RAG chatbot", "computer vision")
        filters:     Optional dict to filter cached results:
                       { "budget_max": 5000 }
                       { "platform": "Freelancer.com" }
                       { "bid_max": 10 }
                     Filters apply to cached results only — the scraper
                     always returns unfiltered results and stores them in full.
        force_fresh: If True, bypass cache entirely and force a new scrape.
                     Set when user says "search again" or "get fresh results".

    Returns:
        CacheResult with hit=True  (use .listings) or
                         hit=False (run scraper, then call store_results)
    """
    normalized = normalize_query(query)
    status     = get_cache_status(query)

    # ── Step 1: Force fresh requested ─────────────────────────────────────────
    if force_fresh:
        force_fresh_scrape(query)
        print(f"  [Cache] Invalidated for '{normalized}' — fresh scrape requested")
        return CacheResult(
            hit=False,
            listings=[],
            status=status,
            query=normalized,
        )

    # ── Step 2: Cache hit ──────────────────────────────────────────────────────
    if is_cache_valid(query):
        listings = get_cached_results(query, filters=filters)

        if listings:
            print(f"  [Cache] HIT — '{normalized}' ({len(listings)} results, "
                  f"{status.get('age_minutes', 0)} min old)")
            return CacheResult(
                hit=True,
                listings=listings,
                status=status,
                query=normalized,
            )

    # ── Step 3: Cache miss ─────────────────────────────────────────────────────
    print(f"  [Cache] MISS — '{normalized}' not in cache or expired")
    return CacheResult(
        hit=False,
        listings=[],
        status=status,
        query=normalized,
    )


def store_results(
    query: str,
    listings: list,
    filters_applied: dict | None = None,
) -> None:
    """
    Stores fresh scrape results into MongoDB after a cache miss.
    Called by the chatbot immediately after scrape() returns.

    Args:
        query:           The original search query
        listings:        List of ProjectListing objects returned by scrape()
        filters_applied: Any filters the user specified (stored for reference)
    """
    from db.queries import insert_scrape_run, insert_listings

    platforms = list({p.platform for p in listings})

    run_id = insert_scrape_run(
        query=query,
        platforms=platforms,
        total_results=len(listings),
        triggered_by="user_query",
        filters_applied=filters_applied or {},
    )

    inserted = insert_listings(run_id, listings)

    print(f"  [Cache] Stored run {run_id} — "
          f"{inserted} new listings, {len(listings) - inserted} updated")


def check_status(query: str) -> dict:
    """
    Returns the cache status for a query in a format
    the chatbot can use directly in a response.

    Args:
        query: The search query to check

    Returns:
        Dict with all cache metadata:
        {
            "is_valid":           True,
            "last_run":           "2026-09-01 14:32:00",
            "age_minutes":        47,
            "expires_in_minutes": 193,
            "total_results":      24,
        }
    """
    return get_cache_status(query)


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from db.connection import close_db
    from db.models import create_indexes
    from db.queries import insert_scrape_run, insert_listings
    from datetime import datetime

    print("Testing core/cache.py...\n")
    create_indexes()

    TEST_QUERY = "computer vision cache test"

    # ── Test 1: cache miss on fresh query ─────────────────────────────────────
    print("Test 1 — cache miss (query never run):")
    result = get_or_invalidate(TEST_QUERY)
    print(f"  hit={result.hit} | {result.summary()}")
    assert result.hit is False, "Expected cache miss"

    # ── Test 2: store dummy results ────────────────────────────────────────────
    print("\nTest 2 — store_results:")

    # Create a minimal fake listing to store
    from core.scraper import ProjectListing
    fake_listings = [
        ProjectListing(
            title="Test CV Project",
            platform="Freelancer.com",
            description="A computer vision project for testing",
            budget_min=1000,
            budget_max=3000,
            currency="USD",
            skills=["Python", "OpenCV"],
            bid_count=5,
            posted_date="2026-09-01",
            url="https://freelancer.com/test-cache-001",
            source_type="freelance",
            keyword="computer vision",
            relevance_score=35.0,
            preference_rank=1,
        )
    ]
    store_results(TEST_QUERY, fake_listings)
    print("  Stored 1 fake listing")

    # ── Test 3: cache hit after storing ───────────────────────────────────────
    print("\nTest 3 — cache hit (just stored):")
    result = get_or_invalidate(TEST_QUERY)
    print(f"  hit={result.hit} | {result.summary()}")
    assert result.hit is True, "Expected cache hit"
    assert len(result.listings) > 0, "Expected listings"

    # ── Test 4: force fresh ───────────────────────────────────────────────────
    print("\nTest 4 — force_fresh=True:")
    result = get_or_invalidate(TEST_QUERY, force_fresh=True)
    print(f"  hit={result.hit} | {result.summary()}")
    assert result.hit is False, "Expected miss after force fresh"

    # ── Test 5: filter on cached results ──────────────────────────────────────
    print("\nTest 5 — re-store and filter by budget:")
    store_results(TEST_QUERY, fake_listings)
    result = get_or_invalidate(TEST_QUERY, filters={"budget_max": 5000})
    print(f"  hit={result.hit} | listings={len(result.listings)}")

    # ── Test 6: check_status ──────────────────────────────────────────────────
    print("\nTest 6 — check_status:")
    status = check_status(TEST_QUERY)
    for k, v in status.items():
        print(f"  {k}: {v}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    from db.connection import get_db
    from db.models import SCRAPE_RUNS_COLLECTION, PROJECT_LISTINGS_COLLECTION
    db = get_db()
    db[SCRAPE_RUNS_COLLECTION].delete_many({"normalized_query": "computer vision cache test"})
    db[PROJECT_LISTINGS_COLLECTION].delete_many({"url": "https://freelancer.com/test-cache-001"})
    print("\n  ✓ Test data cleaned up")

    close_db()
    print("\n✓ All cache tests passed")