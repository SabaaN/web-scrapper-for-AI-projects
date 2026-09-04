"""
db/models.py
============
Defines the structure of our two MongoDB collections
and creates the indexes that make queries fast.

Collections:
  - scrape_runs       : one document per scrape execution
  - project_listings  : one document per individual project found

MongoDB doesn't enforce schemas the way SQL does — any document
can have any fields. These functions just ensure the indexes exist
and provide helper functions to build valid documents consistently.
"""

import os
from datetime import datetime
from pymongo.database import Database
from pymongo import ASCENDING, DESCENDING
from db.connection import get_db


# ── Collection Names ──────────────────────────────────────────────────────────

SCRAPE_RUNS_COLLECTION      = "scrape_runs"
PROJECT_LISTINGS_COLLECTION = "project_listings"


# ── Index Setup ───────────────────────────────────────────────────────────────

def create_indexes() -> None:
    """
    Creates all indexes on both collections.
    Safe to call multiple times — MongoDB skips indexes that already exist.
    Call this once when the app starts.
    """
    db = get_db()

    # ── scrape_runs indexes ───────────────────────────────────────────────────

    db[SCRAPE_RUNS_COLLECTION].create_index(
        [("normalized_query", ASCENDING), ("ran_at", DESCENDING)],
        name="query_time_idx"
    )
    # For finding all runs within a time range
    db[SCRAPE_RUNS_COLLECTION].create_index(
        [("ran_at", DESCENDING)],
        name="ran_at_idx"
    )

    # ── project_listings indexes ──────────────────────────────────────────────

    # For fetching all listings from a specific scrape run
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("scrape_run_id", ASCENDING)],
        name="scrape_run_id_idx"
    )
    # For deduplication — checking if a URL was already scraped
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("url", ASCENDING)],
        name="url_idx"
    )
    # For sorting results by relevance
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("relevance_score", DESCENDING)],
        name="relevance_idx"
    )
    # For filtering by platform and recency
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("platform", ASCENDING), ("first_seen_at", DESCENDING)],
        name="platform_time_idx"
    )
    # For budget filtering
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("budget_max", DESCENDING)],
        name="budget_idx"
    )
    # For bid count filtering (competition level)
    db[PROJECT_LISTINGS_COLLECTION].create_index(
        [("bid_count", ASCENDING)],
        name="bid_count_idx"
    )

    print("✓ MongoDB indexes created / verified")


# ── Document Builders ─────────────────────────────────────────────────────────
# These functions construct valid MongoDB documents from raw data.
# Think of them as the "schema" — they define what a document should look like.

def build_scrape_run_doc(
    query: str,
    normalized_query: str,
    platforms: list[str],
    total_results: int,
    triggered_by: str = "user_query",
    filters_applied: dict | None = None,
) -> dict:
    """
    Builds a scrape_runs document.

    Args:
        query:            The raw query as the user typed it ("RAG chatbot")
        normalized_query: Lowercased, stripped version ("rag chatbot")
        platforms:        Which platforms were scraped
        total_results:    How many results passed the filter
        triggered_by:     "user_query" or "cache_miss"
        filters_applied:  Any filters like { "budget_max": 5000 }

    Returns a dict ready to insert into MongoDB.
    """
    return {
        "query":            query,
        "normalized_query": normalized_query,
        "ran_at":           datetime.utcnow(),
        "platforms":        platforms,
        "total_results":    total_results,
        "triggered_by":     triggered_by,
        "filters_applied":  filters_applied or {},
    }


def build_project_listing_doc(
    scrape_run_id,          # ObjectId from the parent scrape_run
    listing: dict,          # the ProjectListing dataclass as a dict
) -> dict:
    """
    Builds a project_listings document from a scraped ProjectListing.

    Adds two fields that don't exist in the scraper output:
      - first_seen_at : set on first insert, never updated
      - last_seen_at  : updated every time the same URL reappears
      - times_seen    : counter, incremented on each re-scrape

    Args:
        scrape_run_id: The _id of the parent scrape_run document
        listing:       A ProjectListing converted to dict via asdict()

    Returns a dict ready to insert into MongoDB.
    """
    now = datetime.utcnow()
    return {
        "scrape_run_id":   scrape_run_id,
        "title":           listing.get("title", ""),
        "platform":        listing.get("platform", ""),
        "description":     listing.get("description", ""),
        "budget_min":      listing.get("budget_min"),
        "budget_max":      listing.get("budget_max"),
        "currency":        listing.get("currency"),
        "skills":          listing.get("skills", []),
        "bid_count":       listing.get("bid_count"),
        "relevance_score": listing.get("relevance_score", 0.0),
        "preference_rank": listing.get("preference_rank"),
        "posted_date":     listing.get("posted_date"),
        "url":             listing.get("url", ""),
        "source_type":     listing.get("source_type", ""),
        "keyword":         listing.get("keyword", ""),
        "first_seen_at":   now,
        "last_seen_at":    now,
        "times_seen":      1,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = get_db()

    # Create indexes
    create_indexes()

    # Verify both collections exist after index creation
    print(f"Collections: {db.list_collection_names()}")

    # Show indexes on each collection
    print("\nscrape_runs indexes:")
    for idx in db[SCRAPE_RUNS_COLLECTION].list_indexes():
        print(f"  - {idx['name']}")

    print("\nproject_listings indexes:")
    for idx in db[PROJECT_LISTINGS_COLLECTION].list_indexes():
        print(f"  - {idx['name']}")