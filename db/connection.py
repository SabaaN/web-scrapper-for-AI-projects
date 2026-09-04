"""
db/connection.py
================
Handles the MongoDB connection.
One single connection is created when the app starts
and reused everywhere — this pattern is called a singleton.
"""

import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI     = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Module-level variable — created once, reused everywhere
_client: MongoClient | None = None
_db: Database | None = None


def get_db() -> Database:
    """
    Returns the database instance.
    Creates the connection on first call, reuses it after that.
    """
    global _client, _db

    if _db is not None:
        return _db

    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)

        # This actually tests the connection — raises if MongoDB isn't running
        _client.admin.command("ping")

        _db = _client[MONGODB_DB_NAME]
        print(f"✓ Connected to MongoDB — database: '{MONGODB_DB_NAME}'")
        return _db

    except ConnectionFailure:
        raise ConnectionError(
            "Could not connect to MongoDB.\n"
            "Make sure MongoDB is running (check Services or run 'mongosh')"
        )


def close_db() -> None:
    """Cleanly close the connection. Call this when the app shuts down."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        print("MongoDB connection closed.")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = get_db()
    print(f"Collections: {db.list_collection_names()}")
    close_db()