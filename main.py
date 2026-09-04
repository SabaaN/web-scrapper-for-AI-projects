"""
main.py
=======
CLI entry point. The scraping logic lives in core/scraper.py.
This file just handles command-line arguments and calls scrape().

Usage:
  python main.py
  python main.py --query "RAG chatbot"
  python main.py --query "computer vision" --no-excel
"""

import argparse
from core.scraper import scrape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape AI project leads from multiple platforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --query "computer vision"
  python main.py --query "RAG chatbot LLM"
  python main.py --query "machine learning pipeline" --no-excel
        """
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query. Prompted interactively if not provided.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip saving the Excel file.",
    )
    args = parser.parse_args()

    query = args.query
    if not query:
        print("\n  AI Project Leads Scraper")
        print("  ─────────────────────────────────────────────────────────────")
        print("  Sources: Freelancer · Remotive · Himalayas · RemoteOK · Arbeitnow")
        print("  Examples: 'computer vision', 'RAG chatbot', 'ML pipeline'")
        query = input("\n  Enter your search query: ").strip()
        if not query:
            print("  No query entered. Exiting.")
            raise SystemExit(0)

    scrape(
        query=query,
        save_json=True,
        save_excel=not args.no_excel,
        silent=False,
    )