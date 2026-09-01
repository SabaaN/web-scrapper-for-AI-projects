# AI Project Leads Scraper

A Python web scraper for finding AI and machine learning project leads across multiple freelance and remote work platforms.

The current active scraper in [`main.py`](./main.py) searches:

- Freelancer.com
- PeoplePerHour
- Toptal
- Remotive

It expands broad AI-related queries into platform-specific keywords, filters out low-relevance results, sorts the listings, prints them to the terminal, and saves the results to JSON files in `results/`.

## Features

- Query expansion for common AI topics like `computer vision`, `NLP`, `LLMs`, `MLOps`, `OCR`, and more
- Scraping from multiple sources in one run
- Relevance filtering to reduce non-AI noise
- Duplicate removal by URL
- Sorting by competition and budget
- Automatic JSON export of each run

## Requirements

- Python 3.10 or newer
- Internet access
- The following Python packages:
  - `httpx`
  - `python-dotenv`
  - `selectolax`

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install httpx python-dotenv selectolax
```

## Usage

Run the scraper with an interactive prompt:

```bash
python main.py
```

Or pass a query directly:

```bash
python main.py --query "computer vision object detection"
```

More examples:

```bash
python main.py --query "NLP chatbot"
python main.py --query "LLM RAG pipeline"
python main.py --query "machine learning"
```

## How It Works

1. Your query is expanded into a set of related search terms.
2. Each source is searched with those terms.
3. Results are deduplicated by URL.
4. Listings are scored for AI relevance and low-signal non-AI jobs are removed.
5. Remaining listings are sorted and printed.
6. The final results are saved as JSON in `results/`.

## Output Files

Each run is saved automatically to the next ordinal JSON file:

- `results/first.json`
- `results/second.json`
- `results/third.json`
- and so on

If more than 20 result files already exist, the script falls back to names like:

- `results/run_21.json`
- `results/run_22.json`

Each saved file includes:

- `scraped_at`
- `query`
- `total_results`
- `results`

## Configuration

You can customize behavior with environment variables in a `.env` file:

```env
RESULTS_PER_KEYWORD=20
REQUEST_DELAY=2.5
REQUEST_TIMEOUT=15
FREELANCER_LIMIT=20
PPH_BASE_URL=https://www.peopleperhour.com
TOPTAL_BASE_URL=https://www.toptal.com
REMOTIVE_API_URL=https://remotive.com/api/remote-jobs
RESULTS_DIR=results
```

## Project Structure

- [`main.py`](./main.py) - main scraper application
- [`alter.py`](./alter.py) - alternate scraper draft / reference version
- [`results/`](./results) - saved JSON output from runs
- [`notes.txt`](./notes.txt) - keyword map notes

## Notes

- The scraper uses public pages and APIs where available.
- Some sites may change their HTML structure, which can affect parsing.
- Results are filtered heuristically, so occasional false positives or misses are possible.

## License

No license file is currently included.
