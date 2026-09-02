# AI Project Leads Scraper

A Python web scraper for finding AI and machine learning project leads across multiple freelance and remote work platforms.

The current active scraper in [`main.py`](./main.py) searches:

- Freelancer.com
- PeoplePerHour
- Toptal
- Remotive

It expands broad AI-related queries into platform-specific project keywords, filters out low-relevance results, sorts the listings by relevance score, prints them to the terminal, and saves the results to JSON and Excel files.

## Features

- Query expansion for common AI topics like `computer vision`, `NLP`, `LLMs`, `MLOps`, `OCR`, and more
- Scraping from multiple sources in one run
- Relevance filtering to reduce non-AI noise
- Duplicate removal by URL
- `relevance_score` calculation for every kept listing
- Top 5 `preference_rank` labels for the strongest matches
- Sorting by `relevance_score` descending, with bid count and budget used as tie-breakers
- Automatic JSON and Excel export of each run

## Requirements

- Python 3.10 or newer
- Internet access
- The following Python packages:
  - `httpx`
  - `python-dotenv`
  - `selectolax`
  - `openpyxl`

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
pip install httpx python-dotenv selectolax openpyxl
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

1. Your query is expanded into related AI terms and project-intent searches.
2. Each source is searched with those terms.
3. Results are deduplicated by URL.
4. Listings are scored for AI relevance and low-signal non-AI jobs are removed.
5. The top 5 matches are marked with `preference_rank`.
6. Remaining listings are sorted by `relevance_score` from highest to lowest.
7. The final results are saved as JSON in `results/` and Excel in `excel/`.

## Output Files

Each run is saved automatically to the next ordinal JSON file and Excel file:

- `results/first.json`
- `results/second.json`
- `results/third.json`
- `excel/first.xlsx`
- `excel/second.xlsx`
- `excel/third.xlsx`
- and so on

If more than 20 output files already exist in a folder, the script falls back to names like:

- `results/run_21.json`
- `results/run_22.json`
- `excel/run_21.xlsx`
- `excel/run_22.xlsx`

Each saved file includes:

- `scraped_at`
- `query`
- `total_results`
- `results`

Each result includes fields such as title, platform, description, budget, skills, bid count, posted date, URL, source type, keyword, `relevance_score`, and `preference_rank`.

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
EXCEL_DIR=excel
```

## Project Structure

- [`main.py`](./main.py) - main scraper application
- [`alter.py`](./alter.py) - alternate scraper draft / reference version
- [`results/`](./results) - saved JSON output from runs
- `excel/` - saved Excel output from runs
- [`notes.txt`](./notes.txt) - keyword map notes

## Notes

- The scraper uses public pages and APIs where available.
- Some sites may change their HTML structure, which can affect parsing.
- Results are filtered heuristically, so occasional false positives or misses are possible.

## License

No license file is currently included.
