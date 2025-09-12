# xmarks

twitter bookmarks, github stars, and other knowledge collation engine

## Components

- `plan.md`: High-level roadmap and architecture notes.
- `playwright_scraper.py`: Fetches Twitter threads and extracts media without API access.
- `content_expander.py`: Expands external links (GitHub, arXiv, Substack) and extracts metadata.
- `tests/test_playwright_scraper.py`: Unit tests for cookie loading, thread parsing, and media URL extraction.
- `tests/test_content_expander.py`: Tests for link extraction and expansion helpers.
