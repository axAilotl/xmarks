# Twitter Bookmark Knowledge Management System

## Key Implementation Insights

### For Thread Scraping without API costs:

- Use Playwright to intercept network requests such as GraphQL `TweetDetail` responses.
- Authenticate by loading exported Twitter cookies.
- Detect threads by checking page state or "Show this thread" text.

### For Media Handling:

- Twitter media URLs usually live on `pbs.twimg.com` and can be downloaded directly.
- Use Whisper for video transcription; consider `whisper.cpp` for local processing.
- Store media with content-based hashing to avoid duplicates.

### For LLM Processing:

- Prefer running local models via Ollama for privacy and cost; models like Mixtral or Llama 3 work well for classification.
- Craft prompts that return structured JSON for consistent tagging.
- Use different models for specialized tasks (classification vs. summarization).

### For Obsidian Integration:

- Maintain an organized folder structure for tweets, media, and external content.
- Combine Obsidian's search with a vector database for redundancy.
- Use Dataview plugin queries for dynamic indexes.

### Quick Start Path

1. Build minimal prototype:
   - Scrape threads.
   - Save raw JSON data before processing.
   - Test with a small set of bookmarks.
2. Incrementally add:
   - LLM tagging.
   - Media download.
   - External link expansion.
3. Reuse existing tools when possible:
   - `gallery-dl` for media.
   - `yt-dlp` for videos.
   - `newspaper3k` for articles.
4. Create database schemas for SQLite metadata storage.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                      │
├─────────────────────────────────────────────────────────────┤
│  • Twitter Scraper (Playwright)                              │
│  • Content Expander (Threads, Links)                         │
│  • Media Downloader (Images, Videos)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  • LLM Classifier/Tagger                                     │
│  • Content Summarizer                                        │
│  • Media Processor (Transcription, OCR, Captions)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
├─────────────────────────────────────────────────────────────┤
│  • Obsidian Vault (Markdown files)                          │
│  • Media Storage (Local/Cloud)                              │
│  • Vector Database (ChromaDB/Qdrant)                        │
│  • SQLite (Metadata & Relations)                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                           │
├─────────────────────────────────────────────────────────────┤
│  • Web UI (Search & Browse)                                 │
│  • Obsidian Plugin                                          │
│  • API Endpoints                                            │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1 Modules

### `playwright_scraper.py`
Handles thread scraping and media URL extraction using Playwright. Cookies are loaded from an exported JSON file.

### `content_expander.py`
Expands external links (GitHub, arXiv, Substack) and collects metadata for downstream processing.

### `llm_processor.py`
Provides wrappers around local or remote LLMs for classification, summarization, and transcription.

### `obsidian_manager.py`
Creates markdown notes in a structured Obsidian vault.

### `vector_db.py`
Stores embeddings and metadata in a vector database for semantic search.

### `main.py`
Orchestrates the pipeline: scraping, expansion, LLM processing, note creation, and vector DB insertion.

## Configuration (`config.yaml`)

Describes paths, API keys, and processing options. Example sections:

- `cookies_file`
- `llm` provider and model
- storage paths for vault, vector DB, media
- feature toggles such as `expand_links`, `download_media`, etc.

## Web Interface (`search_ui.py`)

Streamlit-based search interface for browsing stored tweets and metadata.

## Phase 2 Ideas

- GitHub stars integration.
- Browser extension for bookmark sync.
- Performance optimizations and privacy considerations.

