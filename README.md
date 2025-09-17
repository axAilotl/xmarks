# XMarks 1.0 – Twitter Bookmark Knowledge Pipeline

XMarks turns your Twitter/X bookmarks into an Obsidian-ready knowledge base. It captures rich TweetDetail GraphQL payloads, downloads linked media/documents, optionally runs LLM summaries, and writes everything to `knowledge_vault/` in a single pass. Version **1.0** focuses on a lean, reliable core while keeping the capture workflow simple.

---

## Quick Start
```bash
# 1. Clone the repo and enter it
git clone https://github.com/axAilotl/xmarks.git
cd xmarks

# 2. Create a virtualenv and install runtime deps
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Playwright needs its bundled browsers
playwright install

# 4. Configure
cp config.example.json config.json    # edit paths + feature flags
cp .env.example .env                  # create if you prefer env vars
```

### Whisper / Speech-to-Text Backend
XMarks expects an **OpenAI-compatible Whisper API**. Two common setups:

```bash
# Option A – Speeches Whisper server (recommended)
docker run --rm -p 8080:8080 ghcr.io/speeches-ai/speeches-whisper-cpp:latest
export WHISPER_BASE_URL="http://localhost:8080/v1"
export WHISPER_API_KEY="sk-local"      # dummy key for local servers

# Option B – Ollama Whisper
ollama pull whisper
ollama serve
export WHISPER_BASE_URL="http://localhost:11434/v1"
export WHISPER_API_KEY="sk-local"
```
Enable the backend in `config.json` under `"whisper"` (or use `"deepgram"` if you prefer Deepgram’s API). Make sure `ffmpeg`/`ffprobe` are on your `PATH`.

---

## CLI Overview (`xmarks.py`)
| Command | Purpose | Most useful flags |
| --- | --- | --- |
| `stats` | Snapshot of bookmarks, caches, DB state. | `--verbose` |
| `download` | Fetch TweetDetail GraphQL via Playwright. | `--limit`, `--resume/--no-resume`, `--cookies` |
| `process` | Convert raw bookmarks (JSON) straight to markdown. | `--use-cache`, `--limit`, `--dry-run`* |
| `pipeline` | Single-pass download → enrich → markdown. | `--use-cache`, `--batch-size`, `--dry-run`, `--rerun-llm`, `--tweet-ids` |
| `async-test` | Benchmark async LLM throughput. | `--limit`, `--concurrent`, `--timeout` |
| `youtube` | Refresh cached tweets with YouTube metadata/transcripts. | `--limit`, `--resume` |
| `update-videos` | Regenerate existing markdown with richer video embeds. | `--no-tweets`, `--no-threads`, `--resume` |
| `twitter-transcripts` | Transcribe Twitter-hosted videos. | `--limit`, `--resume`, `--verbose` |
| `github-stars` / `huggingface-likes` | Summarise starred/liked repos. | `--limit`, `--resume`, `--include-*` |
| `db` | Inspect or vacuum the SQLite database. | `db stats`, `db vacuum`, `db export` |
| `migrate-filenames` | Normalise filenames/backlinks in the vault. | `--dry-run`, `--analyze` |

\*`--dry-run` works with `--use-cache` and prints a plan without touching the filesystem.

---

## Capture API (`uvicorn xmarks_api:app --reload`)
| Endpoint | Description |
| --- | --- |
| `GET /health` | Health probe for the service. |
| `POST /api/bookmark` | Accepts TweetDetail payloads from the extension/userscript. |
| `GET /api/bookmarks` | Last 100 captures from `realtime_bookmarks.json`. |
| `GET /api/bookmarks/pending` | Outstanding queue entries (SQLite). |
| `POST /api/bookmarks/status` | Bulk lookup (`status`, `processed_with_graphql`, etc.). |
| `POST /api/process` | Process everything currently queued. |
| `POST /api/triggers/github-stars` | Run the GitHub stars pipeline. |
| `POST /api/triggers/huggingface-likes` | Run the HuggingFace likes pipeline. |
| `GET /api/stats` | Capture stats + queue depth. |

All writes to `realtime_bookmarks.json` are funneled through an async lock so multiple browser submissions can’t corrupt the cache. Treat the file as a diagnostic aid—the durable source of truth lives in `.xmarks/meta.db`.

---

## Project Structure (high level)
```
xmarks/
├── xmarks.py                  # CLI entry point
├── xmarks_api.py              # FastAPI capture service
├── core/
│   ├── config.py              # Config loader + validation
│   ├── graphql_engine.py      # Playwright TweetDetail collector
│   ├── graphql_cache.py       # Shared cache cleanup helper
│   ├── pipeline_registry.py   # Stage metadata & predicates
│   ├── metadata_db.py         # SQLite helpers (bookmark queue, files index)
│   └── download_tracker.py    # Legacy JSON download cache helper
├── processors/
│   ├── pipeline_processor.py  # Single-pass orchestrator
│   ├── document_factory.py    # ArXiv/PDF/README dispatch
│   ├── youtube_processor.py   # YouTube metadata + transcripts
│   ├── transcription_processor.py # Chooses Whisper/Deepgram backends
│   ├── async_llm_processor.py # Async LLM orchestration
│   └── …                      # Content/thread/media/LLM/doc processors
├── knowledge_vault/           # Markdown, media, transcripts (git-ignored)
├── graphql_cache/             # Cached TweetDetail responses
├── realtime_bookmarks.json    # Latest captures (git-ignored)
├── download_tracking.json     # Download cache (git-ignored)
└── .xmarks/meta.db            # SQLite metadata DB (git-ignored)
```

### Configuration & Data
- `config.json` and `.env` hold your paths, feature flags, and API keys (`OPENAI_API_KEY`, `ANTHROPIC_API`, `GITHUB_API`, `YOUTUBE_API_KEY`, `DEEPGRAM_API_KEY`, etc.).
- Pipeline stages live under `pipeline.stages.*`; switch features on/off without code changes.
- `knowledge_vault/` can be deleted and regenerated; `.xmarks/meta.db` preserves queue and file metadata between runs.

---

## Tips
- **FFmpeg** – required for audio extraction (`ffmpeg` & `ffprobe`).
- **Respect rate limits** – `download` defaults to `--resume`; use small `--limit` values when iterating.
- **Dry run first** – `python xmarks.py pipeline --use-cache --dry-run` shows exactly what will happen.
- **Keep caches tidy** – set `pipeline.keep_graphql_cache` to `false` to auto-clean TweetDetail JSON after successful runs.

---

## Testing & Dev Tooling
```bash
pip install pytest pytest-asyncio black flake8 mypy
pytest
black .
flake8 core processors xmarks.py
mypy core processors
```

---

## License
MIT — see [LICENSE](./LICENSE).
