# XMarks - Modular Twitter Knowledge Management System (v2.0)

## 🎯 **Overview**

XMarks is a powerful Twitter bookmark processing system that transforms your saved tweets into an Obsidian-compatible knowledge vault. Built on a **GraphQL-first architecture**, it extracts rich data from Twitter using Playwright automation, then applies modular processors to generate comprehensive markdown documentation with linked media, papers, and documents.

### 🆕 **Version 2.0 Improvements**
- **30% Less Code**: Eliminated 200+ lines of duplication with MarkdownGenerator
- **2x Faster Processing**: Single-pass pipeline architecture
- **Better Async**: Enhanced LLM processing with semaphore control
- **Auto-Config**: Environment variable support with `.env` file loading
- **Extensible**: Document factory pattern for easy processor additions

### **Key Features**
- 📡 **GraphQL Data Collection** - Direct Twitter API access via Playwright
- 🧵 **Smart Thread Detection** - Uses Twitter's own thread indicators
- 📄 **Document Processing** - ArXiv papers, PDFs, GitHub/HuggingFace READMEs
- 🖼️ **Media Management** - Downloads and links images, videos, GIFs
- 🎬 **YouTube Processing** - Video metadata, optional embeds, and transcripts
- 🎤 **Video Transcripts** - Local Whisper or Deepgram transcription for Twitter videos (60s+)
- 🤖 **AI Enhancement** - LLM-powered tags, summaries, and alt text (async-capable)
- 🔄 **Resume Capability** - Interrupt and restart anytime
- ⚡ **Single-Pass Pipeline** - End-to-end processing in one optimized pass
- 🌐 **Live Capture API** - FastAPI server + browser extension/userscript for real-time capture
- 📚 **Obsidian Integration** - Ready-to-use markdown with wikilinks

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers (required for GraphQL)
playwright install
```

### **Basic Commands**

```bash
# Check current status
python xmarks.py stats

# Process cached data (fast, no rate limits)
python xmarks.py process --use-cache --limit 100

# NEW: Optimized single-pass pipeline (2x faster)
python xmarks.py pipeline --use-cache --batch-size 10

# Preview without writing files (requires --use-cache)
python xmarks.py pipeline --use-cache --dry-run

# NEW: Test async LLM with performance monitoring
python xmarks.py async-test --limit 5 --concurrent 8

# Process Twitter video transcripts (Whisper or Deepgram)
python xmarks.py twitter-transcripts --limit 50 --resume

# Download new GraphQL data (rate limited)
python xmarks.py download --resume --limit 50

# Full pipeline (download + process)
python xmarks.py full --resume

# NEW: Post-process tweets for YouTube videos (embeds + transcripts)
python xmarks.py youtube --limit 100 --no-resume

# NEW: Update existing tweet/thread files with clickable video thumbnails
python xmarks.py update-videos --no-resume --no-threads

# NEW: Process your GitHub stars to summaries + README captures
python xmarks.py github-stars --limit 50 --resume

# NEW: Process your HuggingFace likes (models/datasets/spaces)
python xmarks.py huggingface-likes --limit 50 --resume
```

---

## 📁 **Project Structure**

```
xmarks/
├── xmarks.py                        # Main CLI interface
├── xmarks_api.py                    # FastAPI server for live capture
├── core/                            # Core engine components
│   ├── graphql_engine.py           # Twitter GraphQL collection
│   ├── data_models.py              # Data structures
│   ├── config.py                   # Configuration management
│   ├── llm_interface.py            # Multi-provider LLM support
│   ├── download_tracker.py         # Download status tracking
│   └── github_utils.py             # Repository utilities
├── processors/                      # Modular processors
│   ├── content_processor.py        # Tweet markdown generation
│   ├── thread_processor.py         # Thread detection & files
│   ├── url_processor.py            # URL expansion & PDFs
│   ├── media_processor.py          # Media downloads
│   ├── arxiv_processor.py          # ArXiv paper processing
│   ├── llm_processor.py            # AI-powered features
│   ├── async_llm_processor.py      # Async LLM with concurrency control
│   ├── youtube_processor.py        # YouTube video processing
│   ├── twitter_video_transcript_processor.py # Twitter video transcripts
│   ├── deepgram_transcript_processor.py      # Optional Deepgram support
│   └── cache_loader.py             # Cache data loading
├── knowledge_vault/                 # Generated output
│   ├── tweets/                     # Individual tweet files
│   ├── threads/                    # Thread compilations
│   ├── papers/                     # ArXiv papers (PDFs)
│   ├── pdfs/                       # General PDF documents
│   ├── media/                      # Downloaded media
│   ├── readmes/                    # Repository READMEs
│   ├── repos/                      # Saved repository README files
│   ├── stars/                      # GitHub/HF repo summaries
│   └── transcripts/                # YouTube/Twitter video transcripts
├── graphql_cache/                   # Cached API responses
├── config.json                      # Configuration file
├── twitter-bookmarks-merged.json    # Source bookmarks
└── twitter_cookies.json             # Authentication
```

### Additional Components

- `browser_extension/` and `userscript/`: Real-time capture from Twitter/X to the local API.
- `vector_store/`: Optional local vector artifacts (if used).

---

## 🔧 **CLI Commands**

### **`xmarks.py stats`**
Shows current processing status and statistics.

### **`xmarks.py download [options]`**
Downloads GraphQL data from Twitter (rate limited).

**Options:**
- `--limit N` - Process only N tweets
- `--resume` - Skip already cached tweets (default: true)
- `--cookies FILE` - Custom cookies file
- `--verbose` - Enable detailed logging

### **`xmarks.py process [options]`**
Processes tweets into markdown files.

**Options:**
- `--use-cache` - Use cached GraphQL data (recommended)
- `--limit N` - Process only N tweets
- `--resume` - Skip existing files (default: true)
- `--verbose` - Enable detailed logging

### **`xmarks.py full [options]`**
Complete pipeline: download + process.

**Options:**
- `--limit N` - Process only N tweets
- `--resume` - Enable resume capability (default: true)
- `--verbose` - Enable detailed logging

### 🆕 **`xmarks.py pipeline [options]`**
Optimized single-pass processing (2x faster than standard process).

**Options:**
- `--use-cache` - Use cached GraphQL data
- `--limit N` - Process only N tweets
- `--batch-size N` - Concurrent batch size (default: 10)
- `--resume` - Skip existing files (default: true)

### 🆕 **`xmarks.py async-test [options]`**
Test enhanced async LLM processing with performance monitoring.

**Options:**
- `--limit N` - Number of tweets to test (default: 5)
- `--concurrent N` - Max concurrent requests (default: 8)
- `--timeout N` - Request timeout in seconds (default: 20)

---

## 🧩 **Core Components**

### **GraphQL Engine**
- Playwright-based Twitter data collection
- Rate limiting (45 requests/15 minutes)
- Response caching for resume capability
- Extracts threads, URLs, media metadata

### **Data Models**
- `Tweet` - Enhanced tweet with GraphQL data
- `MediaItem` - Media file information
- `URLMapping` - t.co → expanded URL mappings
- `ThreadInfo` - Thread metadata
- `ProcessingStats` - Operation statistics

### **Processors**

| Processor | Function | Status |
|-----------|----------|---------|
| **ContentProcessor** | Generates individual tweet markdown files | ✅ Core |
| **ThreadProcessor** | Detects and compiles thread files | ✅ Core |
| **CacheLoader** | Loads cached GraphQL for fast processing | ✅ Core |
| **URLProcessor** | Expands URLs and manages PDFs/READMEs | ✅ Enhanced |
| **MediaProcessor** | Downloads media with Obsidian links | ✅ Enhanced |
| **ArXivProcessor** | Processes academic papers with metadata | ✅ Enhanced |
| **LLMProcessor** | Adds AI-powered tags, summaries, alt text | ✅ Enhanced |
| **AsyncLLMProcessor** | Async LLM with concurrency control | ✅ Enhanced |
| **YouTubeProcessor** | YouTube metadata, transcripts, embeds | ✅ Enhanced |
| **TwitterVideoTranscriptProcessor** | Whisper/Deepgram Twitter transcripts | ✅ Enhanced |
| **VideoUpdater** | Updates tweet/thread files with video thumbnails | ✅ Utility |
| **GitHubStarsProcessor** | Processes your GitHub starred repos | ✅ Utility |
| **HuggingFaceLikesProcessor** | Processes your HuggingFace likes | ✅ Utility |
| **MarkdownGenerator** | Shared markdown generation utility | ✅ Utility |
| **MetadataDB** | SQLite metadata (tweets, downloads, files, LLM cache) | 🆕 |

---

## 🤖 **LLM Integration**

### **Multi-Provider Support**
- OpenAI (GPT models)
- Anthropic (Claude models)
- OpenRouter (various models)
- Local (Ollama)

### **AI Features**
- **Tags** - Automatic hashtag generation
- **Summaries** - Concise summaries for long tweets/threads
- **Alt Text** - Accessibility descriptions for images
- **README Summaries** - Repository description extraction

### **Configuration Example**
```json
{
  "llm": {
    "anthropic": {
      "enabled": true,
      "model": "claude-sonnet-4-20250514"
    },
    "tasks": {
      "tags": {"provider": "anthropic", "enabled": true},
      "summary": {"provider": "anthropic", "enabled": true},
      "alt_text": {"provider": "openrouter", "enabled": true}
    }
  },
  "whisper": {
    "enabled": true,
    "base_url": "http://localhost:11434/v1",
    "model": "Systran/faster-distil-whisper-large-v3",
    "min_duration_seconds": 60,
    "max_chunk_mb": 25,
    "target_bitrate_kbps": 128
  }
}
```

### **Video Transcript Setup**
1. **Install ffmpeg**: Required for audio extraction
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Start Whisper Server**: Use Ollama or compatible server
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Whisper model
   ollama pull whisper
   
   # Start server (default port 11434)
   ollama serve
   ```

3. **Configure**: Update `config.json` with your Whisper server URL

#### Deepgram (Optional, faster)
1. Set environment variable `DEEPGRAM_API_KEY`.
2. Enable in `config.json` under `deepgram.enabled: true` and set model (e.g., `nova-2`).
3. Run the same transcript commands; the system will auto-prefer Deepgram if configured.

---

## 📝 **Generated Markdown Structure**

Each tweet file includes:

```markdown
---
type: tweet
id: 1234567890
author: "username"
created_at: 2024-01-01 12:00:00
enhanced: true
---

# Tweet by @username

## Content
[Tweet text with expanded URLs]
![[media_file.jpg]]

## ArXiv Papers
### Paper 1: [Title]
- **ArXiv ID**: [2401.12345](https://arxiv.org/abs/2401.12345)
- **Abstract**: [Paper abstract]
- **PDF**: [[2401.12345.pdf]]

## PDF Documents
### Document 1: [Title]
- **URL**: [https://example.com/doc.pdf]
- **PDF**: [[document_title.pdf]]

## Summary
[LLM-generated summary for long tweets]

#tag1 #tag2 #tag3
```

---

## 🎯 **Key Advantages**

### **1. GraphQL-First Architecture**
Direct access to Twitter's rich data structure, not just basic text.

### **2. Real Thread Detection**
Uses Twitter's `tweetDisplayType: "SelfThread"` for accurate thread grouping.

### **3. Comprehensive Document Processing**
- ArXiv papers with full metadata
- PDF downloads with smart naming
- GitHub/HuggingFace README extraction

### **4. Video Transcript Processing**
- Local Whisper transcription for Twitter videos (60+ seconds)
- Automatic audio extraction with ffmpeg
- LLM-powered transcript cleaning and formatting
- Chunked processing for large audio files

### **5. Resume Capability**
All operations support interruption and continuation.

### **6. Fast Cached Processing**
Once GraphQL data is cached, processing has no rate limits.

---

## 🔍 **Usage Examples**

### **Scenario 1: Quick Processing**
Process existing cached data with all enhancements:
```bash
python xmarks.py process --use-cache
```

### **Scenario 2: Limited Download**
Download GraphQL data for new bookmarks:
```bash
python xmarks.py download --limit 100 --resume
```

### **Scenario 3: Full Pipeline**
Complete processing for new bookmarks file:
```bash
python xmarks.py full --bookmarks new-bookmarks.json
```

### **Scenario 4: YouTube Post-Processing**
Process existing cached tweets to pull YouTube metadata, optional embeds, and transcripts:
```bash
python xmarks.py youtube --limit 100 --no-resume
```

### **Scenario 5: Testing**
Test with small dataset:
```bash
python xmarks.py process --use-cache --limit 10 --verbose
```

---

## 🚨 **Important Notes**

### **Authentication**
- Requires `twitter_cookies.json` with valid session
- Use browser developer tools to export cookies
- Test with small batches first

### **Rate Limiting**
- GraphQL: 45 requests per 15 minutes
- Processing from cache: No limits
- LLM APIs: Configurable delays

### **Resume Capability**
- Downloads skip cached tweets
- Processing skips existing files
- Safe to interrupt at any time

### **File Management**
- Media files use unique names to avoid collisions
- PDFs organized by type (ArXiv vs general)
- All documents linked with Obsidian wikilinks

---

## 🐛 **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| GraphQL fails | Check `twitter_cookies.json` validity |
| Rate limited | Wait 15 minutes or reduce batch size |
| LLM errors | Verify API keys in environment variables |
| Missing media | Check `knowledge_vault/media/` permissions |
| YouTube disabled | Set `YOUTUBE_API_KEY` and enable in config |
| Transcript server | For Whisper: ensure Ollama running; For Deepgram: set `DEEPGRAM_API_KEY` |
| API not reachable | Start `xmarks_api.py` (FastAPI) and check `/health` |

### **Debug Mode**
```bash
# Enable verbose logging
python xmarks.py process --verbose

# Check logs
tail -f xmarks.log
```

---

## 🔮 **Roadmap**

### **Completed** ✅
- GraphQL engine with caching
- Modular processor architecture
- Thread detection and compilation
- URL expansion and PDF downloads
- ArXiv paper processing
- Media downloads with Obsidian links
- Multi-provider LLM integration
- Resume capability for all operations
 - YouTube processing (embeds + transcripts)
 - Live capture API server and browser extension

### **Planned** 🚧
- [ ] Plugin system for custom processors
- [ ] Web UI for monitoring progress
- [ ] Export to other formats (Notion, Roam)
- [ ] Incremental bookmark updates
- [ ] Vector database integration

---

## 🌐 API Server (Live Capture)

Start the API server to receive real-time bookmarks from the browser extension/userscript:

```bash
uvicorn xmarks_api:app --reload
```

Endpoints:

- `GET /health` – health check
- `POST /api/bookmark` – receive bookmark payloads
- `GET /api/bookmarks` – recent captured bookmarks
- `POST /api/process` – trigger processing of captured items
- `GET /api/stats` – capture statistics

All writes to `realtime_bookmarks.json` are funneled through an async lock so concurrent browser submissions cannot corrupt the capture cache; prefer using these endpoints instead of editing the file by hand.

See `browser_extension/README.md` for installation and usage.

---

## 🗄️ Metadata Database (SQLite)

- Location: configurable via `database.path` (default `.xmarks/meta.db`)
- Tables: `tweets`, `url_mappings`, `downloads`, `llm_cache`, `files_index`, `graphql_cache_index`
- CLI: `python xmarks.py db stats|vacuum|export`
- Purpose: reuse across runs, speed resume, power browser/API lookups

---

## 🧰 Configuration Highlights

- Granular pipeline flags in `config.json` under `pipeline.stages`
- LLM prompts under `llm.prompts.*` (tags, summary, alt_text, readme, transcript)
- File naming under `files.naming_patterns.*` (tweets prefixed with `tweets_`)
- Downloads: timeouts/retries/user-agent under `downloads.*`
- Database: `database.enabled`, `database.path`, `database.wal_mode`

---

## 📄 **License**

MIT License - See LICENSE file for details

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly with small datasets
4. Submit a pull request

---

**Built with a focus on modularity, reliability, and comprehensive knowledge capture.**