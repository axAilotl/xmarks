#!/usr/bin/env python3
"""
XMarks API Server - Receives real-time bookmark captures from browser extension
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add current directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core import Tweet, GraphQLEngine, config
from core.metadata_db import get_metadata_db, BookmarkQueueEntry
from processors import ContentProcessor, ThreadProcessor
from processors.pipeline_processor import PipelineProcessor
from processors.cache_loader import CacheLoader
from processors.github_stars_processor import GitHubStarsProcessor
from processors.huggingface_likes_processor import HuggingFaceLikesProcessor
from core.graphql_cache import maybe_cleanup_graphql_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="XMarks API", version="1.0.0")

# Shared pipeline instance protected by an asyncio lock to avoid overlapping runs
pipeline_runner = PipelineProcessor()
pipeline_lock = asyncio.Lock()
github_trigger_lock = asyncio.Lock()
huggingface_trigger_lock = asyncio.Lock()


async def run_pipeline_for_tweets(
    tweets: List[Tweet],
    url_mappings: Optional[Dict[str, str]] = None,
    resume: bool = True,
    rerun_llm: bool = False,
    llm_only: bool = False,
):
    """Execute the unified pipeline for a set of tweets."""
    if not tweets:
        return None

    batch_size = max(1, len(tweets))

    async with pipeline_lock:
        return await pipeline_runner.process_tweets_pipeline(
            tweets,
            url_mappings=url_mappings,
            resume=resume,
            batch_size=batch_size,
            rerun_llm=rerun_llm,
            llm_only=llm_only,
        )


def upsert_bookmark_queue_entry(bookmark_data: Dict[str, Any]):
    """Persist bookmark metadata in the durable queue."""
    tweet_id = bookmark_data.get("tweet_id")
    if not tweet_id:
        return

    captured_at = bookmark_data.get("timestamp") or datetime.now().isoformat()
    entry = BookmarkQueueEntry(
        tweet_id=tweet_id,
        source=bookmark_data.get("source"),
        captured_at=captured_at,
        status="pending",
        payload_json=json.dumps(bookmark_data, ensure_ascii=False),
        next_attempt_at=captured_at,
    )
    try:
        db = get_metadata_db()
        db.upsert_bookmark_entry(entry)
    except Exception as exc:
        logger.error(f"Failed to persist bookmark queue entry {tweet_id}: {exc}")


async def enqueue_bookmark_payload(
    bookmark_data: Dict[str, Any], delay_seconds: float = 0.0
):
    """Schedule a bookmark for processing via the async queue."""
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)

    tweet_id = bookmark_data.get("tweet_id")
    if not tweet_id:
        logger.warning("Bookmark payload missing tweet_id; skipping queue")
        return

    await PROCESSING_QUEUE.put(bookmark_data)
    logger.debug(f"Enqueued bookmark {tweet_id} for processing")


def bookmark_entry_to_payload(
    entry: Optional[BookmarkQueueEntry],
) -> Optional[Dict[str, Any]]:
    """Convert a stored bookmark entry back into a payload dictionary."""
    if not entry:
        return None

    try:
        payload = json.loads(entry.payload_json) if entry.payload_json else {}
    except json.JSONDecodeError as exc:
        logger.error(
            f"Failed to deserialize bookmark payload for {entry.tweet_id}: {exc}"
        )
        payload = {}

    payload.setdefault("tweet_id", entry.tweet_id)
    if entry.source and not payload.get("source"):
        payload["source"] = entry.source
    if entry.captured_at and not payload.get("timestamp"):
        payload["timestamp"] = entry.captured_at

    return payload


async def schedule_retry(tweet_id: str, next_attempt_iso: Optional[str]):
    """Schedule a retry for a bookmark when the next attempt is due."""
    if not next_attempt_iso:
        return

    try:
        next_attempt = datetime.fromisoformat(next_attempt_iso)
    except ValueError:
        logger.debug(f"Invalid next_attempt_at for {tweet_id}: {next_attempt_iso}")
        return

    delay_seconds = max(0.0, (next_attempt - datetime.now()).total_seconds())
    await asyncio.sleep(delay_seconds)

    db = get_metadata_db()
    entry = db.get_bookmark_entry(tweet_id)
    if not entry or entry.status != "pending":
        return

    payload = bookmark_entry_to_payload(entry)
    if payload:
        await PROCESSING_QUEUE.put(payload)
        logger.info(f"Re-queued bookmark {tweet_id} after failure")


def serialize_bookmark_entry(entry: BookmarkQueueEntry) -> Dict[str, Any]:
    """Serialize a bookmark queue entry for API responses."""
    return {
        "tweet_id": entry.tweet_id,
        "source": entry.source,
        "captured_at": entry.captured_at,
        "status": entry.status,
        "attempts": entry.attempts,
        "last_error": entry.last_error,
        "last_attempt_at": entry.last_attempt_at,
        "processed_at": entry.processed_at,
        "next_attempt_at": entry.next_attempt_at,
        "processed_with_graphql": entry.processed_with_graphql,
    }


def serialize_processing_stats(stats) -> Dict[str, Any]:
    """Convert a ProcessingStats-like object into a response dictionary."""
    return {
        "created": getattr(stats, "created", 0),
        "updated": getattr(stats, "updated", 0),
        "skipped": getattr(stats, "skipped", 0),
        "errors": getattr(stats, "errors", 0),
        "total_processed": getattr(stats, "total_processed", 0),
    }


async def load_pending_bookmarks_from_db():
    """Load pending bookmarks from the durable queue into memory on startup."""
    db = get_metadata_db()
    entries = db.get_unprocessed_bookmarks()
    now = datetime.now()

    for entry in entries:
        payload = bookmark_entry_to_payload(entry)
        if not payload:
            continue

        delay = 0.0
        if entry.status == "pending" and entry.next_attempt_at:
            try:
                next_attempt = datetime.fromisoformat(entry.next_attempt_at)
                delay = max(0.0, (next_attempt - now).total_seconds())
            except ValueError:
                delay = 0.0

        if entry.status == "failed":
            # Do not auto enqueue permanently failed items
            continue

        asyncio.create_task(enqueue_bookmark_payload(payload, delay))


# Configure CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://twitter.com", "https://x.com", "chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class BookmarkCapture(BaseModel):
    """Bookmark data from browser extension"""

    tweet_id: str
    tweet_data: Optional[Dict[str, Any]] = None
    graphql_response: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    source: Optional[str] = "browser_extension"
    force: Optional[bool] = False


class ProcessingStatus(BaseModel):
    """Processing status response"""

    status: str
    message: str
    tweet_id: Optional[str] = None
    processed_at: Optional[str] = None


class GitHubTriggerRequest(BaseModel):
    """Request payload for triggering the GitHub stars pipeline."""

    limit: Optional[int] = None
    resume: bool = True


class HuggingFaceTriggerRequest(BaseModel):
    """Request payload for triggering the HuggingFace likes pipeline."""

    limit: Optional[int] = None
    resume: bool = True
    include_models: bool = True
    include_datasets: bool = True
    include_spaces: bool = True


class BookmarkStatusRequest(BaseModel):
    """Request body for bookmark status lookups."""

    tweet_ids: List[str]


# Storage
REALTIME_BOOKMARKS_FILE = Path("realtime_bookmarks.json")
PROCESSING_QUEUE = asyncio.Queue()
BOOKMARKS_FILE_LOCK = asyncio.Lock()


def load_realtime_bookmarks() -> list:
    """Load existing realtime bookmarks"""
    if REALTIME_BOOKMARKS_FILE.exists():
        try:
            with open(REALTIME_BOOKMARKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


async def mutate_realtime_bookmarks(
    mutator: Callable[[List[dict]], Tuple[bool, Any]],
) -> Any:
    """Apply a mutation to realtime bookmarks under an async lock."""

    async with BOOKMARKS_FILE_LOCK:
        bookmarks = load_realtime_bookmarks()
        dirty, result = mutator(bookmarks)
        if dirty:
            with open(REALTIME_BOOKMARKS_FILE, "w", encoding="utf-8") as f:
                json.dump(bookmarks, f, indent=2, ensure_ascii=False)
        return result


def save_bookmark(bookmark_data: dict) -> Tuple[bool, dict]:
    """Save bookmark to local storage and return (is_new, stored_data)."""
    bookmarks = load_realtime_bookmarks()

    if "timestamp" not in bookmark_data or not bookmark_data.get("timestamp"):
        bookmark_data = dict(bookmark_data)
        bookmark_data["timestamp"] = datetime.now().isoformat()

    existing = None
    for entry in bookmarks:
        if entry.get("tweet_id") == bookmark_data["tweet_id"]:
            existing = entry
            break

    if existing is None:
        # If we have a graphql_response, save it to cache and store only the filename
        if bookmark_data.get("graphql_response"):
            tweet_id = bookmark_data["tweet_id"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_filename = f"tweet_{tweet_id}_{timestamp}.json"
            cache_path = Path("graphql_cache") / cache_filename

            # Ensure cache directory exists
            cache_path.parent.mkdir(exist_ok=True)

            # Save the GraphQL response to file
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    bookmark_data["graphql_response"], f, indent=2, ensure_ascii=False
                )

            # Replace the full response with just the filename reference
            bookmark_data_to_save = bookmark_data.copy()
            bookmark_data_to_save["graphql_cache_file"] = str(cache_filename)
            bookmark_data_to_save.pop("graphql_response", None)
            bookmark_data_to_save.pop("force", None)

            logger.info(f"Cached GraphQL response for {tweet_id}")
        else:
            bookmark_data_to_save = bookmark_data.copy()
            bookmark_data_to_save.pop("force", None)

        bookmarks.append(bookmark_data_to_save)

        with open(REALTIME_BOOKMARKS_FILE, "w", encoding="utf-8") as f:
            json.dump(bookmarks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved bookmark for tweet {bookmark_data['tweet_id']}")
        return True, bookmark_data_to_save

    # Existing entry: optionally update cached GraphQL and metadata
    updated = False
    existing_payload = dict(existing)

    if bookmark_data.get("graphql_response"):
        tweet_id = bookmark_data["tweet_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_filename = f"tweet_{tweet_id}_{timestamp}.json"
        cache_path = Path("graphql_cache") / cache_filename

        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                bookmark_data["graphql_response"], f, indent=2, ensure_ascii=False
            )

        existing["graphql_cache_file"] = str(cache_filename)
        existing_payload["graphql_cache_file"] = str(cache_filename)
        updated = True

    if bookmark_data.get("tweet_data"):
        existing["tweet_data"] = bookmark_data["tweet_data"]
        updated = True

    if updated:
        with open(REALTIME_BOOKMARKS_FILE, "w", encoding="utf-8") as f:
            json.dump(bookmarks, f, indent=2, ensure_ascii=False)

    existing_payload.pop("graphql_response", None)
    return False, existing_payload


async def process_bookmark_async(bookmark_data: dict):
    """Process a bookmark asynchronously"""
    tweet_id = bookmark_data.get("tweet_id")
    if not tweet_id:
        logger.warning("Bookmark payload missing tweet_id; skipping")
        return

    logger.info(f"Processing bookmark {tweet_id}")

    db = get_metadata_db()
    db_entry = db.mark_bookmark_processing(tweet_id)

    try:
        # Ensure cache reference is indexed in the metadata DB when available
        cache_filename = bookmark_data.get("graphql_cache_file")
        if cache_filename and config.get("database.enabled", False):
            try:
                cache_file = Path("graphql_cache") / cache_filename
                db.upsert_graphql_cache_entry(tweet_id, str(cache_file))
            except Exception as exc:
                logger.debug(f"Failed to index graphql cache for {tweet_id}: {exc}")

        tweets_to_process = []
        loader = CacheLoader()

        # Prefer cached enhancements when available
        enhanced_map = loader.load_cached_enhancements([tweet_id])
        cache_file = None

        if tweet_id in enhanced_map:
            tweets_to_process.append(enhanced_map[tweet_id])
            cache_dir = Path("graphql_cache")
            for candidate in cache_dir.glob(f"tweet_{tweet_id}_*.json"):
                cache_file = candidate
                break
        elif cache_filename:
            try:
                cache_file = Path("graphql_cache") / cache_filename
                enhanced_tweet = loader._load_tweet_from_cache(cache_file, tweet_id)
                if enhanced_tweet:
                    tweets_to_process.append(enhanced_tweet)
            except Exception as exc:
                logger.debug(f"Could not load just-saved cache for {tweet_id}: {exc}")

        # Expand to entire thread when available
        if tweets_to_process and tweets_to_process[0].is_self_thread and cache_file:
            try:
                thread_tweets = loader.extract_all_thread_tweets_from_cache(cache_file)
                if len(thread_tweets) > 1:
                    tweets_to_process = thread_tweets
            except Exception as exc:
                logger.warning(f"Failed to extract full thread for {tweet_id}: {exc}")

        if not tweets_to_process:
            tweet_data = bookmark_data.get("tweet_data") or {}
            tweets_to_process.append(
                Tweet(
                    id=tweet_id,
                    full_text=tweet_data.get("text", ""),
                    created_at=bookmark_data.get(
                        "timestamp", datetime.now().isoformat()
                    ),
                    screen_name=tweet_data.get("author", "unknown"),
                    name=tweet_data.get("author", "Unknown"),
                )
            )

        # Build URL expansion mappings from enhanced tweets
        url_mappings: Dict[str, str] = {}
        for tw in tweets_to_process:
            if hasattr(tw, "url_mappings") and tw.url_mappings:
                for mapping in tw.url_mappings:
                    short_url = getattr(mapping, "short_url", None)
                    expanded_url = getattr(mapping, "expanded_url", None)
                    if short_url and expanded_url and short_url != expanded_url:
                        url_mappings[short_url] = expanded_url

        pipeline_stats = await run_pipeline_for_tweets(
            tweets_to_process, url_mappings=url_mappings or None, resume=False
        )
        if pipeline_stats:
            logger.info(
                "Pipeline processed %s/%s tweets for bookmark %s",
                pipeline_stats.processed_tweets,
                len(tweets_to_process),
                tweet_id,
            )

        maybe_cleanup_graphql_cache(tweets_to_process, pipeline_stats, logger=logger)

        # Persist processed state
        has_graphql = bool(
            bookmark_data.get("graphql_cache_file")
            or bookmark_data.get("graphql_response")
        )
        db.mark_bookmark_processed(tweet_id, with_graphql=has_graphql)

        try:

            def mark_processed(bookmarks: List[dict]) -> Tuple[bool, None]:
                for entry in bookmarks:
                    if entry.get("tweet_id") == tweet_id:
                        if entry.get("processed") is not True:
                            entry["processed"] = True
                            return True, None
                        return False, None
                return False, None

            await mutate_realtime_bookmarks(mark_processed)
        except Exception as exc:
            logger.debug(f"Failed to update processed flag for {tweet_id}: {exc}")

    except Exception as exc:
        logger.error(f"Error processing bookmark {tweet_id}: {exc}")
        failure_entry = db.mark_bookmark_failed(tweet_id, str(exc))
        if (
            failure_entry
            and failure_entry.status == "pending"
            and failure_entry.next_attempt_at
        ):
            asyncio.create_task(schedule_retry(tweet_id, failure_entry.next_attempt_at))
    else:
        logger.info(f"Bookmark {tweet_id} processed successfully")


# API Endpoints


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "XMarks API"}


@app.post("/api/bookmark", response_model=ProcessingStatus)
async def receive_bookmark(bookmark: BookmarkCapture):
    """Receive a bookmark from the browser extension"""
    try:
        # Convert to dict
        bookmark_data = bookmark.dict()

        # Ensure we have a baseline record for this tweet (no GraphQL yet)
        def ensure_bookmark(bookmarks: List[dict]) -> Tuple[bool, None]:
            for entry in bookmarks:
                if entry.get("tweet_id") == bookmark.tweet_id:
                    return False, None
            bookmarks.append(
                {
                    "tweet_id": bookmark.tweet_id,
                    "tweet_data": bookmark_data.get("tweet_data"),
                    "graphql_cache_file": bookmark_data.get("graphql_cache_file"),
                    "timestamp": bookmark_data.get("timestamp")
                    or datetime.now().isoformat(),
                    "source": bookmark_data.get("source"),
                    "processed": False,
                }
            )
            return True, None

        await mutate_realtime_bookmarks(ensure_bookmark)

        # Only proceed when GraphQL payload is present or cache file reference exists
        if not bookmark_data.get("graphql_response") and not bookmark_data.get(
            "graphql_cache_file"
        ):
            return ProcessingStatus(
                status="queued",
                message="Bookmark recorded; awaiting GraphQL detail",
                tweet_id=bookmark.tweet_id,
            )

        # Persist GraphQL payload
        cache_filename = bookmark_data.get("graphql_cache_file")
        if bookmark_data.get("graphql_response"):
            tweet_id = bookmark.tweet_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_filename = f"tweet_{tweet_id}_{timestamp}.json"
            cache_path = Path("graphql_cache") / cache_filename
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    bookmark_data["graphql_response"], f, indent=2, ensure_ascii=False
                )

        def attach_graphql(bookmarks: List[dict]) -> Tuple[bool, None]:
            for entry in bookmarks:
                if entry.get("tweet_id") == bookmark.tweet_id:
                    dirty = False
                    if entry.get("graphql_cache_file") != cache_filename:
                        entry["graphql_cache_file"] = cache_filename
                        dirty = True
                    if entry.get("processed") is not False:
                        entry["processed"] = False
                        dirty = True
                    if bookmark_data.get("tweet_data"):
                        if entry.get("tweet_data") != bookmark_data.get("tweet_data"):
                            entry["tweet_data"] = bookmark_data.get("tweet_data")
                            dirty = True
                    return dirty, None

            # Fallback: append if entry disappeared between mutations
            bookmarks.append(
                {
                    "tweet_id": bookmark.tweet_id,
                    "tweet_data": bookmark_data.get("tweet_data"),
                    "graphql_cache_file": cache_filename,
                    "timestamp": bookmark_data.get("timestamp")
                    or datetime.now().isoformat(),
                    "source": bookmark_data.get("source"),
                    "processed": False,
                }
            )
            return True, None

        await mutate_realtime_bookmarks(attach_graphql)

        # Queue for processing with GraphQL
        payload_for_queue = {
            "tweet_id": bookmark.tweet_id,
            "tweet_data": bookmark_data.get("tweet_data"),
            "graphql_cache_file": cache_filename,
            "timestamp": bookmark_data.get("timestamp") or datetime.now().isoformat(),
            "source": bookmark_data.get("source"),
            "force": True,
        }

        upsert_bookmark_queue_entry(payload_for_queue)
        await enqueue_bookmark_payload(payload_for_queue)

        return ProcessingStatus(
            status="accepted",
            message="Bookmark queued with GraphQL detail",
            tweet_id=bookmark.tweet_id,
            processed_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error receiving bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bookmarks")
async def get_bookmarks(limit: int = 100, processed: Optional[bool] = None):
    """Get recent bookmarks with optional processed filter"""
    bookmarks = load_realtime_bookmarks()
    # Optional filter by processed flag if present
    if processed is not None:
        filtered = [
            b for b in bookmarks if bool(b.get("processed", False)) == processed
        ]
    else:
        filtered = bookmarks
    return {"total": len(filtered), "bookmarks": filtered[-limit:]}


@app.get("/api/bookmarks/pending")
async def get_pending_bookmarks(limit: int = 100):
    """Return bookmarks that have not been processed yet."""
    db = get_metadata_db()
    entries = db.get_unprocessed_bookmarks(limit=limit)
    unprocessed = [entry for entry in entries if entry.status != "processed"]
    return {
        "total": len(unprocessed),
        "bookmarks": [serialize_bookmark_entry(entry) for entry in unprocessed],
    }


@app.post("/api/bookmarks/status")
async def bookmark_status(request: BookmarkStatusRequest):
    """Return processing status for a list of tweet IDs."""
    db = get_metadata_db()
    statuses = db.get_bookmark_statuses(request.tweet_ids)

    # Fill in defaults for tweet IDs that are unknown to the queue
    response = {}
    for tweet_id in request.tweet_ids:
        if tweet_id in statuses:
            response[tweet_id] = statuses[tweet_id]
        else:
            response[tweet_id] = {
                "status": "missing",
                "captured_at": None,
                "processed_at": None,
                "attempts": 0,
                "last_error": None,
                "next_attempt_at": None,
                "processed_with_graphql": False,
            }

    return {"statuses": response}


@app.post("/api/reprocess/{tweet_id}")
async def reprocess_tweet(
    tweet_id: str, no_resume: bool = Query(False, alias="no-resume")
):
    """Force reprocess a tweet (and thread if applicable) using cached GraphQL"""
    try:
        loader = CacheLoader()
        enhanced_map = loader.load_cached_enhancements([tweet_id])
        tweets_to_process = []
        cache_file = None

        if tweet_id in enhanced_map:
            tweets_to_process.append(enhanced_map[tweet_id])
            # Try find cache file
            cache_dir = Path("graphql_cache")
            for f in cache_dir.glob(f"tweet_{tweet_id}_*.json"):
                cache_file = f
                break
        else:
            # Fall back to scanning cache dir for this tweet
            cache_dir = Path("graphql_cache")
            for f in cache_dir.glob(f"tweet_{tweet_id}_*.json"):
                cache_file = f
                break
            if cache_file:
                tw = loader._load_tweet_from_cache(cache_file, tweet_id)
                if tw:
                    tweets_to_process.append(tw)

        if not tweets_to_process:
            raise HTTPException(
                status_code=404, detail="No cached GraphQL found for tweet"
            )

        # If part of a thread, load all
        if tweets_to_process[0].is_self_thread and cache_file:
            thread_tweets = loader.extract_all_thread_tweets_from_cache(cache_file)
            if len(thread_tweets) > 1:
                tweets_to_process = thread_tweets

        # Build URL mappings
        url_mappings: Dict[str, str] = {}
        for tw in tweets_to_process:
            if hasattr(tw, "url_mappings") and tw.url_mappings:
                for m in tw.url_mappings:
                    su = getattr(m, "short_url", None)
                    eu = getattr(m, "expanded_url", None)
                    if su and eu and su != eu:
                        url_mappings[su] = eu

        try:
            pipeline_stats = await run_pipeline_for_tweets(
                tweets_to_process,
                url_mappings=url_mappings or None,
                resume=not no_resume,
            )
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise

        maybe_cleanup_graphql_cache(tweets_to_process, pipeline_stats, logger=logger)

        try:
            db = get_metadata_db()
            db.mark_bookmark_processed(tweet_id, with_graphql=bool(cache_file))
        except Exception:
            pass

        return {"status": "ok", "reprocessed": len(tweets_to_process)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing tweet {tweet_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/triggers/github-stars")
async def trigger_github_stars(request: GitHubTriggerRequest):
    """Trigger the GitHub stars processor manually."""
    async with github_trigger_lock:
        try:
            processor = GitHubStarsProcessor()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize GitHub stars processor: {exc}")
            raise HTTPException(
                status_code=500, detail="Failed to initialize GitHub stars processor"
            )

        try:
            stats = await processor.fetch_and_process_starred_repos(
                limit=request.limit, resume=request.resume
            )
        except Exception as exc:
            logger.error(f"GitHub stars processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

        logger.info(
            "GitHub stars trigger completed: %s processed, %s skipped, %s errors",
            stats.updated,
            stats.skipped,
            stats.errors,
        )

        return {
            "status": "ok",
            "resume": request.resume,
            "limit": request.limit,
            "stats": serialize_processing_stats(stats),
        }


@app.post("/api/triggers/huggingface-likes")
async def trigger_huggingface_likes(request: HuggingFaceTriggerRequest):
    """Trigger the HuggingFace likes processor manually."""
    async with huggingface_trigger_lock:
        try:
            processor = HuggingFaceLikesProcessor()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize HuggingFace likes processor: {exc}")
            raise HTTPException(
                status_code=500, detail="Failed to initialize HuggingFace processor"
            )

        try:
            stats = await processor.fetch_and_process_liked_repos(
                limit=request.limit,
                resume=request.resume,
                include_models=request.include_models,
                include_datasets=request.include_datasets,
                include_spaces=request.include_spaces,
            )
        except Exception as exc:
            logger.error(f"HuggingFace likes processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

        logger.info(
            "HuggingFace likes trigger completed: %s processed, %s skipped, %s errors",
            stats.updated,
            stats.skipped,
            stats.errors,
        )

        return {
            "status": "ok",
            "resume": request.resume,
            "limit": request.limit,
            "include": {
                "models": request.include_models,
                "datasets": request.include_datasets,
                "spaces": request.include_spaces,
            },
            "stats": serialize_processing_stats(stats),
        }


@app.post("/api/process")
async def trigger_processing():
    """Trigger processing of all pending bookmarks"""
    try:
        bookmarks = load_realtime_bookmarks()

        # Create Tweet objects
        tweets = []
        for bookmark in bookmarks:
            tweet_id = bookmark["tweet_id"]
            tweet_data = bookmark.get("tweet_data", {})

            tweet = Tweet(
                id=tweet_id,
                full_text=tweet_data.get("text", ""),
                created_at=bookmark["timestamp"],
                screen_name=tweet_data.get("author", "unknown"),
                name=tweet_data.get("author", "Unknown"),
            )
            tweets.append(tweet)

        stats = await run_pipeline_for_tweets(tweets)

        try:
            db = get_metadata_db()
            for tweet in tweets:
                db.mark_bookmark_processed(tweet.id, with_graphql=False)
        except Exception:
            pass

        # Mark processed in realtime storage
        try:

            def mark_all_processed(entries: List[dict]) -> Tuple[bool, None]:
                dirty = False
                for entry in entries:
                    if entry.get("processed") is not True:
                        entry["processed"] = True
                        dirty = True
                return dirty, None

            await mutate_realtime_bookmarks(mark_all_processed)
        except Exception as e:
            logger.debug(f"Failed marking bookmarks processed: {e}")

        return {
            "status": "completed",
            "processed": stats.processed_tweets if stats else 0,
            "total": len(tweets),
        }

    except Exception as e:
        logger.error(f"Error processing bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/bookmark/{tweet_id}")
async def delete_bookmark(tweet_id: str, dry_run: bool = Query(False)):
    """Delete a bookmark and all its associated artifacts"""
    try:
        # Import the delete function from xmarks.py
        sys.path.insert(0, str(Path(__file__).parent))
        from xmarks import delete_tweet_artifacts
        
        # Perform deletion
        stats = delete_tweet_artifacts(tweet_id, dry_run)
        
        # Calculate totals
        total_files = (
            len(stats["tweet_files"]) + 
            len(stats["thread_files"]) + 
            len(stats["media_files"]) + 
            len(stats["transcript_files"]) +
            len(stats["cache_files"]) +
            len(stats["pdf_files"]) +
            len(stats["repo_files"])
        )
        
        # Remove from processing queue if present
        if not dry_run:
            try:
                db = get_metadata_db()
                db.delete_bookmark_entry(tweet_id)
            except Exception as e:
                logger.warning(f"Failed to remove from bookmark queue: {e}")
        
        return {
            "status": "ok" if not stats["errors"] else "partial",
            "tweet_id": tweet_id,
            "dry_run": dry_run,
            "deleted": {
                "total_files": total_files,
                "tweet_files": len(stats["tweet_files"]),
                "thread_files": len(stats["thread_files"]),
                "media_files": len(stats["media_files"]),
                "transcript_files": len(stats["transcript_files"]),
                "cache_files": len(stats["cache_files"]),
                "database_entries": stats["database_entries"]
            },
            "errors": stats["errors"][:10] if stats["errors"] else []
        }
        
    except Exception as e:
        logger.error(f"Error deleting bookmark {tweet_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    bookmarks = load_realtime_bookmarks()

    # Count by source
    sources = {}
    for bookmark in bookmarks:
        source = bookmark.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    # Count by date
    dates = {}
    for bookmark in bookmarks:
        date = bookmark["timestamp"][:10]
        dates[date] = dates.get(date, 0) + 1

    try:
        queue_counts = get_metadata_db().get_bookmark_queue_counts()
    except Exception:
        queue_counts = {"pending": 0, "processing": 0, "processed": 0, "failed": 0}

    return {
        "total_bookmarks": len(bookmarks),
        "by_source": sources,
        "by_date": dates,
        "queue_size": PROCESSING_QUEUE.qsize(),
        "queue_counts": queue_counts,
    }


# Background processor
async def background_processor():
    """Process bookmarks from queue in background"""
    while True:
        try:
            bookmark_data = await PROCESSING_QUEUE.get()
            tweet_id = bookmark_data.get("tweet_id")
            logger.debug(f"Dequeued bookmark {tweet_id} for processing")

            await process_bookmark_async(bookmark_data)

        except Exception as e:
            logger.error(f"Background processor error: {e}")
        finally:
            PROCESSING_QUEUE.task_done()


@app.on_event("startup")
async def startup_event():
    """Start background processor on startup"""
    asyncio.create_task(background_processor())
    await load_pending_bookmarks_from_db()
    logger.info("XMarks API server started")


def main():
    """Run the API server"""
    uvicorn.run(
        "xmarks_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
