"""
GraphQL Engine - Core Twitter data collection via Playwright
Extracted and cleaned from real_thread_and_url_fix.py
"""

import asyncio
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from .data_models import (
    Tweet,
    ThreadInfo,
    ProcessingStats,
    extract_full_text_from_result,
)

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

logger = logging.getLogger(__name__)


class GraphQLEngine:
    """Core GraphQL data collection engine using Playwright"""

    def __init__(self, cookies_file: str = "twitter_cookies.json"):
        self.cookies_file = cookies_file

        # Conservative rate limiting: 45 requests per 15 minutes
        self.requests_per_window = 45
        self.window_duration = 900  # 15 minutes in seconds
        self.base_interval = (
            self.window_duration / self.requests_per_window
        )  # ~20 seconds

        # Cache for GraphQL responses
        self.cache_dir = Path("graphql_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Rate limiting tracking
        self.request_times: List[float] = []
        self.last_request = 0

        # Data storage
        self.enhanced_tweets: Dict[str, Dict] = {}
        self.thread_data: Dict[str, List[Dict]] = {}
        self.url_mappings: Dict[str, str] = {}

    async def _enforce_rate_limit(self):
        """Enforce Twitter's rate limits with randomization without blocking the loop."""
        import random

        current_time = time.time()

        # Remove requests older than 15 minutes
        self.request_times = [
            t for t in self.request_times if current_time - t < self.window_duration
        ]

        # Check if at limit
        if len(self.request_times) >= self.requests_per_window:
            oldest_request = min(self.request_times)
            wait_until = oldest_request + self.window_duration + 10  # 10 second buffer
            wait_time = wait_until - current_time

            if wait_time > 0:
                logger.info(
                    f"⏰ Rate limit reached ({len(self.request_times)}/{self.requests_per_window}), waiting {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
                current_time = time.time()

        # Randomized interval: base ± 30%
        randomization = random.uniform(-0.3, 0.3)
        interval = self.base_interval * (1 + randomization)

        # Ensure minimum time since last request
        time_since_last = current_time - self.last_request
        if time_since_last < interval:
            sleep_time = interval - time_since_last
            logger.debug(f"⏱️ Waiting: {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)

        # Record this request
        self.last_request = time.time()
        self.request_times.append(self.last_request)

        # Show rate limit status
        window_requests = len(self.request_times)
        logger.debug(
            f"📊 Rate limit: {window_requests}/{self.requests_per_window} requests in window"
        )

    async def collect_graphql_data(
        self, tweets: List[Tweet], resume: bool = True
    ) -> ProcessingStats:
        """Collect GraphQL data for tweets using Playwright"""
        if async_playwright is None:
            logger.error("❌ Playwright not installed!")
            return ProcessingStats()

        stats = ProcessingStats()

        # Handle resume functionality
        if resume:
            cached_ids = self.get_cached_tweet_ids()
            original_count = len(tweets)

            uncached_tweets = [tweet for tweet in tweets if tweet.id not in cached_ids]
            skipped_count = original_count - len(uncached_tweets)

            if skipped_count > 0:
                logger.info(f"⏭️ Resume: Skipping {skipped_count} cached tweets")
                logger.info(f"🔄 Remaining: {len(uncached_tweets)} tweets")

            if not uncached_tweets:
                logger.info("✅ All tweets already cached!")
                return stats

            tweets = uncached_tweets

        logger.info(f"🚀 Starting GraphQL collection for {len(tweets)} tweets")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            context = await self._setup_browser_context(browser)

            if not context:
                await browser.close()
                return stats

            try:
                page = await context.new_page()

                # Set headers
                await page.set_extra_http_headers(
                    {
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    }
                )

                # Set up GraphQL response capture
                graphql_responses = []

                async def handle_response(response):
                    url = response.url
                    if (
                        "TweetDetail" in url or "threaded_conversation" in url
                    ) and response.status == 200:
                        try:
                            data = await response.json()
                            graphql_responses.append(data)
                            logger.debug(f"📡 Captured GraphQL response")
                            self._cache_graphql_response(data, url)
                        except Exception as e:
                            logger.debug(f"Failed to parse GraphQL response: {e}")

                page.on("response", handle_response)

                # Test cookie validity
                logger.info("🧪 Testing cookies...")
                try:
                    await page.goto("https://x.com/home", timeout=10000)
                    await asyncio.sleep(2)

                    title = await page.title()
                    if "login" in title.lower() or "sign" in title.lower():
                        logger.error("❌ Cookies expired - update twitter_cookies.json")
                        await browser.close()
                        return stats
                    else:
                        logger.info("✅ Cookies valid")
                except Exception as e:
                    logger.warning(f"⚠️ Cookie test failed: {e}")

                # Process tweets
                consecutive_failures = 0
                start_time = time.time()

                for i, tweet in enumerate(tweets, 1):
                    try:
                        await self._enforce_rate_limit()

                        # Progress
                        percent = (i / len(tweets)) * 100
                        elapsed = time.time() - start_time

                        if i > 1:
                            rate = (i - 1) / elapsed
                            remaining = len(tweets) - i
                            eta_mins = (remaining / rate / 60) if rate > 0 else 0
                            logger.info(
                                f"🔍 [{i}/{len(tweets)}] ({percent:.1f}%) {tweet.id} | ETA: {eta_mins:.1f}m"
                            )
                        else:
                            logger.info(
                                f"🔍 [{i}/{len(tweets)}] ({percent:.1f}%) {tweet.id}"
                            )

                        # Try multiple access strategies
                        success = await self._try_access_strategies(
                            page, tweet, graphql_responses
                        )

                        if success:
                            stats.successful += 1
                            consecutive_failures = 0
                        else:
                            stats.failed += 1
                            consecutive_failures += 1

                        # Handle consecutive failures
                        if consecutive_failures >= 10:
                            logger.error(
                                "🚨 10 consecutive failures - possibly rate limited or cookies expired"
                            )
                            logger.info("⏳ Waiting 1 hour...")
                            await asyncio.sleep(3600)
                            consecutive_failures = 0

                        # Process GraphQL responses
                        for response_data in graphql_responses:
                            self._extract_tweet_data(response_data, tweet.id)

                        graphql_responses.clear()
                        stats.total_processed += 1

                    except Exception as e:
                        logger.warning(f"⚠️ Error processing {tweet.id}: {e}")
                        stats.failed += 1
                        stats.total_processed += 1
                        continue

                # Final stats
                success_rate = (
                    (stats.successful / stats.total_processed * 100)
                    if stats.total_processed > 0
                    else 0
                )
                logger.info(
                    f"📊 Collection Complete: {stats.successful}/{stats.total_processed} ({success_rate:.1f}%)"
                )

            finally:
                await browser.close()

        return stats

    async def _try_access_strategies(self, page, tweet: Tweet, responses: List) -> bool:
        """Try multiple strategies to access tweet data"""

        screen_name = tweet.screen_name or "i/web"
        strategies = [
            f"https://x.com/{screen_name}/status/{tweet.id}",
            f"https://twitter.com/{screen_name}/status/{tweet.id}",
            f"https://x.com/i/web/status/{tweet.id}",
            f"https://twitter.com/i/web/status/{tweet.id}",
            f"https://x.com/i/status/{tweet.id}",
            f"https://twitter.com/i/status/{tweet.id}",
        ]

        for url in strategies:
            try:
                logger.debug(f"Trying: {url}")
                await page.goto(url, wait_until="load", timeout=10000)
                await asyncio.sleep(3)

                if responses:
                    logger.debug("✅ Got GraphQL data")
                    return True

            except Exception as e:
                logger.debug(f"Strategy failed: {e}")
                continue

        return False

    async def _setup_browser_context(self, browser):
        """Setup browser with Twitter cookies"""
        try:
            cookies_path = Path(self.cookies_file)
            if not cookies_path.exists():
                logger.error(f"❌ Cookies file not found: {self.cookies_file}")
                return None

            with open(cookies_path, "r") as f:
                cookies = json.load(f)

            cleaned_cookies = self._clean_cookies(cookies)
            context = await browser.new_context()
            await context.add_cookies(cleaned_cookies)

            logger.info(f"🍪 Loaded {len(cleaned_cookies)} cookies")
            return context

        except Exception as e:
            logger.error(f"❌ Failed to setup browser: {e}")
            return None

    def _clean_cookies(self, cookies: List[Dict]) -> List[Dict]:
        """Clean cookies for Playwright compatibility"""
        cleaned = []

        for cookie in cookies:
            cleaned_cookie = {}

            # Copy basic fields
            for key in ["name", "value", "domain", "path"]:
                if key in cookie:
                    cleaned_cookie[key] = cookie[key]

            # Handle expires
            if "expirationDate" in cookie:
                cleaned_cookie["expires"] = int(cookie["expirationDate"])

            # Handle boolean flags
            if cookie.get("httpOnly", False):
                cleaned_cookie["httpOnly"] = True
            if cookie.get("secure", False):
                cleaned_cookie["secure"] = True

            # Clean sameSite
            same_site = cookie.get("sameSite")
            if same_site:
                same_site = same_site.lower()
                if same_site == "no_restriction":
                    cleaned_cookie["sameSite"] = "None"
                elif same_site in ["lax", "strict"]:
                    cleaned_cookie["sameSite"] = same_site.capitalize()

            cleaned.append(cleaned_cookie)

        return cleaned

    def _extract_tweet_data(self, graphql_data: Dict, target_tweet_id: str):
        """Extract tweet data from GraphQL response"""
        try:
            instructions = (
                graphql_data.get("data", {})
                .get("threaded_conversation_with_injections_v2", {})
                .get("instructions", [])
            )

            for instruction in instructions:
                if instruction.get("type") != "TimelineAddEntries":
                    continue

                for entry in instruction.get("entries", []):
                    tweet_info = self._extract_tweet_from_entry(entry)
                    if tweet_info and tweet_info.get("id"):
                        tweet_id = tweet_info["id"]

                        # Store enhanced data
                        self.enhanced_tweets[tweet_id] = tweet_info

                        # Extract URL mappings
                        if "urls" in tweet_info:
                            for url_data in tweet_info["urls"]:
                                tco_url = url_data.get("url")
                                expanded_url = url_data.get("expanded_url")
                                if tco_url and expanded_url:
                                    self.url_mappings[tco_url] = expanded_url

                        # Check for thread indicators
                        if tweet_info.get("display_type") == "SelfThread":
                            conversation_id = tweet_info.get("conversation_id")
                            if conversation_id:
                                if conversation_id not in self.thread_data:
                                    self.thread_data[conversation_id] = []
                                self.thread_data[conversation_id].append(tweet_info)

        except Exception as e:
            logger.debug(f"Error extracting tweet data: {e}")

    def _extract_tweet_from_entry(self, entry: Dict) -> Optional[Dict]:
        """Extract tweet info from timeline entry"""
        try:
            content = entry.get("content", {})
            item_content = content.get("itemContent", {})
            tweet_results = item_content.get("tweet_results", {})
            result = tweet_results.get("result", {})

            if not result or result.get("__typename") != "Tweet":
                return None

            legacy = result.get("legacy", {})
            if not legacy:
                return None

            # Extract comprehensive data
            full_text = extract_full_text_from_result(result)

            note_result = (
                (result.get("note_tweet") or {})
                .get("note_tweet_results", {})
                .get("result", {})
            )
            note_urls = []
            if isinstance(note_result, dict):
                entity_set = note_result.get("entity_set", {}) or {}
                note_urls = entity_set.get("urls", []) or []

            tweet_info = {
                "id": legacy.get("id_str"),
                "full_text": full_text,
                "created_at": legacy.get("created_at"),
                "favorite_count": legacy.get("favorite_count", 0),
                "retweet_count": legacy.get("retweet_count", 0),
                "reply_count": legacy.get("reply_count", 0),
                "display_type": result.get("tweetDisplayType"),
                "is_self_thread": result.get("tweetDisplayType") == "SelfThread",
                "conversation_id": legacy.get("conversation_id_str"),
                # URLs and media
                "urls": (legacy.get("entities", {}).get("urls", []) or []) + note_urls,
                "media": legacy.get("extended_entities", {}).get("media", []),
                # User info
                "screen_name": result.get("core", {})
                .get("user_results", {})
                .get("result", {})
                .get("legacy", {})
                .get("screen_name"),
                "name": result.get("core", {})
                .get("user_results", {})
                .get("result", {})
                .get("legacy", {})
                .get("name"),
            }

            return tweet_info

        except Exception as e:
            logger.debug(f"Error extracting tweet: {e}")
            return None

    def _cache_graphql_response(self, data: Dict, url: str):
        """Cache raw GraphQL response"""
        try:
            tweet_id_match = re.search(r"focalTweetId%22%3A%22(\\d+)%22", url)
            if tweet_id_match:
                tweet_id = tweet_id_match.group(1)
                cache_filename = f"tweet_{tweet_id}_{int(time.time())}.json"
                cache_path = self.cache_dir / cache_filename

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.debug(f"💾 Cached: {cache_filename}")
        except Exception as e:
            logger.debug(f"Failed to cache response: {e}")

    def get_cached_tweet_ids(self) -> Set[str]:
        """Get set of tweet IDs with cached GraphQL data"""
        cached_ids = set()

        if not self.cache_dir.exists():
            return cached_ids

        cache_files = list(self.cache_dir.glob("tweet_*.json"))

        # Group by tweet ID, keep latest
        tweet_files = {}
        for cache_file in cache_files:
            try:
                filename = cache_file.stem
                parts = filename.split("_")
                if len(parts) >= 3 and parts[0] == "tweet":
                    tweet_id = parts[1]
                    timestamp = int(parts[2])

                    if (
                        tweet_id not in tweet_files
                        or timestamp > tweet_files[tweet_id]["timestamp"]
                    ):
                        tweet_files[tweet_id] = {
                            "file": cache_file,
                            "timestamp": timestamp,
                        }
            except Exception:
                continue

        # Validate files
        for tweet_id, file_info in tweet_files.items():
            try:
                with open(file_info["file"], "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and ("data" in data or "errors" in data):
                        cached_ids.add(tweet_id)
            except Exception:
                # Remove corrupted files
                try:
                    file_info["file"].unlink()
                except:
                    pass

        logger.info(f"📂 Found {len(cached_ids)} valid cached tweets")
        return cached_ids

    def get_enhanced_tweets(self) -> Dict[str, Dict]:
        """Get enhanced tweet data"""
        return self.enhanced_tweets

    def get_thread_data(self) -> Dict[str, List[Dict]]:
        """Get thread data"""
        return self.thread_data

    def get_url_mappings(self) -> Dict[str, str]:
        """Get URL mappings"""
        return self.url_mappings

    def get_statistics(self) -> Dict:
        """Get enhancement statistics"""
        cache_files = len(list(self.cache_dir.glob("*.json")))

        return {
            "tweets_enhanced": len(self.enhanced_tweets),
            "url_mappings_found": len(self.url_mappings),
            "threads_detected": len(self.thread_data),
            "graphql_responses_cached": cache_files,
            "github_urls": sum(
                1 for url in self.url_mappings.values() if "github.com" in url
            ),
            "arxiv_urls": sum(
                1 for url in self.url_mappings.values() if "arxiv.org" in url
            ),
            "huggingface_urls": sum(
                1 for url in self.url_mappings.values() if "huggingface.co" in url
            ),
        }
