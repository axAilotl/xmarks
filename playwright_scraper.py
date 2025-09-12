import json
from typing import Dict, List, Optional

try:  # pragma: no cover - dependency may be missing during tests
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None


class TwitterScraper:
    """Scrape tweet threads and media using Playwright."""

    def __init__(self, cookies_file: str):
        self.cookies = self.load_cookies(cookies_file)

    @staticmethod
    def load_cookies(cookies_file: str) -> List[Dict]:
        """Load cookies from a JSON file exported from the browser."""
        with open(cookies_file, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        if not isinstance(cookies, list):
            raise ValueError("Cookie file must contain a list of cookies")
        return cookies

    async def get_full_thread(self, tweet_url: str) -> List[Dict]:
        """Scrape full thread if tweet is part of a self-thread."""
        thread_tweets: List[Dict] = []
        if async_playwright is None:  # pragma: no cover
            raise RuntimeError("playwright is not installed")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            await context.add_cookies(self.cookies)
            page = await context.new_page()

            async def handle_response(response):
                if "TweetDetail" in response.url:
                    try:
                        data = await response.json()
                        thread_tweets.extend(self.parse_thread_data(data))
                    except Exception:
                        pass

            page.on("response", handle_response)
            await page.goto(tweet_url)
            await page.wait_for_selector('[data-testid="tweet"]', timeout=10000)
            await browser.close()
        return thread_tweets

    def parse_thread_data(self, data: Dict) -> List[Dict]:
        """Parse tweets from a TweetDetail GraphQL response."""
        tweets: List[Dict] = []
        instructions = (
            data.get("data", {})
            .get("threaded_conversation_with_injections_v2", {})
            .get("instructions", [])
        )
        for inst in instructions:
            if inst.get("type") != "TimelineAddEntries":
                continue
            for entry in inst.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                tweet_results = item_content.get("tweet_results", {})
                result = tweet_results.get("result")
                if not result:
                    continue
                legacy = None
                if result.get("__typename") == "Tweet":
                    legacy = result.get("legacy")
                elif result.get("tweet") and result["tweet"].get("legacy"):
                    legacy = result["tweet"]["legacy"]
                if not legacy:
                    continue
                tweet = {
                    "id": legacy.get("id_str"),
                    "full_text": legacy.get("full_text"),
                    "created_at": legacy.get("created_at"),
                    "favorite_count": legacy.get("favorite_count"),
                    "retweet_count": legacy.get("retweet_count"),
                    "reply_count": legacy.get("reply_count"),
                    "entities": legacy.get("entities", {}),
                    "extended_entities": legacy.get("extended_entities", {}),
                }
                tweets.append(tweet)
        return tweets

    def get_media_urls(self, tweet_data: Dict) -> Dict[str, List[str]]:
        """Extract all media URLs from tweet data."""
        media = {"images": [], "videos": [], "gifs": []}
        media_entities = (
            tweet_data.get("extended_entities", {}).get("media", [])
        )
        for item in media_entities:
            mtype = item.get("type")
            if mtype == "photo":
                url = item.get("media_url_https") or item.get("media_url")
                if url:
                    media["images"].append(url)
            elif mtype == "video":
                variants = item.get("video_info", {}).get("variants", [])
                mp4s = [v for v in variants if v.get("content_type") == "video/mp4"]
                if mp4s:
                    url = max(mp4s, key=lambda v: v.get("bitrate", 0)).get("url")
                    if url:
                        media["videos"].append(url)
            elif mtype == "animated_gif":
                variants = item.get("video_info", {}).get("variants", [])
                mp4s = [v for v in variants if v.get("content_type") == "video/mp4"]
                if mp4s:
                    url = mp4s[0].get("url")
                    if url:
                        media["gifs"].append(url)
        return media
