import json
from pathlib import Path

import pytest

from playwright_scraper import TwitterScraper


def test_load_cookies(tmp_path: Path):
    cookies = [
        {
            "name": "auth_token",
            "value": "abc",
            "domain": ".twitter.com",
            "path": "/",
            "expires": 1700000000,
        }
    ]
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text(json.dumps(cookies), encoding="utf-8")
    scraper = TwitterScraper(str(cookie_file))
    assert scraper.cookies == cookies


def test_parse_thread_data():
    sample = {
        "data": {
            "threaded_conversation_with_injections_v2": {
                "instructions": [
                    {
                        "type": "TimelineAddEntries",
                        "entries": [
                            {
                                "entryId": "tweet-1",
                                "content": {
                                    "itemContent": {
                                        "tweet_results": {
                                            "result": {
                                                "__typename": "Tweet",
                                                "legacy": {
                                                    "id_str": "1",
                                                    "full_text": "Hello world",
                                                    "created_at": "Mon",
                                                    "favorite_count": 5,
                                                    "retweet_count": 2,
                                                    "reply_count": 1,
                                                },
                                            }
                                        }
                                    }
                                },
                            },
                            {
                                "entryId": "tweet-2",
                                "content": {
                                    "itemContent": {
                                        "tweet_results": {
                                            "result": {
                                                "__typename": "Tweet",
                                                "legacy": {
                                                    "id_str": "2",
                                                    "full_text": "Second tweet",
                                                    "created_at": "Tue",
                                                    "favorite_count": 3,
                                                    "retweet_count": 1,
                                                    "reply_count": 0,
                                                },
                                            }
                                        }
                                    }
                                },
                            },
                        ],
                    }
                ]
            }
        }
    }
    scraper = TwitterScraper.__new__(TwitterScraper)
    tweets = scraper.parse_thread_data(sample)
    assert len(tweets) == 2
    assert tweets[0]["full_text"] == "Hello world"
    assert tweets[1]["id"] == "2"


def test_get_media_urls():
    tweet = {
        "extended_entities": {
            "media": [
                {
                    "type": "photo",
                    "media_url_https": "https://pbs.twimg.com/media/abc.jpg",
                },
                {
                    "type": "video",
                    "video_info": {
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bitrate": 1000000,
                                "url": "https://video1.mp4",
                            }
                        ]
                    },
                },
                {
                    "type": "animated_gif",
                    "video_info": {
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "url": "https://gif1.mp4",
                            }
                        ]
                    },
                },
            ]
        }
    }
    scraper = TwitterScraper.__new__(TwitterScraper)
    media = scraper.get_media_urls(tweet)
    assert media["images"] == ["https://pbs.twimg.com/media/abc.jpg"]
    assert media["videos"] == ["https://video1.mp4"]
    assert media["gifs"] == ["https://gif1.mp4"]
