import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Dict, List
from urllib import request, error
from urllib.parse import urlparse


class _SubstackParser(HTMLParser):
    """Minimal HTML parser to grab title and first paragraph."""

    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.in_p = False
        self._got_p = False
        self.title = ""
        self.summary = ""

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag == "p" and not self._got_p:
            self.in_p = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag == "p" and self.in_p:
            self.in_p = False
            self._got_p = True

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.in_p:
            self.summary += data


class ContentExpander:
    """Expand external links and collect basic metadata."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _extract_urls(self, text: str) -> List[str]:
        """Return valid URLs from text without trailing punctuation."""
        pattern = re.compile(r'https?://[^\s\)\]\}]+')
        urls = [u.rstrip('.,;!?') for u in pattern.findall(text)]
        valid: List[str] = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                valid.append(parsed.geturl())
        return valid

    async def expand_links(self, tweet_text: str) -> Dict[str, Dict[str, str]]:
        """Extract URLs and expand supported ones."""
        expanded: Dict[str, Dict[str, str]] = {}
        for url in self._extract_urls(tweet_text):
            host = urlparse(url).netloc
            try:
                if 'github.com' in host:
                    expanded[url] = await self.expand_github(url)
                elif 'arxiv.org' in host:
                    expanded[url] = await self.expand_arxiv(url)
                elif 'substack.com' in host:
                    expanded[url] = await self.expand_substack(url)
            except Exception:
                expanded[url] = {"title": "", "summary": ""}
        return expanded

    async def expand_github(self, url: str) -> Dict[str, str]:
        """Fetch GitHub repo metadata and README summary."""
        try:
            path_parts = urlparse(url).path.strip('/').split('/')
            owner, repo = path_parts[0], path_parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            with request.urlopen(api_url, timeout=10) as resp:
                repo_data = json.loads(resp.read().decode())
            readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md"
            try:
                with request.urlopen(readme_url, timeout=10) as resp:
                    summary = resp.read().decode().strip()
            except error.URLError:
                summary = repo_data.get('description', '')
            title = repo_data.get('full_name', f"{owner}/{repo}")
            return {"title": title, "summary": summary}
        except error.URLError:
            return {"title": "", "summary": ""}

    async def expand_arxiv(self, url: str) -> Dict[str, str]:
        """Download and parse arXiv metadata."""
        try:
            match = re.search(r'arxiv.org/(?:abs|pdf)/([^/?#]+)', url)
            if not match:
                return {"title": "", "summary": ""}
            paper_id = match.group(1).replace('.pdf', '')
            api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
            with request.urlopen(api_url, timeout=10) as resp:
                xml_data = resp.read().decode()
            root = ET.fromstring(xml_data)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is None:
                return {"title": "", "summary": ""}
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            return {"title": title, "summary": summary}
        except (error.URLError, ET.ParseError, AttributeError):
            return {"title": "", "summary": ""}

    async def expand_substack(self, url: str) -> Dict[str, str]:
        """Grab title and first paragraph from Substack articles."""
        try:
            with request.urlopen(url, timeout=10) as resp:
                html = resp.read().decode()
            parser = _SubstackParser()
            parser.feed(html)
            return {"title": parser.title.strip(), "summary": parser.summary.strip()}
        except error.URLError:
            return {"title": "", "summary": ""}
