import asyncio
import pytest
from content_expander import ContentExpander
from urllib import request


def test_extract_urls_no_trailing_punctuation():
    ce = ContentExpander()
    urls = ce._extract_urls('Check https://github.com/foo/bar, and https://arxiv.org/abs/1234.5678.')
    assert urls == ['https://github.com/foo/bar', 'https://arxiv.org/abs/1234.5678']


def test_expand_links_dispatch(monkeypatch):
    ce = ContentExpander()

    async def fake_github(url):
        return {'title': 'gh', 'summary': ''}

    async def fake_arxiv(url):
        return {'title': 'ax', 'summary': ''}

    async def fake_substack(url):
        return {'title': 'sb', 'summary': ''}

    monkeypatch.setattr(ce, 'expand_github', fake_github)
    monkeypatch.setattr(ce, 'expand_arxiv', fake_arxiv)
    monkeypatch.setattr(ce, 'expand_substack', fake_substack)

    text = (
        'Repo https://github.com/foo/bar end '
        'Paper https://arxiv.org/abs/1234.5678 more '
        'Blog https://my.substack.com/p/post'
    )
    expanded = asyncio.run(ce.expand_links(text))
    assert len(expanded) == 3
    assert expanded['https://github.com/foo/bar']['title'] == 'gh'
    assert expanded['https://arxiv.org/abs/1234.5678']['title'] == 'ax'
    assert expanded['https://my.substack.com/p/post']['title'] == 'sb'


class FakeResp:
    def __init__(self, text=None, data=None):
        self._text = text
        self._data = data

    def read(self):
        if self._text is not None:
            return self._text.encode()
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_expand_github(monkeypatch):
    def fake_urlopen(url, timeout=10):
        if 'api.github.com' in url:
            return FakeResp(text='{"full_name": "foo/bar", "description": "desc"}')
        return FakeResp(text='README')

    monkeypatch.setattr(request, 'urlopen', fake_urlopen)
    ce = ContentExpander()
    info = asyncio.run(ce.expand_github('https://github.com/foo/bar'))
    assert info['title'] == 'foo/bar'
    assert 'README' in info['summary']


def test_expand_arxiv(monkeypatch):
    xml_resp = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry><title>Test Paper</title><summary>Summary text.</summary></entry>'
        '</feed>'
    )

    def fake_urlopen(url, timeout=10):
        return FakeResp(text=xml_resp)

    monkeypatch.setattr(request, 'urlopen', fake_urlopen)
    ce = ContentExpander()
    info = asyncio.run(ce.expand_arxiv('https://arxiv.org/abs/1234.5678'))
    assert info['title'] == 'Test Paper'
    assert info['summary'] == 'Summary text.'


def test_expand_substack(monkeypatch):
    html_resp = '<html><head><title>Substack Title</title></head><body><p>First para.</p></body></html>'

    def fake_urlopen(url, timeout=10):
        return FakeResp(text=html_resp)

    monkeypatch.setattr(request, 'urlopen', fake_urlopen)
    ce = ContentExpander()
    info = asyncio.run(ce.expand_substack('https://newsletter.substack.com/p/post'))
    assert info['title'] == 'Substack Title'
    assert info['summary'] == 'First para.'
