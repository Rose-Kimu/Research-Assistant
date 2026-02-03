import feedparser
from urllib.parse import urlencode

base_url = "http://export.arxiv.org/api/query"

params = {
    "search_query": "all:large language models cognition",
    "max_results": 20
}

query_string = urlencode(params)
url = f"{base_url}?{query_string}"

feed = feedparser.parse(url)

for entry in feed.entries:
    print("Title:", entry.title)
    print("Published:", entry.published)
    print("Link:", entry.link)
    print("-" * 40)
