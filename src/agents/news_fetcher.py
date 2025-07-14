import os
import re
import feedparser
import concurrent.futures
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# RSS feed URLs for commodity and financial news
COMMODITY_RSS_FEEDS = {
    "Reuters Commodities": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best&best-topics=commodities",
    "Bloomberg Markets": "https://www.bloomberg.com/markets/feeds/sitemap_news.xml",
    "Mining.com": "https://www.mining.com/feed/",
    "Kitco News": "https://www.kitco.com/rss/",
    "Investing.com Commodities": "https://www.investing.com/rss/commodities_Commodity_News.rss",
    "S&P Global": "https://www.spglobal.com/commodityinsights/en/rss-feed/news",
    "Financial Times Commodities": "https://www.ft.com/commodities?format=rss",
}

def fetch_recent_news(commodity_name: str) -> str:
    """
    Fetches recent news articles for a given commodity using multiple sources:
    1. Event Registry API for high-quality curated news
    2. RSS feeds from major financial and commodity news sources

    Args:
        commodity_name (str): The name of the commodity to search for (e.g., 'Copper').

    Returns:
        str: A formatted string of news headlines from multiple sources.
    """
    # Combine news from multiple sources
    event_registry_news = fetch_from_event_registry(commodity_name)
    rss_news = fetch_from_rss_feeds(commodity_name)
    
    # Combine and format all news
    all_news = []
    
    # Add Event Registry news if available
    if not event_registry_news.startswith("Error") and not event_registry_news.startswith("No "):
        all_news.extend(parse_event_registry_news(event_registry_news))
    
    # Add RSS feed news
    all_news.extend(rss_news)
    
    # Sort by date (newest first) and deduplicate
    all_news = deduplicate_news(all_news)
    all_news = sorted(all_news, key=lambda x: x.get('date', datetime.now()), reverse=True)
    
    # Format the combined news
    if not all_news:
        return f"No relevant news found for {commodity_name} in the last 7 days."
    
    formatted_news = f"\n--- Recent {commodity_name} Market News ---\n"
    for i, article in enumerate(all_news[:10], 1):  # Show top 10 news items
        title = article.get('title', 'No Title')
        source = article.get('source', 'Unknown Source')
        date_str = article.get('date', datetime.now()).strftime('%Y-%m-%d')
        formatted_news += f"{i}. {title} (Source: {source}, Date: {date_str})\n"
    
    return formatted_news

def fetch_from_event_registry(commodity_name: str) -> str:
    """
    Fetches news from Event Registry API.
    """
    load_dotenv()
    api_key = os.getenv("EVENT_REGISTRY_API_KEY")
    if not api_key:
        return "Error: EVENT_REGISTRY_API_KEY not found. Please set it in your .env file."

    try:
        er = EventRegistry(apiKey=api_key)

        # Get the URI for the commodity concept
        concept_uri = er.getConceptUri(commodity_name)
        if not concept_uri:
            return f"No concept found for '{commodity_name}' in Event Registry."

        # Get the URI for the Business category
        business_category_uri = er.getCategoryUri("Business")

        # Create a more specific query
        q = QueryArticlesIter(
            conceptUri=concept_uri,
            categoryUri=business_category_uri,
            keywords=QueryItems.AND([commodity_name, "price", "market", "supply", "demand"]),
            dateStart=(date.today() - timedelta(days=7)).isoformat(),
            dataType=["news"]
        )

        # Execute query and format results
        articles = list(q.execQuery(er, sortBy="date", maxItems=10))

        if not articles:
            return f"No relevant news articles found for {commodity_name} in Event Registry."

        formatted_news = "\n--- Recent News Headlines (from Event Registry) ---\n"
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            source = article.get('source', {}).get('title', 'Unknown Source')
            date_str = article.get('date', '')
            formatted_news += f"{i}. {title} (Source: {source}, Date: {date_str})\n"
        
        return formatted_news

    except Exception as e:
        if "Invalid API key" in str(e):
            return "Error: Invalid Event Registry API key. Please check your .env file."
        return f"An error occurred while fetching news from Event Registry: {e}"

def parse_event_registry_news(news_text: str) -> List[Dict[str, Any]]:
    """
    Parse the formatted Event Registry news text back into structured data.
    """
    news_items = []
    
    # Skip the header line
    lines = news_text.strip().split('\n')[1:]
    
    for line in lines:
        if line.startswith('---'):
            continue
            
        # Extract information using regex
        match = re.match(r'\d+\. (.+) \(Source: (.+), Date: (.+)\)', line)
        if match:
            title, source, date_str = match.groups()
            try:
                news_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                news_date = datetime.now()
                
            news_items.append({
                'title': title,
                'source': source,
                'date': news_date,
                'origin': 'Event Registry'
            })
    
    return news_items

def fetch_from_rss_feeds(commodity_name: str) -> List[Dict[str, Any]]:
    """
    Fetch and parse news from multiple RSS feeds related to commodities.
    """
    news_items = []
    commodity_keywords = [commodity_name.lower()]
    
    # Add related keywords for better matching
    if commodity_name.lower() == 'copper':
        commodity_keywords.extend(['metal', 'mining', 'minerals'])
    elif commodity_name.lower() == 'crude oil':
        commodity_keywords.extend(['oil', 'petroleum', 'energy'])
    elif commodity_name.lower() == 'gold':
        commodity_keywords.extend(['precious metal', 'bullion'])
    
    # Use ThreadPoolExecutor to fetch feeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_feed = {executor.submit(fetch_single_feed, url, source_name, commodity_keywords): 
                         (url, source_name) for source_name, url in COMMODITY_RSS_FEEDS.items()}
        
        for future in concurrent.futures.as_completed(future_to_feed):
            url, source_name = future_to_feed[future]
            try:
                feed_news = future.result()
                news_items.extend(feed_news)
            except Exception as e:
                print(f"Error fetching {source_name}: {e}")
    
    return news_items

def fetch_single_feed(url: str, source_name: str, keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch and parse a single RSS feed, filtering for relevant commodity news.
    """
    try:
        feed = feedparser.parse(url)
        news_items = []
        
        for entry in feed.entries[:20]:  # Limit to first 20 entries
            title = entry.get('title', '')
            
            # Check if any keyword is in the title
            if any(keyword in title.lower() for keyword in keywords):
                # Extract date
                if 'published_parsed' in entry:
                    news_date = datetime(*entry.published_parsed[:6])
                elif 'updated_parsed' in entry:
                    news_date = datetime(*entry.updated_parsed[:6])
                else:
                    news_date = datetime.now()
                
                # Only include news from the last 7 days
                if (datetime.now() - news_date).days <= 7:
                    news_items.append({
                        'title': title,
                        'source': source_name,
                        'date': news_date,
                        'origin': 'RSS'
                    })
        
        return news_items
    except Exception as e:
        print(f"Error parsing feed {url}: {e}")
        return []

def deduplicate_news(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate news items based on similar titles.
    """
    unique_news = []
    seen_titles = set()
    
    for item in news_items:
        # Normalize the title by removing punctuation and converting to lowercase
        normalized_title = re.sub(r'[^\w\s]', '', item['title'].lower())
        
        # Check if we've seen a similar title
        if not any(normalized_title in seen_title for seen_title in seen_titles):
            seen_titles.add(normalized_title)
            unique_news.append(item)
    
    return unique_news

if __name__ == '__main__':
    # Example usage
    commodity = "Copper"
    news = fetch_recent_news(commodity)
    print(news)
