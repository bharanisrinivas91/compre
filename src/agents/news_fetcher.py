import os
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import date, timedelta
from dotenv import load_dotenv

def fetch_recent_news(commodity_name: str) -> str:
    """
    Fetches recent news articles for a given commodity using Event Registry.

    Args:
        commodity_name (str): The name of the commodity to search for (e.g., 'Copper').

    Returns:
        str: A formatted string of the top 5 news headlines, or an error message.
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
            dateStart=(date.today() - timedelta(days=5)).isoformat(),
            dataType=["news"]
        )

        # Execute query and format results
        articles = list(q.execQuery(er, sortBy="date", maxItems=5))

        if not articles:
            return f"No relevant news articles found for {commodity_name} in the last 5 days."

        formatted_news = "\n--- Recent News Headlines (from Event Registry) ---\n"
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No Title')
            source = article.get('source', {}).get('title', 'Unknown Source')
            formatted_news += f"{i}. {title} (Source: {source})\n"
        
        return formatted_news

    except Exception as e:
        if "Invalid API key" in str(e):
            return "Error: Invalid Event Registry API key. Please check your .env file."
        return f"An error occurred while fetching news from Event Registry: {e}"

if __name__ == '__main__':
    # Example usage
    commodity = "Copper"
    news = fetch_recent_news(commodity)
    print(news)
