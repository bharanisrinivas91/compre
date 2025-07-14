import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def make_procurement_decision(commodity_name: str, data: pd.DataFrame, research_summary: str, news_headlines: str) -> str:
    """
    Analyzes data and research to make a procurement recommendation.

    Args:
        commodity_name (str): The name of the commodity.
        data (pd.DataFrame): A DataFrame with historical price data.
        research_summary (str): A summary of the market research.
        news_headlines (str): A string of recent news headlines.

    Returns:
        str: A final report including a procurement recommendation.
    """
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ_API_KEY not found. Please set it in your .env file."

    try:
        chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

        system_prompt = (
            "You are a seasoned procurement expert for a major construction company. "
            "Your task is to provide a clear, actionable 60-day procurement recommendation for a commodity. "
            "Your decision must be one of two options: 'Procure Now' or 'Wait'. "
            "Justify your decision by synthesizing the provided quantitative data and qualitative research."
        )
        
        data_summary = data.tail(10).to_string()

        human_prompt = (
            "Commodity: {commodity}\n\n"
            "Quantitative Data (Price history for the last 10 days):\n"
            "```\n{data_summary}\n```\n\n"
            "Qualitative Research Summary:\n"
            "---\n{research_summary}\n---\n\n"
            "Recent News Headlines:\n"
            "---\n{news_headlines}\n---\n\n"
            "Based on all the information above, provide your 60-day procurement recommendation. "
            "Start your response with 'Recommendation:' followed by 'Procure Now' or 'Wait'. "
            "Then, provide a section called 'Justification:' explaining your reasoning."
        )

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        
        chain = prompt | chat
        result = chain.invoke({
            "commodity": commodity_name,
            "data_summary": data_summary,
            "research_summary": research_summary,
            "news_headlines": news_headlines
        })
        
        return result.content
    except Exception as e:
        return f"An error occurred while making the decision: {e}"

if __name__ == '__main__':
    # This is a placeholder for testing; in a real run, data would be fetched dynamically.
    mock_data = pd.DataFrame({
        'Open': [3.5, 3.6, 3.55, 3.65, 3.7],
        'High': [3.6, 3.65, 3.6, 3.7, 3.75],
        'Low': [3.45, 3.55, 3.5, 3.6, 3.65],
        'Close': [3.58, 3.62, 3.58, 3.68, 3.72]
    }, index=pd.to_datetime(['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05']))
    
    mock_research = "Market sentiment for copper is currently bullish due to strong demand from the renewable energy sector and recent supply disruptions in Chile. Analysts predict a potential short-term price increase before a possible correction."
    mock_news = "1. Major Copper Mine in Peru Announces Production Halt.\n2. Tech Giant Unveils New EV Battery Requiring Less Copper."
    
    decision = make_procurement_decision("Copper", mock_data, mock_research, mock_news)
    print("--- Procurement Decision ---")
    print(decision)
    print("--------------------------")
