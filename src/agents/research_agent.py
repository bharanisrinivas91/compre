import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def research_commodity(commodity_name: str) -> str:
    """
    Performs research on a given commodity using the Groq LLM.

    Args:
        commodity_name (str): The name of the commodity to research (e.g., 'Copper').

    Returns:
        str: A research summary for the commodity.
    """
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ_API_KEY not found. Please set it in your .env file."

    try:
        chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

        system_prompt = "You are a world-class financial analyst and commodity expert. Your goal is to provide a concise yet comprehensive summary of the current market situation for a given commodity."
        human_prompt = (
            "Please provide a research summary for {commodity}. Include the following points:\n"
            "1. Price-driving news and events from the last 120 days.\n"
            "2. Current market sentiment (bullish, bearish, or neutral) and the reasons behind it.\n"
            "3. Key supply and demand factors at play.\n"
            "4. A brief outlook for the next 60 days."
        )

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
        
        chain = prompt | chat
        result = chain.invoke({"commodity": commodity_name})
        
        return result.content
    except Exception as e:
        return f"An error occurred during the research: {e}"

if __name__ == '__main__':
    # Example usage: Research Copper
    commodity = "Copper"
    research_summary = research_commodity(commodity)
    print(f"--- Research Summary for {commodity} ---\n")
    print(research_summary)
    print("\n--------------------------------------\n")

    # Example usage: Research Steel
    commodity = "Steel"
    research_summary = research_commodity(commodity)
    print(f"--- Research Summary for {commodity} ---\n")
    print(research_summary)
    print("\n--------------------------------------\n")
