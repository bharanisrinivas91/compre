import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents import (
    fetch_commodity_data,
    research_commodity,
    make_procurement_decision,
    fetch_recent_news
)

# Define the state for our graph
class AgentState(TypedDict):
    commodity: str
    ticker: str
    data: pd.DataFrame
    research_summary: str
    news_headlines: str
    decision: str
    # The final structured report
    report: dict

# Define the nodes for our graph
def data_fetcher_node(state: AgentState) -> AgentState:
    print("---NODE: DATA FETCHER---")
    data = fetch_commodity_data(state['ticker'])
    return {"data": data}

def research_agent_node(state: AgentState) -> AgentState:
    print("---NODE: RESEARCH AGENT---")
    summary = research_commodity(state['commodity'])
    return {"research_summary": summary}

def news_fetcher_node(state: AgentState) -> AgentState:
    print("---NODE: NEWS FETCHER---")
    headlines = fetch_recent_news(state['commodity'])
    return {"news_headlines": headlines}

def decision_agent_node(state: AgentState) -> AgentState:
    print("---NODE: DECISION AGENT---")
    decision = make_procurement_decision(
        state['commodity'],
        state['data'],
        state['research_summary'],
        state['news_headlines']
    )
    return {"decision": decision}

def final_output_node(state: AgentState) -> AgentState:
    print("---NODE: FINAL OUTPUT---")
    # Format price data into a more API-friendly list of objects
    price_data = state['data'].tail(5).reset_index()
    # First, find and format the date column to a string
    date_col_name = None
    if 'index' in price_data.columns and pd.api.types.is_datetime64_any_dtype(price_data['index']):
        date_col_name = 'index'
        price_data.rename(columns={'index': 'Date'}, inplace=True)
        price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')
    elif 'Date' in price_data.columns and pd.api.types.is_datetime64_any_dtype(price_data['Date']):
        date_col_name = 'Date'
        price_data['Date'] = price_data['Date'].dt.strftime('%Y-%m-%d')

    # Now, convert all numeric columns to standard Python floats
    numeric_cols = price_data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        price_data[col] = price_data[col].astype(float).round(4)

    price_data_json = price_data.to_dict(orient='records')

    # Clean up news headlines into a clean list of strings
    raw_headlines = state['news_headlines'].strip().split('\n')
    headlines_list = [line.strip() for line in raw_headlines if line.strip() and not line.strip().startswith('---')]

    report = {
        "commodity": state['commodity'].title(),
        "recommendation_summary": state['decision'],
        "qualitative_research": state['research_summary'],
        "recent_news": headlines_list,
        "recent_price_data": price_data_json
    }
    return {"report": report}

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("data_fetcher", data_fetcher_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("news_fetcher", news_fetcher_node)
workflow.add_node("decision_agent", decision_agent_node)
workflow.add_node("final_output", final_output_node)

workflow.set_entry_point("data_fetcher")
workflow.add_edge("data_fetcher", "research_agent")
workflow.add_edge("data_fetcher", "news_fetcher")
workflow.add_edge("research_agent", "decision_agent")
workflow.add_edge("news_fetcher", "decision_agent")
workflow.add_edge("decision_agent", "final_output")
workflow.add_edge("final_output", END)

# Compile the graph
app_graph = workflow.compile()
