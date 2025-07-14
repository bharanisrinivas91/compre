import pandas as pd
from typing import TypedDict, Annotated, List
import operator
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
    # The final output of the graph
    result: str

# Define the nodes for our graph
def data_fetcher_node(state: AgentState) -> AgentState:
    """Fetches historical data for the commodity."""
    print("---NODE: DATA FETCHER---")
    commodity = state['commodity']
    ticker = state['ticker']
    
    data = fetch_commodity_data(ticker)
    print(f"Successfully fetched {len(data)} data points.")
    
    return {"data": data}

def research_agent_node(state: AgentState) -> AgentState:
    """Performs research on the commodity."""
    print("---NODE: RESEARCH AGENT---")
    commodity = state['commodity']
    summary = research_commodity(commodity)
    print("Successfully generated research summary.")
    return {"research_summary": summary}

def news_fetcher_node(state: AgentState) -> AgentState:
    """Fetches recent news for the commodity."""
    print("---NODE: NEWS FETCHER---")
    commodity = state['commodity']
    headlines = fetch_recent_news(commodity)
    print("Successfully fetched recent news.")
    return {"news_headlines": headlines}

def decision_agent_node(state: AgentState) -> AgentState:
    """Makes a procurement decision based on all available info."""
    print("---NODE: DECISION AGENT---")
    decision = make_procurement_decision(
        state['commodity'], 
        state['data'], 
        state['research_summary'], 
        state['news_headlines']
    )
    print("Successfully generated procurement decision.")
    return {"decision": decision}

def final_output_node(state: AgentState) -> AgentState:
    """Combines all information into the final report."""
    print("---NODE: FINAL OUTPUT---")
    
    final_report = (
        f"# Procurement Recommendation Report for {state['commodity']}\n\n"
        f"## Executive Summary\n"
        f"{state['decision']}\n\n"
        f"## Supporting Information\n"
        f"### Qualitative Research Summary:\n{state['research_summary']}\n\n"
        f"### Recent News Headlines:\n{state['news_headlines']}\n\n"
        f"### Recent Price Data (Last 5 Days):\n```\n{state['data'].tail(5).to_string()}\n```"
    )
    
    return {"result": final_report}

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("data_fetcher", data_fetcher_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("news_fetcher", news_fetcher_node)
workflow.add_node("decision_agent", decision_agent_node)
workflow.add_node("final_output", final_output_node)

# Define the graph's execution flow
workflow.set_entry_point("data_fetcher")

# After fetching data, run research and news fetching in parallel
workflow.add_edge("data_fetcher", "research_agent")
workflow.add_edge("data_fetcher", "news_fetcher")

# After both research and news are done, make a decision
# LangGraph ensures that 'decision_agent' waits for all its inputs
workflow.add_edge("research_agent", "decision_agent")
workflow.add_edge("news_fetcher", "decision_agent")

# From decision to the final output
workflow.add_edge("decision_agent", "final_output")
workflow.add_edge("final_output", END)

# Compile the graph
app = workflow.compile()

if __name__ == '__main__':
    # Run the workflow for Copper
    inputs = {"commodity": "Copper", "ticker": "HG=F"}
    final_state = app.invoke(inputs)
    
    print("\n========================================")
    print(final_state['result'])
    print("========================================\n")
