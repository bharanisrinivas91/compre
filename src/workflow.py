import pandas as pd
from typing import TypedDict
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END
from src.agents import (
    fetch_commodity_data,
    research_commodity,
    make_procurement_decision,
    fetch_recent_news,
    forecast_prices,
    export_report_to_excel
)

# Define the state for our graph
class AgentState(TypedDict):
    commodity: str
    ticker: str
    data: pd.DataFrame
    price_forecast: pd.DataFrame
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

def forecasting_node(state: AgentState) -> AgentState:
    print("---NODE: FORECASTING AGENT---")
    forecast = forecast_prices(state['data'])
    return {"price_forecast": forecast}

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

    try:
        # 1. Format historical price data
        historical_data = state['data'].copy()
        print(f"Historical data columns: {historical_data.columns.tolist()}")
        
        # Handle missing columns
        if 'Close' not in historical_data.columns and 'Adj Close' in historical_data.columns:
            historical_data['Close'] = historical_data['Adj Close']
        
        # Ensure Date is a column, not an index
        if 'Date' not in historical_data.columns and historical_data.index.name == 'Date':
            historical_data = historical_data.reset_index()
        
        # Create a simplified version with just Date and Close
        simple_historical = historical_data[['Date', 'Close']].copy()
        
        # Convert dates to strings
        simple_historical['Date'] = pd.to_datetime(simple_historical['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert to records
        historical_data_json = simple_historical.to_dict(orient='records')
        
        # 2. Handle forecast data
        forecast_data = state['price_forecast']
        
        # If forecast data is empty, create a placeholder
        if forecast_data.empty:
            print("Warning: Empty forecast data. Creating placeholder.")
            last_date = pd.to_datetime(historical_data['Date'].iloc[-1])
            last_price = historical_data['Close'].iloc[-1]
            
            # Create dummy forecast data (flat line)
            dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(60)]
            forecast_data_json = [{
                'Date': d,
                'Forecast': last_price,
                'Lower_CI': last_price * 0.9,
                'Upper_CI': last_price * 1.1
            } for d in dates]
            
            # Create a DataFrame for the Excel export
            forecast_data = pd.DataFrame(forecast_data_json)
        else:
            # Use the forecast data as is
            print(f"Forecast data columns: {forecast_data.columns.tolist()}")
            forecast_data_json = forecast_data.to_dict(orient='records')
    
    except Exception as e:
        print(f"Error in final output node: {e}")
        # Create empty placeholders if anything fails
        historical_data_json = []
        forecast_data_json = []
        historical_data = pd.DataFrame()
        forecast_data = pd.DataFrame()

    # 3. Clean up news headlines
    raw_headlines = state['news_headlines'].strip().split('\n')
    headlines_list = [line.strip() for line in raw_headlines if line.strip() and not line.strip().startswith('---')]

    # 4. Assemble the final report
    report = {
        "commodity": state['commodity'].title(),
        "recommendation_summary": state['decision'],
        "qualitative_research": state['research_summary'],
        "recent_news": headlines_list,
        "historical_prices": historical_data_json,
        "forecasted_prices": forecast_data_json
    }
    
    # 5. Export the report to Excel
    try:
        excel_path = export_report_to_excel(
            state['commodity'],
            historical_data,
            forecast_data,
            state['decision'],
            state['research_summary'],
            state['news_headlines']
        )
        print(f"Excel report successfully exported to: {excel_path}")
        report["excel_report_path"] = excel_path
    except Exception as e:
        print(f"Error exporting Excel report: {e}")
        report["excel_report_path"] = f"Error: {str(e)}"
    
    return {"report": report}

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("data_fetcher", data_fetcher_node)
workflow.add_node("forecasting_node", forecasting_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("news_fetcher", news_fetcher_node)
workflow.add_node("decision_agent", decision_agent_node)
workflow.add_node("final_output", final_output_node)

workflow.set_entry_point("data_fetcher")
workflow.add_edge("data_fetcher", "forecasting_node")

# The decision agent depends on research and news, which can run in parallel
workflow.add_edge("forecasting_node", "research_agent")
workflow.add_edge("forecasting_node", "news_fetcher")

# Both research and news feed into the decision agent
workflow.add_edge("research_agent", "decision_agent")
workflow.add_edge("news_fetcher", "decision_agent")

workflow.add_edge("decision_agent", "final_output")
workflow.add_edge("final_output", END)

# Compile the graph
app_graph = workflow.compile()
