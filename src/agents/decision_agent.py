import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import datetime

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

def export_report_to_excel(commodity_name: str, historical_data: pd.DataFrame, forecast_data: pd.DataFrame, 
                      decision: str, research_summary: str, news_headlines: str) -> str:
    """
    Exports the procurement report to an Excel file with forecasted predictions, final decision, and summary.
    
    Args:
        commodity_name (str): The name of the commodity.
        historical_data (pd.DataFrame): Historical price data.
        forecast_data (pd.DataFrame): Forecasted price data.
        decision (str): The procurement decision.
        research_summary (str): Summary of market research.
        news_headlines (str): Recent news headlines.
        
    Returns:
        str: Path to the saved Excel file.
    """
    try:
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{commodity_name.lower()}_procurement_report_{timestamp}.xlsx"
        
        # Create Excel writer with XlsxWriter engine
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Extract decision and justification from the decision text
            decision_lines = decision.strip().split('\n')
            recommendation = "Unknown"
            justification = ""
            
            for i, line in enumerate(decision_lines):
                if line.startswith("Recommendation:"):
                    recommendation = line.replace("Recommendation:", "").strip()
                if i > 0 and line.startswith("Justification:"):
                    justification = "\n".join(decision_lines[i+1:])
                    break
            
            # Create summary sheet
            summary_data = {
                'Item': ['Commodity', 'Recommendation', 'Date of Analysis', 'Analyst'],
                'Value': [commodity_name, recommendation, 
                         datetime.datetime.now().strftime("%Y-%m-%d"), 
                         'CompRe AI Procurement System']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format the summary sheet using openpyxl
            worksheet = writer.sheets['Summary']
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 40
            
            # Add justification to summary sheet
            justification_df = pd.DataFrame({
                'Justification': [justification]
            })
            justification_df.to_excel(writer, sheet_name='Summary', startrow=6, index=False)
            
            # Add research summary to summary sheet
            research_df = pd.DataFrame({
                'Market Research Summary': [research_summary]
            })
            research_df.to_excel(writer, sheet_name='Summary', startrow=10, index=False)
            
            # Add news headlines
            news_lines = news_headlines.strip().split('\n')
            news_df = pd.DataFrame({
                'Recent News Headlines': news_lines
            })
            news_df.to_excel(writer, sheet_name='Summary', startrow=14, index=False)
            
            # Add historical data sheet
            if not historical_data.empty:
                # Ensure the index is reset if it's a DatetimeIndex
                if isinstance(historical_data.index, pd.DatetimeIndex):
                    historical_data = historical_data.reset_index()
                historical_data.to_excel(writer, sheet_name='Historical Data', index=False)
            
            # Add forecast data sheet
            if not forecast_data.empty:
                forecast_data.to_excel(writer, sheet_name='Price Forecast', index=False)
            
            # Create a chart sheet with historical and forecasted prices
            if not historical_data.empty and not forecast_data.empty:
                # Prepare data for the chart
                chart_data = pd.DataFrame()
                
                # Process historical data
                hist_data = historical_data.copy()
                if 'Date' in hist_data.columns:
                    hist_data['Date'] = pd.to_datetime(hist_data['Date'])
                    hist_data = hist_data.sort_values('Date')
                    
                # Process forecast data
                forecast = forecast_data.copy()
                if 'Date' in forecast.columns:
                    forecast['Date'] = pd.to_datetime(forecast['Date'])
                    forecast = forecast.sort_values('Date')
                
                # Create a combined dataset for charting
                chart_data = pd.DataFrame()
                
                if 'Date' in hist_data.columns and 'Close' in hist_data.columns:
                    chart_data['Date'] = hist_data['Date']
                    chart_data['Historical Price'] = hist_data['Close']
                
                if 'Date' in forecast.columns and 'Forecast' in forecast.columns:
                    # Merge forecast data
                    forecast_subset = forecast[['Date', 'Forecast', 'Lower_CI', 'Upper_CI']]
                    chart_data = pd.merge(chart_data, forecast_subset, on='Date', how='outer')
                
                # Save the chart data
                if not chart_data.empty:
                    chart_data.to_excel(writer, sheet_name='Price Chart Data', index=False)
        
        return filename
    except Exception as e:
        print(f"Error exporting report to Excel: {e}")
        return f"Error: {str(e)}"

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
    
    # Create mock forecast data
    mock_forecast = pd.DataFrame({
        'Date': pd.date_range(start='2023-10-06', periods=10),
        'Forecast': [3.75, 3.78, 3.80, 3.82, 3.85, 3.87, 3.90, 3.92, 3.95, 3.97],
        'Lower_CI': [3.70, 3.72, 3.74, 3.76, 3.78, 3.80, 3.82, 3.84, 3.86, 3.88],
        'Upper_CI': [3.80, 3.84, 3.86, 3.88, 3.92, 3.94, 3.98, 4.00, 4.04, 4.06]
    })
    
    decision = make_procurement_decision("Copper", mock_data, mock_research, mock_news)
    print("--- Procurement Decision ---")
    print(decision)
    print("--------------------------")
    
    # Test the Excel export
    excel_path = export_report_to_excel("Copper", mock_data, mock_forecast, decision, mock_research, mock_news)
    print(f"\nExcel report saved to: {excel_path}")
