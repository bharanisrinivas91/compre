from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.workflow import app_graph
from src.config import COMMODITY_TICKERS

app = FastAPI(
    title="AI Commodity Procurement Advisor",
    description="An API for getting AI-powered procurement recommendations for commodities.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ProcurementRequest(BaseModel):
    commodity: str

@app.post("/procurement-analysis")
async def get_procurement_analysis(request: ProcurementRequest):
    """
    Runs the full agentic workflow to generate a procurement recommendation.
    
    - **commodity**: The common name of the commodity (e.g., "Copper").
    """
    commodity_name = request.commodity.lower()
    ticker = COMMODITY_TICKERS.get(commodity_name)

    if not ticker:
        raise HTTPException(
            status_code=404,
            detail=f"Commodity '{request.commodity}' not found. "
                   f"Available commodities are: {', '.join(COMMODITY_TICKERS.keys())}"
        )

    inputs = {"commodity": commodity_name, "ticker": ticker}
    final_state = app_graph.invoke(inputs)
    # The 'report' key now holds our structured JSON object
    return final_state.get('report', {'error': 'Could not generate report.'})

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Commodity Procurement Advisor API. See /docs for details."}
