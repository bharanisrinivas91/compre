# AI-Powered Commodity Procurement System

This project is an AI-native system designed to help companies optimize commodity procurement costs. It leverages agentic AI for autonomous data collection, machine learning for price forecasting, and a procurement optimizer to provide actionable recommendations.

## Features

- **Agentic Data Collection**: Autonomous agents gather real-time and historical data from various sources.
- **Price Forecasting**: Advanced machine learning models predict future price movements.
- **Procurement Recommendations**: The system advises on the best time to procure commodities based on forecasts, project timelines, and budgets.

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm
- Git

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bharanisrinivas91/compre.git
    cd compre
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root of the project and add the following:
    ```
    GROQ_API_KEY="your_groq_api_key"
    EVENT_REGISTRY_API_KEY="your_event_registry_api_key"
    ```

5.  **Run the backend:**
    The backend is set up to be run from `src/main.py`.
    ```bash
    python src/main.py
    ```

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install the dependencies:**
    ```bash
    npm install
    ```

3.  **Run the frontend:**
    ```bash
    npm start
    ```

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Groq
- **Data**: yfinance
- **Core**: Python
