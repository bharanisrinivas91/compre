import React, { useState, useContext } from 'react';
import axios from 'axios';
import { ScrollMenu, VisibilityContext } from 'react-horizontal-scrolling-menu';
import 'react-horizontal-scrolling-menu/dist/styles.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { 
  FaCubes, FaFlask, FaNewspaper, FaChartLine, FaBrain, FaChevronLeft, FaChevronRight, FaChevronDown, FaChevronUp, 
  FaGasPump, FaLeaf, FaSeedling, FaIndustry, FaCube, FaCoins 
} from 'react-icons/fa';
import './App.css';
import PriceChart from './components/PriceChart';

const commodityMap = {
  "Copper": { icon: FaCube, name: "Copper" },
  "Crude Oil": { icon: FaGasPump, name: "Crude Oil" },
  "Gold": { icon: FaCoins, name: "Gold" },
  "Natural Gas": { icon: FaGasPump, name: "Natural Gas" },
  "Soybeans": { icon: FaLeaf, name: "Soybeans" },
  "Corn": { icon: FaSeedling, name: "Corn" },
  "Wheat": { icon: FaSeedling, name: "Wheat" },
  "Steel": { icon: FaIndustry, name: "Steel" },
  "Aluminum": { icon: FaCubes, name: "Aluminum" },
  "Silver": { icon: FaCoins, name: "Silver" }
};
const commodities = Object.keys(commodityMap);

function App() {
  const [selectedCommodity, setSelectedCommodity] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async (commodity) => {
    if (loading) return;
    setSelectedCommodity(commodity);
    setLoading(true);
    setError('');
    setReport(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/procurement-analysis', { commodity });
      setReport(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An unexpected error occurred.');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="header">
        <h1><FaCubes /> Aetherium Metals</h1>
        <p>AI-Powered Procurement Analysis for Industrial Commodities</p>
      </header>

      <CommoditySelector selected={selectedCommodity} onSelect={handleAnalyze} />

      {loading && <LoadingSpinner />}
      {error && <p className="error-message">Error: {error}</p>}
      {report && (
        <div className="report-container">
          <div className="report-card recommendation-card">
            <h2 className="card-title"><FaFlask /> Procurement Recommendation</h2>
            <p className={`recommendation-text ${report.recommendation_summary.toLowerCase().includes('procure') ? 'recommendation-procure' : 'recommendation-wait'}`}>{report.recommendation_summary}</p>
          </div>

          <div className="report-card">
            <h2 className="card-title"><FaChartLine /> 60-Day Price Forecast</h2>
            <PriceChart historical={report.historical_prices} forecast={report.forecasted_prices} />
          </div>

          <CollapsibleCard title="AI Insight Summary" icon={<FaBrain />}>
            <p>{report.qualitative_research}</p>
          </CollapsibleCard>

          <CollapsibleCard title="Recent News" icon={<FaNewspaper />}>
            <ul className="news-list">
              {report.recent_news.map((news, index) => <li key={index}>{news}</li>)}
            </ul>
          </CollapsibleCard>
        </div>
      )}
    </div>
  );
}

const LoadingSpinner = () => (
  <div className="spinner-container">
    <div className="spinner"></div>
    <span>Engaging AI agents... Please wait.</span>
  </div>
);

const CollapsibleCard = ({ icon, title, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="report-card">
      <div className="collapsible-header" onClick={() => setIsOpen(!isOpen)}>
        <h2 className="card-title">{icon} {title}</h2>
        {isOpen ? <FaChevronUp /> : <FaChevronDown />}
      </div>
      <div className={`collapsible-content ${isOpen ? 'open' : ''}`}>
        {children}
      </div>
    </div>
  );
};

const CommoditySelector = ({ selected, onSelect }) => (
  <div className="commodity-selector-container">
    <ScrollMenu LeftArrow={LeftArrow} RightArrow={RightArrow}>
      {commodities.map((name) => (
        <div
          className={`commodity-card ${name === selected ? 'selected' : ''}`}
          key={name}
          onClick={() => onSelect(name)}
          tabIndex={0}
        >
          {React.createElement(commodityMap[name].icon, { className: 'commodity-icon' })}
          <span>{commodityMap[name].name}</span>
        </div>
      ))}
    </ScrollMenu>
  </div>
);

function LeftArrow() {
  const { isFirstItemVisible, scrollPrev } = useContext(VisibilityContext);
  return (
    <div className={`arrow ${isFirstItemVisible ? 'disabled' : ''}`} onClick={() => scrollPrev()}>
      <FaChevronLeft />
    </div>
  );
}

function RightArrow() {
  const { isLastItemVisible, scrollNext } = useContext(VisibilityContext);
  return (
    <div className={`arrow ${isLastItemVisible ? 'disabled' : ''}`} onClick={() => scrollNext()}>
      <FaChevronRight />
    </div>
  );
}



export default App;
