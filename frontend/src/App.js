import React, { useState, useContext } from 'react';
import axios from 'axios';
import { ScrollMenu, VisibilityContext } from 'react-horizontal-scrolling-menu';
import 'react-horizontal-scrolling-menu/dist/styles.css';

import { 
  FaCubes, FaFlask, FaNewspaper, FaChartLine, FaBrain, FaChevronLeft, FaChevronRight, 
  FaGasPump, FaLeaf, FaSeedling, FaIndustry, FaCube, FaCoins, 
  FaExpand, FaCompress, FaDownload, FaFileExcel
} from 'react-icons/fa';
import './App.css';
import PriceChart from './components/PriceChart';

const commodityMap = {
  "Copper": { icon: FaCube, name: "Copper" },
  "Crude Oil": { icon: FaGasPump, name: "Crude Oil" },
  "Gold": { icon: FaCoins, name: "Gold" },
  "Natural Gas": { icon: FaGasPump, name: "Natural Gas" },
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
  const [fullscreenQuadrant, setFullscreenQuadrant] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const reportRef = React.useRef(null);

  const toggleFullscreen = (quadrant) => {
    if (fullscreenQuadrant === quadrant) {
      setFullscreenQuadrant(null);
    } else {
      setFullscreenQuadrant(quadrant);
    }
  };

  const downloadReport = async () => {
    if (!report || downloading) return;
    
    setDownloading(true);
    
    try {
      // Check if the Excel report path is available
      if (report.excel_report_path && !report.excel_report_path.startsWith('Error:')) {
        // Create a server URL to download the Excel file
        const excelFilePath = report.excel_report_path;
        const fileName = excelFilePath.split('/').pop();
        
        // Create a server endpoint URL to download the file
        // This assumes your API is serving static files from the root directory
        const downloadUrl = `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/download/${fileName}`;
        
        // Create a link element and trigger the download
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`Downloading Excel report: ${fileName}`);
      } else {
        // If Excel path is not available or has an error, show an error
        console.error('Excel report path not available or has an error:', report.excel_report_path);
        setError('Excel report is not available for download');
      }
    } catch (err) {
      console.error('Error downloading Excel report:', err);
      setError('Failed to download Excel report');
    } finally {
      setDownloading(false);
    }
  };

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
        <h1><FaCubes /> ProcureIQ</h1>
        <p>AI-Powered Procurement Intelligence for Industrial Commodities</p>
      </header>

      <CommoditySelector selected={selectedCommodity} onSelect={handleAnalyze} />

      {loading && <LoadingSpinner />}
      {error && <p className="error-message">Error: {error}</p>}
      {report && (
        <React.Fragment>
          <div className="download-report-container">
            <button 
              className={`download-report-button ${downloading ? 'downloading' : ''}`}
              onClick={downloadReport}
              disabled={downloading}
            >
              {downloading ? (
                <React.Fragment>
                  <div className="button-spinner"></div>
                  <span>Generating Excel...</span>
                </React.Fragment>
              ) : (
                <React.Fragment>
                  <FaFileExcel />
                  <span>Download Excel Report</span>
                </React.Fragment>
              )}
            </button>
          </div>
          <div className="quadrant-layout" ref={reportRef}>
            {/* Top Left Quadrant - Recommendation */}
            <div className={`quadrant recommendation-quadrant ${fullscreenQuadrant === 'recommendation' ? 'fullscreen' : ''}`}>
            <button 
              className="fullscreen-toggle" 
              onClick={() => toggleFullscreen('recommendation')}
              aria-label={fullscreenQuadrant === 'recommendation' ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {fullscreenQuadrant === 'recommendation' ? <FaCompress /> : <FaExpand />}
            </button>
            <h2 className="card-title"><FaFlask /> Procurement Recommendation</h2>
            <div className="quadrant-content scrollable">
              <p className={`recommendation-text ${report.recommendation_summary.toLowerCase().includes('procure') ? 'recommendation-procure' : 'recommendation-wait'}`}>
                <strong>{report.recommendation_summary.toLowerCase().includes('procure') ? 'PROCURE NOW' : 'WAIT'}</strong>
                <br />
                {report.recommendation_summary}
              </p>
            </div>
          </div>

            {/* Top Right Quadrant - Price Forecast */}
            <div className={`quadrant forecast-quadrant ${fullscreenQuadrant === 'forecast' ? 'fullscreen' : ''}`}>
            <button 
              className="fullscreen-toggle" 
              onClick={() => toggleFullscreen('forecast')}
              aria-label={fullscreenQuadrant === 'forecast' ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {fullscreenQuadrant === 'forecast' ? <FaCompress /> : <FaExpand />}
            </button>
            <h2 className="card-title"><FaChartLine /> 60-Day Price Forecast</h2>
            <div className="quadrant-content chart-container">
              <PriceChart historical={report.historical_prices} forecast={report.forecasted_prices} />
            </div>
          </div>

            {/* Bottom Left Quadrant - AI Insight */}
            <div className={`quadrant insight-quadrant ${fullscreenQuadrant === 'insight' ? 'fullscreen' : ''}`}>
            <button 
              className="fullscreen-toggle" 
              onClick={() => toggleFullscreen('insight')}
              aria-label={fullscreenQuadrant === 'insight' ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {fullscreenQuadrant === 'insight' ? <FaCompress /> : <FaExpand />}
            </button>
            <h2 className="card-title"><FaBrain /> AI Insight Summary</h2>
            <div className="quadrant-content scrollable">
              {report.qualitative_research.split('\n\n').map((paragraph, index) => (
                <div key={index} className="insight-paragraph">
                  {paragraph.includes('bullish') ? (
                    <p dangerouslySetInnerHTML={{
                      __html: paragraph
                        .replace(/bullish/gi, '<span class="highlight-bullish">bullish</span>')
                        .replace(/bearish/gi, '<span class="highlight-bearish">bearish</span>')
                        .replace(/neutral/gi, '<span class="highlight-neutral">neutral</span>')
                    }} />
                  ) : (
                    <p>{paragraph}</p>
                  )}
                </div>
              ))}
            </div>
          </div>

            {/* Bottom Right Quadrant - News */}
            <div className={`quadrant news-quadrant ${fullscreenQuadrant === 'news' ? 'fullscreen' : ''}`}>
            <button 
              className="fullscreen-toggle" 
              onClick={() => toggleFullscreen('news')}
              aria-label={fullscreenQuadrant === 'news' ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {fullscreenQuadrant === 'news' ? <FaCompress /> : <FaExpand />}
            </button>
            <h2 className="card-title"><FaNewspaper /> Recent News</h2>
            <div className="quadrant-content scrollable">
              <ul className="news-list">
                {report.recent_news.map((news, index) => (
                  <li key={index}>
                    <div className="news-item">
                      <span className="news-number">{index + 1}.</span>
                      <span className="news-content">{news}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </React.Fragment>
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


const CommoditySelector = ({ selected, onSelect }) => {
  const [scrollPosition, setScrollPosition] = useState(0);
  const commodityListRef = React.useRef(null);
  const [maxScroll, setMaxScroll] = useState(0);
  
  React.useEffect(() => {
    if (commodityListRef.current) {
      setMaxScroll(commodityListRef.current.scrollWidth - commodityListRef.current.clientWidth);
    }
  }, []);
  
  const scrollLeft = () => {
    if (commodityListRef.current) {
      const newPosition = Math.max(0, scrollPosition - 200);
      commodityListRef.current.scrollTo({ left: newPosition, behavior: 'smooth' });
      setScrollPosition(newPosition);
    }
  };
  
  const scrollRight = () => {
    if (commodityListRef.current) {
      const newPosition = Math.min(maxScroll, scrollPosition + 200);
      commodityListRef.current.scrollTo({ left: newPosition, behavior: 'smooth' });
      setScrollPosition(newPosition);
    }
  };
  
  const handleScroll = () => {
    if (commodityListRef.current) {
      setScrollPosition(commodityListRef.current.scrollLeft);
    }
  };
  
  React.useEffect(() => {
    const listElement = commodityListRef.current;
    if (listElement) {
      listElement.addEventListener('scroll', handleScroll);
      return () => listElement.removeEventListener('scroll', handleScroll);
    }
  }, []);
  
  return (
    <div className="commodity-selector-container">
      <div className="commodity-selector">
        <button 
          className={`arrow ${scrollPosition <= 0 ? 'disabled' : ''}`}
          onClick={scrollLeft}
          disabled={scrollPosition <= 0}
          aria-label="Scroll left"
        >
          <FaChevronLeft />
        </button>
        
        <div 
          className="commodity-list" 
          ref={commodityListRef}
          style={{ 
            display: 'flex', 
            overflowX: 'auto',
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
            padding: '1rem 0'
          }}
          onScroll={handleScroll}
        >
          {commodities.map((name) => (
            <button
              key={name}
              className={`commodity-button ${name === selected ? 'active' : ''}`}
              onClick={() => onSelect(name)}
              aria-label={`Select ${name}`}
            >
              <div className="commodity-icon-wrapper">
                {React.createElement(commodityMap[name].icon, { className: 'commodity-icon' })}
              </div>
              <span className="commodity-name">{commodityMap[name].name}</span>
            </button>
          ))}
        </div>
        
        <button 
          className={`arrow ${scrollPosition >= maxScroll ? 'disabled' : ''}`}
          onClick={scrollRight}
          disabled={scrollPosition >= maxScroll}
          aria-label="Scroll right"
        >
          <FaChevronRight />
        </button>
      </div>
    </div>
  );
};

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
