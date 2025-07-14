import React from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';

const PriceChart = ({ historical, forecast }) => {
  console.log('PriceChart received historical data:', historical);
  console.log('PriceChart received forecast data:', forecast);
  
  // Check for empty or invalid data
  if (!historical || !Array.isArray(historical) || historical.length === 0) {
    console.error('Missing or invalid historical data');
    return <div className="chart-error">No historical data available</div>;
  }
  
  if (!forecast || !Array.isArray(forecast) || forecast.length === 0) {
    console.error('Missing or invalid forecast data');
    return <div className="chart-error">No forecast data available</div>;
  }
  
  // Validate that historical data has the required fields
  if (!historical[0].hasOwnProperty('Date') || !historical[0].hasOwnProperty('Close')) {
    console.error('Historical data missing required fields:', historical[0]);
    return <div className="chart-error">Invalid historical data format</div>;
  }
  
  // Validate that forecast data has the required fields
  if (!forecast[0].hasOwnProperty('Date') || !forecast[0].hasOwnProperty('Forecast') ||
      !forecast[0].hasOwnProperty('Lower_CI') || !forecast[0].hasOwnProperty('Upper_CI')) {
    console.error('Forecast data missing required fields:', forecast[0]);
    return <div className="chart-error">Invalid forecast data format</div>;
  }

  // Combine historical and the last point of historical data to connect the lines
  const lastHistoricalPoint = historical[historical.length - 1];
  const combinedForecastData = [lastHistoricalPoint, ...forecast];

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={250}>
      <ComposedChart
        data={historical}
        margin={{
          top: 15,
          right: 25,
          left: 5,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#30363D" opacity={0.3} />
        <XAxis 
          dataKey="Date" 
          tick={{ fill: '#E6EDF3', fontSize: 12 }} 
          stroke="#30363D" 
          tickFormatter={(date) => new Date(date).toLocaleDateString(undefined, {month: 'short', day: 'numeric'})}
          interval="preserveStartEnd"
          height={35}
          padding={{ left: 10, right: 10 }}
        />
        <YAxis 
          tick={{ fill: '#E6EDF3', fontSize: 12 }} 
          width={45}
          tickFormatter={(value) => `$${Math.round(value)}`}
          domain={['auto', 'auto']}
          padding={{ top: 10, bottom: 10 }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#161B22', 
            border: '1px solid #30363D',
            padding: '10px',
            fontSize: '13px',
            borderRadius: '4px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.2)'
          }}
          labelFormatter={(date) => new Date(date).toLocaleDateString()}
          formatter={(value) => [`$${parseFloat(value).toFixed(2)}`, 'Historical Price']}
          cursor={{ stroke: '#58A6FF', strokeWidth: 1, strokeDasharray: '3 3' }}
        />
        <Legend 
          wrapperStyle={{ 
            fontSize: 12, 
            padding: '10px 0',
            color: '#E6EDF3'
          }} 
          height={30}
          iconSize={10}
          iconType="circle"
        />
        <Line 
          type="monotone" 
          dataKey="Close" 
          stroke="#58A6FF" 
          strokeWidth={2.5} 
          dot={false} 
          name="Historical Price"
          isAnimationActive={true}
          animationDuration={1000}
        />
        
        {/* Forecast data with confidence intervals */}
        <Area
          type="monotone"
          data={combinedForecastData}
          dataKey="Lower_CI"
          fill="rgba(63, 185, 80, 0.15)"
          stroke="transparent"
          name="Confidence Interval"
          isAnimationActive={true}
          animationDuration={1500}
        />
        <Area
          type="monotone"
          data={combinedForecastData}
          dataKey="Upper_CI"
          fill="rgba(63, 185, 80, 0.15)"
          stroke="transparent"
          name=""
          isAnimationActive={true}
          animationDuration={1500}
          legendType="none"
        />
        <Line 
          type="monotone" 
          data={combinedForecastData}
          dataKey="Forecast" 
          stroke="#3FB950" 
          strokeWidth={2.5} 
          dot={false} 
          name="Forecast Price"
          isAnimationActive={true}
          animationDuration={1500}
          strokeDasharray="5 5"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;
