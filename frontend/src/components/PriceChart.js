import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
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
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
        <XAxis 
          dataKey="Date" 
          tick={{ fill: '#8B949E' }} 
          stroke="#30363D" 
          // Allow recharts to skip labels to prevent overlap
          interval="preserveStartEnd"
        />
        <YAxis 
          tick={{ fill: '#8B949E' }} 
          stroke="#30363D" 
          domain={['auto', 'auto']} 
          // Format axis ticks to be more readable
          tickFormatter={(value) => `$${value.toLocaleString()}`}
        />
        <Tooltip
          contentStyle={{ 
            backgroundColor: '#161B22',
            borderColor: '#30363D',
            color: '#E6EDF3'
          }}
          formatter={(value, name) => [value.toFixed(2), name]}
        />
        <Legend wrapperStyle={{ color: '#E6EDF3' }} />

        {/* Historical Data Line */}
        <Line 
          type="monotone" 
          data={historical} 
          dataKey="Close" 
          stroke="#58A6FF" 
          strokeWidth={2} 
          dot={false} 
          name="Historical Price"
        />

        {/* Forecast Data Line */}
        <Line 
          type="monotone" 
          data={combinedForecastData}
          dataKey="Forecast"
          stroke="#3FB950"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={false}
          name="Forecasted Price"
        />

        {/* Confidence Interval Area */}
        <Area
          type="monotone"
          data={combinedForecastData}
          dataKey="Upper_CI"
          fill="#3FB950"
          stroke={false}
          fillOpacity={0.1}
          name="Upper Confidence"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          data={combinedForecastData}
          dataKey="Lower_CI"
          fill="#3FB950"
          stroke={false}
          fillOpacity={0.1}
          name="Lower Confidence"
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;
