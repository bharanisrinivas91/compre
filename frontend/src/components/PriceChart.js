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

  // Format dates consistently
  const formatDate = (date) => {
    return new Date(date).toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      timeZone: 'UTC'
    });
  };

  // Combine historical and forecast data for the chart
  const chartData = [
    ...historical.map(item => ({
      ...item,
      Date: new Date(item.Date),
      type: 'historical'
    })),
    ...forecast.map(item => ({
      ...item,
      Date: new Date(item.Date),
      type: 'forecast'
    }))
  ].sort((a, b) => a.Date - b.Date);
  
  // Get the last historical point to connect the lines
  const lastHistoricalPoint = historical[historical.length - 1];
  const firstForecastPoint = forecast[0];
  
  // Create a connecting point if there's a gap between historical and forecast data
  const connectionPoint = {
    Date: firstForecastPoint ? new Date(firstForecastPoint.Date) : new Date(lastHistoricalPoint.Date),
    Close: lastHistoricalPoint.Close,
    type: 'connection'
  };
  
  // Combine all data for the chart
  const combinedData = [
    ...historical.map(item => ({ ...item, Date: new Date(item.Date) })),
    connectionPoint,
    ...forecast.map(item => ({ 
      ...item, 
      Date: new Date(item.Date),
      Close: item.Forecast // Add Close for consistent tooltip access
    }))
  ].sort((a, b) => a.Date - b.Date);

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={250}>
      <ComposedChart
        data={combinedData}
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
          tickFormatter={(date) => formatDate(date)}
          interval="preserveStartEnd"
          height={35}
          padding={{ left: 10, right: 10 }}
          type="number"
          domain={['dataMin', 'dataMax']}
          tickCount={6}
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
          labelFormatter={(date) => formatDate(date)}
          formatter={(value, name, props) => {
            const price = props.dataKey === 'Forecast' ? value : props.payload.Close || value;
            const label = props.dataKey === 'Forecast' ? 'Forecast Price' : 'Historical Price';
            return [`$${parseFloat(price).toFixed(2)}`, label];
          }}
          cursor={{ stroke: '#58A6FF', strokeWidth: 1, strokeDasharray: '3 3' }}
        />
        <Legend 
          wrapperStyle={{ 
            fontSize: 12, 
            padding: '10px 0',
            color: '#E6EDF3',
            display: 'flex',
            justifyContent: 'center',
            gap: '20px'
          }} 
          height={30}
          iconSize={10}
          iconType="circle"
          layout="horizontal"
          verticalAlign="top"
          align="center"
        />
        {/* Historical Price Line */}
        <Line 
          type="monotone" 
          dataKey="Close" 
          stroke="#58A6FF" 
          strokeWidth={2.5} 
          dot={false} 
          name="Historical Price"
          isAnimationActive={true}
          animationDuration={1000}
          connectNulls={true}
        />
        
        {/* Confidence Interval Area */}
        <Area
          type="monotone"
          dataKey="Upper_CI"
          data={combinedData}
          fill="rgba(63, 185, 80, 0.15)"
          stroke="transparent"
          name="Confidence Interval"
          isAnimationActive={true}
          animationDuration={1500}
          connectNulls={true}
        />
        
        {/* Forecast Price Line */}
        <Line 
          type="monotone" 
          dataKey="Forecast" 
          stroke="#3FB950" 
          strokeWidth={2.5} 
          dot={false} 
          name="Forecast Price"
          isAnimationActive={true}
          animationDuration={1500}
          strokeDasharray="5 5"
          connectNulls={true}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

export default PriceChart;
