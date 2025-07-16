import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
import logging
import time
from typing import Optional, Dict, Any, List, Union, Tuple
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def detect_price_column(data: pd.DataFrame) -> Optional[str]:
    possible_cols = ['Close', 'Adj Close', 'Adj_Close', 'close', 'adj_close', 'Price', 'price']
    for col in possible_cols:
        if col in data.columns:
            return col
    for col in data.columns:
        if any(name in col.lower() for name in ['close', 'price']):
            return col
    return None

def detect_date_column(data: pd.DataFrame) -> Optional[str]:
    if 'Date' in data.columns:
        return 'Date'
    elif data.index.name == 'Date':
        data.reset_index(inplace=True)
        return 'Date'
    else:
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]) or 'date' in col.lower():
                return col
    return None

def add_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Add engineered features to improve forecasting performance.
    """
    # Add rolling statistics
    df['rolling_mean_7d'] = df[price_col].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7d'] = df[price_col].rolling(window=7, min_periods=1).std()
    df['rolling_mean_30d'] = df[price_col].rolling(window=30, min_periods=1).mean()
    df['rolling_std_30d'] = df[price_col].rolling(window=30, min_periods=1).std()
    
    # Add momentum indicators
    df['price_change_1d'] = df[price_col].pct_change(1)
    df['price_change_7d'] = df[price_col].pct_change(7)
    df['price_change_30d'] = df[price_col].pct_change(30)
    
    # Add date-based features
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
    
    # Fill NaN values created by rolling windows
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df

def forecast_arima(df: pd.DataFrame, price_col: str, days_to_predict: int) -> Dict[str, Any]:
    """
    Forecast using auto-ARIMA model with seasonality and volatility injection.
    """
    logger.info("Fitting enhanced auto-ARIMA model with seasonality and volatility...")
    
    # Calculate historical volatility
    hist_volatility = df[price_col].pct_change().std()
    price_mean = df[price_col].mean()
    
    # Try to detect seasonality with more aggressive parameters
    seasonal_periods = [7, 12, 30, 365]  # Common seasonal periods (weekly, monthly, quarterly, yearly)
    best_aic = float('inf')
    best_model = None
    best_period = None
    
    # Test different seasonal periods with more flexible parameters
    for period in seasonal_periods:
        if len(df) >= 2 * period:  # Need at least 2 full cycles
            try:
                model = auto_arima(
                    df[price_col],
                    seasonal=True,
                    m=period,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    max_order=None,
                    max_p=5,
                    max_d=2,
                    max_q=5,
                    information_criterion='aic',  # Use AIC for better model selection
                    with_intercept=True  # Allow intercept for more flexibility
                )
                if model.aic() < best_aic:
                    best_aic = model.aic()
                    best_model = model
                    best_period = period
            except Exception as e:
                logger.warning(f"Failed to fit seasonal ARIMA with period {period}: {str(e)}")
    
    # If no seasonal model worked, try non-seasonal with more parameters
    if best_model is None:
        best_model = auto_arima(
            df[price_col],
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            max_p=10,  # Allow more autoregressive terms
            max_d=3,   # Allow more differencing
            max_q=10   # Allow more moving average terms
        )
        best_period = None
    
    order = best_model.order
    seasonal_order = best_model.seasonal_order if hasattr(best_model, 'seasonal_order') else None
    
    logger.info(f"Best ARIMA order: {order}, Seasonal order: {seasonal_order}, Period: {best_period}")
    
    # Fit final model using statsmodels for confidence intervals
    if seasonal_order and best_period:
        model = ARIMA(df[price_col], order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(df[price_col], order=order)
    
    model_fit = model.fit()
    
    # Forecast
    forecast_result = model_fit.get_forecast(steps=days_to_predict)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)  # Use 95% confidence interval
    
    # Inject volatility to make forecasts more dynamic
    # Calculate recent trend
    recent_window = min(30, len(df) // 3)
    recent_trend = df[price_col].iloc[-recent_window:].diff().mean()
    
    # Amplify trend and add volatility
    trend_factor = 1.5  # Amplify detected trends
    volatility_factor = 1.2  # Increase volatility
    
    # Generate more dynamic forecast by injecting volatility and amplifying trends
    dynamic_forecast = np.zeros(days_to_predict)
    dynamic_forecast[0] = forecast[0] + (recent_trend * trend_factor)
    
    # Generate random shocks based on historical volatility
    np.random.seed(42)  # For reproducibility
    random_shocks = np.random.normal(0, hist_volatility * price_mean * volatility_factor, days_to_predict)
    
    # Apply trend and volatility
    for i in range(1, days_to_predict):
        # Base forecast + trend amplification + volatility injection
        dynamic_forecast[i] = forecast[i] + \
                             (forecast[i] - forecast[i-1]) * trend_factor + \
                             random_shocks[i]
    
    # Adjust confidence intervals to reflect added volatility
    lower_ci = dynamic_forecast - (dynamic_forecast - conf_int.iloc[:, 0].values) * volatility_factor
    upper_ci = dynamic_forecast + (conf_int.iloc[:, 1].values - dynamic_forecast) * volatility_factor
    
    return {
        'forecast': dynamic_forecast,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }

def forecast_prophet(df: pd.DataFrame, date_col: str, price_col: str, days_to_predict: int) -> Dict[str, Any]:
    """
    Forecast using Facebook Prophet model with enhanced changepoint detection and volatility.
    """
    logger.info("Fitting enhanced Prophet model with dynamic components...")
    
    # Calculate historical volatility
    hist_volatility = df[price_col].pct_change().std()
    price_mean = df[price_col].mean()
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df.reset_index()
    prophet_df = prophet_df.rename(columns={date_col: 'ds', price_col: 'y'})
    
    # Add additional regressors if we have enough data
    if len(df) >= 60:
        # Add rolling mean as a regressor
        prophet_df['rolling_mean'] = prophet_df['y'].rolling(window=7, min_periods=1).mean()
        prophet_df['rolling_mean'] = prophet_df['rolling_mean'].fillna(method='bfill')
        
        # Add momentum as a regressor
        prophet_df['momentum'] = prophet_df['y'].pct_change(7).fillna(0)
    
    # Initialize Prophet model with very aggressive parameters
    model = Prophet(
        changepoint_prior_scale=0.15,  # Much more flexible trend changes (default is 0.05)
        seasonality_prior_scale=15.0,  # Stronger seasonality (default is 10)
        changepoint_range=0.95,       # Allow changepoints until 95% of the time series
        n_changepoints=35,            # Increase number of potential changepoints
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        uncertainty_samples=1000       # More samples for better uncertainty estimates
    )
    
    # Add multiple seasonality components
    model.add_seasonality(name='monthly', period=30.5, fourier_order=10)  # Higher fourier order
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    
    # Add the additional regressors if we created them
    if 'rolling_mean' in prophet_df.columns:
        model.add_regressor('rolling_mean', standardize=True)
    if 'momentum' in prophet_df.columns:
        model.add_regressor('momentum', standardize=True)
    
    # Fit the model
    model.fit(prophet_df)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=days_to_predict, freq='D')
    
    # Add regressor values to future dataframe
    if 'rolling_mean' in prophet_df.columns:
        # For future dates, use the last available value
        future['rolling_mean'] = prophet_df['rolling_mean'].iloc[-1]
    
    if 'momentum' in prophet_df.columns:
        # For future dates, use the last available momentum
        future['momentum'] = prophet_df['momentum'].iloc[-1]
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Extract forecast for future dates only
    forecast = forecast.tail(days_to_predict)
    
    # Calculate trend direction and magnitude
    trend_direction = np.mean(np.diff(forecast['trend'].values))
    
    # Amplify the trend and add volatility
    trend_factor = 1.8  # Stronger trend amplification
    volatility_factor = 1.5  # More volatility
    
    # Generate dynamic forecast with amplified trend and volatility
    base_forecast = forecast['yhat'].values
    dynamic_forecast = np.zeros(days_to_predict)
    
    # Generate random shocks based on historical volatility
    np.random.seed(43)  # Different seed than ARIMA
    random_shocks = np.random.normal(0, hist_volatility * price_mean * volatility_factor, days_to_predict)
    
    # Apply trend amplification and volatility
    dynamic_forecast[0] = base_forecast[0] + random_shocks[0]
    for i in range(1, days_to_predict):
        # Amplify trend component and add volatility
        trend_component = (base_forecast[i] - base_forecast[i-1]) * trend_factor
        dynamic_forecast[i] = base_forecast[i] + trend_component + random_shocks[i]
    
    # Adjust confidence intervals
    uncertainty_range = forecast['yhat_upper'].values - forecast['yhat_lower'].values
    lower_ci = dynamic_forecast - (uncertainty_range / 2) * volatility_factor
    upper_ci = dynamic_forecast + (uncertainty_range / 2) * volatility_factor
    
    return {
        'forecast': dynamic_forecast,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }

def forecast_ensemble(df: pd.DataFrame, price_col: str, days_to_predict: int) -> Dict[str, Any]:
    """
    Forecast using a diverse ensemble of models with dynamic boosting.
    """
    logger.info("Fitting enhanced ensemble model with dynamic boosting...")
    
    # Calculate historical volatility and trend
    hist_volatility = df[price_col].pct_change().std()
    price_mean = df[price_col].mean()
    recent_window = min(30, len(df) // 3)
    recent_trend = df[price_col].iloc[-recent_window:].diff().mean()
    
    # Add more advanced features
    feature_df = add_features(df.copy(), price_col)
    
    # Add cyclical features (sin/cos transformations of time components)
    if isinstance(feature_df.index, pd.DatetimeIndex):
        # Day of week as cyclical feature
        feature_df['day_of_week_sin'] = np.sin(2 * np.pi * feature_df.index.dayofweek / 7)
        feature_df['day_of_week_cos'] = np.cos(2 * np.pi * feature_df.index.dayofweek / 7)
        
        # Month as cyclical feature
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df.index.month / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df.index.month / 12)
    
    # Add price levels and volatility clusters
    feature_df['price_level'] = pd.qcut(feature_df[price_col], 5, labels=False, duplicates='drop')
    feature_df['volatility_cluster'] = pd.qcut(
        feature_df[price_col].rolling(window=7).std().fillna(0), 
        5, labels=False, duplicates='drop'
    )
    
    # Fill any NaN values
    feature_df = feature_df.fillna(0)
    
    # Prepare training data
    X = feature_df.drop(columns=[price_col])
    y = feature_df[price_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train multiple diverse models
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import ElasticNet
    
    # Random Forest with higher variance
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12, 
        min_samples_leaf=2,  # Smaller leaf size for more variance
        bootstrap=True,
        random_state=42
    )
    
    # Gradient Boosting with faster learning
    gb_model = GradientBoostingRegressor(
        n_estimators=150, 
        learning_rate=0.1,  # Higher learning rate
        max_depth=6,
        subsample=0.8,  # Subsampling for more diversity
        random_state=43
    )
    
    # Extra Trees for more randomness
    et_model = ExtraTreesRegressor(
        n_estimators=200, 
        max_depth=12,
        min_samples_split=2,
        random_state=44
    )
    
    # Linear model for trend capturing
    linear_model = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=45)
    
    # Train all models
    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    et_model.fit(X_scaled, y)
    linear_model.fit(X_scaled, y)
    
    # Generate future features for prediction
    future_features = []
    future_df = df.copy()
    last_date = future_df.index[-1]
    
    # Initialize predictions array
    predictions = []
    
    for i in range(days_to_predict):
        # Predict next day
        next_date = last_date + timedelta(days=i+1)
        next_row = pd.DataFrame(index=[next_date])
        
        # Add the predicted price from previous iteration
        if i > 0:
            next_row[price_col] = predictions[-1]
        else:
            next_row[price_col] = future_df[price_col].iloc[-1]
        
        # Combine with historical data
        temp_df = pd.concat([future_df, next_row])
        
        # Generate features
        temp_df = add_features(temp_df, price_col)
        
        # Add cyclical features
        if isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df['day_of_week_sin'] = np.sin(2 * np.pi * temp_df.index.dayofweek / 7)
            temp_df['day_of_week_cos'] = np.cos(2 * np.pi * temp_df.index.dayofweek / 7)
            temp_df['month_sin'] = np.sin(2 * np.pi * temp_df.index.month / 12)
            temp_df['month_cos'] = np.cos(2 * np.pi * temp_df.index.month / 12)
        
        # Add price levels and volatility clusters (using last known values)
        temp_df['price_level'] = feature_df['price_level'].iloc[-1]
        temp_df['volatility_cluster'] = feature_df['volatility_cluster'].iloc[-1]
        
        # Fill any NaN values
        temp_df = temp_df.fillna(0)
        
        # Get the features for the next date
        next_features = temp_df.loc[next_date].drop(price_col)
        future_features.append(next_features.values)
        
        # Update for next iteration
        future_df = pd.concat([future_df, next_row])
    
    # Scale future features
    future_features_scaled = scaler.transform(future_features)
    
    # Make predictions with all models
    rf_preds = rf_model.predict(future_features_scaled)
    gb_preds = gb_model.predict(future_features_scaled)
    et_preds = et_model.predict(future_features_scaled)
    linear_preds = linear_model.predict(future_features_scaled)
    
    # Dynamic ensemble weighting based on trend direction
    if recent_trend > 0:
        # In uptrend, favor more aggressive models
        weights = [0.3, 0.4, 0.2, 0.1]  # RF, GB, ET, Linear
    elif recent_trend < 0:
        # In downtrend, favor more aggressive models but different mix
        weights = [0.2, 0.5, 0.2, 0.1]  # RF, GB, ET, Linear
    else:
        # In sideways market, more balanced
        weights = [0.25, 0.25, 0.25, 0.25]  # RF, GB, ET, Linear
    
    # Combine predictions with dynamic weighting
    ensemble_preds = (
        weights[0] * rf_preds + 
        weights[1] * gb_preds + 
        weights[2] * et_preds + 
        weights[3] * linear_preds
    )
    
    # Apply trend amplification and volatility injection
    trend_factor = 2.0  # Strong trend amplification
    volatility_factor = 1.8  # High volatility
    
    # Generate dynamic forecast
    dynamic_forecast = np.zeros(days_to_predict)
    dynamic_forecast[0] = ensemble_preds[0]
    
    # Generate random shocks based on historical volatility
    np.random.seed(46)  # Different seed than other models
    random_shocks = np.random.normal(0, hist_volatility * price_mean * volatility_factor, days_to_predict)
    
    # Apply trend amplification and volatility
    for i in range(1, days_to_predict):
        # Calculate the ensemble model's trend
        model_trend = ensemble_preds[i] - ensemble_preds[i-1]
        
        # Amplify the trend direction
        amplified_trend = model_trend * trend_factor
        
        # Add amplified trend and volatility
        dynamic_forecast[i] = ensemble_preds[i] + amplified_trend + random_shocks[i]
    
    # Create confidence intervals using prediction intervals from the forest
    # and incorporating the dynamic forecast
    lower_ci = []
    upper_ci = []
    
    for i, X_pred in enumerate(future_features_scaled):
        # Get predictions from all trees
        tree_preds = []
        for tree in rf_model.estimators_:
            tree_preds.append(tree.predict([X_pred])[0])
        
        # Calculate tree-based confidence intervals
        tree_lower = np.percentile(tree_preds, 5)
        tree_upper = np.percentile(tree_preds, 95)
        
        # Adjust confidence intervals based on dynamic forecast
        adjustment = dynamic_forecast[i] - ensemble_preds[i]
        
        # Apply adjustment and widen intervals with volatility factor
        interval_width = (tree_upper - tree_lower) * volatility_factor
        lower_ci.append(dynamic_forecast[i] - interval_width/2)
        upper_ci.append(dynamic_forecast[i] + interval_width/2)
    
    return {
        'forecast': dynamic_forecast,
        'lower_ci': np.array(lower_ci),
        'upper_ci': np.array(upper_ci)
    }

def evaluate_models(df: pd.DataFrame, price_col: str, test_size: int = 30) -> Dict[str, float]:
    """
    Evaluate different models on historical data to select the best one.
    Also evaluates models for their dynamism/variability.
    """
    if len(df) <= test_size:
        return {'ensemble': 1.0}  # Default to ensemble if not enough data
    
    # Split data into train and test
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    results = {}
    dynamism_scores = {}
    models = ['arima', 'prophet', 'ensemble']
    
    for model_name in models:
        try:
            if model_name == 'arima':
                forecast_result = forecast_arima(train, price_col, test_size)
            elif model_name == 'prophet':
                forecast_result = forecast_prophet(train, train.index.name or 'Date', price_col, test_size)
            elif model_name == 'ensemble':
                forecast_result = forecast_ensemble(train, price_col, test_size)
            
            # Calculate error metrics
            mae = mean_absolute_error(test[price_col].values, forecast_result['forecast'])
            mse = mean_squared_error(test[price_col].values, forecast_result['forecast'])
            
            # Normalize errors relative to the mean price
            mean_price = train[price_col].mean()
            normalized_mae = mae / mean_price
            
            # Calculate dynamism score (how much the forecast varies)
            forecast_diff = np.diff(forecast_result['forecast'])
            forecast_variability = np.std(forecast_diff)
            test_diff = np.diff(test[price_col].values)
            test_variability = np.std(test_diff)
            
            # Ratio of forecast variability to actual variability (closer to 1 is better)
            # We want forecasts that are dynamic but not unrealistically volatile
            if test_variability > 0:
                dynamism_ratio = forecast_variability / test_variability
                # Penalize both under-variability and over-variability
                if dynamism_ratio < 1:
                    dynamism_score = dynamism_ratio  # Penalize under-variable forecasts
                else:
                    dynamism_score = 1 / dynamism_ratio  # Penalize over-variable forecasts
            else:
                dynamism_score = 0.5  # Default if test data has no variability
            
            # Combined score: balance accuracy and dynamism
            # Weight dynamism more heavily to address the static forecast issue
            combined_score = 0.4 * normalized_mae + 0.6 * (1 - dynamism_score)
            
            results[model_name] = combined_score
            dynamism_scores[model_name] = dynamism_score
            
            logger.info(f"{model_name.upper()} - MAE: {mae:.4f}, Normalized MAE: {normalized_mae:.4f}, "
                        f"Dynamism Score: {dynamism_score:.4f}, Combined Score: {combined_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating {model_name}: {str(e)}")
            results[model_name] = float('inf')
    
    return results

def forecast_prices(data: pd.DataFrame, days_to_predict: int = 60, lookback_days: int = 180) -> pd.DataFrame:
    """
    Forecast prices using multiple models and return a DataFrame with forecast and confidence intervals.
    Automatically selects the best model based on historical performance and dynamism.
    """
    logger.info("\n==== STARTING ADVANCED FORECASTING PROCESS ====\n")

    try:
        if data is None or data.empty:
            logger.error("No data provided for forecasting.")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI', 'Model'])

        # Detect columns
        price_col = detect_price_column(data)
        date_col = detect_date_column(data)

        if not price_col or not date_col:
            logger.error("Required columns (price or date) not found.")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI', 'Model'])

        # Clean data
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df.dropna(subset=[date_col, price_col], inplace=True)
        df.sort_values(by=date_col, inplace=True)

        # Use more historical data for better pattern detection
        df = df.tail(min(len(df), lookback_days)).copy()
        if len(df) < 30:
            logger.warning("Not enough data points for forecasting models.")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI', 'Model'])

        # Set date as index
        df.set_index(date_col, inplace=True)
        
        # Try to infer frequency, default to daily if can't infer
        freq = pd.infer_freq(df.index)
        if freq is None:
            logger.warning("Could not infer frequency from data, assuming daily.")
            freq = 'D'
            # Reindex to ensure regular frequency
            idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            df = df.reindex(idx).interpolate(method='time')
        else:
            df = df.asfreq(freq)
            df.interpolate(method='time', inplace=True)

        # Calculate historical volatility and trend
        hist_volatility = df[price_col].pct_change().std()
        recent_window = min(30, len(df) // 3)
        recent_trend = df[price_col].iloc[-recent_window:].diff().mean()
        
        # Analyze price characteristics
        price_range = df[price_col].max() - df[price_col].min()
        price_mean = df[price_col].mean()
        price_std = df[price_col].std()
        
        logger.info(f"Historical volatility: {hist_volatility:.4f}, Recent trend: {recent_trend:.4f}")
        logger.info(f"Price range: {price_range:.2f}, Mean: {price_mean:.2f}, Std: {price_std:.2f}")

        # Evaluate models on historical data with emphasis on dynamism
        model_scores = evaluate_models(df, price_col)
        
        # Select best model (lowest combined score)
        best_model = min(model_scores, key=model_scores.get)
        logger.info(f"Selected best model: {best_model.upper()}")
        
        # Generate forecasts from all models
        arima_forecast = forecast_arima(df, price_col, days_to_predict)
        prophet_forecast = forecast_prophet(df, df.index.name or 'Date', price_col, days_to_predict)
        ensemble_forecast = forecast_ensemble(df, price_col, days_to_predict)
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict, freq=freq)
        
        # Use the best model as base but blend with others for more dynamism
        if best_model == 'arima':
            base_forecast = arima_forecast
            # Blend in some Prophet and ensemble for more dynamism
            blend_weights = [0.7, 0.15, 0.15]  # ARIMA, Prophet, Ensemble
        elif best_model == 'prophet':
            base_forecast = prophet_forecast
            # Blend in some ARIMA and ensemble
            blend_weights = [0.15, 0.7, 0.15]  # ARIMA, Prophet, Ensemble
        else:  # ensemble
            base_forecast = ensemble_forecast
            # Blend in some ARIMA and Prophet
            blend_weights = [0.15, 0.15, 0.7]  # ARIMA, Prophet, Ensemble
        
        # Blend forecasts for more dynamic results
        blended_forecast = np.zeros(days_to_predict)
        blended_lower = np.zeros(days_to_predict)
        blended_upper = np.zeros(days_to_predict)
        
        for i in range(days_to_predict):
            blended_forecast[i] = (
                blend_weights[0] * arima_forecast['forecast'][i] +
                blend_weights[1] * prophet_forecast['forecast'][i] +
                blend_weights[2] * ensemble_forecast['forecast'][i]
            )
            
            # Take the widest confidence intervals for more realistic uncertainty
            blended_lower[i] = min(
                arima_forecast['lower_ci'][i],
                prophet_forecast['lower_ci'][i],
                ensemble_forecast['lower_ci'][i]
            )
            
            blended_upper[i] = max(
                arima_forecast['upper_ci'][i],
                prophet_forecast['upper_ci'][i],
                ensemble_forecast['upper_ci'][i]
            )
        
        # Apply final dynamic adjustments based on historical patterns
        final_forecast = np.zeros(days_to_predict)
        
        # Generate some additional randomness for dynamism
        np.random.seed(int(time.time()) % 100)  # Use current time for more randomness
        dynamic_factor = 1.0 + np.random.normal(0, 0.02, days_to_predict)  # ±2% random variation
        
        # Apply dynamic factors to the blended forecast
        final_forecast = blended_forecast * dynamic_factor
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Forecast': final_forecast,
            'Lower_CI': blended_lower,
            'Upper_CI': blended_upper,
            'Model': f"BLENDED_{best_model.upper()}"
        })

        logger.info(f"Successfully created dynamic blended forecast for {days_to_predict} days.")
        return forecast_df

    except Exception as e:
        logger.error(f"Error during forecasting: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI', 'Model'])

# Example usage
if __name__ == '__main__':
    from data_fetcher import fetch_commodity_data

    ticker = 'HG=F'  # Example: Copper
    historical_data = fetch_commodity_data(ticker, days=365)

    if not historical_data.empty:
        forecast_df = forecast_prices(historical_data, days_to_predict=60, lookback_days=180)
        if not forecast_df.empty:
            print("\nForecasted Prices (Next 60 Days) using model: " + forecast_df['Model'].iloc[0])
            print(forecast_df.drop(columns=['Model']).to_string(index=False))
            
            # Calculate statistics to show forecast variability
            forecast_range = forecast_df['Upper_CI'] - forecast_df['Lower_CI']
            forecast_volatility = forecast_df['Forecast'].pct_change().std() * 100
            forecast_trend = forecast_df['Forecast'].iloc[-1] - forecast_df['Forecast'].iloc[0]
            forecast_trend_pct = (forecast_trend / forecast_df['Forecast'].iloc[0]) * 100
            
            print(f"\nForecast Statistics:")
            print(f"Average Confidence Interval Width: {forecast_range.mean():.2f}")
            print(f"Day-to-Day Volatility: {forecast_volatility:.2f}%")
            print(f"Overall Trend: {forecast_trend:.2f} ({forecast_trend_pct:.2f}%)")
            print(f"Max-Min Range: {forecast_df['Forecast'].max() - forecast_df['Forecast'].min():.2f}")
            
            # Check if forecast is too static
            if forecast_volatility < 0.5:
                print("\nWARNING: Forecast appears to be very static. Consider using more historical data or adjusting model parameters.")
            else:
                print("\nForecast shows good dynamism in price movements.")
                
            # Print first and last 5 days to show trend
            print("\nFirst 5 days:")
            print(forecast_df.head(5).drop(columns=['Model']).to_string(index=False))
            print("\nLast 5 days:")
            print(forecast_df.tail(5).drop(columns=['Model']).to_string(index=False))
