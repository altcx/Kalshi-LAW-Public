# Kalshi Weather Predictor

Public not fully implemented btw

A machine learning-powered system for predicting daily high temperatures in Los Angeles to inform Kalshi weather futures trading decisions.

## Project Structure 
 
``` 
kalshi-weather-predictor/
├── config/
│   └── config.yaml              # Main configuration file
├── data/                        # Data storage directory
├── logs/                        # Application logs
├── src/
│   ├── data_collection/         # Weather API clients
│   ├── feature_engineering/     # Feature creation and processing
│   ├── models/                  # ML models and ensemble
│   ├── trading/                 # Trading recommendation engine
│   ├── utils/                   # Utility functions
│   └── main.py                  # Main application entry point
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Copy the environment template:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your API keys:
   - OpenWeatherMap: https://openweathermap.org/api
   - Tomorrow.io: https://www.tomorrow.io/weather-api/
   - Weatherbit: https://www.weatherbit.io/api
   - Visual Crossing: https://www.visualcrossing.com/weather-api

3. Review and modify `config/config.yaml` as needed

### 3. Run the Application

```bash
python src/main.py
```

## Features

- **Multi-source data collection**: Aggregates weather data from 5+ APIs
- **ML ensemble models**: XGBoost, LightGBM, Linear Regression, Prophet
- **Parquet data storage**: Efficient columnar storage for time series data
- **Trading recommendations**: Specific Kalshi contract analysis
- **Comprehensive backtesting**: Historical performance validation
- **Real-time monitoring**: Daily accuracy tracking and model updates

## Data Sources

- **NWS API**: National Weather Service (free, unlimited)
- **OpenWeatherMap**: 1000 calls/day free tier
- **Tomorrow.io**: 1000 calls/day free tier
- **Weatherbit**: 500 calls/day free tier
- **Visual Crossing**: 1000 records/day free tier
- **NOAA**: Actual temperature observations (LAX station)

## Next Steps

This setup provides the foundation for the weather prediction system. The next tasks will implement:

1. Weather API clients for data collection
2. Data storage system with Parquet files
3. Feature engineering pipeline
4. Machine learning model ensemble
5. Trading recommendation engine
6. Backtesting framework
7. User interface and dashboard

## Requirements

- Python 3.8+
- Internet connection for API access
- ~100MB disk space for data storage
- API keys for weather services (free tiers available)
- A lot of RAM
