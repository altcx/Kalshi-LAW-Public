# Backtesting Framework

This comprehensive backtesting framework provides tools for evaluating temperature prediction models and trading strategies for Kalshi weather contracts.

## Overview

The backtesting framework consists of several key components:

1. **Historical Data Loading** - Loads and manages historical weather data for backtesting
2. **Performance Metrics** - Calculates comprehensive accuracy and error metrics
3. **Trading Simulation** - Simulates Kalshi contract trading with realistic costs and constraints
4. **Walk-Forward Analysis** - Implements time-series cross-validation for robust model evaluation
5. **Model Comparison** - Compares multiple models and strategies systematically
6. **Comprehensive Framework** - Ties everything together with a unified interface

## Key Features

### Historical Data Management
- Loads data from multiple weather sources (NWS, OpenWeatherMap, Tomorrow.io, etc.)
- Validates data quality and completeness
- Simulates real-time prediction scenarios using historical data
- Supports walk-forward analysis with configurable windows

### Performance Metrics
- **Accuracy Metrics**: Within ±3°F, ±5°F, and custom thresholds
- **Error Metrics**: MAE, RMSE, bias, median error
- **Confidence Analysis**: Calibration, correlation with accuracy
- **Seasonal Analysis**: Performance by season and weather patterns
- **Trend Analysis**: Performance stability over time

### Trading Simulation
- **Synthetic Contracts**: Creates realistic Kalshi-style temperature contracts
- **Position Sizing**: Kelly criterion-based optimal sizing
- **Risk Management**: Maximum position limits, transaction costs
- **Performance Tracking**: ROI, Sharpe ratio, maximum drawdown, win rate
- **Realistic Constraints**: Market prices, confidence thresholds, edge requirements

### Model Comparison
- **Multi-Model Testing**: Compare different ML strategies simultaneously
- **Optimization Scoring**: Weighted scoring across multiple metrics
- **Seasonal Performance**: Identify best models for different weather conditions
- **Ensemble Optimization**: Calculate optimal model weights for ensembles

## Usage Examples

### Basic Single Model Backtest

```python
from src.backtesting.backtesting_framework import BacktestingFramework
from src.models.xgboost_model import XGBoostTemperatureModel

# Initialize framework
framework = BacktestingFramework()

# Define model factory
def xgboost_factory(**params):
    return XGBoostTemperatureModel(**params)

# Run backtest
results = framework.run_single_model_backtest(
    model_factory=xgboost_factory,
    model_params={'n_estimators': 100, 'max_depth': 6},
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

print(f"Overall Accuracy: {results['summary']['overall_accuracy_3f']:.1f}%")
print(f"Trading Return: {results['summary']['trading_return_pct']:.1f}%")
```

### Model Comparison

```python
from src.backtesting.model_comparison import ModelConfig

# Define multiple models to compare
model_configs = [
    ModelConfig(
        name="XGBoost",
        model_factory=xgboost_factory,
        params={'n_estimators': 100, 'max_depth': 6},
        description="XGBoost with default parameters"
    ),
    ModelConfig(
        name="LightGBM",
        model_factory=lightgbm_factory,
        params={'n_estimators': 100, 'max_depth': 6},
        description="LightGBM with default parameters"
    )
]

# Run comparison
comparison_results = framework.run_model_comparison_backtest(
    model_configs=model_configs,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31)
)

# Get best model
best_model = comparison_results['comparison_report']['summary']['best_model']
print(f"Best model: {best_model}")
```

### Walk-Forward Analysis

```python
from src.backtesting.walk_forward_analysis import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer()

# Run walk-forward analysis
results = analyzer.run_analysis(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    model_factory=xgboost_factory,
    model_params={'n_estimators': 100},
    train_window_days=365,  # 1 year training
    test_window_days=30,    # 1 month testing
    step_days=7             # Weekly steps
)

# Analyze results
analysis = analyzer.analyze_results(results)
print(f"Overall MAE: {analysis['comprehensive_metrics']['basic_metrics']['mae']:.2f}°F")
```

### Trading Simulation Only

```python
from src.backtesting.trading_simulation import TradingSimulationEngine

# Prepare prediction data
predictions_data = [
    {
        'date': date(2024, 7, 15),
        'predicted_temp': 82.0,
        'actual_temp': 84.0,
        'confidence': 0.8
    },
    # ... more predictions
]

# Run trading simulation
engine = TradingSimulationEngine(initial_bankroll=10000.0)
trading_results = engine.run_backtest(predictions_data)

print(f"Total Return: {trading_results['performance_metrics']['total_return']:.1f}%")
print(f"Win Rate: {trading_results['performance_metrics']['win_rate']:.1f}%")
```

## Configuration Options

### Backtesting Configuration

```python
config = {
    'train_window_days': 365,      # Training window size
    'test_window_days': 30,        # Test window size  
    'step_days': 7,                # Step size between splits
    'initial_bankroll': 10000.0,   # Starting trading capital
    'parallel_execution': True     # Enable parallel processing
}
```

### Trading Configuration

```python
# Trading engine parameters
engine = TradingSimulationEngine(
    initial_bankroll=10000.0,
    transaction_cost_per_contract=0.01,  # $0.01 per contract
    max_position_size_pct=0.10,          # Max 10% of bankroll per trade
    min_confidence_threshold=0.6,        # Minimum confidence to trade
    min_edge_threshold=0.05               # Minimum edge to trade (5%)
)
```

## Output and Results

### Performance Metrics Structure

```python
{
    'basic_metrics': {
        'mae': 2.45,                    # Mean Absolute Error (°F)
        'rmse': 3.12,                   # Root Mean Square Error (°F)
        'accuracy_within_3f': 85.2,     # Accuracy within ±3°F (%)
        'accuracy_within_5f': 94.1,     # Accuracy within ±5°F (%)
        'bias': -0.15,                  # Systematic bias (°F)
        'total_predictions': 365
    },
    'confidence_metrics': {
        'avg_confidence': 0.78,
        'expected_calibration_error': 0.05,
        'confidence_accuracy_correlation': 0.42
    },
    'seasonal_metrics': {
        'winter': {'mae': 2.1, 'accuracy_within_3f': 88.5},
        'spring': {'mae': 2.3, 'accuracy_within_3f': 86.2},
        'summer': {'mae': 2.8, 'accuracy_within_3f': 82.1},
        'fall': {'mae': 2.2, 'accuracy_within_3f': 87.3}
    }
}
```

### Trading Results Structure

```python
{
    'performance_metrics': {
        'total_trades': 156,
        'win_rate': 62.8,              # Win rate (%)
        'total_return': 15.4,          # Total return (%)
        'sharpe_ratio': 1.23,          # Risk-adjusted return
        'max_drawdown': 8.7,           # Maximum drawdown (%)
        'profit_factor': 1.45          # Gross profit / gross loss
    },
    'trading_history': [
        {
            'date': '2024-07-15',
            'contract_id': 'LA_TEMP_ABOVE_85F_20240715',
            'action': 'BUY',
            'quantity': 100,
            'price': 65.0,
            'predicted_temperature': 87.2,
            'actual_temperature': 89.1,
            'confidence': 0.82,
            'pnl': 35.0,
            'is_winner': True
        }
        # ... more trades
    ]
}
```

## File Structure

```
src/backtesting/
├── __init__.py
├── README.md
├── backtesting_framework.py      # Main framework interface
├── historical_data_loader.py     # Historical data management
├── performance_metrics.py        # Metrics calculation
├── trading_simulation.py         # Trading simulation engine
├── walk_forward_analysis.py      # Walk-forward analysis
├── model_comparison.py           # Model comparison tools
├── test_historical_data_loader.py
├── test_trading_simulation.py
└── test_backtesting_framework.py
```

## Requirements Satisfied

This backtesting framework satisfies the following requirements from the specification:

- **4.2**: Comprehensive performance metrics (RMSE, MAE, accuracy within ±3°F)
- **4.3**: Trading simulation with realistic transaction costs and ROI calculation
- **4.4**: Model comparison across different ML strategies and ensemble methods
- **3.2**: Kalshi contract analysis and trading recommendations
- **6.4**: Seasonal performance analysis across different weather patterns

## Testing

Run the test suite to verify functionality:

```bash
python src/backtesting/test_historical_data_loader.py
python src/backtesting/test_trading_simulation.py
python src/backtesting/test_backtesting_framework.py
```

All tests should pass, indicating the framework is working correctly with the available historical data.

## Next Steps

1. **Integration**: Connect with actual ML models from `src/models/`
2. **Real Data**: Use with complete historical weather datasets
3. **Optimization**: Fine-tune trading parameters and model selection
4. **Automation**: Set up automated backtesting pipelines
5. **Visualization**: Add charts and graphs for results analysis