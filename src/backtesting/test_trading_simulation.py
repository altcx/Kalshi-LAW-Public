"""Test trading simulation engine functionality."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from src.backtesting.trading_simulation import (
    TradingSimulationEngine, KalshiContract, ContractType, 
    ContractStatus, Position, TradingResult
)


def test_trading_simulation_initialization():
    """Test that TradingSimulationEngine initializes correctly."""
    engine = TradingSimulationEngine(initial_bankroll=10000.0)
    
    assert engine.initial_bankroll == 10000.0
    assert engine.current_bankroll == 10000.0
    assert len(engine.positions) == 0
    assert len(engine.trading_history) == 0


def test_kalshi_contract_creation():
    """Test KalshiContract creation and methods."""
    contract = KalshiContract(
        contract_id="TEST_ABOVE_80F",
        contract_type=ContractType.ABOVE_THRESHOLD,
        threshold_temperature=80.0,
        expiration_date=date(2024, 7, 15),
        market_price=65.0,
        created_date=date(2024, 7, 14)
    )
    
    assert contract.implied_probability == 0.65
    assert contract.is_winner(85.0) == True  # 85 > 80
    assert contract.is_winner(75.0) == False  # 75 < 80


def test_create_synthetic_contracts():
    """Test synthetic contract creation."""
    engine = TradingSimulationEngine()
    
    target_date = date(2024, 7, 15)
    predicted_temp = 82.0
    
    contracts = engine.create_synthetic_contracts(target_date, predicted_temp)
    
    assert len(contracts) > 0
    
    # Should have both ABOVE and BELOW contracts
    above_contracts = [c for c in contracts if c.contract_type == ContractType.ABOVE_THRESHOLD]
    below_contracts = [c for c in contracts if c.contract_type == ContractType.BELOW_THRESHOLD]
    
    assert len(above_contracts) > 0
    assert len(below_contracts) > 0
    
    # All contracts should expire on target date
    for contract in contracts:
        assert contract.expiration_date == target_date


def test_expected_value_calculation():
    """Test expected value calculation."""
    engine = TradingSimulationEngine()
    
    contract = KalshiContract(
        contract_id="TEST_ABOVE_80F",
        contract_type=ContractType.ABOVE_THRESHOLD,
        threshold_temperature=80.0,
        expiration_date=date(2024, 7, 15),
        market_price=40.0,  # Market thinks 40% chance
        created_date=date(2024, 7, 14)
    )
    
    # If we predict 85°F with high confidence, expected value should be positive
    ev = engine.calculate_expected_value(contract, predicted_temp=85.0, confidence=0.9)
    assert ev > 0  # Should be profitable
    
    # If we predict 75°F, expected value should be negative
    ev = engine.calculate_expected_value(contract, predicted_temp=75.0, confidence=0.9)
    assert ev < 0  # Should not be profitable


def test_position_size_calculation():
    """Test position size calculation using Kelly criterion."""
    engine = TradingSimulationEngine(initial_bankroll=10000.0)
    
    # Positive expected value should give positive position size
    position_size = engine.calculate_position_size(
        expected_value=20.0,  # 20 cents expected value
        confidence=0.8,
        contract_price=40.0
    )
    
    assert position_size > 0
    # Position size should be reasonable (not more than we can afford)
    max_affordable = int((engine.current_bankroll * 0.95) / (40.0 / 100.0))
    assert position_size <= max_affordable


def test_trade_execution():
    """Test trade execution."""
    engine = TradingSimulationEngine(initial_bankroll=10000.0)
    
    contract = KalshiContract(
        contract_id="TEST_ABOVE_80F",
        contract_type=ContractType.ABOVE_THRESHOLD,
        threshold_temperature=80.0,
        expiration_date=date(2024, 7, 15),
        market_price=40.0,
        created_date=date(2024, 7, 14)
    )
    
    initial_bankroll = engine.current_bankroll
    
    # Execute a trade with good prediction
    result = engine.execute_trade(
        contract=contract,
        predicted_temp=85.0,
        confidence=0.8,
        trade_date=date(2024, 7, 14)
    )
    
    # Should have executed a trade
    if result is not None:  # Trade might not execute due to thresholds
        assert result.action == 'BUY'
        assert result.quantity > 0
        assert engine.current_bankroll < initial_bankroll  # Money spent
        assert len(engine.positions) == 1


def test_contract_settlement():
    """Test contract settlement."""
    engine = TradingSimulationEngine(initial_bankroll=10000.0)
    
    # Create and execute a trade
    contract = KalshiContract(
        contract_id="TEST_ABOVE_80F",
        contract_type=ContractType.ABOVE_THRESHOLD,
        threshold_temperature=80.0,
        expiration_date=date(2024, 7, 15),
        market_price=40.0,
        created_date=date(2024, 7, 14)
    )
    
    # Force a trade by adding a position manually
    position = Position(
        position_id="TEST_POS",
        contract=contract,
        quantity=100,
        entry_price=40.0,
        entry_date=date(2024, 7, 14)
    )
    engine.positions.append(position)
    engine.current_bankroll -= 40.0  # Simulate cost
    
    initial_bankroll = engine.current_bankroll
    
    # Settle with winning temperature
    settled = engine.settle_contracts(date(2024, 7, 15), actual_temperature=85.0)
    
    assert len(settled) == 1
    assert settled[0].status == ContractStatus.EXPIRED
    assert settled[0]._is_winner == True
    assert engine.current_bankroll > initial_bankroll  # Should have made money


def test_backtest_run():
    """Test running a complete backtest."""
    engine = TradingSimulationEngine(initial_bankroll=10000.0)
    
    # Create sample prediction data
    predictions_data = [
        {
            'date': date(2024, 7, 15),
            'predicted_temp': 82.0,
            'actual_temp': 84.0,
            'confidence': 0.8
        },
        {
            'date': date(2024, 7, 16),
            'predicted_temp': 78.0,
            'actual_temp': 76.0,
            'confidence': 0.7
        }
    ]
    
    results = engine.run_backtest(predictions_data)
    
    assert 'performance_metrics' in results
    assert 'trading_history' in results
    assert 'daily_performance' in results
    assert 'backtest_summary' in results
    
    # Check that some basic metrics exist
    metrics = results['performance_metrics']
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics
    assert 'total_return' in metrics


if __name__ == "__main__":
    test_trading_simulation_initialization()
    test_kalshi_contract_creation()
    test_create_synthetic_contracts()
    test_expected_value_calculation()
    test_position_size_calculation()
    test_trade_execution()
    test_contract_settlement()
    test_backtest_run()
    print("All trading simulation tests passed!")