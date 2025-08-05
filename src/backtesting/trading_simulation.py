"""Trading simulation engine for Kalshi weather contracts backtesting."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from enum import Enum
import math


class ContractType(Enum):
    """Types of Kalshi weather contracts."""
    ABOVE_THRESHOLD = "above"
    BELOW_THRESHOLD = "below"


class ContractStatus(Enum):
    """Status of a contract position."""
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"


@dataclass
class KalshiContract:
    """Represents a Kalshi weather contract."""
    contract_id: str
    contract_type: ContractType
    threshold_temperature: float
    expiration_date: date
    market_price: float  # Price in cents (0-100)
    created_date: date
    
    @property
    def implied_probability(self) -> float:
        """Get implied probability from market price."""
        return self.market_price / 100.0
    
    def is_winner(self, actual_temperature: float) -> bool:
        """Check if contract is a winner given actual temperature."""
        if self.contract_type == ContractType.ABOVE_THRESHOLD:
            return actual_temperature > self.threshold_temperature
        else:  # BELOW_THRESHOLD
            return actual_temperature < self.threshold_temperature


@dataclass
class Position:
    """Represents a trading position in a contract."""
    position_id: str
    contract: KalshiContract
    quantity: int  # Number of contracts
    entry_price: float  # Price paid per contract (in cents)
    entry_date: date
    exit_price: Optional[float] = None
    exit_date: Optional[date] = None
    status: ContractStatus = ContractStatus.OPEN
    
    @property
    def cost_basis(self) -> float:
        """Total cost of the position."""
        return self.quantity * self.entry_price
    
    @property
    def current_value(self) -> float:
        """Current value of the position."""
        if self.status == ContractStatus.CLOSED and self.exit_price is not None:
            return self.quantity * self.exit_price
        elif self.status == ContractStatus.EXPIRED:
            # Contract expired, check if it was a winner
            return self.quantity * 100.0 if self.is_winner else 0.0
        else:
            # Still open, use current market price
            return self.quantity * self.contract.market_price
    
    @property
    def pnl(self) -> float:
        """Profit and loss of the position."""
        return self.current_value - self.cost_basis
    
    @property
    def is_winner(self) -> bool:
        """Check if position is a winner (for expired contracts)."""
        return hasattr(self, '_is_winner') and self._is_winner


@dataclass
class TradingResult:
    """Result of a single trade."""
    date: date
    contract_id: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    cost: float
    predicted_temperature: float
    actual_temperature: Optional[float]
    confidence: float
    pnl: float
    is_winner: Optional[bool] = None


class TradingSimulationEngine:
    """Simulates trading Kalshi weather contracts based on predictions."""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        """Initialize the trading simulation engine.
        
        Args:
            initial_bankroll: Starting bankroll in dollars
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Trading configuration
        self.transaction_cost_per_contract = 0.01  # $0.01 per contract
        self.max_position_size_pct = 0.10  # Max 10% of bankroll per trade
        self.min_confidence_threshold = 0.6  # Minimum confidence to trade
        self.min_edge_threshold = 0.05  # Minimum edge to trade (5%)
        
        # Position tracking
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.trading_history: List[TradingResult] = []
        
        # Performance tracking
        self.daily_pnl: List[Dict] = []
        self.max_drawdown = 0.0
        self.peak_bankroll = initial_bankroll
        
        logger.info(f"TradingSimulationEngine initialized with ${initial_bankroll:,.2f} bankroll")
    
    def create_synthetic_contracts(self, target_date: date, 
                                 predicted_temp: float,
                                 actual_temp: Optional[float] = None) -> List[KalshiContract]:
        """Create synthetic Kalshi contracts for a given date and prediction.
        
        Args:
            target_date: Date the contracts expire
            predicted_temp: Predicted temperature
            actual_temp: Actual temperature (for backtesting)
            
        Returns:
            List of synthetic contracts
        """
        contracts = []
        
        # Create contracts at various thresholds around the prediction
        base_thresholds = [70, 75, 80, 85, 90, 95]  # Common LA temperature thresholds
        
        for threshold in base_thresholds:
            # Skip thresholds that are too far from prediction (unrealistic)
            if abs(threshold - predicted_temp) > 20:
                continue
            
            # Calculate synthetic market prices based on prediction
            # This simulates market efficiency with some noise
            
            # For ABOVE threshold contracts
            if predicted_temp > threshold:
                # Prediction suggests contract should win
                prob_above = min(0.95, 0.5 + (predicted_temp - threshold) / 20.0)
            else:
                # Prediction suggests contract should lose
                prob_above = max(0.05, 0.5 - (threshold - predicted_temp) / 20.0)
            
            # Add some market noise (±5%)
            noise = np.random.normal(0, 0.05)
            market_prob_above = np.clip(prob_above + noise, 0.05, 0.95)
            market_price_above = market_prob_above * 100
            
            # Create ABOVE contract
            above_contract = KalshiContract(
                contract_id=f"LA_TEMP_ABOVE_{threshold}F_{target_date.strftime('%Y%m%d')}",
                contract_type=ContractType.ABOVE_THRESHOLD,
                threshold_temperature=float(threshold),
                expiration_date=target_date,
                market_price=market_price_above,
                created_date=target_date - timedelta(days=1)
            )
            contracts.append(above_contract)
            
            # Create BELOW contract (complementary)
            market_price_below = 100 - market_price_above
            below_contract = KalshiContract(
                contract_id=f"LA_TEMP_BELOW_{threshold}F_{target_date.strftime('%Y%m%d')}",
                contract_type=ContractType.BELOW_THRESHOLD,
                threshold_temperature=float(threshold),
                expiration_date=target_date,
                market_price=market_price_below,
                created_date=target_date - timedelta(days=1)
            )
            contracts.append(below_contract)
        
        logger.debug(f"Created {len(contracts)} synthetic contracts for {target_date}")
        return contracts
    
    def calculate_expected_value(self, contract: KalshiContract, 
                               predicted_temp: float, confidence: float) -> float:
        """Calculate expected value of a contract given prediction.
        
        Args:
            contract: KalshiContract to evaluate
            predicted_temp: Predicted temperature
            confidence: Confidence in prediction (0-1)
            
        Returns:
            Expected value in cents
        """
        # Calculate our estimated probability of contract winning
        if contract.contract_type == ContractType.ABOVE_THRESHOLD:
            if predicted_temp > contract.threshold_temperature:
                # We think it will be above threshold
                temp_diff = predicted_temp - contract.threshold_temperature
                base_prob = 0.5 + min(0.45, temp_diff / 20.0)  # Max 95% probability
            else:
                # We think it will be below threshold
                temp_diff = contract.threshold_temperature - predicted_temp
                base_prob = 0.5 - min(0.45, temp_diff / 20.0)  # Min 5% probability
        else:  # BELOW_THRESHOLD
            if predicted_temp < contract.threshold_temperature:
                # We think it will be below threshold
                temp_diff = contract.threshold_temperature - predicted_temp
                base_prob = 0.5 + min(0.45, temp_diff / 20.0)  # Max 95% probability
            else:
                # We think it will be above threshold
                temp_diff = predicted_temp - contract.threshold_temperature
                base_prob = 0.5 - min(0.45, temp_diff / 20.0)  # Min 5% probability
        
        # Adjust probability based on confidence
        # Lower confidence means probability moves toward 50%
        adjusted_prob = 0.5 + (base_prob - 0.5) * confidence
        adjusted_prob = np.clip(adjusted_prob, 0.05, 0.95)
        
        # Expected value = (probability of winning * payout) - cost
        payout_if_win = 100.0  # Contracts pay $1.00 (100 cents) if they win
        expected_value = (adjusted_prob * payout_if_win) - contract.market_price
        
        return expected_value
    
    def calculate_position_size(self, expected_value: float, confidence: float,
                              contract_price: float) -> int:
        """Calculate optimal position size using Kelly criterion.
        
        Args:
            expected_value: Expected value of the trade in cents
            confidence: Confidence in prediction (0-1)
            contract_price: Price of contract in cents
            
        Returns:
            Number of contracts to trade
        """
        if expected_value <= 0:
            return 0
        
        # Kelly fraction = (bp - q) / b
        # where b = odds received, p = probability of winning, q = probability of losing
        
        # Convert to Kelly formula terms
        win_prob = (expected_value + contract_price) / 100.0
        win_prob = np.clip(win_prob, 0.01, 0.99)
        
        lose_prob = 1 - win_prob
        odds_received = (100.0 - contract_price) / contract_price
        
        if odds_received <= 0:
            return 0
        
        # Kelly fraction
        kelly_fraction = (win_prob * odds_received - lose_prob) / odds_received
        
        # Apply confidence scaling and safety limits
        kelly_fraction *= confidence  # Scale by confidence
        kelly_fraction = min(kelly_fraction, self.max_position_size_pct)  # Cap at max position size
        kelly_fraction = max(kelly_fraction, 0)  # No negative positions
        
        # Convert to number of contracts
        max_investment = self.current_bankroll * kelly_fraction
        contract_cost_dollars = contract_price / 100.0  # Convert cents to dollars
        
        if contract_cost_dollars <= 0:
            return 0
        
        num_contracts = int(max_investment / contract_cost_dollars)
        
        # Ensure we don't exceed bankroll
        total_cost = num_contracts * contract_cost_dollars
        if total_cost > self.current_bankroll * 0.95:  # Leave 5% buffer
            num_contracts = int((self.current_bankroll * 0.95) / contract_cost_dollars)
        
        return max(0, num_contracts)
    
    def execute_trade(self, contract: KalshiContract, predicted_temp: float,
                     confidence: float, trade_date: date) -> Optional[TradingResult]:
        """Execute a trade based on prediction and contract analysis.
        
        Args:
            contract: KalshiContract to potentially trade
            predicted_temp: Predicted temperature
            confidence: Confidence in prediction
            trade_date: Date of the trade
            
        Returns:
            TradingResult if trade was executed, None otherwise
        """
        # Calculate expected value
        expected_value = self.calculate_expected_value(contract, predicted_temp, confidence)
        
        # Check if trade meets our criteria
        edge = expected_value / contract.market_price if contract.market_price > 0 else 0
        
        if confidence < self.min_confidence_threshold:
            logger.debug(f"Skipping trade: confidence {confidence:.3f} below threshold {self.min_confidence_threshold}")
            return None
        
        if edge < self.min_edge_threshold:
            logger.debug(f"Skipping trade: edge {edge:.3f} below threshold {self.min_edge_threshold}")
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(expected_value, confidence, contract.market_price)
        
        if quantity <= 0:
            logger.debug("Skipping trade: position size is 0")
            return None
        
        # Calculate costs
        contract_cost = (quantity * contract.market_price) / 100.0  # Convert to dollars
        transaction_costs = quantity * self.transaction_cost_per_contract
        total_cost = contract_cost + transaction_costs
        
        # Check if we have enough bankroll
        if total_cost > self.current_bankroll:
            logger.warning(f"Insufficient bankroll for trade: need ${total_cost:.2f}, have ${self.current_bankroll:.2f}")
            return None
        
        # Execute the trade
        self.current_bankroll -= total_cost
        
        # Create position
        position = Position(
            position_id=f"POS_{len(self.positions) + 1}_{trade_date.strftime('%Y%m%d')}",
            contract=contract,
            quantity=quantity,
            entry_price=contract.market_price,
            entry_date=trade_date,
            status=ContractStatus.OPEN
        )
        
        self.positions.append(position)
        
        # Create trading result
        result = TradingResult(
            date=trade_date,
            contract_id=contract.contract_id,
            action='BUY',
            quantity=quantity,
            price=contract.market_price,
            cost=total_cost,
            predicted_temperature=predicted_temp,
            actual_temperature=None,  # Will be filled in later
            confidence=confidence,
            pnl=0.0  # Will be calculated when position closes
        )
        
        self.trading_history.append(result)
        
        logger.info(f"Executed trade: {contract.contract_id}, quantity={quantity}, "
                   f"cost=${total_cost:.2f}, expected_value={expected_value:.2f}")
        
        return result
    
    def settle_contracts(self, settlement_date: date, actual_temperature: float) -> List[Position]:
        """Settle all contracts expiring on the given date.
        
        Args:
            settlement_date: Date to settle contracts
            actual_temperature: Actual temperature for settlement
            
        Returns:
            List of settled positions
        """
        settled_positions = []
        
        # Find positions expiring on this date
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            if position.contract.expiration_date == settlement_date:
                # Determine if contract won
                is_winner = position.contract.is_winner(actual_temperature)
                
                # Calculate settlement value
                settlement_value = 100.0 if is_winner else 0.0
                total_payout = (position.quantity * settlement_value) / 100.0  # Convert to dollars
                
                # Update position
                position.exit_price = settlement_value
                position.exit_date = settlement_date
                position.status = ContractStatus.EXPIRED
                position._is_winner = is_winner
                
                # Update bankroll
                self.current_bankroll += total_payout
                
                # Move to closed positions
                self.positions.remove(position)
                self.closed_positions.append(position)
                settled_positions.append(position)
                
                # Update trading history
                for trade_result in self.trading_history:
                    if (trade_result.contract_id == position.contract.contract_id and 
                        trade_result.date == position.entry_date):
                        trade_result.actual_temperature = actual_temperature
                        trade_result.pnl = position.pnl
                        trade_result.is_winner = is_winner
                        break
                
                logger.info(f"Settled position {position.position_id}: "
                           f"winner={is_winner}, payout=${total_payout:.2f}, pnl=${position.pnl:.2f}")
        
        return settled_positions
    
    def update_daily_performance(self, date: date) -> None:
        """Update daily performance metrics.
        
        Args:
            date: Date to update performance for
        """
        # Calculate current portfolio value
        portfolio_value = self.current_bankroll
        
        # Add value of open positions (mark-to-market)
        for position in self.positions:
            portfolio_value += position.current_value / 100.0  # Convert cents to dollars
            portfolio_value -= position.cost_basis / 100.0    # Subtract what we paid
        
        # Calculate daily P&L
        if self.daily_pnl:
            previous_value = self.daily_pnl[-1]['portfolio_value']
            daily_pnl = portfolio_value - previous_value
        else:
            daily_pnl = portfolio_value - self.initial_bankroll
        
        # Update peak and drawdown
        if portfolio_value > self.peak_bankroll:
            self.peak_bankroll = portfolio_value
        
        current_drawdown = (self.peak_bankroll - portfolio_value) / self.peak_bankroll
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record daily performance
        daily_record = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash_balance': self.current_bankroll,
            'open_positions': len(self.positions),
            'daily_pnl': daily_pnl,
            'total_return': (portfolio_value - self.initial_bankroll) / self.initial_bankroll,
            'drawdown': current_drawdown
        }
        
        self.daily_pnl.append(daily_record)
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trading_history:
            return {'error': 'No trading history available'}
        
        # Basic metrics
        total_trades = len(self.trading_history)
        winning_trades = [t for t in self.trading_history if t.is_winner is True]
        losing_trades = [t for t in self.trading_history if t.is_winner is False]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trading_history)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        if self.daily_pnl:
            daily_returns = [d['daily_pnl'] / self.initial_bankroll for d in self.daily_pnl[1:]]
            
            if daily_returns:
                volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
                
                # Sharpe ratio (assuming 0% risk-free rate)
                avg_daily_return = np.mean(daily_returns)
                sharpe_ratio = (avg_daily_return * 252) / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Current portfolio value
        current_value = self.current_bankroll
        for position in self.positions:
            current_value += position.current_value / 100.0 - position.cost_basis / 100.0
        
        total_return = (current_value - self.initial_bankroll) / self.initial_bankroll
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_return': total_return * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility * 100,
            'initial_bankroll': self.initial_bankroll,
            'current_value': current_value,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions)
        }
        
        return metrics
    
    def run_backtest(self, predictions_data: List[Dict]) -> Dict[str, Any]:
        """Run a complete backtest using historical predictions.
        
        Args:
            predictions_data: List of dictionaries with prediction data
                Each dict should have: date, predicted_temp, actual_temp, confidence
                
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(predictions_data)} predictions")
        
        for pred_data in predictions_data:
            pred_date = pred_data['date']
            if isinstance(pred_date, str):
                pred_date = datetime.strptime(pred_date, '%Y-%m-%d').date()
            
            predicted_temp = pred_data['predicted_temp']
            actual_temp = pred_data.get('actual_temp')
            confidence = pred_data['confidence']
            
            # Create synthetic contracts for this date
            contracts = self.create_synthetic_contracts(pred_date, predicted_temp, actual_temp)
            
            # Evaluate and potentially trade each contract
            for contract in contracts:
                self.execute_trade(contract, predicted_temp, confidence, pred_date)
            
            # Settle contracts if we have actual temperature
            if actual_temp is not None:
                self.settle_contracts(pred_date, actual_temp)
            
            # Update daily performance
            self.update_daily_performance(pred_date)
        
        # Calculate final performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        backtest_results = {
            'performance_metrics': performance_metrics,
            'trading_history': [
                {
                    'date': t.date.isoformat(),
                    'contract_id': t.contract_id,
                    'action': t.action,
                    'quantity': t.quantity,
                    'price': t.price,
                    'cost': t.cost,
                    'predicted_temperature': t.predicted_temperature,
                    'actual_temperature': t.actual_temperature,
                    'confidence': t.confidence,
                    'pnl': t.pnl,
                    'is_winner': t.is_winner
                }
                for t in self.trading_history
            ],
            'daily_performance': [
                {
                    'date': d['date'].isoformat(),
                    'portfolio_value': d['portfolio_value'],
                    'cash_balance': d['cash_balance'],
                    'open_positions': d['open_positions'],
                    'daily_pnl': d['daily_pnl'],
                    'total_return': d['total_return'],
                    'drawdown': d['drawdown']
                }
                for d in self.daily_pnl
            ],
            'backtest_summary': {
                'start_date': min(p['date'] for p in predictions_data).isoformat() if predictions_data else None,
                'end_date': max(p['date'] for p in predictions_data).isoformat() if predictions_data else None,
                'total_predictions': len(predictions_data),
                'total_trades': len(self.trading_history),
                'final_bankroll': self.current_bankroll,
                'total_return_pct': performance_metrics.get('total_return', 0)
            }
        }
        
        logger.info(f"Backtest completed: {len(self.trading_history)} trades, "
                   f"final return: {performance_metrics.get('total_return', 0):.2f}%")
        
        return backtest_results