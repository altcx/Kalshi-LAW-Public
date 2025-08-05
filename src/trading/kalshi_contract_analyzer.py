"""
Kalshi Contract Analysis System

This module provides functionality to analyze Kalshi temperature threshold contracts,
calculate expected values, and rank contracts by probability of success.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import date
import numpy as np
from scipy import stats


@dataclass
class KalshiContract:
    """Represents a Kalshi temperature threshold contract"""
    contract_id: str
    contract_type: str  # "ABOVE" or "BELOW"
    threshold_temp: float  # Temperature threshold (e.g., 85.0 for >85°F)
    current_price: float  # Current market price (0.0 to 1.0)
    expiry_date: date
    description: str
    
    def __post_init__(self):
        """Validate contract data"""
        if self.current_price < 0.0 or self.current_price > 1.0:
            raise ValueError(f"Contract price must be between 0.0 and 1.0, got {self.current_price}")
        if self.contract_type not in ["ABOVE", "BELOW"]:
            raise ValueError(f"Contract type must be 'ABOVE' or 'BELOW', got {self.contract_type}")


@dataclass
class ContractAnalysis:
    """Analysis results for a Kalshi contract"""
    contract: KalshiContract
    predicted_probability: float  # Probability that contract pays out (0.0 to 1.0)
    expected_value: float  # Expected value of the contract
    edge: float  # Edge over market price (positive = favorable)
    confidence_interval: Tuple[float, float]  # 95% confidence interval for probability
    recommendation: str  # "BUY", "SELL", "HOLD"
    reasoning: str  # Explanation of the recommendation


class KalshiContractAnalyzer:
    """
    Analyzes Kalshi temperature threshold contracts and provides trading recommendations
    """
    
    def __init__(self, transaction_cost: float = 0.01):
        """
        Initialize the contract analyzer
        
        Args:
            transaction_cost: Transaction cost per contract (default 1%)
        """
        self.transaction_cost = transaction_cost
    
    def analyze_contracts(
        self, 
        contracts: List[KalshiContract],
        predicted_temp: float,
        prediction_std: float,
        confidence: float
    ) -> List[ContractAnalysis]:
        """
        Analyze a list of Kalshi contracts and provide recommendations
        
        Args:
            contracts: List of available Kalshi contracts
            predicted_temp: Predicted high temperature
            prediction_std: Standard deviation of prediction (uncertainty)
            confidence: Model confidence score (0.0 to 1.0)
            
        Returns:
            List of contract analyses sorted by expected value (descending)
        """
        analyses = []
        
        for contract in contracts:
            analysis = self._analyze_single_contract(
                contract, predicted_temp, prediction_std, confidence
            )
            analyses.append(analysis)
        
        # Sort by expected value (descending)
        analyses.sort(key=lambda x: x.expected_value, reverse=True)
        
        return analyses
    
    def _analyze_single_contract(
        self,
        contract: KalshiContract,
        predicted_temp: float,
        prediction_std: float,
        confidence: float
    ) -> ContractAnalysis:
        """
        Analyze a single contract
        
        Args:
            contract: The contract to analyze
            predicted_temp: Predicted high temperature
            prediction_std: Standard deviation of prediction
            confidence: Model confidence score
            
        Returns:
            ContractAnalysis with recommendation
        """
        # Calculate probability that contract pays out
        prob_payout = self._calculate_payout_probability(
            contract, predicted_temp, prediction_std
        )
        
        # Calculate confidence interval
        conf_interval = self._calculate_confidence_interval(
            contract, predicted_temp, prediction_std
        )
        
        # Calculate expected value
        expected_value = self._calculate_expected_value(contract, prob_payout)
        
        # Calculate edge over market price
        edge = prob_payout - contract.current_price
        
        # Generate recommendation
        recommendation, reasoning = self._generate_recommendation(
            contract, prob_payout, expected_value, edge, confidence
        )
        
        return ContractAnalysis(
            contract=contract,
            predicted_probability=prob_payout,
            expected_value=expected_value,
            edge=edge,
            confidence_interval=conf_interval,
            recommendation=recommendation,
            reasoning=reasoning
        )
    
    def _calculate_payout_probability(
        self,
        contract: KalshiContract,
        predicted_temp: float,
        prediction_std: float
    ) -> float:
        """
        Calculate the probability that a contract will pay out
        
        Uses normal distribution assumption around the predicted temperature
        """
        if contract.contract_type == "ABOVE":
            # Probability that actual temp > threshold
            prob = 1 - stats.norm.cdf(
                contract.threshold_temp, 
                loc=predicted_temp, 
                scale=prediction_std
            )
        else:  # "BELOW"
            # Probability that actual temp < threshold
            prob = stats.norm.cdf(
                contract.threshold_temp,
                loc=predicted_temp,
                scale=prediction_std
            )
        
        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, prob))
    
    def _calculate_confidence_interval(
        self,
        contract: KalshiContract,
        predicted_temp: float,
        prediction_std: float
    ) -> Tuple[float, float]:
        """
        Calculate 95% confidence interval for payout probability
        """
        # For simplicity, we'll use the prediction uncertainty to estimate
        # confidence bounds on the probability
        temp_lower = predicted_temp - 1.96 * prediction_std
        temp_upper = predicted_temp + 1.96 * prediction_std
        
        if contract.contract_type == "ABOVE":
            # When temp is lower, probability of being above threshold is lower
            prob_lower = 1 - stats.norm.cdf(contract.threshold_temp, temp_lower, prediction_std)
            prob_upper = 1 - stats.norm.cdf(contract.threshold_temp, temp_upper, prediction_std)
        else:  # "BELOW"
            # When temp is lower, probability of being below threshold is higher
            prob_lower = stats.norm.cdf(contract.threshold_temp, temp_lower, prediction_std)
            prob_upper = stats.norm.cdf(contract.threshold_temp, temp_upper, prediction_std)
        
        # Ensure proper ordering
        prob_lower, prob_upper = min(prob_lower, prob_upper), max(prob_lower, prob_upper)
        
        return (max(0.0, prob_lower), min(1.0, prob_upper))
    
    def _calculate_expected_value(
        self,
        contract: KalshiContract,
        prob_payout: float
    ) -> float:
        """
        Calculate expected value of the contract
        
        Expected Value = (Probability of Win * Payout) - (Probability of Loss * Cost) - Transaction Cost
        For Kalshi contracts: Payout = $1, Cost = current_price
        """
        payout = 1.0  # Kalshi contracts pay $1 if they hit
        cost = contract.current_price
        
        expected_value = (prob_payout * payout) - ((1 - prob_payout) * cost) - self.transaction_cost
        
        return expected_value
    
    def _generate_recommendation(
        self,
        contract: KalshiContract,
        prob_payout: float,
        expected_value: float,
        edge: float,
        confidence: float
    ) -> Tuple[str, str]:
        """
        Generate trading recommendation and reasoning
        
        Returns:
            Tuple of (recommendation, reasoning)
        """
        # Minimum edge required based on confidence
        min_edge_required = 0.05 if confidence > 0.8 else 0.10
        
        if expected_value > 0.02 and edge > min_edge_required:
            recommendation = "BUY"
            reasoning = (
                f"Strong positive expected value ({expected_value:.3f}) with "
                f"{edge:.1%} edge over market price. "
                f"Predicted probability ({prob_payout:.1%}) significantly exceeds "
                f"market price ({contract.current_price:.1%})."
            )
        elif expected_value > 0 and edge > 0.02:
            recommendation = "BUY"
            reasoning = (
                f"Positive expected value ({expected_value:.3f}) with "
                f"{edge:.1%} edge. Moderate opportunity."
            )
        elif expected_value < -0.02 or edge < -0.05:
            recommendation = "SELL"
            reasoning = (
                f"Negative expected value ({expected_value:.3f}) or "
                f"significant negative edge ({edge:.1%}). Market overpricing contract."
            )
        else:
            recommendation = "HOLD"
            reasoning = (
                f"Expected value ({expected_value:.3f}) and edge ({edge:.1%}) "
                f"too small to justify trading costs."
            )
        
        return recommendation, reasoning
    
    def rank_contracts_by_value(
        self,
        analyses: List[ContractAnalysis]
    ) -> List[ContractAnalysis]:
        """
        Rank contracts by expected value and probability of success
        
        Args:
            analyses: List of contract analyses
            
        Returns:
            Sorted list with best opportunities first
        """
        # Sort by expected value first, then by edge
        return sorted(
            analyses,
            key=lambda x: (x.expected_value, x.edge),
            reverse=True
        )
    
    def get_best_opportunities(
        self,
        analyses: List[ContractAnalysis],
        min_expected_value: float = 0.01,
        min_edge: float = 0.02
    ) -> List[ContractAnalysis]:
        """
        Filter for the best trading opportunities
        
        Args:
            analyses: List of contract analyses
            min_expected_value: Minimum expected value threshold
            min_edge: Minimum edge threshold
            
        Returns:
            Filtered list of best opportunities
        """
        opportunities = [
            analysis for analysis in analyses
            if (analysis.expected_value >= min_expected_value and 
                analysis.edge >= min_edge and
                analysis.recommendation == "BUY")
        ]
        
        return self.rank_contracts_by_value(opportunities)


def create_sample_contracts() -> List[KalshiContract]:
    """
    Create sample Kalshi contracts for testing
    """
    today = date.today()
    
    return [
        KalshiContract(
            contract_id="LA_HIGH_TEMP_ABOVE_85F",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.45,
            expiry_date=today,
            description="LA high temperature above 85°F"
        ),
        KalshiContract(
            contract_id="LA_HIGH_TEMP_ABOVE_90F",
            contract_type="ABOVE", 
            threshold_temp=90.0,
            current_price=0.25,
            expiry_date=today,
            description="LA high temperature above 90°F"
        ),
        KalshiContract(
            contract_id="LA_HIGH_TEMP_BELOW_80F",
            contract_type="BELOW",
            threshold_temp=80.0,
            current_price=0.30,
            expiry_date=today,
            description="LA high temperature below 80°F"
        ),
        KalshiContract(
            contract_id="LA_HIGH_TEMP_ABOVE_95F",
            contract_type="ABOVE",
            threshold_temp=95.0,
            current_price=0.10,
            expiry_date=today,
            description="LA high temperature above 95°F"
        )
    ]


if __name__ == "__main__":
    # Demo usage
    analyzer = KalshiContractAnalyzer()
    contracts = create_sample_contracts()
    
    # Example prediction: 87°F with 2°F standard deviation, 85% confidence
    predicted_temp = 87.0
    prediction_std = 2.0
    confidence = 0.85
    
    analyses = analyzer.analyze_contracts(contracts, predicted_temp, prediction_std, confidence)
    
    print("Kalshi Contract Analysis Results:")
    print("=" * 50)
    
    for analysis in analyses:
        print(f"\nContract: {analysis.contract.description}")
        print(f"Threshold: {analysis.contract.contract_type} {analysis.contract.threshold_temp}°F")
        print(f"Market Price: {analysis.contract.current_price:.1%}")
        print(f"Predicted Probability: {analysis.predicted_probability:.1%}")
        print(f"Expected Value: ${analysis.expected_value:.3f}")
        print(f"Edge: {analysis.edge:.1%}")
        print(f"Recommendation: {analysis.recommendation}")
        print(f"Reasoning: {analysis.reasoning}")
        print("-" * 30)