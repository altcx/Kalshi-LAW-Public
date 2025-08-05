"""
Tests for Kalshi Contract Analyzer
"""

import pytest
from datetime import date
import numpy as np

from src.trading.kalshi_contract_analyzer import (
    KalshiContract,
    ContractAnalysis,
    KalshiContractAnalyzer,
    create_sample_contracts
)


class TestKalshiContract:
    """Test KalshiContract dataclass"""
    
    def test_valid_contract_creation(self):
        """Test creating a valid contract"""
        contract = KalshiContract(
            contract_id="TEST_ABOVE_85",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.45,
            expiry_date=date.today(),
            description="Test contract"
        )
        
        assert contract.contract_id == "TEST_ABOVE_85"
        assert contract.contract_type == "ABOVE"
        assert contract.threshold_temp == 85.0
        assert contract.current_price == 0.45
    
    def test_invalid_price_raises_error(self):
        """Test that invalid prices raise ValueError"""
        with pytest.raises(ValueError, match="Contract price must be between 0.0 and 1.0"):
            KalshiContract(
                contract_id="TEST",
                contract_type="ABOVE",
                threshold_temp=85.0,
                current_price=1.5,  # Invalid price > 1.0
                expiry_date=date.today(),
                description="Test"
            )
    
    def test_invalid_contract_type_raises_error(self):
        """Test that invalid contract types raise ValueError"""
        with pytest.raises(ValueError, match="Contract type must be 'ABOVE' or 'BELOW'"):
            KalshiContract(
                contract_id="TEST",
                contract_type="INVALID",  # Invalid type
                threshold_temp=85.0,
                current_price=0.45,
                expiry_date=date.today(),
                description="Test"
            )


class TestKalshiContractAnalyzer:
    """Test KalshiContractAnalyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = KalshiContractAnalyzer(transaction_cost=0.01)
        self.contracts = create_sample_contracts()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = KalshiContractAnalyzer(transaction_cost=0.02)
        assert analyzer.transaction_cost == 0.02
    
    def test_payout_probability_calculation_above(self):
        """Test probability calculation for ABOVE contracts"""
        contract = KalshiContract(
            contract_id="TEST_ABOVE_85",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.5,
            expiry_date=date.today(),
            description="Test above 85F"
        )
        
        # If predicted temp is 87°F with std=2°F, probability should be high
        prob = self.analyzer._calculate_payout_probability(contract, 87.0, 2.0)
        assert prob > 0.8  # Should be high probability
        
        # If predicted temp is 83°F with std=2°F, probability should be low
        prob = self.analyzer._calculate_payout_probability(contract, 83.0, 2.0)
        assert prob < 0.2  # Should be low probability
    
    def test_payout_probability_calculation_below(self):
        """Test probability calculation for BELOW contracts"""
        contract = KalshiContract(
            contract_id="TEST_BELOW_80",
            contract_type="BELOW",
            threshold_temp=80.0,
            current_price=0.3,
            expiry_date=date.today(),
            description="Test below 80F"
        )
        
        # If predicted temp is 78°F with std=2°F, probability should be high
        prob = self.analyzer._calculate_payout_probability(contract, 78.0, 2.0)
        assert prob > 0.8  # Should be high probability
        
        # If predicted temp is 82°F with std=2°F, probability should be low
        prob = self.analyzer._calculate_payout_probability(contract, 82.0, 2.0)
        assert prob < 0.2  # Should be low probability
    
    def test_expected_value_calculation(self):
        """Test expected value calculation"""
        # Contract with 70% probability of payout, priced at 50%
        prob_payout = 0.7
        contract = KalshiContract(
            contract_id="TEST",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.5,
            expiry_date=date.today(),
            description="Test"
        )
        
        expected_value = self.analyzer._calculate_expected_value(contract, prob_payout)
        
        # EV = (0.7 * 1.0) - (0.3 * 0.5) - 0.01 = 0.7 - 0.15 - 0.01 = 0.54
        expected_ev = (0.7 * 1.0) - (0.3 * 0.5) - 0.01
        assert abs(expected_value - expected_ev) < 0.001
    
    def test_analyze_single_contract(self):
        """Test analysis of a single contract"""
        contract = self.contracts[0]  # ABOVE 85F contract
        
        analysis = self.analyzer._analyze_single_contract(
            contract, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
        
        assert isinstance(analysis, ContractAnalysis)
        assert analysis.contract == contract
        assert 0.0 <= analysis.predicted_probability <= 1.0
        assert analysis.recommendation in ["BUY", "SELL", "HOLD"]
        assert len(analysis.reasoning) > 0
        assert len(analysis.confidence_interval) == 2
    
    def test_analyze_contracts_returns_sorted_list(self):
        """Test that analyze_contracts returns results sorted by expected value"""
        analyses = self.analyzer.analyze_contracts(
            self.contracts, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
        
        assert len(analyses) == len(self.contracts)
        
        # Check that results are sorted by expected value (descending)
        for i in range(len(analyses) - 1):
            assert analyses[i].expected_value >= analyses[i + 1].expected_value
    
    def test_recommendation_generation_buy(self):
        """Test BUY recommendation generation"""
        contract = self.contracts[0]
        
        # High probability, good edge should generate BUY
        recommendation, reasoning = self.analyzer._generate_recommendation(
            contract, prob_payout=0.8, expected_value=0.25, edge=0.35, confidence=0.9
        )
        
        assert recommendation == "BUY"
        assert "positive expected value" in reasoning.lower()
    
    def test_recommendation_generation_sell(self):
        """Test SELL recommendation generation"""
        contract = self.contracts[0]
        
        # Low probability, negative edge should generate SELL
        recommendation, reasoning = self.analyzer._generate_recommendation(
            contract, prob_payout=0.2, expected_value=-0.15, edge=-0.25, confidence=0.9
        )
        
        assert recommendation == "SELL"
        assert "negative expected value" in reasoning.lower()
    
    def test_recommendation_generation_hold(self):
        """Test HOLD recommendation generation"""
        contract = self.contracts[0]
        
        # Small edge should generate HOLD
        recommendation, reasoning = self.analyzer._generate_recommendation(
            contract, prob_payout=0.46, expected_value=0.005, edge=0.01, confidence=0.7
        )
        
        assert recommendation == "HOLD"
        assert "too small" in reasoning.lower()
    
    def test_rank_contracts_by_value(self):
        """Test contract ranking functionality"""
        # Create analyses with different expected values
        analyses = self.analyzer.analyze_contracts(
            self.contracts, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
        
        ranked = self.analyzer.rank_contracts_by_value(analyses)
        
        # Should be sorted by expected value descending
        for i in range(len(ranked) - 1):
            assert ranked[i].expected_value >= ranked[i + 1].expected_value
    
    def test_get_best_opportunities(self):
        """Test filtering for best opportunities"""
        analyses = self.analyzer.analyze_contracts(
            self.contracts, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
        
        opportunities = self.analyzer.get_best_opportunities(
            analyses, min_expected_value=0.01, min_edge=0.02
        )
        
        # All opportunities should meet minimum criteria
        for opp in opportunities:
            assert opp.expected_value >= 0.01
            assert opp.edge >= 0.02
            assert opp.recommendation == "BUY"
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation"""
        contract = self.contracts[0]  # ABOVE 85F
        
        conf_interval = self.analyzer._calculate_confidence_interval(
            contract, predicted_temp=87.0, prediction_std=2.0
        )
        
        assert len(conf_interval) == 2
        assert 0.0 <= conf_interval[0] <= conf_interval[1] <= 1.0


class TestSampleContracts:
    """Test sample contract creation"""
    
    def test_create_sample_contracts(self):
        """Test that sample contracts are created correctly"""
        contracts = create_sample_contracts()
        
        assert len(contracts) == 4
        
        # Check that all contracts are valid
        for contract in contracts:
            assert isinstance(contract, KalshiContract)
            assert contract.contract_type in ["ABOVE", "BELOW"]
            assert 0.0 <= contract.current_price <= 1.0
            assert contract.threshold_temp > 0
            assert len(contract.description) > 0


if __name__ == "__main__":
    # Run a simple test
    analyzer = KalshiContractAnalyzer()
    contracts = create_sample_contracts()
    
    print("Running basic functionality test...")
    
    analyses = analyzer.analyze_contracts(
        contracts, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
    )
    
    print(f"Analyzed {len(analyses)} contracts successfully")
    
    best_opportunities = analyzer.get_best_opportunities(analyses)
    print(f"Found {len(best_opportunities)} good opportunities")
    
    print("All tests passed!")