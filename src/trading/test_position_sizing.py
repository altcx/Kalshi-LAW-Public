"""
Tests for Position Sizing and Risk Management System
"""

import pytest
from datetime import date

from src.trading.position_sizing import (
    RiskLimits,
    PositionSizeRecommendation,
    PortfolioRisk,
    PositionSizer,
    RiskManager
)
from src.trading.kalshi_contract_analyzer import (
    KalshiContract,
    ContractAnalysis,
    create_sample_contracts,
    KalshiContractAnalyzer
)


class TestRiskLimits:
    """Test RiskLimits dataclass"""
    
    def test_default_risk_limits(self):
        """Test default risk limits creation"""
        limits = RiskLimits()
        
        assert limits.max_position_size == 0.25
        assert limits.max_total_exposure == 0.50
        assert limits.max_single_contract_exposure == 0.15
        assert limits.min_bankroll_reserve == 0.20
    
    def test_custom_risk_limits(self):
        """Test custom risk limits"""
        limits = RiskLimits(
            max_position_size=0.10,
            max_total_exposure=0.30,
            max_single_contract_exposure=0.08,
            min_bankroll_reserve=0.25
        )
        
        assert limits.max_position_size == 0.10
        assert limits.max_total_exposure == 0.30
        assert limits.max_single_contract_exposure == 0.08
        assert limits.min_bankroll_reserve == 0.25
    
    def test_invalid_risk_limits_raise_error(self):
        """Test that invalid risk limits raise ValueError"""
        with pytest.raises(ValueError, match="max_position_size must be between 0 and 1"):
            RiskLimits(max_position_size=1.5)
        
        with pytest.raises(ValueError, match="max_total_exposure must be between 0 and 1"):
            RiskLimits(max_total_exposure=0)


class TestPositionSizer:
    """Test PositionSizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.position_sizer = PositionSizer()
        self.analyzer = KalshiContractAnalyzer()
        self.contracts = create_sample_contracts()
        
        # Create a sample analysis
        self.sample_analysis = self.analyzer._analyze_single_contract(
            self.contracts[0], predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization"""
        custom_limits = RiskLimits(max_position_size=0.10)
        sizer = PositionSizer(custom_limits)
        
        assert sizer.risk_limits.max_position_size == 0.10
    
    def test_kelly_fraction_calculation(self):
        """Test Kelly criterion calculation"""
        # Create a contract with known parameters
        contract = KalshiContract(
            contract_id="TEST",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.4,  # 40% market price
            expiry_date=date.today(),
            description="Test contract"
        )
        
        analysis = ContractAnalysis(
            contract=contract,
            predicted_probability=0.7,  # 70% predicted probability
            expected_value=0.2,
            edge=0.3,
            confidence_interval=(0.6, 0.8),
            recommendation="BUY",
            reasoning="Test"
        )
        
        kelly_fraction = self.position_sizer._calculate_kelly_fraction(analysis)
        
        # Kelly = (bp - q) / b
        # b = (1.0 - 0.4) / 0.4 = 1.5
        # p = 0.7, q = 0.3
        # Kelly = (1.5 * 0.7 - 0.3) / 1.5 = (1.05 - 0.3) / 1.5 = 0.5
        expected_kelly = 0.5
        assert abs(kelly_fraction - expected_kelly) < 0.01
    
    def test_kelly_fraction_negative_edge(self):
        """Test Kelly fraction with negative edge returns 0"""
        contract = KalshiContract(
            contract_id="TEST",
            contract_type="ABOVE",
            threshold_temp=85.0,
            current_price=0.8,  # High price, negative edge
            expiry_date=date.today(),
            description="Test contract"
        )
        
        analysis = ContractAnalysis(
            contract=contract,
            predicted_probability=0.6,  # Lower probability than price
            expected_value=-0.1,
            edge=-0.2,
            confidence_interval=(0.5, 0.7),
            recommendation="SELL",
            reasoning="Test"
        )
        
        kelly_fraction = self.position_sizer._calculate_kelly_fraction(analysis)
        assert kelly_fraction == 0.0
    
    def test_confidence_adjustment(self):
        """Test confidence-based adjustments"""
        # High confidence
        adj_high = self.position_sizer._calculate_confidence_adjustment(0.95)
        assert adj_high == 1.0
        
        # Medium confidence
        adj_med = self.position_sizer._calculate_confidence_adjustment(0.75)
        assert adj_med == 0.6
        
        # Low confidence
        adj_low = self.position_sizer._calculate_confidence_adjustment(0.5)
        assert adj_low == 0.2
    
    def test_position_size_calculation(self):
        """Test complete position size calculation"""
        bankroll = 10000
        confidence = 0.85
        
        recommendation = self.position_sizer.calculate_position_size(
            self.sample_analysis, bankroll, confidence
        )
        
        assert isinstance(recommendation, PositionSizeRecommendation)
        assert recommendation.contract_id == self.sample_analysis.contract.contract_id
        assert 0 <= recommendation.adjusted_fraction <= 1.0
        assert 0 <= recommendation.dollar_amount <= bankroll
        assert len(recommendation.reasoning) > 0
    
    def test_position_size_with_existing_positions(self):
        """Test position sizing with existing positions"""
        bankroll = 10000
        confidence = 0.85
        current_positions = {"EXISTING_1": 2000, "EXISTING_2": 1500}
        
        recommendation = self.position_sizer.calculate_position_size(
            self.sample_analysis, bankroll, confidence, current_positions
        )
        
        # Should be smaller due to existing exposure
        recommendation_no_positions = self.position_sizer.calculate_position_size(
            self.sample_analysis, bankroll, confidence
        )
        
        assert recommendation.adjusted_fraction <= recommendation_no_positions.adjusted_fraction
    
    def test_risk_adjustment_high_exposure(self):
        """Test risk adjustment with high existing exposure"""
        bankroll = 10000
        current_positions = {"POS_1": 4000, "POS_2": 1000}  # 50% exposure
        
        risk_adj = self.position_sizer._calculate_risk_adjustment(
            self.sample_analysis, bankroll, current_positions
        )
        
        assert risk_adj < 1.0  # Should reduce position size


class TestRiskManager:
    """Test RiskManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager()
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        custom_limits = RiskLimits(max_total_exposure=0.30)
        manager = RiskManager(custom_limits)
        
        assert manager.risk_limits.max_total_exposure == 0.30
    
    def test_portfolio_risk_assessment_empty(self):
        """Test portfolio risk assessment with no positions"""
        risk = self.risk_manager.assess_portfolio_risk({}, 10000)
        
        assert risk.total_exposure == 0.0
        assert risk.position_count == 0
        assert risk.overall_risk_score >= 0.0
    
    def test_portfolio_risk_assessment_with_positions(self):
        """Test portfolio risk assessment with positions"""
        positions = {"CONTRACT_1": 2000, "CONTRACT_2": 1500, "CONTRACT_3": -1000}
        bankroll = 10000
        
        risk = self.risk_manager.assess_portfolio_risk(positions, bankroll)
        
        expected_exposure = (2000 + 1500 + 1000) / 10000  # 45%
        assert abs(risk.total_exposure - expected_exposure) < 0.01
        assert risk.position_count == 3
        assert risk.concentration_risk > 0  # Largest position is 20%
    
    def test_check_risk_limits_within_limits(self):
        """Test risk limit checking when within limits"""
        recommendations = [
            PositionSizeRecommendation(
                contract_id="TEST_1",
                kelly_fraction=0.2,
                adjusted_fraction=0.1,
                dollar_amount=1000,
                confidence_adjustment=0.8,
                risk_adjustment=1.0,
                reasoning="Test",
                warnings=[]
            )
        ]
        
        current_positions = {"EXISTING": 1000}
        bankroll = 10000
        
        approved, warnings = self.risk_manager.check_risk_limits(
            recommendations, current_positions, bankroll
        )
        
        assert len(approved) == 1
        assert len(warnings) == 0
        assert approved[0].dollar_amount == 1000
    
    def test_check_risk_limits_exceeds_total_exposure(self):
        """Test risk limit checking when total exposure is exceeded"""
        # Create recommendations that would exceed total exposure
        recommendations = [
            PositionSizeRecommendation(
                contract_id="TEST_1",
                kelly_fraction=0.3,
                adjusted_fraction=0.3,
                dollar_amount=3000,
                confidence_adjustment=1.0,
                risk_adjustment=1.0,
                reasoning="Test",
                warnings=[]
            ),
            PositionSizeRecommendation(
                contract_id="TEST_2",
                kelly_fraction=0.3,
                adjusted_fraction=0.3,
                dollar_amount=3000,
                confidence_adjustment=1.0,
                risk_adjustment=1.0,
                reasoning="Test",
                warnings=[]
            )
        ]
        
        current_positions = {"EXISTING": 2000}  # 20% existing exposure
        bankroll = 10000
        
        approved, warnings = self.risk_manager.check_risk_limits(
            recommendations, current_positions, bankroll
        )
        
        assert len(warnings) > 0
        assert "Total exposure" in warnings[0]
        
        # Positions should be scaled down
        total_approved = sum(rec.dollar_amount for rec in approved)
        assert total_approved < 6000  # Less than original 6000
    
    def test_check_risk_limits_individual_position_too_large(self):
        """Test risk limit checking for individual position size"""
        recommendations = [
            PositionSizeRecommendation(
                contract_id="TEST_1",
                kelly_fraction=0.5,
                adjusted_fraction=0.4,  # Exceeds 25% limit
                dollar_amount=4000,
                confidence_adjustment=1.0,
                risk_adjustment=1.0,
                reasoning="Test",
                warnings=[]
            )
        ]
        
        current_positions = {}
        bankroll = 10000
        
        approved, warnings = self.risk_manager.check_risk_limits(
            recommendations, current_positions, bankroll
        )
        
        assert len(warnings) > 0
        assert approved[0].adjusted_fraction == 0.25  # Capped at limit
        assert approved[0].dollar_amount == 2500  # 25% of 10000
    
    def test_generate_risk_report(self):
        """Test risk report generation"""
        portfolio_risk = PortfolioRisk(
            total_exposure=0.3,
            position_count=2,
            correlation_risk=0.1,
            concentration_risk=0.15,
            liquidity_risk=0.05,
            overall_risk_score=0.2
        )
        
        recommendations = [
            PositionSizeRecommendation(
                contract_id="TEST_1",
                kelly_fraction=0.2,
                adjusted_fraction=0.1,
                dollar_amount=1000,
                confidence_adjustment=0.8,
                risk_adjustment=1.0,
                reasoning="Good opportunity",
                warnings=[]
            )
        ]
        
        report = self.risk_manager.generate_risk_report(portfolio_risk, recommendations)
        
        assert "RISK MANAGEMENT REPORT" in report
        assert "Portfolio Risk Score: 20.0%" in report
        assert "Total Exposure: 30.0%" in report
        assert "TEST_1" in report
        assert "LOW" in report  # Risk level


class TestIntegration:
    """Integration tests combining position sizing and risk management"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        
        # Create sample contracts and analyses
        contracts = create_sample_contracts()
        self.analyses = self.analyzer.analyze_contracts(
            contracts, predicted_temp=87.0, prediction_std=2.0, confidence=0.85
        )
    
    def test_full_position_sizing_workflow(self):
        """Test complete workflow from analysis to position sizing"""
        bankroll = 10000
        confidence = 0.85
        current_positions = {"EXISTING": 1000}
        
        # Get buy recommendations
        buy_analyses = [a for a in self.analyses if a.recommendation == "BUY"]
        
        # Calculate position sizes
        recommendations = []
        for analysis in buy_analyses[:2]:  # Top 2
            rec = self.position_sizer.calculate_position_size(
                analysis, bankroll, confidence, current_positions
            )
            recommendations.append(rec)
        
        # Check risk limits
        approved_recs, warnings = self.risk_manager.check_risk_limits(
            recommendations, current_positions, bankroll
        )
        
        # Assess portfolio risk
        portfolio_risk = self.risk_manager.assess_portfolio_risk(
            current_positions, bankroll
        )
        
        # Generate report
        report = self.risk_manager.generate_risk_report(portfolio_risk, approved_recs)
        
        # Verify results
        assert len(approved_recs) > 0
        assert isinstance(report, str)
        assert len(report) > 100  # Substantial report
        
        # All recommendations should be reasonable
        for rec in approved_recs:
            assert 0 <= rec.adjusted_fraction <= 0.25  # Within limits
            assert 0 <= rec.dollar_amount <= bankroll * 0.25
            assert len(rec.reasoning) > 0


if __name__ == "__main__":
    # Run basic functionality test
    print("Running Position Sizing and Risk Management Tests...")
    
    # Test position sizer
    sizer = PositionSizer()
    analyzer = KalshiContractAnalyzer()
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 87.0, 2.0, 0.85)
    
    # Test with first BUY recommendation
    buy_analysis = next(a for a in analyses if a.recommendation == "BUY")
    rec = sizer.calculate_position_size(buy_analysis, 10000, 0.85)
    
    print(f"✓ Position sizing calculated: {rec.adjusted_fraction:.1%} of bankroll")
    
    # Test risk manager
    risk_manager = RiskManager()
    portfolio_risk = risk_manager.assess_portfolio_risk({"TEST": 2000}, 10000)
    
    print(f"✓ Portfolio risk assessed: {portfolio_risk.overall_risk_score:.1%} risk score")
    
    print("All basic tests passed!")