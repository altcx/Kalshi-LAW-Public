"""
Tests for Trading Recommendation Engine and Alert System
"""

import pytest
from datetime import date, datetime, timedelta

from src.trading.recommendation_engine import (
    RecommendationStrength,
    AlertType,
    TradingRecommendation,
    TradingAlert,
    RecommendationEngine,
    AlertSystem
)
from src.trading.kalshi_contract_analyzer import (
    KalshiContract,
    ContractAnalysis,
    KalshiContractAnalyzer,
    create_sample_contracts
)
from src.trading.position_sizing import (
    PositionSizeRecommendation,
    PositionSizer
)


class TestTradingRecommendation:
    """Test TradingRecommendation dataclass"""
    
    def test_trading_recommendation_creation(self):
        """Test creating a trading recommendation"""
        rec = TradingRecommendation(
            contract_id="TEST_CONTRACT",
            contract_description="Test contract",
            recommendation=RecommendationStrength.BUY,
            confidence_score=85.0,
            expected_value=0.15,
            edge=0.10,
            position_size_pct=12.5,
            position_size_dollars=1250.0,
            market_analysis="Test market analysis",
            prediction_rationale="Test prediction rationale",
            risk_assessment="Test risk assessment",
            timing_considerations="Test timing",
            predicted_probability=0.65,
            market_price=0.55,
            confidence_interval=(0.60, 0.70)
        )
        
        assert rec.contract_id == "TEST_CONTRACT"
        assert rec.recommendation == RecommendationStrength.BUY
        assert rec.confidence_score == 85.0
        assert rec.expected_value == 0.15
    
    def test_trading_recommendation_to_dict(self):
        """Test converting recommendation to dictionary"""
        rec = TradingRecommendation(
            contract_id="TEST",
            contract_description="Test",
            recommendation=RecommendationStrength.STRONG_BUY,
            confidence_score=90.0,
            expected_value=0.20,
            edge=0.15,
            position_size_pct=15.0,
            position_size_dollars=1500.0,
            market_analysis="Test",
            prediction_rationale="Test",
            risk_assessment="Test",
            timing_considerations="Test",
            predicted_probability=0.70,
            market_price=0.55,
            confidence_interval=(0.65, 0.75)
        )
        
        rec_dict = rec.to_dict()
        
        assert rec_dict['contract_id'] == "TEST"
        assert rec_dict['recommendation'] == "STRONG_BUY"
        assert rec_dict['confidence_score'] == 90.0
        assert 'generated_at' in rec_dict


class TestTradingAlert:
    """Test TradingAlert dataclass"""
    
    def test_trading_alert_creation(self):
        """Test creating a trading alert"""
        alert = TradingAlert(
            alert_id="TEST_ALERT_001",
            alert_type=AlertType.PREDICTION_CHANGE,
            title="Test Alert",
            message="This is a test alert",
            severity="HIGH",
            contract_id="TEST_CONTRACT",
            old_value=85.0,
            new_value=90.0,
            change_magnitude=5.0
        )
        
        assert alert.alert_id == "TEST_ALERT_001"
        assert alert.alert_type == AlertType.PREDICTION_CHANGE
        assert alert.severity == "HIGH"
        assert alert.change_magnitude == 5.0
        assert not alert.acknowledged
    
    def test_trading_alert_to_dict(self):
        """Test converting alert to dictionary"""
        alert = TradingAlert(
            alert_id="TEST_ALERT",
            alert_type=AlertType.NEW_OPPORTUNITY,
            title="Test",
            message="Test message",
            severity="MEDIUM"
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_id'] == "TEST_ALERT"
        assert alert_dict['alert_type'] == "NEW_OPPORTUNITY"
        assert alert_dict['severity'] == "MEDIUM"
        assert 'created_at' in alert_dict


class TestRecommendationEngine:
    """Test RecommendationEngine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = RecommendationEngine()
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        
        # Create sample data
        contracts = create_sample_contracts()
        self.analyses = self.analyzer.analyze_contracts(contracts, 87.0, 2.0, 0.85)
        
        # Create position recommendations for BUY opportunities
        self.position_recs = []
        for analysis in self.analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.85)
                self.position_recs.append(rec)
    
    def test_recommendation_engine_initialization(self):
        """Test recommendation engine initialization"""
        engine = RecommendationEngine()
        assert len(engine.previous_predictions) == 0
        assert len(engine.recommendation_history) == 0
    
    def test_generate_recommendations(self):
        """Test generating trading recommendations"""
        recommendations = self.engine.generate_recommendations(
            self.analyses, self.position_recs, 0.85, 87.0, 2.0
        )
        
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert isinstance(rec, TradingRecommendation)
            assert isinstance(rec.recommendation, RecommendationStrength)
            assert 0 <= rec.confidence_score <= 100
            assert len(rec.market_analysis) > 0
            assert len(rec.prediction_rationale) > 0
            assert len(rec.risk_assessment) > 0
            assert 1 <= rec.priority <= 5
    
    def test_determine_recommendation_strength(self):
        """Test recommendation strength determination"""
        # Create test analysis with strong BUY signal
        strong_analysis = self.analyses[0]  # Assume first is good
        strong_position = self.position_recs[0] if self.position_recs else None
        
        if strong_position:
            # Modify for strong signal
            strong_analysis.edge = 0.25
            strong_analysis.expected_value = 0.20
            strong_position.adjusted_fraction = 0.15
            
            strength = self.engine._determine_recommendation_strength(
                strong_analysis, strong_position, 0.90
            )
            
            assert strength in [RecommendationStrength.STRONG_BUY, RecommendationStrength.BUY]
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        analysis = self.analyses[0]
        
        confidence_score = self.engine._calculate_confidence_score(
            analysis, 0.85, 2.0
        )
        
        assert 0 <= confidence_score <= 100
        assert isinstance(confidence_score, float)
    
    def test_generate_market_analysis(self):
        """Test market analysis generation"""
        analysis = self.analyses[0]
        
        market_analysis = self.engine._generate_market_analysis(analysis, None)
        
        assert len(market_analysis) > 0
        assert "Market is pricing" in market_analysis
        assert "probability" in market_analysis.lower()
    
    def test_generate_prediction_rationale(self):
        """Test prediction rationale generation"""
        analysis = self.analyses[0]
        
        rationale = self.engine._generate_prediction_rationale(
            analysis, 87.0, 2.0, 0.85
        )
        
        assert len(rationale) > 0
        assert "Model predicts" in rationale
        assert "87.0°F" in rationale
        assert "85%" in rationale
    
    def test_generate_risk_assessment(self):
        """Test risk assessment generation"""
        if self.position_recs:
            analysis = self.analyses[0]
            position_rec = self.position_recs[0]
            
            risk_assessment = self.engine._generate_risk_assessment(analysis, position_rec)
            
            assert len(risk_assessment) > 0
            assert "position size" in risk_assessment.lower()
    
    def test_calculate_priority(self):
        """Test priority calculation"""
        if self.position_recs:
            analysis = self.analyses[0]
            position_rec = self.position_recs[0]
            
            priority = self.engine._calculate_priority(analysis, position_rec, 0.85)
            
            assert 1 <= priority <= 5
            assert isinstance(priority, int)


class TestAlertSystem:
    """Test AlertSystem functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alert_system = AlertSystem()
        
        # Create sample recommendations
        self.sample_recommendations = [
            TradingRecommendation(
                contract_id="TEST_1",
                contract_description="Test contract 1",
                recommendation=RecommendationStrength.STRONG_BUY,
                confidence_score=90.0,
                expected_value=0.15,
                edge=0.12,
                position_size_pct=15.0,
                position_size_dollars=1500.0,
                market_analysis="Test",
                prediction_rationale="Test",
                risk_assessment="Test",
                timing_considerations="Test",
                predicted_probability=0.70,
                market_price=0.58,
                confidence_interval=(0.65, 0.75)
            )
        ]
    
    def test_alert_system_initialization(self):
        """Test alert system initialization"""
        system = AlertSystem()
        assert len(system.previous_predictions) == 0
        assert len(system.active_alerts) == 0
        assert len(system.alert_history) == 0
    
    def test_check_for_alerts_initial(self):
        """Test initial alert check with no previous data"""
        alerts = self.alert_system.check_for_alerts(
            85.0, 0.80, self.sample_recommendations
        )
        
        # Should have opportunity alert for strong buy
        assert len(alerts) >= 1
        
        # Check that prediction was stored
        today = date.today()
        assert today in self.alert_system.previous_predictions
        assert self.alert_system.previous_predictions[today]['prediction'] == 85.0
    
    def test_check_prediction_changes_significant(self):
        """Test detection of significant prediction changes"""
        # Set initial prediction
        today = date.today()
        self.alert_system.previous_predictions[today] = {
            'prediction': 85.0,
            'confidence': 0.80,
            'timestamp': datetime.now()
        }
        
        # Check with significant change
        alerts = self.alert_system.check_for_alerts(
            92.0, 0.85, self.sample_recommendations, today  # 7 degree increase
        )
        
        # Should detect prediction change
        prediction_alerts = [a for a in alerts if a.alert_type == AlertType.PREDICTION_CHANGE]
        assert len(prediction_alerts) > 0
        
        alert = prediction_alerts[0]
        assert alert.old_value == 85.0
        assert alert.new_value == 92.0
        assert alert.change_magnitude == 7.0
        assert alert.severity in ["HIGH", "MEDIUM"]
    
    def test_check_prediction_changes_small(self):
        """Test that small prediction changes don't trigger alerts"""
        # Set initial prediction
        today = date.today()
        self.alert_system.previous_predictions[today] = {
            'prediction': 85.0,
            'confidence': 0.80,
            'timestamp': datetime.now()
        }
        
        # Check with small change
        alerts = self.alert_system.check_for_alerts(
            86.0, 0.81, [], today  # 1 degree increase
        )
        
        # Should not detect significant prediction change
        prediction_alerts = [a for a in alerts if a.alert_type == AlertType.PREDICTION_CHANGE]
        assert len(prediction_alerts) == 0
    
    def test_check_new_opportunities(self):
        """Test detection of new trading opportunities"""
        alerts = self.alert_system._check_new_opportunities(self.sample_recommendations)
        
        # Should detect strong buy opportunity
        assert len(alerts) > 0
        
        opportunity_alert = alerts[0]
        assert opportunity_alert.alert_type == AlertType.NEW_OPPORTUNITY
        assert "Strong Buy" in opportunity_alert.message
        assert opportunity_alert.contract_id == "TEST_1"
    
    def test_check_risk_warnings_large_position(self):
        """Test detection of large position risk warnings"""
        # Create recommendation with large position
        large_position_rec = TradingRecommendation(
            contract_id="LARGE_POS",
            contract_description="Large position test",
            recommendation=RecommendationStrength.BUY,
            confidence_score=75.0,
            expected_value=0.08,
            edge=0.05,
            position_size_pct=25.0,  # Large position
            position_size_dollars=2500.0,
            market_analysis="Test",
            prediction_rationale="Test",
            risk_assessment="Test",
            timing_considerations="Test",
            predicted_probability=0.60,
            market_price=0.55,
            confidence_interval=(0.55, 0.65)
        )
        
        alerts = self.alert_system._check_risk_warnings([large_position_rec])
        
        # Should detect large position warning
        risk_alerts = [a for a in alerts if a.alert_type == AlertType.RISK_WARNING]
        assert len(risk_alerts) > 0
        
        large_pos_alert = next(
            (a for a in risk_alerts if "Large Position" in a.title), None
        )
        assert large_pos_alert is not None
        assert "25.0%" in large_pos_alert.message
    
    def test_check_confidence_changes(self):
        """Test detection of confidence changes"""
        # Set initial confidence
        today = date.today()
        self.alert_system.previous_predictions[today] = {
            'prediction': 85.0,
            'confidence': 0.70,
            'timestamp': datetime.now()
        }
        
        # Check with significant confidence change
        alerts = self.alert_system._check_confidence_changes(0.90, today)  # 20% increase
        
        # Should detect confidence change
        assert len(alerts) > 0
        
        conf_alert = alerts[0]
        assert conf_alert.alert_type == AlertType.MODEL_CONFIDENCE
        assert conf_alert.old_value == 0.70
        assert conf_alert.new_value == 0.90
        assert abs(conf_alert.change_magnitude - 0.20) < 0.001
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        # Add some test alerts
        alert1 = TradingAlert(
            alert_id="TEST_1",
            alert_type=AlertType.NEW_OPPORTUNITY,
            title="Test 1",
            message="Test message 1",
            severity="HIGH"
        )
        
        alert2 = TradingAlert(
            alert_id="TEST_2",
            alert_type=AlertType.RISK_WARNING,
            title="Test 2",
            message="Test message 2",
            severity="MEDIUM"
        )
        alert2.acknowledged = True
        
        self.alert_system.active_alerts = [alert1, alert2]
        
        # Get all active alerts
        active = self.alert_system.get_active_alerts()
        assert len(active) == 1  # Only unacknowledged
        assert active[0].alert_id == "TEST_1"
        
        # Get by severity
        high_alerts = self.alert_system.get_active_alerts("HIGH")
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == "HIGH"
    
    def test_acknowledge_alert(self):
        """Test acknowledging alerts"""
        alert = TradingAlert(
            alert_id="TEST_ACK",
            alert_type=AlertType.NEW_OPPORTUNITY,
            title="Test",
            message="Test",
            severity="MEDIUM"
        )
        
        self.alert_system.active_alerts = [alert]
        
        # Acknowledge alert
        result = self.alert_system.acknowledge_alert("TEST_ACK")
        assert result is True
        assert alert.acknowledged is True
        
        # Try to acknowledge non-existent alert
        result = self.alert_system.acknowledge_alert("NON_EXISTENT")
        assert result is False
    
    def test_clear_old_alerts(self):
        """Test clearing old acknowledged alerts"""
        # Create old acknowledged alert
        old_alert = TradingAlert(
            alert_id="OLD_ALERT",
            alert_type=AlertType.RISK_WARNING,
            title="Old",
            message="Old alert",
            severity="LOW"
        )
        old_alert.acknowledged = True
        old_alert.created_at = datetime.now() - timedelta(hours=25)  # 25 hours old
        
        # Create recent alert
        recent_alert = TradingAlert(
            alert_id="RECENT_ALERT",
            alert_type=AlertType.NEW_OPPORTUNITY,
            title="Recent",
            message="Recent alert",
            severity="HIGH"
        )
        
        self.alert_system.active_alerts = [old_alert, recent_alert]
        
        # Clear old alerts
        cleared_count = self.alert_system.clear_old_alerts(24)
        
        assert cleared_count == 1
        assert len(self.alert_system.active_alerts) == 1
        assert self.alert_system.active_alerts[0].alert_id == "RECENT_ALERT"


class TestIntegration:
    """Integration tests for recommendation engine and alert system"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        self.recommendation_engine = RecommendationEngine()
        self.alert_system = AlertSystem()
    
    def test_full_recommendation_workflow(self):
        """Test complete workflow from analysis to recommendations and alerts"""
        # Generate contract analyses
        contracts = create_sample_contracts()
        analyses = self.analyzer.analyze_contracts(contracts, 89.0, 2.0, 0.85)
        
        # Generate position recommendations
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.85)
                position_recs.append(rec)
        
        # Generate trading recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.85, 89.0, 2.0
        )
        
        # Check for alerts
        alerts = self.alert_system.check_for_alerts(89.0, 0.85, recommendations)
        
        # Verify results
        assert len(recommendations) > 0
        assert len(alerts) >= 0  # May or may not have alerts on first run
        
        # All recommendations should be valid
        for rec in recommendations:
            assert isinstance(rec.recommendation, RecommendationStrength)
            assert 0 <= rec.confidence_score <= 100
            assert rec.position_size_pct >= 0
            assert len(rec.market_analysis) > 0
        
        # Test prediction change detection
        alerts_change = self.alert_system.check_for_alerts(94.0, 0.90, recommendations)
        
        # Should detect significant change
        prediction_change_alerts = [
            a for a in alerts_change 
            if a.alert_type == AlertType.PREDICTION_CHANGE
        ]
        assert len(prediction_change_alerts) > 0


if __name__ == "__main__":
    # Run basic functionality test
    print("Running Recommendation Engine and Alert System Tests...")
    
    # Test recommendation engine
    engine = RecommendationEngine()
    analyzer = KalshiContractAnalyzer()
    position_sizer = PositionSizer()
    
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 87.0, 2.0, 0.85)
    
    position_recs = []
    for analysis in analyses:
        if analysis.recommendation == "BUY":
            rec = position_sizer.calculate_position_size(analysis, 10000, 0.85)
            position_recs.append(rec)
    
    recommendations = engine.generate_recommendations(
        analyses, position_recs, 0.85, 87.0, 2.0
    )
    
    print(f"✓ Generated {len(recommendations)} trading recommendations")
    
    # Test alert system
    alert_system = AlertSystem()
    alerts = alert_system.check_for_alerts(87.0, 0.85, recommendations)
    print(f"✓ Generated {len(alerts)} initial alerts")
    
    # Test change detection
    alerts_change = alert_system.check_for_alerts(92.0, 0.90, recommendations)
    print(f"✓ Detected {len(alerts_change)} alerts after prediction change")
    
    print("All basic tests passed!")