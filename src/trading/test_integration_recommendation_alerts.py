"""
Integration tests for complete recommendation generation and alert system

This module tests the full workflow from contract analysis to recommendation
generation with alerts, ensuring all requirements are met.
"""

import pytest
from datetime import date, datetime, timedelta

from src.trading.recommendation_engine import (
    RecommendationEngine,
    AlertSystem,
    RecommendationStrength,
    AlertType
)
from src.trading.kalshi_contract_analyzer import (
    KalshiContractAnalyzer,
    create_sample_contracts
)
from src.trading.position_sizing import PositionSizer


class TestRecommendationGenerationIntegration:
    """Test complete recommendation generation workflow"""
    
    def setup_method(self):
        """Set up test components"""
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        self.recommendation_engine = RecommendationEngine()
        self.alert_system = AlertSystem()
        self.contracts = create_sample_contracts()
    
    def test_high_confidence_recommendations(self):
        """Test recommendations with high confidence (>80%) - Requirement 1.5"""
        # High confidence scenario: 89°F ± 1.5°F, 90% confidence
        analyses = self.analyzer.analyze_contracts(self.contracts, 89.0, 1.5, 0.90)
        
        # Generate position recommendations
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.90)
                position_recs.append(rec)
        
        # Generate trading recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.90, 89.0, 1.5
        )
        
        # Verify high confidence requirements
        assert len(recommendations) > 0
        
        for rec in recommendations:
            # Should have strong buy recommendations with high confidence
            assert rec.confidence_score >= 80
            assert rec.recommendation in [
                RecommendationStrength.STRONG_BUY, 
                RecommendationStrength.BUY
            ]
            
            # Should have specific position sizing
            assert rec.position_size_pct > 0
            assert rec.position_size_dollars > 0
            
            # Should have detailed reasoning (Requirement 3.2)
            assert len(rec.market_analysis) > 0
            assert len(rec.prediction_rationale) > 0
            assert len(rec.risk_assessment) > 0
            assert len(rec.timing_considerations) > 0
            
            # Should specify temperature threshold contracts (Requirement 1.4)
            assert "temperature" in rec.contract_description.lower()
            assert any(temp in rec.contract_description for temp in ["85°F", "90°F", "80°F", "95°F"])
    
    def test_moderate_confidence_recommendations(self):
        """Test recommendations with moderate confidence (60-80%) - Requirement 1.6"""
        # Moderate confidence scenario: 85°F ± 2.5°F, 70% confidence
        analyses = self.analyzer.analyze_contracts(self.contracts, 85.0, 2.5, 0.70)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.70)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.70, 85.0, 2.5
        )
        
        # Verify moderate confidence requirements
        for rec in recommendations:
            # Should have moderate buy recommendations
            assert 60 <= rec.confidence_score <= 80 or rec.recommendation in [
                RecommendationStrength.BUY,
                RecommendationStrength.WEAK_BUY
            ]
            
            # Position sizes should be smaller than high confidence
            assert rec.position_size_pct <= 25.0  # Should be conservative
    
    def test_low_confidence_recommendations(self):
        """Test recommendations with low confidence (<60%) - Requirement 1.7"""
        # Low confidence scenario: 82°F ± 4.0°F, 50% confidence
        analyses = self.analyzer.analyze_contracts(self.contracts, 82.0, 4.0, 0.50)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.50)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.50, 82.0, 4.0
        )
        
        # Verify low confidence requirements
        for rec in recommendations:
            # Should have weak recommendations or no trade
            if rec.confidence_score < 60:
                assert rec.recommendation in [
                    RecommendationStrength.WEAK_BUY,
                    RecommendationStrength.HOLD
                ]
                
                # Position sizes should be very small
                assert rec.position_size_pct <= 10.0
    
    def test_clear_buy_sell_recommendations_with_reasoning(self):
        """Test clear buy/sell recommendations with reasoning - Requirement 3.2"""
        analyses = self.analyzer.analyze_contracts(self.contracts, 88.0, 2.0, 0.85)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation in ["BUY", "SELL"]:
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.85)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.85, 88.0, 2.0
        )
        
        for rec in recommendations:
            # Should have clear recommendation
            assert rec.recommendation in [
                RecommendationStrength.STRONG_BUY,
                RecommendationStrength.BUY,
                RecommendationStrength.WEAK_BUY,
                RecommendationStrength.HOLD,
                RecommendationStrength.WEAK_SELL,
                RecommendationStrength.SELL,
                RecommendationStrength.STRONG_SELL
            ]
            
            # Should have comprehensive reasoning
            assert "Market is pricing" in rec.market_analysis
            assert "Model predicts" in rec.prediction_rationale
            assert "position size" in rec.risk_assessment.lower()
            assert "Contract expires" in rec.timing_considerations
            
            # Should have quantitative metrics
            assert isinstance(rec.expected_value, float)
            assert isinstance(rec.edge, float)
            assert isinstance(rec.confidence_score, float)
            assert 0 <= rec.confidence_score <= 100


class TestAlertSystemIntegration:
    """Test complete alert system functionality"""
    
    def setup_method(self):
        """Set up test components"""
        self.alert_system = AlertSystem()
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        self.recommendation_engine = RecommendationEngine()
        self.contracts = create_sample_contracts()
    
    def test_significant_prediction_change_alerts(self):
        """Test alerts for significant prediction changes - Requirement 3.5"""
        # Initial prediction
        initial_temp = 85.0
        initial_confidence = 0.80
        
        # Generate initial recommendations
        analyses = self.analyzer.analyze_contracts(self.contracts, initial_temp, 2.0, initial_confidence)
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, initial_confidence)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, initial_confidence, initial_temp, 2.0
        )
        
        # Initial check (should store prediction)
        alerts = self.alert_system.check_for_alerts(initial_temp, initial_confidence, recommendations)
        
        # Significant change: 7°F increase
        new_temp = 92.0
        new_confidence = 0.85
        
        # Generate new recommendations
        new_analyses = self.analyzer.analyze_contracts(self.contracts, new_temp, 2.0, new_confidence)
        new_position_recs = []
        for analysis in new_analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, new_confidence)
                new_position_recs.append(rec)
        
        new_recommendations = self.recommendation_engine.generate_recommendations(
            new_analyses, new_position_recs, new_confidence, new_temp, 2.0
        )
        
        # Check for alerts after significant change
        change_alerts = self.alert_system.check_for_alerts(new_temp, new_confidence, new_recommendations)
        
        # Should detect prediction change
        prediction_change_alerts = [
            alert for alert in change_alerts 
            if alert.alert_type == AlertType.PREDICTION_CHANGE
        ]
        
        assert len(prediction_change_alerts) > 0
        
        change_alert = prediction_change_alerts[0]
        assert change_alert.old_value == initial_temp
        assert change_alert.new_value == new_temp
        assert change_alert.change_magnitude == 7.0
        assert change_alert.severity in ["HIGH", "MEDIUM"]
        assert "increased by 7.0°F" in change_alert.message
    
    def test_new_opportunity_alerts(self):
        """Test alerts for new high-value opportunities"""
        # High-value opportunity scenario
        analyses = self.analyzer.analyze_contracts(self.contracts, 91.0, 1.5, 0.90)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.90)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.90, 91.0, 1.5
        )
        
        # Check for opportunity alerts
        alerts = self.alert_system.check_for_alerts(91.0, 0.90, recommendations)
        
        opportunity_alerts = [
            alert for alert in alerts 
            if alert.alert_type == AlertType.NEW_OPPORTUNITY
        ]
        
        # Should detect high-value opportunities
        assert len(opportunity_alerts) > 0
        
        for alert in opportunity_alerts:
            assert "opportunity" in alert.message.lower()
            assert alert.severity in ["HIGH", "MEDIUM"]
            assert alert.contract_id is not None
    
    def test_risk_warning_alerts(self):
        """Test risk warning alerts for large positions and low confidence"""
        # Create scenario with large position recommendation
        from src.trading.recommendation_engine import TradingRecommendation
        
        large_position_rec = TradingRecommendation(
            contract_id="LARGE_POS_TEST",
            contract_description="Large position test",
            recommendation=RecommendationStrength.BUY,
            confidence_score=75.0,
            expected_value=0.08,
            edge=0.05,
            position_size_pct=22.0,  # Large position
            position_size_dollars=2200.0,
            market_analysis="Test",
            prediction_rationale="Test",
            risk_assessment="Test",
            timing_considerations="Test",
            predicted_probability=0.60,
            market_price=0.55,
            confidence_interval=(0.55, 0.65)
        )
        
        # Check for risk warnings
        alerts = self.alert_system.check_for_alerts(88.0, 0.75, [large_position_rec])
        
        risk_alerts = [
            alert for alert in alerts 
            if alert.alert_type == AlertType.RISK_WARNING
        ]
        
        # Should detect large position warning
        assert len(risk_alerts) > 0
        
        large_pos_alert = next(
            (alert for alert in risk_alerts if "Large Position" in alert.title), 
            None
        )
        assert large_pos_alert is not None
        assert "22.0%" in large_pos_alert.message
    
    def test_confidence_change_alerts(self):
        """Test alerts for model confidence changes"""
        # Initial confidence
        today = date.today()
        initial_confidence = 0.70
        
        # Set initial prediction
        self.alert_system.previous_predictions[today] = {
            'prediction': 85.0,
            'confidence': initial_confidence,
            'timestamp': datetime.now()
        }
        
        # Significant confidence increase
        new_confidence = 0.90
        
        alerts = self.alert_system.check_for_alerts(85.0, new_confidence, [], today)
        
        confidence_alerts = [
            alert for alert in alerts 
            if alert.alert_type == AlertType.MODEL_CONFIDENCE
        ]
        
        assert len(confidence_alerts) > 0
        
        conf_alert = confidence_alerts[0]
        assert conf_alert.old_value == initial_confidence
        assert conf_alert.new_value == new_confidence
        assert abs(conf_alert.change_magnitude - 0.20) < 0.001
        assert "increased by 20.0%" in conf_alert.message
    
    def test_alert_management_functionality(self):
        """Test alert management features"""
        # Create test alerts
        from src.trading.recommendation_engine import TradingAlert
        
        alert1 = TradingAlert(
            alert_id="TEST_ALERT_1",
            alert_type=AlertType.NEW_OPPORTUNITY,
            title="Test Opportunity",
            message="Test opportunity message",
            severity="HIGH"
        )
        
        alert2 = TradingAlert(
            alert_id="TEST_ALERT_2",
            alert_type=AlertType.RISK_WARNING,
            title="Test Risk Warning",
            message="Test risk warning message",
            severity="MEDIUM"
        )
        
        self.alert_system.active_alerts = [alert1, alert2]
        
        # Test getting active alerts
        active_alerts = self.alert_system.get_active_alerts()
        assert len(active_alerts) == 2
        
        # Test filtering by severity
        high_alerts = self.alert_system.get_active_alerts("HIGH")
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == "HIGH"
        
        # Test acknowledging alerts
        success = self.alert_system.acknowledge_alert("TEST_ALERT_1")
        assert success is True
        assert alert1.acknowledged is True
        
        # Test getting unacknowledged alerts
        unack_alerts = self.alert_system.get_active_alerts()
        assert len(unack_alerts) == 1  # Only unacknowledged
        assert unack_alerts[0].alert_id == "TEST_ALERT_2"


class TestRequirementsCompliance:
    """Test compliance with specific requirements"""
    
    def setup_method(self):
        """Set up test components"""
        self.analyzer = KalshiContractAnalyzer()
        self.position_sizer = PositionSizer()
        self.recommendation_engine = RecommendationEngine()
        self.alert_system = AlertSystem()
        self.contracts = create_sample_contracts()
    
    def test_requirement_1_4_specific_threshold_contracts(self):
        """Test Requirement 1.4: Specify which temperature threshold contracts to buy"""
        analyses = self.analyzer.analyze_contracts(self.contracts, 88.0, 2.0, 0.85)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.85)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.85, 88.0, 2.0
        )
        
        for rec in recommendations:
            # Should specify specific temperature thresholds
            assert any(threshold in rec.contract_description for threshold in [
                "85°F", "90°F", "80°F", "95°F"
            ])
            
            # Should specify above/below
            assert any(direction in rec.contract_description.lower() for direction in [
                "above", "below"
            ])
    
    def test_requirement_1_5_strong_buy_with_position_sizing(self):
        """Test Requirement 1.5: Strong buy recommendations with position sizing (>80% confidence)"""
        # High confidence scenario that should generate strong buy
        analyses = self.analyzer.analyze_contracts(self.contracts, 92.0, 1.0, 0.95)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation == "BUY":
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.95)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.95, 92.0, 1.0
        )
        
        # Should have strong buy recommendations
        strong_buys = [
            rec for rec in recommendations 
            if rec.recommendation == RecommendationStrength.STRONG_BUY
        ]
        
        assert len(strong_buys) > 0
        
        for rec in strong_buys:
            assert rec.confidence_score > 80
            assert rec.position_size_pct > 0
            assert rec.position_size_dollars > 0
            assert "Strong" in rec.recommendation.value or rec.confidence_score >= 85
    
    def test_requirement_3_2_clear_recommendations_with_reasoning(self):
        """Test Requirement 3.2: Clear buy/sell recommendations with reasoning"""
        analyses = self.analyzer.analyze_contracts(self.contracts, 87.0, 2.0, 0.80)
        
        position_recs = []
        for analysis in analyses:
            if analysis.recommendation in ["BUY", "SELL"]:
                rec = self.position_sizer.calculate_position_size(analysis, 10000, 0.80)
                position_recs.append(rec)
        
        recommendations = self.recommendation_engine.generate_recommendations(
            analyses, position_recs, 0.80, 87.0, 2.0
        )
        
        for rec in recommendations:
            # Should have clear recommendation
            assert rec.recommendation in [
                RecommendationStrength.STRONG_BUY,
                RecommendationStrength.BUY,
                RecommendationStrength.WEAK_BUY,
                RecommendationStrength.HOLD,
                RecommendationStrength.WEAK_SELL,
                RecommendationStrength.SELL,
                RecommendationStrength.STRONG_SELL
            ]
            
            # Should include reasoning based on model predictions, API consensus, and historical accuracy
            assert "model predicts" in rec.prediction_rationale.lower()
            assert "market" in rec.market_analysis.lower()
            assert len(rec.market_analysis) > 20  # Substantial reasoning
            assert len(rec.prediction_rationale) > 20
            assert len(rec.risk_assessment) > 20
    
    def test_requirement_3_5_alert_system_for_prediction_changes(self):
        """Test Requirement 3.5: Alert system for significant prediction changes"""
        # Initial prediction
        initial_recommendations = []
        alerts = self.alert_system.check_for_alerts(85.0, 0.80, initial_recommendations)
        
        # Significant change
        alerts = self.alert_system.check_for_alerts(91.0, 0.85, initial_recommendations)  # 6°F change
        
        # Should detect significant change
        prediction_alerts = [
            alert for alert in alerts 
            if alert.alert_type == AlertType.PREDICTION_CHANGE
        ]
        
        assert len(prediction_alerts) > 0
        
        alert = prediction_alerts[0]
        assert alert.old_value == 85.0
        assert alert.new_value == 91.0
        assert alert.change_magnitude == 6.0
        assert "change drivers" in alert.message.lower() or "review" in alert.message.lower()


if __name__ == "__main__":
    # Run integration tests
    print("Running Recommendation Generation and Alert System Integration Tests...")
    
    # Test recommendation generation
    test_rec = TestRecommendationGenerationIntegration()
    test_rec.setup_method()
    
    try:
        test_rec.test_high_confidence_recommendations()
        print("✅ High confidence recommendations test passed")
        
        test_rec.test_moderate_confidence_recommendations()
        print("✅ Moderate confidence recommendations test passed")
        
        test_rec.test_low_confidence_recommendations()
        print("✅ Low confidence recommendations test passed")
        
        test_rec.test_clear_buy_sell_recommendations_with_reasoning()
        print("✅ Clear recommendations with reasoning test passed")
        
    except Exception as e:
        print(f"❌ Recommendation generation test failed: {e}")
    
    # Test alert system
    test_alerts = TestAlertSystemIntegration()
    test_alerts.setup_method()
    
    try:
        test_alerts.test_significant_prediction_change_alerts()
        print("✅ Prediction change alerts test passed")
        
        test_alerts.test_new_opportunity_alerts()
        print("✅ New opportunity alerts test passed")
        
        test_alerts.test_risk_warning_alerts()
        print("✅ Risk warning alerts test passed")
        
        test_alerts.test_confidence_change_alerts()
        print("✅ Confidence change alerts test passed")
        
        test_alerts.test_alert_management_functionality()
        print("✅ Alert management test passed")
        
    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
    
    # Test requirements compliance
    test_req = TestRequirementsCompliance()
    test_req.setup_method()
    
    try:
        test_req.test_requirement_1_4_specific_threshold_contracts()
        print("✅ Requirement 1.4 compliance test passed")
        
        test_req.test_requirement_1_5_strong_buy_with_position_sizing()
        print("✅ Requirement 1.5 compliance test passed")
        
        test_req.test_requirement_3_2_clear_recommendations_with_reasoning()
        print("✅ Requirement 3.2 compliance test passed")
        
        test_req.test_requirement_3_5_alert_system_for_prediction_changes()
        print("✅ Requirement 3.5 compliance test passed")
        
    except Exception as e:
        print(f"❌ Requirements compliance test failed: {e}")
    
    print("\nAll integration tests completed!")