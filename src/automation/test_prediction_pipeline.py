#!/usr/bin/env python3
"""Tests for the prediction pipeline and alert system."""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import date, datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.automation.prediction_pipeline import PredictionPipeline
from src.automation.alert_system import AlertSystem, AlertSeverity, AlertChannel


class TestPredictionPipeline(unittest.TestCase):
    """Test cases for the prediction pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = PredictionPipeline()
        
        # Mock dependencies
        self.pipeline.data_manager = Mock()
        self.pipeline.feature_pipeline = Mock()
        self.pipeline.ensemble_model = Mock()
        self.pipeline.recommendation_engine = Mock()
        self.pipeline.contract_analyzer = Mock()
    
    def test_generate_features_for_prediction_success(self):
        """Test successful feature generation."""
        target_date = date.today()
        
        # Mock feature generation
        mock_features = pd.DataFrame({
            'temperature_mean': [75.0],
            'humidity_mean': [65.0],
            'pressure_mean': [1013.2]
        })
        
        self.pipeline.feature_pipeline.create_features_for_date_range.return_value = mock_features
        
        # Execute feature generation
        result = self.pipeline.generate_features_for_prediction(target_date)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIn('target_date', result.columns)
        self.assertEqual(result['target_date'].iloc[0], target_date)
    
    def test_generate_features_for_prediction_failure(self):
        """Test feature generation failure."""
        target_date = date.today()
        
        # Mock feature generation failure
        self.pipeline.feature_pipeline.create_features_for_date_range.return_value = pd.DataFrame()
        
        # Execute feature generation
        result = self.pipeline.generate_features_for_prediction(target_date)
        
        # Verify failure handling
        self.assertIsNone(result)
    
    def test_generate_temperature_prediction_success(self):
        """Test successful temperature prediction."""
        target_date = date.today()
        features = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        
        # Mock prediction result
        mock_prediction = Mock()
        mock_prediction.prediction = 78.5
        mock_prediction.confidence = 0.85
        mock_prediction.weather_condition = 'normal'
        mock_prediction.ensemble_method = 'weighted_average'
        mock_prediction.model_predictions = []
        
        self.pipeline.ensemble_model.predict_with_confidence.return_value = mock_prediction
        self.pipeline.ensemble_model.get_feature_importance.return_value = {'feature1': 0.6, 'feature2': 0.4}
        
        # Execute prediction
        result = self.pipeline.generate_temperature_prediction(features, target_date)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result['predicted_high'], 78.5)
        self.assertEqual(result['confidence'], 0.85)
        self.assertEqual(result['date'], target_date)
        self.assertIn('model_contributions', result)
        self.assertIn('feature_importance', result)
    
    def test_generate_temperature_prediction_failure(self):
        """Test temperature prediction failure."""
        target_date = date.today()
        features = pd.DataFrame({'feature1': [1.0]})
        
        # Mock prediction failure
        self.pipeline.ensemble_model.predict_with_confidence.return_value = None
        
        # Execute prediction
        result = self.pipeline.generate_temperature_prediction(features, target_date)
        
        # Verify failure handling
        self.assertIsNone(result)
    
    def test_generate_trading_recommendations_success(self):
        """Test successful trading recommendation generation."""
        prediction_data = {
            'date': date.today(),
            'predicted_high': 78.5,
            'confidence': 0.85,
            'weather_condition': 'normal',
            'model_contributions': {}
        }
        
        # Mock contract and recommendation
        mock_contract = Mock()
        mock_contract.contract_id = 'TEST_CONTRACT'
        mock_contract.description = 'LA High Temp Above 75F'
        mock_contract.threshold = 75.0
        mock_contract.contract_type = 'above'
        
        mock_recommendation = Mock()
        mock_recommendation.recommendation.value = 'BUY'
        mock_recommendation.confidence_score = 85.0
        mock_recommendation.expected_value = 0.15
        mock_recommendation.edge = 0.10
        mock_recommendation.position_size_pct = 5.0
        mock_recommendation.position_size_dollars = 100.0
        mock_recommendation.reasoning = 'High confidence prediction above threshold'
        mock_recommendation.risk_factors = []
        
        self.pipeline.contract_analyzer.get_available_contracts.return_value = [mock_contract]
        self.pipeline.recommendation_engine.generate_recommendation.return_value = mock_recommendation
        
        # Execute recommendation generation
        result = self.pipeline.generate_trading_recommendations(prediction_data)
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['contract_id'], 'TEST_CONTRACT')
        self.assertEqual(result[0]['recommendation'], 'BUY')
        self.assertEqual(result[0]['expected_value'], 0.15)
    
    def test_generate_trading_recommendations_no_contracts(self):
        """Test trading recommendation generation with no contracts."""
        prediction_data = {
            'date': date.today(),
            'predicted_high': 78.5,
            'confidence': 0.85
        }
        
        # Mock no contracts available
        self.pipeline.contract_analyzer.get_available_contracts.return_value = []
        
        # Execute recommendation generation
        result = self.pipeline.generate_trading_recommendations(prediction_data)
        
        # Verify empty result
        self.assertEqual(len(result), 0)
    
    def test_detect_significant_changes_first_prediction(self):
        """Test change detection for first prediction."""
        current_prediction = {
            'date': date.today(),
            'predicted_high': 78.5,
            'confidence': 0.85
        }
        
        # Execute change detection
        alerts = self.pipeline.detect_significant_changes(current_prediction)
        
        # Verify new opportunity alert
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'NEW_OPPORTUNITY')
    
    def test_detect_significant_changes_temperature_change(self):
        """Test change detection for significant temperature change."""
        # Set up previous prediction
        previous_prediction = {
            'date': date.today(),
            'predicted_high': 75.0,
            'confidence': 0.80
        }
        
        current_prediction = {
            'date': date.today(),
            'predicted_high': 79.0,  # 4°F increase
            'confidence': 0.85
        }
        
        # Store previous prediction
        self.pipeline.previous_predictions[str(date.today())] = previous_prediction
        
        # Execute change detection
        alerts = self.pipeline.detect_significant_changes(current_prediction)
        
        # Verify temperature change alert
        temp_alerts = [a for a in alerts if a['type'] == 'PREDICTION_CHANGE']
        self.assertTrue(len(temp_alerts) > 0)
    
    def test_run_daily_prediction_pipeline_success(self):
        """Test successful daily prediction pipeline execution."""
        target_date = date.today()
        
        # Mock successful pipeline steps
        mock_features = pd.DataFrame({'feature1': [1.0]})
        self.pipeline.generate_features_for_prediction = Mock(return_value=mock_features)
        
        mock_prediction = {
            'date': target_date,
            'predicted_high': 78.5,
            'confidence': 0.85
        }
        self.pipeline.generate_temperature_prediction = Mock(return_value=mock_prediction)
        
        mock_recommendations = [{'contract_id': 'TEST', 'recommendation': 'BUY'}]
        self.pipeline.generate_trading_recommendations = Mock(return_value=mock_recommendations)
        
        self.pipeline.detect_significant_changes = Mock(return_value=[])
        self.pipeline.store_prediction_and_recommendations = Mock()
        
        # Execute pipeline
        result = self.pipeline.run_daily_prediction_pipeline(target_date)
        
        # Verify success
        self.assertTrue(result['success'])
        self.assertEqual(result['target_date'], target_date)
        self.assertIsNotNone(result['prediction'])
        self.assertEqual(len(result['recommendations']), 1)
    
    def test_run_daily_prediction_pipeline_feature_failure(self):
        """Test pipeline failure at feature generation step."""
        target_date = date.today()
        
        # Mock feature generation failure
        self.pipeline.generate_features_for_prediction = Mock(return_value=None)
        
        # Execute pipeline
        result = self.pipeline.run_daily_prediction_pipeline(target_date)
        
        # Verify failure
        self.assertFalse(result['success'])
        self.assertTrue(len(result['errors']) > 0)
    
    def test_get_pipeline_status(self):
        """Test pipeline status reporting."""
        # Mock recent predictions
        mock_predictions = pd.DataFrame({
            'date': [date.today() - timedelta(days=1)],
            'predicted_high': [78.5],
            'confidence': [0.85]
        })
        self.pipeline.data_manager.load_predictions.return_value = mock_predictions
        
        # Execute status check
        status = self.pipeline.get_pipeline_status()
        
        # Verify status
        self.assertIn('timestamp', status)
        self.assertIn('components', status)
        self.assertIn('overall_health', status)
        self.assertIn('recent_predictions', status)


class TestAlertSystem(unittest.TestCase):
    """Test cases for the alert system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alert_system = AlertSystem()
    
    def test_create_alert(self):
        """Test alert creation."""
        alert = self.alert_system.create_alert(
            alert_type='test_alert',
            title='Test Alert',
            message='This is a test alert',
            data={'test_key': 'test_value'},
            severity=AlertSeverity.INFO
        )
        
        # Verify alert creation
        self.assertIsNotNone(alert)
        self.assertEqual(alert.type, 'test_alert')
        self.assertEqual(alert.title, 'Test Alert')
        self.assertEqual(alert.severity, AlertSeverity.INFO)
        self.assertEqual(alert.data['test_key'], 'test_value')
        
        # Verify alert is stored
        self.assertIn(alert.id, self.alert_system.active_alerts)
        self.assertIn(alert, self.alert_system.alerts_history)
    
    def test_check_high_confidence_opportunity(self):
        """Test high confidence opportunity detection."""
        prediction_data = {
            'date': date.today(),
            'predicted_high': 78.5,
            'confidence': 0.90  # High confidence
        }
        
        recommendations = [
            {
                'contract_description': 'LA High Temp Above 75F',
                'recommendation': 'BUY',
                'expected_value': 0.15  # High expected value
            }
        ]
        
        # Execute opportunity check
        alerts = self.alert_system.check_high_confidence_opportunity(prediction_data, recommendations)
        
        # Verify alert creation
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].type, 'high_confidence_opportunity')
        self.assertEqual(alerts[0].severity, AlertSeverity.SUCCESS)
    
    def test_check_prediction_changes(self):
        """Test prediction change detection."""
        previous_prediction = {
            'date': date.today(),
            'predicted_high': 75.0,
            'confidence': 0.80
        }
        
        current_prediction = {
            'date': date.today(),
            'predicted_high': 79.0,  # 4°F change
            'confidence': 0.85
        }
        
        # Execute change check
        alerts = self.alert_system.check_prediction_changes(current_prediction, previous_prediction)
        
        # Verify temperature change alert
        temp_alerts = [a for a in alerts if 'Temperature' in a.title]
        self.assertTrue(len(temp_alerts) > 0)
        self.assertEqual(temp_alerts[0].severity, AlertSeverity.WARNING)
    
    def test_check_low_confidence_warning(self):
        """Test low confidence warning detection."""
        prediction_data = {
            'date': date.today(),
            'predicted_high': 78.5,
            'confidence': 0.45  # Low confidence
        }
        
        # Execute confidence check
        alerts = self.alert_system.check_low_confidence_warning(prediction_data)
        
        # Verify warning alert
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].type, 'low_confidence_warning')
        self.assertEqual(alerts[0].severity, AlertSeverity.WARNING)
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        # Create test alert
        alert = self.alert_system.create_alert(
            alert_type='test_alert',
            title='Test Alert',
            message='Test message'
        )
        
        # Acknowledge alert
        success = self.alert_system.acknowledge_alert(alert.id)
        
        # Verify acknowledgment
        self.assertTrue(success)
        self.assertTrue(alert.acknowledged)
    
    def test_get_active_alerts(self):
        """Test active alerts retrieval."""
        # Create test alerts with different severities
        self.alert_system.create_alert(
            alert_type='test_info',
            title='Info Alert',
            message='Info message',
            severity=AlertSeverity.INFO
        )
        
        self.alert_system.create_alert(
            alert_type='test_warning',
            title='Warning Alert',
            message='Warning message',
            severity=AlertSeverity.WARNING
        )
        
        # Get all active alerts
        all_alerts = self.alert_system.get_active_alerts()
        self.assertEqual(len(all_alerts), 2)
        
        # Get filtered alerts
        warning_alerts = self.alert_system.get_active_alerts(AlertSeverity.WARNING)
        self.assertEqual(len(warning_alerts), 1)
        self.assertEqual(warning_alerts[0].severity, AlertSeverity.WARNING)
    
    def test_get_alert_summary(self):
        """Test alert summary generation."""
        # Create test alerts
        self.alert_system.create_alert(
            alert_type='test_info',
            title='Info Alert',
            message='Info message',
            severity=AlertSeverity.INFO
        )
        
        self.alert_system.create_alert(
            alert_type='test_warning',
            title='Warning Alert',
            message='Warning message',
            severity=AlertSeverity.WARNING
        )
        
        # Get summary
        summary = self.alert_system.get_alert_summary(hours=24)
        
        # Verify summary
        self.assertEqual(summary['total_alerts'], 2)
        self.assertEqual(summary['active_alerts'], 2)
        self.assertEqual(summary['severity_breakdown']['info'], 1)
        self.assertEqual(summary['severity_breakdown']['warning'], 1)
        self.assertIn('test_info', summary['type_breakdown'])
        self.assertIn('test_warning', summary['type_breakdown'])


def run_prediction_pipeline_tests():
    """Run all prediction pipeline tests."""
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestPredictionPipeline))
    test_suite.addTests(loader.loadTestsFromTestCase(TestAlertSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_prediction_pipeline_tests()
    sys.exit(0 if success else 1)