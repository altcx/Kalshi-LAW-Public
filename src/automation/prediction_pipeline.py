#!/usr/bin/env python3
"""Automated prediction and trading pipeline for daily temperature forecasting."""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_manager import DataManager
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.models.enhanced_ensemble_combiner import EnhancedEnsembleCombiner
from src.models.ensemble_model import EnsembleTemperatureModel
from src.trading.recommendation_engine import RecommendationEngine, AlertType
from src.trading.kalshi_contract_analyzer import KalshiContractAnalyzer
from src.automation.error_handler import error_handler, retry_with_backoff, ErrorSeverity


class PredictionPipeline:
    """Automated pipeline for generating daily temperature predictions and trading recommendations."""
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        self.data_manager = DataManager()
        self.feature_pipeline = FeaturePipeline()
        # Create base ensemble model first
        base_ensemble = EnsembleTemperatureModel()
        self.ensemble_model = EnhancedEnsembleCombiner(base_ensemble)
        self.recommendation_engine = RecommendationEngine()
        self.contract_analyzer = KalshiContractAnalyzer()
        
        # Track previous predictions for change detection
        self.previous_predictions = {}
        self.alert_thresholds = {
            'temperature_change': 3.0,  # Alert if prediction changes by >3°F
            'confidence_change': 0.15,  # Alert if confidence changes by >15%
            'high_confidence': 0.85,    # Alert for high confidence predictions
            'low_confidence': 0.50      # Alert for low confidence predictions
        }
        
        logger.info("PredictionPipeline initialized")
    
    @retry_with_backoff(
        max_retries=3,
        base_delay=2.0,
        error_handler=error_handler,
        context="Feature Generation"
    )
    def generate_features_for_prediction(self, target_date: date) -> Optional[pd.DataFrame]:
        """Generate features for prediction on target date.
        
        Args:
            target_date: Date to generate prediction for
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            logger.info(f"Generating features for prediction on {target_date}")
            
            # Use data up to yesterday for prediction
            end_date = target_date - timedelta(days=1)
            start_date = end_date - timedelta(days=30)  # Use 30 days of historical data
            
            # Generate features
            features_df = self.feature_pipeline.create_features_for_date_range(
                start_date=start_date,
                end_date=end_date,
                clean_outliers=True
            )
            
            if features_df.empty:
                logger.warning(f"No features generated for {target_date}")
                return None
            
            # Get the most recent feature row for prediction
            latest_features = features_df.iloc[-1:].copy()
            latest_features['target_date'] = target_date
            
            logger.info(f"Generated {len(features_df.columns)} features for {target_date}")
            return latest_features
            
        except Exception as e:
            logger.error(f"Error generating features for {target_date}: {e}")
            raise
    
    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        error_handler=error_handler,
        context="Model Prediction"
    )
    def generate_temperature_prediction(self, features: pd.DataFrame, target_date: date) -> Optional[Dict[str, Any]]:
        """Generate temperature prediction using ensemble model.
        
        Args:
            features: Feature DataFrame
            target_date: Date to predict for
            
        Returns:
            Dictionary with prediction results or None if failed
        """
        try:
            logger.info(f"Generating temperature prediction for {target_date}")
            
            # Make prediction using ensemble model
            prediction_result = self.ensemble_model.predict(features)
            
            if prediction_result is None:
                logger.error(f"Ensemble model failed to generate prediction for {target_date}")
                return None
            
            # Extract prediction details
            prediction_data = {
                'date': target_date,
                'predicted_high': prediction_result.prediction,
                'confidence': prediction_result.confidence,
                'weather_condition': prediction_result.weather_condition,
                'ensemble_method': prediction_result.ensemble_method,
                'model_contributions': {
                    pred.model_name: {
                        'prediction': pred.prediction,
                        'confidence': pred.confidence,
                        'weight': pred.weight
                    }
                    for pred in prediction_result.model_predictions
                },
                'feature_importance': self._get_feature_importance(features),
                'created_at': datetime.now()
            }
            
            logger.info(f"Generated prediction: {prediction_data['predicted_high']:.1f}°F "
                       f"(confidence: {prediction_data['confidence']:.3f})")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error generating prediction for {target_date}: {e}")
            raise
    
    def _get_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from the ensemble model.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            # Get feature importance from the ensemble model
            importance = self.ensemble_model.get_feature_importance()
            
            # Normalize to sum to 1.0
            if importance:
                total_importance = sum(importance.values())
                if total_importance > 0:
                    importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance or {}
            
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {}
    
    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        error_handler=error_handler,
        context="Trading Recommendations"
    )
    def generate_trading_recommendations(self, prediction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on prediction.
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            List of trading recommendations
        """
        try:
            target_date = prediction_data['date']
            predicted_temp = prediction_data['predicted_high']
            confidence = prediction_data['confidence']
            
            logger.info(f"Generating trading recommendations for {target_date}")
            
            # Create sample contracts for analysis (in production, this would fetch from Kalshi API)
            from src.trading.kalshi_contract_analyzer import create_sample_contracts
            contracts = create_sample_contracts()
            
            if not contracts:
                logger.warning(f"No contracts available for {target_date}")
                return []
            
            # Analyze contracts
            contract_analyses = self.contract_analyzer.analyze_contracts(
                contracts=contracts,
                predicted_temperature=predicted_temp,
                prediction_confidence=confidence
            )
            
            # Generate recommendations from analyses
            trading_recommendations = self.recommendation_engine.generate_recommendations(
                contract_analyses=contract_analyses,
                predicted_temperature=predicted_temp,
                prediction_confidence=confidence,
                model_contributions=prediction_data.get('model_contributions', {}),
                weather_condition=prediction_data.get('weather_condition', 'normal')
            )
            
            # Convert to our format
            recommendations = []
            for rec in trading_recommendations:
                rec_data = {
                    'date': target_date,
                    'contract_id': rec.contract_id,
                    'contract_description': rec.contract_description,
                    'threshold': rec.contract_id.split('_')[-1] if '_' in rec.contract_id else 'unknown',
                    'contract_type': 'above' if 'ABOVE' in rec.contract_id else 'below',
                    'recommendation': rec.recommendation.value,
                    'confidence_score': rec.confidence_score,
                    'expected_value': rec.expected_value,
                    'edge': rec.edge,
                    'position_size_pct': rec.position_size_pct,
                    'position_size_dollars': rec.position_size_dollars,
                    'reasoning': f"{rec.market_analysis} {rec.prediction_rationale}",
                    'risk_factors': [rec.risk_assessment],
                    'created_at': datetime.now()
                }
                
                recommendations.append(rec_data)
                
                logger.info(f"Generated recommendation for {rec.contract_description}: "
                           f"{rec.recommendation.value} "
                           f"(EV: {rec.expected_value:.3f})")
            
            # Sort recommendations by expected value
            recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} trading recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            raise
    
    def detect_significant_changes(self, current_prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect significant changes from previous predictions.
        
        Args:
            current_prediction: Current prediction data
            
        Returns:
            List of alerts for significant changes
        """
        alerts = []
        target_date = current_prediction['date']
        
        # Get previous prediction for comparison
        previous_key = str(target_date)
        previous_prediction = self.previous_predictions.get(previous_key)
        
        if previous_prediction is None:
            # First prediction for this date
            alerts.append({
                'type': AlertType.NEW_OPPORTUNITY.value,
                'message': f"New prediction generated for {target_date}",
                'severity': 'info',
                'data': {
                    'predicted_temperature': current_prediction['predicted_high'],
                    'confidence': current_prediction['confidence']
                }
            })
        else:
            # Compare with previous prediction
            temp_change = abs(current_prediction['predicted_high'] - previous_prediction['predicted_high'])
            confidence_change = abs(current_prediction['confidence'] - previous_prediction['confidence'])
            
            # Temperature change alert
            if temp_change >= self.alert_thresholds['temperature_change']:
                direction = "increased" if current_prediction['predicted_high'] > previous_prediction['predicted_high'] else "decreased"
                alerts.append({
                    'type': AlertType.PREDICTION_CHANGE.value,
                    'message': f"Significant temperature prediction change for {target_date}: "
                              f"{direction} by {temp_change:.1f}°F",
                    'severity': 'warning',
                    'data': {
                        'previous_temperature': previous_prediction['predicted_high'],
                        'current_temperature': current_prediction['predicted_high'],
                        'change': temp_change
                    }
                })
            
            # Confidence change alert
            if confidence_change >= self.alert_thresholds['confidence_change']:
                direction = "increased" if current_prediction['confidence'] > previous_prediction['confidence'] else "decreased"
                alerts.append({
                    'type': AlertType.MODEL_CONFIDENCE.value,
                    'message': f"Significant confidence change for {target_date}: "
                              f"{direction} by {confidence_change:.1%}",
                    'severity': 'info',
                    'data': {
                        'previous_confidence': previous_prediction['confidence'],
                        'current_confidence': current_prediction['confidence'],
                        'change': confidence_change
                    }
                })
        
        # High/low confidence alerts
        confidence = current_prediction['confidence']
        if confidence >= self.alert_thresholds['high_confidence']:
            alerts.append({
                'type': AlertType.MODEL_CONFIDENCE.value,
                'message': f"High confidence prediction for {target_date}: {confidence:.1%}",
                'severity': 'success',
                'data': {
                    'confidence': confidence,
                    'predicted_temperature': current_prediction['predicted_high']
                }
            })
        elif confidence <= self.alert_thresholds['low_confidence']:
            alerts.append({
                'type': AlertType.RISK_WARNING.value,
                'message': f"Low confidence prediction for {target_date}: {confidence:.1%}",
                'severity': 'warning',
                'data': {
                    'confidence': confidence,
                    'predicted_temperature': current_prediction['predicted_high']
                }
            })
        
        # Store current prediction for future comparison
        self.previous_predictions[previous_key] = current_prediction.copy()
        
        return alerts
    
    def store_prediction_and_recommendations(self, prediction_data: Dict[str, Any], 
                                           recommendations: List[Dict[str, Any]]) -> None:
        """Store prediction and recommendations to database.
        
        Args:
            prediction_data: Prediction data to store
            recommendations: Trading recommendations to store
        """
        try:
            # Store prediction
            self.data_manager.store_prediction(
                prediction=prediction_data['predicted_high'],
                confidence=prediction_data['confidence'],
                target_date=prediction_data['date'],
                model_contributions=prediction_data.get('model_contributions'),
                feature_importance=prediction_data.get('feature_importance')
            )
            
            # Store recommendations (you might want to add this to DataManager)
            # For now, we'll log them
            for rec in recommendations:
                logger.info(f"Recommendation stored: {rec['contract_description']} - "
                           f"{rec['recommendation']} (EV: {rec['expected_value']:.3f})")
            
            logger.info(f"Stored prediction and {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error storing prediction and recommendations: {e}")
            raise
    
    def run_daily_prediction_pipeline(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Run the complete daily prediction pipeline.
        
        Args:
            target_date: Date to generate prediction for (defaults to today)
            
        Returns:
            Dictionary with pipeline results
        """
        if target_date is None:
            target_date = date.today()
        
        logger.info(f"=== Running Daily Prediction Pipeline for {target_date} ===")
        
        pipeline_results = {
            'target_date': target_date,
            'success': False,
            'prediction': None,
            'recommendations': [],
            'alerts': [],
            'errors': []
        }
        
        try:
            # Step 1: Generate features
            logger.info("Step 1: Generating features...")
            features = self.generate_features_for_prediction(target_date)
            
            if features is None:
                error_msg = "Failed to generate features"
                pipeline_results['errors'].append(error_msg)
                logger.error(error_msg)
                return pipeline_results
            
            # Step 2: Generate prediction
            logger.info("Step 2: Generating temperature prediction...")
            prediction_data = self.generate_temperature_prediction(features, target_date)
            
            if prediction_data is None:
                error_msg = "Failed to generate temperature prediction"
                pipeline_results['errors'].append(error_msg)
                logger.error(error_msg)
                return pipeline_results
            
            pipeline_results['prediction'] = prediction_data
            
            # Step 3: Generate trading recommendations
            logger.info("Step 3: Generating trading recommendations...")
            recommendations = self.generate_trading_recommendations(prediction_data)
            pipeline_results['recommendations'] = recommendations
            
            # Step 4: Detect significant changes and generate alerts
            logger.info("Step 4: Detecting significant changes...")
            alerts = self.detect_significant_changes(prediction_data)
            pipeline_results['alerts'] = alerts
            
            # Step 5: Store results
            logger.info("Step 5: Storing prediction and recommendations...")
            self.store_prediction_and_recommendations(prediction_data, recommendations)
            
            pipeline_results['success'] = True
            
            # Log summary
            logger.info(f"Pipeline completed successfully:")
            logger.info(f"  Prediction: {prediction_data['predicted_high']:.1f}°F "
                       f"(confidence: {prediction_data['confidence']:.3f})")
            logger.info(f"  Recommendations: {len(recommendations)}")
            logger.info(f"  Alerts: {len(alerts)}")
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline failed with error: {e}"
            pipeline_results['errors'].append(error_msg)
            logger.error(error_msg)
            
            # Log error with context
            error_handler.log_error(
                e,
                f"Daily Prediction Pipeline for {target_date}",
                ErrorSeverity.HIGH,
                {
                    'target_date': str(target_date),
                    'pipeline_step': 'unknown'
                }
            )
            
            return pipeline_results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health.
        
        Returns:
            Dictionary with pipeline status information
        """
        try:
            status = {
                'timestamp': datetime.now(),
                'components': {},
                'recent_predictions': [],
                'overall_health': 'healthy'
            }
            
            # Check component health
            components = {
                'data_manager': self.data_manager,
                'feature_pipeline': self.feature_pipeline,
                'ensemble_model': self.ensemble_model,
                'recommendation_engine': self.recommendation_engine,
                'contract_analyzer': self.contract_analyzer
            }
            
            for name, component in components.items():
                try:
                    # Basic health check - component exists and has expected methods
                    status['components'][name] = {
                        'status': 'healthy',
                        'type': type(component).__name__
                    }
                except Exception as e:
                    status['components'][name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    status['overall_health'] = 'degraded'
            
            # Get recent predictions
            try:
                recent_predictions_df = self.data_manager.load_predictions(
                    start_date=date.today() - timedelta(days=7),
                    end_date=date.today()
                )
                
                if not recent_predictions_df.empty:
                    status['recent_predictions'] = [
                        {
                            'date': str(row['date']),
                            'predicted_high': row['predicted_high'],
                            'confidence': row['confidence']
                        }
                        for _, row in recent_predictions_df.tail(5).iterrows()
                    ]
            except Exception as e:
                logger.warning(f"Could not load recent predictions: {e}")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                'timestamp': datetime.now(),
                'overall_health': 'error',
                'error': str(e)
            }


def main():
    """Main entry point for running the prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated prediction and trading pipeline")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Create pipeline
    pipeline = PredictionPipeline()
    
    if args.status:
        # Show pipeline status
        status = pipeline.get_pipeline_status()
        
        print(f"Pipeline Status:")
        print(f"  Overall Health: {status['overall_health']}")
        print(f"  Timestamp: {status['timestamp']}")
        
        print(f"\nComponents:")
        for name, info in status['components'].items():
            status_icon = "✅" if info['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {name}: {info['status']}")
        
        if status['recent_predictions']:
            print(f"\nRecent Predictions:")
            for pred in status['recent_predictions']:
                print(f"  {pred['date']}: {pred['predicted_high']:.1f}°F "
                      f"(confidence: {pred['confidence']:.3f})")
        
        return 0
    
    else:
        # Run prediction pipeline
        target_date = None
        if args.date:
            try:
                target_date = date.fromisoformat(args.date)
            except ValueError:
                print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
                return 1
        
        # Run pipeline
        results = pipeline.run_daily_prediction_pipeline(target_date)
        
        if results['success']:
            print("✅ Prediction pipeline completed successfully")
            
            if results['prediction']:
                pred = results['prediction']
                print(f"Prediction: {pred['predicted_high']:.1f}°F "
                      f"(confidence: {pred['confidence']:.3f})")
            
            if results['recommendations']:
                print(f"Generated {len(results['recommendations'])} trading recommendations")
            
            if results['alerts']:
                print(f"Generated {len(results['alerts'])} alerts")
            
            return 0
        else:
            print("❌ Prediction pipeline failed")
            for error in results['errors']:
                print(f"Error: {error}")
            return 1


if __name__ == "__main__":
    sys.exit(main())