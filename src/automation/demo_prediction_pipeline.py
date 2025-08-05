#!/usr/bin/env python3
"""Demo script for testing the prediction pipeline and alert system."""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.automation.prediction_pipeline import PredictionPipeline
from src.automation.alert_system import alert_system, AlertSeverity


def setup_demo_logging():
    """Setup logging for demo."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def demo_prediction_pipeline():
    """Demonstrate the prediction pipeline."""
    print("\n" + "="*60)
    print("DEMO: Prediction Pipeline")
    print("="*60)
    
    pipeline = PredictionPipeline()
    
    print("Running daily prediction pipeline for today...")
    
    # Run prediction pipeline
    results = pipeline.run_daily_prediction_pipeline()
    
    if results['success']:
        print("✅ Prediction pipeline completed successfully!")
        
        prediction = results['prediction']
        print(f"\nPrediction Results:")
        print(f"  Date: {prediction['date']}")
        print(f"  Predicted High: {prediction['predicted_high']:.1f}°F")
        print(f"  Confidence: {prediction['confidence']:.1%}")
        print(f"  Weather Condition: {prediction['weather_condition']}")
        print(f"  Ensemble Method: {prediction['ensemble_method']}")
        
        # Show model contributions
        if prediction.get('model_contributions'):
            print(f"\nModel Contributions:")
            for model, contrib in prediction['model_contributions'].items():
                print(f"  {model}: {contrib['prediction']:.1f}°F "
                      f"(weight: {contrib['weight']:.3f}, confidence: {contrib['confidence']:.3f})")
        
        # Show recommendations
        recommendations = results['recommendations']
        if recommendations:
            print(f"\nTrading Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {i}. {rec['contract_description']}")
                print(f"     Recommendation: {rec['recommendation']}")
                print(f"     Expected Value: {rec['expected_value']:.3f}")
                print(f"     Position Size: {rec['position_size_pct']:.1f}% (${rec['position_size_dollars']:.0f})")
                print(f"     Reasoning: {rec['reasoning']}")
        else:
            print("\nNo trading recommendations generated")
        
        # Show alerts
        alerts = results['alerts']
        if alerts:
            print(f"\nGenerated Alerts ({len(alerts)}):")
            for alert in alerts:
                print(f"  - {alert.get('type', 'Unknown')}: {alert.get('message', 'No message')}")
        else:
            print("\nNo alerts generated")
        
        return True
    else:
        print("❌ Prediction pipeline failed")
        for error in results['errors']:
            print(f"Error: {error}")
        return False


def demo_alert_system():
    """Demonstrate the alert system."""
    print("\n" + "="*60)
    print("DEMO: Alert System")
    print("="*60)
    
    print("Creating sample alerts...")
    
    # Create high confidence opportunity alert
    prediction_data = {
        'date': date.today(),
        'predicted_high': 85.5,
        'confidence': 0.92
    }
    
    recommendations = [
        {
            'contract_description': 'LA High Temp Above 80F',
            'recommendation': 'BUY',
            'expected_value': 0.18,
            'position_size_pct': 8.0,
            'position_size_dollars': 200.0
        }
    ]
    
    high_conf_alerts = alert_system.check_high_confidence_opportunity(prediction_data, recommendations)
    
    # Create prediction change alert
    previous_prediction = {
        'date': date.today(),
        'predicted_high': 78.0,
        'confidence': 0.75
    }
    
    change_alerts = alert_system.check_prediction_changes(prediction_data, previous_prediction)
    
    # Create low confidence warning
    low_conf_prediction = {
        'date': date.today() + timedelta(days=1),
        'predicted_high': 82.0,
        'confidence': 0.45
    }
    
    low_conf_alerts = alert_system.check_low_confidence_warning(low_conf_prediction)
    
    # Show active alerts
    active_alerts = alert_system.get_active_alerts()
    
    print(f"\nActive Alerts ({len(active_alerts)}):")
    for alert in active_alerts:
        severity_icon = {
            AlertSeverity.CRITICAL: '🚨',
            AlertSeverity.ERROR: '❌',
            AlertSeverity.WARNING: '⚠️',
            AlertSeverity.SUCCESS: '✅',
            AlertSeverity.INFO: 'ℹ️'
        }.get(alert.severity, '📢')
        
        print(f"  {severity_icon} [{alert.type.upper()}] {alert.title}")
        print(f"     {alert.message}")
        print(f"     Severity: {alert.severity.value.title()}")
        print(f"     Created: {alert.timestamp.strftime('%H:%M:%S')}")
        
        if alert.data:
            print(f"     Data: {alert.data}")
        print()
    
    # Show alert summary
    summary = alert_system.get_alert_summary(hours=1)
    
    print(f"Alert Summary (last hour):")
    print(f"  Total Alerts: {summary['total_alerts']}")
    print(f"  Active Alerts: {summary['active_alerts']}")
    print(f"  By Severity:")
    for severity, count in summary['severity_breakdown'].items():
        if count > 0:
            print(f"    {severity.title()}: {count}")
    
    return True


def demo_pipeline_status():
    """Demonstrate pipeline status monitoring."""
    print("\n" + "="*60)
    print("DEMO: Pipeline Status Monitoring")
    print("="*60)
    
    pipeline = PredictionPipeline()
    
    print("Getting pipeline status...")
    status = pipeline.get_pipeline_status()
    
    print(f"\nPipeline Status:")
    print(f"  Overall Health: {status['overall_health']}")
    print(f"  Timestamp: {status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nComponents:")
    for name, info in status['components'].items():
        status_icon = "✅" if info['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {name}: {info['status']}")
        if 'error' in info:
            print(f"     Error: {info['error']}")
    
    if status['recent_predictions']:
        print(f"\nRecent Predictions:")
        for pred in status['recent_predictions']:
            print(f"  {pred['date']}: {pred['predicted_high']:.1f}°F "
                  f"(confidence: {pred['confidence']:.3f})")
    else:
        print(f"\nNo recent predictions found")
    
    return True


def demo_feature_generation():
    """Demonstrate feature generation for prediction."""
    print("\n" + "="*60)
    print("DEMO: Feature Generation")
    print("="*60)
    
    pipeline = PredictionPipeline()
    target_date = date.today()
    
    print(f"Generating features for prediction on {target_date}...")
    
    try:
        features = pipeline.generate_features_for_prediction(target_date)
        
        if features is not None:
            print("✅ Feature generation successful!")
            print(f"  Features shape: {features.shape}")
            print(f"  Feature columns: {len(features.columns)}")
            
            # Show sample features
            print(f"\nSample Features:")
            for col in list(features.columns)[:10]:  # Show first 10 features
                value = features[col].iloc[0]
                if isinstance(value, (int, float)):
                    print(f"  {col}: {value:.3f}")
                else:
                    print(f"  {col}: {value}")
            
            if len(features.columns) > 10:
                print(f"  ... and {len(features.columns) - 10} more features")
            
            return True
        else:
            print("❌ Feature generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Feature generation error: {e}")
        return False


def demo_trading_recommendations():
    """Demonstrate trading recommendation generation."""
    print("\n" + "="*60)
    print("DEMO: Trading Recommendations")
    print("="*60)
    
    pipeline = PredictionPipeline()
    
    # Sample prediction data
    prediction_data = {
        'date': date.today(),
        'predicted_high': 83.5,
        'confidence': 0.78,
        'weather_condition': 'normal',
        'model_contributions': {
            'xgboost': {'prediction': 84.0, 'confidence': 0.80, 'weight': 0.4},
            'lightgbm': {'prediction': 83.0, 'confidence': 0.75, 'weight': 0.3},
            'prophet': {'prediction': 83.5, 'confidence': 0.70, 'weight': 0.3}
        }
    }
    
    print(f"Generating trading recommendations for prediction:")
    print(f"  Temperature: {prediction_data['predicted_high']:.1f}°F")
    print(f"  Confidence: {prediction_data['confidence']:.1%}")
    
    try:
        recommendations = pipeline.generate_trading_recommendations(prediction_data)
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} trading recommendations!")
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['contract_description']}")
                print(f"   Threshold: {rec['threshold']}°F ({rec['contract_type']})")
                print(f"   Recommendation: {rec['recommendation']}")
                print(f"   Confidence Score: {rec['confidence_score']:.1f}%")
                print(f"   Expected Value: {rec['expected_value']:.3f}")
                print(f"   Edge: {rec['edge']:.3f}")
                print(f"   Position Size: {rec['position_size_pct']:.1f}% (${rec['position_size_dollars']:.0f})")
                print(f"   Reasoning: {rec['reasoning']}")
                
                if rec.get('risk_factors'):
                    print(f"   Risk Factors: {', '.join(rec['risk_factors'])}")
            
            return True
        else:
            print("⚠️  No trading recommendations generated")
            print("This could be due to:")
            print("  - No contracts available for the target date")
            print("  - No profitable opportunities identified")
            print("  - Contract analyzer or recommendation engine issues")
            return False
            
    except Exception as e:
        print(f"❌ Trading recommendation error: {e}")
        return False


def run_comprehensive_demo():
    """Run comprehensive demo of all prediction pipeline features."""
    print("🚀 Prediction Pipeline and Alert System Demo")
    print("=" * 80)
    
    setup_demo_logging()
    
    demos = [
        ("Pipeline Status", demo_pipeline_status),
        ("Feature Generation", demo_feature_generation),
        ("Trading Recommendations", demo_trading_recommendations),
        ("Alert System", demo_alert_system),
        ("Full Prediction Pipeline", demo_prediction_pipeline),
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🔄 Running {demo_name} demo...")
            success = demo_func()
            results[demo_name] = success
            
            if success:
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"❌ {demo_name} demo failed")
                
        except Exception as e:
            print(f"💥 {demo_name} demo crashed: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    successful_demos = sum(1 for success in results.values() if success)
    total_demos = len(results)
    
    print(f"Completed: {successful_demos}/{total_demos} demos successful")
    
    for demo_name, success in results.items():
        status_icon = "✅" if success else "❌"
        print(f"  {status_icon} {demo_name}")
    
    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("The prediction pipeline and alert system are ready for production use.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demos failed.")
        print("Please review the errors above and fix any issues.")
    
    return successful_demos == total_demos


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)