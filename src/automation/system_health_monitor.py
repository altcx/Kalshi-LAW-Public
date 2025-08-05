#!/usr/bin/env python3
"""Comprehensive system health monitoring for the Kalshi Weather Predictor."""

import sys
import psutil
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_manager import DataManager
from src.data_collection.client_factory import WeatherDataCollector
from src.data_collection.actual_temperature_collector import ActualTemperatureCollector
from src.automation.alert_system import alert_system, AlertSeverity
from src.automation.error_handler import error_handler


class SystemHealthMonitor:
    """Comprehensive system health monitoring and alerting."""
    
    def __init__(self):
        """Initialize the system health monitor."""
        self.data_manager = DataManager()
        self.weather_collector = WeatherDataCollector()
        self.actual_temp_collector = ActualTemperatureCollector()
        
        # Health thresholds
        self.thresholds = {
            'api_success_rate': 0.8,        # 80% minimum success rate
            'data_quality_score': 0.7,      # 70% minimum quality score
            'collection_rate': 0.8,         # 80% minimum collection rate
            'disk_usage': 0.9,              # 90% maximum disk usage
            'memory_usage': 0.85,           # 85% maximum memory usage
            'cpu_usage': 0.8,               # 80% maximum CPU usage
            'prediction_accuracy': 0.7,     # 70% minimum prediction accuracy
            'model_confidence': 0.6,        # 60% minimum average confidence
            'error_rate': 0.1               # 10% maximum error rate
        }
        
        logger.info("SystemHealthMonitor initialized")
    
    def get_comprehensive_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status.
        
        Returns:
            Dictionary with detailed health information
        """
        logger.info("Performing comprehensive system health check...")
        
        health_status = {
            'timestamp': datetime.now(),
            'overall_health': 'healthy',
            'components': {},
            'metrics': {},
            'alerts': [],
            'recommendations': []
        }
        
        try:
            # Check API health
            api_health = self._check_api_health()
            health_status['components']['apis'] = api_health
            
            # Check data collection health
            collection_health = self._check_data_collection_health()
            health_status['components']['data_collection'] = collection_health
            
            # Check data quality
            quality_health = self._check_data_quality_health()
            health_status['components']['data_quality'] = quality_health
            
            # Check system resources
            resource_health = self._check_system_resources()
            health_status['components']['system_resources'] = resource_health
            
            # Check prediction system health
            prediction_health = self._check_prediction_health()
            health_status['components']['prediction_system'] = prediction_health
            
            # Check storage health
            storage_health = self._check_storage_health()
            health_status['components']['storage'] = storage_health
            
            # Calculate overall health
            health_status['overall_health'] = self._calculate_overall_health(health_status['components'])
            
            # Generate health metrics
            health_status['metrics'] = self._generate_health_metrics(health_status['components'])
            
            # Generate alerts and recommendations
            health_status['alerts'] = self._generate_health_alerts(health_status)
            health_status['recommendations'] = self._generate_recommendations(health_status)
            
            logger.info(f"Health check completed: {health_status['overall_health']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status['overall_health'] = 'error'
            health_status['error'] = str(e)
            return health_status
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health and connectivity."""
        logger.info("Checking API health...")
        
        try:
            # Test API connections
            connection_results = self.weather_collector.test_all_connections()
            
            # Calculate success rate
            total_apis = len(connection_results)
            successful_apis = sum(1 for result in connection_results.values() if result)
            success_rate = successful_apis / total_apis if total_apis > 0 else 0
            
            # Determine health status
            if success_rate >= self.thresholds['api_success_rate']:
                status = 'healthy'
            elif success_rate >= 0.5:
                status = 'degraded'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'success_rate': success_rate,
                'total_apis': total_apis,
                'successful_apis': successful_apis,
                'failed_apis': total_apis - successful_apis,
                'api_results': connection_results,
                'threshold': self.thresholds['api_success_rate']
            }
            
        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_data_collection_health(self) -> Dict[str, Any]:
        """Check data collection health and completeness."""
        logger.info("Checking data collection health...")
        
        try:
            # Check temperature collection status
            temp_status = self.actual_temp_collector.get_temperature_collection_status(days=7)
            
            if 'error' in temp_status:
                return {
                    'status': 'error',
                    'error': temp_status['error']
                }
            
            collection_rate = temp_status.get('collection_rate', 0)
            
            # Determine health status
            if collection_rate >= self.thresholds['collection_rate']:
                status = 'healthy'
            elif collection_rate >= 0.6:
                status = 'degraded'
            else:
                status = 'critical'
            
            # Check forecast data collection
            forecast_health = self._check_forecast_collection()
            
            return {
                'status': status,
                'temperature_collection': {
                    'rate': collection_rate,
                    'collected': temp_status.get('total_collected', 0),
                    'expected': temp_status.get('total_expected', 0),
                    'missing_dates': temp_status.get('missing_dates', [])
                },
                'forecast_collection': forecast_health,
                'threshold': self.thresholds['collection_rate']
            }
            
        except Exception as e:
            logger.error(f"Error checking data collection health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_forecast_collection(self) -> Dict[str, Any]:
        """Check forecast data collection health."""
        try:
            # Get data summary for all sources
            data_summary = self.data_manager.get_data_summary()
            
            sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
            collection_status = {}
            
            for source in sources:
                if source in data_summary:
                    source_info = data_summary[source]
                    if 'error' in source_info:
                        collection_status[source] = {
                            'status': 'error',
                            'error': source_info['error']
                        }
                    elif source_info.get('status') == 'file_not_found':
                        collection_status[source] = {
                            'status': 'missing',
                            'records': 0
                        }
                    else:
                        # Check if data is recent (within last 2 days)
                        date_range = source_info.get('date_range')
                        if date_range:
                            end_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()
                            days_old = (date.today() - end_date).days
                            
                            if days_old <= 2:
                                status = 'healthy'
                            elif days_old <= 7:
                                status = 'stale'
                            else:
                                status = 'very_stale'
                        else:
                            status = 'unknown'
                        
                        collection_status[source] = {
                            'status': status,
                            'records': source_info.get('records', 0),
                            'file_size_mb': source_info.get('file_size_mb', 0),
                            'date_range': date_range,
                            'days_old': days_old if date_range else None
                        }
            
            # Calculate overall forecast collection health
            healthy_sources = sum(1 for info in collection_status.values() 
                                if info.get('status') == 'healthy')
            total_sources = len(sources)
            
            if healthy_sources >= total_sources * 0.8:
                overall_status = 'healthy'
            elif healthy_sources >= total_sources * 0.5:
                overall_status = 'degraded'
            else:
                overall_status = 'critical'
            
            return {
                'overall_status': overall_status,
                'healthy_sources': healthy_sources,
                'total_sources': total_sources,
                'sources': collection_status
            }
            
        except Exception as e:
            logger.error(f"Error checking forecast collection: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _check_data_quality_health(self) -> Dict[str, Any]:
        """Check data quality across all sources."""
        logger.info("Checking data quality health...")
        
        try:
            sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
            quality_results = {}
            quality_scores = []
            
            for source in sources:
                quality_summary = self.data_manager.get_data_quality_summary(source, days=7)
                
                if 'error' in quality_summary:
                    quality_results[source] = {
                        'status': 'error',
                        'error': quality_summary['error']
                    }
                else:
                    avg_quality = quality_summary.get('avg_quality_score', 0)
                    quality_scores.append(avg_quality)
                    
                    if avg_quality >= self.thresholds['data_quality_score']:
                        status = 'healthy'
                    elif avg_quality >= 0.5:
                        status = 'degraded'
                    else:
                        status = 'poor'
                    
                    quality_results[source] = {
                        'status': status,
                        'avg_quality_score': avg_quality,
                        'total_records': quality_summary.get('total_records', 0),
                        'low_quality_count': quality_summary.get('low_quality_count', 0),
                        'high_quality_count': quality_summary.get('high_quality_count', 0)
                    }
            
            # Calculate overall quality
            if quality_scores:
                overall_quality = sum(quality_scores) / len(quality_scores)
                
                if overall_quality >= self.thresholds['data_quality_score']:
                    overall_status = 'healthy'
                elif overall_quality >= 0.5:
                    overall_status = 'degraded'
                else:
                    overall_status = 'poor'
            else:
                overall_status = 'unknown'
                overall_quality = 0
            
            return {
                'status': overall_status,
                'overall_quality_score': overall_quality,
                'sources': quality_results,
                'threshold': self.thresholds['data_quality_score']
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        logger.info("Checking system resources...")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100
            
            # Determine overall resource health
            resource_issues = []
            
            if cpu_percent / 100 > self.thresholds['cpu_usage']:
                resource_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.thresholds['memory_usage']:
                resource_issues.append(f"High memory usage: {memory_percent:.1%}")
            
            if disk_percent > self.thresholds['disk_usage']:
                resource_issues.append(f"High disk usage: {disk_percent:.1%}")
            
            if len(resource_issues) == 0:
                status = 'healthy'
            elif len(resource_issues) <= 1:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'cpu_usage': cpu_percent / 100,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'issues': resource_issues,
                'thresholds': {
                    'cpu': self.thresholds['cpu_usage'],
                    'memory': self.thresholds['memory_usage'],
                    'disk': self.thresholds['disk_usage']
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_prediction_health(self) -> Dict[str, Any]:
        """Check prediction system health."""
        logger.info("Checking prediction system health...")
        
        try:
            # Load recent predictions
            recent_predictions = self.data_manager.load_predictions(
                start_date=date.today() - timedelta(days=7),
                end_date=date.today()
            )
            
            if recent_predictions.empty:
                return {
                    'status': 'no_data',
                    'message': 'No recent predictions found',
                    'predictions_count': 0
                }
            
            # Calculate prediction metrics
            predictions_with_actuals = recent_predictions.dropna(subset=['actual_temperature'])
            
            if predictions_with_actuals.empty:
                return {
                    'status': 'no_validation',
                    'message': 'No predictions with actual temperatures for validation',
                    'predictions_count': len(recent_predictions),
                    'avg_confidence': recent_predictions['confidence'].mean()
                }
            
            # Calculate accuracy
            errors = abs(predictions_with_actuals['predicted_high'] - 
                        predictions_with_actuals['actual_temperature'])
            accuracy_3f = (errors <= 3.0).mean()  # Within ±3°F
            rmse = (errors ** 2).mean() ** 0.5
            mae = errors.mean()
            
            # Average confidence
            avg_confidence = recent_predictions['confidence'].mean()
            
            # Determine health status
            if accuracy_3f >= self.thresholds['prediction_accuracy'] and avg_confidence >= self.thresholds['model_confidence']:
                status = 'healthy'
            elif accuracy_3f >= 0.5 or avg_confidence >= 0.4:
                status = 'degraded'
            else:
                status = 'poor'
            
            return {
                'status': status,
                'predictions_count': len(recent_predictions),
                'validated_predictions': len(predictions_with_actuals),
                'accuracy_3f': accuracy_3f,
                'rmse': rmse,
                'mae': mae,
                'avg_confidence': avg_confidence,
                'thresholds': {
                    'accuracy': self.thresholds['prediction_accuracy'],
                    'confidence': self.thresholds['model_confidence']
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking prediction health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """Check data storage health."""
        logger.info("Checking storage health...")
        
        try:
            data_dir = self.data_manager.data_dir
            
            # Check if data directory exists
            if not data_dir.exists():
                return {
                    'status': 'critical',
                    'error': f'Data directory does not exist: {data_dir}'
                }
            
            # Check data files
            expected_files = [
                'nws_data.parquet',
                'openweather_data.parquet',
                'tomorrow_data.parquet',
                'weatherbit_data.parquet',
                'visual_crossing_data.parquet',
                'actual_temperatures.parquet'
            ]
            
            file_status = {}
            missing_files = []
            total_size = 0
            
            for filename in expected_files:
                file_path = data_dir / filename
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size += size_mb
                    file_status[filename] = {
                        'exists': True,
                        'size_mb': size_mb,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    }
                else:
                    missing_files.append(filename)
                    file_status[filename] = {
                        'exists': False
                    }
            
            # Determine storage health
            if len(missing_files) == 0:
                status = 'healthy'
            elif len(missing_files) <= 2:
                status = 'degraded'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'data_directory': str(data_dir),
                'total_files': len(expected_files),
                'existing_files': len(expected_files) - len(missing_files),
                'missing_files': missing_files,
                'total_size_mb': total_size,
                'file_details': file_status
            }
            
        except Exception as e:
            logger.error(f"Error checking storage health: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_overall_health(self, components: Dict[str, Any]) -> str:
        """Calculate overall system health from component health."""
        component_statuses = []
        
        for component_name, component_info in components.items():
            status = component_info.get('status', 'unknown')
            component_statuses.append(status)
        
        # Count status types
        critical_count = component_statuses.count('critical')
        error_count = component_statuses.count('error')
        degraded_count = component_statuses.count('degraded')
        poor_count = component_statuses.count('poor')
        
        # Determine overall health
        if critical_count > 0 or error_count > 2:
            return 'critical'
        elif error_count > 0 or degraded_count > 2 or poor_count > 1:
            return 'degraded'
        elif degraded_count > 0 or poor_count > 0:
            return 'warning'
        else:
            return 'healthy'
    
    def _generate_health_metrics(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Generate health metrics summary."""
        metrics = {
            'component_count': len(components),
            'healthy_components': 0,
            'degraded_components': 0,
            'critical_components': 0,
            'error_components': 0
        }
        
        for component_info in components.values():
            status = component_info.get('status', 'unknown')
            
            if status == 'healthy':
                metrics['healthy_components'] += 1
            elif status in ['degraded', 'warning', 'stale']:
                metrics['degraded_components'] += 1
            elif status in ['critical', 'poor']:
                metrics['critical_components'] += 1
            elif status == 'error':
                metrics['error_components'] += 1
        
        # Calculate health percentage
        total_components = metrics['component_count']
        if total_components > 0:
            metrics['health_percentage'] = metrics['healthy_components'] / total_components
        else:
            metrics['health_percentage'] = 0
        
        return metrics
    
    def _generate_health_alerts(self, health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on health status."""
        alerts = []
        
        overall_health = health_status['overall_health']
        components = health_status['components']
        
        # Overall health alerts
        if overall_health == 'critical':
            alert_system.check_system_health_alerts({'overall_health': 'critical'})
        elif overall_health == 'degraded':
            alert_system.check_system_health_alerts({'overall_health': 'degraded'})
        
        # Component-specific alerts
        for component_name, component_info in components.items():
            status = component_info.get('status')
            
            if status in ['critical', 'error']:
                alerts.append({
                    'type': 'component_critical',
                    'component': component_name,
                    'status': status,
                    'message': f"{component_name} is in {status} state",
                    'details': component_info
                })
            elif status in ['degraded', 'poor', 'warning']:
                alerts.append({
                    'type': 'component_degraded',
                    'component': component_name,
                    'status': status,
                    'message': f"{component_name} is in {status} state",
                    'details': component_info
                })
        
        return alerts
    
    def _generate_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        components = health_status['components']
        
        # API health recommendations
        api_health = components.get('apis', {})
        if api_health.get('status') in ['degraded', 'critical']:
            failed_apis = api_health.get('failed_apis', 0)
            if failed_apis > 0:
                recommendations.append(f"Check {failed_apis} failed API connections and verify API keys")
        
        # Data collection recommendations
        collection_health = components.get('data_collection', {})
        if collection_health.get('status') in ['degraded', 'critical']:
            temp_collection = collection_health.get('temperature_collection', {})
            missing_dates = temp_collection.get('missing_dates', [])
            if missing_dates:
                recommendations.append(f"Backfill missing temperature data for {len(missing_dates)} dates")
        
        # Data quality recommendations
        quality_health = components.get('data_quality', {})
        if quality_health.get('status') in ['degraded', 'poor']:
            recommendations.append("Review data quality issues and improve validation rules")
        
        # System resource recommendations
        resource_health = components.get('system_resources', {})
        if resource_health.get('status') in ['warning', 'critical']:
            issues = resource_health.get('issues', [])
            for issue in issues:
                recommendations.append(f"Address resource issue: {issue}")
        
        # Prediction system recommendations
        prediction_health = components.get('prediction_system', {})
        if prediction_health.get('status') == 'no_data':
            recommendations.append("Train and deploy prediction models")
        elif prediction_health.get('status') in ['degraded', 'poor']:
            recommendations.append("Retrain models to improve prediction accuracy")
        
        # Storage recommendations
        storage_health = components.get('storage', {})
        if storage_health.get('status') in ['degraded', 'critical']:
            missing_files = storage_health.get('missing_files', [])
            if missing_files:
                recommendations.append(f"Initialize missing data files: {', '.join(missing_files)}")
        
        return recommendations
    
    def generate_health_report(self) -> str:
        """Generate a formatted health report."""
        health_status = self.get_comprehensive_health_status()
        
        report = []
        report.append("=" * 60)
        report.append("SYSTEM HEALTH REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {health_status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Overall Health: {health_status['overall_health'].upper()}")
        report.append("")
        
        # Component status
        report.append("COMPONENT STATUS:")
        report.append("-" * 20)
        
        for component_name, component_info in health_status['components'].items():
            status = component_info.get('status', 'unknown')
            status_icon = {
                'healthy': '✅',
                'degraded': '⚠️',
                'warning': '⚠️',
                'critical': '❌',
                'error': '💥',
                'poor': '❌'
            }.get(status, '❓')
            
            report.append(f"  {status_icon} {component_name.replace('_', ' ').title()}: {status}")
        
        report.append("")
        
        # Metrics
        metrics = health_status['metrics']
        report.append("HEALTH METRICS:")
        report.append("-" * 15)
        report.append(f"  Total Components: {metrics['component_count']}")
        report.append(f"  Healthy: {metrics['healthy_components']}")
        report.append(f"  Degraded: {metrics['degraded_components']}")
        report.append(f"  Critical: {metrics['critical_components']}")
        report.append(f"  Health Percentage: {metrics['health_percentage']:.1%}")
        report.append("")
        
        # Alerts
        alerts = health_status['alerts']
        if alerts:
            report.append("ACTIVE ALERTS:")
            report.append("-" * 14)
            for alert in alerts:
                report.append(f"  🚨 {alert['message']}")
            report.append("")
        
        # Recommendations
        recommendations = health_status['recommendations']
        if recommendations:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 16)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Main entry point for system health monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System health monitoring")
    parser.add_argument("--report", action="store_true", help="Generate health report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--component", type=str, help="Check specific component")
    
    args = parser.parse_args()
    
    monitor = SystemHealthMonitor()
    
    if args.report:
        # Generate formatted report
        report = monitor.generate_health_report()
        print(report)
    elif args.json:
        # Generate JSON output
        import json
        health_status = monitor.get_comprehensive_health_status()
        # Convert datetime objects to strings for JSON serialization
        health_status['timestamp'] = health_status['timestamp'].isoformat()
        print(json.dumps(health_status, indent=2, default=str))
    else:
        # Default: simple status check
        health_status = monitor.get_comprehensive_health_status()
        
        print(f"System Health: {health_status['overall_health'].upper()}")
        print(f"Timestamp: {health_status['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        metrics = health_status['metrics']
        print(f"Components: {metrics['healthy_components']}/{metrics['component_count']} healthy")
        
        if health_status['alerts']:
            print(f"Alerts: {len(health_status['alerts'])} active")
        
        if health_status['recommendations']:
            print(f"Recommendations: {len(health_status['recommendations'])} available")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())