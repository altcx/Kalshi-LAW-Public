"""Comprehensive logging and monitoring system for weather prediction."""

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import json
import sys
import traceback
from collections import defaultdict, deque

from ..utils.data_manager import DataManager
from .performance_tracker import PerformanceTracker
from .model_retrainer import ModelRetrainer


class SystemMonitor:
    """Comprehensive system monitoring and logging for weather prediction system."""
    
    def __init__(self, data_manager: Optional[DataManager] = None,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 model_retrainer: Optional[ModelRetrainer] = None):
        """Initialize system monitor.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
            performance_tracker: PerformanceTracker instance (creates new one if None)
            model_retrainer: ModelRetrainer instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.performance_tracker = performance_tracker or PerformanceTracker(self.data_manager)
        self.model_retrainer = model_retrainer or ModelRetrainer(self.data_manager, self.performance_tracker)
        
        # Monitoring configuration
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # System health thresholds
        self.health_thresholds = {
            'api_success_rate': 0.80,      # Minimum API success rate
            'data_quality_score': 0.70,    # Minimum average data quality
            'prediction_accuracy': 0.70,   # Minimum prediction accuracy (±3°F)
            'model_age_days': 14,          # Maximum days since model retraining
            'disk_usage_mb': 1000,         # Maximum disk usage in MB
            'memory_usage_mb': 500,        # Maximum memory usage in MB
            'response_time_seconds': 30    # Maximum response time for operations
        }
        
        # Error tracking
        self.error_history = deque(maxlen=1000)  # Keep last 1000 errors
        self.api_failure_counts = defaultdict(int)
        self.system_alerts = []
        
        # Performance metrics history
        self.metrics_history = deque(maxlen=100)  # Keep last 100 metric snapshots
        
        # Setup logging configuration
        self._setup_logging()
        
        logger.info("SystemMonitor initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        try:
            # Remove default logger
            logger.remove()
            
            # Add console logging with colors
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO",
                colorize=True
            )
            
            # Add file logging for all messages
            logger.add(
                self.logs_dir / "weather_predictor.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="30 days",
                compression="zip"
            )
            
            # Add separate error log
            logger.add(
                self.logs_dir / "errors.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation="5 MB",
                retention="60 days",
                compression="zip"
            )
            
            # Add performance log
            logger.add(
                self.logs_dir / "performance.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
                level="INFO",
                rotation="5 MB",
                retention="30 days",
                filter=lambda record: "PERFORMANCE" in record["message"]
            )
            
            # Add API monitoring log
            logger.add(
                self.logs_dir / "api_monitoring.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
                level="INFO",
                rotation="5 MB",
                retention="30 days",
                filter=lambda record: "API_MONITOR" in record["message"]
            )
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def log_prediction(self, prediction: float, confidence: float, target_date: date,
                      model_info: Optional[Dict] = None, features_used: Optional[List] = None) -> None:
        """Log a temperature prediction with full context.
        
        Args:
            prediction: Predicted temperature
            confidence: Confidence score
            target_date: Date the prediction is for
            model_info: Information about the model used
            features_used: List of features used in prediction
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'prediction',
                'target_date': target_date.isoformat(),
                'predicted_temperature': prediction,
                'confidence': confidence,
                'model_info': model_info or {},
                'features_used': features_used or [],
                'session_id': self._get_session_id()
            }
            
            logger.info(f"PREDICTION | {json.dumps(log_entry)}")
            
            # Store prediction in data manager
            self.data_manager.store_prediction(
                prediction, confidence, target_date,
                model_info, {'features_used': features_used}
            )
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def log_actual_temperature(self, target_date: date, actual_temperature: float,
                              source: str = "NOAA") -> None:
        """Log actual temperature observation.
        
        Args:
            target_date: Date of observation
            actual_temperature: Actual temperature
            source: Source of observation
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'actual_temperature',
                'target_date': target_date.isoformat(),
                'actual_temperature': actual_temperature,
                'source': source,
                'session_id': self._get_session_id()
            }
            
            logger.info(f"ACTUAL_TEMP | {json.dumps(log_entry)}")
            
            # Store in data manager
            self.data_manager.store_actual_temperature(target_date, actual_temperature, source)
            
            # Check for prediction accuracy
            self._check_prediction_accuracy(target_date, actual_temperature)
            
        except Exception as e:
            logger.error(f"Error logging actual temperature: {e}")
    
    def log_api_call(self, api_name: str, endpoint: str, success: bool,
                    response_time: float, error_message: Optional[str] = None,
                    data_quality: Optional[float] = None) -> None:
        """Log API call with performance metrics.
        
        Args:
            api_name: Name of the API
            endpoint: API endpoint called
            success: Whether the call was successful
            response_time: Response time in seconds
            error_message: Error message if failed
            data_quality: Quality score of returned data
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'api_call',
                'api_name': api_name,
                'endpoint': endpoint,
                'success': success,
                'response_time_seconds': response_time,
                'error_message': error_message,
                'data_quality': data_quality,
                'session_id': self._get_session_id()
            }
            
            logger.info(f"API_MONITOR | {json.dumps(log_entry)}")
            
            # Track API failures
            if not success:
                self.api_failure_counts[api_name] += 1
                self._check_api_health(api_name)
            
            # Check response time
            if response_time > self.health_thresholds['response_time_seconds']:
                self._create_alert(
                    'slow_api_response',
                    f"{api_name} response time {response_time:.1f}s exceeds threshold",
                    'warning'
                )
            
        except Exception as e:
            logger.error(f"Error logging API call: {e}")
    
    def log_error(self, error: Exception, context: Optional[Dict] = None,
                 severity: str = 'error') -> None:
        """Log error with full context and stack trace.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            severity: Error severity level
        """
        try:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'error',
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc(),
                'context': context or {},
                'severity': severity,
                'session_id': self._get_session_id()
            }
            
            # Add to error history
            self.error_history.append(error_entry)
            
            # Log based on severity
            if severity == 'critical':
                logger.critical(f"CRITICAL_ERROR | {json.dumps(error_entry)}")
                self._create_alert('critical_error', f"Critical error: {str(error)}", 'critical')
            elif severity == 'error':
                logger.error(f"ERROR | {json.dumps(error_entry)}")
            elif severity == 'warning':
                logger.warning(f"WARNING | {json.dumps(error_entry)}")
            else:
                logger.info(f"INFO_ERROR | {json.dumps(error_entry)}")
            
        except Exception as e:
            # Fallback logging if our error logging fails
            logger.error(f"Error in error logging: {e}")
            logger.error(f"Original error: {error}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        try:
            metrics_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'performance_metrics',
                'metrics': metrics,
                'session_id': self._get_session_id()
            }
            
            # Add to metrics history
            self.metrics_history.append(metrics_entry)
            
            logger.info(f"PERFORMANCE | {json.dumps(metrics_entry)}")
            
            # Check for performance issues
            self._check_performance_health(metrics)
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor overall system health and generate health report.
        
        Returns:
            Dictionary with system health status
        """
        try:
            logger.info("Starting system health monitoring")
            
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'components': {},
                'alerts': [],
                'recommendations': []
            }
            
            # Check API health
            api_health = self._check_all_apis_health()
            health_report['components']['apis'] = api_health
            
            # Check data quality
            data_health = self._check_data_health()
            health_report['components']['data'] = data_health
            
            # Check model performance
            model_health = self._check_model_health()
            health_report['components']['models'] = model_health
            
            # Check system resources
            resource_health = self._check_resource_health()
            health_report['components']['resources'] = resource_health
            
            # Check recent errors
            error_health = self._check_error_health()
            health_report['components']['errors'] = error_health
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in health_report['components'].values()]
            if 'critical' in component_statuses:
                health_report['overall_status'] = 'critical'
            elif 'warning' in component_statuses:
                health_report['overall_status'] = 'warning'
            elif 'degraded' in component_statuses:
                health_report['overall_status'] = 'degraded'
            
            # Add current alerts
            health_report['alerts'] = self.system_alerts[-10:]  # Last 10 alerts
            
            # Generate recommendations
            health_report['recommendations'] = self._generate_recommendations(health_report)
            
            logger.info(f"System health check completed: {health_report['overall_status']}")
            self.log_performance_metrics({'system_health': health_report})
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error in system health monitoring: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _check_prediction_accuracy(self, target_date: date, actual_temperature: float) -> None:
        """Check accuracy of predictions for a specific date.
        
        Args:
            target_date: Date to check
            actual_temperature: Actual temperature observed
        """
        try:
            # Get predictions for this date
            predictions_df = self.data_manager.load_source_data('predictions')
            if predictions_df.empty:
                return
            
            predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
            date_predictions = predictions_df[predictions_df['date'] == target_date]
            
            if not date_predictions.empty:
                for _, pred_row in date_predictions.iterrows():
                    predicted_temp = pred_row['predicted_high']
                    confidence = pred_row['confidence']
                    error = abs(predicted_temp - actual_temperature)
                    
                    accuracy_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'prediction_accuracy',
                        'target_date': target_date.isoformat(),
                        'predicted_temperature': predicted_temp,
                        'actual_temperature': actual_temperature,
                        'error': error,
                        'confidence': confidence,
                        'accurate_within_3f': error <= 3.0,
                        'session_id': self._get_session_id()
                    }
                    
                    logger.info(f"ACCURACY | {json.dumps(accuracy_entry)}")
                    
                    # Alert on large errors
                    if error > 5.0:
                        self._create_alert(
                            'large_prediction_error',
                            f"Large prediction error: {error:.1f}°F for {target_date}",
                            'warning'
                        )
            
        except Exception as e:
            logger.error(f"Error checking prediction accuracy: {e}")
    
    def _check_api_health(self, api_name: str) -> None:
        """Check health of a specific API.
        
        Args:
            api_name: Name of the API to check
        """
        failure_count = self.api_failure_counts.get(api_name, 0)
        
        if failure_count > 5:  # More than 5 failures
            self._create_alert(
                'api_failures',
                f"{api_name} has {failure_count} recent failures",
                'warning'
            )
        
        if failure_count > 10:  # More than 10 failures
            self._create_alert(
                'api_critical',
                f"{api_name} has {failure_count} failures - may be down",
                'critical'
            )
    
    def _check_all_apis_health(self) -> Dict[str, Any]:
        """Check health of all APIs.
        
        Returns:
            Dictionary with API health status
        """
        try:
            api_sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
            api_health = {
                'status': 'healthy',
                'apis': {},
                'total_failures': sum(self.api_failure_counts.values()),
                'most_failures': max(self.api_failure_counts.items(), key=lambda x: x[1]) if self.api_failure_counts else None
            }
            
            for api in api_sources:
                failure_count = self.api_failure_counts.get(api, 0)
                if failure_count == 0:
                    status = 'healthy'
                elif failure_count <= 5:
                    status = 'warning'
                else:
                    status = 'critical'
                
                api_health['apis'][api] = {
                    'status': status,
                    'failure_count': failure_count
                }
            
            # Determine overall API health
            api_statuses = [api['status'] for api in api_health['apis'].values()]
            if 'critical' in api_statuses:
                api_health['status'] = 'critical'
            elif 'warning' in api_statuses:
                api_health['status'] = 'warning'
            
            return api_health
            
        except Exception as e:
            logger.error(f"Error checking API health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_data_health(self) -> Dict[str, Any]:
        """Check health of data quality and availability.
        
        Returns:
            Dictionary with data health status
        """
        try:
            data_summary = self.data_manager.get_data_summary()
            
            data_health = {
                'status': 'healthy',
                'sources': {},
                'total_records': 0,
                'avg_quality_score': 0.0,
                'recent_data_available': True
            }
            
            quality_scores = []
            total_records = 0
            
            for source, info in data_summary.items():
                if source in ['model_performance', 'predictions']:
                    continue
                
                if 'error' in info:
                    data_health['sources'][source] = {
                        'status': 'error',
                        'error': info['error']
                    }
                    continue
                
                records = info.get('records', 0)
                total_records += records
                
                # Check if we have recent data (within last 2 days)
                date_range = info.get('date_range')
                recent_data = False
                if date_range and date_range.get('end'):
                    try:
                        end_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()
                        recent_data = (date.today() - end_date).days <= 2
                    except:
                        pass
                
                # Get quality score for this source
                quality_summary = self.data_manager.get_data_quality_summary(source)
                avg_quality = quality_summary.get('avg_quality_score', 0.8)
                quality_scores.append(avg_quality)
                
                # Determine source status
                if records == 0:
                    status = 'critical'
                elif not recent_data:
                    status = 'warning'
                elif avg_quality < self.health_thresholds['data_quality_score']:
                    status = 'degraded'
                else:
                    status = 'healthy'
                
                data_health['sources'][source] = {
                    'status': status,
                    'records': records,
                    'quality_score': avg_quality,
                    'recent_data': recent_data
                }
            
            data_health['total_records'] = total_records
            data_health['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0.0
            
            # Determine overall data health
            source_statuses = [src['status'] for src in data_health['sources'].values()]
            if 'critical' in source_statuses:
                data_health['status'] = 'critical'
            elif 'warning' in source_statuses or 'degraded' in source_statuses:
                data_health['status'] = 'warning'
            
            return data_health
            
        except Exception as e:
            logger.error(f"Error checking data health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check health of prediction models.
        
        Returns:
            Dictionary with model health status
        """
        try:
            model_health = {
                'status': 'healthy',
                'models': {},
                'last_retraining': None,
                'performance_trend': 'stable'
            }
            
            # Check recent model performance
            recent_performance = self.performance_tracker.track_all_sources_performance(window_days=7)
            
            for source, metrics in recent_performance.items():
                if source == 'ensemble' or 'error' in metrics:
                    continue
                
                accuracy = metrics.get('accuracy_3f', 0)
                mae = metrics.get('mae', float('inf'))
                
                if accuracy >= self.health_thresholds['prediction_accuracy']:
                    status = 'healthy'
                elif accuracy >= 0.60:
                    status = 'warning'
                else:
                    status = 'critical'
                
                model_health['models'][source] = {
                    'status': status,
                    'accuracy_3f': accuracy,
                    'mae': mae,
                    'performance_category': metrics.get('performance_category', 'unknown')
                }
            
            # Check if models need retraining
            retraining_check = self.model_retrainer.check_retraining_needed('ensemble')
            if retraining_check.get('should_retrain', False):
                model_health['needs_retraining'] = True
                model_health['retraining_reasons'] = retraining_check.get('reasons', [])
            
            # Determine overall model health
            model_statuses = [model['status'] for model in model_health['models'].values()]
            if 'critical' in model_statuses:
                model_health['status'] = 'critical'
            elif 'warning' in model_statuses:
                model_health['status'] = 'warning'
            
            return model_health
            
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_resource_health(self) -> Dict[str, Any]:
        """Check system resource health.
        
        Returns:
            Dictionary with resource health status
        """
        try:
            import psutil
            
            # Get system metrics
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            resource_health = {
                'status': 'healthy',
                'memory': {
                    'used_mb': memory.used / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'percent': memory.percent
                },
                'disk': {
                    'used_mb': disk.used / (1024 * 1024),
                    'free_mb': disk.free / (1024 * 1024),
                    'percent': (disk.used / disk.total) * 100
                }
            }
            
            # Check thresholds
            if memory.percent > 90:
                resource_health['status'] = 'critical'
                self._create_alert('high_memory_usage', f"Memory usage at {memory.percent:.1f}%", 'critical')
            elif memory.percent > 80:
                resource_health['status'] = 'warning'
                self._create_alert('high_memory_usage', f"Memory usage at {memory.percent:.1f}%", 'warning')
            
            if resource_health['disk']['percent'] > 90:
                resource_health['status'] = 'critical'
                self._create_alert('high_disk_usage', f"Disk usage at {resource_health['disk']['percent']:.1f}%", 'critical')
            elif resource_health['disk']['percent'] > 80:
                resource_health['status'] = 'warning'
            
            return resource_health
            
        except ImportError:
            return {
                'status': 'unknown',
                'error': 'psutil not available for resource monitoring'
            }
        except Exception as e:
            logger.error(f"Error checking resource health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_error_health(self) -> Dict[str, Any]:
        """Check recent error patterns.
        
        Returns:
            Dictionary with error health status
        """
        try:
            recent_errors = [err for err in self.error_history 
                           if datetime.fromisoformat(err['timestamp']) > datetime.now() - timedelta(hours=24)]
            
            error_health = {
                'status': 'healthy',
                'recent_errors_24h': len(recent_errors),
                'critical_errors_24h': len([err for err in recent_errors if err['severity'] == 'critical']),
                'error_types': {}
            }
            
            # Count error types
            for error in recent_errors:
                error_type = error['error_type']
                if error_type not in error_health['error_types']:
                    error_health['error_types'][error_type] = 0
                error_health['error_types'][error_type] += 1
            
            # Determine status
            if error_health['critical_errors_24h'] > 0:
                error_health['status'] = 'critical'
            elif error_health['recent_errors_24h'] > 20:
                error_health['status'] = 'warning'
            elif error_health['recent_errors_24h'] > 50:
                error_health['status'] = 'critical'
            
            return error_health
            
        except Exception as e:
            logger.error(f"Error checking error health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_performance_health(self, metrics: Dict[str, Any]) -> None:
        """Check performance metrics against thresholds.
        
        Args:
            metrics: Performance metrics to check
        """
        try:
            # Check prediction accuracy
            if 'accuracy_3f' in metrics:
                accuracy = metrics['accuracy_3f']
                if accuracy < self.health_thresholds['prediction_accuracy']:
                    self._create_alert(
                        'low_prediction_accuracy',
                        f"Prediction accuracy {accuracy:.1%} below threshold",
                        'warning'
                    )
            
            # Check response times
            if 'response_time' in metrics:
                response_time = metrics['response_time']
                if response_time > self.health_thresholds['response_time_seconds']:
                    self._create_alert(
                        'slow_response_time',
                        f"Response time {response_time:.1f}s exceeds threshold",
                        'warning'
                    )
            
        except Exception as e:
            logger.error(f"Error checking performance health: {e}")
    
    def _create_alert(self, alert_type: str, message: str, severity: str) -> None:
        """Create a system alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, critical)
        """
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity,
                'session_id': self._get_session_id()
            }
            
            self.system_alerts.append(alert)
            
            # Log alert
            if severity == 'critical':
                logger.critical(f"ALERT | {json.dumps(alert)}")
            elif severity == 'warning':
                logger.warning(f"ALERT | {json.dumps(alert)}")
            else:
                logger.info(f"ALERT | {json.dumps(alert)}")
            
            # Keep only recent alerts
            if len(self.system_alerts) > 100:
                self.system_alerts = self.system_alerts[-50:]
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def _generate_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health report.
        
        Args:
            health_report: System health report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # API recommendations
            api_health = health_report['components'].get('apis', {})
            if api_health.get('status') in ['warning', 'critical']:
                recommendations.append("Check API connectivity and consider backup data sources")
            
            # Data recommendations
            data_health = health_report['components'].get('data', {})
            if data_health.get('status') in ['warning', 'critical']:
                recommendations.append("Review data quality and collection processes")
            
            # Model recommendations
            model_health = health_report['components'].get('models', {})
            if model_health.get('status') in ['warning', 'critical']:
                recommendations.append("Consider model retraining or parameter adjustment")
            
            # Resource recommendations
            resource_health = health_report['components'].get('resources', {})
            if resource_health.get('status') in ['warning', 'critical']:
                recommendations.append("Monitor system resources and consider scaling")
            
            # Error recommendations
            error_health = health_report['components'].get('errors', {})
            if error_health.get('status') in ['warning', 'critical']:
                recommendations.append("Investigate recent error patterns and implement fixes")
            
            if not recommendations:
                recommendations.append("System is operating normally")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to error")
        
        return recommendations
    
    def _get_session_id(self) -> str:
        """Get current session ID for tracking.
        
        Returns:
            Session ID string
        """
        # Simple session ID based on current date
        return datetime.now().strftime("%Y%m%d_%H")
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data.
        
        Returns:
            Dictionary with dashboard information
        """
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.monitor_system_health(),
                'recent_metrics': list(self.metrics_history)[-10:] if self.metrics_history else [],
                'recent_errors': list(self.error_history)[-10:] if self.error_history else [],
                'recent_alerts': self.system_alerts[-10:] if self.system_alerts else [],
                'api_failure_counts': dict(self.api_failure_counts),
                'data_summary': self.data_manager.get_data_summary()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating system dashboard: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """Clean up old log files.
        
        Args:
            days_to_keep: Number of days of logs to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.logs_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Cleaned up old log file: {log_file}")
            
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")