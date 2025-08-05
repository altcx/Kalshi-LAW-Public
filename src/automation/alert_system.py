#!/usr/bin/env python3
"""Alert system for high-confidence trading opportunities and significant changes."""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.automation.error_handler import error_handler, ErrorSeverity


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    CONSOLE = "console"
    FILE = "file"
    EMAIL = "email"  # Future implementation
    SLACK = "slack"  # Future implementation
    SMS = "sms"      # Future implementation


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    timestamp: datetime
    type: str
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    expires_at: Optional[datetime] = None


class AlertSystem:
    """Comprehensive alert system for trading opportunities and system events."""
    
    def __init__(self):
        """Initialize the alert system."""
        self.alerts_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Setup default alert handlers
        self._setup_default_handlers()
        
        # Setup default alert rules
        self._setup_default_rules()
        
        logger.info("AlertSystem initialized")
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        self.alert_handlers[AlertChannel.LOG] = self._handle_log_alert
        self.alert_handlers[AlertChannel.CONSOLE] = self._handle_console_alert
        self.alert_handlers[AlertChannel.FILE] = self._handle_file_alert
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        self.alert_rules = {
            'high_confidence_opportunity': {
                'confidence_threshold': 0.85,
                'expected_value_threshold': 0.1,
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
                'severity': AlertSeverity.SUCCESS
            },
            'significant_prediction_change': {
                'temperature_change_threshold': 3.0,
                'confidence_change_threshold': 0.15,
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE],
                'severity': AlertSeverity.WARNING
            },
            'low_confidence_warning': {
                'confidence_threshold': 0.50,
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE],
                'severity': AlertSeverity.WARNING
            },
            'model_performance_degradation': {
                'accuracy_threshold': 0.70,
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
                'severity': AlertSeverity.ERROR
            },
            'api_failure': {
                'failure_count_threshold': 3,
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
                'severity': AlertSeverity.ERROR
            },
            'system_health_critical': {
                'channels': [AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.FILE],
                'severity': AlertSeverity.CRITICAL
            }
        }
    
    def create_alert(self, alert_type: str, title: str, message: str, 
                    data: Optional[Dict[str, Any]] = None,
                    severity: Optional[AlertSeverity] = None,
                    channels: Optional[List[AlertChannel]] = None,
                    expires_in_hours: Optional[int] = None) -> Alert:
        """Create a new alert.
        
        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            data: Additional alert data
            severity: Alert severity (uses rule default if not specified)
            channels: Alert channels (uses rule default if not specified)
            expires_in_hours: Hours until alert expires
            
        Returns:
            Created Alert object
        """
        # Generate unique alert ID
        alert_id = f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get rule configuration
        rule_config = self.alert_rules.get(alert_type, {})
        
        # Use provided values or fall back to rule defaults
        if severity is None:
            severity = rule_config.get('severity', AlertSeverity.INFO)
        
        if channels is None:
            channels = rule_config.get('channels', [AlertChannel.LOG])
        
        # Calculate expiration time
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=data or {},
            channels=channels,
            expires_at=expires_at
        )
        
        # Store alert
        self.alerts_history.append(alert)
        self.active_alerts[alert_id] = alert
        
        # Send alert through configured channels
        self._send_alert(alert)
        
        logger.info(f"Created alert: {alert_type} - {title}")
        return alert
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels.
        
        Args:
            alert: Alert to send
        """
        for channel in alert.channels:
            handler = self.alert_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert through {channel.value}: {e}")
                    error_handler.log_error(
                        e,
                        f"Alert delivery via {channel.value}",
                        ErrorSeverity.MEDIUM,
                        {'alert_id': alert.id, 'alert_type': alert.type}
                    )
    
    def _handle_log_alert(self, alert: Alert):
        """Handle log channel alert."""
        log_message = f"ALERT [{alert.type}] {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        elif alert.severity == AlertSeverity.SUCCESS:
            logger.success(log_message)
        else:
            logger.info(log_message)
    
    def _handle_console_alert(self, alert: Alert):
        """Handle console channel alert."""
        # Color codes for different severities
        colors = {
            AlertSeverity.CRITICAL: '\033[91m',  # Red
            AlertSeverity.ERROR: '\033[91m',     # Red
            AlertSeverity.WARNING: '\033[93m',   # Yellow
            AlertSeverity.SUCCESS: '\033[92m',   # Green
            AlertSeverity.INFO: '\033[94m'       # Blue
        }
        reset_color = '\033[0m'
        
        color = colors.get(alert.severity, '')
        severity_icon = {
            AlertSeverity.CRITICAL: '🚨',
            AlertSeverity.ERROR: '❌',
            AlertSeverity.WARNING: '⚠️',
            AlertSeverity.SUCCESS: '✅',
            AlertSeverity.INFO: 'ℹ️'
        }.get(alert.severity, '📢')
        
        print(f"{color}{severity_icon} ALERT [{alert.type.upper()}]{reset_color}")
        print(f"{color}{alert.title}{reset_color}")
        print(f"{alert.message}")
        
        # Print additional data if available
        if alert.data:
            print("Additional Information:")
            for key, value in alert.data.items():
                print(f"  {key}: {value}")
        
        print("-" * 50)
    
    def _handle_file_alert(self, alert: Alert):
        """Handle file channel alert."""
        alerts_file = Path("logs/alerts.log")
        alerts_file.parent.mkdir(exist_ok=True)
        
        alert_record = {
            'id': alert.id,
            'timestamp': alert.timestamp.isoformat(),
            'type': alert.type,
            'severity': alert.severity.value,
            'title': alert.title,
            'message': alert.message,
            'data': alert.data
        }
        
        with open(alerts_file, 'a') as f:
            f.write(json.dumps(alert_record) + '\n')
    
    def check_high_confidence_opportunity(self, prediction_data: Dict[str, Any], 
                                        recommendations: List[Dict[str, Any]]) -> List[Alert]:
        """Check for high-confidence trading opportunities.
        
        Args:
            prediction_data: Prediction data
            recommendations: Trading recommendations
            
        Returns:
            List of generated alerts
        """
        alerts = []
        rule = self.alert_rules['high_confidence_opportunity']
        
        confidence = prediction_data.get('confidence', 0)
        predicted_temp = prediction_data.get('predicted_high', 0)
        
        # Check for high confidence predictions
        if confidence >= rule['confidence_threshold']:
            # Look for high expected value recommendations
            high_ev_recs = [
                rec for rec in recommendations
                if rec.get('expected_value', 0) >= rule['expected_value_threshold']
            ]
            
            if high_ev_recs:
                best_rec = max(high_ev_recs, key=lambda x: x.get('expected_value', 0))
                
                alert = self.create_alert(
                    alert_type='high_confidence_opportunity',
                    title=f"High Confidence Trading Opportunity - {prediction_data.get('date')}",
                    message=f"High confidence prediction ({confidence:.1%}) with profitable opportunity: "
                           f"{best_rec['contract_description']} - {best_rec['recommendation']} "
                           f"(EV: {best_rec['expected_value']:.3f})",
                    data={
                        'predicted_temperature': predicted_temp,
                        'confidence': confidence,
                        'best_recommendation': best_rec,
                        'total_opportunities': len(high_ev_recs)
                    },
                    expires_in_hours=24
                )
                alerts.append(alert)
        
        return alerts
    
    def check_prediction_changes(self, current_prediction: Dict[str, Any], 
                               previous_prediction: Optional[Dict[str, Any]]) -> List[Alert]:
        """Check for significant prediction changes.
        
        Args:
            current_prediction: Current prediction data
            previous_prediction: Previous prediction data
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        if previous_prediction is None:
            return alerts
        
        rule = self.alert_rules['significant_prediction_change']
        
        # Check temperature change
        temp_change = abs(
            current_prediction.get('predicted_high', 0) - 
            previous_prediction.get('predicted_high', 0)
        )
        
        if temp_change >= rule['temperature_change_threshold']:
            direction = "increased" if current_prediction.get('predicted_high', 0) > previous_prediction.get('predicted_high', 0) else "decreased"
            
            alert = self.create_alert(
                alert_type='significant_prediction_change',
                title=f"Significant Temperature Prediction Change - {current_prediction.get('date')}",
                message=f"Temperature prediction {direction} by {temp_change:.1f}°F from previous forecast",
                data={
                    'previous_temperature': previous_prediction.get('predicted_high'),
                    'current_temperature': current_prediction.get('predicted_high'),
                    'change': temp_change,
                    'direction': direction
                },
                expires_in_hours=12
            )
            alerts.append(alert)
        
        # Check confidence change
        confidence_change = abs(
            current_prediction.get('confidence', 0) - 
            previous_prediction.get('confidence', 0)
        )
        
        if confidence_change >= rule['confidence_change_threshold']:
            direction = "increased" if current_prediction.get('confidence', 0) > previous_prediction.get('confidence', 0) else "decreased"
            
            alert = self.create_alert(
                alert_type='significant_prediction_change',
                title=f"Significant Confidence Change - {current_prediction.get('date')}",
                message=f"Model confidence {direction} by {confidence_change:.1%}",
                data={
                    'previous_confidence': previous_prediction.get('confidence'),
                    'current_confidence': current_prediction.get('confidence'),
                    'change': confidence_change,
                    'direction': direction
                },
                expires_in_hours=12
            )
            alerts.append(alert)
        
        return alerts
    
    def check_low_confidence_warning(self, prediction_data: Dict[str, Any]) -> List[Alert]:
        """Check for low confidence warnings.
        
        Args:
            prediction_data: Prediction data
            
        Returns:
            List of generated alerts
        """
        alerts = []
        rule = self.alert_rules['low_confidence_warning']
        
        confidence = prediction_data.get('confidence', 0)
        
        if confidence <= rule['confidence_threshold']:
            alert = self.create_alert(
                alert_type='low_confidence_warning',
                title=f"Low Confidence Prediction Warning - {prediction_data.get('date')}",
                message=f"Model confidence is low ({confidence:.1%}). "
                       f"Consider avoiding trading or reducing position sizes.",
                data={
                    'confidence': confidence,
                    'predicted_temperature': prediction_data.get('predicted_high'),
                    'weather_condition': prediction_data.get('weather_condition')
                },
                expires_in_hours=24
            )
            alerts.append(alert)
        
        return alerts
    
    def check_system_health_alerts(self, health_status: Dict[str, Any]) -> List[Alert]:
        """Check for system health alerts.
        
        Args:
            health_status: System health status
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        overall_health = health_status.get('overall_health', 'unknown')
        
        if overall_health == 'critical':
            alert = self.create_alert(
                alert_type='system_health_critical',
                title="Critical System Health Issue",
                message="System health is critical. Immediate attention required.",
                data=health_status,
                expires_in_hours=1
            )
            alerts.append(alert)
        
        elif overall_health == 'degraded':
            alert = self.create_alert(
                alert_type='system_health_critical',
                title="System Health Degraded",
                message="System health is degraded. Some components may not be functioning properly.",
                data=health_status,
                severity=AlertSeverity.WARNING,
                expires_in_hours=6
            )
            alerts.append(alert)
        
        # Check API failures
        api_connections = health_status.get('api_connections', {})
        failed_apis = [api for api, status in api_connections.items() if not status]
        
        if len(failed_apis) >= self.alert_rules['api_failure']['failure_count_threshold']:
            alert = self.create_alert(
                alert_type='api_failure',
                title="Multiple API Failures Detected",
                message=f"Multiple APIs are failing: {', '.join(failed_apis)}",
                data={
                    'failed_apis': failed_apis,
                    'total_apis': len(api_connections),
                    'failure_count': len(failed_apis)
                },
                expires_in_hours=2
            )
            alerts.append(alert)
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts.
        
        Args:
            severity_filter: Filter by severity level
            
        Returns:
            List of active alerts
        """
        # Clean up expired alerts
        self._cleanup_expired_alerts()
        
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with alert summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts_history
            if alert.timestamp > cutoff_time
        ]
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in recent_alerts
                if alert.severity == severity
            ])
        
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            type_counts[alert.type] = type_counts.get(alert.type, 0) + 1
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'most_recent_alert': recent_alerts[-1] if recent_alerts else None
        }
    
    def _cleanup_expired_alerts(self):
        """Clean up expired alerts."""
        now = datetime.now()
        expired_alert_ids = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.expires_at and alert.expires_at < now
        ]
        
        for alert_id in expired_alert_ids:
            del self.active_alerts[alert_id]
            logger.debug(f"Expired alert removed: {alert_id}")


# Global alert system instance
alert_system = AlertSystem()