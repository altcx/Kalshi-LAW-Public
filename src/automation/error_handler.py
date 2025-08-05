"""Error handling and retry logic for automated data collection."""

import time
import functools
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"


class ErrorHandler:
    """Handles errors and implements retry logic for automated tasks."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.alert_thresholds = {
            ErrorSeverity.LOW: 10,      # Alert after 10 low severity errors
            ErrorSeverity.MEDIUM: 5,    # Alert after 5 medium severity errors
            ErrorSeverity.HIGH: 2,      # Alert after 2 high severity errors
            ErrorSeverity.CRITICAL: 1   # Alert immediately for critical errors
        }
    
    def log_error(self, error: Exception, context: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  additional_info: Optional[Dict] = None) -> None:
        """Log an error with context and severity.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            severity: Severity level of the error
            additional_info: Additional information about the error
        """
        error_record = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'severity': severity.value,
            'traceback': traceback.format_exc(),
            'additional_info': additional_info or {}
        }
        
        # Add to history
        self.error_history.append(error_record)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR in {context}: {error}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR in {context}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR in {context}: {error}")
        else:
            logger.info(f"LOW SEVERITY ERROR in {context}: {error}")
        
        # Check if we should send alerts
        self._check_alert_thresholds(severity)
    
    def _check_alert_thresholds(self, severity: ErrorSeverity) -> None:
        """Check if error count exceeds alert thresholds."""
        # Count recent errors of this severity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = [
            err for err in self.error_history
            if err['timestamp'] > one_hour_ago and err['severity'] == severity.value
        ]
        
        threshold = self.alert_thresholds.get(severity, 5)
        
        if len(recent_errors) >= threshold:
            logger.critical(f"ALERT: {len(recent_errors)} {severity.value} errors in the last hour (threshold: {threshold})")
            # Here you could implement actual alerting (email, Slack, etc.)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of errors in the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with error summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [err for err in self.error_history if err['timestamp'] > cutoff_time]
        
        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = len([
                err for err in recent_errors if err['severity'] == severity.value
            ])
        
        # Count by context
        context_counts = {}
        for error in recent_errors:
            context = error['context']
            context_counts[context] = context_counts.get(context, 0) + 1
        
        # Count by error type
        error_type_counts = {}
        for error in recent_errors:
            error_type = error['error_type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'severity_breakdown': severity_counts,
            'context_breakdown': context_counts,
            'error_type_breakdown': error_type_counts,
            'most_recent_error': recent_errors[-1] if recent_errors else None
        }


def retry_with_backoff(
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exceptions: tuple = (Exception,),
    error_handler: Optional[ErrorHandler] = None,
    context: str = "unknown"
):
    """Decorator for retrying functions with configurable backoff strategies.
    
    Args:
        max_retries: Maximum number of retry attempts
        strategy: Retry strategy to use
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        error_handler: ErrorHandler instance for logging
        context: Context string for error logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        if error_handler:
                            severity = ErrorSeverity.HIGH if max_retries > 1 else ErrorSeverity.MEDIUM
                            error_handler.log_error(
                                e, 
                                f"{context} (final attempt {attempt + 1}/{max_retries + 1})",
                                severity,
                                {'function': func.__name__, 'attempt': attempt + 1}
                            )
                        raise e
                    
                    # Calculate delay for next attempt
                    if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    elif strategy == RetryStrategy.LINEAR_BACKOFF:
                        delay = min(base_delay * (attempt + 1), max_delay)
                    else:  # FIXED_INTERVAL
                        delay = base_delay
                    
                    # Log retry attempt
                    if error_handler:
                        error_handler.log_error(
                            e,
                            f"{context} (attempt {attempt + 1}/{max_retries + 1}, retrying in {delay:.1f}s)",
                            ErrorSeverity.LOW,
                            {'function': func.__name__, 'attempt': attempt + 1, 'delay': delay}
                        )
                    else:
                        logger.warning(f"Retry attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for failing services."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: type = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that triggers the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                    logger.info(f"Circuit breaker for {func.__name__} is HALF_OPEN, attempting recovery")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        if self.state == 'HALF_OPEN':
            logger.info("Circuit breaker recovery successful, closing circuit")
        
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")


class GracefulDegradation:
    """Implements graceful degradation for system components."""
    
    def __init__(self):
        """Initialize graceful degradation handler."""
        self.degraded_services = set()
        self.fallback_strategies = {}
    
    def register_fallback(self, service_name: str, fallback_func: Callable) -> None:
        """Register a fallback function for a service.
        
        Args:
            service_name: Name of the service
            fallback_func: Function to call when service is degraded
        """
        self.fallback_strategies[service_name] = fallback_func
        logger.info(f"Registered fallback strategy for {service_name}")
    
    def mark_degraded(self, service_name: str, reason: str = "") -> None:
        """Mark a service as degraded.
        
        Args:
            service_name: Name of the service
            reason: Reason for degradation
        """
        self.degraded_services.add(service_name)
        logger.warning(f"Service {service_name} marked as degraded: {reason}")
    
    def mark_recovered(self, service_name: str) -> None:
        """Mark a service as recovered.
        
        Args:
            service_name: Name of the service
        """
        if service_name in self.degraded_services:
            self.degraded_services.remove(service_name)
            logger.info(f"Service {service_name} marked as recovered")
    
    def is_degraded(self, service_name: str) -> bool:
        """Check if a service is currently degraded.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is degraded, False otherwise
        """
        return service_name in self.degraded_services
    
    def execute_with_fallback(self, service_name: str, primary_func: Callable, 
                             *args, **kwargs) -> Any:
        """Execute function with fallback if service is degraded.
        
        Args:
            service_name: Name of the service
            primary_func: Primary function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result from primary or fallback function
        """
        if not self.is_degraded(service_name):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed for {service_name}: {e}")
                self.mark_degraded(service_name, str(e))
        
        # Use fallback if available
        if service_name in self.fallback_strategies:
            logger.info(f"Using fallback strategy for {service_name}")
            return self.fallback_strategies[service_name](*args, **kwargs)
        else:
            raise Exception(f"No fallback available for degraded service: {service_name}")


# Global instances
error_handler = ErrorHandler()
graceful_degradation = GracefulDegradation()