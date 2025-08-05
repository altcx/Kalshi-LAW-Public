#!/usr/bin/env python3
"""Demo script for testing the daily scheduler automation system."""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.automation.daily_scheduler import DailyScheduler
from src.automation.service_manager import ServiceManager
from src.automation.error_handler import error_handler, ErrorSeverity


def setup_demo_logging():
    """Setup logging for demo."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def demo_morning_collection():
    """Demonstrate morning data collection."""
    print("\n" + "="*60)
    print("DEMO: Morning Data Collection")
    print("="*60)
    
    scheduler = DailyScheduler()
    
    print("Starting morning forecast collection from all APIs...")
    success = scheduler.morning_data_collection()
    
    if success:
        print("✅ Morning collection completed successfully!")
    else:
        print("❌ Morning collection failed")
    
    return success


def demo_evening_collection():
    """Demonstrate evening actual temperature collection."""
    print("\n" + "="*60)
    print("DEMO: Evening Actual Temperature Collection")
    print("="*60)
    
    scheduler = DailyScheduler()
    
    print("Starting actual temperature collection for yesterday...")
    success = scheduler.evening_actual_temperature_collection()
    
    if success:
        print("✅ Evening collection completed successfully!")
    else:
        print("❌ Evening collection failed")
    
    return success


def demo_health_check():
    """Demonstrate system health check."""
    print("\n" + "="*60)
    print("DEMO: System Health Check")
    print("="*60)
    
    scheduler = DailyScheduler()
    
    print("Performing comprehensive system health check...")
    health_status = scheduler.health_check()
    
    print(f"\nHealth Check Results:")
    print(f"  Overall Health: {health_status.get('overall_health', 'unknown')}")
    
    # API Connections
    api_connections = health_status.get('api_connections', {})
    healthy_apis = sum(1 for status in api_connections.values() if status)
    total_apis = len(api_connections)
    print(f"  API Health: {healthy_apis}/{total_apis} APIs responding")
    
    for api, status in api_connections.items():
        status_icon = "✅" if status else "❌"
        print(f"    {status_icon} {api}")
    
    # Collection Status
    collection_status = health_status.get('collection_status', {})
    if 'error' not in collection_status:
        collection_rate = collection_status.get('collection_rate', 0)
        print(f"  Temperature Collection: {collection_rate:.1%} success rate")
        
        missing_dates = collection_status.get('missing_dates', [])
        if missing_dates:
            print(f"  Missing Data: {len(missing_dates)} days")
            if len(missing_dates) <= 3:
                print(f"    Missing: {', '.join(missing_dates)}")
    
    return 'error' not in health_status


def demo_error_handling():
    """Demonstrate error handling and logging."""
    print("\n" + "="*60)
    print("DEMO: Error Handling System")
    print("="*60)
    
    print("Simulating various error scenarios...")
    
    # Simulate different types of errors
    try:
        raise ConnectionError("API connection failed")
    except Exception as e:
        error_handler.log_error(e, "API Connection Test", ErrorSeverity.HIGH)
    
    try:
        raise ValueError("Invalid temperature reading: -999°F")
    except Exception as e:
        error_handler.log_error(e, "Data Validation", ErrorSeverity.MEDIUM)
    
    try:
        raise TimeoutError("Request timed out after 30 seconds")
    except Exception as e:
        error_handler.log_error(e, "Network Request", ErrorSeverity.LOW)
    
    # Get error summary
    summary = error_handler.get_error_summary(hours=1)
    
    print(f"\nError Summary (last hour):")
    print(f"  Total Errors: {summary['total_errors']}")
    print(f"  By Severity:")
    for severity, count in summary['severity_breakdown'].items():
        if count > 0:
            print(f"    {severity.title()}: {count}")
    
    print(f"  By Context:")
    for context, count in summary['context_breakdown'].items():
        print(f"    {context}: {count}")
    
    return True


def demo_service_management():
    """Demonstrate service management capabilities."""
    print("\n" + "="*60)
    print("DEMO: Service Management")
    print("="*60)
    
    manager = ServiceManager()
    
    print("Checking service status...")
    status = manager.get_service_status()
    
    print(f"Service Status:")
    print(f"  Running: {'Yes' if status['running'] else 'No'}")
    
    if status['running']:
        print(f"  PID: {status['pid']}")
        print(f"  Uptime: {status['uptime']}")
        print(f"  Memory Usage: {status['memory_usage']:.1f} MB")
        print(f"  CPU Usage: {status['cpu_percent']:.1f}%")
    
    if status['last_log_entry']:
        print(f"  Last Log: {status['last_log_entry']}")
    
    # Test running individual tasks
    print("\nTesting individual task execution...")
    
    tasks_to_test = ["health"]  # Start with health check as it's safest
    
    for task in tasks_to_test:
        print(f"  Running {task} task...")
        success = manager.run_task_once(task)
        print(f"    {'✅ Success' if success else '❌ Failed'}")
    
    return True


def demo_retry_logic():
    """Demonstrate retry logic and backoff strategies."""
    print("\n" + "="*60)
    print("DEMO: Retry Logic and Backoff")
    print("="*60)
    
    from src.automation.error_handler import retry_with_backoff, RetryStrategy
    
    # Simulate a function that fails a few times then succeeds
    attempt_count = 0
    
    @retry_with_backoff(
        max_retries=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.5,
        error_handler=error_handler,
        context="Demo Retry Test"
    )
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < 3:
            raise ConnectionError(f"Simulated failure (attempt {attempt_count})")
        
        return f"Success on attempt {attempt_count}!"
    
    print("Testing retry logic with exponential backoff...")
    try:
        result = flaky_function()
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Final failure: {e}")
    
    # Show error summary
    summary = error_handler.get_error_summary(hours=1)
    recent_errors = [
        err for err in error_handler.error_history
        if err['context'] == "Demo Retry Test"
    ]
    
    print(f"\nRetry attempts logged: {len(recent_errors)}")
    for i, err in enumerate(recent_errors, 1):
        print(f"  Attempt {i}: {err['error_message']}")
    
    return True


def demo_data_quality_monitoring():
    """Demonstrate data quality monitoring."""
    print("\n" + "="*60)
    print("DEMO: Data Quality Monitoring")
    print("="*60)
    
    scheduler = DailyScheduler()
    
    print("Checking data quality for all sources...")
    
    sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
    
    for source in sources:
        quality_summary = scheduler.data_manager.get_data_quality_summary(source, days=7)
        
        if 'error' in quality_summary:
            print(f"  ❌ {source}: {quality_summary['error']}")
        else:
            avg_quality = quality_summary.get('avg_quality_score', 0)
            total_records = quality_summary.get('total_records', 0)
            low_quality_count = quality_summary.get('low_quality_count', 0)
            
            quality_icon = "✅" if avg_quality >= 0.8 else "⚠️" if avg_quality >= 0.6 else "❌"
            
            print(f"  {quality_icon} {source}:")
            print(f"    Average Quality: {avg_quality:.3f}")
            print(f"    Total Records: {total_records}")
            print(f"    Low Quality Records: {low_quality_count}")
    
    return True


def run_comprehensive_demo():
    """Run comprehensive demo of all scheduler features."""
    print("🚀 Daily Scheduler Automation System Demo")
    print("=" * 80)
    
    setup_demo_logging()
    
    demos = [
        ("Health Check", demo_health_check),
        ("Error Handling", demo_error_handling),
        ("Retry Logic", demo_retry_logic),
        ("Data Quality Monitoring", demo_data_quality_monitoring),
        ("Service Management", demo_service_management),
        ("Morning Collection", demo_morning_collection),
        ("Evening Collection", demo_evening_collection),
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
        print("The daily scheduler automation system is ready for production use.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demos failed.")
        print("Please review the errors above and fix any issues.")
    
    return successful_demos == total_demos


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)