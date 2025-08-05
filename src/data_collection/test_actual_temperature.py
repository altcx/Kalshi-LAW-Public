"""Test script for actual temperature collection functionality."""

import sys
from pathlib import Path
from datetime import date, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.actual_temperature_collector import ActualTemperatureCollector
from src.data_collection.noaa_observations_client import NOAAObservationsClient
from src.utils.data_manager import DataManager
from loguru import logger


def test_noaa_client():
    """Test NOAA observations client."""
    print("\n=== Testing NOAA Observations Client ===")
    
    client = NOAAObservationsClient()
    
    # Test connection
    print("Testing connection...")
    if client.test_connection():
        print("✓ Connection test passed")
    else:
        print("✗ Connection test failed")
        return False
    
    # Test station info
    print("Getting station info...")
    station_info = client.get_station_info()
    if station_info:
        print(f"✓ Station: {station_info['name']}")
        print(f"  Elevation: {station_info['elevation']}m")
        print(f"  Timezone: {station_info['timezone']}")
    else:
        print("✗ Failed to get station info")
        return False
    
    # Test temperature collection for yesterday
    yesterday = date.today() - timedelta(days=1)
    print(f"Getting temperature for {yesterday}...")
    temp = client.get_daily_high_temperature(yesterday)
    if temp is not None:
        print(f"✓ Temperature: {temp:.1f}°F")
    else:
        print("✗ Failed to get temperature (may be normal if data not available yet)")
    
    return True


def test_data_manager():
    """Test data manager actual temperature functionality."""
    print("\n=== Testing Data Manager ===")
    
    data_manager = DataManager()
    test_date = date.today() - timedelta(days=3)
    test_temp = 78.5
    
    # Test storing actual temperature
    print(f"Storing test temperature: {test_temp}°F for {test_date}")
    data_manager.store_actual_temperature(test_date, test_temp, source="TEST")
    
    # Test retrieving actual temperature
    print("Retrieving stored temperature...")
    retrieved_temp = data_manager.get_actual_temperature(test_date)
    if retrieved_temp == test_temp:
        print(f"✓ Retrieved temperature: {retrieved_temp:.1f}°F")
    else:
        print(f"✗ Temperature mismatch: expected {test_temp}, got {retrieved_temp}")
        return False
    
    # Test validation
    print("Testing temperature validation...")
    valid_temp = data_manager.validate_actual_temperature(75.0, test_date)
    invalid_temp = data_manager.validate_actual_temperature(150.0, test_date)
    
    if valid_temp and not invalid_temp:
        print("✓ Temperature validation working correctly")
    else:
        print("✗ Temperature validation failed")
        return False
    
    return True


def test_temperature_collector():
    """Test actual temperature collector."""
    print("\n=== Testing Temperature Collector ===")
    
    collector = ActualTemperatureCollector()
    
    # Test system
    print("Testing collection system...")
    if collector.test_collection_system():
        print("✓ Collection system test passed")
    else:
        print("✗ Collection system test failed")
        return False
    
    # Test collecting recent temperatures
    print("Testing recent temperature collection...")
    recent_temps = collector.collect_recent_temperatures(days=3)
    
    successful_collections = sum(1 for temp in recent_temps.values() if temp is not None)
    print(f"✓ Collected {successful_collections}/{len(recent_temps)} recent temperatures")
    
    for date_key, temp in recent_temps.items():
        if temp is not None:
            print(f"  {date_key}: {temp:.1f}°F")
        else:
            print(f"  {date_key}: No data available")
    
    # Test collection status
    print("Getting collection status...")
    status = collector.get_temperature_collection_status(days=7)
    if 'error' not in status:
        print(f"✓ Collection rate: {status['collection_rate']:.1%}")
        print(f"  Collected: {status['total_collected']}/{status['total_expected']}")
        if status['missing_dates']:
            print(f"  Missing dates: {', '.join(status['missing_dates'][:5])}")
    else:
        print(f"✗ Error getting status: {status['error']}")
    
    # Test data quality validation
    print("Validating data quality...")
    quality = collector.validate_temperature_data_quality(days=7)
    if 'error' not in quality:
        print(f"✓ Average temperature: {quality.get('avg_temperature', 0):.1f}°F")
        print(f"  Temperature range: {quality.get('temperature_range', 0):.1f}°F")
        print(f"  Outliers: {quality.get('outlier_count', 0)} ({quality.get('outlier_percentage', 0):.1f}%)")
    else:
        print(f"✗ Error validating quality: {quality['error']}")
    
    return True


def main():
    """Run all tests."""
    print("Starting actual temperature collection tests...")
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    tests = [
        ("NOAA Client", test_noaa_client),
        ("Data Manager", test_data_manager),
        ("Temperature Collector", test_temperature_collector)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Results ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Actual temperature collection is working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()