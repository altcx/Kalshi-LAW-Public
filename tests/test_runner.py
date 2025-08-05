"""
Comprehensive test runner for the Kalshi Weather Predictor system.
This module provides a centralized way to run all unit tests and generate coverage reports.
"""

import sys
import os
import unittest
import importlib
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test discovery patterns
TEST_PATTERNS = [
    'src/*/test_*.py',
    'tests/test_*.py'
]

class TestResult:
    """Container for test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.execution_time = 0.0
        self.failures = []
        self.errors = []
        self.module_results = {}

class ComprehensiveTestRunner:
    """Comprehensive test runner for all system components."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_modules = []
        self.results = TestResult()
    
    def discover_test_modules(self) -> List[str]:
        """Discover all test modules in the project."""
        test_files = []
        
        # Find all test files
        for pattern in TEST_PATTERNS:
            test_files.extend(self.project_root.glob(pattern))
        
        # Convert to module names
        modules = []
        for test_file in test_files:
            if test_file.name.startswith('test_') and test_file.suffix == '.py':
                # Convert file path to module name
                relative_path = test_file.relative_to(self.project_root)
                module_name = str(relative_path).replace('/', '.').replace('\\', '.')[:-3]
                modules.append(module_name)
        
        return sorted(modules)
    
    def run_module_tests(self, module_name: str) -> Dict[str, Any]:
        """Run tests for a specific module."""
        print(f"\n--- Running tests for {module_name} ---")
        
        try:
            # Import the test module
            test_module = importlib.import_module(module_name)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            runner = unittest.TextTestRunner(
                verbosity=1,
                stream=sys.stdout,
                buffer=True
            )
            
            start_time = time.time()
            result = runner.run(suite)
            execution_time = time.time() - start_time
            
            # Collect results
            module_result = {
                'module': module_name,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success': result.wasSuccessful(),
                'execution_time': execution_time,
                'failure_details': result.failures,
                'error_details': result.errors
            }
            
            return module_result
            
        except Exception as e:
            print(f"Error running tests for {module_name}: {e}")
            return {
                'module': module_name,
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'execution_time': 0.0,
                'failure_details': [],
                'error_details': [('Module Import Error', str(e))]
            }
    
    def run_all_tests(self) -> TestResult:
        """Run all discovered tests."""
        print("=" * 80)
        print("KALSHI WEATHER PREDICTOR - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Discover test modules
        self.test_modules = self.discover_test_modules()
        print(f"\nDiscovered {len(self.test_modules)} test modules:")
        for module in self.test_modules:
            print(f"  - {module}")
        
        # Run tests for each module
        start_time = time.time()
        
        for module_name in self.test_modules:
            module_result = self.run_module_tests(module_name)
            self.results.module_results[module_name] = module_result
            
            # Update overall results
            self.results.total_tests += module_result['tests_run']
            if module_result['success']:
                self.results.passed_tests += module_result['tests_run'] - module_result['failures'] - module_result['errors']
            self.results.failed_tests += module_result['failures']
            self.results.error_tests += module_result['errors']
            self.results.skipped_tests += module_result['skipped']
            
            # Collect failure and error details
            self.results.failures.extend(module_result['failure_details'])
            self.results.errors.extend(module_result['error_details'])
        
        self.results.execution_time = time.time() - start_time
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"Total Tests Run: {self.results.total_tests}")
        print(f"Passed: {self.results.passed_tests}")
        print(f"Failed: {self.results.failed_tests}")
        print(f"Errors: {self.results.error_tests}")
        print(f"Skipped: {self.results.skipped_tests}")
        print(f"Execution Time: {self.results.execution_time:.2f} seconds")
        
        success_rate = (self.results.passed_tests / self.results.total_tests * 100) if self.results.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Module-by-module results
        print("\nMODULE RESULTS:")
        print("-" * 80)
        for module_name, result in self.results.module_results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"{status} {module_name:<40} ({result['tests_run']} tests, {result['execution_time']:.2f}s)")
        
        # Failure details
        if self.results.failures:
            print(f"\nFAILURE DETAILS ({len(self.results.failures)} failures):")
            print("-" * 80)
            for i, (test, traceback) in enumerate(self.results.failures[:5], 1):  # Show first 5
                print(f"{i}. {test}")
                print(f"   {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See full traceback'}")
            
            if len(self.results.failures) > 5:
                print(f"   ... and {len(self.results.failures) - 5} more failures")
        
        # Error details
        if self.results.errors:
            print(f"\nERROR DETAILS ({len(self.results.errors)} errors):")
            print("-" * 80)
            for i, (test, traceback) in enumerate(self.results.errors[:3], 1):  # Show first 3
                print(f"{i}. {test}")
                print(f"   {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See full traceback'}")
            
            if len(self.results.errors) > 3:
                print(f"   ... and {len(self.results.errors) - 3} more errors")
        
        print("\n" + "=" * 80)
        
        overall_success = self.results.failed_tests == 0 and self.results.error_tests == 0
        if overall_success:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED - Review failures and errors above")
        
        print("=" * 80)
        
        return overall_success
    
    def run_component_tests(self, component: str) -> bool:
        """Run tests for a specific component."""
        component_modules = [m for m in self.test_modules if component in m]
        
        if not component_modules:
            print(f"No test modules found for component: {component}")
            return False
        
        print(f"Running tests for component: {component}")
        print(f"Found modules: {component_modules}")
        
        success = True
        for module_name in component_modules:
            result = self.run_module_tests(module_name)
            if not result['success']:
                success = False
        
        return success


def run_all_tests():
    """Run all tests and return success status."""
    runner = ComprehensiveTestRunner()
    results = runner.run_all_tests()
    success = runner.print_summary()
    return success


def run_component_tests(component: str):
    """Run tests for a specific component."""
    runner = ComprehensiveTestRunner()
    runner.test_modules = runner.discover_test_modules()
    success = runner.run_component_tests(component)
    return success


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kalshi Weather Predictor Test Runner')
    parser.add_argument('--component', '-c', help='Run tests for specific component')
    parser.add_argument('--list', '-l', action='store_true', help='List all test modules')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner()
    
    if args.list:
        modules = runner.discover_test_modules()
        print("Available test modules:")
        for module in modules:
            print(f"  - {module}")
        return
    
    if args.component:
        success = run_component_tests(args.component)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()