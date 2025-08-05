"""Demo script for the backtesting dashboard."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import matplotlib.pyplot as plt
from loguru import logger

from src.dashboard.backtesting_dashboard import BacktestingDashboard
from src.utils.data_manager import DataManager


def demo_backtesting_dashboard():
    """Demonstrate the backtesting dashboard functionality."""
    print("=" * 60)
    print("BACKTESTING DASHBOARD DEMO")
    print("=" * 60)
    
    try:
        # Initialize dashboard
        print("\n1. Initializing backtesting dashboard...")
        dashboard = BacktestingDashboard()
        
        # Set up date range for testing
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # 30-day backtest
        
        print(f"   Test period: {start_date} to {end_date}")
        
        # Run custom backtest
        print("\n2. Running custom backtest...")
        custom_params = {
            'models': ['ensemble', 'xgboost'],
            'retraining_frequency': 7,
            'min_training_days': 30,
            'confidence_threshold': 0.7,
            'trading_enabled': True,
            'initial_bankroll': 10000
        }
        
        backtest_results = dashboard.run_custom_backtest(start_date, end_date, custom_params)
        
        if 'error' not in backtest_results:
            predictions = backtest_results.get('predictions', [])
            print(f"   Backtest completed: {len(predictions)} predictions generated")
            
            performance_metrics = backtest_results.get('performance_metrics', {})
            if performance_metrics:
                print(f"   Performance metrics:")
                print(f"     MAE: {performance_metrics.get('mae', 0):.2f}°F")
                print(f"     RMSE: {performance_metrics.get('rmse', 0):.2f}°F")
                print(f"     Accuracy (±3°F): {performance_metrics.get('accuracy_3f', 0):.3f}")
            
            trading_metrics = backtest_results.get('trading_metrics', {})
            if trading_metrics:
                print(f"   Trading metrics:")
                print(f"     Total return: {trading_metrics.get('total_return', 0)*100:.1f}%")
                print(f"     Number of trades: {trading_metrics.get('num_trades', 0)}")
                print(f"     Win rate: {trading_metrics.get('win_rate', 0)*100:.1f}%")
                print(f"     Sharpe ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"   Backtest failed: {backtest_results['error']}")
        
        # Compare models
        print("\n3. Comparing model performance...")
        models_to_compare = ['ensemble', 'xgboost', 'lightgbm']
        comparison_results = dashboard.compare_models(start_date, end_date, models_to_compare)
        
        if 'error' not in comparison_results:
            model_results = comparison_results.get('model_results', {})
            print(f"   Model comparison completed for {len(model_results)} models:")
            
            for model_name, metrics in model_results.items():
                print(f"     {model_name}:")
                print(f"       Accuracy (±3°F): {metrics.get('accuracy_3f', 0):.3f}")
                print(f"       MAE: {metrics.get('mae', 0):.2f}°F")
                print(f"       RMSE: {metrics.get('rmse', 0):.2f}°F")
        else:
            print(f"   Model comparison failed: {comparison_results['error']}")
        
        # Strategy optimization
        print("\n4. Running strategy optimization...")
        optimization_params = {
            'confidence_thresholds': [0.6, 0.7, 0.8],
            'position_sizes': [0.02, 0.05, 0.10],
            'retraining_frequencies': [7, 14],
            'objective': 'sharpe_ratio'
        }
        
        optimization_results = dashboard.optimize_strategy(start_date, end_date, optimization_params)
        
        if 'error' not in optimization_results:
            best_params = optimization_results.get('best_parameters', {})
            best_score = optimization_results.get('best_score', 0)
            num_combinations = len(optimization_results.get('optimization_results', []))
            
            print(f"   Optimization completed: tested {num_combinations} parameter combinations")
            print(f"   Best score ({optimization_params['objective']}): {best_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"     {param}: {value}")
        else:
            print(f"   Strategy optimization failed: {optimization_results['error']}")
        
        # What-if analysis
        print("\n5. Running what-if analysis...")
        base_scenario = {
            'start_date': start_date,
            'end_date': end_date,
            'parameters': {
                'confidence_threshold': 0.7,
                'position_size': 0.05,
                'trading_enabled': True,
                'initial_bankroll': 10000
            }
        }
        
        variations = [
            {
                'name': 'High Confidence',
                'parameter_changes': {'confidence_threshold': 0.9}
            },
            {
                'name': 'Large Position',
                'parameter_changes': {'position_size': 0.10}
            },
            {
                'name': 'Conservative',
                'parameter_changes': {'confidence_threshold': 0.8, 'position_size': 0.02}
            }
        ]
        
        whatif_results = dashboard.run_what_if_analysis(base_scenario, variations)
        
        if 'error' not in whatif_results:
            results = whatif_results.get('results', [])
            print(f"   What-if analysis completed: {len(results)} scenarios tested")
            
            for result in results:
                scenario_name = result['scenario']
                scenario_result = result['result']
                
                if 'error' not in scenario_result:
                    trading_metrics = scenario_result.get('trading_metrics', {})
                    if trading_metrics:
                        total_return = trading_metrics.get('total_return', 0) * 100
                        win_rate = trading_metrics.get('win_rate', 0) * 100
                        print(f"     {scenario_name}: Return {total_return:.1f}%, Win Rate {win_rate:.1f}%")
                else:
                    print(f"     {scenario_name}: Failed - {scenario_result['error']}")
        else:
            print(f"   What-if analysis failed: {whatif_results['error']}")
        
        # Create and display plots
        print("\n6. Creating backtesting dashboard plots...")
        try:
            # Create backtest results plot
            if 'error' not in backtest_results:
                print("   Creating backtest results plot...")
                fig1 = dashboard.create_backtest_results_plot(backtest_results)
                plt.show(block=False)
                plt.pause(2)  # Show for 2 seconds
                plt.close(fig1)
            
            # Create model comparison plot
            if 'error' not in comparison_results:
                print("   Creating model comparison plot...")
                fig2 = dashboard.create_model_comparison_plot(comparison_results)
                plt.show(block=False)
                plt.pause(2)  # Show for 2 seconds
                plt.close(fig2)
            
            # Create optimization results plot
            if 'error' not in optimization_results:
                print("   Creating optimization results plot...")
                fig3 = dashboard.create_optimization_results_plot(optimization_results)
                plt.show(block=False)
                plt.pause(2)  # Show for 2 seconds
                plt.close(fig3)
            
            print("   Plots created successfully!")
            
        except Exception as e:
            print(f"   Error creating plots: {e}")
        
        # Save plots to files
        print("\n7. Saving backtesting dashboard plots to files...")
        try:
            saved_files = dashboard.save_backtesting_plots()
            if saved_files:
                print("   Plots saved:")
                for plot_name, file_path in saved_files.items():
                    print(f"     {plot_name}: {file_path}")
            else:
                print("   No plots were saved")
        except Exception as e:
            print(f"   Error saving plots: {e}")
        
        print("\n" + "=" * 60)
        print("BACKTESTING DASHBOARD DEMO COMPLETED")
        print("=" * 60)
        
        # Summary
        print("\nBacktesting Dashboard Features Demonstrated:")
        print("✓ Custom backtest execution with parameters")
        print("✓ Model performance comparison")
        print("✓ Strategy parameter optimization")
        print("✓ What-if scenario analysis")
        print("✓ Performance metrics calculation")
        print("✓ Trading simulation and metrics")
        print("✓ Comprehensive visualization plots")
        print("✓ Results export and saving")
        
    except Exception as e:
        logger.error(f"Error in backtesting dashboard demo: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_backtesting_dashboard()