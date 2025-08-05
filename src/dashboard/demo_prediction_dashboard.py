"""Demo script for the prediction dashboard."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import matplotlib.pyplot as plt
from loguru import logger

from src.dashboard.prediction_dashboard import PredictionDashboard
from src.utils.data_manager import DataManager


def demo_prediction_dashboard():
    """Demonstrate the prediction dashboard functionality."""
    print("=" * 60)
    print("PREDICTION DASHBOARD DEMO")
    print("=" * 60)
    
    try:
        # Initialize dashboard
        print("\n1. Initializing prediction dashboard...")
        dashboard = PredictionDashboard()
        
        # Check data availability
        data_manager = DataManager()
        print("\n2. Checking data availability...")
        
        # Check predictions data
        try:
            predictions_df = data_manager.load_predictions()
            print(f"   Predictions available: {len(predictions_df)} records")
            if not predictions_df.empty:
                print(f"   Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        except Exception as e:
            print(f"   No predictions data: {e}")
        
        # Check actual temperatures
        try:
            actuals_df = data_manager.load_actual_temperatures()
            print(f"   Actual temperatures: {len(actuals_df)} records")
            if not actuals_df.empty:
                print(f"   Date range: {actuals_df['date'].min()} to {actuals_df['date'].max()}")
        except Exception as e:
            print(f"   No actual temperature data: {e}")
        
        # Get latest prediction
        print("\n3. Getting latest prediction...")
        latest_prediction = dashboard.get_latest_prediction()
        if latest_prediction:
            print(f"   Date: {latest_prediction['date']}")
            print(f"   Predicted High: {latest_prediction['predicted_high']:.1f}°F")
            print(f"   Confidence: {latest_prediction['confidence']:.1f}%")
            print(f"   Created: {latest_prediction['created_at']}")
        else:
            print("   No predictions available")
        
        # Get source contributions
        print("\n4. Getting source contributions for today...")
        target_date = date.today()
        contributions = dashboard.get_source_contributions(target_date)
        if contributions:
            print(f"   Source contributions for {target_date}:")
            for source, temp in contributions.items():
                print(f"     {source}: {temp:.1f}°F")
        else:
            print("   No source contributions available")
        
        # Generate comprehensive report
        print("\n5. Generating prediction report...")
        report = dashboard.generate_prediction_report()
        
        if 'error' not in report:
            print(f"   Report generated for: {report['date']}")
            print(f"   Generated at: {report['generated_at']}")
            
            if report['prediction']:
                pred = report['prediction']
                print(f"   Prediction: {pred['predicted_high']:.1f}°F (confidence: {pred['confidence']:.1f}%)")
            
            if report['source_contributions']:
                print(f"   Sources contributing: {len(report['source_contributions'])}")
            
            if report['recommendations']:
                print("   Recommendations:")
                for rec in report['recommendations']:
                    print(f"     - {rec}")
        else:
            print(f"   Error generating report: {report['error']}")
        
        # Create and save dashboard plots
        print("\n6. Creating dashboard plots...")
        try:
            # Create prediction summary plot
            print("   Creating prediction summary plot...")
            fig1 = dashboard.create_prediction_summary_plot(days_back=14)
            plt.show(block=False)
            plt.pause(2)  # Show for 2 seconds
            plt.close(fig1)
            
            # Create source contributions plot
            print("   Creating source contributions plot...")
            fig2 = dashboard.create_source_contributions_plot(target_date)
            plt.show(block=False)
            plt.pause(2)  # Show for 2 seconds
            plt.close(fig2)
            
            # Create accuracy trends plot
            print("   Creating accuracy trends plot...")
            fig3 = dashboard.create_accuracy_trends_plot(days_back=30)
            plt.show(block=False)
            plt.pause(2)  # Show for 2 seconds
            plt.close(fig3)
            
            print("   Plots created successfully!")
            
        except Exception as e:
            print(f"   Error creating plots: {e}")
        
        # Save plots to files
        print("\n7. Saving dashboard plots to files...")
        try:
            saved_files = dashboard.save_dashboard_plots()
            if saved_files:
                print("   Plots saved:")
                for plot_name, file_path in saved_files.items():
                    print(f"     {plot_name}: {file_path}")
            else:
                print("   No plots were saved")
        except Exception as e:
            print(f"   Error saving plots: {e}")
        
        print("\n" + "=" * 60)
        print("PREDICTION DASHBOARD DEMO COMPLETED")
        print("=" * 60)
        
        # Summary
        print("\nDashboard Features Demonstrated:")
        print("✓ Latest prediction retrieval")
        print("✓ Source contribution analysis")
        print("✓ Prediction vs actual comparison plots")
        print("✓ Historical accuracy trends")
        print("✓ Confidence calibration analysis")
        print("✓ Comprehensive prediction reports")
        print("✓ Plot generation and saving")
        
    except Exception as e:
        logger.error(f"Error in prediction dashboard demo: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_prediction_dashboard()