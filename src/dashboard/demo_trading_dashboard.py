"""Demo script for the trading dashboard."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from datetime import date, timedelta
import matplotlib.pyplot as plt
from loguru import logger

from src.dashboard.trading_dashboard import TradingDashboard
from src.utils.data_manager import DataManager


def demo_trading_dashboard():
    """Demonstrate the trading dashboard functionality."""
    print("=" * 60)
    print("TRADING DASHBOARD DEMO")
    print("=" * 60)
    
    try:
        # Initialize dashboard
        print("\n1. Initializing trading dashboard...")
        dashboard = TradingDashboard()
        
        # Check data availability
        data_manager = DataManager()
        print("\n2. Checking prediction data availability...")
        
        try:
            predictions_df = data_manager.load_predictions()
            print(f"   Predictions available: {len(predictions_df)} records")
            if not predictions_df.empty:
                print(f"   Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        except Exception as e:
            print(f"   No predictions data: {e}")
        
        # Get latest recommendations
        print("\n3. Getting latest trading recommendations...")
        recommendations = dashboard.get_latest_recommendations()
        
        if recommendations:
            print(f"   Generated {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations[:5]):  # Show first 5
                print(f"     {i+1}. {rec.get('contract_type', 'Unknown')}: {rec.get('recommendation', 'N/A')} "
                      f"(EV: ${rec.get('expected_value', 0):.3f}, Confidence: {rec.get('confidence', 0):.1f}%)")
        else:
            print("   No recommendations generated")
        
        # Analyze contract opportunities
        print("\n4. Analyzing contract opportunities...")
        target_date = date.today()
        contract_analysis = dashboard.analyze_contract_opportunities(target_date)
        
        if 'error' not in contract_analysis:
            print(f"   Contract analysis completed for {target_date}")
            print(f"   Predicted temperature: {contract_analysis.get('predicted_temp', 'N/A'):.1f}°F")
            print(f"   Confidence: {contract_analysis.get('confidence', 0):.1f}%")
            
            contracts = contract_analysis.get('contracts', {})
            positive_ev_contracts = [name for name, data in contracts.items() 
                                   if data.get('expected_value', 0) > 0]
            print(f"   Positive EV opportunities: {len(positive_ev_contracts)}")
            
            if positive_ev_contracts:
                print("   Best opportunities:")
                for contract_name in positive_ev_contracts[:3]:  # Show top 3
                    contract_data = contracts[contract_name]
                    print(f"     - {contract_name}: EV=${contract_data.get('expected_value', 0):.3f}, "
                          f"Win Prob: {contract_data.get('win_probability', 0):.3f}")
        else:
            print(f"   Error in contract analysis: {contract_analysis['error']}")
        
        # Calculate position sizing
        print("\n5. Calculating position sizing...")
        bankroll = 10000  # $10,000 bankroll
        
        if recommendations:
            sized_recommendations = dashboard.calculate_position_sizing(recommendations, bankroll)
            
            total_position = sum(r.get('position_size_dollars', 0) for r in sized_recommendations)
            buy_positions = [r for r in sized_recommendations if r.get('position_size_dollars', 0) > 0]
            
            print(f"   Bankroll: ${bankroll:,}")
            print(f"   Total position size: ${total_position:.2f}")
            print(f"   Portfolio allocation: {(total_position/bankroll)*100:.1f}%")
            print(f"   Number of positions: {len(buy_positions)}")
            
            if buy_positions:
                print("   Position details:")
                for pos in buy_positions[:3]:  # Show top 3 positions
                    print(f"     - {pos.get('contract_type', 'Unknown')}: ${pos.get('position_size_dollars', 0):.0f} "
                          f"({pos.get('position_size_contracts', 0)} contracts)")
        else:
            print("   No recommendations to size")
        
        # Generate comprehensive report
        print("\n6. Generating comprehensive trading report...")
        report = dashboard.generate_trading_report(target_date, bankroll)
        
        if 'error' not in report:
            print(f"   Report generated for: {report['date']}")
            
            summary = report.get('summary', {})
            print(f"   Total recommendations: {summary.get('total_recommendations', 0)}")
            print(f"   Buy recommendations: {summary.get('buy_recommendations', 0)}")
            print(f"   Recommendation strength: {summary.get('recommendation_strength', 'Unknown')}")
            
            risk_metrics = report.get('risk_metrics', {})
            if risk_metrics:
                print(f"   Portfolio allocation: {risk_metrics.get('portfolio_allocation', 0)*100:.1f}%")
                print(f"   Max portfolio loss: {risk_metrics.get('max_portfolio_loss', 0)*100:.1f}%")
                print(f"   Risk-reward ratio: {risk_metrics.get('risk_reward_ratio', 0):.2f}")
            
            best_opportunity = summary.get('highest_ev_contract')
            if best_opportunity:
                print(f"   Best opportunity: {best_opportunity.get('contract_type', 'Unknown')} "
                      f"(EV: ${best_opportunity.get('expected_value', 0):.3f})")
        else:
            print(f"   Error generating report: {report['error']}")
        
        # Create and save plots
        print("\n7. Creating trading dashboard plots...")
        try:
            # Create recommendations plot
            if recommendations:
                print("   Creating recommendations plot...")
                sized_recs = dashboard.calculate_position_sizing(recommendations, bankroll)
                fig1 = dashboard.create_recommendations_plot(sized_recs)
                plt.show(block=False)
                plt.pause(2)  # Show for 2 seconds
                plt.close(fig1)
            
            # Create contract analysis plot
            if 'error' not in contract_analysis:
                print("   Creating contract analysis plot...")
                fig2 = dashboard.create_contract_analysis_plot(contract_analysis)
                plt.show(block=False)
                plt.pause(2)  # Show for 2 seconds
                plt.close(fig2)
            
            print("   Plots created successfully!")
            
        except Exception as e:
            print(f"   Error creating plots: {e}")
        
        # Save plots to files
        print("\n8. Saving trading dashboard plots to files...")
        try:
            saved_files = dashboard.save_trading_plots()
            if saved_files:
                print("   Plots saved:")
                for plot_name, file_path in saved_files.items():
                    print(f"     {plot_name}: {file_path}")
            else:
                print("   No plots were saved")
        except Exception as e:
            print(f"   Error saving plots: {e}")
        
        print("\n" + "=" * 60)
        print("TRADING DASHBOARD DEMO COMPLETED")
        print("=" * 60)
        
        # Summary
        print("\nTrading Dashboard Features Demonstrated:")
        print("✓ Trading recommendation generation")
        print("✓ Contract opportunity analysis")
        print("✓ Position sizing with Kelly criterion")
        print("✓ Risk-reward analysis")
        print("✓ Expected value calculations")
        print("✓ Portfolio risk metrics")
        print("✓ Comprehensive trading reports")
        print("✓ Trading visualization plots")
        
    except Exception as e:
        logger.error(f"Error in trading dashboard demo: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_trading_dashboard()