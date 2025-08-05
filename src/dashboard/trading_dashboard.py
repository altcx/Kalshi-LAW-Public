"""Trading recommendation dashboard for Kalshi weather futures."""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import seaborn as sns
from loguru import logger
from pathlib import Path

from src.utils.data_manager import DataManager
from src.utils.config import config
from src.trading.recommendation_engine import RecommendationEngine
from src.trading.kalshi_contract_analyzer import KalshiContractAnalyzer
from src.trading.position_sizing import PositionSizer


class TradingDashboard:
    """Dashboard for displaying Kalshi trading recommendations and analysis."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize trading dashboard.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.recommendation_engine = RecommendationEngine()
        self.contract_analyzer = KalshiContractAnalyzer()
        self.position_sizing = PositionSizer()
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Dashboard configuration
        self.figure_size = (15, 10)
        self.dpi = 100
        
        logger.info("TradingDashboard initialized")
    
    def get_latest_recommendations(self) -> List[Dict[str, Any]]:
        """Get the most recent trading recommendations.
        
        Returns:
            List of recommendation dictionaries
        """
        try:
            # Get latest prediction
            predictions_df = self.data_manager.load_predictions()
            if predictions_df.empty:
                logger.warning("No predictions available for recommendations")
                return []
            
            latest_prediction = predictions_df.loc[predictions_df['date'].idxmax()]
            predicted_temp = latest_prediction['predicted_high']
            confidence = latest_prediction['confidence']
            
            # Generate simple recommendations for common temperature thresholds
            recommendations = []
            temperature_thresholds = [75, 80, 85, 90, 95]
            
            for threshold in temperature_thresholds:
                # Analyze "above" contract
                above_prob = 1.0 / (1.0 + np.exp(-(predicted_temp - threshold) / 2.0))  # Sigmoid
                above_ev = above_prob * 1.0 - (1 - above_prob) * 0.5  # Assume 50 cent price
                
                if above_ev > 0.05:  # Only recommend if EV > 5 cents
                    recommendations.append({
                        'contract_type': f'LA_HIGH_TEMP_ABOVE_{threshold}F',
                        'recommendation': 'BUY',
                        'expected_value': above_ev,
                        'confidence': confidence,
                        'contract_price': 0.5,
                        'win_probability': above_prob,
                        'reasoning': f'Predicted temp {predicted_temp:.1f}°F has {above_prob:.1%} chance above {threshold}°F'
                    })
                
                # Analyze "below" contract
                below_prob = 1 - above_prob
                below_ev = below_prob * 1.0 - (1 - below_prob) * 0.5
                
                if below_ev > 0.05:  # Only recommend if EV > 5 cents
                    recommendations.append({
                        'contract_type': f'LA_HIGH_TEMP_BELOW_{threshold}F',
                        'recommendation': 'BUY',
                        'expected_value': below_ev,
                        'confidence': confidence,
                        'contract_price': 0.5,
                        'win_probability': below_prob,
                        'reasoning': f'Predicted temp {predicted_temp:.1f}°F has {below_prob:.1%} chance below {threshold}°F'
                    })
            
            # Sort by expected value (highest first)
            recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting latest recommendations: {e}")
            return []
    
    def analyze_contract_opportunities(self, target_date: date) -> Dict[str, Any]:
        """Analyze available Kalshi contract opportunities.
        
        Args:
            target_date: Date to analyze contracts for
            
        Returns:
            Dictionary with contract analysis data
        """
        try:
            # Get prediction for target date
            predictions_df = self.data_manager.load_predictions()
            if predictions_df.empty:
                return {'error': 'No predictions available'}
            
            # Find prediction for target date
            predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
            target_prediction = predictions_df[predictions_df['date'] == target_date]
            
            if target_prediction.empty:
                return {'error': f'No prediction available for {target_date}'}
            
            prediction_row = target_prediction.iloc[0]
            predicted_temp = prediction_row['predicted_high']
            confidence = prediction_row['confidence']
            
            # Analyze contracts for common temperature thresholds
            temperature_thresholds = [75, 80, 85, 90, 95]
            contract_analysis = {}
            
            for threshold in temperature_thresholds:
                # Analyze "above" contract
                above_analysis = self.contract_analyzer.analyze_temperature_contract(
                    predicted_temp=predicted_temp,
                    confidence=confidence,
                    threshold=threshold,
                    contract_type='above',
                    current_price=0.5  # Assume 50 cents as default
                )
                
                # Analyze "below" contract  
                below_analysis = self.contract_analyzer.analyze_temperature_contract(
                    predicted_temp=predicted_temp,
                    confidence=confidence,
                    threshold=threshold,
                    contract_type='below',
                    current_price=0.5  # Assume 50 cents as default
                )
                
                contract_analysis[f'above_{threshold}'] = above_analysis
                contract_analysis[f'below_{threshold}'] = below_analysis
            
            return {
                'target_date': target_date,
                'predicted_temp': predicted_temp,
                'confidence': confidence,
                'contracts': contract_analysis,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contract opportunities: {e}")
            return {'error': str(e)}
    
    def calculate_position_sizing(self, recommendations: List[Dict[str, Any]], 
                                 bankroll: float = 10000) -> List[Dict[str, Any]]:
        """Calculate position sizing for recommendations.
        
        Args:
            recommendations: List of trading recommendations
            bankroll: Available bankroll in dollars
            
        Returns:
            List of recommendations with position sizing added
        """
        try:
            sized_recommendations = []
            
            for rec in recommendations:
                if rec.get('recommendation') == 'BUY':
                    # Calculate position size based on Kelly criterion
                    edge = rec.get('expected_value', 0)
                    confidence = rec.get('confidence', 0.5)
                    
                    # Convert confidence to probability
                    win_probability = confidence / 100.0 if confidence > 1 else confidence
                    
                    # Create a mock ContractAnalysis for position sizing
                    # In a real implementation, this would come from the contract analyzer
                    from src.trading.kalshi_contract_analyzer import ContractAnalysis
                    
                    mock_analysis = ContractAnalysis(
                        contract_id=rec.get('contract_type', 'unknown'),
                        expected_value=edge,
                        win_probability=win_probability,
                        current_price=rec.get('contract_price', 0.5),
                        recommendation='BUY' if edge > 0 else 'HOLD',
                        confidence_score=confidence,
                        reasoning="Mock analysis for dashboard"
                    )
                    
                    # Calculate position size using PositionSizer
                    position_recommendation = self.position_sizing.calculate_position_size(
                        analysis=mock_analysis,
                        bankroll=bankroll,
                        confidence=confidence
                    )
                    
                    kelly_fraction = position_recommendation.kelly_fraction
                    position_size = position_recommendation.dollar_amount
                    
                    # Add position sizing to recommendation
                    rec_with_sizing = rec.copy()
                    rec_with_sizing.update({
                        'kelly_fraction': kelly_fraction,
                        'position_size_dollars': position_size,
                        'position_size_contracts': int(position_size / rec.get('contract_price', 0.5)),
                        'max_loss': position_size,
                        'max_gain': position_size * (1.0 / rec.get('contract_price', 0.5) - 1),
                        'risk_reward_ratio': (1.0 / rec.get('contract_price', 0.5) - 1) if rec.get('contract_price', 0.5) > 0 else 0
                    })
                    
                    sized_recommendations.append(rec_with_sizing)
                else:
                    # For non-buy recommendations, just add with zero position
                    rec_with_sizing = rec.copy()
                    rec_with_sizing.update({
                        'kelly_fraction': 0,
                        'position_size_dollars': 0,
                        'position_size_contracts': 0,
                        'max_loss': 0,
                        'max_gain': 0,
                        'risk_reward_ratio': 0
                    })
                    sized_recommendations.append(rec_with_sizing)
            
            return sized_recommendations
            
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return recommendations
    
    def create_recommendations_plot(self, recommendations: List[Dict[str, Any]]) -> Figure:
        """Create a plot showing trading recommendations.
        
        Args:
            recommendations: List of trading recommendations
            
        Returns:
            Matplotlib figure with recommendations plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            if not recommendations:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No recommendations available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            # Filter buy recommendations
            buy_recs = [r for r in recommendations if r.get('recommendation') == 'BUY']
            
            if not buy_recs:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No buy recommendations', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            # Plot 1: Expected Value by Contract
            contracts = [r.get('contract_type', 'Unknown') for r in buy_recs]
            expected_values = [r.get('expected_value', 0) for r in buy_recs]
            
            bars1 = ax1.bar(range(len(contracts)), expected_values, alpha=0.7)
            ax1.set_title('Expected Value by Contract', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Expected Value ($)', fontsize=12)
            ax1.set_xticks(range(len(contracts)))
            ax1.set_xticklabels(contracts, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add value labels
            for i, (bar, ev) in enumerate(zip(bars1, expected_values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'${ev:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 2: Position Sizing
            position_sizes = [r.get('position_size_dollars', 0) for r in buy_recs]
            
            bars2 = ax2.bar(range(len(contracts)), position_sizes, alpha=0.7, color='green')
            ax2.set_title('Recommended Position Sizes', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Position Size ($)', fontsize=12)
            ax2.set_xticks(range(len(contracts)))
            ax2.set_xticklabels(contracts, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, size) in enumerate(zip(bars2, position_sizes)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(position_sizes) * 0.01,
                        f'${size:.0f}', ha='center', va='bottom', fontsize=10)
            
            # Plot 3: Risk-Reward Analysis
            max_losses = [r.get('max_loss', 0) for r in buy_recs]
            max_gains = [r.get('max_gain', 0) for r in buy_recs]
            
            x_pos = np.arange(len(contracts))
            width = 0.35
            
            bars3a = ax3.bar(x_pos - width/2, max_losses, width, label='Max Loss', alpha=0.7, color='red')
            bars3b = ax3.bar(x_pos + width/2, max_gains, width, label='Max Gain', alpha=0.7, color='green')
            
            ax3.set_title('Risk-Reward Analysis', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Amount ($)', fontsize=12)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(contracts, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Confidence vs Expected Value Scatter
            confidences = [r.get('confidence', 0) for r in buy_recs]
            
            scatter = ax4.scatter(confidences, expected_values, 
                                s=[r.get('position_size_dollars', 0)/10 for r in buy_recs],
                                alpha=0.7, c=range(len(buy_recs)), cmap='viridis')
            
            ax4.set_title('Confidence vs Expected Value', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Confidence (%)', fontsize=12)
            ax4.set_ylabel('Expected Value ($)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add contract labels to scatter points
            for i, (conf, ev, contract) in enumerate(zip(confidences, expected_values, contracts)):
                ax4.annotate(contract, (conf, ev), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating recommendations plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def create_contract_analysis_plot(self, contract_analysis: Dict[str, Any]) -> Figure:
        """Create a plot showing contract analysis.
        
        Args:
            contract_analysis: Contract analysis data
            
        Returns:
            Matplotlib figure with contract analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            if 'error' in contract_analysis:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, f"Error: {contract_analysis['error']}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            contracts = contract_analysis.get('contracts', {})
            predicted_temp = contract_analysis.get('predicted_temp', 0)
            
            if not contracts:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No contract data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
            
            # Extract data for plotting
            thresholds = []
            above_probs = []
            below_probs = []
            above_evs = []
            below_evs = []
            
            for contract_name, analysis in contracts.items():
                if 'above_' in contract_name:
                    threshold = int(contract_name.split('_')[1])
                    thresholds.append(threshold)
                    above_probs.append(analysis.get('win_probability', 0))
                    above_evs.append(analysis.get('expected_value', 0))
                elif 'below_' in contract_name:
                    threshold = int(contract_name.split('_')[1])
                    if threshold not in [t for t in thresholds]:  # Avoid duplicates
                        continue
                    below_probs.append(analysis.get('win_probability', 0))
                    below_evs.append(analysis.get('expected_value', 0))
            
            # Sort by threshold
            sorted_data = sorted(zip(thresholds, above_probs, below_probs, above_evs, below_evs))
            thresholds, above_probs, below_probs, above_evs, below_evs = zip(*sorted_data)
            
            # Plot 1: Win Probabilities
            ax1.plot(thresholds, above_probs, 'o-', label='Above Threshold', linewidth=2, markersize=6)
            ax1.plot(thresholds, below_probs, 's-', label='Below Threshold', linewidth=2, markersize=6)
            ax1.axvline(x=predicted_temp, color='red', linestyle='--', alpha=0.7, label=f'Predicted: {predicted_temp:.1f}°F')
            ax1.set_title('Win Probabilities by Threshold', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Temperature Threshold (°F)', fontsize=12)
            ax1.set_ylabel('Win Probability', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Expected Values
            ax2.plot(thresholds, above_evs, 'o-', label='Above Threshold', linewidth=2, markersize=6)
            ax2.plot(thresholds, below_evs, 's-', label='Below Threshold', linewidth=2, markersize=6)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(x=predicted_temp, color='red', linestyle='--', alpha=0.7, label=f'Predicted: {predicted_temp:.1f}°F')
            ax2.set_title('Expected Values by Threshold', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Temperature Threshold (°F)', fontsize=12)
            ax2.set_ylabel('Expected Value ($)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Best Opportunities (Positive EV)
            positive_evs = []
            positive_contracts = []
            positive_thresholds = []
            
            for i, threshold in enumerate(thresholds):
                if above_evs[i] > 0:
                    positive_evs.append(above_evs[i])
                    positive_contracts.append(f'Above {threshold}°F')
                    positive_thresholds.append(threshold)
                if below_evs[i] > 0:
                    positive_evs.append(below_evs[i])
                    positive_contracts.append(f'Below {threshold}°F')
                    positive_thresholds.append(threshold)
            
            if positive_evs:
                bars3 = ax3.bar(range(len(positive_contracts)), positive_evs, alpha=0.7, color='green')
                ax3.set_title('Positive Expected Value Opportunities', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Expected Value ($)', fontsize=12)
                ax3.set_xticks(range(len(positive_contracts)))
                ax3.set_xticklabels(positive_contracts, rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, ev in zip(bars3, positive_evs):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'${ev:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                ax3.text(0.5, 0.5, 'No positive EV opportunities', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            
            # Plot 4: Temperature Distribution Visualization
            temp_range = np.linspace(predicted_temp - 10, predicted_temp + 10, 100)
            # Simple normal distribution around prediction
            confidence = contract_analysis.get('confidence', 0.8)
            std_dev = (100 - confidence * 100) / 10  # Lower confidence = higher std dev
            prob_density = np.exp(-0.5 * ((temp_range - predicted_temp) / std_dev) ** 2)
            prob_density = prob_density / np.max(prob_density)  # Normalize
            
            ax4.plot(temp_range, prob_density, 'b-', linewidth=2, label='Predicted Distribution')
            ax4.axvline(x=predicted_temp, color='red', linestyle='--', alpha=0.7, label=f'Predicted: {predicted_temp:.1f}°F')
            
            # Add threshold lines
            for threshold in thresholds:
                ax4.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
                ax4.text(threshold, 0.9, f'{threshold}°F', rotation=90, ha='right', va='top', fontsize=8)
            
            ax4.set_title('Temperature Prediction Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Temperature (°F)', fontsize=12)
            ax4.set_ylabel('Probability Density', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating contract analysis plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        return fig
    
    def generate_trading_report(self, target_date: Optional[date] = None, 
                               bankroll: float = 10000) -> Dict[str, Any]:
        """Generate a comprehensive trading report.
        
        Args:
            target_date: Date to generate report for (defaults to today)
            bankroll: Available bankroll for position sizing
            
        Returns:
            Dictionary containing trading report data
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            report = {
                'date': target_date,
                'generated_at': datetime.now(),
                'bankroll': bankroll,
                'recommendations': [],
                'contract_analysis': {},
                'risk_metrics': {},
                'summary': {}
            }
            
            # Get recommendations
            recommendations = self.get_latest_recommendations()
            if recommendations:
                # Add position sizing
                sized_recommendations = self.calculate_position_sizing(recommendations, bankroll)
                report['recommendations'] = sized_recommendations
            
            # Get contract analysis
            contract_analysis = self.analyze_contract_opportunities(target_date)
            report['contract_analysis'] = contract_analysis
            
            # Calculate risk metrics
            if report['recommendations']:
                total_position = sum(r.get('position_size_dollars', 0) for r in report['recommendations'])
                max_loss = sum(r.get('max_loss', 0) for r in report['recommendations'])
                max_gain = sum(r.get('max_gain', 0) for r in report['recommendations'])
                
                report['risk_metrics'] = {
                    'total_position_size': total_position,
                    'portfolio_allocation': total_position / bankroll if bankroll > 0 else 0,
                    'max_portfolio_loss': max_loss / bankroll if bankroll > 0 else 0,
                    'max_portfolio_gain': max_gain / bankroll if bankroll > 0 else 0,
                    'risk_reward_ratio': max_gain / max_loss if max_loss > 0 else 0,
                    'number_of_positions': len([r for r in report['recommendations'] if r.get('position_size_dollars', 0) > 0])
                }
            
            # Generate summary
            buy_recs = [r for r in report['recommendations'] if r.get('recommendation') == 'BUY']
            report['summary'] = {
                'total_recommendations': len(report['recommendations']),
                'buy_recommendations': len(buy_recs),
                'highest_ev_contract': max(buy_recs, key=lambda x: x.get('expected_value', 0)) if buy_recs else None,
                'largest_position': max(buy_recs, key=lambda x: x.get('position_size_dollars', 0)) if buy_recs else None,
                'recommendation_strength': 'Strong' if len(buy_recs) > 2 else 'Moderate' if len(buy_recs) > 0 else 'Weak'
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trading report: {e}")
            return {
                'date': target_date,
                'generated_at': datetime.now(),
                'error': str(e)
            }
    
    def save_trading_plots(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save all trading dashboard plots to files.
        
        Args:
            output_dir: Directory to save plots (defaults to logs directory)
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            output_dir = config.logs_dir / 'dashboard_plots'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        try:
            # Get data for plots
            recommendations = self.get_latest_recommendations()
            sized_recommendations = self.calculate_position_sizing(recommendations)
            contract_analysis = self.analyze_contract_opportunities(date.today())
            
            # Save recommendations plot
            if sized_recommendations:
                fig1 = self.create_recommendations_plot(sized_recommendations)
                path1 = output_dir / f'trading_recommendations_{date.today()}.png'
                fig1.savefig(path1, dpi=self.dpi, bbox_inches='tight')
                saved_files['trading_recommendations'] = path1
                plt.close(fig1)
            
            # Save contract analysis plot
            if 'error' not in contract_analysis:
                fig2 = self.create_contract_analysis_plot(contract_analysis)
                path2 = output_dir / f'contract_analysis_{date.today()}.png'
                fig2.savefig(path2, dpi=self.dpi, bbox_inches='tight')
                saved_files['contract_analysis'] = path2
                plt.close(fig2)
            
            logger.info(f"Trading dashboard plots saved to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving trading plots: {e}")
            return saved_files


if __name__ == "__main__":
    # Demo usage
    dashboard = TradingDashboard()
    
    # Generate and display trading analysis
    print("Creating trading dashboard...")
    
    # Get recommendations
    recommendations = dashboard.get_latest_recommendations()
    print(f"Generated {len(recommendations)} recommendations")
    
    # Analyze contracts
    contract_analysis = dashboard.analyze_contract_opportunities(date.today())
    print(f"Contract analysis completed for {date.today()}")
    
    # Generate comprehensive report
    report = dashboard.generate_trading_report(bankroll=10000)
    print(f"Trading report generated")
    
    # Save plots
    saved_files = dashboard.save_trading_plots()
    print(f"Plots saved: {saved_files}")
    
    # Print summary
    if 'summary' in report:
        summary = report['summary']
        print(f"\nTrading Summary:")
        print(f"  Total recommendations: {summary['total_recommendations']}")
        print(f"  Buy recommendations: {summary['buy_recommendations']}")
        print(f"  Recommendation strength: {summary['recommendation_strength']}")
        
        if summary['highest_ev_contract']:
            best = summary['highest_ev_contract']
            print(f"  Best opportunity: {best.get('contract_type', 'Unknown')} (EV: ${best.get('expected_value', 0):.3f})")