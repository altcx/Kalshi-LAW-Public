"""Prediction display dashboard for weather forecasting system."""

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
from src.monitoring.performance_tracker import PerformanceTracker


class PredictionDashboard:
    """Dashboard for displaying temperature predictions and model performance."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """Initialize prediction dashboard.
        
        Args:
            data_manager: DataManager instance (creates new one if None)
        """
        self.data_manager = data_manager or DataManager()
        self.performance_tracker = PerformanceTracker(self.data_manager)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Dashboard configuration
        self.figure_size = (15, 10)
        self.dpi = 100
        
        logger.info("PredictionDashboard initialized")
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent temperature prediction.
        
        Returns:
            Dictionary with prediction details or None if no predictions available
        """
        try:
            predictions_df = self.data_manager.load_predictions()
            if predictions_df.empty:
                logger.warning("No predictions available")
                return None
            
            # Get the most recent prediction
            latest = predictions_df.loc[predictions_df['date'].idxmax()]
            
            return {
                'date': latest['date'],
                'predicted_high': latest['predicted_high'],
                'confidence': latest['confidence'],
                'model_contributions': latest.get('model_contributions', {}),
                'feature_importance': latest.get('feature_importance', {}),
                'created_at': latest['created_at']
            }
            
        except Exception as e:
            logger.error(f"Error getting latest prediction: {e}")
            return None
    
    def get_source_contributions(self, target_date: date) -> Dict[str, float]:
        """Get individual weather source contributions for a specific date.
        
        Args:
            target_date: Date to get contributions for
            
        Returns:
            Dictionary mapping source names to their predicted temperatures
        """
        contributions = {}
        
        try:
            # Load data from each source for the target date
            sources = ['nws', 'openweather', 'tomorrow', 'weatherbit', 'visual_crossing']
            
            for source in sources:
                try:
                    source_data = self.data_manager.load_source_data(
                        source, target_date, target_date
                    )
                    
                    if not source_data.empty:
                        # Get the most recent forecast for the target date
                        forecast = source_data[source_data['date'] == pd.Timestamp(target_date)]
                        if not forecast.empty:
                            contributions[source] = forecast.iloc[-1]['predicted_high']
                        
                except Exception as e:
                    logger.warning(f"Could not load {source} data: {e}")
                    continue
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error getting source contributions: {e}")
            return {}
    
    def create_prediction_summary_plot(self, days_back: int = 7) -> Figure:
        """Create a summary plot showing recent predictions vs actuals.
        
        Args:
            days_back: Number of days to include in the plot
            
        Returns:
            Matplotlib figure with the prediction summary
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        try:
            # Load recent predictions and actuals
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            predictions_df = self.data_manager.load_predictions(start_date, end_date)
            actuals_df = self.data_manager.load_actual_temperatures(start_date, end_date)
            
            if predictions_df.empty:
                ax1.text(0.5, 0.5, 'No prediction data available', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax2.text(0.5, 0.5, 'No data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                return fig
            
            # Merge predictions with actuals
            merged_df = pd.merge(predictions_df, actuals_df, on='date', how='left')
            merged_df = merged_df.sort_values('date')
            
            # Plot predictions vs actuals
            ax1.plot(merged_df['date'], merged_df['predicted_high'], 
                    'o-', label='Predicted High', linewidth=2, markersize=6)
            
            if 'actual_high' in merged_df.columns:
                ax1.plot(merged_df['date'], merged_df['actual_high'], 
                        's-', label='Actual High', linewidth=2, markersize=6)
                
                # Add error bars for confidence
                if 'confidence' in merged_df.columns:
                    confidence_range = (100 - merged_df['confidence']) / 10  # Convert to temp range
                    ax1.fill_between(merged_df['date'], 
                                   merged_df['predicted_high'] - confidence_range,
                                   merged_df['predicted_high'] + confidence_range,
                                   alpha=0.2, label='Confidence Range')
            
            ax1.set_title('Temperature Predictions vs Actuals', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Temperature (°F)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot confidence scores
            if 'confidence' in merged_df.columns:
                ax2.bar(merged_df['date'], merged_df['confidence'], 
                       alpha=0.7, color='green', label='Confidence %')
                ax2.set_ylabel('Confidence (%)', fontsize=12)
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            # Format x-axis for confidence plot
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating prediction summary plot: {e}")
            ax1.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        return fig
    
    def create_source_contributions_plot(self, target_date: date) -> Figure:
        """Create a plot showing individual source contributions.
        
        Args:
            target_date: Date to analyze source contributions for
            
        Returns:
            Matplotlib figure with source contributions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        try:
            contributions = self.get_source_contributions(target_date)
            
            if not contributions:
                ax1.text(0.5, 0.5, 'No source data available', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax2.text(0.5, 0.5, 'No source data available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                return fig
            
            sources = list(contributions.keys())
            temperatures = list(contributions.values())
            
            # Bar chart of source predictions
            bars = ax1.bar(sources, temperatures, alpha=0.7)
            ax1.set_title(f'Source Predictions for {target_date}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Predicted High Temperature (°F)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, temp in zip(bars, temperatures):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{temp:.1f}°F', ha='center', va='bottom', fontsize=10)
            
            # Pie chart showing source agreement
            if len(temperatures) > 1:
                temp_range = max(temperatures) - min(temperatures)
                mean_temp = np.mean(temperatures)
                
                # Calculate how close each source is to the mean
                agreements = [1 / (1 + abs(temp - mean_temp)) for temp in temperatures]
                
                wedges, texts, autotexts = ax2.pie(agreements, labels=sources, autopct='%1.1f%%',
                                                  startangle=90)
                ax2.set_title(f'Source Agreement\n(Range: {temp_range:.1f}°F)', 
                             fontsize=14, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'Need multiple sources\nfor agreement analysis', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating source contributions plot: {e}")
            ax1.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        
        return fig
    
    def create_accuracy_trends_plot(self, days_back: int = 30) -> Figure:
        """Create a plot showing historical accuracy trends.
        
        Args:
            days_back: Number of days to include in the analysis
            
        Returns:
            Matplotlib figure with accuracy trends
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size)
        
        try:
            # Get performance data
            end_date = date.today()
            start_date = end_date - timedelta(days=days_back)
            
            performance_data = self.performance_tracker.get_performance_summary(
                start_date, end_date
            )
            
            if not performance_data:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No performance data available', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                return fig
            
            # Plot 1: Overall accuracy over time
            if 'daily_accuracy' in performance_data:
                daily_acc = performance_data['daily_accuracy']
                dates = list(daily_acc.keys())
                accuracies = list(daily_acc.values())
                
                ax1.plot(dates, accuracies, 'o-', linewidth=2, markersize=4)
                ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (80%)')
                ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Acceptable (70%)')
                ax1.set_title('Daily Accuracy Trend', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Accuracy (within ±3°F)', fontsize=10)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Source performance comparison
            if 'source_performance' in performance_data:
                source_perf = performance_data['source_performance']
                sources = list(source_perf.keys())
                accuracies = [source_perf[s].get('accuracy_7d', 0) for s in sources]
                
                bars = ax2.bar(sources, accuracies, alpha=0.7)
                ax2.set_title('7-Day Source Accuracy', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Accuracy', fontsize=10)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Error distribution
            if 'error_distribution' in performance_data:
                errors = performance_data['error_distribution']
                ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect')
                ax3.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Error (°F)', fontsize=10)
                ax3.set_ylabel('Frequency', fontsize=10)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Confidence calibration
            if 'confidence_calibration' in performance_data:
                cal_data = performance_data['confidence_calibration']
                confidence_bins = list(cal_data.keys())
                actual_accuracies = list(cal_data.values())
                
                ax4.plot(confidence_bins, actual_accuracies, 'o-', label='Actual')
                ax4.plot([0, 100], [0, 100], '--', alpha=0.7, label='Perfect Calibration')
                ax4.set_title('Confidence Calibration', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Predicted Confidence (%)', fontsize=10)
                ax4.set_ylabel('Actual Accuracy (%)', fontsize=10)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating accuracy trends plot: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        return fig
    
    def generate_prediction_report(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """Generate a comprehensive prediction report.
        
        Args:
            target_date: Date to generate report for (defaults to today)
            
        Returns:
            Dictionary containing prediction report data
        """
        if target_date is None:
            target_date = date.today()
        
        try:
            report = {
                'date': target_date,
                'generated_at': datetime.now(),
                'prediction': None,
                'source_contributions': {},
                'performance_metrics': {},
                'confidence_analysis': {},
                'recommendations': []
            }
            
            # Get latest prediction
            latest_prediction = self.get_latest_prediction()
            if latest_prediction:
                report['prediction'] = latest_prediction
            
            # Get source contributions
            report['source_contributions'] = self.get_source_contributions(target_date)
            
            # Get performance metrics
            performance_data = self.performance_tracker.get_performance_summary(
                target_date - timedelta(days=30), target_date
            )
            if performance_data:
                report['performance_metrics'] = performance_data
            
            # Generate recommendations based on confidence and accuracy
            if latest_prediction:
                confidence = latest_prediction.get('confidence', 0)
                if confidence > 85:
                    report['recommendations'].append("High confidence prediction - suitable for trading")
                elif confidence > 70:
                    report['recommendations'].append("Moderate confidence - consider position sizing")
                else:
                    report['recommendations'].append("Low confidence - avoid trading or wait for better data")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating prediction report: {e}")
            return {
                'date': target_date,
                'generated_at': datetime.now(),
                'error': str(e)
            }
    
    def save_dashboard_plots(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save all dashboard plots to files.
        
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
            # Save prediction summary plot
            fig1 = self.create_prediction_summary_plot()
            path1 = output_dir / f'prediction_summary_{date.today()}.png'
            fig1.savefig(path1, dpi=self.dpi, bbox_inches='tight')
            saved_files['prediction_summary'] = path1
            plt.close(fig1)
            
            # Save source contributions plot
            fig2 = self.create_source_contributions_plot(date.today())
            path2 = output_dir / f'source_contributions_{date.today()}.png'
            fig2.savefig(path2, dpi=self.dpi, bbox_inches='tight')
            saved_files['source_contributions'] = path2
            plt.close(fig2)
            
            # Save accuracy trends plot
            fig3 = self.create_accuracy_trends_plot()
            path3 = output_dir / f'accuracy_trends_{date.today()}.png'
            fig3.savefig(path3, dpi=self.dpi, bbox_inches='tight')
            saved_files['accuracy_trends'] = path3
            plt.close(fig3)
            
            logger.info(f"Dashboard plots saved to {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving dashboard plots: {e}")
            return saved_files


if __name__ == "__main__":
    # Demo usage
    dashboard = PredictionDashboard()
    
    # Generate and display plots
    print("Creating prediction dashboard...")
    
    # Save plots
    saved_files = dashboard.save_dashboard_plots()
    print(f"Plots saved: {saved_files}")
    
    # Generate report
    report = dashboard.generate_prediction_report()
    print(f"Prediction report generated for {report['date']}")
    
    if 'prediction' in report and report['prediction']:
        pred = report['prediction']
        print(f"Latest prediction: {pred['predicted_high']:.1f}°F (confidence: {pred['confidence']:.1f}%)")
    
    if report['source_contributions']:
        print("Source contributions:")
        for source, temp in report['source_contributions'].items():
            print(f"  {source}: {temp:.1f}°F")