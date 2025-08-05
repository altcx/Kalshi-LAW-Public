"""
Demo script for Position Sizing and Risk Management System

This script demonstrates how to use the position sizing and risk management
components to calculate optimal position sizes and manage portfolio risk.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.trading.kalshi_contract_analyzer import (
    KalshiContractAnalyzer,
    create_sample_contracts
)
from src.trading.position_sizing import (
    PositionSizer,
    RiskManager,
    RiskLimits
)


def demo_position_sizing():
    """Demonstrate position sizing calculations"""
    
    print("=" * 60)
    print("POSITION SIZING AND RISK MANAGEMENT DEMO")
    print("=" * 60)
    
    # Set up components
    analyzer = KalshiContractAnalyzer()
    position_sizer = PositionSizer()
    risk_manager = RiskManager()
    
    # Portfolio parameters
    bankroll = 25000  # $25,000 bankroll
    confidence = 0.85  # 85% model confidence
    
    print(f"Portfolio Bankroll: ${bankroll:,}")
    print(f"Model Confidence: {confidence:.0%}")
    print()
    
    # Get contract analyses
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(
        contracts, predicted_temp=89.0, prediction_std=2.0, confidence=confidence
    )
    
    # Show available opportunities
    print("Available Trading Opportunities:")
    print("-" * 40)
    buy_opportunities = [a for a in analyses if a.recommendation == "BUY"]
    
    for i, analysis in enumerate(buy_opportunities, 1):
        print(f"{i}. {analysis.contract.description}")
        print(f"   Market Price: {analysis.contract.current_price:.1%}")
        print(f"   Predicted Probability: {analysis.predicted_probability:.1%}")
        print(f"   Expected Value: ${analysis.expected_value:.3f}")
        print(f"   Edge: {analysis.edge:+.1%}")
        print()
    
    # Calculate position sizes for top opportunities
    print("POSITION SIZE CALCULATIONS")
    print("=" * 40)
    
    current_positions = {
        "EXISTING_TEMP_CONTRACT_1": 3000,  # $3,000 existing position
        "EXISTING_TEMP_CONTRACT_2": 2000   # $2,000 existing position
    }
    
    print(f"Existing Positions: ${sum(current_positions.values()):,} ({sum(current_positions.values())/bankroll:.1%} of bankroll)")
    print()
    
    recommendations = []
    for i, analysis in enumerate(buy_opportunities[:3], 1):  # Top 3 opportunities
        print(f"Position {i}: {analysis.contract.description}")
        print("-" * 30)
        
        rec = position_sizer.calculate_position_size(
            analysis, bankroll, confidence, current_positions
        )
        recommendations.append(rec)
        
        print(f"Kelly Fraction: {rec.kelly_fraction:.1%}")
        print(f"Confidence Adjustment: {rec.confidence_adjustment:.1%}")
        print(f"Risk Adjustment: {rec.risk_adjustment:.1%}")
        print(f"Final Position Size: {rec.adjusted_fraction:.1%} (${rec.dollar_amount:,.0f})")
        print(f"Reasoning: {rec.reasoning}")
        
        if rec.warnings:
            print(f"⚠️  Warnings: {'; '.join(rec.warnings)}")
        
        print()
    
    # Risk management check
    print("RISK MANAGEMENT ANALYSIS")
    print("=" * 40)
    
    # Check risk limits
    approved_recs, risk_warnings = risk_manager.check_risk_limits(
        recommendations, current_positions, bankroll
    )
    
    if risk_warnings:
        print("Risk Warnings:")
        for warning in risk_warnings:
            print(f"⚠️  {warning}")
        print()
    
    # Portfolio risk assessment
    portfolio_risk = risk_manager.assess_portfolio_risk(current_positions, bankroll)
    
    print(f"Current Portfolio Risk Score: {portfolio_risk.overall_risk_score:.1%}")
    print(f"Total Exposure: {portfolio_risk.total_exposure:.1%}")
    print(f"Position Count: {portfolio_risk.position_count}")
    print(f"Concentration Risk: {portfolio_risk.concentration_risk:.1%}")
    print()
    
    # Generate comprehensive risk report
    report = risk_manager.generate_risk_report(portfolio_risk, approved_recs)
    print(report)


def demo_different_confidence_levels():
    """Demonstrate how position sizes change with different confidence levels"""
    
    print("\n" + "=" * 60)
    print("CONFIDENCE LEVEL IMPACT ON POSITION SIZING")
    print("=" * 60)
    
    analyzer = KalshiContractAnalyzer()
    position_sizer = PositionSizer()
    
    # Get a good trading opportunity
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 89.0, 2.0, 0.9)
    best_analysis = next(a for a in analyses if a.recommendation == "BUY")
    
    bankroll = 10000
    confidence_levels = [0.95, 0.85, 0.75, 0.65, 0.55]
    
    print(f"Contract: {best_analysis.contract.description}")
    print(f"Expected Value: ${best_analysis.expected_value:.3f}")
    print(f"Edge: {best_analysis.edge:+.1%}")
    print()
    
    print("Confidence Level → Position Size")
    print("-" * 35)
    
    for confidence in confidence_levels:
        rec = position_sizer.calculate_position_size(
            best_analysis, bankroll, confidence
        )
        
        print(f"{confidence:.0%} confidence → {rec.adjusted_fraction:.1%} of bankroll (${rec.dollar_amount:,.0f})")
    
    print()
    print("Key Insight: Lower confidence leads to smaller position sizes,")
    print("protecting against model uncertainty.")


def demo_risk_limits():
    """Demonstrate different risk limit scenarios"""
    
    print("\n" + "=" * 60)
    print("RISK LIMITS DEMONSTRATION")
    print("=" * 60)
    
    # Conservative risk limits
    conservative_limits = RiskLimits(
        max_position_size=0.10,      # 10% max per position
        max_total_exposure=0.30,     # 30% max total exposure
        max_single_contract_exposure=0.08,  # 8% max per contract type
        min_bankroll_reserve=0.30    # 30% reserve
    )
    
    # Aggressive risk limits
    aggressive_limits = RiskLimits(
        max_position_size=0.40,      # 40% max per position
        max_total_exposure=0.80,     # 80% max total exposure
        max_single_contract_exposure=0.30,  # 30% max per contract type
        min_bankroll_reserve=0.10    # 10% reserve
    )
    
    # Test with same opportunity
    analyzer = KalshiContractAnalyzer()
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 92.0, 1.5, 0.9)
    best_analysis = next(a for a in analyses if a.recommendation == "BUY")
    
    bankroll = 10000
    confidence = 0.9
    
    print(f"Testing with: {best_analysis.contract.description}")
    print(f"High confidence prediction (90%)")
    print()
    
    # Conservative approach
    conservative_sizer = PositionSizer(conservative_limits)
    conservative_rec = conservative_sizer.calculate_position_size(
        best_analysis, bankroll, confidence
    )
    
    print("CONSERVATIVE RISK MANAGEMENT:")
    print(f"Position Size: {conservative_rec.adjusted_fraction:.1%} (${conservative_rec.dollar_amount:,.0f})")
    print(f"Max Position Limit: {conservative_limits.max_position_size:.1%}")
    print()
    
    # Aggressive approach
    aggressive_sizer = PositionSizer(aggressive_limits)
    aggressive_rec = aggressive_sizer.calculate_position_size(
        best_analysis, bankroll, confidence
    )
    
    print("AGGRESSIVE RISK MANAGEMENT:")
    print(f"Position Size: {aggressive_rec.adjusted_fraction:.1%} (${aggressive_rec.dollar_amount:,.0f})")
    print(f"Max Position Limit: {aggressive_limits.max_position_size:.1%}")
    print()
    
    print("Key Insight: Risk limits significantly impact position sizes.")
    print("Conservative limits protect capital but may reduce returns.")
    print("Aggressive limits increase potential returns but also risk.")


def demo_portfolio_scenarios():
    """Demonstrate different portfolio scenarios"""
    
    print("\n" + "=" * 60)
    print("PORTFOLIO SCENARIO ANALYSIS")
    print("=" * 60)
    
    risk_manager = RiskManager()
    bankroll = 20000
    
    scenarios = [
        {
            "name": "Empty Portfolio",
            "positions": {},
            "description": "Starting fresh with no positions"
        },
        {
            "name": "Moderate Exposure",
            "positions": {
                "CONTRACT_A": 3000,
                "CONTRACT_B": 2000
            },
            "description": "25% total exposure across 2 positions"
        },
        {
            "name": "High Exposure",
            "positions": {
                "CONTRACT_A": 4000,
                "CONTRACT_B": 3000,
                "CONTRACT_C": 2000,
                "CONTRACT_D": 1000
            },
            "description": "50% total exposure across 4 positions"
        },
        {
            "name": "Concentrated Risk",
            "positions": {
                "LARGE_POSITION": 8000,
                "SMALL_POSITION": 1000
            },
            "description": "45% exposure but highly concentrated"
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 40)
        
        risk = risk_manager.assess_portfolio_risk(scenario['positions'], bankroll)
        
        print(f"Total Exposure: {risk.total_exposure:.1%}")
        print(f"Position Count: {risk.position_count}")
        print(f"Concentration Risk: {risk.concentration_risk:.1%}")
        print(f"Overall Risk Score: {risk.overall_risk_score:.1%}")
        
        # Risk level
        if risk.overall_risk_score < 0.3:
            risk_level = "LOW 🟢"
        elif risk.overall_risk_score < 0.6:
            risk_level = "MODERATE 🟡"
        else:
            risk_level = "HIGH 🔴"
        
        print(f"Risk Level: {risk_level}")


if __name__ == "__main__":
    # Run all demos
    demo_position_sizing()
    demo_different_confidence_levels()
    demo_risk_limits()
    demo_portfolio_scenarios()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe Position Sizing and Risk Management System successfully:")
    print("✓ Implements Kelly criterion for optimal position sizing")
    print("✓ Adjusts positions based on model confidence")
    print("✓ Applies comprehensive risk management controls")
    print("✓ Monitors portfolio exposure and concentration")
    print("✓ Provides detailed reasoning for all decisions")
    print("✓ Generates comprehensive risk reports")
    print("✓ Handles various portfolio scenarios")
    print("✓ Protects against excessive risk-taking")