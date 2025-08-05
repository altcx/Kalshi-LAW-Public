"""
Demo script for Kalshi Contract Analyzer

This script demonstrates how to use the KalshiContractAnalyzer to analyze
temperature threshold contracts and generate trading recommendations.
"""

from datetime import date
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.trading.kalshi_contract_analyzer import (
    KalshiContract,
    KalshiContractAnalyzer,
    create_sample_contracts
)


def demo_contract_analysis():
    """Demonstrate contract analysis with different scenarios"""
    
    print("=" * 60)
    print("KALSHI CONTRACT ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = KalshiContractAnalyzer(transaction_cost=0.01)
    
    # Create sample contracts
    contracts = create_sample_contracts()
    
    print(f"\nAvailable Contracts ({len(contracts)}):")
    print("-" * 40)
    for i, contract in enumerate(contracts, 1):
        print(f"{i}. {contract.description}")
        print(f"   Type: {contract.contract_type} {contract.threshold_temp}°F")
        print(f"   Market Price: {contract.current_price:.1%}")
        print()
    
    # Scenario 1: Hot day prediction
    print("\n" + "=" * 60)
    print("SCENARIO 1: HOT DAY PREDICTION")
    print("Predicted Temperature: 92°F ± 2°F, Confidence: 90%")
    print("=" * 60)
    
    analyses_hot = analyzer.analyze_contracts(
        contracts, 
        predicted_temp=92.0, 
        prediction_std=2.0, 
        confidence=0.90
    )
    
    print_analysis_results(analyses_hot)
    
    # Scenario 2: Cool day prediction
    print("\n" + "=" * 60)
    print("SCENARIO 2: COOL DAY PREDICTION")
    print("Predicted Temperature: 78°F ± 3°F, Confidence: 75%")
    print("=" * 60)
    
    analyses_cool = analyzer.analyze_contracts(
        contracts,
        predicted_temp=78.0,
        prediction_std=3.0,
        confidence=0.75
    )
    
    print_analysis_results(analyses_cool)
    
    # Scenario 3: Uncertain prediction
    print("\n" + "=" * 60)
    print("SCENARIO 3: UNCERTAIN PREDICTION")
    print("Predicted Temperature: 85°F ± 5°F, Confidence: 60%")
    print("=" * 60)
    
    analyses_uncertain = analyzer.analyze_contracts(
        contracts,
        predicted_temp=85.0,
        prediction_std=5.0,
        confidence=0.60
    )
    
    print_analysis_results(analyses_uncertain)
    
    # Show best opportunities
    print("\n" + "=" * 60)
    print("BEST TRADING OPPORTUNITIES")
    print("=" * 60)
    
    all_analyses = analyses_hot + analyses_cool + analyses_uncertain
    best_opportunities = analyzer.get_best_opportunities(
        all_analyses, 
        min_expected_value=0.01, 
        min_edge=0.02
    )
    
    if best_opportunities:
        print(f"Found {len(best_opportunities)} good opportunities:")
        print()
        for i, opp in enumerate(best_opportunities[:5], 1):  # Top 5
            print(f"{i}. {opp.contract.description}")
            print(f"   Expected Value: ${opp.expected_value:.3f}")
            print(f"   Edge: {opp.edge:.1%}")
            print(f"   Recommendation: {opp.recommendation}")
            print(f"   Reasoning: {opp.reasoning}")
            print()
    else:
        print("No good opportunities found with current criteria.")


def print_analysis_results(analyses):
    """Print formatted analysis results"""
    
    print(f"\nAnalysis Results (sorted by Expected Value):")
    print("-" * 50)
    
    for i, analysis in enumerate(analyses, 1):
        contract = analysis.contract
        
        print(f"{i}. {contract.description}")
        print(f"   Market Price: {contract.current_price:.1%}")
        print(f"   Predicted Probability: {analysis.predicted_probability:.1%}")
        print(f"   Expected Value: ${analysis.expected_value:.3f}")
        print(f"   Edge: {analysis.edge:+.1%}")
        print(f"   Confidence Interval: {analysis.confidence_interval[0]:.1%} - {analysis.confidence_interval[1]:.1%}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Reasoning: {analysis.reasoning}")
        print()


def demo_custom_contracts():
    """Demonstrate analysis with custom contracts"""
    
    print("\n" + "=" * 60)
    print("CUSTOM CONTRACT ANALYSIS")
    print("=" * 60)
    
    # Create custom contracts for a specific scenario
    custom_contracts = [
        KalshiContract(
            contract_id="LA_TEMP_ABOVE_88F",
            contract_type="ABOVE",
            threshold_temp=88.0,
            current_price=0.35,  # Market thinks 35% chance
            expiry_date=date.today(),
            description="LA high temperature above 88°F"
        ),
        KalshiContract(
            contract_id="LA_TEMP_BELOW_82F",
            contract_type="BELOW",
            threshold_temp=82.0,
            current_price=0.20,  # Market thinks 20% chance
            expiry_date=date.today(),
            description="LA high temperature below 82°F"
        )
    ]
    
    analyzer = KalshiContractAnalyzer()
    
    # Our model predicts 89°F with high confidence
    print("Our Prediction: 89°F ± 1.5°F, Confidence: 95%")
    print()
    
    analyses = analyzer.analyze_contracts(
        custom_contracts,
        predicted_temp=89.0,
        prediction_std=1.5,
        confidence=0.95
    )
    
    print_analysis_results(analyses)


def demo_edge_cases():
    """Demonstrate edge cases and error handling"""
    
    print("\n" + "=" * 60)
    print("EDGE CASES AND ERROR HANDLING")
    print("=" * 60)
    
    analyzer = KalshiContractAnalyzer()
    
    # Test with extreme predictions
    extreme_contract = KalshiContract(
        contract_id="EXTREME_TEST",
        contract_type="ABOVE",
        threshold_temp=85.0,
        current_price=0.5,
        expiry_date=date.today(),
        description="Extreme test contract"
    )
    
    print("Testing extreme predictions:")
    
    # Very hot prediction
    analysis_hot = analyzer._analyze_single_contract(
        extreme_contract, predicted_temp=110.0, prediction_std=1.0, confidence=0.99
    )
    print(f"Very hot (110°F): Probability = {analysis_hot.predicted_probability:.1%}, Recommendation = {analysis_hot.recommendation}")
    
    # Very cold prediction
    analysis_cold = analyzer._analyze_single_contract(
        extreme_contract, predicted_temp=50.0, prediction_std=1.0, confidence=0.99
    )
    print(f"Very cold (50°F): Probability = {analysis_cold.predicted_probability:.1%}, Recommendation = {analysis_cold.recommendation}")
    
    # High uncertainty
    analysis_uncertain = analyzer._analyze_single_contract(
        extreme_contract, predicted_temp=85.0, prediction_std=10.0, confidence=0.30
    )
    print(f"High uncertainty (85°F ± 10°F): Probability = {analysis_uncertain.predicted_probability:.1%}, Recommendation = {analysis_uncertain.recommendation}")


if __name__ == "__main__":
    # Run all demos
    demo_contract_analysis()
    demo_custom_contracts()
    demo_edge_cases()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nThe Kalshi Contract Analyzer successfully:")
    print("✓ Analyzes temperature threshold contracts")
    print("✓ Calculates expected values and probabilities")
    print("✓ Generates BUY/SELL/HOLD recommendations")
    print("✓ Ranks contracts by profitability")
    print("✓ Handles various prediction scenarios")
    print("✓ Provides detailed reasoning for decisions")