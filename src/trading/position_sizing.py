"""
Position Sizing and Risk Management System

This module implements Kelly criterion-based position sizing, confidence-based
adjustments, and comprehensive risk management controls for Kalshi trading.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import numpy as np
import logging

from src.trading.kalshi_contract_analyzer import ContractAnalysis


@dataclass
class RiskLimits:
    """Risk management limits and controls"""
    max_position_size: float = 0.25  # Maximum 25% of bankroll per position
    max_total_exposure: float = 0.50  # Maximum 50% of bankroll total exposure
    max_single_contract_exposure: float = 0.15  # Maximum 15% per contract type
    min_bankroll_reserve: float = 0.20  # Keep 20% in reserve
    max_correlation_exposure: float = 0.30  # Max exposure to correlated positions
    stop_loss_threshold: float = 0.10  # Stop trading if down 10%
    
    def __post_init__(self):
        """Validate risk limits"""
        if not (0 < self.max_position_size <= 1.0):
            raise ValueError("max_position_size must be between 0 and 1")
        if not (0 < self.max_total_exposure <= 1.0):
            raise ValueError("max_total_exposure must be between 0 and 1")


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation for a contract"""
    contract_id: str
    kelly_fraction: float  # Raw Kelly criterion result
    adjusted_fraction: float  # Risk-adjusted position size
    dollar_amount: float  # Recommended dollar amount
    confidence_adjustment: float  # Confidence-based adjustment factor
    risk_adjustment: float  # Risk-based adjustment factor
    reasoning: str  # Explanation of sizing decision
    warnings: List[str]  # Any risk warnings


@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics"""
    total_exposure: float  # Total exposure as fraction of bankroll
    position_count: int  # Number of open positions
    correlation_risk: float  # Estimated correlation risk
    concentration_risk: float  # Concentration in single contracts
    liquidity_risk: float  # Risk from illiquid positions
    overall_risk_score: float  # Overall risk score (0-1)


class PositionSizer:
    """
    Implements Kelly criterion-based position sizing with risk management
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize position sizer
        
        Args:
            risk_limits: Risk management limits (uses defaults if None)
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        analysis: ContractAnalysis,
        bankroll: float,
        confidence: float,
        current_positions: Optional[Dict[str, float]] = None
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size using Kelly criterion with risk adjustments
        
        Args:
            analysis: Contract analysis with expected value and probability
            bankroll: Current bankroll amount
            confidence: Model confidence (0-1)
            current_positions: Current positions {contract_id: dollar_amount}
            
        Returns:
            Position size recommendation
        """
        current_positions = current_positions or {}
        
        # Calculate raw Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(analysis)
        
        # Apply confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(confidence)
        
        # Apply risk adjustments
        risk_adjustment = self._calculate_risk_adjustment(
            analysis, bankroll, current_positions
        )
        
        # Calculate final adjusted fraction
        adjusted_fraction = kelly_fraction * confidence_adjustment * risk_adjustment
        
        # Apply hard limits
        adjusted_fraction = min(adjusted_fraction, self.risk_limits.max_position_size)
        
        # Calculate dollar amount
        dollar_amount = adjusted_fraction * bankroll
        
        # Generate reasoning and warnings
        reasoning, warnings = self._generate_sizing_reasoning(
            kelly_fraction, confidence_adjustment, risk_adjustment, 
            adjusted_fraction, confidence
        )
        
        return PositionSizeRecommendation(
            contract_id=analysis.contract.contract_id,
            kelly_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            dollar_amount=dollar_amount,
            confidence_adjustment=confidence_adjustment,
            risk_adjustment=risk_adjustment,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _calculate_kelly_fraction(self, analysis: ContractAnalysis) -> float:
        """
        Calculate Kelly criterion fraction
        
        Kelly fraction = (bp - q) / b
        where:
        - b = odds received on the wager (payout/cost - 1)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        p = analysis.predicted_probability  # Probability of winning
        q = 1 - p  # Probability of losing
        
        # For Kalshi contracts: cost = current_price, payout = 1.0
        cost = analysis.contract.current_price
        payout = 1.0
        
        if cost <= 0 or cost >= 1:
            return 0.0  # Invalid price
        
        # Odds = (payout - cost) / cost
        b = (payout - cost) / cost
        
        if b <= 0:
            return 0.0  # No positive odds
        
        # Kelly fraction
        kelly_fraction = (b * p - q) / b
        
        # Ensure non-negative (don't bet if Kelly is negative)
        return max(0.0, kelly_fraction)
    
    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Adjust position size based on model confidence
        
        Higher confidence = larger positions
        Lower confidence = smaller positions
        """
        if confidence >= 0.9:
            return 1.0  # Full Kelly at very high confidence
        elif confidence >= 0.8:
            return 0.8  # 80% of Kelly at high confidence
        elif confidence >= 0.7:
            return 0.6  # 60% of Kelly at moderate confidence
        elif confidence >= 0.6:
            return 0.4  # 40% of Kelly at low confidence
        else:
            return 0.2  # 20% of Kelly at very low confidence
    
    def _calculate_risk_adjustment(
        self,
        analysis: ContractAnalysis,
        bankroll: float,
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate risk-based adjustment to position size
        """
        adjustments = []
        
        # Total exposure adjustment
        total_exposure = sum(abs(pos) for pos in current_positions.values()) / bankroll
        if total_exposure > self.risk_limits.max_total_exposure * 0.8:
            exposure_adj = max(0.5, 1.0 - (total_exposure - 0.4) / 0.1)
            adjustments.append(exposure_adj)
        
        # Contract concentration adjustment
        same_type_exposure = sum(
            abs(pos) for contract_id, pos in current_positions.items()
            if self._contracts_are_similar(analysis.contract.contract_id, contract_id)
        ) / bankroll
        
        if same_type_exposure > self.risk_limits.max_single_contract_exposure:
            concentration_adj = max(0.3, 1.0 - same_type_exposure / 0.2)
            adjustments.append(concentration_adj)
        
        # Edge size adjustment (smaller positions for smaller edges)
        edge = analysis.edge
        if edge < 0.05:  # Small edge
            edge_adj = max(0.5, edge / 0.05)
            adjustments.append(edge_adj)
        
        # Return minimum adjustment (most conservative)
        return min(adjustments) if adjustments else 1.0
    
    def _contracts_are_similar(self, contract1: str, contract2: str) -> bool:
        """Check if two contracts are similar (same underlying, similar strikes)"""
        # Simple heuristic: same if they contain similar temperature thresholds
        # In practice, this would be more sophisticated
        return "TEMP" in contract1 and "TEMP" in contract2
    
    def _generate_sizing_reasoning(
        self,
        kelly_fraction: float,
        confidence_adj: float,
        risk_adj: float,
        final_fraction: float,
        confidence: float
    ) -> Tuple[str, List[str]]:
        """Generate reasoning and warnings for position sizing"""
        
        reasoning_parts = []
        warnings = []
        
        # Kelly component
        reasoning_parts.append(f"Kelly criterion suggests {kelly_fraction:.1%} of bankroll")
        
        # Confidence adjustment
        if confidence_adj < 1.0:
            reasoning_parts.append(
                f"reduced to {kelly_fraction * confidence_adj:.1%} due to "
                f"{confidence:.0%} model confidence"
            )
        
        # Risk adjustment
        if risk_adj < 1.0:
            reasoning_parts.append(
                f"further reduced to {final_fraction:.1%} due to risk management"
            )
            warnings.append("Position size reduced due to portfolio risk limits")
        
        # Final recommendation
        if final_fraction == 0:
            reasoning_parts.append("No position recommended due to insufficient edge or high risk")
            warnings.append("Zero position size recommended")
        elif final_fraction < 0.02:
            warnings.append("Very small position size - consider if trade is worthwhile")
        elif final_fraction > 0.15:
            warnings.append("Large position size - ensure you're comfortable with the risk")
        
        reasoning = ". ".join(reasoning_parts) + "."
        
        return reasoning, warnings


class RiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """Initialize risk manager"""
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(__name__)
    
    def assess_portfolio_risk(
        self,
        current_positions: Dict[str, float],
        bankroll: float,
        market_data: Optional[Dict] = None
    ) -> PortfolioRisk:
        """
        Assess current portfolio risk
        
        Args:
            current_positions: Current positions {contract_id: dollar_amount}
            bankroll: Current bankroll
            market_data: Optional market data for correlation analysis
            
        Returns:
            Portfolio risk assessment
        """
        total_exposure = sum(abs(pos) for pos in current_positions.values()) / bankroll
        position_count = len([pos for pos in current_positions.values() if abs(pos) > 0])
        
        # Calculate concentration risk
        max_position = max(abs(pos) for pos in current_positions.values()) if current_positions else 0
        concentration_risk = max_position / bankroll if bankroll > 0 else 0
        
        # Estimate correlation risk (simplified)
        correlation_risk = min(0.8, total_exposure * 0.5) if position_count > 1 else 0
        
        # Liquidity risk (simplified - assume all Kalshi contracts are liquid)
        liquidity_risk = 0.1 if total_exposure > 0.3 else 0
        
        # Overall risk score
        risk_factors = [
            total_exposure,
            concentration_risk,
            correlation_risk,
            liquidity_risk
        ]
        overall_risk_score = np.mean(risk_factors)
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            position_count=position_count,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            overall_risk_score=overall_risk_score
        )
    
    def check_risk_limits(
        self,
        position_recommendations: List[PositionSizeRecommendation],
        current_positions: Dict[str, float],
        bankroll: float
    ) -> Tuple[List[PositionSizeRecommendation], List[str]]:
        """
        Check position recommendations against risk limits
        
        Returns:
            Tuple of (approved_recommendations, risk_warnings)
        """
        approved_recommendations = []
        risk_warnings = []
        
        # Calculate total proposed exposure
        current_exposure = sum(abs(pos) for pos in current_positions.values())
        proposed_exposure = sum(rec.dollar_amount for rec in position_recommendations)
        total_exposure = (current_exposure + proposed_exposure) / bankroll
        
        # Check total exposure limit
        if total_exposure > self.risk_limits.max_total_exposure:
            risk_warnings.append(
                f"Total exposure ({total_exposure:.1%}) exceeds limit "
                f"({self.risk_limits.max_total_exposure:.1%})"
            )
            # Scale down all recommendations proportionally
            scale_factor = (self.risk_limits.max_total_exposure * bankroll - current_exposure) / proposed_exposure
            scale_factor = max(0, scale_factor)
            
            for rec in position_recommendations:
                scaled_rec = PositionSizeRecommendation(
                    contract_id=rec.contract_id,
                    kelly_fraction=rec.kelly_fraction,
                    adjusted_fraction=rec.adjusted_fraction * scale_factor,
                    dollar_amount=rec.dollar_amount * scale_factor,
                    confidence_adjustment=rec.confidence_adjustment,
                    risk_adjustment=rec.risk_adjustment * scale_factor,
                    reasoning=rec.reasoning + f" (Scaled down by {scale_factor:.1%} due to exposure limits)",
                    warnings=rec.warnings + ["Position scaled down due to total exposure limits"]
                )
                approved_recommendations.append(scaled_rec)
        else:
            approved_recommendations = position_recommendations.copy()
        
        # Check individual position limits
        for i, rec in enumerate(approved_recommendations):
            if rec.adjusted_fraction > self.risk_limits.max_position_size:
                risk_warnings.append(
                    f"Position {rec.contract_id} exceeds individual limit"
                )
                # Cap at maximum
                approved_recommendations[i] = PositionSizeRecommendation(
                    contract_id=rec.contract_id,
                    kelly_fraction=rec.kelly_fraction,
                    adjusted_fraction=self.risk_limits.max_position_size,
                    dollar_amount=self.risk_limits.max_position_size * bankroll,
                    confidence_adjustment=rec.confidence_adjustment,
                    risk_adjustment=rec.risk_adjustment,
                    reasoning=rec.reasoning + " (Capped at maximum position size)",
                    warnings=rec.warnings + ["Position capped at maximum size limit"]
                )
        
        # Check minimum bankroll reserve
        total_proposed = sum(rec.dollar_amount for rec in approved_recommendations)
        if (current_exposure + total_proposed) > bankroll * (1 - self.risk_limits.min_bankroll_reserve):
            risk_warnings.append(
                f"Insufficient bankroll reserve. Keeping {self.risk_limits.min_bankroll_reserve:.1%} in reserve."
            )
        
        return approved_recommendations, risk_warnings
    
    def generate_risk_report(
        self,
        portfolio_risk: PortfolioRisk,
        position_recommendations: List[PositionSizeRecommendation]
    ) -> str:
        """Generate a comprehensive risk report"""
        
        report_lines = [
            "RISK MANAGEMENT REPORT",
            "=" * 30,
            "",
            f"Portfolio Risk Score: {portfolio_risk.overall_risk_score:.1%}",
            f"Total Exposure: {portfolio_risk.total_exposure:.1%}",
            f"Position Count: {portfolio_risk.position_count}",
            f"Concentration Risk: {portfolio_risk.concentration_risk:.1%}",
            f"Correlation Risk: {portfolio_risk.correlation_risk:.1%}",
            "",
            "Position Recommendations:",
            "-" * 25
        ]
        
        for rec in position_recommendations:
            report_lines.extend([
                f"Contract: {rec.contract_id}",
                f"  Recommended Size: {rec.adjusted_fraction:.1%} (${rec.dollar_amount:.0f})",
                f"  Kelly Fraction: {rec.kelly_fraction:.1%}",
                f"  Adjustments: Confidence={rec.confidence_adjustment:.1%}, Risk={rec.risk_adjustment:.1%}",
                f"  Reasoning: {rec.reasoning}",
            ])
            
            if rec.warnings:
                report_lines.append(f"  Warnings: {'; '.join(rec.warnings)}")
            
            report_lines.append("")
        
        # Risk level assessment
        if portfolio_risk.overall_risk_score < 0.3:
            risk_level = "LOW"
        elif portfolio_risk.overall_risk_score < 0.6:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        report_lines.extend([
            f"Overall Risk Level: {risk_level}",
            "",
            "Risk Management Status: ACTIVE" if portfolio_risk.overall_risk_score < 0.8 else "⚠️  HIGH RISK - REVIEW POSITIONS"
        ])
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # Demo usage
    from src.trading.kalshi_contract_analyzer import create_sample_contracts, KalshiContractAnalyzer
    
    print("Position Sizing and Risk Management Demo")
    print("=" * 50)
    
    # Create sample analysis
    analyzer = KalshiContractAnalyzer()
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 87.0, 2.0, 0.85)
    
    # Initialize position sizer and risk manager
    position_sizer = PositionSizer()
    risk_manager = RiskManager()
    
    # Calculate position sizes
    bankroll = 10000  # $10,000 bankroll
    current_positions = {"EXISTING_CONTRACT": 1500}  # $1,500 existing position
    
    recommendations = []
    for analysis in analyses[:2]:  # Top 2 opportunities
        if analysis.recommendation == "BUY":
            rec = position_sizer.calculate_position_size(
                analysis, bankroll, 0.85, current_positions
            )
            recommendations.append(rec)
    
    # Check risk limits
    approved_recs, warnings = risk_manager.check_risk_limits(
        recommendations, current_positions, bankroll
    )
    
    # Assess portfolio risk
    portfolio_risk = risk_manager.assess_portfolio_risk(current_positions, bankroll)
    
    # Generate report
    report = risk_manager.generate_risk_report(portfolio_risk, approved_recs)
    print(report)
    
    if warnings:
        print("\nRisk Warnings:")
        for warning in warnings:
            print(f"⚠️  {warning}")