"""
Trading Recommendation Engine and Alert System

This module generates comprehensive trading recommendations with reasoning,
confidence scoring, and alert system for significant prediction changes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime, timedelta
from enum import Enum
import json
import logging

from src.trading.kalshi_contract_analyzer import ContractAnalysis
from src.trading.position_sizing import PositionSizeRecommendation


class RecommendationStrength(Enum):
    """Recommendation strength levels"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class AlertType(Enum):
    """Types of alerts"""
    PREDICTION_CHANGE = "PREDICTION_CHANGE"
    NEW_OPPORTUNITY = "NEW_OPPORTUNITY"
    RISK_WARNING = "RISK_WARNING"
    POSITION_UPDATE = "POSITION_UPDATE"
    MODEL_CONFIDENCE = "MODEL_CONFIDENCE"


@dataclass
class TradingRecommendation:
    """Comprehensive trading recommendation"""
    contract_id: str
    contract_description: str
    recommendation: RecommendationStrength
    confidence_score: float  # 0-100
    expected_value: float
    edge: float
    position_size_pct: float  # Recommended position size as % of bankroll
    position_size_dollars: float  # Recommended position size in dollars
    
    # Detailed reasoning
    market_analysis: str
    prediction_rationale: str
    risk_assessment: str
    timing_considerations: str
    
    # Supporting data
    predicted_probability: float
    market_price: float
    confidence_interval: Tuple[float, float]
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    priority: int = 1  # 1=highest, 5=lowest
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'contract_id': self.contract_id,
            'contract_description': self.contract_description,
            'recommendation': self.recommendation.value,
            'confidence_score': self.confidence_score,
            'expected_value': self.expected_value,
            'edge': self.edge,
            'position_size_pct': self.position_size_pct,
            'position_size_dollars': self.position_size_dollars,
            'market_analysis': self.market_analysis,
            'prediction_rationale': self.prediction_rationale,
            'risk_assessment': self.risk_assessment,
            'timing_considerations': self.timing_considerations,
            'predicted_probability': self.predicted_probability,
            'market_price': self.market_price,
            'confidence_interval': self.confidence_interval,
            'generated_at': self.generated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'priority': self.priority
        }


@dataclass
class TradingAlert:
    """Trading alert for significant changes or opportunities"""
    alert_id: str
    alert_type: AlertType
    title: str
    message: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    contract_id: Optional[str] = None
    
    # Change details (for prediction changes)
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    change_magnitude: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'title': self.title,
            'message': self.message,
            'severity': self.severity,
            'contract_id': self.contract_id,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'change_magnitude': self.change_magnitude,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged
        }


class RecommendationEngine:
    """
    Generates comprehensive trading recommendations with reasoning and confidence scoring
    """
    
    def __init__(self):
        """Initialize recommendation engine"""
        self.logger = logging.getLogger(__name__)
        self.previous_predictions = {}  # Store for change detection
        self.recommendation_history = []  # Store recommendation history
    
    def generate_recommendations(
        self,
        contract_analyses: List[ContractAnalysis],
        position_recommendations: List[PositionSizeRecommendation],
        model_confidence: float,
        prediction_temp: float,
        prediction_std: float,
        market_context: Optional[Dict] = None
    ) -> List[TradingRecommendation]:
        """
        Generate comprehensive trading recommendations
        
        Args:
            contract_analyses: List of contract analyses
            position_recommendations: List of position size recommendations
            model_confidence: Overall model confidence (0-1)
            prediction_temp: Predicted temperature
            prediction_std: Prediction standard deviation
            market_context: Optional market context information
            
        Returns:
            List of trading recommendations
        """
        recommendations = []
        
        # Create position size lookup
        position_lookup = {rec.contract_id: rec for rec in position_recommendations}
        
        for analysis in contract_analyses:
            # Skip if no position recommendation (likely HOLD)
            if analysis.contract.contract_id not in position_lookup:
                continue
            
            position_rec = position_lookup[analysis.contract.contract_id]
            
            # Generate comprehensive recommendation
            recommendation = self._create_recommendation(
                analysis, position_rec, model_confidence, 
                prediction_temp, prediction_std, market_context
            )
            
            recommendations.append(recommendation)
        
        # Sort by priority and expected value
        recommendations.sort(key=lambda x: (x.priority, -x.expected_value))
        
        # Store for history tracking
        self.recommendation_history.extend(recommendations)
        
        return recommendations
    
    def _create_recommendation(
        self,
        analysis: ContractAnalysis,
        position_rec: PositionSizeRecommendation,
        model_confidence: float,
        prediction_temp: float,
        prediction_std: float,
        market_context: Optional[Dict]
    ) -> TradingRecommendation:
        """Create a comprehensive trading recommendation"""
        
        # Determine recommendation strength
        strength = self._determine_recommendation_strength(
            analysis, position_rec, model_confidence
        )
        
        # Calculate confidence score (0-100)
        confidence_score = self._calculate_confidence_score(
            analysis, model_confidence, prediction_std
        )
        
        # Generate detailed reasoning
        market_analysis = self._generate_market_analysis(analysis, market_context)
        prediction_rationale = self._generate_prediction_rationale(
            analysis, prediction_temp, prediction_std, model_confidence
        )
        risk_assessment = self._generate_risk_assessment(analysis, position_rec)
        timing_considerations = self._generate_timing_considerations(analysis)
        
        # Determine priority
        priority = self._calculate_priority(analysis, position_rec, model_confidence)
        
        # Set expiration (recommendations expire at end of day)
        expires_at = datetime.now().replace(hour=23, minute=59, second=59)
        
        return TradingRecommendation(
            contract_id=analysis.contract.contract_id,
            contract_description=analysis.contract.description,
            recommendation=strength,
            confidence_score=confidence_score,
            expected_value=analysis.expected_value,
            edge=analysis.edge,
            position_size_pct=position_rec.adjusted_fraction * 100,
            position_size_dollars=position_rec.dollar_amount,
            market_analysis=market_analysis,
            prediction_rationale=prediction_rationale,
            risk_assessment=risk_assessment,
            timing_considerations=timing_considerations,
            predicted_probability=analysis.predicted_probability,
            market_price=analysis.contract.current_price,
            confidence_interval=analysis.confidence_interval,
            expires_at=expires_at,
            priority=priority
        )
    
    def _determine_recommendation_strength(
        self,
        analysis: ContractAnalysis,
        position_rec: PositionSizeRecommendation,
        model_confidence: float
    ) -> RecommendationStrength:
        """Determine the strength of the recommendation"""
        
        edge = analysis.edge
        expected_value = analysis.expected_value
        position_size = position_rec.adjusted_fraction
        
        if analysis.recommendation == "BUY":
            if edge > 0.20 and expected_value > 0.15 and model_confidence > 0.85:
                return RecommendationStrength.STRONG_BUY
            elif edge > 0.10 and expected_value > 0.05 and position_size > 0.10:
                return RecommendationStrength.BUY
            elif edge > 0.02 and expected_value > 0.01:
                return RecommendationStrength.WEAK_BUY
            else:
                return RecommendationStrength.HOLD
        
        elif analysis.recommendation == "SELL":
            if edge < -0.20 and expected_value < -0.15:
                return RecommendationStrength.STRONG_SELL
            elif edge < -0.10 and expected_value < -0.05:
                return RecommendationStrength.SELL
            elif edge < -0.02:
                return RecommendationStrength.WEAK_SELL
            else:
                return RecommendationStrength.HOLD
        
        else:  # HOLD
            return RecommendationStrength.HOLD
    
    def _calculate_confidence_score(
        self,
        analysis: ContractAnalysis,
        model_confidence: float,
        prediction_std: float
    ) -> float:
        """Calculate overall confidence score (0-100)"""
        
        # Base confidence from model
        base_confidence = model_confidence * 100
        
        # Adjust for prediction uncertainty
        uncertainty_penalty = min(20, prediction_std * 5)  # Max 20 point penalty
        
        # Adjust for edge size (larger edge = more confidence)
        edge_bonus = min(15, abs(analysis.edge) * 30)  # Max 15 point bonus
        
        # Adjust for expected value
        ev_bonus = min(10, abs(analysis.expected_value) * 20)  # Max 10 point bonus
        
        # Calculate final score
        confidence_score = base_confidence - uncertainty_penalty + edge_bonus + ev_bonus
        
        return max(0, min(100, confidence_score))
    
    def _generate_market_analysis(
        self,
        analysis: ContractAnalysis,
        market_context: Optional[Dict]
    ) -> str:
        """Generate market analysis reasoning"""
        
        contract = analysis.contract
        market_price_pct = contract.current_price * 100
        predicted_prob_pct = analysis.predicted_probability * 100
        
        analysis_parts = [
            f"Market is pricing {contract.description} at {market_price_pct:.1f}% probability."
        ]
        
        if predicted_prob_pct > market_price_pct + 5:
            analysis_parts.append(
                f"Our model predicts {predicted_prob_pct:.1f}% probability, "
                f"suggesting the market is underpricing this contract by {predicted_prob_pct - market_price_pct:.1f} percentage points."
            )
        elif predicted_prob_pct < market_price_pct - 5:
            analysis_parts.append(
                f"Our model predicts {predicted_prob_pct:.1f}% probability, "
                f"suggesting the market is overpricing this contract by {market_price_pct - predicted_prob_pct:.1f} percentage points."
            )
        else:
            analysis_parts.append(
                f"Our model predicts {predicted_prob_pct:.1f}% probability, "
                f"which is close to market pricing."
            )
        
        # Add market context if available
        if market_context:
            if 'volume' in market_context:
                analysis_parts.append(f"Trading volume: {market_context['volume']}")
            if 'bid_ask_spread' in market_context:
                analysis_parts.append(f"Bid-ask spread: {market_context['bid_ask_spread']:.1%}")
        
        return " ".join(analysis_parts)
    
    def _generate_prediction_rationale(
        self,
        analysis: ContractAnalysis,
        prediction_temp: float,
        prediction_std: float,
        model_confidence: float
    ) -> str:
        """Generate prediction rationale"""
        
        contract = analysis.contract
        threshold = contract.threshold_temp
        
        rationale_parts = [
            f"Model predicts LA high temperature of {prediction_temp:.1f}°F ± {prediction_std:.1f}°F "
            f"with {model_confidence:.0%} confidence."
        ]
        
        if contract.contract_type == "ABOVE":
            temp_diff = prediction_temp - threshold
            if temp_diff > 2 * prediction_std:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F above the {threshold}°F threshold, "
                    f"well outside the uncertainty range."
                )
            elif temp_diff > prediction_std:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F above the {threshold}°F threshold, "
                    f"moderately above the uncertainty range."
                )
            elif temp_diff > 0:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F above the {threshold}°F threshold, "
                    f"but within the uncertainty range."
                )
            else:
                rationale_parts.append(
                    f"Predicted temperature is {abs(temp_diff):.1f}°F below the {threshold}°F threshold."
                )
        
        else:  # BELOW
            temp_diff = threshold - prediction_temp
            if temp_diff > 2 * prediction_std:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F below the {threshold}°F threshold, "
                    f"well outside the uncertainty range."
                )
            elif temp_diff > prediction_std:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F below the {threshold}°F threshold, "
                    f"moderately below the uncertainty range."
                )
            elif temp_diff > 0:
                rationale_parts.append(
                    f"Predicted temperature is {temp_diff:.1f}°F below the {threshold}°F threshold, "
                    f"but within the uncertainty range."
                )
            else:
                rationale_parts.append(
                    f"Predicted temperature is {abs(temp_diff):.1f}°F above the {threshold}°F threshold."
                )
        
        return " ".join(rationale_parts)
    
    def _generate_risk_assessment(
        self,
        analysis: ContractAnalysis,
        position_rec: PositionSizeRecommendation
    ) -> str:
        """Generate risk assessment"""
        
        risk_parts = []
        
        # Position size risk
        position_pct = position_rec.adjusted_fraction * 100
        if position_pct > 20:
            risk_parts.append(f"Large position size ({position_pct:.1f}% of bankroll) increases risk.")
        elif position_pct > 10:
            risk_parts.append(f"Moderate position size ({position_pct:.1f}% of bankroll).")
        else:
            risk_parts.append(f"Conservative position size ({position_pct:.1f}% of bankroll).")
        
        # Edge risk
        edge_pct = abs(analysis.edge) * 100
        if edge_pct > 20:
            risk_parts.append("Large edge suggests high conviction but verify model accuracy.")
        elif edge_pct > 10:
            risk_parts.append("Moderate edge provides reasonable risk-reward ratio.")
        else:
            risk_parts.append("Small edge requires careful consideration of transaction costs.")
        
        # Confidence interval risk
        ci_width = analysis.confidence_interval[1] - analysis.confidence_interval[0]
        if ci_width > 0.4:
            risk_parts.append("Wide confidence interval indicates high uncertainty.")
        elif ci_width > 0.2:
            risk_parts.append("Moderate confidence interval suggests some uncertainty.")
        else:
            risk_parts.append("Narrow confidence interval indicates high certainty.")
        
        # Warnings from position sizing
        if position_rec.warnings:
            risk_parts.append(f"Additional considerations: {'; '.join(position_rec.warnings)}")
        
        return " ".join(risk_parts)
    
    def _generate_timing_considerations(self, analysis: ContractAnalysis) -> str:
        """Generate timing considerations"""
        
        contract = analysis.contract
        expiry = contract.expiry_date
        today = date.today()
        
        if expiry == today:
            return "Contract expires today - execute immediately if trading."
        elif expiry == today + timedelta(days=1):
            return "Contract expires tomorrow - monitor for any prediction updates."
        else:
            days_to_expiry = (expiry - today).days
            return f"Contract expires in {days_to_expiry} days - time to monitor for better entry points."
    
    def _calculate_priority(
        self,
        analysis: ContractAnalysis,
        position_rec: PositionSizeRecommendation,
        model_confidence: float
    ) -> int:
        """Calculate recommendation priority (1=highest, 5=lowest)"""
        
        # Base priority on expected value and edge
        score = analysis.expected_value * 10 + abs(analysis.edge) * 5
        
        # Adjust for position size
        score += position_rec.adjusted_fraction * 2
        
        # Adjust for model confidence
        score += model_confidence
        
        # Convert to priority (higher score = higher priority)
        if score > 2.0:
            return 1  # Highest priority
        elif score > 1.0:
            return 2  # High priority
        elif score > 0.5:
            return 3  # Medium priority
        elif score > 0.1:
            return 4  # Low priority
        else:
            return 5  # Lowest priority


class AlertSystem:
    """
    Alert system for significant prediction changes and trading opportunities
    """
    
    def __init__(self):
        """Initialize alert system"""
        self.logger = logging.getLogger(__name__)
        self.previous_predictions = {}  # {date: prediction_data}
        self.active_alerts = []
        self.alert_history = []
    
    def check_for_alerts(
        self,
        current_prediction: float,
        current_confidence: float,
        recommendations: List[TradingRecommendation],
        prediction_date: date = None
    ) -> List[TradingAlert]:
        """
        Check for conditions that should trigger alerts
        
        Args:
            current_prediction: Current temperature prediction
            current_confidence: Current model confidence
            recommendations: Current trading recommendations
            prediction_date: Date of prediction (defaults to today)
            
        Returns:
            List of new alerts
        """
        if prediction_date is None:
            prediction_date = date.today()
        
        new_alerts = []
        
        # Check for significant prediction changes
        prediction_alerts = self._check_prediction_changes(
            current_prediction, current_confidence, prediction_date
        )
        new_alerts.extend(prediction_alerts)
        
        # Check for new high-value opportunities
        opportunity_alerts = self._check_new_opportunities(recommendations)
        new_alerts.extend(opportunity_alerts)
        
        # Check for risk warnings
        risk_alerts = self._check_risk_warnings(recommendations)
        new_alerts.extend(risk_alerts)
        
        # Check for model confidence changes
        confidence_alerts = self._check_confidence_changes(
            current_confidence, prediction_date
        )
        new_alerts.extend(confidence_alerts)
        
        # Store current prediction for next comparison
        self.previous_predictions[prediction_date] = {
            'prediction': current_prediction,
            'confidence': current_confidence,
            'timestamp': datetime.now()
        }
        
        # Add to active alerts and history
        self.active_alerts.extend(new_alerts)
        self.alert_history.extend(new_alerts)
        
        return new_alerts
    
    def _check_prediction_changes(
        self,
        current_prediction: float,
        current_confidence: float,
        prediction_date: date
    ) -> List[TradingAlert]:
        """Check for significant prediction changes"""
        
        alerts = []
        
        # Get previous prediction for same date
        if prediction_date in self.previous_predictions:
            prev_data = self.previous_predictions[prediction_date]
            prev_prediction = prev_data['prediction']
            
            # Calculate change magnitude
            change = abs(current_prediction - prev_prediction)
            change_pct = change / prev_prediction if prev_prediction != 0 else 0
            
            # Alert thresholds
            if change >= 5.0:  # 5+ degree change
                severity = "HIGH" if change >= 8.0 else "MEDIUM"
                direction = "increased" if current_prediction > prev_prediction else "decreased"
                
                alert = TradingAlert(
                    alert_id=f"PRED_CHANGE_{prediction_date}_{datetime.now().strftime('%H%M%S')}",
                    alert_type=AlertType.PREDICTION_CHANGE,
                    title=f"Significant Temperature Prediction Change",
                    message=(
                        f"Temperature prediction for {prediction_date} has {direction} "
                        f"by {change:.1f}°F (from {prev_prediction:.1f}°F to {current_prediction:.1f}°F). "
                        f"Review trading positions and consider adjustments."
                    ),
                    severity=severity,
                    old_value=prev_prediction,
                    new_value=current_prediction,
                    change_magnitude=change
                )
                alerts.append(alert)
            
            elif change_pct >= 0.05:  # 5%+ relative change
                alert = TradingAlert(
                    alert_id=f"PRED_CHANGE_PCT_{prediction_date}_{datetime.now().strftime('%H%M%S')}",
                    alert_type=AlertType.PREDICTION_CHANGE,
                    title=f"Temperature Prediction Updated",
                    message=(
                        f"Temperature prediction for {prediction_date} changed "
                        f"from {prev_prediction:.1f}°F to {current_prediction:.1f}°F "
                        f"({change_pct:.1%} change). Monitor for trading implications."
                    ),
                    severity="MEDIUM",
                    old_value=prev_prediction,
                    new_value=current_prediction,
                    change_magnitude=change
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_new_opportunities(
        self,
        recommendations: List[TradingRecommendation]
    ) -> List[TradingAlert]:
        """Check for new high-value trading opportunities"""
        
        alerts = []
        
        # Find high-value opportunities
        strong_recommendations = [
            rec for rec in recommendations
            if rec.recommendation in [RecommendationStrength.STRONG_BUY, RecommendationStrength.STRONG_SELL]
            and rec.confidence_score >= 80
            and rec.expected_value > 0.10
        ]
        
        for rec in strong_recommendations:
            alert = TradingAlert(
                alert_id=f"OPPORTUNITY_{rec.contract_id}_{datetime.now().strftime('%H%M%S')}",
                alert_type=AlertType.NEW_OPPORTUNITY,
                title=f"High-Value Trading Opportunity",
                message=(
                    f"Strong {rec.recommendation.value.replace('_', ' ').title()} opportunity identified: "
                    f"{rec.contract_description}. "
                    f"Expected value: ${rec.expected_value:.3f}, "
                    f"Confidence: {rec.confidence_score:.0f}%, "
                    f"Recommended position: {rec.position_size_pct:.1f}% of bankroll."
                ),
                severity="HIGH" if rec.confidence_score >= 90 else "MEDIUM",
                contract_id=rec.contract_id
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_risk_warnings(
        self,
        recommendations: List[TradingRecommendation]
    ) -> List[TradingAlert]:
        """Check for risk-related warnings"""
        
        alerts = []
        
        # Check for large position sizes
        large_positions = [
            rec for rec in recommendations
            if rec.position_size_pct > 20  # >20% of bankroll
        ]
        
        for rec in large_positions:
            alert = TradingAlert(
                alert_id=f"RISK_LARGE_POS_{rec.contract_id}_{datetime.now().strftime('%H%M%S')}",
                alert_type=AlertType.RISK_WARNING,
                title=f"Large Position Size Warning",
                message=(
                    f"Recommended position size for {rec.contract_description} "
                    f"is {rec.position_size_pct:.1f}% of bankroll (${rec.position_size_dollars:,.0f}). "
                    f"Ensure you're comfortable with this level of risk exposure."
                ),
                severity="MEDIUM",
                contract_id=rec.contract_id
            )
            alerts.append(alert)
        
        # Check for low confidence recommendations
        low_confidence = [
            rec for rec in recommendations
            if rec.confidence_score < 60 and rec.position_size_pct > 5
        ]
        
        for rec in low_confidence:
            alert = TradingAlert(
                alert_id=f"RISK_LOW_CONF_{rec.contract_id}_{datetime.now().strftime('%H%M%S')}",
                alert_type=AlertType.RISK_WARNING,
                title=f"Low Confidence Warning",
                message=(
                    f"Recommendation for {rec.contract_description} has low confidence "
                    f"({rec.confidence_score:.0f}%) but suggests {rec.position_size_pct:.1f}% position. "
                    f"Consider reducing position size or waiting for better data."
                ),
                severity="MEDIUM",
                contract_id=rec.contract_id
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_confidence_changes(
        self,
        current_confidence: float,
        prediction_date: date
    ) -> List[TradingAlert]:
        """Check for significant model confidence changes"""
        
        alerts = []
        
        if prediction_date in self.previous_predictions:
            prev_confidence = self.previous_predictions[prediction_date]['confidence']
            confidence_change = abs(current_confidence - prev_confidence)
            
            if confidence_change >= 0.15:  # 15+ percentage point change
                direction = "increased" if current_confidence > prev_confidence else "decreased"
                severity = "HIGH" if confidence_change >= 0.25 else "MEDIUM"
                
                alert = TradingAlert(
                    alert_id=f"CONF_CHANGE_{prediction_date}_{datetime.now().strftime('%H%M%S')}",
                    alert_type=AlertType.MODEL_CONFIDENCE,
                    title=f"Model Confidence Change",
                    message=(
                        f"Model confidence has {direction} by {confidence_change:.1%} "
                        f"(from {prev_confidence:.1%} to {current_confidence:.1%}). "
                        f"Review position sizes and trading recommendations."
                    ),
                    severity=severity,
                    old_value=prev_confidence,
                    new_value=current_confidence,
                    change_magnitude=confidence_change
                )
                alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[TradingAlert]:
        """Get active alerts, optionally filtered by severity"""
        if severity_filter:
            return [alert for alert in self.active_alerts 
                   if alert.severity == severity_filter and not alert.acknowledged]
        return [alert for alert in self.active_alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_old_alerts(self, hours_old: int = 24) -> int:
        """Clear acknowledged alerts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        initial_count = len(self.active_alerts)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not alert.acknowledged or alert.created_at > cutoff_time
        ]
        
        return initial_count - len(self.active_alerts)


if __name__ == "__main__":
    # Demo usage
    from src.trading.kalshi_contract_analyzer import KalshiContractAnalyzer, create_sample_contracts
    from src.trading.position_sizing import PositionSizer
    
    print("Trading Recommendation Engine and Alert System Demo")
    print("=" * 60)
    
    # Set up components
    analyzer = KalshiContractAnalyzer()
    position_sizer = PositionSizer()
    recommendation_engine = RecommendationEngine()
    alert_system = AlertSystem()
    
    # Generate sample data
    contracts = create_sample_contracts()
    analyses = analyzer.analyze_contracts(contracts, 89.0, 2.0, 0.85)
    
    # Get position recommendations for BUY opportunities
    position_recs = []
    for analysis in analyses:
        if analysis.recommendation == "BUY":
            rec = position_sizer.calculate_position_size(analysis, 10000, 0.85)
            position_recs.append(rec)
    
    # Generate trading recommendations
    recommendations = recommendation_engine.generate_recommendations(
        analyses, position_recs, 0.85, 89.0, 2.0
    )
    
    print(f"Generated {len(recommendations)} trading recommendations:")
    print()
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.contract_description}")
        print(f"   Recommendation: {rec.recommendation.value}")
        print(f"   Confidence Score: {rec.confidence_score:.0f}%")
        print(f"   Expected Value: ${rec.expected_value:.3f}")
        print(f"   Position Size: {rec.position_size_pct:.1f}%")
        print(f"   Market Analysis: {rec.market_analysis}")
        print(f"   Priority: {rec.priority}")
        print()
    
    # Test alert system
    print("Testing Alert System:")
    print("-" * 30)
    
    # Simulate prediction change
    alerts = alert_system.check_for_alerts(89.0, 0.85, recommendations)
    print(f"Initial check: {len(alerts)} alerts")
    
    # Simulate significant change
    alerts = alert_system.check_for_alerts(94.0, 0.90, recommendations)  # 5 degree increase
    print(f"After 5°F increase: {len(alerts)} alerts")
    
    for alert in alerts:
        print(f"  {alert.severity}: {alert.title}")
        print(f"    {alert.message}")
    
    print("\nDemo complete!")