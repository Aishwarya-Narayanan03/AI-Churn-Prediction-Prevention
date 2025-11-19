"""
Churn Prevention System - Core Engine
Integrates prediction, root cause analysis, and intervention recommendations
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Customer churn risk levels"""
    CRITICAL = "critical"  # 80-100
    HIGH = "high"          # 60-79
    MEDIUM = "medium"      # 40-59
    LOW = "low"            # 20-39
    MINIMAL = "minimal"    # 0-19


class IssueCategory(Enum):
    """Common customer complaint categories"""
    BILLING = "billing"
    SERVICE_QUALITY = "service_quality"
    TECHNICAL = "technical"
    CUSTOMER_SUPPORT = "customer_support"
    PRICING = "pricing"
    COMPETITION = "competition"
    OTHER = "other"


@dataclass
class Customer:
    """Customer data model"""
    customer_id: str
    name: str
    email: str
    phone: str
    tenure_months: int
    monthly_revenue: float
    contract_type: str
    last_interaction: datetime
    total_tickets: int
    avg_resolution_time: float


@dataclass
class ChurnPrediction:
    """Churn prediction result"""
    customer_id: str
    churn_probability: float
    sentiment_score: float
    risk_level: RiskLevel
    risk_score: int
    predicted_churn_date: Optional[datetime]
    confidence: float


@dataclass
class RootCause:
    """Identified issue root cause"""
    category: IssueCategory
    description: str
    severity: float  # 0-1
    keywords: List[str]
    sentiment: float


@dataclass
class Intervention:
    """Recommended intervention action"""
    action_type: str
    priority: str  # critical, high, medium, low
    description: str
    estimated_cost: float
    expected_retention_rate: float
    timeline: str
    channel: str  # email, sms, call, in-app
    template_id: Optional[str]
    offer_details: Optional[Dict]


class ChurnPreventionEngine:
    """
    Main engine for churn prevention system
    Combines prediction, analysis, and intervention recommendation
    """
    
    def __init__(self, model_path: str, vectorizer_path: str):
        """Initialize the churn prevention engine"""
        self.model = self._load_model(model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_path)
        
        # Issue detection keywords
        self.issue_keywords = {
            IssueCategory.BILLING: ['bill', 'charge', 'overcharge', 'expensive', 'cost', 'price increase', 'fee'],
            IssueCategory.SERVICE_QUALITY: ['outage', 'slow', 'unreliable', 'disconnect', 'poor service', 'down'],
            IssueCategory.TECHNICAL: ['not working', 'broken', 'error', 'technical issue', 'malfunction', 'bug'],
            IssueCategory.CUSTOMER_SUPPORT: ['support', 'wait time', 'representative', 'on hold', 'unhelpful', 'rude'],
            IssueCategory.PRICING: ['expensive', 'competitor cheaper', 'price', 'discount', 'promotional rate'],
            IssueCategory.COMPETITION: ['switch', 'competitor', 'alternative', 'other provider', 'att', 'verizon']
        }
        
    def _load_model(self, path: str):
        """Load the sentiment/churn prediction model"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_vectorizer(self, path: str):
        """Load the TF-IDF vectorizer"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            raise
    
    def predict_churn(self, review_text: str, customer: Customer) -> ChurnPrediction:
        """
        Predict churn risk for a customer based on review text and customer data
        
        Args:
            review_text: Customer review or complaint text
            customer: Customer information
            
        Returns:
            ChurnPrediction with risk assessment
        """
        # Get sentiment prediction
        text_clean = review_text.lower().strip()
        features = self.vectorizer.transform([text_clean])
        sentiment_pred = self.model.predict(features)[0]
        sentiment_proba = self.model.predict_proba(features)[0]
        
        # Sentiment score: -1 (very negative) to +1 (very positive)
        sentiment_score = sentiment_proba[1] - sentiment_proba[0]
        
        # Calculate multi-factor risk score (0-100)
        risk_score = self._calculate_risk_score(
            sentiment_score=sentiment_score,
            tenure_months=customer.tenure_months,
            total_tickets=customer.total_tickets,
            avg_resolution_time=customer.avg_resolution_time
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_score)
        
        # Estimate churn probability
        churn_probability = self._estimate_churn_probability(risk_score, sentiment_score)
        
        # Predict potential churn date
        predicted_date = self._predict_churn_date(risk_score, customer.contract_type)
        
        return ChurnPrediction(
            customer_id=customer.customer_id,
            churn_probability=churn_probability,
            sentiment_score=sentiment_score,
            risk_level=risk_level,
            risk_score=risk_score,
            predicted_churn_date=predicted_date,
            confidence=max(sentiment_proba)
        )
    
    def _calculate_risk_score(self, sentiment_score: float, tenure_months: int, 
                            total_tickets: int, avg_resolution_time: float) -> int:
        """Calculate overall churn risk score (0-100)"""
        # Base score from sentiment (inverted: negative = high risk)
        sentiment_risk = (1 - (sentiment_score + 1) / 2) * 40  # Max 40 points
        
        # Tenure risk (shorter tenure = higher risk)
        tenure_risk = max(0, (24 - tenure_months) / 24 * 20)  # Max 20 points
        
        # Support ticket risk
        ticket_risk = min(total_tickets / 10 * 25, 25)  # Max 25 points
        
        # Resolution time risk (slower = higher risk)
        resolution_risk = min(avg_resolution_time / 72 * 15, 15)  # Max 15 points
        
        total_risk = sentiment_risk + tenure_risk + ticket_risk + resolution_risk
        return int(min(100, total_risk))
    
    def _get_risk_level(self, risk_score: int) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        elif risk_score >= 20:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _estimate_churn_probability(self, risk_score: int, sentiment_score: float) -> float:
        """Estimate probability of churn (0-1)"""
        # Weighted combination
        base_prob = risk_score / 100 * 0.7
        sentiment_prob = (1 - (sentiment_score + 1) / 2) * 0.3
        return min(1.0, base_prob + sentiment_prob)
    
    def _predict_churn_date(self, risk_score: int, contract_type: str) -> Optional[datetime]:
        """Predict when customer might churn"""
        if risk_score < 40:
            return None
        
        # Days until likely churn based on risk
        if risk_score >= 80:
            days = 7  # Critical: within a week
        elif risk_score >= 60:
            days = 30  # High: within a month
        else:
            days = 90  # Medium: within 3 months
        
        # Adjust for contract type
        if contract_type == "annual":
            days *= 2  # Annual contracts have more friction
        
        return datetime.now() + timedelta(days=days)
    
    def extract_root_causes(self, review_text: str) -> List[RootCause]:
        """
        Extract root causes from review text using NLP
        
        Args:
            review_text: Customer review or complaint
            
        Returns:
            List of identified root causes with severity
        """
        text_lower = review_text.lower()
        root_causes = []
        
        for category, keywords in self.issue_keywords.items():
            # Count keyword matches
            matches = [kw for kw in keywords if kw in text_lower]
            
            if matches:
                # Calculate severity based on number of matches and sentiment
                features = self.vectorizer.transform([text_lower])
                sentiment_proba = self.model.predict_proba(features)[0]
                sentiment = sentiment_proba[1] - sentiment_proba[0]
                
                severity = min(1.0, len(matches) / 3 * (1 - sentiment))
                
                root_cause = RootCause(
                    category=category,
                    description=f"Issue related to {category.value}",
                    severity=severity,
                    keywords=matches,
                    sentiment=sentiment
                )
                root_causes.append(root_cause)
        
        # Sort by severity
        root_causes.sort(key=lambda x: x.severity, reverse=True)
        return root_causes
    
    def recommend_interventions(self, prediction: ChurnPrediction, 
                               root_causes: List[RootCause],
                               customer: Customer) -> List[Intervention]:
        """
        Recommend intervention actions based on prediction and root causes
        
        Args:
            prediction: Churn prediction results
            root_causes: Identified issue root causes
            customer: Customer information
            
        Returns:
            List of recommended interventions, prioritized
        """
        interventions = []
        
        # Determine customer value tier
        clv = customer.monthly_revenue * customer.tenure_months
        is_high_value = clv > 5000 or customer.monthly_revenue > 200
        
        # Critical risk interventions
        if prediction.risk_level == RiskLevel.CRITICAL:
            if is_high_value:
                interventions.append(Intervention(
                    action_type="executive_outreach",
                    priority="critical",
                    description="Executive/VP call within 24 hours",
                    estimated_cost=500,
                    expected_retention_rate=0.75,
                    timeline="24 hours",
                    channel="call",
                    template_id="exec_outreach_001",
                    offer_details={"type": "custom", "max_discount": "30%"}
                ))
            
            interventions.append(Intervention(
                action_type="retention_offer",
                priority="critical",
                description="Immediate service credit + priority support",
                estimated_cost=customer.monthly_revenue * 2,
                expected_retention_rate=0.65,
                timeline="immediate",
                channel="email",
                template_id="retention_critical_001",
                offer_details={
                    "credit_amount": customer.monthly_revenue * 2,
                    "priority_support": True,
                    "contract_guarantee": "6 months"
                }
            ))
        
        # High risk interventions
        elif prediction.risk_level == RiskLevel.HIGH:
            interventions.append(Intervention(
                action_type="retention_call",
                priority="high",
                description="Retention specialist call within 48 hours",
                estimated_cost=100,
                expected_retention_rate=0.60,
                timeline="48 hours",
                channel="call",
                template_id="retention_call_001",
                offer_details={"discount": "15%", "duration": "3 months"}
            ))
            
            interventions.append(Intervention(
                action_type="service_upgrade",
                priority="high",
                description="Free service upgrade for 3 months",
                estimated_cost=customer.monthly_revenue * 0.5,
                expected_retention_rate=0.55,
                timeline="3 days",
                channel="email",
                template_id="upgrade_offer_001",
                offer_details={"upgrade_tier": "premium", "duration": "3 months"}
            ))
        
        # Medium risk interventions
        elif prediction.risk_level == RiskLevel.MEDIUM:
            interventions.append(Intervention(
                action_type="proactive_checkIn",
                priority="medium",
                description="Customer success check-in call",
                estimated_cost=50,
                expected_retention_rate=0.45,
                timeline="1 week",
                channel="call",
                template_id="checkin_001",
                offer_details={"satisfaction_survey": True}
            ))
            
            interventions.append(Intervention(
                action_type="loyalty_discount",
                priority="medium",
                description="10% loyalty discount for 6 months",
                estimated_cost=customer.monthly_revenue * 0.1 * 6,
                expected_retention_rate=0.50,
                timeline="3 days",
                channel="email",
                template_id="loyalty_001",
                offer_details={"discount": "10%", "duration": "6 months"}
            ))
        
        # Issue-specific interventions
        for cause in root_causes[:2]:  # Top 2 issues
            if cause.category == IssueCategory.BILLING:
                interventions.append(Intervention(
                    action_type="billing_review",
                    priority="high",
                    description="Billing audit and correction",
                    estimated_cost=25,
                    expected_retention_rate=0.70,
                    timeline="48 hours",
                    channel="email",
                    template_id="billing_review_001",
                    offer_details={"audit": True, "credit_if_error": True}
                ))
            
            elif cause.category == IssueCategory.TECHNICAL:
                interventions.append(Intervention(
                    action_type="technical_support",
                    priority="high",
                    description="Priority technical support visit",
                    estimated_cost=150,
                    expected_retention_rate=0.65,
                    timeline="24 hours",
                    channel="sms",
                    template_id="tech_support_001",
                    offer_details={"on_site_visit": True, "priority": "urgent"}
                ))
            
            elif cause.category == IssueCategory.CUSTOMER_SUPPORT:
                interventions.append(Intervention(
                    action_type="support_escalation",
                    priority="high",
                    description="Escalate to senior support team",
                    estimated_cost=75,
                    expected_retention_rate=0.60,
                    timeline="immediate",
                    channel="call",
                    template_id="escalation_001",
                    offer_details={"dedicated_agent": True}
                ))
        
        # Sort by expected ROI (retention_rate / cost)
        interventions.sort(
            key=lambda x: (x.expected_retention_rate * customer.monthly_revenue * 12) - x.estimated_cost,
            reverse=True
        )
        
        return interventions
    
    def generate_prevention_plan(self, review_text: str, customer: Customer) -> Dict:
        """
        Generate comprehensive churn prevention plan
        
        Args:
            review_text: Customer review or complaint
            customer: Customer information
            
        Returns:
            Complete prevention plan with prediction, causes, and interventions
        """
        logger.info(f"Generating prevention plan for customer {customer.customer_id}")
        
        # Step 1: Predict churn risk
        prediction = self.predict_churn(review_text, customer)
        logger.info(f"Churn risk: {prediction.risk_level.value} ({prediction.risk_score}/100)")
        
        # Step 2: Extract root causes
        root_causes = self.extract_root_causes(review_text)
        logger.info(f"Identified {len(root_causes)} root causes")
        
        # Step 3: Recommend interventions
        interventions = self.recommend_interventions(prediction, root_causes, customer)
        logger.info(f"Generated {len(interventions)} intervention recommendations")
        
        # Calculate CLV and potential loss
        clv = customer.monthly_revenue * customer.tenure_months
        potential_loss = customer.monthly_revenue * 24  # 2-year value
        
        return {
            "customer": {
                "id": customer.customer_id,
                "name": customer.name,
                "clv": clv,
                "monthly_revenue": customer.monthly_revenue
            },
            "prediction": {
                "risk_level": prediction.risk_level.value,
                "risk_score": prediction.risk_score,
                "churn_probability": prediction.churn_probability,
                "sentiment_score": prediction.sentiment_score,
                "predicted_churn_date": prediction.predicted_churn_date.isoformat() if prediction.predicted_churn_date else None,
                "potential_revenue_loss": potential_loss
            },
            "root_causes": [
                {
                    "category": rc.category.value,
                    "description": rc.description,
                    "severity": rc.severity,
                    "keywords": rc.keywords
                }
                for rc in root_causes
            ],
            "interventions": [
                {
                    "action_type": i.action_type,
                    "priority": i.priority,
                    "description": i.description,
                    "estimated_cost": i.estimated_cost,
                    "expected_retention_rate": i.expected_retention_rate,
                    "roi": (i.expected_retention_rate * potential_loss) - i.estimated_cost,
                    "timeline": i.timeline,
                    "channel": i.channel,
                    "offer_details": i.offer_details
                }
                for i in interventions
            ],
            "summary": {
                "total_estimated_cost": sum(i.estimated_cost for i in interventions[:3]),
                "expected_retention_probability": max(i.expected_retention_rate for i in interventions),
                "recommended_action": interventions[0].description if interventions else "Monitor customer",
                "urgency": "immediate" if prediction.risk_level == RiskLevel.CRITICAL else "high" if prediction.risk_level == RiskLevel.HIGH else "normal"
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = ChurnPreventionEngine(
        model_path="Data/processed/model.pkl",
        vectorizer_path="Data/processed/tfidf.pkl"
    )
    
    # Example customer
    customer = Customer(
        customer_id="CUST001",
        name="John Doe",
        email="john.doe@example.com",
        phone="+1-555-0100",
        tenure_months=8,
        monthly_revenue=150.00,
        contract_type="monthly",
        last_interaction=datetime.now() - timedelta(days=5),
        total_tickets=3,
        avg_resolution_time=48.0
    )
    
    # Example review
    review = """
    I've been a customer for months and the service keeps getting worse. 
    The internet is constantly down and when I call support, I'm on hold for hours. 
    The bills keep increasing too. I'm seriously considering switching to a competitor.
    """
    
    # Generate prevention plan
    plan = engine.generate_prevention_plan(review, customer)
    
    print("\n" + "="*80)
    print("CHURN PREVENTION PLAN")
    print("="*80)
    print(f"\nCustomer: {plan['customer']['name']} ({plan['customer']['id']})")
    print(f"Monthly Revenue: ${plan['customer']['monthly_revenue']:.2f}")
    print(f"Customer Lifetime Value: ${plan['customer']['clv']:.2f}")
    print(f"\nRisk Assessment:")
    print(f"  Risk Level: {plan['prediction']['risk_level'].upper()}")
    print(f"  Risk Score: {plan['prediction']['risk_score']}/100")
    print(f"  Churn Probability: {plan['prediction']['churn_probability']:.1%}")
    print(f"  Potential Loss: ${plan['prediction']['potential_revenue_loss']:.2f}")
    
    print(f"\nRoot Causes Identified:")
    for rc in plan['root_causes']:
        print(f"  • {rc['category']}: {rc['description']} (Severity: {rc['severity']:.2f})")
        print(f"    Keywords: {', '.join(rc['keywords'])}")
    
    print(f"\nRecommended Interventions:")
    for idx, intervention in enumerate(plan['interventions'][:3], 1):
        print(f"\n  {idx}. {intervention['action_type'].upper()} [{intervention['priority']}]")
        print(f"     {intervention['description']}")
        print(f"     Cost: ${intervention['estimated_cost']:.2f}")
        print(f"     Expected Retention: {intervention['expected_retention_rate']:.1%}")
        print(f"     ROI: ${intervention['roi']:.2f}")
        print(f"     Timeline: {intervention['timeline']}")
        print(f"     Channel: {intervention['channel']}")
    
    print(f"\nExecution Summary:")
    print(f"  Urgency: {plan['summary']['urgency'].upper()}")
    print(f"  Recommended Action: {plan['summary']['recommended_action']}")
    print(f"  Total Investment: ${plan['summary']['total_estimated_cost']:.2f}")
    print(f"  Expected Success Rate: {plan['summary']['expected_retention_probability']:.1%}")
    print("="*80 + "\n")
