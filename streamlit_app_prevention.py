"""
Enhanced Streamlit App for Churn Prevention System
Integrates prediction, root cause analysis, and intervention management
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.churn_prevention_engine import (
    ChurnPreventionEngine, Customer, RiskLevel, IssueCategory
)
from src.intervention_tracker import InterventionTracker, InterventionStatus, OutcomeType

# Page configuration
st.set_page_config(
    page_title="Comcast Churn Prevention System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .risk-critical { background-color: #ff4444; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-high { background-color: #ff8800; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-medium { background-color: #ffaa00; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-low { background-color: #88cc00; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-minimal { background-color: #00cc66; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .intervention-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load the churn prevention engine"""
    return ChurnPreventionEngine(
        model_path="Data/processed/model.pkl",
        vectorizer_path="Data/processed/tfidf.pkl"
    )


@st.cache_resource
def load_tracker():
    """Load the intervention tracker"""
    return InterventionTracker()


def render_risk_badge(risk_level: str, risk_score: int):
    """Render a risk level badge"""
    css_class = f"risk-{risk_level.lower()}"
    st.markdown(f'<div class="{css_class}">⚠️ {risk_level.upper()} RISK: {risk_score}/100</div>', 
                unsafe_allow_html=True)


def render_intervention_card(intervention: dict, idx: int):
    """Render an intervention recommendation card"""
    st.markdown(f"""
    <div class="intervention-card">
        <h4>#{idx} {intervention['action_type'].replace('_', ' ').title()} 
            <span style="color: {'red' if intervention['priority'] == 'critical' else 'orange' if intervention['priority'] == 'high' else 'gray'};">
                [{intervention['priority'].upper()}]
            </span>
        </h4>
        <p><strong>{intervention['description']}</strong></p>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <div>💰 Cost: ${intervention['estimated_cost']:,.2f}</div>
            <div>📈 Expected Retention: {intervention['expected_retention_rate']:.1%}</div>
            <div>💵 ROI: ${intervention['roi']:,.2f}</div>
        </div>
        <div style="margin-top: 10px;">
            <div>⏰ Timeline: {intervention['timeline']}</div>
            <div>📞 Channel: {intervention['channel'].upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.title("🛡️ Comcast Churn Prevention System")
    st.markdown("**Predict churn risk, identify issues, and recommend retention actions**")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🎯 Churn Analysis", "📊 Intervention Dashboard", "📈 Analytics & Reports", "ℹ️ System Info"]
    )
    
    # Load engine and tracker
    engine = load_engine()
    tracker = load_tracker()
    
    if page == "🎯 Churn Analysis":
        render_churn_analysis_page(engine, tracker)
    elif page == "📊 Intervention Dashboard":
        render_intervention_dashboard(tracker)
    elif page == "📈 Analytics & Reports":
        render_analytics_page(tracker)
    else:
        render_system_info_page()


def render_churn_analysis_page(engine, tracker):
    """Main churn analysis and prevention page"""
    st.header("Customer Churn Risk Analysis")
    
    # Customer information form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        customer_id = st.text_input("Customer ID", value="CUST_" + datetime.now().strftime("%Y%m%d%H%M"))
        name = st.text_input("Customer Name", value="John Doe")
        email = st.text_input("Email", value="john.doe@example.com")
        phone = st.text_input("Phone", value="+1-555-0100")
    
    with col2:
        st.subheader("Account Details")
        tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=240, value=12)
        monthly_revenue = st.number_input("Monthly Revenue ($)", min_value=0.0, value=150.0, step=10.0)
        contract_type = st.selectbox("Contract Type", ["monthly", "annual", "2-year"])
        total_tickets = st.number_input("Total Support Tickets", min_value=0, value=2)
        avg_resolution_time = st.number_input("Avg Resolution Time (hours)", min_value=0.0, value=24.0)
    
    # Customer review/complaint
    st.subheader("Customer Feedback/Complaint")
    review_text = st.text_area(
        "Enter customer review, complaint, or support ticket text",
        value="Service has been terrible lately. Internet keeps dropping and customer support is unhelpful. Bills are too expensive.",
        height=150
    )
    
    if st.button("🔍 Analyze Churn Risk", type="primary"):
        with st.spinner("Analyzing customer churn risk..."):
            # Create customer object
            customer = Customer(
                customer_id=customer_id,
                name=name,
                email=email,
                phone=phone,
                tenure_months=tenure_months,
                monthly_revenue=monthly_revenue,
                contract_type=contract_type,
                last_interaction=datetime.now(),
                total_tickets=total_tickets,
                avg_resolution_time=avg_resolution_time
            )
            
            # Generate prevention plan
            plan = engine.generate_prevention_plan(review_text, customer)
            
            # Store in session state
            st.session_state.current_plan = plan
            st.session_state.current_customer = customer
            st.session_state.current_review = review_text
            
            st.success("✅ Analysis complete!")
    
    # Display results
    if 'current_plan' in st.session_state:
        plan = st.session_state.current_plan
        
        st.markdown("---")
        st.header("📊 Risk Assessment Results")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_risk_badge(plan['prediction']['risk_level'], plan['prediction']['risk_score'])
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Churn Probability", f"{plan['prediction']['churn_probability']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sentiment Score", f"{plan['prediction']['sentiment_score']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Potential Loss", f"${plan['prediction']['potential_revenue_loss']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predicted churn date
        if plan['prediction']['predicted_churn_date']:
            st.warning(f"⏰ **Predicted Churn Date:** {plan['prediction']['predicted_churn_date']}")
        
        # Root causes
        st.markdown("---")
        st.header("🔍 Root Cause Analysis")
        
        if plan['root_causes']:
            for rc in plan['root_causes']:
                severity_color = 'red' if rc['severity'] > 0.7 else 'orange' if rc['severity'] > 0.4 else 'yellow'
                st.markdown(f"""
                **{rc['category'].replace('_', ' ').title()}** 
                <span style="color: {severity_color};">● Severity: {rc['severity']:.0%}</span>
                
                *{rc['description']}*
                
                Keywords detected: `{', '.join(rc['keywords'])}`
                """, unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No specific issues identified in the review text.")
        
        # Intervention recommendations
        st.header("💡 Recommended Interventions")
        
        if plan['interventions']:
            st.markdown(f"""
            **Summary:**
            - Total estimated cost: ${plan['summary']['total_estimated_cost']:,.2f}
            - Expected retention probability: {plan['summary']['expected_retention_probability']:.1%}
            - Urgency: {plan['summary']['urgency'].upper()}
            """)
            
            for idx, intervention in enumerate(plan['interventions'][:5], 1):
                render_intervention_card(intervention, idx)
                
                # Add button to execute intervention
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button(f"📋 Schedule Intervention #{idx}", key=f"schedule_{idx}"):
                        # Create intervention in tracker
                        int_id = tracker.create_intervention(
                            customer_id=plan['customer']['id'],
                            action_type=intervention['action_type'],
                            priority=intervention['priority'],
                            description=intervention['description'],
                            estimated_cost=intervention['estimated_cost'],
                            expected_retention_rate=intervention['expected_retention_rate'],
                            channel=intervention['channel'],
                            schedule_datetime=datetime.now() + timedelta(hours=2),
                            metadata={
                                'risk_score': plan['prediction']['risk_score'],
                                'churn_probability': plan['prediction']['churn_probability'],
                                'potential_loss': plan['prediction']['potential_revenue_loss'],
                                'offer_details': intervention.get('offer_details', {})
                            }
                        )
                        st.success(f"✅ Intervention scheduled: {int_id}")
        else:
            st.info("No interventions recommended at this risk level. Continue monitoring.")


def render_intervention_dashboard(tracker):
    """Dashboard for managing interventions"""
    st.header("📊 Intervention Management Dashboard")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["⏳ Pending", "⚠️ Overdue", "✅ Completed"])
    
    with tab1:
        st.subheader("Pending Interventions")
        pending = tracker.get_pending_interventions()
        
        if len(pending) > 0:
            # Filter by priority
            priority_filter = st.multiselect(
                "Filter by priority",
                ["critical", "high", "medium", "low"],
                default=["critical", "high"]
            )
            
            filtered = pending[pending['priority'].isin(priority_filter)]
            
            st.dataframe(
                filtered[['intervention_id', 'customer_id', 'action_type', 'priority', 
                         'description', 'estimated_cost', 'channel', 'created_at']],
                use_container_width=True
            )
            
            # Action buttons
            st.markdown("### Quick Actions")
            selected_id = st.selectbox("Select intervention", filtered['intervention_id'].tolist())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Start Execution"):
                    tracker.update_status(selected_id, InterventionStatus.IN_PROGRESS, "Execution started")
                    st.success(f"Started: {selected_id}")
                    st.rerun()
            
            with col2:
                if st.button("✅ Mark Complete"):
                    tracker.update_status(selected_id, InterventionStatus.COMPLETED, "Completed successfully")
                    st.success(f"Completed: {selected_id}")
                    st.rerun()
            
            with col3:
                if st.button("❌ Cancel"):
                    tracker.update_status(selected_id, InterventionStatus.CANCELLED, "Cancelled by user")
                    st.warning(f"Cancelled: {selected_id}")
                    st.rerun()
        else:
            st.info("No pending interventions")
    
    with tab2:
        st.subheader("Overdue Interventions")
        overdue = tracker.get_overdue_interventions()
        
        if len(overdue) > 0:
            st.warning(f"⚠️ {len(overdue)} overdue interventions require attention!")
            st.dataframe(
                overdue[['intervention_id', 'customer_id', 'action_type', 'priority', 
                        'scheduled_for', 'description']],
                use_container_width=True
            )
        else:
            st.success("✅ No overdue interventions")
    
    with tab3:
        st.subheader("Completed Interventions")
        
        if len(tracker.records) > 0:
            completed = tracker.records[tracker.records['outcome'] != 'pending']
            
            if len(completed) > 0:
                st.dataframe(
                    completed[['intervention_id', 'customer_id', 'action_type', 'outcome',
                              'customer_retained', 'actual_cost', 'revenue_saved', 'completed_at']],
                    use_container_width=True
                )
                
                # Outcome recording
                st.markdown("### Record Outcome")
                int_id = st.selectbox("Intervention ID", completed['intervention_id'].tolist())
                
                col1, col2 = st.columns(2)
                with col1:
                    retained = st.checkbox("Customer Retained?")
                    actual_cost = st.number_input("Actual Cost ($)", min_value=0.0, value=100.0)
                
                with col2:
                    outcome_type = st.selectbox("Outcome", ["retained", "churned", "partially_successful"])
                    revenue_saved = st.number_input("Revenue Saved ($)", min_value=0.0, value=0.0)
                
                if st.button("💾 Save Outcome"):
                    tracker.record_outcome(
                        intervention_id=int_id,
                        outcome=OutcomeType(outcome_type),
                        customer_retained=retained,
                        actual_cost=actual_cost,
                        revenue_saved=revenue_saved if retained else 0
                    )
                    st.success("✅ Outcome recorded")
                    st.rerun()
            else:
                st.info("No completed interventions yet")
        else:
            st.info("No intervention records available")


def render_analytics_page(tracker):
    """Analytics and reporting page"""
    st.header("📈 Analytics & Performance Reports")
    
    # Period selector
    period = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
    
    metrics = tracker.calculate_metrics(period_days=period)
    
    # Key metrics
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interventions", metrics['total_interventions'])
    
    with col2:
        st.metric("Retention Rate", f"{metrics['retention_rate']:.1%}")
    
    with col3:
        st.metric("Total ROI", f"{metrics['roi']:.1f}%")
    
    with col4:
        st.metric("Revenue Saved", f"${metrics['total_revenue_saved']:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Success Rate by Intervention Type")
        if metrics['success_rate_by_type']:
            df_success = pd.DataFrame([
                {'Type': k.replace('_', ' ').title(), 'Success Rate': v['success_rate']}
                for k, v in metrics['success_rate_by_type'].items()
            ])
            fig = px.bar(df_success, x='Type', y='Success Rate', 
                        title="Intervention Success Rates")
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Interventions by Priority")
        if metrics['interventions_by_priority']:
            df_priority = pd.DataFrame([
                {'Priority': k.title(), 'Count': v}
                for k, v in metrics['interventions_by_priority'].items()
            ])
            fig = px.pie(df_priority, values='Count', names='Priority',
                        title="Distribution by Priority")
            st.plotly_chart(fig, use_container_width=True)
    
    # Financial impact
    st.subheader("Financial Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Cost", f"${metrics['total_cost']:,.2f}")
        st.metric("Avg Cost per Intervention", f"${metrics['avg_cost_per_intervention']:,.2f}")
    
    with col2:
        st.metric("Total Revenue Saved", f"${metrics['total_revenue_saved']:,.2f}")
        st.metric("Avg Revenue per Retention", f"${metrics['avg_revenue_saved_per_retention']:,.2f}")
    
    # Detailed report
    st.subheader("Detailed Report")
    report_text = tracker.generate_report(period_days=period)
    st.text(report_text)


def render_system_info_page():
    """System information and documentation"""
    st.header("ℹ️ System Information")
    
    st.markdown("""
    ## Comcast Churn Prevention System
    
    ### Overview
    This system uses AI/ML to predict customer churn risk and recommend targeted retention interventions.
    
    ### Key Features
    
    #### 1. Churn Risk Prediction
    - Multi-factor risk scoring (0-100)
    - Sentiment analysis from customer reviews
    - Risk levels: Critical, High, Medium, Low, Minimal
    - Predicted churn date estimation
    
    #### 2. Root Cause Analysis
    - Automatic issue categorization
    - Categories: Billing, Service Quality, Technical, Customer Support, Pricing, Competition
    - Severity scoring for each issue
    - Keyword extraction
    
    #### 3. Intervention Recommendations
    - AI-powered action recommendations
    - ROI-optimized intervention ranking
    - Multiple channels: Call, Email, SMS, In-app
    - Cost-benefit analysis
    
    #### 4. Intervention Management
    - Track execution status
    - Record outcomes
    - Measure effectiveness
    - Real-time dashboard
    
    #### 5. Analytics & Reporting
    - Retention rate tracking
    - ROI calculation
    - Success rate by intervention type
    - Financial impact analysis
    
    ### Risk Score Calculation
    
    The risk score (0-100) combines multiple factors:
    - **Sentiment Analysis** (40%): Negative sentiment = higher risk
    - **Tenure** (20%): Shorter tenure = higher risk
    - **Support Tickets** (25%): More tickets = higher risk
    - **Resolution Time** (15%): Slower resolution = higher risk
    
    ### Intervention Types
    
    - **Executive Outreach**: VP/executive call for high-value customers
    - **Retention Call**: Specialist call with retention offer
    - **Service Credit**: Bill credit or service upgrade
    - **Technical Support**: Priority tech support visit
    - **Billing Review**: Account audit and correction
    - **Loyalty Discount**: Long-term discount offer
    - **Proactive Check-in**: Customer success outreach
    
    ### System Architecture
    
    ```
    Customer Review/Complaint
            ↓
    [Sentiment Analysis Model]
            ↓
    [Multi-Factor Risk Scoring]
            ↓
    [Root Cause Extraction]
            ↓
    [Intervention Recommendation Engine]
            ↓
    [Execution & Tracking]
            ↓
    [Outcome Measurement]
    ```
    
    ### Model Information
    - **Algorithm**: Logistic Regression with TF-IDF
    - **Accuracy**: 98.5%
    - **Features**: 5000 TF-IDF features
    - **Training Data**: Comcast customer reviews
    
    ### Contact & Support
    For questions or issues, contact the Data Science team.
    """)


if __name__ == "__main__":
    main()
