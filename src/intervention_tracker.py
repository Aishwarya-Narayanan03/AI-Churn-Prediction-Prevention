"""
Intervention Tracker
Tracks execution and outcomes of churn prevention interventions
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os


class InterventionStatus(Enum):
    """Status of intervention execution"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutcomeType(Enum):
    """Outcome of intervention"""
    RETAINED = "retained"
    CHURNED = "churned"
    PENDING = "pending"
    PARTIALLY_SUCCESSFUL = "partially_successful"


@dataclass
class InterventionRecord:
    """Record of an intervention action"""
    intervention_id: str
    customer_id: str
    action_type: str
    priority: str
    description: str
    estimated_cost: float
    expected_retention_rate: float
    channel: str
    
    created_at: datetime
    scheduled_for: Optional[datetime]
    executed_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    status: InterventionStatus
    outcome: OutcomeType
    
    actual_cost: Optional[float]
    customer_retained: Optional[bool]
    revenue_saved: Optional[float]
    
    notes: str
    metadata: Dict


class InterventionTracker:
    """
    Tracks and analyzes intervention executions and outcomes
    """
    
    def __init__(self, storage_path: str = "Data/interventions/"):
        """Initialize tracker with storage location"""
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.records_file = os.path.join(storage_path, "intervention_records.csv")
        self.analytics_file = os.path.join(storage_path, "analytics.json")
        
        # Load existing records
        self.records = self._load_records()
    
    def _load_records(self) -> pd.DataFrame:
        """Load intervention records from storage"""
        if os.path.exists(self.records_file):
            df = pd.read_csv(self.records_file)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['scheduled_for'] = pd.to_datetime(df['scheduled_for'])
            df['executed_at'] = pd.to_datetime(df['executed_at'])
            df['completed_at'] = pd.to_datetime(df['completed_at'])
            return df
        else:
            return pd.DataFrame()
    
    def _save_records(self):
        """Save records to storage"""
        self.records.to_csv(self.records_file, index=False)
    
    def create_intervention(self, customer_id: str, action_type: str, priority: str,
                          description: str, estimated_cost: float, 
                          expected_retention_rate: float, channel: str,
                          schedule_datetime: Optional[datetime] = None,
                          metadata: Dict = None) -> str:
        """
        Create a new intervention record
        
        Args:
            customer_id: Customer identifier
            action_type: Type of intervention
            priority: Priority level
            description: Action description
            estimated_cost: Estimated cost
            expected_retention_rate: Expected success rate
            channel: Communication channel
            schedule_datetime: When to execute (None = immediate)
            metadata: Additional metadata
            
        Returns:
            intervention_id
        """
        intervention_id = f"INT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{customer_id}"
        
        record = {
            'intervention_id': intervention_id,
            'customer_id': customer_id,
            'action_type': action_type,
            'priority': priority,
            'description': description,
            'estimated_cost': estimated_cost,
            'expected_retention_rate': expected_retention_rate,
            'channel': channel,
            'created_at': datetime.now(),
            'scheduled_for': schedule_datetime,
            'executed_at': None,
            'completed_at': None,
            'status': InterventionStatus.SCHEDULED.value if schedule_datetime else InterventionStatus.PENDING.value,
            'outcome': OutcomeType.PENDING.value,
            'actual_cost': None,
            'customer_retained': None,
            'revenue_saved': None,
            'notes': '',
            'metadata': json.dumps(metadata or {})
        }
        
        self.records = pd.concat([self.records, pd.DataFrame([record])], ignore_index=True)
        self._save_records()
        
        return intervention_id
    
    def update_status(self, intervention_id: str, status: InterventionStatus, notes: str = ""):
        """Update intervention status"""
        idx = self.records[self.records['intervention_id'] == intervention_id].index
        
        if len(idx) == 0:
            raise ValueError(f"Intervention {intervention_id} not found")
        
        self.records.loc[idx, 'status'] = status.value
        
        if notes:
            current_notes = self.records.loc[idx, 'notes'].values[0]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            new_notes = f"{current_notes}\n[{timestamp}] {notes}" if current_notes else f"[{timestamp}] {notes}"
            self.records.loc[idx, 'notes'] = new_notes
        
        if status == InterventionStatus.IN_PROGRESS:
            self.records.loc[idx, 'executed_at'] = datetime.now()
        elif status in [InterventionStatus.COMPLETED, InterventionStatus.FAILED]:
            self.records.loc[idx, 'completed_at'] = datetime.now()
        
        self._save_records()
    
    def record_outcome(self, intervention_id: str, outcome: OutcomeType, 
                      customer_retained: bool, actual_cost: float,
                      revenue_saved: float = None):
        """
        Record the outcome of an intervention
        
        Args:
            intervention_id: Intervention identifier
            outcome: Outcome type
            customer_retained: Whether customer was retained
            actual_cost: Actual cost incurred
            revenue_saved: Revenue saved (if retained)
        """
        idx = self.records[self.records['intervention_id'] == intervention_id].index
        
        if len(idx) == 0:
            raise ValueError(f"Intervention {intervention_id} not found")
        
        self.records.loc[idx, 'outcome'] = outcome.value
        self.records.loc[idx, 'customer_retained'] = customer_retained
        self.records.loc[idx, 'actual_cost'] = actual_cost
        self.records.loc[idx, 'revenue_saved'] = revenue_saved
        self.records.loc[idx, 'status'] = InterventionStatus.COMPLETED.value
        self.records.loc[idx, 'completed_at'] = datetime.now()
        
        self._save_records()
    
    def get_pending_interventions(self, priority: Optional[str] = None) -> pd.DataFrame:
        """Get all pending interventions, optionally filtered by priority"""
        pending = self.records[
            self.records['status'].isin([InterventionStatus.PENDING.value, InterventionStatus.SCHEDULED.value])
        ].copy()
        
        if priority:
            pending = pending[pending['priority'] == priority]
        
        return pending.sort_values('created_at')
    
    def get_overdue_interventions(self) -> pd.DataFrame:
        """Get interventions that are past their scheduled time"""
        now = datetime.now()
        overdue = self.records[
            (self.records['scheduled_for'].notna()) &
            (self.records['scheduled_for'] < now) &
            (self.records['status'] == InterventionStatus.SCHEDULED.value)
        ].copy()
        
        return overdue.sort_values('scheduled_for')
    
    def get_customer_interventions(self, customer_id: str) -> pd.DataFrame:
        """Get all interventions for a specific customer"""
        return self.records[self.records['customer_id'] == customer_id].sort_values('created_at', ascending=False)
    
    def calculate_metrics(self, period_days: int = 30) -> Dict:
        """
        Calculate intervention effectiveness metrics
        
        Args:
            period_days: Analysis period in days
            
        Returns:
            Dictionary of metrics
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent = self.records[self.records['created_at'] >= cutoff_date].copy()
        
        completed = recent[recent['outcome'] != OutcomeType.PENDING.value]
        
        if len(completed) == 0:
            return {
                'total_interventions': len(recent),
                'completed_interventions': 0,
                'retention_rate': 0,
                'total_cost': 0,
                'total_revenue_saved': 0,
                'roi': 0,
                'avg_cost_per_intervention': 0,
                'success_rate_by_type': {}
            }
        
        retained = completed[completed['customer_retained'] == True]
        
        total_cost = completed['actual_cost'].sum()
        total_revenue_saved = retained['revenue_saved'].sum() if 'revenue_saved' in retained else 0
        roi = ((total_revenue_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        # Success rate by intervention type
        success_by_type = {}
        for action_type in completed['action_type'].unique():
            type_records = completed[completed['action_type'] == action_type]
            type_retained = type_records[type_records['customer_retained'] == True]
            success_rate = len(type_retained) / len(type_records) if len(type_records) > 0 else 0
            success_by_type[action_type] = {
                'success_rate': success_rate,
                'count': len(type_records),
                'retained': len(type_retained)
            }
        
        metrics = {
            'period_days': period_days,
            'total_interventions': len(recent),
            'completed_interventions': len(completed),
            'retention_rate': len(retained) / len(completed) if len(completed) > 0 else 0,
            'total_cost': total_cost,
            'total_revenue_saved': total_revenue_saved,
            'roi': roi,
            'avg_cost_per_intervention': total_cost / len(completed) if len(completed) > 0 else 0,
            'avg_revenue_saved_per_retention': total_revenue_saved / len(retained) if len(retained) > 0 else 0,
            'success_rate_by_type': success_by_type,
            'interventions_by_priority': recent['priority'].value_counts().to_dict(),
            'interventions_by_channel': recent['channel'].value_counts().to_dict()
        }
        
        # Save analytics
        with open(self.analytics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return metrics
    
    def generate_report(self, period_days: int = 30) -> str:
        """Generate a text report of intervention performance"""
        metrics = self.calculate_metrics(period_days)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CHURN PREVENTION INTERVENTION REPORT                      ║
║                          Last {period_days} Days                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERVIEW
─────────────────────────────────────────────────────────────────────────────
Total Interventions:        {metrics['total_interventions']}
Completed Interventions:    {metrics['completed_interventions']}
Overall Retention Rate:     {metrics['retention_rate']:.1%}

FINANCIAL IMPACT
─────────────────────────────────────────────────────────────────────────────
Total Cost:                 ${metrics['total_cost']:,.2f}
Total Revenue Saved:        ${metrics['total_revenue_saved']:,.2f}
Return on Investment:       {metrics['roi']:.1f}%
Avg Cost per Intervention:  ${metrics['avg_cost_per_intervention']:,.2f}
Avg Revenue per Retention:  ${metrics['avg_revenue_saved_per_retention']:,.2f}

SUCCESS RATE BY INTERVENTION TYPE
─────────────────────────────────────────────────────────────────────────────
"""
        
        for action_type, stats in metrics['success_rate_by_type'].items():
            report += f"{action_type:30} {stats['success_rate']:>6.1%}  ({stats['retained']}/{stats['count']})\n"
        
        report += f"""
INTERVENTIONS BY PRIORITY
─────────────────────────────────────────────────────────────────────────────
"""
        for priority, count in metrics['interventions_by_priority'].items():
            report += f"{priority:20} {count:>4}\n"
        
        report += f"""
INTERVENTIONS BY CHANNEL
─────────────────────────────────────────────────────────────────────────────
"""
        for channel, count in metrics['interventions_by_channel'].items():
            report += f"{channel:20} {count:>4}\n"
        
        report += "═" * 78 + "\n"
        
        return report


# Example usage
if __name__ == "__main__":
    tracker = InterventionTracker()
    
    # Create sample interventions
    print("Creating sample interventions...")
    
    int1 = tracker.create_intervention(
        customer_id="CUST001",
        action_type="retention_call",
        priority="critical",
        description="Immediate retention call",
        estimated_cost=100,
        expected_retention_rate=0.65,
        channel="call",
        schedule_datetime=datetime.now() + timedelta(hours=2),
        metadata={"risk_score": 85, "clv": 3600}
    )
    
    int2 = tracker.create_intervention(
        customer_id="CUST002",
        action_type="service_credit",
        priority="high",
        description="Service credit offer",
        estimated_cost=200,
        expected_retention_rate=0.70,
        channel="email",
        metadata={"risk_score": 75, "clv": 4200}
    )
    
    # Update status
    tracker.update_status(int1, InterventionStatus.IN_PROGRESS, "Call initiated")
    tracker.update_status(int1, InterventionStatus.COMPLETED, "Call completed successfully")
    
    # Record outcome
    tracker.record_outcome(
        intervention_id=int1,
        outcome=OutcomeType.RETAINED,
        customer_retained=True,
        actual_cost=95,
        revenue_saved=3600
    )
    
    # Get pending interventions
    pending = tracker.get_pending_interventions()
    print(f"\nPending interventions: {len(pending)}")
    
    # Calculate metrics
    print("\n" + tracker.generate_report(period_days=7))
