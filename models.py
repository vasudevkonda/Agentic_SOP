"""
S&OP Data Models — Pydantic schemas for all planning entities
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, date
import random

# ─── Enums ────────────────────────────────────────────────────────────────────

class ScenarioType(str, Enum):
    BASE = "base_case"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    STRESS = "stress_test"
    CUSTOM = "custom"

class ReviewCycle(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ProductFamily(str, Enum):
    ELECTRONICS = "Consumer Electronics"
    INDUSTRIAL = "Industrial Equipment"
    HEALTHCARE = "Healthcare Devices"
    AUTOMOTIVE = "Automotive Components"

class RiskTier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

# ─── Core Planning Models ──────────────────────────────────────────────────────

class DemandSignal(BaseModel):
    period: str
    product_family: str
    actual_demand: float
    forecast_demand: float
    bias: float = 0.0
    mape: float = 0.0
    channel: str = "direct"

class InventoryPosition(BaseModel):
    product_family: str
    on_hand: float
    in_transit: float
    on_order: float
    safety_stock: float
    weeks_of_supply: float
    doh: float  # Days on Hand
    excess_value: float = 0.0

class CapacityData(BaseModel):
    resource: str
    plant: str
    installed_capacity: float
    available_capacity: float
    utilized_capacity: float
    utilization_pct: float
    constraint_type: str = "machine"  # machine, labor, material

class FinancialBridge(BaseModel):
    revenue_plan: float
    revenue_latest_estimate: float
    revenue_variance: float
    gross_margin_plan: float
    gross_margin_le: float
    cogs_plan: float
    cogs_le: float
    inventory_value: float
    working_capital: float

class ScenarioAssumption(BaseModel):
    category: str
    assumption: str
    value: Any
    confidence: float
    impact: str

class Scenario(BaseModel):
    id: str
    name: str
    type: ScenarioType
    description: str
    assumptions: List[ScenarioAssumption] = []
    demand_uplift: float = 0.0        # % change vs base
    cost_uplift: float = 0.0          # % change in COGS
    margin_impact: float = 0.0        # $ impact on gross margin
    revenue_impact: float = 0.0       # $ impact on revenue
    inventory_impact: float = 0.0     # % change in inventory
    service_level_impact: float = 0.0 # % change in OTIF
    risk_tier: RiskTier = RiskTier.MODERATE
    probability: float = 0.33
    recommended_actions: List[str] = []
    created_at: str = ""
    ai_narrative: str = ""

class SOPCycleData(BaseModel):
    cycle_month: str
    review_date: str
    demand_signals: List[DemandSignal] = []
    inventory_positions: List[InventoryPosition] = []
    capacity_data: List[CapacityData] = []
    financial_bridge: Optional[FinancialBridge] = None
    scenarios: List[Scenario] = []
    consensus_forecast: Dict[str, float] = {}
    executive_summary: str = ""
    key_decisions: List[str] = []
    open_issues: List[str] = []
    agent_insights: List[Dict[str, Any]] = []
    overall_health: float = 75.0
    plan_confidence: float = 0.70

# ─── Sample Data Generator ────────────────────────────────────────────────────

def generate_sop_data() -> SOPCycleData:
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current_month = datetime.now().strftime("%B %Y")

    # Demand signals
    demand_signals = []
    families = ["Consumer Electronics", "Industrial Equipment", "Healthcare Devices", "Automotive Components"]
    for fam in families:
        base = {"Consumer Electronics": 8500, "Industrial Equipment": 3200,
                "Healthcare Devices": 1800, "Automotive Components": 5400}[fam]
        for i, m in enumerate(months[:6]):
            actual = base * (1 + 0.05 * np.sin(i) + np.random.normal(0, 0.04))
            forecast = actual * (1 + np.random.normal(0, 0.06))
            bias = (forecast - actual) / actual
            mape = abs(bias)
            demand_signals.append(DemandSignal(
                period=m, product_family=fam,
                actual_demand=round(actual, 0), forecast_demand=round(forecast, 0),
                bias=round(bias, 3), mape=round(mape, 3),
                channel=random.choice(["direct", "distributor", "ecommerce"])
            ))

    # Inventory positions
    inventory_positions = [
        InventoryPosition(
            product_family="Consumer Electronics", on_hand=24500, in_transit=8200,
            on_order=12000, safety_stock=7000, weeks_of_supply=4.2, doh=29.4,
            excess_value=185000
        ),
        InventoryPosition(
            product_family="Industrial Equipment", on_hand=9800, in_transit=2100,
            on_order=4500, safety_stock=3500, weeks_of_supply=6.1, doh=42.7,
            excess_value=420000
        ),
        InventoryPosition(
            product_family="Healthcare Devices", on_hand=5200, in_transit=800,
            on_order=2200, safety_stock=2000, weeks_of_supply=5.8, doh=40.6,
            excess_value=95000
        ),
        InventoryPosition(
            product_family="Automotive Components", on_hand=18700, in_transit=5600,
            on_order=9000, safety_stock=6000, weeks_of_supply=3.9, doh=27.3,
            excess_value=310000
        ),
    ]

    # Capacity data
    capacity_data = [
        CapacityData(resource="Assembly Line A", plant="Plant North",
                     installed_capacity=10000, available_capacity=9200,
                     utilized_capacity=8740, utilization_pct=95.0, constraint_type="machine"),
        CapacityData(resource="Assembly Line B", plant="Plant North",
                     installed_capacity=8000, available_capacity=7600,
                     utilized_capacity=6080, utilization_pct=80.0, constraint_type="machine"),
        CapacityData(resource="Precision Machining", plant="Plant South",
                     installed_capacity=5000, available_capacity=4800,
                     utilized_capacity=4560, utilization_pct=95.0, constraint_type="machine"),
        CapacityData(resource="Electronics Test", plant="Plant East",
                     installed_capacity=6000, available_capacity=5700,
                     utilized_capacity=4560, utilization_pct=80.0, constraint_type="labor"),
        CapacityData(resource="Packaging & Fulfillment", plant="DC Central",
                     installed_capacity=15000, available_capacity=14000,
                     utilized_capacity=9800, utilization_pct=70.0, constraint_type="labor"),
    ]

    # Financial bridge
    financial_bridge = FinancialBridge(
        revenue_plan=142_500_000, revenue_latest_estimate=138_200_000,
        revenue_variance=-4_300_000,
        gross_margin_plan=0.418, gross_margin_le=0.401,
        cogs_plan=82_950_000, cogs_le=82_700_000,
        inventory_value=47_800_000, working_capital=31_200_000
    )

    return SOPCycleData(
        cycle_month=current_month,
        review_date=datetime.now().strftime("%Y-%m-%d"),
        demand_signals=demand_signals,
        inventory_positions=inventory_positions,
        capacity_data=capacity_data,
        financial_bridge=financial_bridge,
        key_decisions=[
            "Approve Q3 capacity expansion for Assembly Line A (+15%)",
            "Resolve $420K excess inventory in Industrial Equipment",
            "Set consensus demand uplift for Consumer Electronics H2",
            "Authorize spot procurement for Automotive Components shortage",
        ],
        open_issues=[
            "Supplier lead time extension — Tier-1 components +3 weeks",
            "Demand forecast bias in Healthcare: +8% systematic overforecast",
            "Plant South precision machining at 95% utilization — constraint risk",
            "FX headwind impacting EMEA revenue by ~$1.2M",
        ],
        overall_health=68.0,
        plan_confidence=0.64,
    )


def get_historical_demand_df() -> pd.DataFrame:
    """12-month trailing demand for charts"""
    np.random.seed(7)
    months = ["Mar '24", "Apr '24", "May '24", "Jun '24", "Jul '24", "Aug '24",
              "Sep '24", "Oct '24", "Nov '24", "Dec '24", "Jan '25", "Feb '25"]
    families = ["Consumer Electronics", "Industrial Equipment", "Healthcare Devices", "Automotive Components"]
    rows = []
    for fam in families:
        base = {"Consumer Electronics": 8500, "Industrial Equipment": 3200,
                "Healthcare Devices": 1800, "Automotive Components": 5400}[fam]
        trend = {"Consumer Electronics": 1.012, "Industrial Equipment": 0.995,
                 "Healthcare Devices": 1.018, "Automotive Components": 1.005}[fam]
        for i, m in enumerate(months):
            val = base * (trend ** i) * (1 + 0.06 * np.sin(i * 0.8) + np.random.normal(0, 0.03))
            rows.append({"Month": m, "Product Family": fam, "Demand": round(val, 0)})
    return pd.DataFrame(rows)


def get_scenario_comparison_df(scenarios: List[Scenario]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        rows.append({
            "Scenario": s.name,
            "Type": s.type.value,
            "Revenue Impact ($M)": round(s.revenue_impact / 1e6, 2),
            "Margin Impact ($M)": round(s.margin_impact / 1e6, 2),
            "Demand Change (%)": round(s.demand_uplift * 100, 1),
            "Inventory Change (%)": round(s.inventory_impact * 100, 1),
            "Service Level (%)": round(s.service_level_impact * 100, 1),
            "Probability (%)": round(s.probability * 100, 0),
            "Risk": s.risk_tier.value,
        })
    return pd.DataFrame(rows)
