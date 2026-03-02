"""
S&OP AI Agent System — LangGraph orchestration of specialized planning agents
Each agent handles a distinct S&OP function using gpt-4o-mini (small, fast, cost-effective)
"""

import json
import os
from typing import Any, Dict, List, TypedDict, Annotated, Optional
import operator
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from models import (
    SOPCycleData, Scenario, ScenarioType, ScenarioAssumption,
    RiskTier, DemandSignal, InventoryPosition
)

# ─── LangGraph State ──────────────────────────────────────────────────────────

class SOPGraphState(TypedDict):
    sop_data: dict                              # Serialized SOPCycleData
    messages: Annotated[List[dict], operator.add]
    phase: str
    scenarios: List[dict]
    iteration: int
    error: Optional[str]

# ─── Shared LLM Factory ───────────────────────────────────────────────────────

def llm(temperature: float = 0.15) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


def safe_json(text: str) -> dict:
    """Safely parse JSON, stripping markdown fences"""
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception as e:
        return {"error": str(e), "raw": text[:200]}

# ─── Agent 1: Demand Intelligence Agent ──────────────────────────────────────

def demand_intelligence_agent(state: SOPGraphState) -> SOPGraphState:
    """
    Analyzes demand signals, identifies forecast bias, recommends
    consensus adjustments. Uses gpt-4o-mini.
    """
    sop = SOPCycleData(**state["sop_data"])
    model = llm(0.1)

    # Summarize demand data
    demand_summary = []
    seen = set()
    for d in sop.demand_signals:
        key = d.product_family
        if key not in seen:
            seen.add(key)
            family_signals = [x for x in sop.demand_signals if x.product_family == key]
            avg_mape = sum(x.mape for x in family_signals) / len(family_signals)
            avg_bias = sum(x.bias for x in family_signals) / len(family_signals)
            avg_actual = sum(x.actual_demand for x in family_signals) / len(family_signals)
            demand_summary.append({
                "product_family": key,
                "avg_monthly_demand": round(avg_actual, 0),
                "avg_mape": round(avg_mape, 3),
                "avg_bias": round(avg_bias, 3),
                "bias_direction": "over-forecast" if avg_bias > 0 else "under-forecast",
            })

    inv_summary = [
        {"product_family": i.product_family, "weeks_of_supply": i.weeks_of_supply,
         "doh": i.doh, "excess_value_usd": i.excess_value}
        for i in sop.inventory_positions
    ]

    prompt = f"""You are the Demand Intelligence Agent in an AI-powered S&OP system.
Analyze these demand signals and inventory positions for {sop.cycle_month}.

DEMAND SIGNALS (6-month averages):
{json.dumps(demand_summary, indent=2)}

INVENTORY POSITIONS:
{json.dumps(inv_summary, indent=2)}

OPEN ISSUES: {json.dumps(sop.open_issues)}

Perform expert S&OP demand analysis. Return JSON only:
{{
  "demand_health_score": 0-100,
  "forecast_quality": {{
    "overall_mape_assessment": "good|acceptable|poor",
    "bias_issues": ["list of families with systematic bias"],
    "root_causes": ["identified root causes"]
  }},
  "demand_insights": [
    {{
      "product_family": "...",
      "insight": "specific finding with numbers",
      "urgency": "low|medium|high",
      "recommended_adjustment": "+/-X% to consensus forecast"
    }}
  ],
  "consensus_adjustments": {{
    "Consumer Electronics": 0.03,
    "Industrial Equipment": -0.02,
    "Healthcare Devices": -0.08,
    "Automotive Components": 0.01
  }},
  "demand_risks": ["list of 3-4 specific demand risks"],
  "narrative": "2-3 sentence executive summary of demand health"
}}

Be specific with percentages, dollar amounts, and actionable recommendations.
Return ONLY valid JSON."""

    response = model.invoke([
        SystemMessage(content="You are a senior S&OP demand planner. Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = safe_json(response.content)

    # Apply consensus adjustments to sop data
    adjustments = result.get("consensus_adjustments", {})
    for fam, adj in adjustments.items():
        base = next((d.actual_demand for d in sop.demand_signals if d.product_family == fam), 8000)
        sop.consensus_forecast[fam] = round(base * (1 + adj), 0)

    health = result.get("demand_health_score", 72)

    message = {
        "agent": "Demand Intelligence",
        "icon": "📊",
        "model": "gpt-4o-mini",
        "phase": "demand",
        "timestamp": datetime.now().isoformat(),
        "content": result.get("narrative", "Demand analysis complete."),
        "details": result,
        "health_score": health,
    }

    return {
        **state,
        "sop_data": sop.dict(),
        "messages": [message],
        "phase": "supply",
        "iteration": state["iteration"] + 1,
    }

# ─── Agent 2: Supply & Capacity Agent ────────────────────────────────────────

def supply_capacity_agent(state: SOPGraphState) -> SOPGraphState:
    """
    Reviews capacity utilization, identifies constraints, aligns supply plan
    to consensus demand. Uses gpt-4o-mini.
    """
    sop = SOPCycleData(**state["sop_data"])
    model = llm(0.1)

    cap_data = [
        {"resource": c.resource, "plant": c.plant,
         "utilization_pct": c.utilization_pct,
         "available": c.available_capacity, "used": c.utilized_capacity,
         "constraint_type": c.constraint_type,
         "slack_units": round(c.available_capacity - c.utilized_capacity, 0)}
        for c in sop.capacity_data
    ]

    inv_data = [
        {"product_family": i.product_family,
         "weeks_of_supply": i.weeks_of_supply,
         "safety_stock": i.safety_stock,
         "on_hand": i.on_hand}
        for i in sop.inventory_positions
    ]

    consensus = sop.consensus_forecast or {
        "Consumer Electronics": 8760, "Industrial Equipment": 3150,
        "Healthcare Devices": 1660, "Automotive Components": 5450
    }

    prompt = f"""You are the Supply & Capacity Agent in an AI-powered S&OP system.
Analyze capacity constraints and supply plan alignment for {sop.cycle_month}.

CAPACITY DATA:
{json.dumps(cap_data, indent=2)}

INVENTORY POSITIONS:
{json.dumps(inv_data, indent=2)}

CONSENSUS DEMAND FORECAST: {json.dumps(consensus)}

Perform expert supply-demand balancing. Return JSON only:
{{
  "supply_health_score": 0-100,
  "capacity_assessment": {{
    "constrained_resources": ["resources at >90% utilization with names"],
    "underutilized_resources": ["resources <65% with names"],
    "critical_bottleneck": "most critical constraint",
    "total_capacity_risk": "low|medium|high"
  }},
  "supply_gaps": [
    {{
      "product_family": "...",
      "gap_units": 500,
      "gap_direction": "short|long",
      "root_cause": "specific cause",
      "resolution_options": ["option1", "option2"]
    }}
  ],
  "recommended_production_plan": {{
    "Consumer Electronics": 8800,
    "Industrial Equipment": 3200,
    "Healthcare Devices": 1700,
    "Automotive Components": 5400
  }},
  "capacity_actions": [
    {{
      "action": "specific capacity action",
      "resource": "resource name",
      "timing": "immediate|next_quarter|H2",
      "cost_estimate": 250000,
      "benefit": "quantified benefit"
    }}
  ],
  "narrative": "2-3 sentence supply situation summary"
}}

Return ONLY valid JSON."""

    response = model.invoke([
        SystemMessage(content="You are a senior supply chain capacity planner. Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = safe_json(response.content)
    health = result.get("supply_health_score", 68)

    message = {
        "agent": "Supply & Capacity",
        "icon": "⚙️",
        "model": "gpt-4o-mini",
        "phase": "supply",
        "timestamp": datetime.now().isoformat(),
        "content": result.get("narrative", "Supply analysis complete."),
        "details": result,
        "health_score": health,
    }

    return {
        **state,
        "sop_data": sop.dict(),
        "messages": [message],
        "phase": "financial",
        "iteration": state["iteration"] + 1,
    }

# ─── Agent 3: Financial Reconciliation Agent ─────────────────────────────────

def financial_reconciliation_agent(state: SOPGraphState) -> SOPGraphState:
    """
    Bridges operational plan to P&L, quantifies revenue/margin gaps,
    recommends financial levers. Uses gpt-4o-mini.
    """
    sop = SOPCycleData(**state["sop_data"])
    model = llm(0.1)

    fb = sop.financial_bridge
    fin_data = fb.dict() if fb else {}

    # Pull demand agent consensus
    demand_msg = next((m for m in state["messages"] if m.get("agent") == "Demand Intelligence"), {})
    demand_details = demand_msg.get("details", {})

    prompt = f"""You are the Financial Reconciliation Agent in an AI-powered S&OP system.
Bridge the operational S&OP plan to financial P&L for {sop.cycle_month}.

FINANCIAL BRIDGE:
{json.dumps(fin_data, indent=2)}

DEMAND ADJUSTMENTS: {json.dumps(demand_details.get("consensus_adjustments", {}))}
DEMAND RISKS: {json.dumps(demand_details.get("demand_risks", []))}
OPEN ISSUES: {json.dumps(sop.open_issues)}

Perform financial S&OP reconciliation. Return JSON only:
{{
  "financial_health_score": 0-100,
  "p_and_l_bridge": {{
    "revenue_gap_usd": -4300000,
    "revenue_gap_pct": -3.0,
    "margin_gap_usd": -2100000,
    "margin_gap_pct": -1.7,
    "primary_drivers": ["driver1: $Xm", "driver2: $Xm"],
    "recovery_potential": 1800000
  }},
  "financial_risks": [
    {{
      "risk": "specific financial risk",
      "exposure_usd": 1200000,
      "probability": "low|medium|high",
      "mitigation": "specific mitigation action"
    }}
  ],
  "levers": [
    {{
      "lever": "lever name",
      "type": "revenue|cost|working_capital",
      "impact_usd": 800000,
      "timing": "Q3|Q4|FY",
      "owner": "function owner",
      "confidence": "low|medium|high"
    }}
  ],
  "working_capital_actions": ["action1", "action2"],
  "financial_narrative": "2-3 sentence CFO-ready financial summary"
}}

Return ONLY valid JSON."""

    response = model.invoke([
        SystemMessage(content="You are a senior FP&A / S&OP financial analyst. Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = safe_json(response.content)
    health = result.get("financial_health_score", 70)

    message = {
        "agent": "Financial Reconciliation",
        "icon": "💰",
        "model": "gpt-4o-mini",
        "phase": "financial",
        "timestamp": datetime.now().isoformat(),
        "content": result.get("financial_narrative", "Financial reconciliation complete."),
        "details": result,
        "health_score": health,
    }

    return {
        **state,
        "sop_data": sop.dict(),
        "messages": [message],
        "phase": "scenarios",
        "iteration": state["iteration"] + 1,
    }

# ─── Agent 4: Scenario Planning Agent ────────────────────────────────────────

def scenario_planning_agent(state: SOPGraphState) -> SOPGraphState:
    """
    The core S&OP innovation: AI-driven scenario planning.
    Generates Base / Optimistic / Pessimistic / Stress scenarios with
    full quantification and decision trees. Uses gpt-4o-mini.
    """
    sop = SOPCycleData(**state["sop_data"])
    model = llm(0.35)  # Slightly more creative for scenario generation

    # Gather context from prior agents
    demand_msg = next((m for m in state["messages"] if m.get("agent") == "Demand Intelligence"), {})
    supply_msg = next((m for m in state["messages"] if m.get("agent") == "Supply & Capacity"), {})
    fin_msg = next((m for m in state["messages"] if m.get("agent") == "Financial Reconciliation"), {})

    fb = sop.financial_bridge
    base_revenue = fb.revenue_latest_estimate if fb else 138_200_000

    context = {
        "cycle": sop.cycle_month,
        "base_revenue_usd": base_revenue,
        "demand_risks": demand_msg.get("details", {}).get("demand_risks", []),
        "capacity_constraints": supply_msg.get("details", {}).get("capacity_assessment", {}),
        "financial_gaps": fin_msg.get("details", {}).get("p_and_l_bridge", {}),
        "open_issues": sop.open_issues,
        "key_decisions": sop.key_decisions,
    }

    prompt = f"""You are the Scenario Planning Agent — the strategic brain of an AI-powered S&OP system.
Generate 4 comprehensive planning scenarios for {sop.cycle_month} executive review.

PLANNING CONTEXT:
{json.dumps(context, indent=2)}

Generate 4 scenarios (Base, Optimistic, Pessimistic, Stress Test). Return JSON only:
{{
  "scenarios": [
    {{
      "id": "S001",
      "name": "Base Case",
      "type": "base_case",
      "description": "Detailed 2-3 sentence scenario description with specific market conditions",
      "probability": 0.45,
      "risk_tier": "moderate",
      "assumptions": [
        {{"category": "Demand", "assumption": "specific assumption", "value": "specific value or %", "confidence": 0.75, "impact": "description"}},
        {{"category": "Supply", "assumption": "specific assumption", "value": "specific value or %", "confidence": 0.80, "impact": "description"}},
        {{"category": "Macro", "assumption": "specific assumption", "value": "specific value or %", "confidence": 0.65, "impact": "description"}}
      ],
      "demand_uplift": 0.00,
      "cost_uplift": 0.00,
      "margin_impact": 0,
      "revenue_impact": 0,
      "inventory_impact": 0.00,
      "service_level_impact": 0.00,
      "recommended_actions": [
        "Specific action with owner and timeline",
        "Another specific action"
      ],
      "ai_narrative": "CFO/COO-ready 2-sentence scenario narrative with specific numbers"
    }},
    {{
      "id": "S002",
      "name": "Market Recovery",
      "type": "optimistic",
      "description": "...",
      "probability": 0.25,
      "risk_tier": "low",
      "assumptions": [...],
      "demand_uplift": 0.08,
      "cost_uplift": 0.02,
      "margin_impact": 3200000,
      "revenue_impact": 9800000,
      "inventory_impact": -0.12,
      "service_level_impact": 0.02,
      "recommended_actions": ["Pre-position capacity now", "..."],
      "ai_narrative": "..."
    }},
    {{
      "id": "S003",
      "name": "Demand Contraction",
      "type": "pessimistic",
      "description": "...",
      "probability": 0.22,
      "risk_tier": "high",
      "assumptions": [...],
      "demand_uplift": -0.12,
      "cost_uplift": 0.03,
      "margin_impact": -5800000,
      "revenue_impact": -14200000,
      "inventory_impact": 0.25,
      "service_level_impact": -0.03,
      "recommended_actions": ["Implement demand destruction playbook", "..."],
      "ai_narrative": "..."
    }},
    {{
      "id": "S004",
      "name": "Supply Shock",
      "type": "stress_test",
      "description": "...",
      "probability": 0.08,
      "risk_tier": "critical",
      "assumptions": [...],
      "demand_uplift": 0.05,
      "cost_uplift": 0.18,
      "margin_impact": -9200000,
      "revenue_impact": -6500000,
      "inventory_impact": 0.40,
      "service_level_impact": -0.15,
      "recommended_actions": ["Activate emergency procurement protocols", "..."],
      "ai_narrative": "..."
    }}
  ],
  "scenario_summary": "3-sentence executive framing of the scenario space",
  "recommended_hedging_strategy": "specific hedging recommendation",
  "decision_triggers": [
    {{"trigger": "observable market event", "action": "specific response action", "scenario": "which scenario it confirms"}}
  ]
}}

Use real dollar amounts based on base_revenue of ${base_revenue:,.0f}.
Make assumptions specific and quantified. Return ONLY valid JSON."""

    response = model.invoke([
        SystemMessage(content="You are a strategic S&OP scenario planner. Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = safe_json(response.content)
    scenarios_raw = result.get("scenarios", [])

    # Build Scenario objects
    scenarios = []
    for s in scenarios_raw:
        try:
            assumptions = [
                ScenarioAssumption(**a) for a in s.get("assumptions", [])
            ]
            scenario = Scenario(
                id=s.get("id", "S001"),
                name=s.get("name", "Scenario"),
                type=ScenarioType(s.get("type", "base_case")),
                description=s.get("description", ""),
                assumptions=assumptions,
                demand_uplift=s.get("demand_uplift", 0.0),
                cost_uplift=s.get("cost_uplift", 0.0),
                margin_impact=s.get("margin_impact", 0.0),
                revenue_impact=s.get("revenue_impact", 0.0),
                inventory_impact=s.get("inventory_impact", 0.0),
                service_level_impact=s.get("service_level_impact", 0.0),
                risk_tier=RiskTier(s.get("risk_tier", "moderate")),
                probability=s.get("probability", 0.25),
                recommended_actions=s.get("recommended_actions", []),
                created_at=datetime.now().isoformat(),
                ai_narrative=s.get("ai_narrative", ""),
            )
            scenarios.append(scenario)
        except Exception as e:
            pass

    sop.scenarios = scenarios

    message = {
        "agent": "Scenario Planning",
        "icon": "🎯",
        "model": "gpt-4o-mini",
        "phase": "scenarios",
        "timestamp": datetime.now().isoformat(),
        "content": result.get("scenario_summary", f"Generated {len(scenarios)} planning scenarios."),
        "details": {
            "scenarios_count": len(scenarios),
            "hedging": result.get("recommended_hedging_strategy", ""),
            "triggers": result.get("decision_triggers", []),
            "summary": result.get("scenario_summary", ""),
        },
        "health_score": 80,
        "scenarios": [s.dict() for s in scenarios],
    }

    return {
        **state,
        "sop_data": sop.dict(),
        "messages": [message],
        "scenarios": [s.dict() for s in scenarios],
        "phase": "executive",
        "iteration": state["iteration"] + 1,
    }

# ─── Agent 5: Executive Synthesis Agent ──────────────────────────────────────

def executive_synthesis_agent(state: SOPGraphState) -> SOPGraphState:
    """
    Synthesizes all agent outputs into executive-ready S&OP package.
    Generates key decisions, risk register, and recommended plan.
    Uses gpt-4o-mini.
    """
    sop = SOPCycleData(**state["sop_data"])
    model = llm(0.2)

    # Gather all agent insights
    agent_summaries = []
    for msg in state["messages"]:
        agent_summaries.append({
            "agent": msg.get("agent"),
            "finding": msg.get("content"),
            "health": msg.get("health_score", 75),
        })

    scenarios_summary = []
    for s in sop.scenarios:
        scenarios_summary.append({
            "name": s.name,
            "probability": s.probability,
            "revenue_impact": s.revenue_impact,
            "margin_impact": s.margin_impact,
            "risk": s.risk_tier.value,
        })

    fb = sop.financial_bridge
    prompt = f"""You are the Executive Synthesis Agent for an AI-powered S&OP system.
Synthesize all planning inputs into a CEO/CFO/COO-ready S&OP recommendation package.

CYCLE: {sop.cycle_month}
FINANCIAL POSITION: Revenue LE ${fb.revenue_latest_estimate/1e6:.1f}M vs Plan ${fb.revenue_plan/1e6:.1f}M (gap: ${fb.revenue_variance/1e6:.1f}M)

AGENT FINDINGS:
{json.dumps(agent_summaries, indent=2)}

SCENARIOS:
{json.dumps(scenarios_summary, indent=2)}

KEY DECISIONS REQUIRED: {json.dumps(sop.key_decisions)}
OPEN ISSUES: {json.dumps(sop.open_issues)}

Generate the executive S&OP synthesis. Return JSON only:
{{
  "overall_health_score": 0-100,
  "plan_confidence": 0.0-1.0,
  "executive_summary": "4-5 sentence CEO-ready summary: current state, key risks, recommended path, financial impact",
  "recommended_plan": "base_case|optimistic|pessimistic — which scenario to plan to and why",
  "key_decisions": [
    {{
      "decision": "clear decision statement",
      "owner": "function/role",
      "deadline": "date or cycle",
      "financial_impact": "$X million",
      "urgency": "immediate|this_cycle|next_cycle",
      "recommendation": "AI recommendation"
    }}
  ],
  "risk_register": [
    {{
      "risk": "specific risk name",
      "category": "demand|supply|financial|external",
      "likelihood": "low|medium|high",
      "impact": "low|medium|high|critical",
      "mitigation": "specific mitigation",
      "owner": "function"
    }}
  ],
  "30_60_90_actions": {{
    "30_days": ["action1", "action2", "action3"],
    "60_days": ["action1", "action2"],
    "90_days": ["action1", "action2"]
  }},
  "metrics_to_watch": [
    {{"metric": "metric name", "current": "current value", "target": "target value", "trend": "improving|stable|declining"}}
  ]
}}

Return ONLY valid JSON."""

    response = model.invoke([
        SystemMessage(content="You are a C-suite S&OP facilitator. Return only valid JSON."),
        HumanMessage(content=prompt)
    ])

    result = safe_json(response.content)

    # Update sop state
    sop.executive_summary = result.get("executive_summary", "")
    sop.overall_health = result.get("overall_health_score", 72)
    sop.plan_confidence = result.get("plan_confidence", 0.68)

    # Store agent insights
    sop.agent_insights = state["messages"] + [{"agent": "Executive Synthesis", "details": result}]

    message = {
        "agent": "Executive Synthesis",
        "icon": "🏆",
        "model": "gpt-4o-mini",
        "phase": "executive",
        "timestamp": datetime.now().isoformat(),
        "content": result.get("executive_summary", "Executive synthesis complete."),
        "details": result,
        "health_score": result.get("overall_health_score", 72),
    }

    return {
        **state,
        "sop_data": sop.dict(),
        "messages": [message],
        "phase": "done",
        "iteration": state["iteration"] + 1,
    }


# ─── Routing Logic ────────────────────────────────────────────────────────────

def route_phase(state: SOPGraphState) -> str:
    phase = state.get("phase", "supply")
    routing = {
        "supply": "supply",
        "financial": "financial",
        "scenarios": "scenarios",
        "executive": "executive",
        "done": END,
    }
    return routing.get(phase, END)

# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_sop_graph():
    """Assemble the LangGraph S&OP workflow"""
    workflow = StateGraph(SOPGraphState)

    workflow.add_node("demand", demand_intelligence_agent)
    workflow.add_node("supply", supply_capacity_agent)
    workflow.add_node("financial", financial_reconciliation_agent)
    workflow.add_node("scenarios", scenario_planning_agent)
    workflow.add_node("executive", executive_synthesis_agent)

    workflow.set_entry_point("demand")

    workflow.add_conditional_edges(
        "demand", route_phase,
        {"supply": "supply", "financial": "financial", "scenarios": "scenarios",
         "executive": "executive", END: END}
    )
    workflow.add_conditional_edges(
        "supply", route_phase,
        {"supply": "supply", "financial": "financial", "scenarios": "scenarios",
         "executive": "executive", END: END}
    )
    workflow.add_conditional_edges(
        "financial", route_phase,
        {"supply": "supply", "financial": "financial", "scenarios": "scenarios",
         "executive": "executive", END: END}
    )
    workflow.add_conditional_edges(
        "scenarios", route_phase,
        {"supply": "supply", "financial": "financial", "scenarios": "scenarios",
         "executive": "executive", END: END}
    )
    workflow.add_edge("executive", END)

    return workflow.compile()


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run_sop_cycle(sop_data: SOPCycleData, api_key: str):
    """Run the complete AI S&OP cycle"""
    os.environ["OPENAI_API_KEY"] = api_key

    graph = build_sop_graph()

    initial = SOPGraphState(
        sop_data=sop_data.dict(),
        messages=[],
        phase="supply",        # demand is entry point, after demand → supply
        scenarios=[],
        iteration=0,
        error=None,
    )

    result = graph.invoke(initial)

    final_sop = SOPCycleData(**result["sop_data"])
    return final_sop, result["messages"], result.get("scenarios", [])
