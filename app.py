"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI-DRIVEN S&OP PLANNING SYSTEM                                            ║
║   LangGraph × GPT-4o-mini × Streamlit                                      ║
║   Redesigning Sales & Operations Planning with Autonomous AI Agents         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI S&OP Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Typography + Styling ──────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
html, body, [class*="css"] {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'IBM Plex Mono', monospace;
}
.stApp { background: #0d1117; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown { font-size: 0.78rem; }

/* ── Main header ── */
.sop-header {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 1rem 0 1.2rem 0;
    margin-bottom: 1.2rem;
    position: relative;
}
.sop-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, #e6a817 0%, #2dd4bf 35%, #79c0ff 70%, #f85149 100%);
}
.sop-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.sop-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #656d76;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-top: 2px solid;
    border-radius: 4px;
    padding: 1rem 1.1rem 0.9rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 40px;
    border-radius: 0 0 0 40px;
    opacity: 0.06;
}
.kpi-label {
    font-size: 0.62rem;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.7rem;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-amber  { border-top-color: #e6a817; }
.kpi-teal   { border-top-color: #2dd4bf; }
.kpi-azure  { border-top-color: #79c0ff; }
.kpi-sage   { border-top-color: #7ee787; }
.kpi-crimson{ border-top-color: #f85149; }
.kpi-violet { border-top-color: #bc8cff; }

/* ── Section Labels ── */
.section-label {
    font-size: 0.62rem;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* ── Agent Cards ── */
.agent-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid;
    border-radius: 3px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    animation: fadeSlide 0.4s ease-out;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.agent-demand   { border-left-color: #79c0ff; }
.agent-supply   { border-left-color: #2dd4bf; }
.agent-financial{ border-left-color: #e6a817; }
.agent-scenarios{ border-left-color: #bc8cff; }
.agent-executive{ border-left-color: #7ee787; }

.agent-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 0.7rem;
}
.agent-name { font-weight: 600; color: #f0f6fc; font-size: 0.78rem; }
.agent-pill {
    background: #21262d;
    color: #656d76;
    padding: 1px 6px;
    border-radius: 2px;
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
}
.agent-body {
    font-size: 0.8rem;
    color: #8b949e;
    line-height: 1.55;
}

/* ── Scenario Cards ── */
.scenario-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 1rem 1.1rem;
    margin: 0.5rem 0;
    position: relative;
}
.scenario-card-base     { border-top: 2px solid #79c0ff; }
.scenario-card-optimistic { border-top: 2px solid #7ee787; }
.scenario-card-pessimistic{ border-top: 2px solid #ffa657; }
.scenario-card-stress_test{ border-top: 2px solid #f85149; }

.scenario-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #f0f6fc;
    margin-bottom: 4px;
}
.scenario-prob {
    font-size: 0.65rem;
    color: #656d76;
    margin-bottom: 8px;
}
.scenario-narrative {
    font-size: 0.77rem;
    color: #8b949e;
    line-height: 1.5;
    margin-bottom: 10px;
    padding: 8px;
    background: #0d1117;
    border-radius: 3px;
    border-left: 2px solid #21262d;
    font-style: italic;
}
.impact-row {
    display: flex;
    gap: 12px;
    font-size: 0.7rem;
    flex-wrap: wrap;
}
.impact-chip {
    padding: 2px 8px;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.impact-pos { background: #0d2117; color: #7ee787; }
.impact-neg { background: #2d0f0f; color: #f85149; }
.impact-neu { background: #161b22; color: #656d76; border: 1px solid #21262d; }

/* ── Risk Register Table ── */
.risk-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.76rem;
}
.risk-badge {
    padding: 2px 7px;
    border-radius: 2px;
    font-size: 0.65rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    min-width: 56px;
    text-align: center;
}
.risk-critical { background: #67100d; color: #f85149; }
.risk-high     { background: #3d1f00; color: #ffa657; }
.risk-moderate { background: #332200; color: #e6a817; }
.risk-low      { background: #0d2117; color: #7ee787; }

/* ── Decision Items ── */
.decision-item {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 3px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.decision-header {
    font-size: 0.8rem;
    color: #f0f6fc;
    font-weight: 500;
    margin-bottom: 6px;
}
.decision-meta {
    display: flex;
    gap: 10px;
    font-size: 0.68rem;
    color: #656d76;
    flex-wrap: wrap;
}
.decision-chip {
    background: #21262d;
    padding: 1px 7px;
    border-radius: 2px;
}
.decision-chip.urgent { background: #3d1f00; color: #ffa657; }
.decision-chip.impact { color: #7ee787; background: #0d2117; }

/* ── Action Timeline ── */
.action-day {
    border-left: 2px solid #21262d;
    padding-left: 1rem;
    margin-left: 0.5rem;
    margin-bottom: 1rem;
    position: relative;
}
.action-day::before {
    content: '';
    position: absolute;
    left: -5px; top: 4px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #e6a817;
}
.action-day-label {
    font-size: 0.65rem;
    color: #e6a817;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.action-item {
    font-size: 0.78rem;
    color: #8b949e;
    padding: 3px 0;
}
.action-item::before { content: "→ "; color: #656d76; }

/* ── Assumption Tags ── */
.assumption-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 3px;
    padding: 6px 10px;
    margin: 3px 0;
    font-size: 0.73rem;
}
.assumption-cat {
    color: #e6a817;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
}

/* ── Pipeline Steps ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
    margin: 8px 0;
}
.pipeline-node {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 0.68rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
}
.pipeline-node.active {
    border-color: #e6a817;
    color: #e6a817;
    background: #1a1400;
}
.pipeline-arrow { color: #30363d; font-size: 0.8rem; }

/* ── Metric Table ── */
.metric-table-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.76rem;
}
.metric-table-label { color: #656d76; }
.metric-table-val { color: #c9d1d9; font-family: 'IBM Plex Mono', monospace; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #656d76;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e6a817;
    border-bottom-color: #e6a817;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 3px;
}

/* ── Scrollable log ── */
.log-scroll {
    max-height: 480px;
    overflow-y: auto;
    padding-right: 4px;
}
.log-scroll::-webkit-scrollbar { width: 3px; }
.log-scroll::-webkit-scrollbar-thumb { background: #30363d; }

/* ── Status indicator ── */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: #7ee787; box-shadow: 0 0 6px #7ee787; }
.dot-amber { background: #e6a817; box-shadow: 0 0 6px #e6a817; animation: pulse 2s infinite; }
.dot-red   { background: #f85149; box-shadow: 0 0 6px #f85149; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Imports ───────────────────────────────────────────────────────────────────
from models import (
    SOPCycleData, Scenario, ScenarioType, RiskTier,
    generate_sop_data, get_historical_demand_df, get_scenario_comparison_df
)
from visualizations import (
    demand_trend_chart, forecast_accuracy_chart,
    capacity_utilization_chart, inventory_doh_chart,
    scenario_tornado_chart, scenario_probability_chart,
    financial_bridge_chart, plan_confidence_gauge
)

# ─── Session State Init ────────────────────────────────────────────────────────
if "sop_data" not in st.session_state:
    st.session_state.sop_data = generate_sop_data()
if "agent_log" not in st.session_state:
    st.session_state.agent_log = []
if "running" not in st.session_state:
    st.session_state.running = False
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0

sop: SOPCycleData = st.session_state.sop_data
fb = sop.financial_bridge


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:#f0f6fc;">
            S&OP<span style="color:#e6a817;">AI</span>
        </div>
        <div style="font-size:0.62rem; color:#656d76; text-transform:uppercase; letter-spacing:0.1em;">
            Command Center
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown('<div class="section-label" style="margin-top:1rem">LangGraph Pipeline</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline" style="flex-direction:column; align-items:flex-start;">
        <div class="pipeline-node">📊 Demand Intelligence</div>
        <div class="pipeline-arrow" style="padding-left:12px">↓</div>
        <div class="pipeline-node">⚙️ Supply & Capacity</div>
        <div class="pipeline-arrow" style="padding-left:12px">↓</div>
        <div class="pipeline-node">💰 Financial Reconciliation</div>
        <div class="pipeline-arrow" style="padding-left:12px">↓</div>
        <div class="pipeline-node">🎯 Scenario Planning</div>
        <div class="pipeline-arrow" style="padding-left:12px">↓</div>
        <div class="pipeline-node">🏆 Executive Synthesis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">S&OP Cycle Parameters</div>',
                unsafe_allow_html=True)

    planning_horizon = st.selectbox("Planning Horizon", ["18 months", "12 months", "24 months"])
    review_frequency = st.selectbox("Review Frequency", ["Monthly", "Quarterly"])
    scenario_count = st.slider("Scenarios to Generate", 3, 5, 4)

    st.markdown('<div class="section-label" style="margin-top:1rem">Actions</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.sop_data = generate_sop_data()
            st.session_state.agent_log = []
            st.session_state.cycle_count = 0
            st.rerun()
    with col_b:
        if st.button("Demo", use_container_width=True):
            # Set a more stressed state for demo
            d = generate_sop_data()
            d.overall_health = 58.0
            d.plan_confidence = 0.52
            d.open_issues.append("Critical: Tier-1 supplier bankruptcy risk identified")
            st.session_state.sop_data = d
            st.session_state.agent_log = []
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.65rem; color:#30363d; line-height:1.8;">
    Cycle: <span style="color:#656d76">{sop.cycle_month}</span><br>
    Agents: <span style="color:#656d76">5 × gpt-4o-mini</span><br>
    Framework: <span style="color:#656d76">LangGraph 0.1+</span><br>
    Cycles run: <span style="color:#656d76">{st.session_state.cycle_count}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

status_dot = '<span class="status-dot dot-amber"></span>' if not st.session_state.agent_log else \
             '<span class="status-dot dot-green"></span>'

st.markdown(f"""
<div class="sop-header">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div class="sop-title">
                AI-Driven Sales & Operations Planning
            </div>
            <div class="sop-subtitle">
                {status_dot}LangGraph Multi-Agent · GPT-4o-mini · {sop.cycle_month} Cycle · {planning_horizon if 'planning_horizon' in dir() else '18 months'} Horizon
            </div>
        </div>
        <div style="text-align:right; font-size:0.65rem; color:#30363d;">
            <div style="color:#656d76">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            <div>{review_frequency if 'review_frequency' in dir() else 'Monthly'} Review</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5, k6 = st.columns(6)

rev_gap = fb.revenue_variance / 1e6 if fb else -4.3
margin_pct = fb.gross_margin_le * 100 if fb else 40.1
inv_val = fb.inventory_value / 1e6 if fb else 47.8
open_issues = len(sop.open_issues)
scenarios_count = len(sop.scenarios)
conf_pct = sop.plan_confidence * 100

health_color = "#f85149" if sop.overall_health < 55 else "#e6a817" if sop.overall_health < 70 else "#7ee787"
gap_color = "#f85149" if rev_gap < 0 else "#7ee787"
conf_color = "#f85149" if conf_pct < 55 else "#e6a817" if conf_pct < 72 else "#7ee787"

with k1:
    st.markdown(f"""<div class="kpi-card kpi-amber">
    <div class="kpi-label">S&OP Health</div>
    <div class="kpi-value" style="color:{health_color}">{sop.overall_health:.0f}</div>
    <div class="kpi-delta" style="color:#656d76">/ 100 composite score</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="kpi-card kpi-{'crimson' if rev_gap < 0 else 'sage'}">
    <div class="kpi-label">Rev vs Plan</div>
    <div class="kpi-value" style="color:{gap_color}">{rev_gap:+.1f}M</div>
    <div class="kpi-delta" style="color:#656d76">USD latest estimate gap</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="kpi-card kpi-teal">
    <div class="kpi-label">Gross Margin</div>
    <div class="kpi-value" style="color:#2dd4bf">{margin_pct:.1f}%</div>
    <div class="kpi-delta" style="color:#656d76">LE vs {fb.gross_margin_plan*100:.1f}% plan</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card kpi-azure">
    <div class="kpi-label">Inventory Value</div>
    <div class="kpi-value" style="color:#79c0ff">${inv_val:.1f}M</div>
    <div class="kpi-delta" style="color:#656d76">on-hand + in-transit</div>
    </div>""", unsafe_allow_html=True)

with k5:
    issue_color = "#f85149" if open_issues > 4 else "#e6a817" if open_issues > 2 else "#7ee787"
    st.markdown(f"""<div class="kpi-card kpi-{'crimson' if open_issues > 4 else 'amber'}">
    <div class="kpi-label">Open Issues</div>
    <div class="kpi-value" style="color:{issue_color}">{open_issues}</div>
    <div class="kpi-delta" style="color:#656d76">requiring resolution</div>
    </div>""", unsafe_allow_html=True)

with k6:
    st.markdown(f"""<div class="kpi-card kpi-violet">
    <div class="kpi-label">Plan Confidence</div>
    <div class="kpi-value" style="color:{conf_color}">{conf_pct:.0f}%</div>
    <div class="kpi-delta" style="color:#656d76">AI composite signal</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_demand, tab_supply, tab_financial, tab_scenarios, tab_executive, tab_agents = st.tabs([
    "⚡ OVERVIEW",
    "📊 DEMAND",
    "⚙️ SUPPLY",
    "💰 FINANCIAL",
    "🎯 SCENARIOS",
    "🏆 EXECUTIVE",
    "🤖 AGENTS",
])


# ── TAB 1: OVERVIEW ────────────────────────────────────────────────────────────
with tab_overview:
    col_gauges, col_issues = st.columns([2, 1])

    with col_gauges:
        st.markdown('<div class="section-label">Plan Confidence & S&OP Health</div>',
                    unsafe_allow_html=True)
        fig_conf = plan_confidence_gauge(sop.plan_confidence, sop.overall_health)
        st.plotly_chart(fig_conf, use_container_width=True, key="conf_gauge")

        st.markdown('<div class="section-label" style="margin-top:0.5rem">Trailing Demand Trend</div>',
                    unsafe_allow_html=True)
        hist_df = get_historical_demand_df()
        fig_demand = demand_trend_chart(hist_df)
        st.plotly_chart(fig_demand, use_container_width=True, key="overview_demand")

    with col_issues:
        st.markdown('<div class="section-label">Open Issues</div>', unsafe_allow_html=True)
        for i, issue in enumerate(sop.open_issues, 1):
            urgency = "crimson" if "Critical" in issue else "amber"
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #21262d;
                        border-left:3px solid {'#f85149' if urgency=='crimson' else '#e6a817'};
                        border-radius:3px; padding:8px 10px; margin:4px 0;
                        font-size:0.76rem; color:#8b949e;">
                <span style="color:{'#f85149' if urgency=='crimson' else '#e6a817'}; font-size:0.65rem;">
                #{i}</span> {issue}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Key Decisions</div>',
                    unsafe_allow_html=True)
        for decision in sop.key_decisions:
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #21262d; border-radius:3px;
                        padding:8px 10px; margin:4px 0; font-size:0.76rem; color:#8b949e;">
                <span style="color:#79c0ff;">▸</span> {decision}
            </div>
            """, unsafe_allow_html=True)

    # Financial bridge
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Revenue Bridge: Plan to Latest Estimate</div>',
                unsafe_allow_html=True)
    fig_bridge = financial_bridge_chart(sop)
    st.plotly_chart(fig_bridge, use_container_width=True, key="overview_bridge")


# ── TAB 2: DEMAND ──────────────────────────────────────────────────────────────
with tab_demand:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown('<div class="section-label">Demand Trend — Trailing 12 Months</div>',
                    unsafe_allow_html=True)
        fig_dt = demand_trend_chart(hist_df)
        st.plotly_chart(fig_dt, use_container_width=True, key="demand_trend")

    with col_d2:
        st.markdown('<div class="section-label">Forecast Accuracy — MAPE Heatmap</div>',
                    unsafe_allow_html=True)
        fig_fa = forecast_accuracy_chart(sop)
        st.plotly_chart(fig_fa, use_container_width=True, key="forecast_acc")

    # Demand signal table
    st.markdown('<div class="section-label">Demand Signals — Last 6 Periods</div>',
                unsafe_allow_html=True)

    demand_rows = []
    seen = set()
    for d in sop.demand_signals:
        key = d.product_family
        if key not in seen:
            seen.add(key)
            family_signals = [x for x in sop.demand_signals if x.product_family == key]
            avg_mape = sum(x.mape for x in family_signals) / len(family_signals)
            avg_bias = sum(x.bias for x in family_signals) / len(family_signals)
            avg_actual = sum(x.actual_demand for x in family_signals) / len(family_signals)
            avg_forecast = sum(x.forecast_demand for x in family_signals) / len(family_signals)
            consensus = sop.consensus_forecast.get(key, avg_actual)
            demand_rows.append({
                "Product Family": key,
                "Avg Actual": f"{avg_actual:,.0f}",
                "Avg Forecast": f"{avg_forecast:,.0f}",
                "Consensus": f"{consensus:,.0f}",
                "MAPE": f"{avg_mape*100:.1f}%",
                "Bias": f"{avg_bias*100:+.1f}%",
                "Bias Direction": "↑ Over" if avg_bias > 0 else "↓ Under",
            })

    df_demand = pd.DataFrame(demand_rows)
    st.dataframe(df_demand, use_container_width=True, hide_index=True)

    # Consensus adjustments if AI ran
    if sop.consensus_forecast:
        st.markdown('<div class="section-label" style="margin-top:0.5rem">AI Consensus Adjustments</div>',
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (fam, val) in enumerate(sop.consensus_forecast.items()):
            with cols[i % 4]:
                base_val = next((d.actual_demand for d in sop.demand_signals
                                 if d.product_family == fam), val)
                adj_pct = (val / base_val - 1) * 100 if base_val > 0 else 0
                color = "#7ee787" if adj_pct > 0 else "#f85149" if adj_pct < 0 else "#656d76"
                st.markdown(f"""
                <div class="kpi-card" style="border-top:2px solid {color};">
                <div class="kpi-label">{fam.split()[0]}</div>
                <div style="font-size:1.1rem; font-weight:700; color:{color}; font-family:'Syne',sans-serif;">
                    {val:,.0f}
                </div>
                <div style="font-size:0.68rem; color:#656d76;">{adj_pct:+.1f}% vs actuals</div>
                </div>""", unsafe_allow_html=True)


# ── TAB 3: SUPPLY ──────────────────────────────────────────────────────────────
with tab_supply:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="section-label">Capacity Utilization</div>', unsafe_allow_html=True)
        fig_cap = capacity_utilization_chart(sop)
        st.plotly_chart(fig_cap, use_container_width=True, key="cap_util")

    with col_s2:
        st.markdown('<div class="section-label">Inventory: DOH & Excess Value</div>',
                    unsafe_allow_html=True)
        fig_inv = inventory_doh_chart(sop)
        st.plotly_chart(fig_inv, use_container_width=True, key="inv_doh")

    # Capacity detail table
    st.markdown('<div class="section-label">Capacity Detail by Resource</div>', unsafe_allow_html=True)
    cap_rows = []
    for c in sop.capacity_data:
        util = c.utilization_pct
        status = "🔴 Constrained" if util >= 90 else "🟡 Elevated" if util >= 75 else "🟢 Normal"
        cap_rows.append({
            "Resource": c.resource,
            "Plant": c.plant,
            "Type": c.constraint_type,
            "Installed": f"{c.installed_capacity:,.0f}",
            "Available": f"{c.available_capacity:,.0f}",
            "Utilized": f"{c.utilized_capacity:,.0f}",
            "Utilization": f"{util:.0f}%",
            "Status": status,
        })
    df_cap = pd.DataFrame(cap_rows)
    st.dataframe(df_cap, use_container_width=True, hide_index=True)

    # Inventory detail
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Inventory Position Detail</div>',
                unsafe_allow_html=True)
    inv_rows = []
    for inv in sop.inventory_positions:
        wos_color = "🔴" if inv.weeks_of_supply > 6 else "🟡" if inv.weeks_of_supply > 4 else "🟢"
        inv_rows.append({
            "Product Family": inv.product_family,
            "On Hand": f"{inv.on_hand:,.0f}",
            "In Transit": f"{inv.in_transit:,.0f}",
            "On Order": f"{inv.on_order:,.0f}",
            "Safety Stock": f"{inv.safety_stock:,.0f}",
            "WOS": f"{wos_color} {inv.weeks_of_supply:.1f}w",
            "DOH": f"{inv.doh:.1f}d",
            "Excess $": f"${inv.excess_value/1000:.0f}K",
        })
    df_inv = pd.DataFrame(inv_rows)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)


# ── TAB 4: FINANCIAL ───────────────────────────────────────────────────────────
with tab_financial:
    if fb:
        # Key financial metrics
        f1, f2, f3, f4 = st.columns(4)
        metrics = [
            ("Revenue Plan", f"${fb.revenue_plan/1e6:.1f}M", "annual plan", "azure"),
            ("Revenue LE",   f"${fb.revenue_latest_estimate/1e6:.1f}M",
             f"{fb.revenue_variance/1e6:+.1f}M vs plan", "crimson" if fb.revenue_variance < 0 else "sage"),
            ("Gross Margin LE", f"{fb.gross_margin_le*100:.1f}%",
             f"{(fb.gross_margin_le - fb.gross_margin_plan)*100:+.1f}pp vs plan", "teal"),
            ("Working Capital", f"${fb.working_capital/1e6:.1f}M", "net working capital", "amber"),
        ]
        for col, (label, val, delta, accent) in zip([f1, f2, f3, f4], metrics):
            with col:
                st.markdown(f"""<div class="kpi-card kpi-{accent}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{'#79c0ff' if accent=='azure' else '#f85149' if accent=='crimson' else '#2dd4bf' if accent=='teal' else '#e6a817'};">
                    {val}</div>
                <div class="kpi-delta" style="color:#656d76;">{delta}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Revenue Bridge Waterfall</div>', unsafe_allow_html=True)
        fig_fb = financial_bridge_chart(sop)
        st.plotly_chart(fig_fb, use_container_width=True, key="fin_bridge")

        # Financial levers from AI (if available)
        exec_msg = next((m for m in st.session_state.agent_log
                        if m.get("agent") == "Financial Reconciliation"), None)
        if exec_msg:
            levers = exec_msg.get("details", {}).get("levers", [])
            risks = exec_msg.get("details", {}).get("financial_risks", [])

            if levers:
                st.markdown('<div class="section-label" style="margin-top:0.5rem">AI-Identified Financial Levers</div>',
                            unsafe_allow_html=True)
                lever_rows = []
                for lev in levers:
                    lever_rows.append({
                        "Lever": lev.get("lever", ""),
                        "Type": lev.get("type", ""),
                        "Impact ($)": f"${lev.get('impact_usd', 0)/1000:.0f}K",
                        "Timing": lev.get("timing", ""),
                        "Owner": lev.get("owner", ""),
                        "Confidence": lev.get("confidence", "").upper(),
                    })
                st.dataframe(pd.DataFrame(lever_rows), use_container_width=True, hide_index=True)

            if risks:
                st.markdown('<div class="section-label" style="margin-top:0.5rem">Financial Risk Register</div>',
                            unsafe_allow_html=True)
                for r in risks:
                    exposure = r.get("exposure_usd", 0)
                    prob = r.get("probability", "medium")
                    color = {"high": "#f85149", "medium": "#e6a817", "low": "#7ee787"}.get(prob, "#656d76")
                    st.markdown(f"""
                    <div style="background:#161b22; border:1px solid #21262d;
                                border-left:3px solid {color}; border-radius:3px;
                                padding:8px 10px; margin:4px 0;">
                        <div style="font-size:0.78rem; color:#c9d1d9; font-weight:500;">
                            {r.get('risk', '')}
                            <span style="color:{color}; font-size:0.65rem; margin-left:8px;">
                            ${exposure/1000:.0f}K exposure</span>
                        </div>
                        <div style="font-size:0.72rem; color:#656d76; margin-top:3px;">
                            ↳ {r.get('mitigation', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Run the AI S&OP cycle to generate financial reconciliation.")


# ── TAB 5: SCENARIOS ───────────────────────────────────────────────────────────
with tab_scenarios:
    if sop.scenarios:
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.markdown('<div class="section-label">Scenario Financial Impact</div>',
                        unsafe_allow_html=True)
            fig_tornado = scenario_tornado_chart(sop.scenarios)
            st.plotly_chart(fig_tornado, use_container_width=True, key="tornado")

        with sc_col2:
            st.markdown('<div class="section-label">Scenario Probability Space</div>',
                        unsafe_allow_html=True)
            fig_prob = scenario_probability_chart(sop.scenarios)
            st.plotly_chart(fig_prob, use_container_width=True, key="sc_prob")

        st.markdown('<div class="section-label">Scenario Detail Cards</div>', unsafe_allow_html=True)

        type_order = ["base_case", "optimistic", "pessimistic", "stress_test", "custom"]
        sorted_scenarios = sorted(sop.scenarios,
                                  key=lambda s: type_order.index(s.type.value)
                                  if s.type.value in type_order else 99)

        for scen in sorted_scenarios:
            type_colors = {
                "base_case": "#79c0ff", "optimistic": "#7ee787",
                "pessimistic": "#ffa657", "stress_test": "#f85149", "custom": "#bc8cff"
            }
            sc_color = type_colors.get(scen.type.value, "#656d76")

            rev_chip = f'<span class="impact-chip {"impact-pos" if scen.revenue_impact > 0 else "impact-neg"}">Rev: ${scen.revenue_impact/1e6:+.1f}M</span>'
            mar_chip = f'<span class="impact-chip {"impact-pos" if scen.margin_impact > 0 else "impact-neg"}">Margin: ${scen.margin_impact/1e6:+.1f}M</span>'
            dem_chip = f'<span class="impact-chip {"impact-pos" if scen.demand_uplift > 0 else "impact-neg" if scen.demand_uplift < 0 else "impact-neu"}">Demand: {scen.demand_uplift*100:+.0f}%</span>'
            inv_chip = f'<span class="impact-chip {"impact-neg" if scen.inventory_impact > 0.1 else "impact-pos" if scen.inventory_impact < -0.05 else "impact-neu"}">Inv: {scen.inventory_impact*100:+.0f}%</span>'
            risk_chip = f'<span class="impact-chip {{"low":"impact-pos","moderate":"impact-neu","high":"impact-neg","critical":"impact-neg"}}.get("{scen.risk_tier.value}","impact-neu")">Risk: {scen.risk_tier.value.upper()}</span>'

            with st.expander(
                f"{scen.name}  ·  P={scen.probability:.0%}  ·  {scen.type.value.replace('_',' ').title()}",
                expanded=(scen.type == ScenarioType.BASE)
            ):
                st.markdown(f"""
                <div style="border-left:3px solid {sc_color}; padding-left:12px; margin-bottom:10px;">
                    <div style="font-size:0.78rem; color:#8b949e; font-style:italic; margin-bottom:8px;">
                        {scen.description}
                    </div>
                    <div class="impact-row">{rev_chip}{mar_chip}{dem_chip}{inv_chip}</div>
                </div>
                """, unsafe_allow_html=True)

                if scen.ai_narrative:
                    st.markdown(f"""
                    <div style="background:#0d1117; border:1px solid #21262d; border-radius:3px;
                                padding:10px 12px; font-size:0.77rem; color:#8b949e; font-style:italic;
                                border-left:3px solid {sc_color}; margin-bottom:10px;">
                        <span style="font-size:0.62rem; color:#656d76; text-transform:uppercase;
                                     letter-spacing:0.1em; display:block; margin-bottom:4px;">
                        AI Narrative</span>
                        {scen.ai_narrative}
                    </div>
                    """, unsafe_allow_html=True)

                if scen.assumptions:
                    st.markdown('<div style="font-size:0.65rem; color:#656d76; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">Key Assumptions</div>', unsafe_allow_html=True)
                    for a in scen.assumptions:
                        conf_bar = "█" * int(a.confidence * 10) + "░" * (10 - int(a.confidence * 10))
                        st.markdown(f"""
                        <div class="assumption-block">
                            <div class="assumption-cat">{a.category}</div>
                            <div style="color:#c9d1d9; margin-bottom:2px;">{a.assumption}: <span style="color:#e6a817">{a.value}</span></div>
                            <div style="font-size:0.65rem; color:#30363d; font-family:'IBM Plex Mono',monospace;">
                                Conf: <span style="color:#656d76">{conf_bar}</span> {a.confidence:.0%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                if scen.recommended_actions:
                    st.markdown('<div style="font-size:0.65rem; color:#656d76; text-transform:uppercase; letter-spacing:0.1em; margin:8px 0 6px;">Recommended Actions</div>', unsafe_allow_html=True)
                    for action in scen.recommended_actions:
                        st.markdown(f'<div class="action-item">{action}</div>',
                                    unsafe_allow_html=True)

        # Comparison table
        st.markdown('<div class="section-label" style="margin-top:0.5rem">Scenario Comparison Matrix</div>',
                    unsafe_allow_html=True)
        df_sc = get_scenario_comparison_df(sop.scenarios)
        st.dataframe(df_sc, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:4px;
                    padding:2.5rem; text-align:center; margin-top:1rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1rem; color:#f0f6fc; margin-bottom:8px;">
                🎯 No Scenarios Generated
            </div>
            <div style="font-size:0.8rem; color:#656d76; margin-bottom:1.5rem;">
                Run the AI S&OP cycle to generate Base / Optimistic / Pessimistic / Stress scenarios
                with full quantification, assumption transparency, and decision triggers.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 6: EXECUTIVE ───────────────────────────────────────────────────────────
with tab_executive:
    exec_msg = next((m for m in st.session_state.agent_log
                    if m.get("agent") == "Executive Synthesis"), None)

    if exec_msg and exec_msg.get("details"):
        details = exec_msg["details"]

        # Executive summary
        st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)
        summary = sop.executive_summary or details.get("executive_summary", "")
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:4px;
                    padding:1.2rem 1.4rem; border-top:2px solid #e6a817;
                    font-size:0.82rem; color:#8b949e; line-height:1.7;">
            <span style="font-size:0.62rem; color:#e6a817; text-transform:uppercase;
                         letter-spacing:0.12em; display:block; margin-bottom:8px;">
            CEO / CFO / COO Briefing · {sop.cycle_month}</span>
            {summary}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ex_col1, ex_col2 = st.columns(2)

        # Key decisions
        with ex_col1:
            st.markdown('<div class="section-label">Key Decisions Required</div>', unsafe_allow_html=True)
            decisions = details.get("key_decisions", [])
            for dec in decisions:
                urgency_map = {"immediate": "urgent", "this_cycle": "", "next_cycle": ""}
                urgency_class = urgency_map.get(dec.get("urgency", ""), "")
                impact = dec.get("financial_impact", "")
                rec = dec.get("recommendation", "")
                st.markdown(f"""
                <div class="decision-item">
                    <div class="decision-header">{dec.get('decision', '')}</div>
                    <div class="decision-meta">
                        <span class="decision-chip">👤 {dec.get('owner', '')}</span>
                        <span class="decision-chip {urgency_class}">⏰ {dec.get('deadline', '')}</span>
                        {f'<span class="decision-chip impact">💰 {impact}</span>' if impact else ''}
                    </div>
                    {f'<div style="font-size:0.72rem; color:#656d76; margin-top:6px;">AI: {rec}</div>' if rec else ''}
                </div>
                """, unsafe_allow_html=True)

        # Risk register
        with ex_col2:
            st.markdown('<div class="section-label">Risk Register</div>', unsafe_allow_html=True)
            risks = details.get("risk_register", [])
            for risk in risks:
                impact_map = {"critical": "risk-critical", "high": "risk-high",
                              "moderate": "risk-moderate", "low": "risk-low"}
                badge_class = impact_map.get(risk.get("impact", "moderate"), "risk-moderate")
                cat_colors = {
                    "demand": "#79c0ff", "supply": "#2dd4bf",
                    "financial": "#e6a817", "external": "#bc8cff"
                }
                cat_color = cat_colors.get(risk.get("category", ""), "#656d76")
                st.markdown(f"""
                <div class="risk-row">
                    <span class="risk-badge {badge_class}">{risk.get('impact', '').upper()}</span>
                    <div style="flex:1;">
                        <div style="color:#c9d1d9; font-size:0.76rem;">{risk.get('risk', '')}</div>
                        <div style="font-size:0.68rem; color:#656d76; margin-top:2px;">
                            <span style="color:{cat_color}">{risk.get('category', '').title()}</span>
                            · {risk.get('mitigation', '')[:60]}...
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 30/60/90 actions
        st.markdown('<div class="section-label" style="margin-top:0.5rem">30 / 60 / 90 Day Action Plan</div>',
                    unsafe_allow_html=True)
        actions = details.get("30_60_90_actions", {})
        ac1, ac2, ac3 = st.columns(3)

        for col, (label, key, color) in zip(
            [ac1, ac2, ac3],
            [("30 DAYS", "30_days", "#f85149"),
             ("60 DAYS", "60_days", "#e6a817"),
             ("90 DAYS", "90_days", "#7ee787")]
        ):
            with col:
                items = actions.get(key, [])
                items_html = "".join(f'<div class="action-item">{a}</div>' for a in items)
                st.markdown(f"""
                <div style="border-left:2px solid {color}; padding-left:12px;">
                    <div style="font-size:0.65rem; color:{color}; text-transform:uppercase;
                                letter-spacing:0.1em; margin-bottom:8px;">{label}</div>
                    {items_html}
                </div>
                """, unsafe_allow_html=True)

        # Watch metrics
        watch = details.get("metrics_to_watch", [])
        if watch:
            st.markdown('<div class="section-label" style="margin-top:0.5rem">Metrics to Watch</div>',
                        unsafe_allow_html=True)
            mw_cols = st.columns(min(len(watch), 4))
            for i, (col, m) in enumerate(zip(mw_cols, watch[:4])):
                with col:
                    trend_emoji = {"improving": "↑", "stable": "→", "declining": "↓"}.get(
                        m.get("trend", "stable"), "→")
                    trend_color = {"improving": "#7ee787", "stable": "#e6a817", "declining": "#f85149"}.get(
                        m.get("trend", "stable"), "#656d76")
                    st.markdown(f"""
                    <div class="kpi-card" style="border-top:2px solid {trend_color};">
                        <div class="kpi-label">{m.get('metric', '')[:30]}</div>
                        <div style="font-size:1rem; font-weight:700; color:{trend_color};
                                    font-family:'Syne',sans-serif;">{m.get('current', '')}</div>
                        <div style="font-size:0.68rem; color:#656d76;">
                            Target: {m.get('target', '')} {trend_emoji}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:4px;
                    padding:2.5rem; text-align:center; margin-top:1rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1rem; color:#f0f6fc; margin-bottom:8px;">
                🏆 Executive Package Not Yet Generated
            </div>
            <div style="font-size:0.8rem; color:#656d76;">
                Run the AI S&OP cycle to generate the executive synthesis: decisions, risk register,
                30/60/90 action plan, and metrics dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 7: AGENTS ──────────────────────────────────────────────────────────────
with tab_agents:
    # Run button
    can_run = bool(api_key) and not st.session_state.running

    bt_col, info_col = st.columns([2, 3])
    with bt_col:
        if st.button(
            "▶ RUN AI S&OP CYCLE" if not st.session_state.running else "⏳ RUNNING...",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
            key="run_sop"
        ):
            st.session_state.running = True
            st.rerun()

    with info_col:
        if not api_key:
            st.warning("Enter OpenAI API key to activate the LangGraph agent pipeline")
        else:
            st.success(f"Ready · 5 agents · gpt-4o-mini · {sop.cycle_month} cycle")

    if st.session_state.running and api_key:
        from agents import run_sop_cycle

        prog = st.progress(0)
        status = st.empty()

        phase_names = [
            "📊 Demand Intelligence Agent analyzing forecast signals...",
            "⚙️  Supply & Capacity Agent reviewing constraints...",
            "💰 Financial Reconciliation Agent bridging P&L...",
            "🎯 Scenario Planning Agent generating 4 scenarios...",
            "🏆 Executive Synthesis Agent preparing package...",
        ]

        for i, name in enumerate(phase_names[:2]):
            status.info(name)
            prog.progress((i + 1) / 6)
            time.sleep(0.3)

        try:
            status.info("🧠 LangGraph pipeline executing (gpt-4o-mini × 5 agents)...")
            prog.progress(0.35)

            final_sop, messages, scenarios = run_sop_cycle(sop, api_key)

            prog.progress(1.0)
            status.success("✅ AI S&OP cycle complete — all 5 agents executed")

            st.session_state.sop_data = final_sop
            st.session_state.agent_log = messages
            st.session_state.cycle_count += 1
            st.session_state.running = False

            time.sleep(1)
            st.rerun()

        except Exception as e:
            prog.empty()
            status.empty()
            st.session_state.running = False
            st.error(f"Pipeline error: {str(e)}")
            st.info("Verify your OpenAI API key has access to gpt-4o-mini")

    # Agent log display
    if st.session_state.agent_log:
        st.markdown('<div class="section-label" style="margin-top:1rem">Agent Execution Log</div>',
                    unsafe_allow_html=True)

        phase_class_map = {
            "demand": "agent-demand", "supply": "agent-supply",
            "financial": "agent-financial", "scenarios": "agent-scenarios",
            "executive": "agent-executive",
        }

        st.markdown('<div class="log-scroll">', unsafe_allow_html=True)
        for msg in st.session_state.agent_log:
            phase_class = phase_class_map.get(msg.get("phase", ""), "agent-demand")
            ts = msg.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts).strftime("%H:%M:%S")
            except:
                pass

            health = msg.get("health_score", None)
            health_html = ""
            if health:
                hc = "#f85149" if health < 60 else "#e6a817" if health < 75 else "#7ee787"
                health_html = f'<span style="color:{hc}; font-size:0.65rem; margin-left:auto;">health: {health:.0f}</span>'

            st.markdown(f"""
            <div class="agent-card {phase_class}">
                <div class="agent-meta">
                    <span style="font-size:1rem">{msg.get('icon','🤖')}</span>
                    <span class="agent-name">{msg.get('agent','')}</span>
                    <span class="agent-pill">{msg.get('model','gpt-4o-mini')}</span>
                    <span class="agent-pill">{msg.get('phase','').upper()}</span>
                    <span style="color:#30363d; font-size:0.65rem;">{ts}</span>
                    {health_html}
                </div>
                <div class="agent-body">{msg.get('content','')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Show expandable details
            details = msg.get("details", {})
            if details and isinstance(details, dict) and not details.get("error"):
                agent = msg.get("agent", "")
                if agent == "Scenario Planning" and "triggers" in details:
                    with st.expander("View decision triggers & hedging strategy"):
                        st.markdown(f"**Hedging:** {details.get('hedging', '')}")
                        triggers = details.get("triggers", [])
                        for t in triggers:
                            st.markdown(f"""
                            <div style="background:#0d1117; border:1px solid #21262d;
                                        border-radius:3px; padding:6px 10px; margin:3px 0;
                                        font-size:0.75rem; color:#8b949e;">
                                <span style="color:#e6a817;">Trigger:</span> {t.get('trigger','')} →
                                <span style="color:#7ee787;">{t.get('action','')}</span>
                            </div>
                            """, unsafe_allow_html=True)

                elif agent == "Supply & Capacity" and "capacity_actions" in details:
                    with st.expander("View capacity actions"):
                        for action in details.get("capacity_actions", []):
                            cost = action.get('cost_estimate', 0)
                            st.markdown(f"""
                            <div style="background:#0d1117; border:1px solid #21262d;
                                        border-radius:3px; padding:8px 10px; margin:3px 0;
                                        font-size:0.75rem; color:#8b949e;">
                                <b style="color:#c9d1d9;">{action.get('action','')}</b>
                                <span style="color:#2dd4bf; margin-left:8px;">{action.get('resource','')}</span>
                                · ${cost/1000:.0f}K · {action.get('timing','')}
                                <div style="color:#656d76; margin-top:2px;">{action.get('benefit','')}</div>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Architecture diagram
        st.markdown("""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:4px;
                    padding:2rem; margin-top:1rem;">
            <div style="font-size:0.65rem; color:#656d76; text-transform:uppercase;
                        letter-spacing:0.12em; margin-bottom:1.2rem;">
                LangGraph S&OP Agent Architecture
            </div>
            <div style="display:flex; flex-direction:column; gap:0; max-width:400px;">
                <div style="background:#0d1117; border:1px solid #79c0ff; border-radius:3px;
                            padding:10px 14px; font-size:0.78rem; color:#79c0ff;">
                    📊 Demand Intelligence Agent
                    <div style="font-size:0.65rem; color:#656d76; margin-top:2px;">
                    Forecast analysis · MAPE assessment · Consensus adjustments
                    </div>
                </div>
                <div style="padding:3px 0 3px 18px; color:#30363d; font-size:0.8rem;">↓</div>
                <div style="background:#0d1117; border:1px solid #2dd4bf; border-radius:3px;
                            padding:10px 14px; font-size:0.78rem; color:#2dd4bf;">
                    ⚙️ Supply & Capacity Agent
                    <div style="font-size:0.65rem; color:#656d76; margin-top:2px;">
                    Constraint analysis · Supply-demand balancing · Capacity actions
                    </div>
                </div>
                <div style="padding:3px 0 3px 18px; color:#30363d; font-size:0.8rem;">↓</div>
                <div style="background:#0d1117; border:1px solid #e6a817; border-radius:3px;
                            padding:10px 14px; font-size:0.78rem; color:#e6a817;">
                    💰 Financial Reconciliation Agent
                    <div style="font-size:0.65rem; color:#656d76; margin-top:2px;">
                    P&L bridge · Revenue gap analysis · Financial levers
                    </div>
                </div>
                <div style="padding:3px 0 3px 18px; color:#30363d; font-size:0.8rem;">↓</div>
                <div style="background:#0d1117; border:1px solid #bc8cff; border-radius:3px;
                            padding:10px 14px; font-size:0.78rem; color:#bc8cff;">
                    🎯 Scenario Planning Agent
                    <div style="font-size:0.65rem; color:#656d76; margin-top:2px;">
                    4 scenarios · Quantified assumptions · Decision triggers
                    </div>
                </div>
                <div style="padding:3px 0 3px 18px; color:#30363d; font-size:0.8rem;">↓</div>
                <div style="background:#0d1117; border:1px solid #7ee787; border-radius:3px;
                            padding:10px 14px; font-size:0.78rem; color:#7ee787;">
                    🏆 Executive Synthesis Agent
                    <div style="font-size:0.65rem; color:#656d76; margin-top:2px;">
                    CEO package · Risk register · 30/60/90 action plan
                    </div>
                </div>
            </div>
            <div style="font-size:0.68rem; color:#30363d; margin-top:1.2rem;">
                All agents use <code style="color:#656d76; background:#0d1117; padding:1px 4px;">gpt-4o-mini</code>
                via LangGraph conditional routing · Enter API key to activate
            </div>
        </div>
        """, unsafe_allow_html=True)
