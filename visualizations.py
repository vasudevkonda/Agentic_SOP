"""
S&OP Visualization Library
Bloomberg terminal aesthetic: dark slate, amber + teal accents, dense data displays
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from models import SOPCycleData, Scenario, RiskTier

# ─── Color Palette ─────────────────────────────────────────────────────────────
SLATE_900 = "#0d1117"
SLATE_800 = "#161b22"
SLATE_700 = "#21262d"
SLATE_600 = "#30363d"
SLATE_400 = "#656d76"
SLATE_200 = "#c9d1d9"
AMBER   = "#e6a817"
TEAL    = "#2dd4bf"
CRIMSON = "#f85149"
SAGE    = "#7ee787"
VIOLET  = "#bc8cff"
AZURE   = "#79c0ff"
ORANGE  = "#ffa657"

SCENARIO_COLORS = {
    "base_case":   AZURE,
    "optimistic":  SAGE,
    "pessimistic": ORANGE,
    "stress_test": CRIMSON,
    "custom":      VIOLET,
}

RISK_COLORS = {
    "low":      SAGE,
    "moderate": AMBER,
    "high":     ORANGE,
    "critical": CRIMSON,
}

FAMILY_COLORS = [AZURE, TEAL, VIOLET, AMBER]

LAYOUT_BASE = dict(
    paper_bgcolor=SLATE_900,
    plot_bgcolor=SLATE_800,
    font=dict(family="'IBM Plex Mono', monospace", color=SLATE_200, size=11),
    margin=dict(l=50, r=30, t=40, b=40),
)


def _apply_base(fig, title="", height=300):
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(color=SLATE_200, size=12), x=0, xanchor='left'),
        height=height,
        xaxis=dict(gridcolor=SLATE_600, zeroline=False, tickfont=dict(color=SLATE_400)),
        yaxis=dict(gridcolor=SLATE_600, zeroline=False, tickfont=dict(color=SLATE_400)),
        legend=dict(bgcolor=SLATE_800, bordercolor=SLATE_600, borderwidth=1,
                    font=dict(color=SLATE_200, size=10)),
    )
    return fig


# ─── Demand Trend Chart ────────────────────────────────────────────────────────

def demand_trend_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    families = df["Product Family"].unique()
    for i, fam in enumerate(families):
        sub = df[df["Product Family"] == fam]
        color = FAMILY_COLORS[i % len(FAMILY_COLORS)]
        fig.add_trace(go.Scatter(
            x=sub["Month"], y=sub["Demand"],
            name=fam, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{fam}</b><br>%{{x}}: %{{y:,.0f}} units<extra></extra>",
        ))
    _apply_base(fig, "TRAILING 12-MONTH DEMAND TREND", height=280)
    fig.update_layout(
        xaxis=dict(gridcolor=SLATE_600, tickfont=dict(color=SLATE_400, size=9)),
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    return fig


# ─── Forecast Accuracy Heatmap ────────────────────────────────────────────────

def forecast_accuracy_chart(sop: SOPCycleData) -> go.Figure:
    families = list({d.product_family for d in sop.demand_signals})
    periods = list({d.period for d in sop.demand_signals})[:6]

    z = []
    text = []
    for fam in families:
        row = []
        text_row = []
        for per in periods:
            sig = next((d for d in sop.demand_signals if d.product_family == fam and d.period == per), None)
            if sig:
                row.append(sig.mape * 100)
                text_row.append(f"{sig.mape*100:.1f}%")
            else:
                row.append(0)
                text_row.append("")
        z.append(row)
        text.append(text_row)

    short_families = [f.split()[0] for f in families]


    
    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=short_families,
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[[0, SAGE], [0.3, AMBER], [0.6, ORANGE], [1.0, CRIMSON]],
        zmin=0, zmax=15,
        showscale=True,
        colorbar=dict(
            title="MAPE %", title=dict(text='My title', font=dict(color=SLATE_400, size=10),
            tickfont=dict(color=SLATE_400, size=9),
            ticksuffix="%",
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: MAPE = %{z:.1f}%<extra></extra>",
    ))
    _apply_base(fig, "FORECAST ACCURACY — MAPE BY FAMILY × PERIOD", height=220)
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=9, color=SLATE_400)),
        yaxis=dict(tickfont=dict(size=9, color=SLATE_400)),
    )
    return fig


# ─── Capacity Waterfall ────────────────────────────────────────────────────────

def capacity_utilization_chart(sop: SOPCycleData) -> go.Figure:
    resources = [c.resource for c in sop.capacity_data]
    utilized = [c.utilized_capacity for c in sop.capacity_data]
    available = [c.available_capacity - c.utilized_capacity for c in sop.capacity_data]
    pcts = [c.utilization_pct for c in sop.capacity_data]
    colors = [CRIMSON if p >= 90 else AMBER if p >= 75 else TEAL for p in pcts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Utilized", x=resources, y=utilized,
        marker_color=colors,
        text=[f"{p:.0f}%" for p in pcts],
        textposition="inside", textfont=dict(color="white", size=10),
        hovertemplate="<b>%{x}</b><br>Utilized: %{y:,.0f}<br>Utilization: %{text}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Available Slack", x=resources, y=available,
        marker_color=SLATE_600, opacity=0.5,
        hovertemplate="<b>%{x}</b><br>Slack: %{y:,.0f} units<extra></extra>",
    ))

    # 90% constraint line
    fig.add_hline(y=max(c.available_capacity for c in sop.capacity_data) * 0.9,
                  line_dash="dash", line_color=CRIMSON, opacity=0.6,
                  annotation_text="90% constraint threshold",
                  annotation_font_color=CRIMSON, annotation_font_size=9)

    _apply_base(fig, "CAPACITY UTILIZATION BY RESOURCE", height=280)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


# ─── Inventory DOH Chart ──────────────────────────────────────────────────────

def inventory_doh_chart(sop: SOPCycleData) -> go.Figure:
    fams = [i.product_family.split()[0] for i in sop.inventory_positions]
    doh = [i.doh for i in sop.inventory_positions]
    wos = [i.weeks_of_supply for i in sop.inventory_positions]
    excess = [i.excess_value / 1000 for i in sop.inventory_positions]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = [CRIMSON if d > 45 else AMBER if d > 35 else TEAL for d in doh]

    fig.add_trace(go.Bar(
        name="Days on Hand", x=fams, y=doh,
        marker_color=colors,
        text=[f"{d:.1f}d" for d in doh],
        textposition="outside", textfont=dict(color=SLATE_200, size=10),
        hovertemplate="<b>%{x}</b><br>DOH: %{y:.1f} days<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name="Excess Value ($K)", x=fams, y=excess,
        mode="lines+markers",
        line=dict(color=ORANGE, width=2, dash="dot"),
        marker=dict(size=8, color=ORANGE, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Excess: $%{y:,.0f}K<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=30, line_dash="dash", line_color=AMBER, opacity=0.5,
                  annotation_text="Target: 30 DOH", annotation_font_color=AMBER,
                  annotation_font_size=9)

    _apply_base(fig, "INVENTORY: DAYS ON HAND + EXCESS VALUE", height=280)
    fig.update_yaxes(title_text="Days on Hand", title_font_color=SLATE_400,
                     gridcolor=SLATE_600, secondary_y=False)
    fig.update_yaxes(title_text="Excess Value ($K)", title_font_color=ORANGE,
                     secondary_y=True, showgrid=False)
    return fig


# ─── Scenario Tornado Chart ───────────────────────────────────────────────────

def scenario_tornado_chart(scenarios: list[Scenario]) -> go.Figure:
    fig = go.Figure()

    names = [s.name for s in scenarios]
    rev_impacts = [s.revenue_impact / 1e6 for s in scenarios]
    mar_impacts = [s.margin_impact / 1e6 for s in scenarios]
    colors_rev = [CRIMSON if v < 0 else SAGE for v in rev_impacts]
    colors_mar = [CRIMSON if v < 0 else TEAL for v in mar_impacts]

    fig.add_trace(go.Bar(
        name="Revenue Impact ($M)", x=names, y=rev_impacts,
        marker_color=colors_rev,
        text=[f"${v:+.1f}M" for v in rev_impacts],
        textposition="outside", textfont=dict(color=SLATE_200, size=10),
        hovertemplate="<b>%{x}</b><br>Revenue: %{text}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Margin Impact ($M)", x=names, y=mar_impacts,
        marker_color=colors_mar, opacity=0.85,
        text=[f"${v:+.1f}M" for v in mar_impacts],
        textposition="inside", textfont=dict(color="white", size=10),
        hovertemplate="<b>%{x}</b><br>Margin: %{text}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=SLATE_400, line_width=1)

    _apply_base(fig, "SCENARIO FINANCIAL IMPACT — REVENUE & MARGIN ($M)", height=300)
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(tickprefix="$", ticksuffix="M", gridcolor=SLATE_600),
    )
    return fig


# ─── Scenario Probability Bubble ─────────────────────────────────────────────

def scenario_probability_chart(scenarios: list[Scenario]) -> go.Figure:
    fig = go.Figure()

    for s in scenarios:
        color = SCENARIO_COLORS.get(s.type.value, VIOLET)
        fig.add_trace(go.Scatter(
            x=[s.revenue_impact / 1e6],
            y=[s.margin_impact / 1e6],
            mode="markers+text",
            name=s.name,
            text=[s.name],
            textposition="top center",
            textfont=dict(color=color, size=9),
            marker=dict(
                size=s.probability * 120 + 20,
                color=color,
                opacity=0.75,
                line=dict(color=color, width=2),
            ),
            hovertemplate=(
                f"<b>{s.name}</b><br>"
                f"Revenue: ${s.revenue_impact/1e6:+.1f}M<br>"
                f"Margin: ${s.margin_impact/1e6:+.1f}M<br>"
                f"Probability: {s.probability:.0%}<br>"
                f"Risk: {s.risk_tier.value}<extra></extra>"
            ),
        ))

    fig.add_vline(x=0, line_color=SLATE_400, line_width=1, opacity=0.5)
    fig.add_hline(y=0, line_color=SLATE_400, line_width=1, opacity=0.5)

    _apply_base(fig, "SCENARIO SPACE — PROBABILITY × FINANCIAL IMPACT", height=300)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Revenue Impact ($M)", title_font_color=SLATE_400,
                   tickprefix="$", ticksuffix="M"),
        yaxis=dict(title="Margin Impact ($M)", title_font_color=SLATE_400,
                   tickprefix="$", ticksuffix="M"),
    )
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="Bubble size = probability",
                       font=dict(color=SLATE_400, size=9),
                       showarrow=False, xanchor="left", yanchor="top")
    return fig


# ─── Financial Bridge Waterfall ───────────────────────────────────────────────

def financial_bridge_chart(sop: SOPCycleData) -> go.Figure:
    fb = sop.financial_bridge
    if not fb:
        return go.Figure()

    plan = fb.revenue_plan / 1e6
    le = fb.revenue_latest_estimate / 1e6
    gap = fb.revenue_variance / 1e6

    # Simulate bridge components
    demand_var = gap * 0.55
    mix_var = gap * 0.20
    price_var = gap * 0.15
    fx_var = gap * 0.10

    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    x_labels = ["Plan", "Demand", "Mix/Volume", "Price", "FX/Other", "LE"]
    y_vals = [plan, demand_var, mix_var, price_var, fx_var, 0]

    colors_list = [AZURE, CRIMSON if demand_var < 0 else SAGE,
                   CRIMSON if mix_var < 0 else SAGE,
                   CRIMSON if price_var < 0 else SAGE,
                   CRIMSON if fx_var < 0 else SAGE, AMBER]

    fig = go.Figure(go.Waterfall(
        measure=measures,
        x=x_labels,
        y=y_vals,
        text=[f"${v:+.1f}M" if i > 0 and i < 5 else f"${v:.1f}M" for i, v in enumerate(y_vals)],
        textposition="outside",
        textfont=dict(color=SLATE_200, size=10),
        connector=dict(line=dict(color=SLATE_400, dash="dot", width=1)),
        increasing=dict(marker=dict(color=SAGE)),
        decreasing=dict(marker=dict(color=CRIMSON)),
        totals=dict(marker=dict(color=AMBER)),
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
    ))

    _apply_base(fig, "REVENUE BRIDGE: PLAN TO LATEST ESTIMATE ($M)", height=280)
    fig.update_layout(
        yaxis=dict(tickprefix="$", ticksuffix="M"),
        showlegend=False,
    )
    return fig


# ─── Plan Confidence Gauge ────────────────────────────────────────────────────

def plan_confidence_gauge(confidence: float, health: float) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )

    conf_color = CRIMSON if confidence < 0.55 else AMBER if confidence < 0.75 else SAGE

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title=dict(text="PLAN CONFIDENCE", font=dict(size=11, color=SLATE_400)),
        number=dict(suffix="%", font=dict(size=28, color=conf_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=SLATE_400,
                      tickfont=dict(color=SLATE_400, size=9)),
            bar=dict(color=conf_color),
            bgcolor=SLATE_700,
            bordercolor=SLATE_600,
            steps=[
                dict(range=[0, 55], color="#2d0f0f"),
                dict(range=[55, 75], color="#2d2000"),
                dict(range=[75, 100], color="#0f2d1a"),
            ],
        ),
    ), row=1, col=1)

    health_color = CRIMSON if health < 55 else AMBER if health < 72 else SAGE

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=health,
        title=dict(text="S&OP HEALTH", font=dict(size=11, color=SLATE_400)),
        number=dict(suffix="/100", font=dict(size=28, color=health_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=SLATE_400,
                      tickfont=dict(color=SLATE_400, size=9)),
            bar=dict(color=health_color),
            bgcolor=SLATE_700,
            bordercolor=SLATE_600,
            steps=[
                dict(range=[0, 55], color="#2d0f0f"),
                dict(range=[55, 72], color="#2d2000"),
                dict(range=[72, 100], color="#0f2d1a"),
            ],
        ),
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=SLATE_900,
        font=dict(family="'IBM Plex Mono', monospace", color=SLATE_200),
        height=200,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig
