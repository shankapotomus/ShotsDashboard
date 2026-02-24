"""
app.py  –  Shots Dashboard
--------------------------
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_all_data
from metrics import (
    BUCKET_LABELS, BUCKET_ORDER,
    CLOCK_LABELS, CLOCK_ORDER,
    ZONE_LABELS, ZONE_ORDER,
    compute_assisted_metrics,
    compute_bucket_metrics,
    compute_clock_zone_metrics,
    compute_four_factors,
    compute_poss_type_summary,
    compute_team_conference,
    compute_team_record,
    compute_zone_metrics,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shots Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* hide default header padding */
    .block-container { padding-top: 1rem; }

    /* metric card style */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3150;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { font-size: 0.75rem; color: #9aa0b0; }
    [data-testid="stMetricValue"]  { font-size: 1.7rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.75rem; }

    /* section dividers */
    .section-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #9aa0b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.2rem 0 0.4rem 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2d3150;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
ZONE_COLORS = {
    "Rim":              "#e05c5c",
    "Paint (non-rim)":  "#e07c40",
    "Mid-Range":        "#9a9a9a",
    "Corner 3":         "#4e9af1",
    "Above-Break 3":    "#a06cf5",
}
BUCKET_COLORS = {
    "Transition":           "#4ec9b0",
    "1st HC Attempt":       "#4e9af1",
    "Putbacks / Scrambles": "#e07c40",
    "2nd HC Attempt":       "#a06cf5",
    "3rd+ HC Attempt":      "#e05c5c",
}
PLOTLY_TEMPLATE = "plotly_dark"


# ── Helpers ────────────────────────────────────────────────────────────────────
def pct_fmt(v, decimals=1) -> str:
    return "—" if pd.isna(v) else f"{v:.{decimals}f}%"


def _bar_fig(df, x, y, color_map, title="", xrange=None, orientation="h"):
    """Thin wrapper to make a horizontal bar chart with custom colours."""
    df = df.copy()
    df["_color"] = df[color_map].map(
        lambda k: ZONE_COLORS.get(k, BUCKET_COLORS.get(k, "#6c7a9c"))
    )
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row[x]] if orientation == "h" else [row[y]],
                y=[row[y]] if orientation == "h" else [row[x]],
                orientation=orientation,
                name=row[color_map],
                marker_color=row["_color"],
                showlegend=False,
                text=[f"{row[x]:.1f}%"] if orientation == "h" else [f"{row[y]:.1f}%"],
                textposition="outside",
                insidetextanchor="middle",
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font_size=13),
        margin=dict(l=10, r=40, t=35, b=10),
        height=280,
        barmode="relative",
        xaxis=dict(range=xrange or [0, df[x].max() * 1.25], showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Load data (cached) ─────────────────────────────────────────────────────────
shots, possessions, four_factors, games = load_all_data()

all_teams = sorted(
    shots["team"].dropna().unique().tolist()
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 Shots Dashboard")
    st.caption("2025–26 Season")

    selected_team = st.selectbox(
        "Team", all_teams, index=all_teams.index("Duke") if "Duke" in all_teams else 0
    )

    side = st.radio("Perspective", ["Offense", "Defense"], horizontal=True)

    st.divider()
    st.caption(
        "Shot data from CFBD/ESPN via daily pipeline. "
        "FTs excluded from shot charts."
    )

# ── Team header ────────────────────────────────────────────────────────────────
conf  = compute_team_conference(games, selected_team)
rec   = compute_team_record(games, selected_team)

h1, h2 = st.columns([3, 1])
with h1:
    st.title(f"{selected_team}")
    st.caption(f"{conf}  ·  {rec}  ·  **{side}**")
with h2:
    st.write("")

st.markdown("---")

# ── Compute metrics ────────────────────────────────────────────────────────────
ff_stats   = compute_four_factors(four_factors, shots, selected_team, side)
zone_df    = compute_zone_metrics(shots, selected_team, side)
bucket_df  = compute_bucket_metrics(shots, selected_team, side)
clock_df   = compute_clock_zone_metrics(shots, selected_team, side)
ast_df     = compute_assisted_metrics(shots, selected_team, side)
poss_sum   = compute_poss_type_summary(possessions, games, selected_team, side)

# ── SECTION 1: Four Factors ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Four Factors</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("eFG%", pct_fmt(ff_stats.get("efg")))
with c2:
    st.metric("Turnover %", pct_fmt(ff_stats.get("to_pct")))
with c3:
    st.metric("Off Reb %", pct_fmt(ff_stats.get("orb_pct")))
with c4:
    st.metric("FT Rate (FTM/FGA)", pct_fmt(ff_stats.get("ft_rate")))
with c5:
    st.metric("Tempo (poss/g)", f"{ff_stats.get('tempo', 0):.1f}")

# ── SECTION 2: Possession-type summary ────────────────────────────────────────
st.markdown('<div class="section-header">Possession Breakdown</div>', unsafe_allow_html=True)

if not poss_sum.empty:
    p_cols = st.columns(len(poss_sum))
    ptype_color = {
        "Transition":       "#4ec9b0",
        "Half Court":       "#4e9af1",
        "Second Chance":    "#e07c40",
        "Scramble Putback": "#e05c5c",
    }
    for i, row in poss_sum.iterrows():
        with p_cols[i]:
            c = ptype_color.get(row["poss_type"], "#9aa0b0")
            st.markdown(
                f"""
                <div style="background:#1e2130;border:1px solid #2d3150;border-top:3px solid {c};
                            border-radius:8px;padding:12px;text-align:center;">
                  <div style="font-size:0.72rem;color:#9aa0b0;text-transform:uppercase;
                               letter-spacing:.06em;">{row['poss_type']}</div>
                  <div style="font-size:1.5rem;font-weight:700;margin:4px 0">{row['share']:.1f}%</div>
                  <div style="font-size:0.75rem;color:#9aa0b0">
                    FG%&nbsp;{pct_fmt(row['fg_pct'])} &nbsp;|&nbsp; TO%&nbsp;{pct_fmt(row['to_pct'])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── SECTION 3: Possession bucket × shot efficiency ────────────────────────────
st.markdown(
    '<div class="section-header">Shot Attempt Context — Possession Buckets</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Transition → 1st half-court set → Putbacks/scrambles → 2nd HC (after OREB) → 3rd+ HC"
)

if not bucket_df.empty:
    b_left, b_right = st.columns(2)

    with b_left:
        fig_bshare = go.Figure()
        for _, row in bucket_df.iterrows():
            c = BUCKET_COLORS.get(row["bucket"], "#6c7a9c")
            fig_bshare.add_trace(
                go.Bar(
                    x=[row["fga_pct"]],
                    y=[row["bucket"]],
                    orientation="h",
                    name=row["bucket"],
                    marker_color=c,
                    showlegend=False,
                    text=[f"{row['fga_pct']:.1f}%"],
                    textposition="outside",
                )
            )
        fig_bshare.update_layout(
            template=PLOTLY_TEMPLATE,
            title="FGA Share by Bucket",
            margin=dict(l=10, r=50, t=35, b=10),
            height=260,
            xaxis=dict(range=[0, bucket_df["fga_pct"].max() * 1.3], showgrid=False),
            yaxis=dict(
                categoryorder="array",
                categoryarray=[BUCKET_LABELS[b] for b in reversed(BUCKET_ORDER)],
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bshare, use_container_width=True)

    with b_right:
        # Grouped bar: FG% and eFG% per bucket
        fig_befg = go.Figure()
        bcolors = [BUCKET_COLORS.get(b, "#6c7a9c") for b in bucket_df["bucket"]]
        fig_befg.add_trace(
            go.Bar(
                x=[BUCKET_LABELS[b] for b in BUCKET_ORDER],
                y=bucket_df["fg_pct"].tolist(),
                name="FG%",
                marker_color=bcolors,
                opacity=0.5,
                text=[pct_fmt(v) for v in bucket_df["fg_pct"]],
                textposition="outside",
            )
        )
        fig_befg.add_trace(
            go.Bar(
                x=[BUCKET_LABELS[b] for b in BUCKET_ORDER],
                y=bucket_df["efg_pct"].tolist(),
                name="eFG%",
                marker_color=bcolors,
                text=[pct_fmt(v) for v in bucket_df["efg_pct"]],
                textposition="outside",
            )
        )
        fig_befg.update_layout(
            template=PLOTLY_TEMPLATE,
            title="FG% and eFG% by Bucket",
            margin=dict(l=10, r=10, t=35, b=60),
            height=260,
            barmode="group",
            yaxis=dict(range=[0, 90], showgrid=True, gridcolor="#2d3150"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_befg, use_container_width=True)

    # Bucket summary table
    with st.expander("Bucket detail table", expanded=False):
        disp = bucket_df[["bucket", "fga", "fga_pct", "fg_pct", "efg_pct", "ast_pct"]].copy()
        disp.columns = ["Bucket", "FGA", "FGA%", "FG%", "eFG%", "Ast%"]
        st.dataframe(disp.set_index("Bucket"), use_container_width=True)

# ── SECTION 4: Shot Zones ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Shot Zones</div>', unsafe_allow_html=True)

if not zone_df.empty:
    z_left, z_right = st.columns(2)

    with z_left:
        # Horizontal bar: FGA share
        fig_zshare = go.Figure()
        for _, row in zone_df.iterrows():
            c = ZONE_COLORS.get(row["zone"], "#6c7a9c")
            fig_zshare.add_trace(
                go.Bar(
                    x=[row["fga_pct"]],
                    y=[row["zone"]],
                    orientation="h",
                    marker_color=c,
                    showlegend=False,
                    text=[f"{row['fga_pct']:.1f}%"],
                    textposition="outside",
                )
            )
        fig_zshare.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Shot Distribution",
            margin=dict(l=10, r=50, t=35, b=10),
            height=260,
            xaxis=dict(range=[0, zone_df["fga_pct"].max() * 1.3], showgrid=False),
            yaxis=dict(
                categoryorder="array",
                categoryarray=[ZONE_LABELS[z] for z in reversed(ZONE_ORDER)],
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_zshare, use_container_width=True)

    with z_right:
        # eFG% dots per zone
        fig_zefg = go.Figure()
        for _, row in zone_df.iterrows():
            c = ZONE_COLORS.get(row["zone"], "#6c7a9c")
            fig_zefg.add_trace(
                go.Scatter(
                    x=[row["efg_pct"]],
                    y=[row["zone"]],
                    mode="markers+text",
                    marker=dict(size=22, color=c),
                    text=[f"{row['efg_pct']:.1f}%"],
                    textposition="middle center",
                    textfont=dict(size=10, color="white"),
                    name=row["zone"],
                    showlegend=False,
                )
            )
        fig_zefg.update_layout(
            template=PLOTLY_TEMPLATE,
            title="eFG% by Zone",
            margin=dict(l=10, r=20, t=35, b=10),
            height=260,
            xaxis=dict(range=[20, 90], showgrid=True, gridcolor="#2d3150", title="eFG%"),
            yaxis=dict(
                categoryorder="array",
                categoryarray=[ZONE_LABELS[z] for z in reversed(ZONE_ORDER)],
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        # reference line at 50% eFG
        fig_zefg.add_vline(
            x=50, line_dash="dash", line_color="#555", annotation_text="50%", annotation_font_size=10
        )
        st.plotly_chart(fig_zefg, use_container_width=True)

    # Zone summary table
    with st.expander("Zone detail table", expanded=False):
        disp = zone_df[["zone", "fga", "fga_pct", "fg_pct", "efg_pct", "ast_pct"]].copy()
        disp.columns = ["Zone", "FGA", "FGA%", "FG%", "eFG%", "Ast%"]
        st.dataframe(disp.set_index("Zone"), use_container_width=True)

# ── SECTION 5: Shot timing × zone heatmap ─────────────────────────────────────
st.markdown(
    '<div class="section-header">Shot Context — When in the Possession?</div>',
    unsafe_allow_html=True,
)
st.caption("Early = ≤8s | Normal = 9–25s | Late clock = 25–30s | Extra = putback/scramble")

if not clock_df.empty:
    clock_left, clock_right = st.columns(2)

    with clock_left:
        # FGA count heatmap
        pivot_fga = (
            clock_df.pivot(index="zone", columns="clock_bucket", values="fga")
            .reindex(
                index=[ZONE_LABELS[z] for z in ZONE_ORDER],
                columns=[CLOCK_LABELS[c] for c in CLOCK_ORDER],
            )
            .fillna(0)
        )
        fig_heat_fga = px.imshow(
            pivot_fga,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title="FGA Count",
            labels=dict(color="FGA"),
        )
        fig_heat_fga.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=35, b=10),
            height=260,
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat_fga, use_container_width=True)

    with clock_right:
        # eFG% heatmap
        pivot_efg = (
            clock_df.pivot(index="zone", columns="clock_bucket", values="efg_pct")
            .reindex(
                index=[ZONE_LABELS[z] for z in ZONE_ORDER],
                columns=[CLOCK_LABELS[c] for c in CLOCK_ORDER],
            )
        )
        # mask zero-FGA cells
        fga_pivot = (
            clock_df.pivot(index="zone", columns="clock_bucket", values="fga")
            .reindex(
                index=[ZONE_LABELS[z] for z in ZONE_ORDER],
                columns=[CLOCK_LABELS[c] for c in CLOCK_ORDER],
            )
            .fillna(0)
        )
        pivot_efg = pivot_efg.where(fga_pivot >= 5, np.nan)

        fig_heat_efg = px.imshow(
            pivot_efg,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            zmin=30,
            zmax=75,
            aspect="auto",
            title="eFG% (min 5 FGA)",
            labels=dict(color="eFG%"),
        )
        fig_heat_efg.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=35, b=10),
            height=260,
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat_efg, use_container_width=True)

# ── SECTION 6: Assisted / Unassisted ─────────────────────────────────────────
st.markdown(
    '<div class="section-header">Assisted vs Unassisted</div>',
    unsafe_allow_html=True,
)

if not ast_df.empty:
    a_left, a_right = st.columns(2)

    with a_left:
        # Stacked bar: Assisted% vs Unassisted%
        fig_ast = go.Figure()
        zcolors = [ZONE_COLORS.get(ZONE_LABELS[z], "#6c7a9c") for z in ZONE_ORDER]
        zones_labels = [ZONE_LABELS[z] for z in ZONE_ORDER]

        fig_ast.add_trace(
            go.Bar(
                x=zones_labels,
                y=ast_df["ast_share"].tolist(),
                name="Assisted",
                marker_color=zcolors,
                text=[f"{v:.0f}%" for v in ast_df["ast_share"]],
                textposition="inside",
            )
        )
        fig_ast.add_trace(
            go.Bar(
                x=zones_labels,
                y=(100 - ast_df["ast_share"]).tolist(),
                name="Unassisted",
                marker_color=zcolors,
                opacity=0.3,
                text=[f"{100-v:.0f}%" for v in ast_df["ast_share"]],
                textposition="inside",
                showlegend=True,
            )
        )
        fig_ast.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Assisted Shot Share by Zone",
            barmode="stack",
            margin=dict(l=10, r=10, t=35, b=60),
            height=280,
            yaxis=dict(range=[0, 105], showgrid=False, title="% of FGA"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ast, use_container_width=True)

    with a_right:
        # FG% — Assisted vs Unassisted per zone
        fig_ast2 = go.Figure()
        fig_ast2.add_trace(
            go.Bar(
                x=zones_labels,
                y=ast_df["ast_fg_pct"].tolist(),
                name="Assisted FG%",
                marker_color="#4e9af1",
                text=[pct_fmt(v) for v in ast_df["ast_fg_pct"]],
                textposition="outside",
            )
        )
        fig_ast2.add_trace(
            go.Bar(
                x=zones_labels,
                y=ast_df["unast_fg_pct"].tolist(),
                name="Unassisted FG%",
                marker_color="#e05c5c",
                text=[pct_fmt(v) for v in ast_df["unast_fg_pct"]],
                textposition="outside",
            )
        )
        fig_ast2.update_layout(
            template=PLOTLY_TEMPLATE,
            title="FG% — Assisted vs Unassisted",
            barmode="group",
            margin=dict(l=10, r=10, t=35, b=60),
            height=280,
            yaxis=dict(range=[0, 90], showgrid=True, gridcolor="#2d3150"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ast2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: CFBD / ESPN  ·  Pipeline: daily_fetch.py  ·  2025–26 Season")
