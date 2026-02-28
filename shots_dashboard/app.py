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
    GRADE_COLORS, GRADE_LABELS, GRADE_ORDER,
    ZONE_LABELS, ZONE_ORDER,
    _PTYPE_ORDER, _PTYPE_LABELS,
    compute_assisted_metrics,
    compute_bucket_metrics,
    compute_clock_zone_metrics,
    compute_four_factors,
    compute_league_stats,
    compute_last5_game_breakdown,
    compute_poss_ppp_league,
    compute_poss_type_summary,
    compute_team_conference,
    compute_team_record,
    compute_threept_context,
    compute_zone_metrics,
    get_d1_teams,
    percentile_rank,
    poss_percentile_rank,
    _ordinal,
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


def val_fmt(v, decimals=1) -> str:
    """Format a non-percentage number (points, etc.)."""
    return "—" if pd.isna(v) else f"{v:.{decimals}f}"


# YlOrBr ColorBrewer sequential scale — color-blind safe.
# High percentile = dark orange-red (hot/vibrant), low = pale cream (recedes).
_YLORB_SCALE = [
    (10,  "#FFF7BC"),   # 0–10   very poor   — pale yellow (washed out)
    (25,  "#FEE391"),   # 11–25  poor        — light amber
    (40,  "#FEC44F"),   # 26–40  below avg   — amber
    (55,  "#FB9A29"),   # 41–55  around avg  — medium orange
    (70,  "#EC7014"),   # 56–70  above avg   — deep orange
    (85,  "#D94801"),   # 71–85  good        — burnt orange
    (100, "#CC4C02"),   # 86–100 elite       — dark orange-red (hot)
]


def _pct_color(pct: int | None) -> str:
    """Map 0–100 percentile → YlOrBr hex color (high = dark orange-red)."""
    if pct is None:
        return "#6c7a9c"   # muted gray for missing / not ranked
    for threshold, color in _YLORB_SCALE:
        if pct <= threshold:
            return color
    return "#CC4C02"


def _indicator(pct: int | None) -> str:
    """Return colored ✓/✗ indicator HTML based on percentile thresholds.

    Check marks (teal):  ≥66 → ✓,  ≥75 → ✓✓,  ≥95 → ✓✓✓
    Red X marks (red):   ≤33 → ✗,  ≤25 → ✗✗,  ≤5  → ✗✗✗
    """
    if pct is None:
        return ""
    if pct <= 5:
        return '<span style="color:#e05c5c;font-size:0.8rem;">✗✗✗</span>'
    elif pct <= 25:
        return '<span style="color:#e05c5c;font-size:0.8rem;">✗✗</span>'
    elif pct <= 33:
        return '<span style="color:#e05c5c;font-size:0.8rem;">✗</span>'
    elif pct >= 95:
        return '<span style="color:#4ec9b0;font-size:0.8rem;">✓✓✓</span>'
    elif pct >= 75:
        return '<span style="color:#4ec9b0;font-size:0.8rem;">✓✓</span>'
    elif pct >= 66:
        return '<span style="color:#4ec9b0;font-size:0.8rem;">✓</span>'
    return ""


def _ff_card(label: str, value: str, pct: int | None) -> str:
    """Render a Four-Factors metric card with color-scaled percentile + indicator."""
    pct_text  = f"{_ordinal(pct)} pct" if pct is not None else "—"
    color     = _pct_color(pct)
    indicator = _indicator(pct)
    return (
        f'<div style="background:#1e2130;border:1px solid #2d3150;border-radius:8px;'
        f'padding:12px 10px;">'
        f'<div style="font-size:0.65rem;color:#9aa0b0;text-transform:uppercase;'
        f'letter-spacing:.05em;margin-bottom:4px;white-space:nowrap;overflow:hidden;'
        f'text-overflow:ellipsis;">{label}</div>'
        f'<div style="font-size:1.55rem;font-weight:700;color:#ffffff;margin-bottom:6px;white-space:nowrap;">'
        f'{value}</div>'
        f'<div style="font-size:0.68rem;font-weight:600;color:{color};'
        f'display:flex;align-items:center;gap:4px;white-space:nowrap;">'
        f'{pct_text}&nbsp;{indicator}</div>'
        f'</div>'
    )


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


@st.cache_data(show_spinner="Computing D1 percentiles…")
def _get_league_stats(_shots, _ff, _games):
    return compute_league_stats(_shots, _ff, _games)


@st.cache_data(show_spinner="Computing possession percentiles…")
def _get_poss_league_stats(_shots, _poss, _games, _ff):
    return compute_poss_ppp_league(_shots, _poss, _games, _ff)


@st.cache_data(show_spinner="Loading team info…")
def _load_team_info():
    """Fetch team metadata (display name, ESPN logo ID, colors) from CBBD API."""
    import cbbd
    import os
    api_key = os.environ.get("CBBD_API_KEY", "")
    configuration = cbbd.Configuration(access_token=api_key)
    with cbbd.ApiClient(configuration) as api_client:
        teams = cbbd.TeamsApi(api_client).get_teams(season=2026)
    rows = []
    for t in teams:
        d = t.to_dict()
        rows.append({
            "school":       d.get("school", ""),
            "display_name": d.get("displayName", d.get("school", "")),
            "espn_id":      str(d.get("sourceId", "")),
            "primary_color": "#" + str(d.get("primaryColor", "4e9af1")).lstrip("#"),
        })
    return pd.DataFrame(rows).set_index("school")


league_stats      = _get_league_stats(shots, four_factors, games)
poss_league_stats = _get_poss_league_stats(shots, possessions, games, four_factors)
team_info_df      = _load_team_info()

# ── D1 teams (10+ games) ───────────────────────────────────────────────────────
_d1_set = get_d1_teams(games)
all_teams = sorted(
    t for t in shots["team"].dropna().unique() if t in _d1_set
)


def _team_logo_url(school: str) -> str:
    """Return ESPN CDN logo URL for a school, or '' if not found."""
    if school in team_info_df.index:
        espn_id = team_info_df.at[school, "espn_id"]
        if espn_id:
            return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{espn_id}.png"
    return ""


def _display_name(school: str) -> str:
    """Return full display name (e.g. 'Duke Blue Devils') for a school."""
    if school in team_info_df.index:
        return team_info_df.at[school, "display_name"]
    return school

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 Shots Dashboard")
    st.caption("2025–26 Season")

    selected_team = st.selectbox(
        "Team",
        all_teams,
        index=all_teams.index("Duke") if "Duke" in all_teams else 0,
        format_func=_display_name,
    )

    side = st.radio("Perspective", ["Offense", "Defense"], horizontal=True)

    st.divider()
    st.caption(
        "Shot data from CFBD/ESPN via daily pipeline. "
        "FTs excluded from shot charts."
    )

# ── Team header ────────────────────────────────────────────────────────────────
conf      = compute_team_conference(games, selected_team)
rec       = compute_team_record(games, selected_team)
logo_url  = _team_logo_url(selected_team)
full_name = _display_name(selected_team)

logo_html = (
    f'<img src="{logo_url}" width="72" height="72" '
    f'style="border-radius:8px;object-fit:contain;background:transparent;" '
    f'onerror="this.style.display=\'none\'" />'
    if logo_url else ""
)

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;padding:8px 0 4px 0;">
      {logo_html}
      <div>
        <div style="font-size:2rem;font-weight:800;line-height:1.1;">{full_name}</div>
        <div style="font-size:0.9rem;color:#9aa0b0;margin-top:2px;">
          {conf}&nbsp;&nbsp;·&nbsp;&nbsp;{rec}&nbsp;&nbsp;·&nbsp;&nbsp;<strong style="color:#ffffff;">{side}</strong>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Compute metrics ────────────────────────────────────────────────────────────
ff_stats   = compute_four_factors(four_factors, shots, selected_team, side)
zone_df    = compute_zone_metrics(shots, selected_team, side)
bucket_df  = compute_bucket_metrics(shots, selected_team, side)
clock_df   = compute_clock_zone_metrics(shots, selected_team, side)
ast_df     = compute_assisted_metrics(shots, selected_team, side)
poss_sum   = compute_poss_type_summary(possessions, games, selected_team, side, shots=shots, ff=four_factors)
last5_df   = compute_last5_game_breakdown(possessions, games, selected_team, side, shots=shots, ff=four_factors)
threept_ctx = compute_threept_context(shots, selected_team, side)

# ── SECTION 1: Four Factors + Transition ──────────────────────────────────────
st.markdown('<div class="section-header">Four Factors + Transition</div>', unsafe_allow_html=True)

_ff_metrics = [
    ("eFG%",           pct_fmt(ff_stats.get("efg")),              "efg"),
    ("Turnover %",     pct_fmt(ff_stats.get("to_pct")),           "to_pct"),
    ("Off Reb %",      pct_fmt(ff_stats.get("orb_pct")),          "orb_pct"),
    ("FT Rate",        pct_fmt(ff_stats.get("ft_rate")),          "ft_rate"),
    ("Trans Pts / 100", val_fmt(ff_stats.get("trans_pts_per100")), "trans_pts_per100"),
]
for col, (label, value, metric_key) in zip(st.columns(5), _ff_metrics):
    pct = percentile_rank(league_stats, selected_team, side, metric_key)
    with col:
        st.markdown(_ff_card(label, value, pct), unsafe_allow_html=True)

# ── SECTION 2: Possession Breakdown ───────────────────────────────────────────
st.markdown('<div class="section-header">Possession Breakdown</div>', unsafe_allow_html=True)

_PTYPE_COLORS = {
    "transition":       "#4ec9b0",
    "half_court":       "#4e9af1",
    "second_chance":    "#e07c40",
    "scramble_putback": "#e05c5c",
}

if not poss_sum.empty and poss_sum["ppp"].notna().any():
    poss_sum = poss_sum.copy()
    poss_sum["contribution"] = poss_sum["share"] / 100.0 * poss_sum["ppp"].fillna(0)
    overall_ppp = poss_sum["contribution"].sum()

    # ── Stacked PPP bar (season) ───────────────────────────────────────────────
    fig_ppp = go.Figure()
    for _, row in poss_sum.iterrows():
        contrib = row["contribution"]
        c = _PTYPE_COLORS.get(row["poss_type_key"], "#9aa0b0")
        label = _PTYPE_LABELS.get(row["poss_type_key"], row["poss_type"])
        fig_ppp.add_trace(go.Bar(
            x=[contrib],
            y=["PPP"],
            orientation="h",
            name=label,
            marker_color=c,
            showlegend=False,
            text=f"<b>{label}</b><br>{contrib:.2f} pts",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"PPP: {row['ppp']:.2f}<br>"
                f"Freq: {row['share']:.1f}%<br>"
                f"Contribution: {contrib:.2f} pts<extra></extra>"
            ),
        ))

    fig_ppp.add_vline(x=1.0, line_dash="dash", line_color="#555",
                      annotation_text="Avg", annotation_font_size=10,
                      annotation_position="top")
    fig_ppp.add_vline(x=overall_ppp, line_dash="dot", line_color="#9aa0b0",
                      annotation_text=f"{overall_ppp:.2f}",
                      annotation_font_size=10, annotation_position="bottom right")

    fig_ppp.add_annotation(
        text="<b>Season</b>", x=0.5, y=1, xref="paper", yref="paper",
        xanchor="center", yanchor="top", showarrow=False,
        font=dict(size=11, color="#ffffff"),
    )
    fig_ppp.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="stack",
        height=110,
        margin=dict(l=52, r=20, t=14, b=28),
        xaxis=dict(range=[0, 1.5], showgrid=True, gridcolor="#2d3150",
                   tickvals=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ppp, use_container_width=True)

    # ── Last 5 games chart (same PPP scale as season bar) ────────────────────────
    if not last5_df.empty:
        ordered_labels = (
            last5_df.drop_duplicates("game_id")
            .sort_values("game_order")["game_label"]
            .tolist()
        )

        # Convert pts → PPP contribution (pts / total_poss) so scale matches above
        l5 = last5_df.copy()
        l5["contrib"] = l5["pts"] / l5["total_poss"]

        game_summary = (
            l5.drop_duplicates("game_id")
            .sort_values("game_order")
        )

        fig_l5 = go.Figure()

        for pt in _PTYPE_ORDER:
            pt_data = l5[l5["poss_type_key"] == pt].sort_values("game_order").copy()
            if pt_data.empty:
                continue
            c = _PTYPE_COLORS.get(pt, "#9aa0b0")
            label = _PTYPE_LABELS.get(pt, pt)

            # Per-segment label: pts (poss count) + italic PPP for that type
            pt_data["seg_ppp"] = (pt_data["pts"] / pt_data["count"].clip(lower=1)).round(2)
            seg_texts = [
                f"{row['pts']:.0f} pts ({int(row['count'])} poss)  <i>{row['seg_ppp']:.2f} PPP</i>"
                for _, row in pt_data.iterrows()
            ]

            fig_l5.add_trace(go.Bar(
                x=pt_data["contrib"].tolist(),
                y=pt_data["game_label"].tolist(),
                name=label,
                marker_color=c,
                orientation="h",
                showlegend=True,
                text=seg_texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=8, color="#ffffff"),
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "PPP contrib: %{x:.2f}<br>"
                    "Game: %{y}<extra></extra>"
                ),
            ))

        # D1 avg and season PPP reference lines (same as season bar)
        fig_l5.add_vline(x=1.0, line_dash="dash", line_color="#555",
                         annotation_text="Avg", annotation_font_size=10,
                         annotation_position="top")
        fig_l5.add_vline(x=overall_ppp, line_dash="dot", line_color="#9aa0b0",
                         annotation_text=f"Season {overall_ppp:.2f}",
                         annotation_font_size=9, annotation_position="bottom right")

        fig_l5.add_annotation(
            text="<b>Last 5</b>", x=0.5, y=1, xref="paper", yref="paper",
            xanchor="center", yanchor="top", showarrow=False,
            font=dict(size=11, color="#ffffff"),
        )

        # Add opponent logos as layout images on the left side of the y-axis
        for _, gr in game_summary.iterrows():
            logo_url = _team_logo_url(gr["opponent"])
            if logo_url:
                fig_l5.add_layout_image(
                    source=logo_url,
                    x=0,
                    y=gr["game_label"],
                    xref="paper",
                    yref="y",
                    sizex=0.044,   # ~40 px on a 900 px-wide figure
                    sizey=0.72,    # 72% of one categorical row height
                    xanchor="right",
                    yanchor="middle",
                    layer="above",
                )

        fig_l5.update_layout(
            template=PLOTLY_TEMPLATE,
            barmode="stack",
            height=230,
            margin=dict(l=52, r=30, t=14, b=28),
            xaxis=dict(
                range=[0, 1.5],
                showgrid=True,
                gridcolor="#2d3150",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            ),
            yaxis=dict(
                categoryorder="array",
                categoryarray=ordered_labels,
                showgrid=False,
                showticklabels=False,
            ),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                        font_size=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_l5, use_container_width=True)

    # ── Table: frequency + PPP with percentiles ───────────────────────────────
    # Header
    def _td(txt, color="#9aa0b0", bold=False, align="right"):
        fw = "700" if bold else "400"
        return (f'<td style="padding:6px 10px;font-size:0.75rem;color:{color};'
                f'font-weight:{fw};text-align:{align};">{txt}</td>')

    def _pct_badge(pct: int | None) -> str:
        if pct is None:
            return '<span style="color:#6c7a9c;font-size:0.68rem;">—</span>'
        col = _pct_color(pct)
        ind = _indicator(pct)
        return (f'<span style="color:{col};font-size:0.68rem;font-weight:600;">'
                f'{_ordinal(pct)} pct</span>'
                + (f'&nbsp;{ind}' if ind else ''))

    header = (
        '<table style="width:100%;border-collapse:collapse;margin-top:6px;">'
        '<thead><tr style="border-bottom:1px solid #2d3150;">'
        + _td("Possession Type", align="left", bold=True, color="#9aa0b0")
        + _td("Freq %", bold=True, color="#9aa0b0")
        + _td("Freq Pctl", bold=True, color="#9aa0b0")
        + _td("PPP", bold=True, color="#9aa0b0")
        + _td("PPP Pctl", bold=True, color="#9aa0b0")
        + _td("Pts/100", bold=True, color="#9aa0b0")
        + _td("Pts/100 Pctl", bold=True, color="#9aa0b0")
        + '</tr></thead><tbody>'
    )

    body = ""
    for _, row in poss_sum.iterrows():
        pt_key = row["poss_type_key"]
        c = _PTYPE_COLORS.get(pt_key, "#9aa0b0")
        label = _PTYPE_LABELS.get(pt_key, row["poss_type"])

        freq_pct    = poss_percentile_rank(poss_league_stats, selected_team, side, pt_key, "share")
        ppp_pct     = poss_percentile_rank(poss_league_stats, selected_team, side, pt_key, "ppp")
        pts100_pct  = poss_percentile_rank(poss_league_stats, selected_team, side, pt_key, "pts_per100")

        ppp_val    = f"{row['ppp']:.2f}" if pd.notna(row["ppp"]) else "—"
        pts100_val = f"{row['contribution'] * 100:.2f}" if pd.notna(row["ppp"]) else "—"

        body += (
            f'<tr style="border-bottom:1px solid #1e2130;">'
            + _td(
                f'<span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;background:{c};margin-right:7px;"></span>'
                f'<span style="color:#e0e4f0;">{label}</span>',
                align="left",
            )
            + _td(f"{row['share']:.1f}%", color="#e0e4f0", bold=True)
            + _td(_pct_badge(freq_pct))
            + _td(ppp_val, color="#e0e4f0", bold=True)
            + _td(_pct_badge(ppp_pct))
            + _td(pts100_val, color="#e0e4f0", bold=True)
            + _td(_pct_badge(pts100_pct))
            + '</tr>'
        )

    st.markdown(header + body + '</tbody></table>', unsafe_allow_html=True)

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

# ── SECTION 6: Assisted ───────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Assisted Scoring</div>',
    unsafe_allow_html=True,
)
st.caption("Assist rate = % of made FGs credited to an assisting player")

if not ast_df.empty:
    zones_labels = [ZONE_LABELS[z] for z in ZONE_ORDER]
    zcolors = [ZONE_COLORS.get(ZONE_LABELS[z], "#6c7a9c") for z in ZONE_ORDER]
    a_left, a_right = st.columns(2)

    with a_left:
        # FG% per zone
        fig_ast_fg = go.Figure()
        fig_ast_fg.add_trace(
            go.Bar(
                x=zones_labels,
                y=ast_df["fg_pct"].tolist(),
                marker_color=zcolors,
                showlegend=False,
                text=[pct_fmt(v) for v in ast_df["fg_pct"]],
                textposition="outside",
            )
        )
        fig_ast_fg.update_layout(
            template=PLOTLY_TEMPLATE,
            title="FG% by Zone",
            margin=dict(l=10, r=10, t=35, b=60),
            height=280,
            yaxis=dict(range=[0, 80], showgrid=True, gridcolor="#2d3150"),
            xaxis=dict(tickangle=-20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ast_fg, use_container_width=True)

    with a_right:
        # % of makes that were assisted per zone
        fig_ast_rate = go.Figure()
        fig_ast_rate.add_trace(
            go.Bar(
                x=zones_labels,
                y=ast_df["ast_on_makes"].tolist(),
                marker_color=zcolors,
                showlegend=False,
                text=[pct_fmt(v) for v in ast_df["ast_on_makes"]],
                textposition="outside",
            )
        )
        fig_ast_rate.update_layout(
            template=PLOTLY_TEMPLATE,
            title="% of Makes Assisted by Zone",
            margin=dict(l=10, r=10, t=35, b=60),
            height=280,
            yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#2d3150"),
            xaxis=dict(tickangle=-20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ast_rate, use_container_width=True)

# ── SECTION 7: 3-Point Context ────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">3-Point Context</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Shooter grades: Red = close hard (37%+ FG3, 3+ att/g), "
    "Yellow = respect (32–37%), Green = sag off (<32% or low volume)"
)

ctx = threept_ctx
if ctx["total_3pt_fga"] > 0:
    # ── top-line summary metrics ───────────────────────────────────────────
    tm1, tm2, tm3, tm4 = st.columns(4)
    with tm1:
        st.metric("3pt FGA", f"{ctx['total_3pt_fga']:,}")
    with tm2:
        st.metric("3pt FG%", pct_fmt(ctx["overall_fg_pct"]))
    with tm3:
        st.metric("Corner 3 Rate", pct_fmt(ctx["corner_pct"]))
    with tm4:
        st.metric("Ast on Made 3s", pct_fmt(ctx["overall_ast"]))

    st.write("")
    tp_left, tp_mid, tp_right = st.columns(3)

    # ── location: corner vs above-break ───────────────────────────────────
    with tp_left:
        loc_df = ctx["location"]
        fig_loc = go.Figure()
        loc_colors = ["#4e9af1", "#a06cf5"]
        for i, row in loc_df.iterrows():
            fig_loc.add_trace(go.Bar(
                x=[row["fga_pct"]],
                y=[row["zone"]],
                orientation="h",
                marker_color=loc_colors[i],
                showlegend=False,
                text=[f"{row['fga_pct']:.1f}%  FG%:{pct_fmt(row['fg_pct'])}"],
                textposition="outside",
                hovertemplate="%{y}<br>FGA%: %{x:.1f}%<br>FG%: " +
                              pct_fmt(row['fg_pct']) +
                              f"<br>Ast on makes: {pct_fmt(row['ast_on_makes'], 0)}<extra></extra>",
            ))
        fig_loc.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Location",
            margin=dict(l=10, r=10, t=35, b=10),
            height=200,
            xaxis=dict(range=[0, loc_df["fga_pct"].max() + 55], showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_loc, use_container_width=True)

    # ── shooter grade distribution ─────────────────────────────────────────
    with tp_mid:
        grade_df = ctx["grade"]
        fig_grade = go.Figure()
        for _, row in grade_df.iterrows():
            c = GRADE_COLORS[row["grade_key"]]
            fig_grade.add_trace(go.Bar(
                x=[row["fga_pct"]],
                y=[row["grade"]],
                orientation="h",
                marker_color=c,
                showlegend=False,
                text=[f"{row['fga_pct']:.1f}%  (FG%:{pct_fmt(row['fg_pct'])})"],
                textposition="outside",
            ))
        fig_grade.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Shooter Grade",
            margin=dict(l=10, r=10, t=35, b=10),
            height=200,
            xaxis=dict(range=[0, grade_df["fga_pct"].max() + 60], showgrid=False),
            yaxis=dict(
                categoryorder="array",
                categoryarray=[GRADE_LABELS[g] for g in reversed(GRADE_ORDER)],
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_grade, use_container_width=True)

    # ── clock timing ──────────────────────────────────────────────────────
    with tp_right:
        clk_df = ctx["clock"]
        clk_colors = ["#4ec9b0", "#4e9af1", "#e05c5c", "#e07c40"]
        fig_clk = go.Figure()
        for i, row in clk_df.iterrows():
            fig_clk.add_trace(go.Bar(
                x=[row["fga_pct"]],
                y=[row["clock"]],
                orientation="h",
                marker_color=clk_colors[i],
                showlegend=False,
                text=[f"{row['fga_pct']:.1f}%  (FG%:{pct_fmt(row['fg_pct'])})"],
                textposition="outside",
            ))
        fig_clk.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Shot Clock",
            margin=dict(l=10, r=10, t=35, b=10),
            height=200,
            xaxis=dict(range=[0, clk_df["fga_pct"].max() + 60], showgrid=False),
            yaxis=dict(
                categoryorder="array",
                categoryarray=list(reversed([r["clock"] for _, r in clk_df.iterrows()])),
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_clk, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: CFBD / ESPN  ·  Pipeline: daily_fetch.py  ·  2025–26 Season")
