import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="Crypto Trader Sentinel",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0a0d14;
    --surface:  #111520;
    --border:   #1e2535;
    --accent:   #00e5ff;
    --accent2:  #ff3c6e;
    --accent3:  #a259ff;
    --text:     #e2e8f4;
    --muted:    #5a6480;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    letter-spacing: 0.05em;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
}
.kpi-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }
.pos { color: #00e5a0; }
.neg { color: var(--accent2); }

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Table tweaks */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    overflow: hidden;
}

/* Selectbox / slider labels */
label { color: var(--muted) !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    overall_trader_perf = pd.read_csv("data/overall_trader_perf.csv")
    sentiment_pnl       = pd.read_csv("data/sentiment_pnl.csv")
    final_df            = pd.read_csv("data/final_df.csv")
    final_df["date"]    = pd.to_datetime(final_df["date"])
    return overall_trader_perf, sentiment_pnl, final_df

overall_trader_perf, sentiment_pnl, final_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ Sentinel")
    st.markdown("---")

    st.markdown("## Filters")

    # Date range
    min_date = final_df["date"].min().date()
    max_date = final_df["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Sentiment filter
    if "sentiment_score" in final_df.columns:
        sent_vals = sorted(final_df["sentiment_score"].dropna().unique())
        selected_sentiments = st.multiselect(
            "Sentiment Score",
            options=sent_vals,
            default=sent_vals,
        )
    else:
        selected_sentiments = None

    # Score threshold for leaderboard
    if "score" in overall_trader_perf.columns:
        min_score, max_score = float(overall_trader_perf["score"].min()), float(overall_trader_perf["score"].max())
        score_thresh = st.slider(
            "Min Trader Score",
            min_value=min_score,
            max_value=max_score,
            value=min_score,
            step=(max_score - min_score) / 100,
        )
    else:
        score_thresh = None

    st.markdown("---")
    st.markdown("## Trader Explorer")
    selected_trader = st.selectbox(
        "Select Trader",
        options=sorted(final_df["account"].unique()),
    )

    chart_type = st.radio(
        "PnL Chart Style",
        ["Line", "Area", "Candlestick-like (OHLC)"],
        index=1,
    )

# ── Apply global date filter ──────────────────────────────────────────────────
if len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1])
    filtered_df = final_df[(final_df["date"] >= start_dt) & (final_df["date"] <= end_dt)]
else:
    filtered_df = final_df.copy()

if selected_sentiments is not None and "sentiment_score" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["sentiment_score"].isin(selected_sentiments)]

# ── KPI row ───────────────────────────────────────────────────────────────────
total_pnl   = filtered_df["closed_pnl"].sum() if "closed_pnl" in filtered_df.columns else 0
total_trades = len(filtered_df)
winners     = (filtered_df["closed_pnl"] > 0).sum() if "closed_pnl" in filtered_df.columns else 0
win_rate    = winners / total_trades * 100 if total_trades > 0 else 0
n_traders   = filtered_df["account"].nunique()

pnl_class = "pos" if total_pnl >= 0 else "neg"
pnl_sign  = "+" if total_pnl >= 0 else ""

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Total PnL</div>
    <div class="kpi-value {pnl_class}">{pnl_sign}{total_pnl:,.0f}</div>
    <div class="kpi-sub">Across filtered range</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value {'pos' if win_rate >= 50 else 'neg'}">{win_rate:.1f}%</div>
    <div class="kpi-sub">{winners:,} winners / {total_trades:,} trades</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Unique Traders</div>
    <div class="kpi-value" style="color:var(--accent);">{n_traders}</div>
    <div class="kpi-sub">In filtered dataset</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg PnL / Trade</div>
    <div class="kpi-value {'pos' if total_pnl/max(total_trades,1) >= 0 else 'neg'}">{total_pnl/max(total_trades,1):,.1f}</div>
    <div class="kpi-sub">Per closed position</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.6, 1], gap="large")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#5a6480", size=11),
    xaxis=dict(gridcolor="#1e2535", zeroline=False),
    yaxis=dict(gridcolor="#1e2535", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
)

with col_left:
    # ── Trader PnL chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Trader PnL Over Time</div>', unsafe_allow_html=True)

    trader_data = filtered_df[filtered_df["account"] == selected_trader].sort_values("date")

    if trader_data.empty:
        st.info("No data for this trader in the selected range.")
    else:
        cumulative = trader_data["closed_pnl"].cumsum()

        if chart_type == "Line":
            fig = go.Figure(go.Scatter(
                x=trader_data["date"], y=trader_data["closed_pnl"],
                mode="lines",
                line=dict(color="#00e5ff", width=1.5),
                fill=None,
                name="Daily PnL",
            ))
        elif chart_type == "Area":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trader_data["date"], y=cumulative,
                mode="lines",
                line=dict(color="#00e5ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,229,255,0.08)",
                name="Cumulative PnL",
            ))
            fig.add_trace(go.Bar(
                x=trader_data["date"],
                y=trader_data["closed_pnl"],
                marker_color=np.where(trader_data["closed_pnl"] >= 0, "#00e5a0", "#ff3c6e"),
                name="Daily PnL",
                opacity=0.6,
                yaxis="y2",
            ))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", gridcolor="#1e2535", showgrid=False))
        else:  # OHLC-like using daily resampled
            ohlc = trader_data.set_index("date")["closed_pnl"].resample("W").agg(
                Open="first", High="max", Low="min", Close="last"
            ).dropna().reset_index()
            fig = go.Figure(go.Candlestick(
                x=ohlc["date"],
                open=ohlc["Open"], high=ohlc["High"],
                low=ohlc["Low"],   close=ohlc["Close"],
                increasing_line_color="#00e5a0",
                decreasing_line_color="#ff3c6e",
                name="Weekly PnL",
            ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            legend=dict(orientation="h", y=1.05, x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment vs PnL ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Sentiment Score → Avg PnL</div>', unsafe_allow_html=True)

    sent_col = "sentiment_score"
    if sent_col in sentiment_pnl.columns and "closed_pnl" in sentiment_pnl.columns:
        sent_plot = sentiment_pnl.copy()
    elif sent_col in filtered_df.columns and "closed_pnl" in filtered_df.columns:
        sent_plot = filtered_df.groupby(sent_col)["closed_pnl"].mean().reset_index()
        sent_plot.columns = [sent_col, "closed_pnl"]
    else:
        sent_plot = sentiment_pnl.copy()

    sent_plot = sent_plot.sort_values(sent_col)
    colors    = ["#00e5a0" if v >= 0 else "#ff3c6e" for v in sent_plot.iloc[:, 1]]

    fig2 = go.Figure(go.Bar(
        x=sent_plot.iloc[:, 0].astype(str),
        y=sent_plot.iloc[:, 1],
        marker_color=colors,
        text=sent_plot.iloc[:, 1].round(2),
        textposition="outside",
        textfont=dict(size=10, color="#5a6480"),
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, height=280)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Trader Leaderboard</div>', unsafe_allow_html=True)

    leaderboard = overall_trader_perf.copy()
    if score_thresh is not None and "score" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["score"] >= score_thresh]
    leaderboard = leaderboard.sort_values("score", ascending=False).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1  # 1-indexed rank

    # Colour-code score column if it exists
    def style_score(val):
        p = (val - leaderboard["score"].min()) / max(leaderboard["score"].max() - leaderboard["score"].min(), 1)
        r = int(255 * (1 - p)); g = int(200 * p + 55)
        return f"color: rgb({r},{g},100); font-weight:600"

    styled = leaderboard.style.applymap(style_score, subset=["score"]) if "score" in leaderboard.columns else leaderboard.style
    st.dataframe(styled, use_container_width=True, height=280)

    # ── Distribution of trader PnL ────────────────────────────────────────────
    st.markdown('<div class="section-label">PnL Distribution</div>', unsafe_allow_html=True)

    trader_totals = filtered_df.groupby("account")["closed_pnl"].sum()
    fig3 = go.Figure(go.Histogram(
        x=trader_totals,
        nbinsx=30,
        marker=dict(
            color=trader_totals,
            colorscale=[[0, "#ff3c6e"], [0.5, "#a259ff"], [1, "#00e5ff"]],
            showscale=False,
        ),
    ))
    fig3.add_vline(
        x=trader_totals.mean(),
        line_dash="dot",
        line_color="#00e5a0",
        annotation_text="mean",
        annotation_font_size=10,
    )
    fig3.update_layout(**PLOTLY_LAYOUT, height=240)
    st.plotly_chart(fig3, use_container_width=True)

# ── Bottom row: scatter + raw table ───────────────────────────────────────────
st.markdown('<div class="section-label">Trader Score vs Total PnL</div>', unsafe_allow_html=True)

if "score" in overall_trader_perf.columns:
    pnl_by_trader = filtered_df.groupby("account")["closed_pnl"].sum().reset_index()
    pnl_by_trader.columns = ["account", "total_pnl"]
    scatter_df = overall_trader_perf.merge(pnl_by_trader, on="account", how="inner")

    fig4 = px.scatter(
        scatter_df,
        x="score",
        y="total_pnl",
        hover_name="account",
        color="total_pnl",
        color_continuous_scale=["#ff3c6e", "#a259ff", "#00e5ff"],
        size=abs(scatter_df["total_pnl"]).clip(lower=1),
        size_max=28,
    )
    fig4.update_traces(marker=dict(line=dict(width=0.5, color="#0a0d14")))
    fig4.update_coloraxes(showscale=False)
    fig4.update_layout(**PLOTLY_LAYOUT, height=320, hovermode="closest")
    st.plotly_chart(fig4, use_container_width=True)

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("🗂  Raw Data — Filtered Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("📊  Raw Data — Sentiment PnL"):
    st.dataframe(sentiment_pnl, use_container_width=True)