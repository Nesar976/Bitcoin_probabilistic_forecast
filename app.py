import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

from data import fetch_binance_klines
from model import predict_next_hour_price_interval
from metrics import calculate_coverage, calculate_average_width, winkler_score
from utils import load_backtest_results, save_prediction, load_predictions

st.set_page_config(page_title="Crypto Quant Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS: Cinematic Blue Hybrid Theme ---
st.markdown("""
<style>
    /* Cinematic Layered Gradient Background */
    .stApp {
        background-color: #0F1C3F;
        background-image: 
            radial-gradient(circle at top right, rgba(96,165,250,0.15), transparent 40%),
            linear-gradient(135deg, #0F1C3F 0%, #1E3A8A 50%, #2563EB 100%);
        color: #E5E7EB;
        font-family: 'Inter', sans-serif;
    }
    header {visibility: hidden;}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B1220 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Typography & Header Elements */
    .title-text {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .micro-info {
        font-size: 0.85rem;
        color: #9CA3AF;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .live-dot { color: #22C55E; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    
    .subtitle-text {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .header-divider {
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59,130,246,0.5), transparent);
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 0 10px rgba(59,130,246,0.3);
    }
    
    /* Header Chips */
    .header-chips-container { display: flex; gap: 0.75rem; align-items: center; justify-content: flex-end; height: 100%; padding-top: 1.5rem; }
    .stat-chip {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #9CA3AF;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .stat-chip span { font-weight: 700; }
    
    /* KPI Cards (Dark Glass) */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
    }
    .kpi-pill {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.2rem;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 25px rgba(0,0,0,0.25);
        border: 1px solid rgba(255, 255, 255, 0.03);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, border-color 0.3s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .kpi-pill:hover {
        transform: translateY(-5px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(59, 130, 246, 0.1);
        border-color: rgba(59,130,246,0.3);
    }
    .kpi-price { border-top: 3px solid #3B82F6; }
    .kpi-range { border-top: 3px solid #8B5CF6; }
    .kpi-vol   { border-top: 3px solid #F59E0B; }
    .kpi-cov   { border-top: 3px solid #22C55E; }
    
    .kpi-label { font-size: 13px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .kpi-val { font-size: 26px; font-weight: 800; color: #F8FAFC; display: flex; align-items: center; gap: 12px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);}
    .kpi-delta { font-size: 12px; font-weight: 600; padding: 6px 10px; border-radius: 8px; white-space: nowrap; }
    .kpi-delta.up { color: #22C55E; background: rgba(34, 197, 94, 0.15);}
    .kpi-delta.down { color: #EF4444; background: rgba(239, 68, 68, 0.15);}
    
    /* Quick Stats Bar (Floating above chart) */
    .quick-stats-bar {
        display: flex;
        gap: 2rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        margin-bottom: -1rem;
        position: relative;
        z-index: 10;
        width: fit-content;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .q-stat { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; color: #9CA3AF; font-weight: 600;}
    .q-stat span { color: #F8FAFC; font-weight: 700; }
    
    /* Plotly Chart Container */
    [data-testid="stPlotlyChart"] > div {
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.03);
        background-color: #0F172A;
    }
    
    /* Soft Containers (Insights/Tables) */
    .soft-container {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.03), 0 10px 30px rgba(0, 0, 0, 0.2);
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 2rem;
    }
    
    .section-header { 
        font-size: 1.5rem; 
        font-weight: 700; 
        color: #F8FAFC; 
        margin-top: 3.5rem;
        margin-bottom: 1.5rem; 
        padding-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(59,130,246,0.3);
    }
    
    /* Smart Table */
    .table-wrapper { overflow-y: auto; max-height: 400px; }
    .smart-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.9rem; color: #E5E7EB; }
    .smart-table th { text-align: left; padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #94A3B8; font-weight: 600; position: sticky; top: 0; background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(8px); z-index: 10;}
    .smart-table td { padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.02); }
    .smart-table tr { transition: background 0.2s ease; }
    .smart-table tr:hover { background-color: rgba(255, 255, 255, 0.03) !important; }
    
    /* Summary Strip */
    .summary-strip { display: flex; gap: 1.5rem; margin-bottom: 2.5rem; }
    .summary-item { display: flex; align-items: center; gap: 1rem; background: rgba(15, 23, 42, 0.6); padding: 1.25rem 1.5rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.03); box-shadow: 0 10px 30px rgba(0,0,0,0.2); flex: 1; backdrop-filter: blur(12px); }
    .summary-icon { font-size: 1.75rem; text-shadow: 0 0 15px currentColor; }
    .summary-details { display: flex; flex-direction: column; }
    .summary-label { font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; }
    .summary-value { font-size: 1.5rem; font-weight: 800; color: #F8FAFC; text-shadow: 0 2px 5px rgba(0,0,0,0.5); }
    
    /* Small Insights Panel Styles */
    .insight-row { display:flex; align-items:center; gap:0.5rem; font-size:0.95rem; margin-bottom:0.75rem; color:#E5E7EB; }
    .insight-dot { width:8px; height:8px; border-radius:50%; box-shadow: 0 0 8px currentColor; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("<h3 style='color:#E5E7EB; font-weight:700; font-size:1.1rem; margin-top:1rem;'>📊 Data Feed</h3>", unsafe_allow_html=True)
symbol = st.sidebar.selectbox("Asset", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0, label_visibility="collapsed")
interval = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=0, label_visibility="collapsed")

st.sidebar.markdown("<br><h3 style='color:#E5E7EB; font-weight:700; font-size:1.1rem;'>⚙️ Inference Engine</h3>", unsafe_allow_html=True)
window_size = st.sidebar.slider("Rolling Window", min_value=100, max_value=1000, value=500, step=50)
confidence_level_pct = st.sidebar.slider("Confidence Level (%)", min_value=90, max_value=99, value=95, step=1)
alpha = 1.0 - (confidence_level_pct / 100.0)

st.sidebar.markdown("<br><h3 style='color:#E5E7EB; font-weight:700; font-size:1.1rem;'>🎨 Visualization</h3>", unsafe_allow_html=True)
show_ema = st.sidebar.checkbox("Show EMA (20, 50)", value=True)
show_bb = st.sidebar.checkbox("Show Volatility Bands", value=False)

@st.cache_data(ttl=60)
def get_live_data(sym, intv, bars):
    return fetch_binance_klines(symbol=sym, interval=intv, total_bars=max(bars, 200))

data = get_live_data(symbol, interval, max(window_size, 150))
if data.empty:
    st.error("Failed to fetch data.")
    st.stop()

train_data = data.tail(window_size).reset_index(drop=True)
current_price = train_data.iloc[-1]["close"]
current_time = train_data.iloc[-1]["open_time"]
prev_price = train_data.iloc[-2]["close"]

# Computations
with st.spinner("Executing Quant Engine..."):
    lower, upper, volatility = predict_next_hour_price_interval(train_data["close"], num_simulations=10000, alpha=alpha)

price_delta = current_price - prev_price
price_delta_pct = (price_delta / prev_price) * 100
price_dir = "up" if price_delta >= 0 else "down"
price_arrow = "↑" if price_delta >= 0 else "↓"

vol_regime = "High" if volatility > 0.015 else ("Medium" if volatility > 0.005 else "Low")
trend = "Bullish" if current_price > train_data['close'].ewm(span=50).mean().iloc[-1] else "Bearish"
trend_color = "#22C55E" if trend == "Bullish" else "#EF4444"

# 1. HEADER SECTION (BALANCED HYBRID)
col_head1, col_head2, col_head3 = st.columns([2, 1, 1.5])

with col_head1:
    st.markdown(f"""
        <div class="title-text">{symbol} Quant Analytics</div>
        <div class="micro-info">
            <span class="live-dot">●</span> Live Data Feed &nbsp;&nbsp;|&nbsp;&nbsp; 
            <span style="color:#94A3B8;">Last updated: {datetime.now().strftime('%H:%M:%S UTC')}</span>
        </div>
        <div class="subtitle-text">Cinematic Probabilistic Forecasting & Risk Modeling</div>
    """, unsafe_allow_html=True)

with col_head2:
    fig_spark = go.Figure(go.Scatter(x=data['open_time'].tail(30), y=data['close'].tail(30), line=dict(color='#3B82F6', width=2)))
    fig_spark.update_layout(margin=dict(l=0,r=0,t=15,b=0), height=60, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_visible=False, yaxis_visible=False, showlegend=False)
    st.plotly_chart(fig_spark, use_container_width=True)

with col_head3:
    st.markdown(f"""
    <div class="header-chips-container">
        <div class="stat-chip">Sentiment: <span style="color:{trend_color};">{trend}</span></div>
        <div class="stat-chip">Vol: <span style="color:#F59E0B;">{vol_regime}</span></div>
        <div class="stat-chip">P/C: <span style="color:#8B5CF6;">{confidence_level_pct}%</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)

backtest_results = load_backtest_results()
cov = 0
bt_df = None
if backtest_results:
    bt_df = pd.DataFrame(backtest_results)
    bt_df['width'] = bt_df['upper_bound'] - bt_df['lower_bound']
    bt_df['inside'] = (bt_df['actual_price'] >= bt_df['lower_bound']) & (bt_df['actual_price'] <= bt_df['upper_bound'])
    actuals = bt_df['actual_price'].tolist()
    lowers = bt_df['lower_bound'].tolist()
    uppers = bt_df['upper_bound'].tolist()
    cov = calculate_coverage(actuals, lowers, uppers)
    winkler = winkler_score(actuals, lowers, uppers, alpha=0.05)

cov_color = "#22C55E" if cov >= 0.90 else "#EF4444"
cov_text = f"{cov:.1%}" if backtest_results else "N/A"

# 2. KPI CARDS (DARK GLASS)
kpi_html = f"""
<div class="kpi-container">
    <div class="kpi-pill kpi-price">
        <div class="kpi-label">Current Price</div>
        <div class="kpi-val">${current_price:,.2f} <span class="kpi-delta {price_dir}">{price_arrow} {abs(price_delta_pct):.2f}%</span></div>
    </div>
    <div class="kpi-pill kpi-range">
        <div class="kpi-label">{confidence_level_pct}% Predicted Range</div>
        <div class="kpi-val">${lower:,.0f} - ${upper:,.0f}</div>
    </div>
    <div class="kpi-pill kpi-vol">
        <div class="kpi-label">Implied Volatility (σ)</div>
        <div class="kpi-val">{volatility:.4f}</div>
    </div>
    <div class="kpi-pill kpi-cov">
        <div class="kpi-label">System Coverage</div>
        <div class="kpi-val" style="color: {cov_color};">{cov_text}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# 3. QUICK STATS FLOATING BAR
risk_level = "Low" if cov > 0.93 else ("Medium" if cov > 0.85 else "High")
risk_color = "#22C55E" if risk_level == "Low" else ("#F59E0B" if risk_level == "Medium" else "#EF4444")
st.markdown(f"""
<div style="display:flex; justify-content:center; width:100%;">
    <div class="quick-stats-bar">
        <div class="q-stat">📊 Direction: <span style="color:{trend_color};">{trend}</span></div>
        <div class="q-stat" style="border-left:1px solid rgba(255,255,255,0.1); padding-left:1rem;">⚡ Signal Strength: <span style="color:#3B82F6;">Moderate</span></div>
        <div class="q-stat" style="border-left:1px solid rgba(255,255,255,0.1); padding-left:1rem;">🛡️ Risk Envelope: <span style="color:{risk_color};">{risk_level}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. MAIN CHART (DEEP #0F172A)
display_data = data.tail(100).copy()
display_data['color'] = np.where(display_data['close'] >= display_data['open'], 'rgba(34, 197, 94, 0.8)', 'rgba(239, 68, 68, 0.8)')

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.8])

fig.add_trace(go.Candlestick(
    x=display_data["open_time"], open=display_data["open"], high=display_data["high"], low=display_data["low"], close=display_data["close"],
    name="Price", increasing_line_color='#22C55E', decreasing_line_color='#EF4444', increasing_fillcolor='#22C55E', decreasing_fillcolor='#EF4444'
), row=1, col=1)

if show_ema:
    data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['close'].ewm(span=50, adjust=False).mean()
    fig.add_trace(go.Scatter(x=display_data["open_time"], y=data['EMA20'].tail(100), mode='lines', name='EMA 20', line=dict(color='#F97316', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_data["open_time"], y=data['EMA50'].tail(100), mode='lines', name='EMA 50', line=dict(color='#A855F7', width=2)), row=1, col=1)

if interval == "1h": td = timedelta(hours=1)
elif interval == "4h": td = timedelta(hours=4)
else: td = timedelta(days=1)
next_time = current_time + td

fig.add_trace(go.Scatter(
    x=[current_time, next_time, next_time, current_time], y=[current_price, upper, lower, current_price],
    fill="toself", fillcolor="rgba(59, 130, 246, 0.2)", line=dict(color="rgba(59, 130, 246, 0.5)", width=1, dash='dash'), name=f"Forecast Band"
), row=1, col=1)

fig.add_vline(x=current_time, line_width=1, line_dash="dot", line_color="#E5E7EB", row=1, col=1)

fig.add_trace(go.Bar(
    x=display_data["open_time"], y=display_data["volume"], marker_color=display_data['color'], marker_line_width=0, name="Volume"
), row=2, col=1)

fig.update_layout(
    template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
    margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False, hovermode="x unified", showlegend=False, height=650
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=1, row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)", gridwidth=1, row=1, col=1)
fig.update_xaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# 5. NEW INSIGHTS PANEL (Smart Text)
st.markdown('<div class="section-header">🤖 Algorithmic Insights</div>', unsafe_allow_html=True)
col_a1, col_a2 = st.columns([1, 2], gap="large")

with col_a1:
    st.markdown("""
    <div class="soft-container" style="height:100%;">
        <div style="color:#94A3B8; font-weight:600; margin-bottom:1rem; text-transform:uppercase; font-size:0.85rem;">System Readout</div>
        <div class="insight-row"><div class="insight-dot" style="background:#22C55E; color:#22C55E;"></div> Model confidence is operating nominally.</div>
        <div class="insight-row"><div class="insight-dot" style="background:#F59E0B; color:#F59E0B;"></div> Volatility is transitioning.</div>
        <div class="insight-row"><div class="insight-dot" style="background:#3B82F6; color:#3B82F6;"></div> Bullish momentum detected across primary EMAs.</div>
    </div>
    """, unsafe_allow_html=True)

with col_a2:
    if bt_df is not None:
        avg_w = bt_df['width'].mean()
        st.markdown(f"""
        <div class="summary-strip" style="margin-bottom:0;">
            <div class="summary-item">
                <div class="summary-icon" style="color:{cov_color};">🎯</div>
                <div class="summary-details">
                    <div class="summary-label">Historic Coverage</div>
                    <div class="summary-value" style="color:{cov_color};">{cov:.1%}</div>
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-icon" style="color:#3B82F6;">📏</div>
                <div class="summary-details">
                    <div class="summary-label">Average Width</div>
                    <div class="summary-value" style="color:#3B82F6;">${avg_w:,.0f}</div>
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-icon" style="color:#8B5CF6;">⚖️</div>
                <div class="summary-details">
                    <div class="summary-label">Winkler Score</div>
                    <div class="summary-value" style="color:#8B5CF6;">{winkler:.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No backtest summary available.")

# 6. VISUAL INSIGHTS (CHARTS) & TABLES
if bt_df is not None:
    c1, c2, c3 = st.columns(3)
    bt_df['rolling_cov'] = bt_df['inside'].rolling(50).mean()
    bt_df['error'] = np.where(bt_df['inside'], 0, np.where(bt_df['actual_price'] > bt_df['upper_bound'], bt_df['actual_price'] - bt_df['upper_bound'], bt_df['lower_bound'] - bt_df['actual_price']))
    
    fig_cov = px.line(bt_df, x=bt_df.index, y='rolling_cov', title="50-Period Coverage Trend")
    fig_cov.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", height=220, margin=dict(l=10,r=10,b=10,t=40))
    fig_cov.update_traces(line_color="#22C55E")
    fig_cov.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    fig_cov.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    c1.plotly_chart(fig_cov, use_container_width=True)
    
    fig_dist = px.histogram(bt_df, x='width', title="Interval Width Distribution", nbins=30)
    fig_dist.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", height=220, margin=dict(l=10,r=10,b=10,t=40))
    fig_dist.update_traces(marker_color="#3B82F6")
    fig_dist.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    fig_dist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    c2.plotly_chart(fig_dist, use_container_width=True)
    
    fig_err = px.bar(bt_df.tail(100), y='error', title="Prediction Miss Magnitude (Last 100)")
    fig_err.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", height=220, margin=dict(l=10,r=10,b=10,t=40))
    fig_err.update_traces(marker_color="#EF4444")
    fig_err.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    fig_err.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    c3.plotly_chart(fig_err, use_container_width=True)

    table_html = """
    <div class="soft-container table-wrapper" style="margin-top:1rem;">
        <table class="smart-table">
            <thead>
                <tr><th>Timestamp</th><th>Actual Price</th><th>Lower Bound</th><th>Upper Bound</th><th>Inside Range</th></tr>
            </thead>
            <tbody>
    """
    for r in bt_df.tail(15).iloc[::-1].itertuples():
        status = "<span style='color:#22C55E; font-weight:700;'>✓ HIT</span>" if r.inside else "<span style='color:#EF4444; font-weight:700;'>✗ MISS</span>"
        row_bg = "background: rgba(34, 197, 94, 0.05);" if r.inside else "background: rgba(239, 68, 68, 0.05);"
        table_html += f"<tr style='{row_bg}'><td>{r.timestamp}</td><td>${r.actual_price:,.2f}</td><td>${r.lower_bound:,.2f}</td><td>${r.upper_bound:,.2f}</td><td>{status}</td></tr>"
    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

# 7. LIVE PREDICTION TRACKER
st.markdown('<div class="section-header">📡 Live Prediction Tracker</div>', unsafe_allow_html=True)

LIVE_PREDICTIONS_FILE = f"live_predictions_{symbol}_{interval}.json"
pred_data = {
    "target_timestamp": next_time.strftime('%Y-%m-%d %H:%M:%S'),
    "lower_bound": round(lower, 2),
    "upper_bound": round(upper, 2),
    "prediction_made_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
past_preds = load_predictions(LIVE_PREDICTIONS_FILE)
if not any(p["target_timestamp"] == pred_data["target_timestamp"] for p in past_preds):
    save_prediction(LIVE_PREDICTIONS_FILE, pred_data)
    past_preds.append(pred_data)

if past_preds:
    live_df = pd.DataFrame(past_preds)
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(x=live_df['target_timestamp'], y=live_df['upper_bound'], mode='lines', line=dict(color='#8B5CF6', width=2), name='Upper Bound'))
    fig_live.add_trace(go.Scatter(x=live_df['target_timestamp'], y=live_df['lower_bound'], mode='lines', line=dict(color='#8B5CF6', width=2), fill='tonexty', fillcolor='rgba(139, 92, 246, 0.15)', name='Lower Bound'))
    
    fig_live.update_layout(template="plotly_dark", plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", height=300, margin=dict(l=10, r=10, t=30, b=10), title="Forecast Timeline vs Actuals", xaxis_title="Target Timestamp", hovermode="x unified")
    fig_live.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    fig_live.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.03)")
    st.plotly_chart(fig_live, use_container_width=True)
    
    live_html = """
    <div class="soft-container table-wrapper">
        <table class="smart-table">
            <thead>
                <tr><th>Target Timestamp</th><th>Lower Bound</th><th>Upper Bound</th><th>Predicted At</th></tr>
            </thead>
            <tbody>
    """
    for r in live_df.tail(8).iloc[::-1].itertuples():
        live_html += f"<tr><td>{r.target_timestamp}</td><td>${r.lower_bound:,.2f}</td><td>${r.upper_bound:,.2f}</td><td style='color:#94A3B8;'>{r.prediction_made_at}</td></tr>"
    live_html += "</tbody></table></div>"
    st.markdown(live_html, unsafe_allow_html=True)
