"""
MARKET PULSE - Live NSE Market Screener (Streamlit)

Chalane ka tarika: run.bat par double-click karo.
(Manual: pip install streamlit yfinance pandas plotly tzdata  ->  streamlit run app.py)

Note: Yahoo Finance ka data ~15 min delayed ho sakta hai. Ye sirf jaankari ke liye hai,
investment advice nahi.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Market Pulse", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

IST = ZoneInfo("Asia/Kolkata")
BENCHMARK = "^NSEI"
INDICES = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN"}

NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "ULTRACEMCO", "NESTLEIND", "WIPRO", "HCLTECH", "ONGC", "NTPC",
    "POWERGRID", "M&M", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "TECHM", "INDUSINDBK", "HINDALCO", "DRREDDY", "CIPLA", "EICHERMOT",
    "GRASIM", "BRITANNIA", "APOLLOHOSP", "DIVISLAB", "HEROMOTOCO", "BPCL", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO", "SHRIRAMFIN",
]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.1rem; max-width: 1500px;}
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -10%, #1d2556 0%, transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(124,58,237,.22) 0%, transparent 55%),
    #0b1020;
}
section[data-testid="stSidebar"] {background: #0d1330; border-right: 1px solid rgba(255,255,255,.06);}
.hero {
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;
  padding:18px 22px; border-radius:18px; margin-bottom:16px;
  background: linear-gradient(135deg, rgba(99,102,241,.28), rgba(14,165,233,.12));
  border:1px solid rgba(255,255,255,.09);
}
.hero h1 {margin:0; padding:0; font-size:1.75rem; font-weight:800; letter-spacing:-.02em;}
.hero small {color:#9ca3af;}
.badge {
  display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px;
  font-size:.8rem; font-weight:600; border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.05);
}
.dot {width:9px; height:9px; border-radius:50%; display:inline-block;}
.dot.live {background:#22c55e; animation:pulse 1.6s infinite;}
.dot.closed {background:#f59e0b;}
@keyframes pulse {
  0% {box-shadow:0 0 0 0 rgba(34,197,94,.7);}
  70% {box-shadow:0 0 0 9px rgba(34,197,94,0);}
  100% {box-shadow:0 0 0 0 rgba(34,197,94,0);}
}
.card {
  padding:14px 16px; border-radius:16px; height:100%;
  background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09);
  backdrop-filter: blur(6px);
}
.card .lbl {color:#9ca3af; font-size:.7rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700;}
.card .val {font-size:1.5rem; font-weight:800; margin-top:3px; line-height:1.15;}
.card .sub {font-size:.85rem; font-weight:600; margin-top:3px;}
.up {color:#22c55e;} .down {color:#ef4444;} .muted {color:#9ca3af;}
.breadth {height:9px; border-radius:999px; overflow:hidden; display:flex; background:#1f2937; margin-top:9px;}
.breadth span {display:block; height:100%;}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {border-radius:10px; padding:8px 16px; background: rgba(255,255,255,.04);}
.stTabs [aria-selected="true"] {background: rgba(99,102,241,.38) !important;}
.stTabs [data-baseweb="tab-highlight"] {display:none;}
div[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,.09);}
.note {color:#6b7280; font-size:.75rem; margin-top:18px; text-align:center;}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def card(label, value, sub="", cls="muted", extra=""):
    return (f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div>'
            f'<div class="sub {cls}">{sub}</div>{extra}</div>')


def stretch(fn, *args, **kwargs):
    """Streamlit ke naye/purane dono versions me full-width."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def style_fig(fig, height):
    fig.update_layout(template="plotly_dark", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=30, b=0), font=dict(family="Inter, sans-serif"))
    return fig


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_daily(symbols: tuple) -> pd.DataFrame:
    tickers = [s + ".NS" for s in symbols] + list(INDICES.values())
    return yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_chart(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol + ".NS", period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def get(raw, ticker):
    try:
        df = raw[ticker].dropna(how="all")
        return df if not df.empty else None
    except KeyError:
        return None


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def period_return(close, start, end):
    idx = pd.to_datetime(close.index).tz_localize(None)
    mask = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(end) + pd.Timedelta(days=1))
    sub = close[mask]
    if len(sub) < 2:
        return np.nan
    return (sub.iloc[-1] / sub.iloc[0] - 1) * 100


def build_table(raw, symbols, start, end):
    bench = get(raw, BENCHMARK)
    bench_ret = period_return(bench["Close"].dropna(), start, end) if bench is not None else np.nan
    rows = []
    for s in symbols:
        df = get(raw, s + ".NS")
        if df is None:
            continue
        close, vol = df["Close"].dropna(), df["Volume"]
        if len(close) < 2:
            continue
        last, prev = close.iloc[-1], close.iloc[-2]
        sma20, sma50 = close.rolling(20).mean().iloc[-1], close.rolling(50).mean().iloc[-1]
        avg_vol = vol.iloc[-21:-1].mean()
        pr = period_return(close, start, end)
        trend = ("Uptrend" if last > sma20 > sma50
                 else "Downtrend" if last < sma20 < sma50 else "Sideways")
        vs = "NA" if (np.isnan(pr) or np.isnan(bench_ret)) else ("Aage" if pr > bench_ret else "Peeche")
        rows.append({
            "Symbol": s, "LTP": last, "Day %": (last / prev - 1) * 100, "Period %": pr,
            "Vs Nifty": vs, "Trend": trend, "RSI": rsi(close).iloc[-1],
            "Vol x": vol.iloc[-1] / avg_vol if avg_vol else np.nan,
            "From 52W High %": (last / close.max() - 1) * 100,
            "SMA20": sma20, "SMA50": sma50, "Volume": vol.iloc[-1],
        })
    out = pd.DataFrame(rows)
    out.attrs["bench_ret"] = bench_ret
    return out


def tint_pn(v):
    if pd.isna(v):
        return ""
    if v > 0:
        return "background-color: rgba(34,197,94,.14); color:#4ade80; font-weight:700"
    if v < 0:
        return "background-color: rgba(239,68,68,.14); color:#f87171; font-weight:700"
    return "color:#9ca3af"


def tint_rsi(v):
    if pd.isna(v):
        return ""
    if v >= 70:
        return "color:#fbbf24; font-weight:700"
    if v <= 30:
        return "color:#38bdf8; font-weight:700"
    return ""


def tint_txt(v):
    s = str(v)
    if "Aage" in s or "Uptrend" in s:
        return "color:#4ade80; font-weight:600"
    if "Peeche" in s or "Downtrend" in s:
        return "color:#f87171; font-weight:600"
    if "Sideways" in s:
        return "color:#fbbf24; font-weight:600"
    return ""


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Filters")

with st.sidebar.expander("🧺 Stocks", expanded=False):
    universe = st.multiselect("Nifty 50 me se chuno", NIFTY50, default=NIFTY50)
    custom = st.text_input("Extra NSE symbols (comma se)", placeholder="IRCTC, ZOMATO")
symbols = list(dict.fromkeys(universe + [x.strip().upper() for x in custom.split(",") if x.strip()]))

today = dt.datetime.now(IST).date()
with st.sidebar.expander("📅 Date range", expanded=True):
    preset = st.select_slider("Period", ["1W", "1M", "3M", "6M", "1Y", "Custom"], value="1M")
    days = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    if preset == "Custom":
        picked = st.date_input("Start - End", (today - dt.timedelta(days=30), today), max_value=today)
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start_date, end_date = picked
        else:
            start_date, end_date = today - dt.timedelta(days=30), today
    else:
        start_date, end_date = today - dt.timedelta(days=days[preset]), today

with st.sidebar.expander("📊 Badha / Gira", expanded=True):
    basis = st.radio("Kis basis pe?", ["Aaj (Day)", "Selected period"], horizontal=True)
    direction = st.radio("Direction", ["Sab", "Badha 🟢", "Gira 🔴"], horizontal=True)
    min_move = st.slider("Kam se kam move (%)", 0.0, 10.0, 0.0, 0.25)

with st.sidebar.expander("🧭 Aage / Peeche", expanded=True):
    rel = st.radio("Nifty se", ["Sab", "Aage", "Peeche"], horizontal=True)
    trend_sel = st.multiselect("Trend", ["Uptrend", "Sideways", "Downtrend"],
                               default=["Uptrend", "Sideways", "Downtrend"])

with st.sidebar.expander("⚙️ Advanced", expanded=False):
    rsi_range = st.slider("RSI range", 0, 100, (0, 100))
    vol_x = st.slider("Volume spike (x avg) se zyada", 0.0, 5.0, 0.0, 0.1)
    near_high = st.slider("52W High se kitna door (max %)", 0, 100, 100)
    a, b = st.columns(2)
    price_min = a.number_input("Min price", 0.0, value=0.0)
    price_max = b.number_input("Max price", 0.0, value=100000.0)
    search = st.text_input("Symbol search")

with st.sidebar.expander("🔄 Refresh", expanded=False):
    auto = st.toggle("Auto refresh", value=False)
    every = st.select_slider("Har kitne second", [30, 60, 120, 300], value=60)
    if stretch(st.button, "Abhi refresh karo"):
        st.cache_data.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
def dashboard():
    now = dt.datetime.now(IST)
    is_open = now.weekday() < 5 and dt.time(9, 15) <= now.time() <= dt.time(15, 30)
    status = ('<span class="badge"><span class="dot live"></span>Market OPEN</span>' if is_open else
              '<span class="badge"><span class="dot closed"></span>Market CLOSED - last close dikh raha hai</span>')
    st.markdown(
        f'<div class="hero"><div><h1>📈 Market Pulse</h1>'
        f'<small>NSE live screener · {start_date:%d %b} → {end_date:%d %b %Y}</small></div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap">{status}'
        f'<span class="badge">🕒 {now:%I:%M:%S %p} IST</span></div></div>',
        unsafe_allow_html=True)

    if not symbols:
        st.info("Sidebar se kam se kam ek stock chuno.")
        return
    try:
        with st.spinner("Market data la raha hoon..."):
            raw = load_daily(tuple(symbols))
    except Exception as e:
        st.error(f"Data load nahi ho paya: {e}")
        return
    if raw is None or raw.empty:
        st.error("Data nahi mila. Internet ya symbols check karo.")
        return

    df = build_table(raw, symbols, start_date, end_date)
    if df.empty:
        st.error("Koi valid stock data nahi mila.")
        return
    bench_ret = df.attrs.get("bench_ret", np.nan)

    col = "Day %" if basis.startswith("Aaj") else "Period %"
    basis_lbl = "Aaj" if col == "Day %" else "Period"

    # ---- filters ----
    f = df.copy()
    if direction.startswith("Badha"):
        f = f[f[col] > 0]
    elif direction.startswith("Gira"):
        f = f[f[col] < 0]
    f = f[f[col].abs() >= min_move]
    f = f[f["Trend"].isin(trend_sel)]
    if rel != "Sab":
        f = f[f["Vs Nifty"] == rel]
    f = f[f["RSI"].between(*rsi_range)]
    f = f[f["Vol x"].fillna(0) >= vol_x]
    f = f[f["From 52W High %"] >= -near_high]
    f = f[f["LTP"].between(price_min, price_max)]
    if search:
        f = f[f["Symbol"].str.contains(search.strip().upper(), na=False)]

    # ---- index cards + breadth ----
    cols = st.columns(4)
    for c, (name, tk) in zip(cols[:3], INDICES.items()):
        idf = get(raw, tk)
        if idf is None or len(idf["Close"].dropna()) < 2:
            c.markdown(card(name, "NA"), unsafe_allow_html=True)
            continue
        cl = idf["Close"].dropna()
        chg = (cl.iloc[-1] / cl.iloc[-2] - 1) * 100
        arrow, cls = ("▲", "up") if chg >= 0 else ("▼", "down")
        c.markdown(card(name, f"{cl.iloc[-1]:,.2f}", f"{arrow} {chg:+.2f}% aaj", cls),
                   unsafe_allow_html=True)

    adv, dec = int((df[col] > 0).sum()), int((df[col] < 0).sum())
    unch = len(df) - adv - dec
    tot = max(len(df), 1)
    bar = (f'<div class="breadth"><span style="width:{adv / tot * 100:.1f}%;background:#22c55e"></span>'
           f'<span style="width:{unch / tot * 100:.1f}%;background:#6b7280"></span>'
           f'<span style="width:{dec / tot * 100:.1f}%;background:#ef4444"></span></div>')
    cols[3].markdown(card(f"Breadth ({basis_lbl})",
                          f'<span class="up">{adv}↑</span> / <span class="down">{dec}↓</span>',
                          f"{unch} unchanged", "muted", bar), unsafe_allow_html=True)

    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    avg_move = f[col].mean() if len(f) else np.nan
    k1.markdown(card("Filter match", f"{len(f)} / {len(df)}", "stocks", "muted"), unsafe_allow_html=True)
    k2.markdown(card(f"Average move ({basis_lbl})", "-" if np.isnan(avg_move) else f"{avg_move:+.2f}%",
                     "filtered stocks", "up" if avg_move > 0 else "down" if avg_move < 0 else "muted"),
                unsafe_allow_html=True)
    k3.markdown(card("Nifty (selected period)", "NA" if np.isnan(bench_ret) else f"{bench_ret:+.2f}%",
                     f"{int((df['Vs Nifty'] == 'Aage').sum())} stocks Nifty se aage",
                     "up" if bench_ret > 0 else "down" if bench_ret < 0 else "muted"), unsafe_allow_html=True)
    k4.markdown(card("Volume spike 2x+", int((f["Vol x"] >= 2).sum()), "filtered me", "muted"),
                unsafe_allow_html=True)
    st.write("")

    if f.empty:
        st.warning("Is filter me koi stock nahi mila. Sidebar me filters thode loose karo.")
        return

    tab_scr, tab_heat, tab_mov, tab_chart = st.tabs(
        ["📋 Screener", "🗺️ Heatmap", "🔥 Movers", "📈 Chart"])

    # ---------------- Screener ----------------
    with tab_scr:
        c1, c2 = st.columns([3, 1])
        sort_col = c1.selectbox("Sort by", ["Day %", "Period %", "Vol x", "RSI", "LTP",
                                            "From 52W High %", "Symbol"], key="sort_col")
        asc = c2.toggle("Ascending", value=False, key="sort_asc")
        s = f.sort_values(sort_col, ascending=asc).reset_index(drop=True)

        view = s[["Symbol", "LTP", "Day %", "Period %", "Vs Nifty", "Trend", "RSI", "Vol x",
                  "From 52W High %", "SMA20", "SMA50", "Volume"]].copy()
        view["Vs Nifty"] = view["Vs Nifty"].map({"Aage": "⬆ Aage", "Peeche": "⬇ Peeche"}).fillna("-")
        view["Trend"] = view["Trend"].map({"Uptrend": "🟢 Uptrend", "Downtrend": "🔴 Downtrend",
                                           "Sideways": "🟡 Sideways"})
        fmt = {"LTP": "₹{:,.2f}", "Day %": "{:+.2f}%", "Period %": "{:+.2f}%", "Vol x": "{:.2f}x",
               "RSI": "{:.0f}", "From 52W High %": "{:.1f}%", "SMA20": "{:,.2f}",
               "SMA50": "{:,.2f}", "Volume": "{:,.0f}"}
        styled = (view.style.format(fmt, na_rep="-")
                  .map(tint_pn, subset=["Day %", "Period %", "From 52W High %"])
                  .map(tint_rsi, subset=["RSI"])
                  .map(tint_txt, subset=["Vs Nifty", "Trend"]))
        stretch(st.dataframe, styled, height=min(700, 40 + 35 * len(view)), hide_index=True)
        st.caption("RSI: 🟡 70+ overbought, 🔵 30 se neeche oversold  ·  Trend: Close vs 20 & 50 DMA")
        st.download_button("⬇️ CSV download", s.to_csv(index=False).encode(),
                           file_name="market_pulse.csv", mime="text/csv")

    # ---------------- Heatmap ----------------
    with tab_heat:
        hm = f.dropna(subset=[col]).copy()
        hm["Turnover"] = (hm["LTP"] * hm["Volume"]).clip(lower=1)
        lim = max(1.0, float(np.nanpercentile(hm[col].abs(), 90)))
        fig = px.treemap(hm, path=[px.Constant("Market"), "Symbol"], values="Turnover", color=col,
                         color_continuous_scale=["#b91c1c", "#1f2937", "#15803d"],
                         color_continuous_midpoint=0, range_color=[-lim, lim])
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{color:+.2f}%", textfont_size=15,
                          marker_line_width=2, marker_line_color="#0b1020", root_color="rgba(0,0,0,0)",
                          hovertemplate="<b>%{label}</b><br>%{color:+.2f}%<extra></extra>")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 560))
        st.caption(f"Box ka size = turnover (price x volume), rang = {basis_lbl} % change")

    # ---------------- Movers ----------------
    with tab_mov:
        def movers(frame, color, title):
            fig = go.Figure(go.Bar(
                x=frame[col], y=frame["Symbol"], orientation="h", marker_color=color,
                text=[f"{v:+.2f}%" for v in frame[col]], textposition="outside", cliponaxis=False))
            fig.update_layout(title=title, yaxis=dict(autorange="reversed"), xaxis=dict(showgrid=False))
            return style_fig(fig, max(300, 36 * len(frame) + 60))

        g, l = st.columns(2)
        gainers = f[f[col] > 0].nlargest(10, col)
        losers = f[f[col] < 0].nsmallest(10, col)
        with g:
            if gainers.empty:
                st.info("Koi gainer nahi.")
            else:
                st.plotly_chart(movers(gainers, "#22c55e", f"🟢 Top gainers ({basis_lbl})"))
        with l:
            if losers.empty:
                st.info("Koi loser nahi.")
            else:
                st.plotly_chart(movers(losers, "#ef4444", f"🔴 Top losers ({basis_lbl})"))

    # ---------------- Chart ----------------
    with tab_chart:
        c1, c2 = st.columns([2, 2])
        sym = c1.selectbox("Stock", f["Symbol"].tolist(), key="chart_sym")
        tf = c2.radio("Timeframe", ["5m", "15m", "1h", "1d"], index=3, horizontal=True, key="chart_tf")
        interval, period = {"5m": ("5m", "5d"), "15m": ("15m", "1mo"),
                            "1h": ("1h", "3mo"), "1d": ("1d", "1y")}[tf]
        row = f[f["Symbol"] == sym].iloc[0]

        s1, s2, s3, s4, s5 = st.columns(5)
        cls = "up" if row["Day %"] >= 0 else "down"
        s1.markdown(card(sym, f"₹{row['LTP']:,.2f}", f"{row['Day %']:+.2f}% aaj", cls), unsafe_allow_html=True)
        s2.markdown(card("Period", f"{row['Period %']:+.2f}%" if pd.notna(row["Period %"]) else "NA",
                         f"Nifty se {row['Vs Nifty']}", "muted"), unsafe_allow_html=True)
        s3.markdown(card("RSI (14)", f"{row['RSI']:.0f}", row["Trend"], "muted"), unsafe_allow_html=True)
        s4.markdown(card("Volume", f"{row['Vol x']:.2f}x", "20-day avg ke mukable", "muted"), unsafe_allow_html=True)
        s5.markdown(card("52W High se", f"{row['From 52W High %']:.1f}%", "distance", "muted"), unsafe_allow_html=True)
        st.write("")

        cdf = load_chart(sym, interval, period)
        if cdf.empty:
            st.info("Is stock ka chart data nahi mila.")
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                                vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(
                x=cdf.index, open=cdf["Open"], high=cdf["High"], low=cdf["Low"], close=cdf["Close"],
                name=sym, increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444"), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf["Close"].rolling(20).mean(), name="SMA20",
                                     line=dict(width=1.4, color="#38bdf8")), row=1, col=1)
            fig.add_trace(go.Scatter(x=cdf.index, y=cdf["Close"].rolling(50).mean(), name="SMA50",
                                     line=dict(width=1.4, color="#f59e0b")), row=1, col=1)
            vcol = np.where(cdf["Close"] >= cdf["Open"], "#22c55e", "#ef4444")
            fig.add_trace(go.Bar(x=cdf.index, y=cdf["Volume"], name="Volume", marker_color=vcol,
                                 opacity=0.7), row=2, col=1)
            breaks = [dict(bounds=["sat", "mon"])]
            if interval != "1d":
                breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
            fig.update_xaxes(rangebreaks=breaks, showgrid=False)
            fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
            fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04))
            st.plotly_chart(style_fig(fig, 600))

    st.markdown('<div class="note">Data Yahoo Finance se, ~15 min delayed ho sakta hai. '
                'Sirf jaankari ke liye, investment advice nahi.</div>', unsafe_allow_html=True)


st.fragment(run_every=every if auto else None)(dashboard)()
