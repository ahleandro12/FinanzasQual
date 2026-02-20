# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FinDash", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .block-container { padding-top: 1rem; }
    h1,h2,h3 { color: #818cf8; }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 3px solid #6366f1;
    }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #818cf8; }
</style>
""", unsafe_allow_html=True)

# ── PORTFOLIO DATA ──────────────────────────────────────────────────────────────
# Mapeamos tus CEDEARs al ticker real de Yahoo Finance
PORTFOLIO = {
    "AMZN":  {"name": "Amazon",             "qty": 80,    "ppc": 1.66,   "yf_ticker": "AMZN",    "sector": "Tech"},
    "ASML":  {"name": "ASML Holding",       "qty": 17,    "ppc": 9.23,   "yf_ticker": "ASML",    "sector": "Tech"},
    "BRKB":  {"name": "Berkshire Hathaway", "qty": 2,     "ppc": 23.73,  "yf_ticker": "BRK-B",   "sector": "Financiero"},
    "EEM":   {"name": "iShares MSCI EM",    "qty": 4,     "ppc": 12.19,  "yf_ticker": "EEM",     "sector": "ETF"},
    "MA":    {"name": "Mastercard",         "qty": 5,     "ppc": 17.73,  "yf_ticker": "MA",      "sector": "Financiero"},
    "META":  {"name": "Meta Platforms",     "qty": 4,     "ppc": 28.50,  "yf_ticker": "META",    "sector": "Tech"},
    "MSFT":  {"name": "Microsoft",          "qty": 10,    "ppc": 14.30,  "yf_ticker": "MSFT",    "sector": "Tech"},
    "MU":    {"name": "Micron Technology",  "qty": 2,     "ppc": 76.69,  "yf_ticker": "MU",      "sector": "Tech"},
    "NFLX":  {"name": "Netflix",            "qty": 19,    "ppc": 1.82,   "yf_ticker": "NFLX",    "sector": "Tech"},
    "NU":    {"name": "Nu Holdings",        "qty": 5,     "ppc": 8.89,   "yf_ticker": "NU",      "sector": "Fintech"},
    "XOM":   {"name": "Exxon",              "qty": 5,     "ppc": 14.35,  "yf_ticker": "XOM",     "sector": "Energía"},
}

CEDEAR_RATIO = {
    "AMZN": 400, "ASML": 10, "BRK-B": 1, "EEM": 1,
    "MA": 10, "META": 10, "MSFT": 10, "MU": 1,
    "NFLX": 400, "NU": 1, "XOM": 1,
}

# ── HELPERS ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)  # cache 15 min
def get_price(ticker):
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="2d")
        if data.empty:
            return None, None
        price = data["Close"].iloc[-1]
        prev  = data["Close"].iloc[-2] if len(data) > 1 else price
        return price, prev
    except:
        return None, None

@st.cache_data(ttl=900)
def get_history(ticker, period="1y"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        info        = t.info
        financials  = t.financials        # anual income statement
        cashflow    = t.cashflow          # anual cash flow
        balance     = t.balance_sheet     # anual balance sheet
        return info, financials, cashflow, balance
    except:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def cedear_price(yf_price, yf_ticker):
    """Convierte precio ADR a precio CEDEAR aproximado."""
    ratio = CEDEAR_RATIO.get(yf_ticker, 1)
    return yf_price / ratio

def fmt_mill(v):
    if v is None or np.isnan(v): return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:.2f}"

def color_delta(v):
    return "🟢" if v >= 0 else "🔴"

# ── RSI ────────────────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

# ── MACD ───────────────────────────────────────────────────────────────────────
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast).mean()
    ema_slow   = series.ewm(span=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

# ── BUILD PORTFOLIO TABLE ───────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def build_portfolio_df():
    rows = []
    for tk, d in PORTFOLIO.items():
        price_adr, prev_adr = get_price(d["yf_ticker"])
        if price_adr:
            price  = cedear_price(price_adr, d["yf_ticker"])
            prev   = cedear_price(prev_adr, d["yf_ticker"])
            var_d  = price - prev
            var_pct = (var_d / prev) * 100 if prev else 0
        else:
            price = d["ppc"]; prev = d["ppc"]; var_d = 0; var_pct = 0

        total    = price * d["qty"]
        gan_usd  = (price - d["ppc"]) * d["qty"]
        gan_pct  = ((price - d["ppc"]) / d["ppc"]) * 100

        rows.append({
            "Ticker":   tk,
            "Nombre":   d["name"],
            "Sector":   d["sector"],
            "Cant.":    d["qty"],
            "Precio":   price,
            "PPC":      d["ppc"],
            "Var% Día": var_pct,
            "Var$ Día": var_d * d["qty"],
            "Gan%":     gan_pct,
            "Gan USD":  gan_usd,
            "Total":    total,
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 FinDash — Mi Portafolio")

tab1, tab2, tab3 = st.tabs(["📊 Cartera", "🔍 Activo + Fundamentals", "📺 TradingView"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CARTERA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if st.button("🔄 Actualizar precios"):
        st.cache_data.clear()

    df = build_portfolio_df()

    total_val   = df["Total"].sum()
    total_gan   = df["Gan USD"].sum()
    total_gan_p = (total_gan / (total_val - total_gan)) * 100
    var_dia     = df["Var$ Día"].sum()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💼 Valor Total",     f"${total_val:,.2f}")
    c2.metric("📈 P&L Total",       f"${total_gan:,.2f}",  f"{total_gan_p:.2f}%")
    c3.metric("⚡ Variación Hoy",   f"${var_dia:,.2f}")
    c4.metric("🗂️ Posiciones",      f"{len(df)}")

    st.divider()

    col_l, col_r = st.columns(2)

    # Pie — composición por activo
    with col_l:
        fig_pie = px.pie(
            df, values="Total", names="Ticker",
            title="Composición del portafolio",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              hovertemplate="<b>%{label}</b><br>USD %{value:,.2f}<br>%{percent}")
        fig_pie.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", title_font_color="#818cf8",
            legend=dict(font=dict(size=12, color="#e2e8f0")),
            margin=dict(t=50, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar — P&L USD
    with col_r:
        df_sorted = df.sort_values("Gan USD", ascending=True)
        colors = ["#4ade80" if v >= 0 else "#f87171" for v in df_sorted["Gan USD"]]
        fig_bar = go.Figure(go.Bar(
            x=df_sorted["Gan USD"], y=df_sorted["Ticker"],
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.2f}" for v in df_sorted["Gan USD"]],
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
            hovertemplate="<b>%{y}</b><br>P&L: $%{x:,.2f}<extra></extra>"
        ))
        fig_bar.update_layout(
            title="Ganancia / Pérdida por activo (USD)",
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", title_font_color="#818cf8",
            xaxis=dict(gridcolor="#334155", color="#94a3b8", tickfont=dict(size=12)),
            yaxis=dict(gridcolor="#334155", color="#94a3b8", tickfont=dict(size=13)),
            margin=dict(t=50, b=10, r=80)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla
    st.subheader("📋 Posiciones detalladas")
    df_show = df.copy()
    df_show["Precio"]   = df_show["Precio"].map("${:,.2f}".format)
    df_show["PPC"]      = df_show["PPC"].map("${:,.2f}".format)
    df_show["Var% Día"] = df_show["Var% Día"].map("{:+.2f}%".format)
    df_show["Var$ Día"] = df_show["Var$ Día"].map("${:+,.2f}".format)
    df_show["Gan%"]     = df_show["Gan%"].map("{:+.2f}%".format)
    df_show["Gan USD"]  = df_show["Gan USD"].map("${:+,.2f}".format)
    df_show["Total"]    = df_show["Total"].map("${:,.2f}".format)
    st.dataframe(df_show, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ACTIVO + FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    ticker_sel = st.selectbox(
        "Seleccioná un activo",
        options=list(PORTFOLIO.keys()),
        format_func=lambda x: f"{x} — {PORTFOLIO[x]['name']}"
    )

    yf_tk = PORTFOLIO[ticker_sel]["yf_ticker"]
    period = st.select_slider("Período del chart técnico", ["3mo","6mo","1y","2y","5y"], value="1y")

    hist = get_history(yf_tk, period)
    info, fin, cf, bal = get_fundamentals(yf_tk)

    # ── CHART TÉCNICO ──────────────────────────────────────────────────────────
    if not hist.empty:
        rsi  = calc_rsi(hist["Close"])
        macd_line, signal_line, histogram = calc_macd(hist["Close"])
        hist["SMA20"]  = hist["Close"].rolling(20).mean()
        hist["SMA50"]  = hist["Close"].rolling(50).mean()
        hist["SMA200"] = hist["Close"].rolling(200).mean()

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.03,
            subplot_titles=[f"{ticker_sel} — Precio", "RSI (14)", "MACD"]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="Precio",
            increasing_line_color="#4ade80", decreasing_line_color="#f87171"
        ), row=1, col=1)

        for ma, color in [("SMA20","#f59e0b"),("SMA50","#818cf8"),("SMA200","#ec4899")]:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[ma], name=ma,
                                     line=dict(color=color, width=1.5)), row=1, col=1)

        # Volumen
        vol_colors = ["#4ade80" if c >= o else "#f87171"
                      for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volumen",
                             marker_color=vol_colors, opacity=0.4, showlegend=False), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI",
                                  line=dict(color="#f59e0b", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f87171", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", opacity=0.5, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="#f87171", opacity=0.05, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="#4ade80", opacity=0.05, row=2, col=1)

        # MACD
        hist_colors = ["#4ade80" if v >= 0 else "#f87171" for v in histogram]
        fig.add_trace(go.Bar(x=hist.index, y=histogram, name="Histograma",
                             marker_color=hist_colors, opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=macd_line, name="MACD",
                                  line=dict(color="#818cf8", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=signal_line, name="Signal",
                                  line=dict(color="#f59e0b", width=1.5)), row=3, col=1)

        fig.update_layout(
            height=700,
            paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
            margin=dict(t=40, b=10),
        )
        for i in range(1, 4):
            fig.update_xaxes(gridcolor="#1e2d40", row=i, col=1)
            fig.update_yaxes(gridcolor="#1e2d40", row=i, col=1,
                             tickfont=dict(size=11, color="#94a3b8"))

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── FUNDAMENTALS ──────────────────────────────────────────────────────────
    st.subheader(f"🔍 Fundamentals — {ticker_sel}")

    # Info rápida
    ki1, ki2, ki3, ki4, ki5 = st.columns(5)
    pe  = info.get("trailingPE")
    fwd = info.get("forwardPE")
    ps  = info.get("priceToSalesTrailing12Months")
    pb  = info.get("priceToBook")
    mc  = info.get("marketCap")

    ki1.metric("P/E (trailing)", f"{pe:.1f}" if pe else "N/A")
    ki2.metric("P/E (forward)",  f"{fwd:.1f}" if fwd else "N/A")
    ki3.metric("P/S",            f"{ps:.1f}" if ps else "N/A")
    ki4.metric("P/B",            f"{pb:.1f}" if pb else "N/A")
    ki5.metric("Market Cap",     fmt_mill(mc) if mc else "N/A")

    st.divider()

    # ── 1. REVENUE ────────────────────────────────────────────────────────────
    st.markdown("#### 1️⃣ Revenue — ¿Crece de forma constante?")
    if not fin.empty and "Total Revenue" in fin.index:
        rev = fin.loc["Total Revenue"].dropna().sort_index()
        rev_df = pd.DataFrame({"Año": rev.index.year, "Revenue": rev.values})
        fig_rev = px.bar(rev_df, x="Año", y="Revenue",
                         text=[fmt_mill(v) for v in rev_df["Revenue"]],
                         color_discrete_sequence=["#6366f1"])
        fig_rev.update_traces(textposition="outside", textfont=dict(size=13, color="#e2e8f0"))
        fig_rev.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=300, margin=dict(t=20,b=10),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#1e2d40"),
            xaxis=dict(tickfont=dict(size=12), gridcolor="#1e2d40"),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

        # Crecimiento YoY
        if len(rev) >= 2:
            cagr = ((rev.iloc[-1] / rev.iloc[0]) ** (1 / max(len(rev)-1,1)) - 1) * 100
            yoy  = rev.pct_change().dropna() * 100
            cr1, cr2 = st.columns(2)
            cr1.metric("CAGR Revenue", f"{cagr:.1f}%",
                       "✅ Crecimiento sólido" if cagr > 10 else "⚠️ Revisar")
            cr2.metric("Último crecimiento YoY",
                       f"{yoy.iloc[-1]:.1f}%",
                       color_delta(yoy.iloc[-1]))
    else:
        st.info("Revenue no disponible para este activo (ETF o bono).")

    # ── 2. FREE CASH FLOW ─────────────────────────────────────────────────────
    st.markdown("#### 2️⃣ Free Cash Flow — ¿Crece y es positivo?")
    if not cf.empty and "Free Cash Flow" in cf.index:
        fcf = cf.loc["Free Cash Flow"].dropna().sort_index()
        fcf_df = pd.DataFrame({"Año": fcf.index.year, "FCF": fcf.values})
        bar_colors = ["#4ade80" if v >= 0 else "#f87171" for v in fcf_df["FCF"]]
        fig_fcf = px.bar(fcf_df, x="Año", y="FCF",
                         text=[fmt_mill(v) for v in fcf_df["FCF"]],
                         color_discrete_sequence=["#10b981"])
        fig_fcf.update_traces(marker_color=bar_colors, textposition="outside",
                              textfont=dict(size=13, color="#e2e8f0"))
        fig_fcf.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=300, margin=dict(t=20,b=10),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#1e2d40"),
            xaxis=dict(tickfont=dict(size=12), gridcolor="#1e2d40"),
        )
        st.plotly_chart(fig_fcf, use_container_width=True)

        positivos = (fcf > 0).sum()
        st.metric("Años con FCF positivo", f"{positivos}/{len(fcf)}",
                  "✅ Consistente" if positivos == len(fcf) else "⚠️ Años negativos")
    else:
        st.info("FCF no disponible para este activo.")

    # ── 3. MÁRGENES ───────────────────────────────────────────────────────────
    st.markdown("#### 3️⃣ Márgenes — ¿Estables o mejorando?")
    if not fin.empty:
        rows_mg = {}
        if "Total Revenue" in fin.index and "Net Income" in fin.index:
            rev_s  = fin.loc["Total Revenue"].dropna().sort_index()
            ni_s   = fin.loc["Net Income"].dropna().sort_index()
            idx    = rev_s.index.intersection(ni_s.index)
            rows_mg["Margen Neto %"] = (ni_s[idx] / rev_s[idx] * 100).values
            years = idx.year

        if "Total Revenue" in fin.index and "Operating Income" in fin.index:
            oi_s   = fin.loc["Operating Income"].dropna().sort_index()
            idx2   = rev_s.index.intersection(oi_s.index)
            rows_mg["Margen Operativo %"] = (oi_s[idx2] / rev_s[idx2] * 100).values
            years = idx2.year

        if rows_mg:
            mg_df = pd.DataFrame(rows_mg, index=years)
            fig_mg = go.Figure()
            for col, color in zip(mg_df.columns, ["#818cf8","#10b981"]):
                fig_mg.add_trace(go.Scatter(
                    x=mg_df.index, y=mg_df[col], name=col,
                    mode="lines+markers+text",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=8),
                    text=[f"{v:.1f}%" for v in mg_df[col]],
                    textposition="top center",
                    textfont=dict(size=12, color=color)
                ))
            fig_mg.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                font_color="#e2e8f0", height=300, margin=dict(t=20,b=10),
                yaxis=dict(ticksuffix="%", gridcolor="#1e2d40", tickfont=dict(size=11)),
                xaxis=dict(gridcolor="#1e2d40", tickfont=dict(size=12)),
                legend=dict(font=dict(size=12))
            )
            st.plotly_chart(fig_mg, use_container_width=True)
        else:
            st.info("Márgenes no disponibles.")
    else:
        st.info("Datos financieros no disponibles para este activo.")

    # ── 4. DEUDA ──────────────────────────────────────────────────────────────
    st.markdown("#### 4️⃣ Deuda — ¿Controlada respecto al Cash Flow?")
    if not bal.empty and not cf.empty:
        debt_key = next((k for k in ["Total Debt","Long Term Debt"] if k in bal.index), None)
        if debt_key and "Operating Cash Flow" in cf.index:
            debt = bal.loc[debt_key].dropna().sort_index()
            ocf  = cf.loc["Operating Cash Flow"].dropna().sort_index()
            idx  = debt.index.intersection(ocf.index)
            if len(idx) > 0:
                ratio = (debt[idx] / ocf[idx]).abs()
                ratio_df = pd.DataFrame({"Año": idx.year, "Deuda/OCF": ratio.values})
                bar_colors_d = ["#4ade80" if v <= 3 else "#f87171" for v in ratio_df["Deuda/OCF"]]
                fig_debt = px.bar(ratio_df, x="Año", y="Deuda/OCF",
                                  text=[f"{v:.1f}x" for v in ratio_df["Deuda/OCF"]],
                                  color_discrete_sequence=["#f59e0b"])
                fig_debt.update_traces(marker_color=bar_colors_d, textposition="outside",
                                       textfont=dict(size=13, color="#e2e8f0"))
                fig_debt.update_layout(
                    paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                    font_color="#e2e8f0", height=300, margin=dict(t=20,b=10),
                    yaxis=dict(tickfont=dict(size=11), gridcolor="#1e2d40",
                               title="Años para pagar deuda con OCF"),
                    xaxis=dict(tickfont=dict(size=12), gridcolor="#1e2d40"),
                )
                st.plotly_chart(fig_debt, use_container_width=True)
                last_ratio = ratio.iloc[-1]
                st.metric("Ratio actual Deuda/OCF", f"{last_ratio:.1f}x",
                          "✅ Saludable (<3x)" if last_ratio <= 3 else "⚠️ Alto (>3x)")
        else:
            st.info("Datos de deuda no disponibles.")
    else:
        st.info("Balance o cashflow no disponibles para este activo.")

    # ── 5. P/E o FCF YIELD ───────────────────────────────────────────────────
    st.markdown("#### 5️⃣ FCF Yield — ¿Razonable vs su promedio histórico?")
    price_now, _ = get_price(yf_tk)
    mc_val = info.get("marketCap")
    if not cf.empty and "Free Cash Flow" in cf.index and mc_val and price_now:
        fcf_s   = cf.loc["Free Cash Flow"].dropna().sort_index()
        shares  = info.get("sharesOutstanding", 1)
        mcs     = [mc_val] * len(fcf_s)  # simplificado: usamos market cap actual
        fcfy    = (fcf_s.values / np.array(mcs)) * 100
        fcfy_df = pd.DataFrame({"Año": fcf_s.index.year, "FCF Yield %": fcfy})
        avg     = fcfy_df["FCF Yield %"].mean()

        fig_fcfy = go.Figure()
        fig_fcfy.add_trace(go.Scatter(
            x=fcfy_df["Año"], y=fcfy_df["FCF Yield %"],
            mode="lines+markers+text",
            line=dict(color="#c084fc", width=2.5),
            marker=dict(size=9),
            text=[f"{v:.1f}%" for v in fcfy_df["FCF Yield %"]],
            textposition="top center",
            textfont=dict(size=12, color="#c084fc"),
            name="FCF Yield %"
        ))
        fig_fcfy.add_hline(y=avg, line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"Promedio: {avg:.1f}%",
                           annotation_font_color="#f59e0b")
        fig_fcfy.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=300, margin=dict(t=20,b=10),
            yaxis=dict(ticksuffix="%", gridcolor="#1e2d40", tickfont=dict(size=11)),
            xaxis=dict(gridcolor="#1e2d40", tickfont=dict(size=12)),
        )
        st.plotly_chart(fig_fcfy, use_container_width=True)

        curr = fcfy_df["FCF Yield %"].iloc[-1]
        st.metric("FCF Yield actual", f"{curr:.1f}%",
                  "✅ Por encima del promedio (atractivo)" if curr >= avg
                  else "⚠️ Por debajo del promedio (caro)")
    else:
        pe_val = info.get("trailingPE")
        if pe_val:
            st.metric("P/E actual", f"{pe_val:.1f}x",
                      "✅ Razonable (<25x)" if pe_val < 25 else "⚠️ Elevado (>25x)")
        else:
            st.info("P/E o FCF Yield no disponibles para este activo.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRADINGVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📺 TradingView — Chart avanzado")
    tv_map = {
        "AMZN":"NASDAQ:AMZN","ASML":"NASDAQ:ASML","BRKB":"NYSE:BRK.B",
        "EEM":"AMEX:EEM","MA":"NYSE:MA","META":"NASDAQ:META",
        "MSFT":"NASDAQ:MSFT","MU":"NASDAQ:MU","NFLX":"NASDAQ:NFLX",
        "NU":"NYSE:NU","XOM":"NYSE:XOM"
    }
    tv_sel = st.selectbox("Seleccioná activo", list(tv_map.keys()),
                          format_func=lambda x: f"{x} — {PORTFOLIO[x]['name']}")
    tv_symbol = tv_map[tv_sel]
    tv_html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%", "height": 600,
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "America/Argentina/Buenos_Aires",
          "theme": "dark",
          "style": "1",
          "locale": "es",
          "toolbar_bg": "#1e293b",
          "enable_publishing": false,
          "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],
          "container_id": "tradingview_chart"
        }});
      </script>
    </div>
    """
    st.components.v1.html(tv_html, height=620)
