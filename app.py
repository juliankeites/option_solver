import calendar, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Asian Option Strategy Payoff", layout="wide")

PRESETS = {"Jet fuel (USD/t)": 800.0, "Jet fuel (USD/bbl)": 105.0, "Brent (USD/bbl)": 85.0,
           "ICE Gasoil (USD/t)": 750.0, "EUR/USD": 1.08, "GBP/USD": 1.30, "Custom": 100.0}

# ---------------------------------------------------------------- pricing
def fixing_dates(start, expiry):
    """Business days from start date to expiry (last fixing), inclusive."""
    d, out = start, []
    while d <= expiry:
        if d.weekday() < 5: out.append(d)
        d += dt.timedelta(days=1)
    return out or [expiry]

def eff_vol(vol, t_fix, T):
    """Moment-matched vol of the arithmetic average of a flat forward curve.
    E[A]=F ; E[A^2]=F^2/N^2 * sum_ij exp(vol^2*min(ti,tj))."""
    t = np.asarray(t_fix, float)
    m2 = np.exp(vol**2*np.minimum.outer(t, t)).sum()/len(t)**2
    return np.sqrt(max(np.log(m2), 1e-12)/T)

def black76(kind, F, K, T, vol, r):
    if T <= 0 or vol <= 0:
        return max(F-K, 0) if kind == "Call" else max(K-F, 0)
    d1 = (np.log(F/K) + 0.5*vol**2*T)/(vol*np.sqrt(T)); d2 = d1 - vol*np.sqrt(T); df = np.exp(-r*T)
    return df*(F*norm.cdf(d1)-K*norm.cdf(d2)) if kind == "Call" else df*(K*norm.cdf(-d2)-F*norm.cdf(-d1))

def asian_price(kind, F, K, T, vol, r, t_fix):
    return black76(kind, F, K, T, eff_vol(vol, t_fix, T), r)

def intrinsic(kind, A, K):
    return np.maximum(A-K, 0) if kind == "Call" else np.maximum(K-A, 0)

def add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return dt.date(d.year + y, mo + 1, 1)

def last_bday(first):
    d = dt.date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])
    while d.weekday() > 4: d -= dt.timedelta(days=1)
    return d

def times_for(start, expiry, today):
    fixes = fixing_dates(start, expiry)
    return fixes, [max((d - today).days, 0)/365.0 for d in fixes]

def segments(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i-1]*y[i] < 0:
            xs.append(x[i-1] - y[i-1]*(x[i]-x[i-1])/(y[i]-y[i-1])); ys.append(0.0)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)

def red_green(fig, xs, ys, lw=3, label=True):
    fig.add_trace(go.Scatter(x=xs, y=np.where(ys >= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(0,160,0,0.35)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=np.where(ys <= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(220,0,0,0.35)", hoverinfo="skip", showlegend=False))
    for mask, col, nm in [(ys >= 0, "green", "Gain"), (ys <= 0, "red", "Loss")]:
        fig.add_trace(go.Scatter(x=xs, y=np.where(mask, ys, np.nan), mode="lines", line=dict(color=col, width=lw),
                                 name=nm, showlegend=label))

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Market & expiry")
name = sb.selectbox("1) Underlying", list(PRESETS))
if name == "Custom":
    name = sb.text_input("Custom name", "My underlying")
spot = sb.number_input("Underlying / forward price", value=float(PRESETS.get(name, 100.0)), min_value=0.0001,
                       step=0.01, format="%.4f", help="Flat forward curve: forward for every fixing date = this price.")
today = dt.date.today()
months = sb.slider("Months ahead (expiry month)", 1, 6, 3)
first = add_months(today, months)
last = dt.date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])
expiry = sb.date_input(f"Final fixing / expiry date ({first:%B %Y})", value=last_bday(first), min_value=first, max_value=last,
                       help="Last averaging date. Set per product (gasoil, Brent, jet swaps differ).")

sb.header("Averaging period")
mode = sb.selectbox("Averaging start", ["Start of expiry month", "Today", "Tomorrow", "In X days", "Specific date"],
                    help="Start can be anywhere between today and the expiry. A shorter window means a higher effective vol and a higher premium; "
                         "a longer window (earlier start) is cheaper.")
if mode == "Start of expiry month": start = first
elif mode == "Today": start = today
elif mode == "Tomorrow": start = today + dt.timedelta(days=1)
elif mode == "In X days":
    start = today + dt.timedelta(days=int(sb.number_input("Days from today", 0, max(1, (expiry - today).days), 30, step=1)))
else:
    start = sb.date_input("Averaging start date", value=first, min_value=today, max_value=expiry)
start = min(start, expiry)
fixes, t_fix = times_for(start, expiry, today)
T = max(t_fix[-1], 1/365)
sb.caption(f"{len(fixes)} daily fixings, {fixes[0]:%d %b %Y} to {fixes[-1]:%d %b %Y} | T(expiry) = {T:.3f} yrs")

sb.header("Pricing inputs")
vol_all = sb.number_input("Volatility for all strikes (%)", value=30.0, min_value=0.1, step=1.0,
                          help="Flat, no skew. Applied to every leg; override per leg in the table. Underlying vol, not vol of the average.")
r = sb.number_input("Risk-free rate (%)", value=4.0, step=0.25, min_value=-5.0)/100
mult = sb.number_input("Contract size multiplier", value=1.0, min_value=0.0001, step=1.0)
ccy = sb.text_input("Currency label", "USD")

sb.header("Physical exposure")
phys_side = sb.radio("Physical position", ["Buyer of fuel (cost rises with price)", "Seller / producer (revenue rises with price)", "None"])
phys_vol = sb.number_input("Physical volume (same unit as underlying)", value=1000.0, min_value=0.0, step=100.0)
phys_ref = sb.number_input("Physical reference / budget price", value=float(spot), min_value=0.0001, step=0.01, format="%.4f",
                           help="Physical fuel is assumed priced at the same average price as the options.")

# ---------------------------------------------------------------- legs
st.title("Asian (Average Price) Option Strategy – P&L Payoff Diagram")
st.subheader("2–3) Build legs")
TEMPLATES = {
 "Custom": [("Long","Call",100),("Short","Call",115)],
 "Long call": [("Long","Call",100)],
 "Call spread (2-way)": [("Long","Call",100),("Short","Call",115)],
 "Zero-cost collar (2-way)": [("Long","Call",105),("Short","Put",95)],
 "Three-way (call spread + short put)": [("Long","Call",100),("Short","Call",120),("Short","Put",90)],
 "Four-way (call spread + put spread)": [("Long","Call",100),("Short","Call",120),("Short","Put",90),("Long","Put",80)],
}
tpl = st.selectbox("Strategy template (editable afterwards)", list(TEMPLATES), index=5)
qty0 = st.number_input("Default quantity per leg", value=1000.0, min_value=0.0, step=100.0)
t = TEMPLATES[tpl]
n = int(st.number_input("Number of legs", 1, 8, len(t), key=f"n_{tpl}"))
rows0 = [(d, ty, round(spot*k/100, 2), float(vol_all), float(qty0)) for d, ty, k in t]
while len(rows0) < n: rows0.append(("Long", "Call", round(spot, 2), float(vol_all), float(qty0)))
default = pd.DataFrame(rows0[:n], columns=["Direction","Type","Strike","Vol %","Qty"])
legs = st.data_editor(
    default, key=f"legs_{tpl}_{n}_{spot}_{qty0}_{vol_all}", num_rows="fixed", use_container_width=True,
    column_config={"Direction": st.column_config.SelectboxColumn(options=["Long", "Short"]),
                   "Type": st.column_config.SelectboxColumn(options=["Call", "Put"]),
                   "Strike": st.column_config.NumberColumn(min_value=0.0001, format="%.4f"),
                   "Vol %": st.column_config.NumberColumn(min_value=0.1, format="%.2f"),
                   "Qty": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})

def strategy_cost(tf, Tx, asian=True):
    c = 0.0
    for _, l in legs.iterrows():
        sign = 1 if l["Direction"] == "Long" else -1
        v = l["Vol %"]/100
        p = asian_price(l["Type"], spot, l["Strike"], Tx, v, r, tf) if asian else black76(l["Type"], spot, l["Strike"], Tx, v, r)
        c += sign*p*l["Qty"]*mult
    return c

A_grid = np.linspace(spot*0.3, spot*1.9, 1200)
rows, total_cost, payoff, call_net = [], 0.0, np.zeros_like(A_grid), 0.0
for _, l in legs.iterrows():
    sign = 1 if l["Direction"] == "Long" else -1
    q = l["Qty"]*mult; v = l["Vol %"]/100
    p = asian_price(l["Type"], spot, l["Strike"], T, v, r, t_fix)
    total_cost += sign*p*q
    payoff += sign*intrinsic(l["Type"], A_grid, l["Strike"])*q
    if l["Type"] == "Call": call_net += sign*q
    rows.append({**l.to_dict(), "Eff. avg vol %": 100*eff_vol(v, t_fix, T),
                 "European premium": black76(l["Type"], spot, l["Strike"], T, v, r),
                 "Asian unit premium": p, "Cash (+pay/-receive)": sign*p*q})
pnl = payoff - total_cost

def pnl_at(a):
    v = -total_cost
    for _, l in legs.iterrows():
        sign = 1 if l["Direction"] == "Long" else -1
        v += sign*float(intrinsic(l["Type"], a, l["Strike"]))*l["Qty"]*mult
    return v

# ---------------------------------------------------------------- chart 1
xs, ys = segments(A_grid, pnl)
fig = go.Figure()
red_green(fig, xs, ys)
fig.add_vline(x=spot, line_dash="dot", annotation_text=f"Fwd {spot:g}")
fig.add_hline(y=0, line_color="black")
fig.update_layout(title=f"{name} – {tpl} – Asian option P&L vs average price ({fixes[0]:%d %b %Y} to {fixes[-1]:%d %b %Y})",
                  xaxis_title=f"Average {name} price over averaging period", yaxis_title=f"P&L ({ccy})",
                  height=480, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

up_unl, down_unl = call_net > 1e-12, call_net < -1e-12
mx = max(pnl.max(), pnl_at(0.0)); mn = min(pnl.min(), pnl_at(0.0))
be = sorted(set(round(x, 2) for x, y0, y1 in zip(xs[:-1], ys[:-1], ys[1:]) if y0 == 0 and y1 != 0))
st.subheader("4–6) Option strategy results")
eur_cost = strategy_cost(None, T, asian=False)
m = st.columns(5)
m[0].metric("Net cost (+ pay / – receive)", f"{total_cost:,.2f} {ccy}")
m[1].metric("Max gain", "Unlimited" if up_unl else f"{mx:,.2f} {ccy}")
m[2].metric("Max loss", "Unlimited" if down_unl else f"{mn:,.2f} {ccy}")
m[3].metric("Breakevens (average price)", ", ".join(f"{b:,.2f}" for b in be) or "None")
m[4].metric("Same strategy, European", f"{eur_cost:,.2f} {ccy}", delta=f"{total_cost-eur_cost:,.2f} (Asian saving)", delta_color="off")
if down_unl:
    st.error("Net short calls: loss is unlimited if the average price rises – flag for risk committee.")
st.dataframe(pd.DataFrame(rows).style.format({"Eff. avg vol %": "{:,.2f}", "European premium": "{:,.4f}",
             "Asian unit premium": "{:,.4f}", "Cash (+pay/-receive)": "{:,.2f}"}), use_container_width=True)

# ---------------------------------------------------------------- cost vs averaging start
st.subheader("Net cost vs averaging start date (same expiry)")
cands = {"Today": today, "Tomorrow": today + dt.timedelta(days=1), "+30 days": today + dt.timedelta(days=30),
         "+60 days": today + dt.timedelta(days=60), "Start of expiry month": first}
srows = []
for lab, d0 in cands.items():
    d0 = min(max(d0, today), expiry)
    fx, tf = times_for(d0, expiry, today); Tx = max(tf[-1], 1/365)
    c = strategy_cost(tf, Tx)
    srows.append({"Averaging start": lab, "Date": d0.strftime("%d %b %Y"), "Fixings": len(fx),
                  "Net cost": c, "vs European": c - eur_cost})
st.dataframe(pd.DataFrame(srows).style.format({"Net cost": "{:,.0f}", "vs European": "{:,.0f}"}), use_container_width=True)
st.caption("Earlier start = longer averaging window = lower effective vol = cheaper long premium (and less premium received on short legs).")

# ---------------------------------------------------------------- chart 2
st.subheader("7) Physical exposure + option strategy = net position")
sgn = {"Buyer of fuel (cost rises with price)": -1, "Seller / producer (revenue rises with price)": 1, "None": 0}[phys_side]
phys = sgn*(A_grid - phys_ref)*phys_vol
net = phys + pnl
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=A_grid, y=phys, mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=2, dash="dash"), name="Physical (lowlight)"))
fig2.add_trace(go.Scatter(x=A_grid, y=pnl, mode="lines", line=dict(color="rgba(70,130,200,0.55)", width=2), name="Asian option strategy (lowlight)"))
xn, yn = segments(A_grid, net)
red_green(fig2, xn, yn, lw=4)
fig2.add_vline(x=spot, line_dash="dot", annotation_text=f"Fwd {spot:g}")
fig2.add_hline(y=0, line_color="black")
fig2.update_layout(title="Net P&L vs average price (negative = cost increase / loss)",
                   xaxis_title=f"Average {name} price", yaxis_title=f"P&L ({ccy})", height=520, hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

nm = st.columns(3)
nm[0].metric("Net worst case (grid 30%-190% of fwd)", f"{net.min():,.0f} {ccy}")
nm[1].metric("Net best case (grid)", f"{net.max():,.0f} {ccy}")
nm[2].metric("Physical-only worst (grid)", f"{phys.min():,.0f} {ccy}")
sc = []
for s in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
    P = spot*(1+s); ph = sgn*(P-phys_ref)*phys_vol; op = pnl_at(P)
    sc.append({"Avg price move": f"{s:+.0%}", "Avg price": P, "Physical P&L": ph, "Option P&L": op, "Net P&L": ph+op,
               "Effective unit price (buyer)": (phys_ref*phys_vol - ph - op)/phys_vol if phys_vol else np.nan})
st.dataframe(pd.DataFrame(sc).style.format({"Avg price": "{:,.2f}", "Physical P&L": "{:,.0f}", "Option P&L": "{:,.0f}",
             "Net P&L": "{:,.0f}", "Effective unit price (buyer)": "{:,.2f}"}), use_container_width=True)
st.caption("Arithmetic daily-average (business days) options, flat forward curve, flat vol, moment-matched lognormal approximation, "
           "premium discounted from expiry, no transaction costs/basis/margining. Indicative only.")
