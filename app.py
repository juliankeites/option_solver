import calendar, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import minimize

st.set_page_config(page_title="Monthly Asian Option Strip Solver", layout="wide")

def bdays(start, expiry):
    d, out = start, []
    while d <= expiry:
        if d.weekday() < 5: out.append(d)
        d += dt.timedelta(days=1)
    return out or [expiry]

def eff_vol(vol, t_fix, T):
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

def next_month_start(today):
    nm = add_months(today, 1)
    return dt.date(nm.year, nm.month, 1)

def segments(x, y):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if y[i-1]*y[i] < 0:
            xs.append(x[i-1] - y[i-1]*(x[i]-x[i-1])/(y[i]-y[i-1])); ys.append(0.0)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)

def red_green(fig, xs, ys, lw=3):
    fig.add_trace(go.Scatter(x=xs, y=np.where(ys >= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(0,160,0,0.35)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=np.where(ys <= 0, ys, 0), mode="lines", line=dict(width=0), fill="tozeroy",
                             fillcolor="rgba(220,0,0,0.35)", hoverinfo="skip", showlegend=False))
    for mask, col, nm in [(ys >= 0, "green", "Gain"), (ys <= 0, "red", "Loss")]:
        fig.add_trace(go.Scatter(x=xs, y=np.where(mask, ys, np.nan), mode="lines", line=dict(color=col, width=lw), name=nm))

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Global settings")
today = dt.date.today()
r = sb.number_input("Risk-free rate (%)", value=0.0, step=0.25, min_value=-5.0)/100
ccy = sb.text_input("Currency label", "USD")
mult = sb.number_input("Contract size multiplier", value=1.0, min_value=0.0001, step=1.0)
n_months = sb.slider("Number of months in strip", 1, 12, 3)
first_month = add_months(today, sb.slider("First expiry month (months ahead)", 1, 12, 1))
days0 = sb.number_input("Days from today to averaging start", value=(next_month_start(today)-today).days, min_value=0, max_value=365)
start_date = today + dt.timedelta(days=int(days0))
sb.caption(f"Averaging starts {start_date:%d %b %Y} ({days0} days from today, start of next whole month by default).")

sb.header("Physical exposure")
phys_vol = sb.number_input("Physical volume per month (same unit)", value=290000.0, min_value=0.0, step=1000.0)
phys_ref = sb.number_input("Physical reference / budget price", value=196.0, min_value=0.0001, step=0.01, format="%.4f")

sb.header("Solver targets")
max_spend = sb.number_input("Max total premium spend (USD)", value=750000.0, min_value=0.0, step=10000.0)
max_loss = sb.number_input("Max total loss (worst-case net P&L, USD)", value=-2000000.0, step=100000.0)
sb.caption("Hedge ratio is set per month in the table (default 40%). Solver variables: short put strike % and long put strike % only. "
           "Call strikes fixed at your minimums. Minimises premium spend; if infeasible, returns closest solution plus what must change.")

# ---------------------------------------------------------------- monthly inputs
st.title("Monthly Asian Option Strip – 4-Way Solver")
st.subheader("1) Monthly market inputs (forward, vol, hedge ratio per month)")
mrows = []
for i in range(n_months):
    fm = add_months(first_month, i)
    mrows.append({"Month": fm.strftime("%b %Y"), "Forward": 196.0, "Vol %": 30.0, "Hedge %": 40.0,
                  "Start": start_date.strftime("%Y-%m-%d"), "Expiry": last_bday(fm).strftime("%Y-%m-%d")})
months = st.data_editor(pd.DataFrame(mrows), num_rows="fixed", use_container_width=True,
    column_config={"Month": st.column_config.TextColumn(disabled=True),
                   "Forward": st.column_config.NumberColumn(min_value=0.0001, format="%.4f"),
                   "Vol %": st.column_config.NumberColumn(min_value=0.1, format="%.2f"),
                   "Hedge %": st.column_config.NumberColumn(min_value=0.0, max_value=200.0, format="%.1f",
                       help="Hedge ratio as % of physical volume for this month"),
                   "Start": st.column_config.TextColumn(help="Averaging start date (YYYY-MM-DD)"),
                   "Expiry": st.column_config.TextColumn(help="Final fixing / expiry date (YYYY-MM-DD)")})

st.subheader("2) Call strike minimums (% of that month's forward) – fixed by you")
c = st.columns(2)
k1p = c[0].number_input("Long call strike % (minimum)", value=100.0, min_value=1.0)
k2p = c[1].number_input("Short call strike % (minimum)", value=110.0, min_value=1.0)
if k2p <= k1p:
    st.warning("Short call strike % should be above long call strike %.")
k3_lo = st.sidebar.number_input("Short put strike % – lower bound", value=70.0, min_value=1.0)
k3_hi = st.sidebar.number_input("Short put strike % – upper bound", value=100.0, min_value=1.0)
k4_lo = st.sidebar.number_input("Long put strike % – lower bound", value=50.0, min_value=1.0)

# ---------------------------------------------------------------- per-month precompute
md = []
for _, row in months.iterrows():
    fm = dt.datetime.strptime(row["Month"], "%b %Y").date()
    start = dt.datetime.strptime(row["Start"], "%Y-%m-%d").date()
    exp = dt.datetime.strptime(row["Expiry"], "%Y-%m-%d").date()
    fx = bdays(start, exp); tf = np.array([max((d - today).days, 0)/365.0 for d in fx])
    Tx = max(tf[-1], 1/365)
    F = float(row["Forward"]); v = float(row["Vol %"])/100; h = float(row["Hedge %"])/100
    md.append({"label": row["Month"], "F": F, "vol": v, "h": h, "tf": tf, "T": Tx,
               "call_prem": (asian_price("Call", F, F*k1p/100, Tx, v, r, tf)
                             - asian_price("Call", F, F*k2p/100, Tx, v, r, tf))})

pct = np.linspace(-50, 90, 300)

def strip_results(k3p, k4p):
    cost, net = 0.0, np.zeros_like(pct)
    for m in md:
        F, v, T, tf, h = m["F"], m["vol"], m["T"], m["tf"], m["h"]
        K3, K4 = F*k3p/100, F*k4p/100
        prem = m["call_prem"] - asian_price("Put", F, K3, T, v, r, tf) + asian_price("Put", F, K4, T, v, r, tf)
        q = h*phys_vol*mult
        cost += prem*q
        A = F*(1+pct/100)
        p = (np.maximum(A-F*k1p/100, 0) - np.maximum(A-F*k2p/100, 0)
             - np.maximum(K3-A, 0) + np.maximum(K4-A, 0))
        net += (p - prem)*q
    return cost, net.min(), net

def objective(x):
    k3p, k4p = x
    if k4p >= k3p: return 1e9
    cost, worst, _ = strip_results(k3p, k4p)
    v_s = max(0.0, cost - max_spend); v_l = max(0.0, max_loss - worst)
    scale_s = max(abs(max_spend), 1.0); scale_l = max(abs(max_loss), 1.0)
    viol = (v_s/scale_s)**2 + (v_l/scale_l)**2
    return 1e6*viol + cost/scale_s if viol > 0 else cost/scale_s

# ---------------------------------------------------------------- solver
st.subheader("3) Solver")
if st.button("Solve 4-way strip", type="primary"):
    best = None
    for k30 in (k3_lo+5, (k3_lo+k3_hi)/2, k3_hi-5):
        for k40 in (k4_lo, k4_lo+10):
            if k40 >= k30: continue
            res = minimize(objective, x0=[k30, k40], method="Nelder-Mead",
                           bounds=[(k3_lo, k3_hi), (k4_lo, k3_hi-0.5)],
                           options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6})
            if best is None or res.fun < best.fun: best = res
    k3p, k4p = [float(v) for v in best.x]
    k4p = min(k4p, k3p - 0.5)
    cost, worst, net = strip_results(k3p, k4p)
    st.session_state["sol"] = (k3p, k4p, cost, worst, net)

sol = st.session_state.get("sol")
if sol:
    k3p, k4p, cost, worst, net = sol
    feas = cost <= max_spend and worst >= max_loss
    m = st.columns(5)
    m[0].metric("Hedge ratio", "per month (table)")
    m[1].metric("Short put strike", f"{k3p:.1f}% of fwd")
    m[2].metric("Long put strike", f"{k4p:.1f}% of fwd")
    m[3].metric("Total premium spend", f"{cost:,.0f} {ccy}", delta=f"{cost-max_spend:,.0f} vs target", delta_color="inverse")
    m[4].metric("Worst-case net P&L", f"{worst:,.0f} {ccy}", delta=f"{worst-max_loss:,.0f} vs target")
    if feas:
        st.success(f"Both targets met: short put {k3p:.1f}%, long put {k4p:.1f}%. Lowest-spend solution found.")
    else:
        st.warning(f"No solution meets both targets with the current call strikes, put bounds and per-month hedge ratios. Closest: "
                   f"spend over by {max(0,cost-max_spend):,.0f}, loss over by {max(0,max_loss-worst):,.0f}.")
        st.subheader("What must change to get a solution")
        k3s = np.linspace(k3_lo, k3_hi-1.0, 10)
        k4s = np.linspace(k4_lo, k3_hi-2.0, 10)
        min_cost_given_loss, best_wc_given_spend = np.inf, -np.inf
        arg_cost = arg_wc = None
        for k3 in k3s:
            for k4 in k4s:
                if k4 >= k3: continue
                cst, wc, _ = strip_results(k3, k4)
                if wc >= max_loss and cst < min_cost_given_loss:
                    min_cost_given_loss, arg_cost = cst, (k3, k4)
                if cst <= max_spend and wc > best_wc_given_spend:
                    best_wc_given_spend, arg_wc = wc, (k3, k4)
        d = st.columns(2)
        if np.isfinite(min_cost_given_loss):
            d[0].metric("Min spend meeting loss target", f"{min_cost_given_loss:,.0f} {ccy}")
            st.caption(f"Achieved at short put {arg_cost[0]:.1f}%, long put {arg_cost[1]:.1f}%. "
                       f"Raise max spend to at least {min_cost_given_loss:,.0f} (currently {max_spend:,.0f}).")
        else:
            d[0].metric("Min spend meeting loss target", "Not achievable")
            st.caption("Even the widest allowed put strikes cannot meet the loss target. "
                       "Raise the short-put upper bound, lower a hedge ratio, or relax the loss target.")
        if best_wc_given_spend > -np.inf:
            d[1].metric("Best loss meeting spend target", f"{best_wc_given_spend:,.0f} {ccy}")
            st.caption(f"Achieved at short put {arg_wc[0]:.1f}%, long put {arg_wc[1]:.1f}%. "
                       f"Relax max loss to at least {best_wc_given_spend:,.0f} (currently {max_loss:,.0f}).")
        else:
            d[1].metric("Best loss meeting spend target", "Not achievable")
            st.caption("Even the cheapest structure exceeds the spend target. "
                       "Raise max spend, lower a hedge ratio, or lower the long-call strike minimum.")
        st.caption("To meet both: raise max spend to the first figure, relax max loss to the second, "
                   "widen the put strike bounds, or lower one or more monthly hedge ratios and re-solve.")
else:
    k3p, k4p = 90.0, 80.0
    cost, worst, net = strip_results(k3p, k4p)
    st.info("Press 'Solve 4-way strip'. Charts below show an illustrative 90%/80% put structure at the table's hedge ratios.")

# ---------------------------------------------------------------- charts
st.subheader("4) Resulting charts")
fig1 = go.Figure()
for m in md:
    F = m["F"]; K3, K4 = F*k3p/100, F*k4p/100
    A = F*(1+pct/100)
    prem = m["call_prem"] - asian_price("Put", F, K3, m["T"], m["vol"], r, m["tf"]) + asian_price("Put", F, K4, m["T"], m["vol"], r, m["tf"])
    p = (np.maximum(A-F*k1p/100, 0) - np.maximum(A-F*k2p/100, 0) - np.maximum(K3-A, 0) + np.maximum(K4-A, 0))
    fig1.add_trace(go.Scatter(x=A, y=(p-prem)*m["h"]*phys_vol*mult, mode="lines", name=m["label"]))
fig1.add_hline(y=0, line_color="black")
fig1.update_layout(title="Per-month 4-way P&L vs that month's average price",
                   xaxis_title="Average price", yaxis_title=f"P&L ({ccy})", height=450, hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

agg = np.zeros_like(pct)
for m in md:
    F = m["F"]; K3, K4 = F*k3p/100, F*k4p/100
    A = F*(1+pct/100)
    prem = m["call_prem"] - asian_price("Put", F, K3, m["T"], m["vol"], r, m["tf"]) + asian_price("Put", F, K4, m["T"], m["vol"], r, m["tf"])
    p = (np.maximum(A-F*k1p/100, 0) - np.maximum(A-F*k2p/100, 0) - np.maximum(K3-A, 0) + np.maximum(K4-A, 0))
    agg += (p-prem)*m["h"]*phys_vol*mult
xs, ys = segments(1+pct/100, agg)
fig2 = go.Figure()
red_green(fig2, xs, ys, lw=4)
fig2.add_vline(x=1.0, line_dash="dot", annotation_text="Forward")
fig2.add_hline(y=0, line_color="black")
fig2.update_layout(title="Strip total P&L vs % move in each month's average price",
                   xaxis_title="Price move vs forward (%)", yaxis_title=f"P&L ({ccy})", height=480, hovermode="x unified")
st.plotly_chart(fig2, use_container_width=True)

phys = sum((m["F"]*(1+pct/100) - phys_ref)*phys_vol for m in md)*-1
netp = phys + agg
xn, yn = segments(1+pct/100, netp)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=1+pct/100, y=phys, mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=2, dash="dash"), name="Physical (lowlight)"))
fig3.add_trace(go.Scatter(x=1+pct/100, y=agg, mode="lines", line=dict(color="rgba(70,130,200,0.55)", width=2), name="Option strip (lowlight)"))
red_green(fig3, xn, yn, lw=4)
fig3.add_vline(x=1.0, line_dash="dot", annotation_text="Forward")
fig3.add_hline(y=0, line_color="black")
fig3.update_layout(title="Net P&L: physical + option strip (buyer of fuel)",
                   xaxis_title="Price move vs forward (%)", yaxis_title=f"P&L ({ccy})", height=500, hovermode="x unified")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("5) Solved structure per month")
srows = []
for m in md:
    F = m["F"]
    K3, K4 = F*k3p/100, F*k4p/100
    prem = m["call_prem"] - asian_price("Put", F, K3, m["T"], m["vol"], r, m["tf"]) + asian_price("Put", F, K4, m["T"], m["vol"], r, m["tf"])
    srows.append({"Month": m["label"], "Forward": F, "Vol %": m["vol"]*100, "Hedge %": m["h"]*100,
                  "Long call K": F*k1p/100, "Short call K": F*k2p/100,
                  "Short put K": K3, "Long put K": K4, "Volume": m["h"]*phys_vol*mult, "Net premium": prem*m["h"]*phys_vol*mult})
st.dataframe(pd.DataFrame(srows).style.format({"Forward": "{:,.2f}", "Vol %": "{:,.1f}", "Hedge %": "{:,.1f}",
             "Long call K": "{:,.2f}", "Short call K": "{:,.2f}", "Short put K": "{:,.2f}", "Long put K": "{:,.2f}",
             "Volume": "{:,.0f}", "Net premium": "{:,.0f}"}), use_container_width=True)
st.caption("Arithmetic daily-average options, flat forward curve per month, per-month vol and hedge ratio, moment-matched lognormal pricing. "
           "Solver chooses both put strikes; call strikes fixed at your minimums. "
           "If infeasible, diagnostics show the minimum spend or loss relaxation needed.")
