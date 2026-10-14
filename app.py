import calendar, datetime as dt
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import differential_evolution

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
r = sb.number_input("Risk-free rate (%)", value=4.0, step=0.25, min_value=-5.0)/100
ccy = sb.text_input("Currency label", "USD")
mult = sb.number_input("Contract size multiplier", value=1.0, min_value=0.0001, step=1.0)
n_months = sb.slider("Number of months in strip", 1, 12, 3)
first_month = add_months(today, sb.slider("First expiry month (months ahead)", 1, 12, 1))
start_mode = sb.selectbox("Averaging start", ["Start of expiry month", "Today", "Tomorrow", "In X days"])
days0 = sb.number_input("Days from today (if 'In X days')", 0, 365, 30)
sb.caption("Each month has its own forward price and volatility, entered in the table below.")

sb.header("Physical exposure")
phys_vol = sb.number_input("Physical volume per month (same unit)", value=1000.0, min_value=0.0, step=100.0)
phys_ref = sb.number_input("Physical reference / budget price", value=100.0, min_value=0.0001, step=0.01, format="%.4f")

sb.header("Solver targets")
hedge_pct = sb.slider("Starting hedge ratio (% of physical volume)", 0, 200, 100, 5)/100
max_spend = sb.number_input("Max total premium spend", value=50000.0, min_value=0.0, step=1000.0)
max_loss = sb.number_input("Max total loss (worst-case net P&L)", value=-100000.0, step=10000.0)
sb.caption("Strikes are fixed at your chosen offsets. The solver finds the hedge ratio that best meets both targets "
           "(minimum constraint violation; if feasible, minimum spend).")

# ---------------------------------------------------------------- monthly inputs
st.title("Monthly Asian Option Strip – 4-Way Solver")
st.subheader("1) Monthly market inputs (forward price and vol per month)")
st.caption("Edit the Forward and Vol % columns directly for each month. Dates can also be overridden per month.")
mrows = []
for i in range(n_months):
    fm = add_months(first_month, i)
    mrows.append({"Month": fm.strftime("%b %Y"), "Forward": 100.0, "Vol %": 30.0,
                  "Start": fm.strftime("%Y-%m-%d"), "Expiry": last_bday(fm).strftime("%Y-%m-%d")})
months = st.data_editor(pd.DataFrame(mrows), num_rows="fixed", use_container_width=True,
    column_config={"Month": st.column_config.TextColumn(disabled=True),
                   "Forward": st.column_config.NumberColumn(min_value=0.0001, format="%.4f",
                       help="Forward underlying price for this month's average"),
                   "Vol %": st.column_config.NumberColumn(min_value=0.1, format="%.2f",
                       help="Volatility of the underlying for this month"),
                   "Start": st.column_config.TextColumn(help="Averaging start date (YYYY-MM-DD)"),
                   "Expiry": st.column_config.TextColumn(help="Final fixing / expiry date (YYYY-MM-DD)")})

st.subheader("2) 4-way strike offsets (% of that month's forward) – fixed by you")
c = st.columns(4)
k1p = c[0].number_input("Long call strike %", value=100.0, min_value=1.0)
k2p = c[1].number_input("Short call strike %", value=120.0, min_value=1.0)
k3p = c[2].number_input("Short put strike %", value=90.0, min_value=1.0)
k4p = c[3].number_input("Long put strike %", value=80.0, min_value=1.0)
if not (k4p < k3p < k1p < k2p):
    st.warning("Expected ordering: long put % < short put % < long call % < short call %.")

# ---------------------------------------------------------------- build per-month data
md = []
for _, row in months.iterrows():
    fm = dt.datetime.strptime(row["Month"], "%b %Y").date()
    start = dt.datetime.strptime(row["Start"], "%Y-%m-%d").date()
    exp = dt.datetime.strptime(row["Expiry"], "%Y-%m-%d").date()
    fx = bdays(start, exp); tf = np.array([max((d - today).days, 0)/365.0 for d in fx])
    Tx = max(tf[-1], 1/365)
    md.append({"label": row["Month"], "F": float(row["Forward"]), "vol": float(row["Vol %"])/100,
               "fx": fx, "tf": tf, "T": Tx,
               "K": [row["Forward"]*k1p/100, row["Forward"]*k2p/100, row["Forward"]*k3p/100, row["Forward"]*k4p/100]})

def month_cost(m, h):
    F, v, K = m["F"], m["vol"], m["K"]
    legs = [(1,"Call",K[0]),(-1,"Call",K[1]),(-1,"Put",K[2]),(1,"Put",K[3])]
    return sum(s*asian_price(k, F, kk, m["T"], v, r, m["tf"])*h*phys_vol*mult for s,k,kk in legs)

def month_pnl(m, h, A):
    F, K = m["F"], m["K"]
    legs = [(1,"Call",K[0]),(-1,"Call",K[1]),(-1,"Put",K[2]),(1,"Put",K[3])]
    p = sum(s*intrinsic(k, A, kk)*h*phys_vol*mult for s,k,kk in legs)
    return p - month_cost(m, h)

# ---------------------------------------------------------------- solver
st.subheader("3) Solver")
A_grid = np.linspace(0.3*min(m["F"] for m in md), 1.9*max(m["F"] for m in md), 800)
def evaluate(h):
    cost = sum(month_cost(m, h) for m in md)
    net = np.zeros_like(A_grid)
    for m in md:
        net += month_pnl(m, h, A_grid)
    return cost, net.min(), net

def objective(h):
    cost, worst, _ = evaluate(h)
    v_spend = max(0.0, cost - max_spend)
    v_loss = max(0.0, max_loss - worst)
    # normalise violations by target scale so both constraints matter
    scale_s = max(abs(max_spend), 1.0); scale_l = max(abs(max_loss), 1.0)
    viol = (v_spend/scale_s)**2 + (v_loss/scale_l)**2
    if viol > 0:
        return 1e6*viol + cost/scale_s  # nearest feasible: minimise violation, tie-break on cost
    return cost/scale_s  # feasible: minimise spend

if st.button("Solve 4-way strip", type="primary"):
    res = differential_evolution(objective, bounds=[(0.0, 3.0)], seed=1, tol=1e-8, polish=True)
    h = float(res.x[0]); cost, worst, net = evaluate(h)
    st.session_state["sol"] = (h, cost, worst, net)
h_sol = st.session_state.get("sol")
if h_sol:
    h, cost, worst, net = h_sol
    feas = cost <= max_spend and worst >= max_loss
    m = st.columns(5)
    m[0].metric("Solved hedge ratio", f"{h:.0%} of physical")
    m[1].metric("Total premium spend", f"{cost:,.0f} {ccy}")
    m[2].metric("Worst-case net P&L", f"{worst:,.0f} {ccy}")
    m[3].metric("Max spend target", f"{max_spend:,.0f} {ccy}", delta=f"{cost-max_spend:,.0f}", delta_color="inverse")
    m[4].metric("Max loss target", f"{max_loss:,.0f} {ccy}", delta=f"{worst-max_loss:,.0f}", delta_color="normal")
    if feas:
        st.success("Both targets met. This is the lowest-spend hedge ratio that satisfies them.")
    else:
        st.warning("No hedge ratio meets both targets with these strikes. This is the closest solution: "
                   f"spend over by {max(0,cost-max_spend):,.0f}, loss over by {max(0,max_loss-worst):,.0f}. "
                   "Widen the strikes or relax a target.")
else:
    h, cost, worst, net = hedge_pct, sum(month_cost(m, hedge_pct) for m in md), None, None
    st.info("Set targets and press 'Solve 4-way strip'. The charts below currently use the slider hedge ratio.")

# ---------------------------------------------------------------- charts
st.subheader("4) Resulting charts")
fig1 = go.Figure()
for m in md:
    A = np.linspace(m["F"]*0.5, m["F"]*1.5, 400)
    p = month_pnl(m, h, A)
    fig1.add_trace(go.Scatter(x=A, y=p, mode="lines", name=m["label"]))
fig1.add_hline(y=0, line_color="black")
fig1.update_layout(title="Per-month 4-way P&L vs that month's average price",
                   xaxis_title="Average price", yaxis_title=f"P&L ({ccy})", height=450, hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

pct = np.linspace(-50, 90, 500)
agg = np.zeros_like(pct)
for m in md:
    A = m["F"]*(1+pct/100)
    agg += month_pnl(m, h, A)
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

st.subheader("5) Structure per month")
srows = []
for m in md:
    K = m["K"]; c0 = month_cost(m, h)
    srows.append({"Month": m["label"], "Forward": m["F"], "Vol %": m["vol"]*100,
                  "Long call K": K[0], "Short call K": K[1], "Short put K": K[2], "Long put K": K[3],
                  "Volume": h*phys_vol, "Net premium": c0})
st.dataframe(pd.DataFrame(srows).style.format({"Forward": "{:,.2f}", "Vol %": "{:,.1f}", "Long call K": "{:,.2f}",
             "Short call K": "{:,.2f}", "Short put K": "{:,.2f}", "Long put K": "{:,.2f}",
             "Volume": "{:,.0f}", "Net premium": "{:,.0f}"}), use_container_width=True)
st.caption("Arithmetic daily-average options, flat forward curve per month, per-month vol, moment-matched lognormal pricing. "
           "Strikes fixed at user offsets; solver finds hedge ratio closest to meeting spend and loss targets.")
