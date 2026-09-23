"""
Streamlit dashboard for the Personalised Retirement Drawdown model (v2).

Run locally with:
    streamlit run app.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from actuarial_model import (
    AGE_PENSION_AGE,
    AP_MAX_SINGLE,
    LONGEVITY_BUFFER_YEARS,
    STRATEGIES,
    Assumptions,
    asfa_modest,
    asfa_target,
    compare_strategies,
    life_expectancy,
    load_simulated_population,
    margin_sensitivity,
    member_frame,
    run_population_analysis,
)

st.set_page_config(page_title="Retirement Income Dashboard", page_icon="💰", layout="wide")

COLORS = {"minimum": "#8A8F98", "fixed": "#C8553D", "dynamic": "#2A9D8F", "rule": "#1D4E89"}
SHORT = {"minimum": "Minimum drawdown", "fixed": "Fixed 4%", "dynamic": "Dynamic", "rule": "Personalised rule"}
POP_DATA_PATH = Path(__file__).parent / "data" / "simulated_members.csv"


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def smoney(x: float) -> str:
    """Signed money for metric deltas (sign first so the arrow points the right way)."""
    return f"-${-x:,.0f}" if x < 0 else f"+${x:,.0f}"


def md(x: float) -> str:
    """Money for markdown text: escape $ so Streamlit doesn't render it as LaTeX."""
    return money(x).replace("$", "\\$")


def pct(x: float, dp: int = 0) -> str:
    return f"{x * 100:.{dp}f}%"


def base_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
        hovermode="x unified", **kw,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Member profile")
age = st.sidebar.slider("Current age", 60, 85, 65)
gender = st.sidebar.radio("Gender", ["Female", "Male"], horizontal=True)
gender_code = "F" if gender == "Female" else "M"
balance = st.sidebar.number_input(
    "Super balance ($)", min_value=50_000, max_value=3_000_000, value=500_000, step=10_000, format="%d",
)
homeowner = st.sidebar.toggle("Owns their home", value=True,
                              help="Affects the ASFA budget and the Age Pension assets test.")
health_status = st.sidebar.select_slider(
    "Health status", options=["poor", "fair", "good"], value="fair",
    help="Changes mortality for every strategy. The personalised rule also uses it to set its planning horizon.",
)
consumption_level = st.sidebar.radio(
    "Lifestyle target", ["modest", "comfortable"], index=1, horizontal=True,
    help="The ASFA Retirement Standard this member is aiming for.",
)

st.sidebar.header("Strategy settings")
safety_margin = st.sidebar.slider(
    "Safety margin (dynamic strategy)", 1.0, 2.0, 1.2, 0.1,
    help="Plans as if the member lives this many times their life expectancy. Higher = lower, safer income.",
)

with st.sidebar.expander("Model assumptions"):
    ret_mean = st.slider("Expected real return (net of fees)", 0.0, 0.07, 0.035, 0.005, format="%.3f")
    ret_sd = st.slider("Return volatility", 0.0, 0.20, 0.09, 0.01, format="%.2f")
    include_ap = st.toggle("Include Age Pension", value=True)
    enforce_min = st.toggle("Enforce legislated minimum drawdown", value=True)
    n_sim = st.select_slider("Simulation runs", options=[500, 1000, 2000, 5000], value=2000)

st.sidebar.caption(
    "All amounts in today's dollars. Mortality: Australian Life Tables 2020-22 (AGA). "
    "Budgets: ASFA Retirement Standard, March quarter 2026. Age Pension: Services Australia "
    "rates from 20 September 2026."
)

assumptions = Assumptions(
    return_mean=ret_mean, return_sd=ret_sd, planning_return=ret_mean,
    include_age_pension=include_ap, enforce_min_drawdown=enforce_min, safety_margin=safety_margin,
)
member = member_frame(age, gender_code, balance, health_status, consumption_level, homeowner)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_compare(member_key: tuple, a_key: tuple, n: int):
    m = member_frame(*member_key)
    return compare_strategies(m, Assumptions(*a_key), n_sim=n)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_sensitivity(member_key: tuple, a_key: tuple, n: int):
    return margin_sensitivity(member_frame(*member_key), Assumptions(*a_key), n_sim=n)


member_key = (age, gender_code, balance, health_status, consumption_level, homeowner)
a_key = (ret_mean, ret_sd, ret_mean, include_ap, enforce_min, safety_margin)

with st.spinner("Running Monte Carlo simulation..."):
    run = cached_compare(member_key, a_key, n_sim)
summary = run["summary"]
results = run["results"]

target_now = float(asfa_target(np.array([consumption_level]), np.array([age]), np.array([homeowner]))[0])
modest_now = float(asfa_modest(np.array([age]), np.array([homeowner]))[0])

# ---------------------------------------------------------------------------
# Header and key finding
# ---------------------------------------------------------------------------

st.title("Personalised Retirement Drawdown Dashboard")
st.markdown(
    f"How should {'an' if age in (80, 81, 82, 83, 84, 85, 8, 11, 18) else 'a'} **{age}-year-old {gender.lower()}** with **{md(balance)}** in super draw an income? "
    f"We simulate {n_sim:,} possible lifetimes and market paths and test four drawdown strategies against "
    f"the same futures. Their target is the ASFA **{consumption_level}** standard "
    f"({md(target_now)} a year)."
)

rule, base = summary["rule"], summary["minimum"]
st.info(
    f"**Personalised rule vs today's default (minimum drawdown):** covers "
    f"**{pct(rule['target_coverage'])}** of the lifestyle target on average (vs {pct(base['target_coverage'])}), "
    f"and in a typical lifetime the leanest year still pays **{md(rule['median_lowest_income'])}** "
    f"(vs {md(base['median_lowest_income'])}). The chance income ever drops below the ASFA Modest standard "
    f"is **{pct(rule['prob_ever_below_modest'], 1)}** (vs {pct(base['prob_ever_below_modest'], 1)}).",
    icon="🎯",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target coverage", pct(rule["target_coverage"]),
          f"{(rule['target_coverage'] - base['target_coverage']) * 100:+.0f} pts vs default",
          help="Average share of the lifestyle target that income (super + Age Pension) covers each year.")
c2.metric("Average yearly income", money(rule["avg_annual_income"]),
          f"{smoney(rule['avg_annual_income'] - base['avg_annual_income'])} vs default")
c3.metric("Leanest year (typical)", money(rule["median_lowest_income"]),
          f"{smoney(rule['median_lowest_income'] - base['median_lowest_income'])} vs default",
          help="Median across simulations of the lowest income in any single year of retirement.")
c4.metric("Balance left at death (median)", money(rule["median_bequest"]),
          f"{smoney(rule['median_bequest'] - base['median_bequest'])} vs default", delta_color="off",
          help="Unspent super. Higher isn't automatically better: it is income the member didn't get to use.")

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

st.subheader("All four strategies, same futures")
table = pd.DataFrame([
    {
        "Strategy": STRATEGIES[k],
        "Target coverage": s["target_coverage"] * 100,
        "Years on target": s["target_met_share"] * 100,
        "Avg income / yr": s["avg_annual_income"],
        "Leanest year": s["median_lowest_income"],
        "Ever below Modest": s["prob_ever_below_modest"] * 100,
        "Super runs out": s["prob_ran_out"] * 100,
        "Left at death": s["median_bequest"],
    }
    for k, s in summary.items()
])
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "Target coverage": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Years on target": st.column_config.NumberColumn(format="%.0f%%"),
        "Avg income / yr": st.column_config.NumberColumn(format="$%,.0f"),
        "Leanest year": st.column_config.NumberColumn(format="$%,.0f"),
        "Ever below Modest": st.column_config.NumberColumn(format="%.1f%%"),
        "Super runs out": st.column_config.NumberColumn(format="%.1f%%"),
        "Left at death": st.column_config.NumberColumn(format="$%,.0f"),
    },
)
st.caption(
    f"Income = super withdrawals + Age Pension (from age {AGE_PENSION_AGE}, max {md(AP_MAX_SINGLE)}/yr single). "
    f"ASFA Modest: {md(modest_now)}/yr. This member's target: {md(target_now)}/yr. "
    "'Super runs out' means the balance falls below \\$1,000 while the member is alive."
)

# ---------------------------------------------------------------------------
# Charts over the lifetime
# ---------------------------------------------------------------------------

st.subheader("What retirement looks like, year by year")
p = run["paths"]
ages = age + np.arange(p.T)
focus = st.segmented_control(
    "Look closer at", options=list(STRATEGIES), default="rule",
    format_func=lambda k: SHORT[k], key="focus",
) or "rule"


def alive_pctl(mat: np.ndarray, q: float, min_alive: int = 50) -> np.ndarray:
    """Percentile by age among members still alive; hidden once too few remain."""
    alive = np.sum(~np.isnan(mat), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanpercentile(mat, q, axis=0)
    out[alive < min(min_alive, mat.shape[0] // 10)] = np.nan
    return out


# Stop the charts once fewer than 2% of simulated members are still alive:
# beyond that, medians rest on a handful of lives and jump around or drop to zero.
alive_by_age = np.sum(~np.isnan(results["minimum"].income_path), axis=0)
enough = np.nonzero(alive_by_age >= max(20, 0.02 * p.n))[0]
horizon = int(min(p.T, 101 - age, (enough[-1] + 1) if len(enough) else p.T))
tab_inc, tab_bal, tab_paths = st.tabs(["Income", "Super balance", "Sample lifetimes"])

with tab_inc:
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        for k in STRATEGIES:
            fig.add_trace(go.Scatter(
                x=ages[:horizon], y=alive_pctl(results[k].income_path, 50)[:horizon], name=SHORT[k],
                line=dict(color=COLORS[k], width=3.5 if k == focus else 1.8),
            ))
        tgt = asfa_target(np.full(horizon, consumption_level), ages[:horizon], np.full(horizon, homeowner))
        fig.add_trace(go.Scatter(x=ages[:horizon], y=tgt, name="Lifestyle target",
                                 line=dict(color="#222", dash="dash", width=1.2)))
        base_layout(fig, title="Median yearly income, all strategies",
                    xaxis_title="Age", yaxis_title="Income ($ today)", yaxis_tickformat="$,.0f")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        r = results[focus]
        med_sup = alive_pctl(r.super_path, 50)[:horizon]
        med_ap = alive_pctl(r.ap_path, 50)[:horizon]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ages[:horizon], y=med_ap, name="Age Pension", stackgroup="one",
                                 line=dict(width=0), fillcolor="rgba(233,196,106,0.75)"))
        fig.add_trace(go.Scatter(x=ages[:horizon], y=med_sup, name="From super", stackgroup="one",
                                 line=dict(width=0), fillcolor=COLORS[focus]))
        fig.add_trace(go.Scatter(x=ages[:horizon], y=tgt, name="Lifestyle target",
                                 line=dict(color="#222", dash="dash", width=1.2)))
        base_layout(fig, title=f"Where the income comes from: {SHORT[focus]} (median)",
                    xaxis_title="Age", yaxis_title="Income ($ today)", yaxis_tickformat="$,.0f")
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Medians are taken across simulated members still alive at each age. Steps at 75, 80, 85, 90 "
        "and 95 are where the legislated minimum drawdown rate rises; the target dips at 85 because "
        "ASFA's budget for over-85s is lower."
    )

with tab_bal:
    r = results[focus]
    bal_ages = age + np.arange(p.T + 1)
    h1 = horizon + 1
    fig = go.Figure()
    for lo, hi, op in [(10, 90, 0.15), (25, 75, 0.30)]:
        fig.add_trace(go.Scatter(x=bal_ages[:h1], y=alive_pctl(r.balance_path, hi)[:h1],
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bal_ages[:h1], y=alive_pctl(r.balance_path, lo)[:h1], fill="tonexty",
                                 line=dict(width=0), fillcolor=rgba(COLORS[focus], op),
                                 name=f"{lo}th–{hi}th percentile"))
    fig.add_trace(go.Scatter(x=bal_ages[:h1], y=alive_pctl(r.balance_path, 50)[:h1], name="Median",
                             line=dict(color=COLORS[focus], width=3)))
    base_layout(fig, title=f"Super balance range: {SHORT[focus]}",
                xaxis_title="Age", yaxis_title="Balance ($ today)", yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded bands show how wide the range of outcomes is, driven by market returns.")

with tab_paths:
    r = results[focus]
    fig = go.Figure()
    for i in range(40):
        row = r.balance_path[i]
        keep = ~np.isnan(row)
        fig.add_trace(go.Scatter(x=(age + np.arange(p.T + 1))[keep], y=row[keep], mode="lines",
                                 line=dict(color=COLORS[focus], width=1), opacity=0.35,
                                 showlegend=False, hoverinfo="skip"))
    base_layout(fig, title=f"40 simulated members: {SHORT[focus]}", xaxis_title="Age",
                yaxis_title="Balance ($ today)", yaxis_tickformat="$,.0f")
    fig.update_layout(hovermode=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each line is one simulated lifetime, ending at death. {n_sim:,} were simulated in total.")

# ---------------------------------------------------------------------------
# Safety margin trade-off
# ---------------------------------------------------------------------------

st.subheader("The trade-off behind the safety margin")
st.markdown(
    "A larger safety margin plans for a longer life: income is lower and more is left unspent, "
    "but the money is less likely to run out. The personalised rule is shown as a reference point."
)
sens = cached_sensitivity(member_key, a_key, n_sim)
col_l, col_r = st.columns(2)
with col_l:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sens["safety_margin"], y=sens["avg_annual_income"], mode="lines+markers",
                             name="Dynamic: avg income", line=dict(color=COLORS["dynamic"], width=3)))
    fig.add_hline(y=rule["avg_annual_income"], line_dash="dot", line_color=COLORS["rule"],
                  annotation_text="Personalised rule", annotation_position="top left")
    fig.add_vline(x=safety_margin, line_color="#bbb")
    base_layout(fig, 320, title="Average yearly income", xaxis_title="Safety margin",
                yaxis_tickformat="$,.0f")
    st.plotly_chart(fig, width="stretch")
with col_r:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sens["safety_margin"], y=sens["median_bequest"], mode="lines+markers",
                             name="Dynamic: left at death", line=dict(color=COLORS["dynamic"], width=3)))
    fig.add_trace(go.Scatter(x=sens["safety_margin"], y=sens["prob_ran_out"] * 100, mode="lines+markers",
                             name="Dynamic: super runs out (%)", yaxis="y2",
                             line=dict(color=COLORS["fixed"], width=2, dash="dash")))
    fig.add_vline(x=safety_margin, line_color="#bbb")
    base_layout(fig, 320, title="Money left vs risk of running out", xaxis_title="Safety margin",
                yaxis=dict(tickformat="$,.0f"),
                yaxis2=dict(overlaying="y", side="right", ticksuffix="%", tickformat=".1f",
                             rangemode="tozero", showgrid=False))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Population view
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Across our whole target segment")
st.markdown(
    "One member can be cherry-picked. Here the same four strategies are run for **every member in a "
    "simulated population of 1,000 retirees** (built from ABS income and asset distributions), each with "
    "their own age, balance, health and lifestyle target."
)


@st.cache_data(show_spinner=False)
def cached_population(csv_path: str, a_key: tuple):
    return run_population_analysis(load_simulated_population(csv_path), Assumptions(*a_key), n_sim_per_member=200)


if POP_DATA_PATH.exists():
    population = load_simulated_population(POP_DATA_PATH)
    if st.button("Run population analysis", type="primary"):
        with st.spinner(f"Simulating {len(population):,} members × 4 strategies × 200 lifetimes..."):
            st.session_state["pop"] = (a_key, cached_population(str(POP_DATA_PATH), a_key))
    st.caption("Takes a few seconds. Uses the assumptions and safety margin set in the sidebar.")

    if "pop" in st.session_state:
        pop_key, pop = st.session_state["pop"]
        if pop_key != a_key:
            st.warning("Assumptions have changed since this was run. Run it again to update.")

        pop_table = pd.DataFrame([
            {
                "Strategy": STRATEGIES[k],
                "Avg target coverage": pop[f"{k}_target_coverage"].mean() * 100,
                "Members with 90%+ coverage": (pop[f"{k}_target_coverage"] >= 0.9).mean() * 100,
                "Members almost never below Modest": (pop[f"{k}_prob_below_modest"] < 0.05).mean() * 100,
                "Avg chance super runs out": pop[f"{k}_prob_ran_out"].mean() * 100,
                "Avg income / yr": pop[f"{k}_avg_income"].mean(),
            }
            for k in STRATEGIES
        ])
        st.dataframe(
            pop_table, hide_index=True, width="stretch",
            column_config={
                "Avg target coverage": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                "Members with 90%+ coverage": st.column_config.NumberColumn(format="%.0f%%"),
                "Members almost never below Modest": st.column_config.NumberColumn(
                    format="%.0f%%", help="Share of members with under a 5% chance of income ever "
                                          "dropping below the ASFA Modest standard."),
                "Avg chance super runs out": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg income / yr": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

        col_l, col_r = st.columns(2)
        with col_l:
            fig = go.Figure()
            for k in STRATEGIES:
                fig.add_trace(go.Box(y=pop[f"{k}_target_coverage"] * 100, name=SHORT[k],
                                     marker_color=COLORS[k], boxpoints=False))
            base_layout(fig, 360, title="Target coverage per member", yaxis_title="Coverage (%)",
                        showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with col_r:
            by = st.radio("Break down by", ["health_status", "consumption_level"], horizontal=True,
                          format_func=lambda c: {"health_status": "Health", "consumption_level": "Lifestyle"}[c])
            grp = pop.groupby(by)[[f"{k}_target_coverage" for k in STRATEGIES]].mean() * 100
            order = ["poor", "fair", "good"] if by == "health_status" else ["modest", "comfortable"]
            grp = grp.reindex([o for o in order if o in grp.index])
            fig = go.Figure()
            for k in STRATEGIES:
                fig.add_trace(go.Bar(x=grp.index, y=grp[f"{k}_target_coverage"], name=SHORT[k],
                                     marker_color=COLORS[k]))
            base_layout(fig, 320, barmode="group", yaxis_title="Avg coverage (%)",
                        yaxis_range=[max(0, grp.values.min() - 10), 100])
            st.plotly_chart(fig, width="stretch")

        st.download_button("Download member-level results (CSV)", pop.to_csv(index=False).encode("utf-8"),
                           "population_results.csv", "text/csv")
else:
    st.info("Add the simulated population at `data/simulated_members.csv` to enable this section.")

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

with st.expander("📖 Methodology, assumptions and data sources"):
    st.markdown(f"""
**The four strategies**

- **Minimum drawdown (status quo):** the legislated minimum rate for the member's age (5% at 65–74,
  rising to 14% at 95+). Most retirees in account-based pensions draw close to this.
- **Fixed 4% rule:** 4% of the starting balance every year, constant in today's dollars.
- **Dynamic personalised:** each year, the balance is spread evenly over *life expectancy × safety margin*
  years, allowing for expected investment returns (an annuity-factor formula).
- **Personalised rule (need vs sustainable):** each year, works out what the member needs from super to
  reach their ASFA target *after* their Age Pension, and pays that, capped at a sustainable rate based on
  health-adjusted life expectancy plus a {LONGEVITY_BUFFER_YEARS}-year longevity buffer. Before age {AGE_PENSION_AGE} it also
  bridges the Age Pension the member is expected to receive (using no more than a quarter of the
  balance in total), so income doesn't dip and then jump.

Where the minimum-drawdown switch is on, every strategy pays at least the legislated minimum.

**How outcomes are measured**

- *Target coverage:* the average share of the lifestyle target that income covers each year (capped at 100%).
- *Leanest year:* the lowest income in any single year of a simulated lifetime (median across simulations).
- *Ever below Modest:* chance that income falls below the ASFA Modest standard in at least one year.
- *Super runs out:* chance the balance falls below \\$1,000 while the member is still alive.

**Simulation**

- Lifetimes are simulated from the Australian Life Tables 2020-22, with mortality scaled by health
  (good ×0.85, fair ×1.00, poor ×1.50). Returns are normally distributed, real and net of fees.
- *Common random numbers:* every strategy faces exactly the same simulated lifetimes and market returns,
  so differences between strategies come from the strategy alone.
- The Age Pension is modelled for a single person with both the assets test and the income test (deeming),
  using rates from 20 September 2026. Thresholds are assumed to keep pace with inflation.

**Simplifications**

Singles only; no tax (most over-60 pension income is tax-free); no other income or assets outside super;
no aged-care costs; ASFA budgets for renters use ASFA's single renter figure for both lifestyle levels.
""")

st.caption(
    "Built for the Actuaries Institute Innovation Challenge 2026. A simplified, illustrative model; "
    "not financial advice."
)
