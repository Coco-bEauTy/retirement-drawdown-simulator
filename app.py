"""
Streamlit dashboard for the Personalised Retirement Drawdown model.

Run locally with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from actuarial_model import (
    ASFA_COMFORTABLE_SINGLE,
    ASFA_MODEST_SINGLE,
    DynamicPersonalisedStrategy,
    FixedDollarStrategy,
    PersonalisedRuleStrategy,
    compare_strategies,
    generate_random_streams,
    get_life_table,
    load_simulated_population,
    run_population_analysis,
    run_simulation,
)
from pathlib import Path

st.set_page_config(
    page_title="Retirement Income Dashboard",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — member inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Member profile")

age = st.sidebar.slider("Current age", min_value=60, max_value=85, value=65)
gender = st.sidebar.radio("Gender", options=["Female", "Male"], horizontal=True)
gender_code = "F" if gender == "Female" else "M"
balance = st.sidebar.number_input(
    "Account balance ($)", min_value=50_000, max_value=3_000_000,
    value=500_000, step=10_000, format="%d",
)
safety_margin = st.sidebar.slider(
    "Risk tolerance (safety margin)", min_value=1.0, max_value=1.8, value=1.2, step=0.1,
    help="Used by the 'Dynamic Personalised' strategy. Higher = more "
         "conservative withdrawals, planning to live longer than average.",
)
health_status = st.sidebar.select_slider(
    "Health status", options=["poor", "fair", "good"], value="fair",
    help="Used by the 'Personalised Rule' strategy - affects both the "
         "planning horizon and realised mortality risk.",
)
consumption_level = st.sidebar.radio(
    "Lifestyle target", options=["modest", "comfortable"], index=1, horizontal=True,
    help="Used by the 'Personalised Rule' strategy - the ASFA living "
         "standard this member is aiming to maintain.",
)
n_sim = st.sidebar.select_slider(
    "Simulation runs", options=[500, 1000, 2000, 5000], value=2000,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Mortality data: Australian Life Tables 2020-22 (Australian Government "
    "Actuary). Retirement cost benchmarks: ASFA Retirement Standard, "
    "March quarter 2026."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💰 Personalised Retirement Drawdown Dashboard")
st.markdown(
    "Comparing three retirement withdrawal strategies: a traditional "
    "**fixed 4% rule**, a **dynamic life-expectancy-based** strategy, and a "
    "**rule-based personalised** strategy that reconciles a member's "
    "lifestyle *need* against what their balance can *sustainably* support "
    "— built on Australian Government Actuary mortality data and the ASFA "
    "Retirement Standard."
)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

with st.spinner("Running Monte Carlo simulation..."):
    result = compare_strategies(
        age=age, gender=gender_code, balance=balance,
        safety_margin=safety_margin, health_status=health_status,
        consumption_level=consumption_level, n_sim=n_sim, seed=42,
    )

fixed_summary = result["fixed_summary"]
dynamic_summary = result["dynamic_summary"]
rule_summary = result["rule_based_summary"]

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

st.subheader("Key outcomes")

metrics_df = pd.DataFrame([
    {
        "Strategy": "Fixed 4% Rule",
        "Prob. money runs out before death": f"{fixed_summary['prob_ran_out_before_death']*100:.1f}%",
        "Prob. income below Modest standard": f"{fixed_summary['prob_income_shortfall_before_death']*100:.1f}%",
    },
    {
        "Strategy": "Dynamic Personalised (life expectancy × margin)",
        "Prob. money runs out before death": f"{dynamic_summary['prob_ran_out_before_death']*100:.1f}%",
        "Prob. income below Modest standard": f"{dynamic_summary['prob_income_shortfall_before_death']*100:.1f}%",
    },
    {
        "Strategy": "Personalised Rule (need vs sustainable)",
        "Prob. money runs out before death": f"{rule_summary['prob_ran_out_before_death']*100:.1f}%",
        "Prob. income below Modest standard": f"{rule_summary['prob_income_shortfall_before_death']*100:.1f}%",
    },
])
st.dataframe(metrics_df, hide_index=True, use_container_width=True)

st.caption(
    f"ASFA Modest Living Standard (single): ${ASFA_MODEST_SINGLE:,}/year · "
    f"ASFA Comfortable Living Standard (single): ${ASFA_COMFORTABLE_SINGLE:,}/year · "
    f"this member's lifestyle target: **{consumption_level}**, health status: **{health_status}**"
)

# ---------------------------------------------------------------------------
# Balance trajectory chart — sample of simulated paths
# ---------------------------------------------------------------------------

st.subheader("Sample simulated balance trajectories")

n_paths_to_show = 40


def simulate_paths_for_display(gender_code, balance, strategy, streams, n_show):
    """Re-run a small number of paths individually so we can capture and
    plot the year-by-year balance trajectory (the vectorised engine only
    returns final outcomes, not full paths, for performance)."""
    table = get_life_table(gender_code)
    qx_by_age = table["qx"]
    start_age = streams.start_age
    n_years = streams.U.shape[1]

    paths = []
    for i in range(min(n_show, streams.U.shape[0])):
        bal = balance
        ages = [start_age]
        balances = [bal]
        for t in range(n_years):
            age_t = start_age + t
            if age_t not in qx_by_age.index:
                break
            qx = qx_by_age.loc[age_t]
            if streams.U[i, t] < qx:
                break
            withdrawal = min(strategy.withdrawal(bal, age_t), bal)
            bal = max((bal - withdrawal) * (1 + streams.R[i, t]), 0)
            ages.append(age_t + 1)
            balances.append(bal)
            if bal <= 0:
                break
        paths.append((ages, balances))
    return paths


fixed_strategy = FixedDollarStrategy(initial_balance=balance, withdrawal_rate=0.04)
dynamic_strategy = DynamicPersonalisedStrategy(gender=gender_code, safety_margin=safety_margin)
rule_strategy = PersonalisedRuleStrategy(
    gender=gender_code, health_status=health_status, consumption_level=consumption_level
)

fixed_paths = simulate_paths_for_display(gender_code, balance, fixed_strategy, result["streams"], n_paths_to_show)
dynamic_paths = simulate_paths_for_display(gender_code, balance, dynamic_strategy, result["streams"], n_paths_to_show)
rule_paths = simulate_paths_for_display(gender_code, balance, rule_strategy, result["streams"], n_paths_to_show)

tab1, tab2, tab3 = st.tabs(["Fixed 4% Rule", "Dynamic Personalised", "Personalised Rule"])

for tab, paths, color in [
    (tab1, fixed_paths, "firebrick"),
    (tab2, dynamic_paths, "seagreen"),
    (tab3, rule_paths, "steelblue"),
]:
    with tab:
        fig = go.Figure()
        for ages, balances in paths:
            fig.add_trace(go.Scatter(
                x=ages, y=balances, mode="lines",
                line=dict(color=color, width=1), opacity=0.25,
                showlegend=False, hoverinfo="skip",
            ))
        fig.update_layout(
            xaxis_title="Age", yaxis_title="Account balance ($)",
            height=400, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(paths)} simulated members shown (out of {n_sim} total simulations)")

# ---------------------------------------------------------------------------
# Safety margin sensitivity
# ---------------------------------------------------------------------------

st.subheader("How does risk tolerance affect outcomes?")

margin_values = [1.0, 1.2, 1.4, 1.6, 1.8]
sensitivity_rows = []
for m in margin_values:
    strat = DynamicPersonalisedStrategy(gender=gender_code, safety_margin=m)
    res = run_simulation(gender_code, balance, strat, result["streams"])
    s = res.summary()
    sensitivity_rows.append({
        "Safety margin": m,
        "Prob. ran out before death (%)": round(s["prob_ran_out_before_death"] * 100, 2),
        "Prob. income shortfall before death (%)": round(s["prob_income_shortfall_before_death"] * 100, 2),
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)

col_chart, col_table = st.columns([2, 1])
with col_chart:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sensitivity_df["Safety margin"],
        y=sensitivity_df["Prob. income shortfall before death (%)"],
        mode="lines+markers", name="Income shortfall risk",
    ))
    fig2.add_hline(
        y=fixed_summary["prob_income_shortfall_before_death"] * 100,
        line_dash="dash", line_color="firebrick",
        annotation_text="Fixed 4% Rule benchmark",
    )
    fig2.update_layout(
        xaxis_title="Safety margin", yaxis_title="Probability (%)",
        height=350, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
with col_table:
    st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Population-level analysis — using the simulated member dataset
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Target segment analysis")
st.markdown(
    "The single-member view above is useful for exploring one scenario at a "
    "time. The analysis below runs the same model across a **simulated "
    "population of 1,000 members**, generated from real ABS income/asset "
    "distributions and AGA mortality data, to show outcomes across our "
    "entire target segment rather than one hand-picked example."
)

POP_DATA_PATH = Path(__file__).parent / "data" / "simulated_members.csv"

if POP_DATA_PATH.exists():
    population = load_simulated_population(POP_DATA_PATH)

    with st.expander(f"Preview simulated population ({len(population)} members)"):
        st.dataframe(population.head(20), hide_index=True, use_container_width=True)

    pop_col1, pop_col2 = st.columns([1, 3])
    with pop_col1:
        run_pop = st.button("Run population analysis", type="primary")
        st.caption(
            f"Runs the model for all {len(population)} simulated members "
            "(uses the risk tolerance slider above). Takes about a minute."
        )

    @st.cache_data(show_spinner=False)
    def _cached_population_analysis(csv_path: str, margin: float, n_members: int):
        pop = load_simulated_population(csv_path)
        return run_population_analysis(pop, safety_margin=margin, n_sim_per_member=200)

    if run_pop:
        with st.spinner(f"Simulating {len(population)} members × 2 strategies × 200 runs each..."):
            pop_results = _cached_population_analysis(
                str(POP_DATA_PATH), safety_margin, len(population)
            )
        st.session_state["pop_results"] = pop_results

    if "pop_results" in st.session_state:
        pop_results = st.session_state["pop_results"]

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(
            "Members with <10% chance of running out (fixed 4%)",
            f"{(pop_results['fixed_prob_ran_out'] < 0.10).mean()*100:.0f}%",
        )
        pc2.metric(
            "Members with <10% chance of running out (dynamic)",
            f"{(pop_results['dynamic_prob_ran_out'] < 0.10).mean()*100:.0f}%",
        )
        pc3.metric(
            "Members with <10% chance of running out (personalised rule)",
            f"{(pop_results['rule_prob_ran_out'] < 0.10).mean()*100:.0f}%",
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=pop_results["fixed_income_shortfall"] * 100, name="Fixed 4% Rule",
            opacity=0.55, marker_color="firebrick", nbinsx=25,
        ))
        fig3.add_trace(go.Histogram(
            x=pop_results["dynamic_income_shortfall"] * 100, name="Dynamic Personalised",
            opacity=0.55, marker_color="seagreen", nbinsx=25,
        ))
        fig3.add_trace(go.Histogram(
            x=pop_results["rule_income_shortfall"] * 100, name="Personalised Rule",
            opacity=0.55, marker_color="steelblue", nbinsx=25,
        ))
        fig3.update_layout(
            barmode="overlay",
            xaxis_title="Probability of income shortfall before death (%)",
            yaxis_title="Number of members",
            height=350, margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

        if "health_status" in pop_results.columns:
            st.markdown("**Outcomes by health status (Personalised Rule strategy)**")
            health_summary = pop_results.groupby("health_status")[
                ["rule_prob_ran_out", "rule_income_shortfall"]
            ].mean().reset_index()
            health_summary.columns = [
                "Health status", "Avg. prob. ran out", "Avg. prob. income shortfall"
            ]
            st.dataframe(health_summary, hide_index=True, use_container_width=True)

        if "consumption_level" in pop_results.columns:
            st.markdown("**Outcomes by lifestyle target (Personalised Rule strategy)**")
            consumption_summary = pop_results.groupby("consumption_level")[
                ["rule_prob_ran_out", "rule_income_shortfall"]
            ].mean().reset_index()
            consumption_summary.columns = [
                "Lifestyle target", "Avg. prob. ran out", "Avg. prob. income shortfall"
            ]
            st.dataframe(consumption_summary, hide_index=True, use_container_width=True)

        st.download_button(
            "Download full population results (CSV)",
            data=pop_results.to_csv(index=False).encode("utf-8"),
            file_name="population_results.csv",
            mime="text/csv",
        )
else:
    st.info(
        "No simulated population file found at `data/simulated_members.csv`. "
        "Add your teammate's generated dataset there to enable this section."
    )



with st.expander("📖 Methodology & data sources"):
    st.markdown("""
    **Strategies compared**
    - **Fixed 4% Rule**: withdraws a fixed dollar amount every year, equal to
      4% of the *starting* balance — the industry-standard benchmark.
    - **Dynamic Personalised**: withdraws `current balance ÷ (life expectancy
      at current age × safety margin)` every year, automatically adjusting
      as the member ages and as markets move.
    - **Personalised Rule (need vs sustainable)**: every year, reconciles
      two sides — how much this member *needs* to maintain their chosen
      lifestyle (the ASFA modest/comfortable benchmark for their age band),
      against the maximum their balance can *sustainably* support (an
      amortisation formula using health-adjusted life expectancy plus a
      5-year longevity buffer). The withdrawal is the need, capped at the
      sustainable ceiling — members whose need exceeds what's sustainable
      are flagged as relying on a supplementary income source (e.g. the Age
      Pension), rather than silently overspending. Health status also
      adjusts this member's realised mortality risk, not just their
      withdrawal rate.

    **Two risk metrics tracked**
    1. *Probability of running out of money before death* — the account
       balance hits exactly $0 while the member is still alive.
    2. *Probability of income shortfall before death* — the annual withdrawal
       amount falls below the ASFA Modest Living Standard, even if the
       account isn't literally empty. A percentage-of-balance strategy can
       never mathematically hit exact $0, so this second metric captures a
       more realistic risk of an inadequate — but not zero — income.

    **Simulation method**: Monte Carlo simulation using *Common Random
    Numbers* — the same simulated set of members (same simulated lifespans,
    same simulated investment returns) is used to test every strategy,
    ensuring a fair, paired comparison rather than comparing independently
    drawn random samples.

    **Data sources**
    - Mortality rates and life expectancy: *Australian Life Tables 2020-22*,
      Australian Government Actuary.
    - Retirement cost benchmarks: *ASFA Retirement Standard*, March quarter
      2026 (by household type, home ownership, and age band).
    """)

st.markdown("---")
st.caption(
    "Built for the Actuaries Institute Innovation Challenge 2026. "
    "This is a simplified illustrative model and does not constitute "
    "financial advice."
)
