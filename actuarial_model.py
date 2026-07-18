"""
Actuarial Model — Personalised Retirement Drawdown Simulation
================================================================
Core Monte Carlo engine comparing a fixed-dollar withdrawal strategy
(the classic industry "4% rule") against a dynamic, personalised
withdrawal strategy based on life expectancy and a member-chosen
safety margin.

Data sources:
- Mortality rates (qx) and life expectancy (ex): Australian Life
  Tables 2020-22, Australian Government Actuary (AGA)
- Retirement cost benchmarks: ASFA Retirement Standard, March
  quarter 2026

Methodology notes:
- Uses "Common Random Numbers" (a variance reduction technique):
  the same simulated set of members (same simulated lifespans, same
  simulated market returns) is used to test every strategy being
  compared, giving a fair, paired comparison instead of comparing
  independently-drawn random samples.
- Tracks TWO distinct risk metrics rather than one, because they
  capture genuinely different things:
    1. Probability of running out of money before death (literal
       depletion of the account to $0)
    2. Probability of income shortfall before death (annual
       withdrawal falls below the ASFA Modest Living Standard,
       even if the account isn't literally empty)
  A withdrawal strategy that takes a fixed PERCENTAGE of the current
  balance each year can mathematically never fully deplete an
  account (it only approaches zero asymptotically), so metric (2) is
  essential to capture the more realistic risk of an inadequate
  income even when the account technically still has a balance.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# ASFA Retirement Standard benchmarks (March quarter 2026, single, homeowner,
# aged 65-84) — flat benchmarks used by the actuarial team's own metrics
# ---------------------------------------------------------------------------
ASFA_COMFORTABLE_SINGLE = 55_923   # per year
ASFA_MODEST_SINGLE = 36_434        # per year
ASFA_COMFORTABLE_COUPLE = 78_566
ASFA_MODEST_COUPLE = 52_473

# ---------------------------------------------------------------------------
# Full ASFA table by household / homeowner status / age band — used by the
# rule-based personalised strategy (contributed by the DS teammate), which
# needs to look up a member's *specific* annual cost of living rather than
# a single flat benchmark.
# ---------------------------------------------------------------------------
ASFA_ANNUAL_TABLE = {
    ("single", True, "65-84"): {"comfortable": 55_923, "modest": 36_434},
    ("single", True, "85+"):   {"comfortable": 53_656, "modest": 34_374},
    ("couple", True, "65-84"): {"comfortable": 78_566, "modest": 52_473},
    ("couple", True, "85+"):   {"comfortable": 73_970, "modest": 49_255},
    # Renters: ASFA only publishes a single "modest" figure, reused for both tiers.
    ("single", False, "65-84"): {"comfortable": 51_164, "modest": 51_164},
    ("couple", False, "65-84"): {"comfortable": 69_002, "modest": 69_002},
}
DEFAULT_HOUSEHOLD = "single"
DEFAULT_HOMEOWNER = True

# Health status -> mortality (qx) multiplier: poorer health means a higher
# chance of death each year than the population-average life table implies.
HEALTH_QX_MULT = {"good": 0.85, "fair": 1.00, "poor": 1.50}
# Health status -> life-expectancy / planning-horizon adjustment: poor health
# shortens the realistic planning horizon, which supports drawing a higher
# (still actuarially reasoned) withdrawal rate.
HEALTH_HORIZON_MULT = {"good": 1.05, "fair": 1.00, "poor": 0.80}

LONGEVITY_BUFFER_YEARS = 5   # plan to "life expectancy + buffer" years, to
                              # hedge against the risk of outliving the average
RULE_REAL_RETURN = 0.035     # assumed real (after-inflation) return used only
                              # to derive the amortisation-based sustainable
                              # withdrawal rate below (a planning assumption,
                              # distinct from the Monte Carlo's simulated
                              # investment returns)
RATE_FLOOR = 0.02
RATE_CAP = 0.10


# ---------------------------------------------------------------------------
# 1. Mortality data loading
# ---------------------------------------------------------------------------

def load_life_table(gender: str) -> pd.DataFrame:
    """Load and clean the AGA Australian Life Table 2020-22 for the given
    gender ("M" or "F"). Returns a DataFrame indexed by Age with columns
    qx (probability of death within the year) and ex (life expectancy)."""
    filename = "ALT_2020-22_Males.csv" if gender == "M" else "ALT_2020-22_Females.csv"
    df = pd.read_csv(DATA_DIR / filename, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    def clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(",", "").str.strip(), errors="coerce"
        )

    df["Age"] = clean_numeric(df["Age"]).astype(int)
    df["qx"] = clean_numeric(df["qx"])
    df["ex"] = clean_numeric(df["e?x"])
    return df.set_index("Age")[["qx", "ex"]]


_LIFE_TABLES = {"M": load_life_table("M"), "F": load_life_table("F")}


def get_life_table(gender: str) -> pd.DataFrame:
    return _LIFE_TABLES[gender]


# ---------------------------------------------------------------------------
# 2. Withdrawal strategies
# ---------------------------------------------------------------------------

@dataclass
class FixedDollarStrategy:
    """The classic '4% rule': withdraw a fixed dollar amount each year,
    equal to `withdrawal_rate` of the STARTING balance. Does not adjust
    for market performance or life expectancy. Can genuinely deplete
    the account if markets underperform."""
    initial_balance: float
    withdrawal_rate: float = 0.04
    name: str = "Fixed 4% Rule"

    def withdrawal(self, balance: float, age: int) -> float:
        return self.initial_balance * self.withdrawal_rate


@dataclass
class DynamicPersonalisedStrategy:
    """Withdraws current_balance / (life_expectancy_at_age * safety_margin).
    Naturally adjusts every year as the member ages and as the balance
    changes with market performance. `safety_margin` represents the
    member's personal buffer against longevity risk (1.0 = plan to the
    average life expectancy exactly; 1.2 = plan as if living 20% longer
    than expected, i.e. more conservative)."""
    gender: str
    safety_margin: float = 1.2
    name: str = "Dynamic Personalised"

    def __post_init__(self):
        self._table = get_life_table(self.gender)

    def withdrawal(self, balance: float, age: int) -> float:
        ex = self._table["ex"].get(age, 5.0)
        if pd.isna(ex) or ex <= 0:
            ex = 5.0
        horizon = ex * self.safety_margin
        return balance / horizon


# --- Contributed by DS teammate: needs-based vs sustainable-rate reconciliation ---

def age_band(age: int) -> str:
    return "65-84" if age < 85 else "85+"


def asfa_need(
    consumption_level: str,
    age: int,
    household: str = DEFAULT_HOUSEHOLD,
    homeowner: bool = DEFAULT_HOMEOWNER,
) -> float:
    """Look up this member's ASFA annual cost-of-living requirement ($),
    based on their chosen lifestyle (modest/comfortable), age band, and
    household type."""
    key = (household, homeowner, age_band(age))
    row = ASFA_ANNUAL_TABLE.get(key)
    if row is None:
        row = ASFA_ANNUAL_TABLE[(DEFAULT_HOUSEHOLD, DEFAULT_HOMEOWNER, age_band(age))]
    return row.get(consumption_level, row["modest"])


def planning_horizon(gender: str, age: int, health_status: str) -> float:
    """Planning horizon = life expectancy (health-adjusted) + longevity buffer."""
    table = get_life_table(gender)
    ex = table["ex"].get(age, 5.0)
    if pd.isna(ex) or ex <= 0:
        ex = 5.0
    ex_adj = ex * HEALTH_HORIZON_MULT.get(health_status, 1.0)
    return max(1.0, ex_adj + LONGEVITY_BUFFER_YEARS)


def sustainable_rate(gender: str, age: int, health_status: str,
                      real_return: float = RULE_REAL_RETURN) -> float:
    """Sustainable withdrawal rate (%): the amortisation/annuity payout
    factor that draws the balance down evenly over the planning horizon n.
        rate = r / (1 - (1+r)^-n)        (degenerates to 1/n when r = 0)
    A shorter horizon (older age / poorer health) supports a higher rate;
    a longer horizon (younger / better health) is more conservative."""
    n = planning_horizon(gender, age, health_status)
    r = real_return
    rate = (1.0 / n) if abs(r) < 1e-9 else r / (1.0 - (1.0 + r) ** (-n))
    return float(np.clip(rate, RATE_FLOOR, RATE_CAP))


@dataclass
class PersonalisedRuleStrategy:
    """Rule-based personalised withdrawal strategy (contributed by the DS
    teammate). Every year, reconciles two sides:
      - NEED side: how much this member needs to maintain their chosen
        lifestyle (ASFA modest/comfortable benchmark for their age band)
      - SUSTAINABLE side: the maximum the account can safely support,
        via an amortisation formula based on health-adjusted life
        expectancy plus a longevity buffer
    The withdrawal is the need, capped at the sustainable ceiling. When the
    need exceeds what's sustainable, the member is flagged as relying on a
    supplementary income source (e.g. the Age Pension) to fully meet their
    target lifestyle - the model does not silently overspend."""
    gender: str
    health_status: str = "fair"          # "good" / "fair" / "poor"
    consumption_level: str = "comfortable"  # "modest" / "comfortable"
    name: str = "Personalised Rule (Need vs Sustainable)"

    def withdrawal(self, balance: float, age: int) -> float:
        s_rate = sustainable_rate(self.gender, age, self.health_status)
        need_dollar = asfa_need(self.consumption_level, age)
        sustainable_dollar = s_rate * balance
        # Draw what's needed, but never more than the account can sustain
        return min(need_dollar, sustainable_dollar) if balance > 0 else 0.0

    @property
    def qx_multiplier(self) -> float:
        """Health status also affects realised mortality risk, not just the
        withdrawal rate - poorer health means a higher chance of death each
        year than the population-average life table implies."""
        return HEALTH_QX_MULT.get(self.health_status, 1.0)


# ---------------------------------------------------------------------------
# 3. Common random numbers — shared simulation inputs across strategies
# ---------------------------------------------------------------------------

@dataclass
class RandomStreams:
    """Pre-generated random numbers used to determine, for each simulated
    member and each year: (a) whether they die that year, and (b) that
    year's investment return. Generating these ONCE and reusing them for
    every strategy under comparison ensures a fair, paired comparison
    (the 'Common Random Numbers' variance reduction technique)."""
    U: np.ndarray   # shape (n_sim, n_years) — mortality draws
    R: np.ndarray   # shape (n_sim, n_years) — investment return draws
    start_age: int
    max_age: int


def generate_random_streams(
    n_sim: int,
    start_age: int,
    max_age: int = 100,
    return_mean: float = 0.05,
    return_sd: float = 0.08,
    rng: np.random.Generator | None = None,
) -> RandomStreams:
    rng = rng or np.random.default_rng()
    n_years = max_age - start_age
    U = rng.random((n_sim, n_years))
    R = rng.normal(return_mean, return_sd, size=(n_sim, n_years))
    return RandomStreams(U=U, R=R, start_age=start_age, max_age=max_age)


# ---------------------------------------------------------------------------
# 4. Core simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    death_age: np.ndarray
    depletion_age: np.ndarray          # NaN if balance never hit $0
    income_shortfall_age: np.ndarray   # NaN if income never fell below ASFA floor
    ending_balance: np.ndarray

    @property
    def ran_out_before_death(self) -> np.ndarray:
        return (~np.isnan(self.depletion_age)) & (self.depletion_age < self.death_age)

    @property
    def income_shortfall_before_death(self) -> np.ndarray:
        return (~np.isnan(self.income_shortfall_age)) & (self.income_shortfall_age < self.death_age)

    def summary(self) -> dict:
        return {
            "prob_ran_out_before_death": float(np.mean(self.ran_out_before_death)),
            "prob_income_shortfall_before_death": float(np.mean(self.income_shortfall_before_death)),
            "mean_death_age": float(np.mean(self.death_age)),
            "median_ending_balance": float(np.median(self.ending_balance)),
        }


def run_simulation(
    gender: str,
    balance: float,
    strategy,
    streams: RandomStreams,
    income_floor: float = ASFA_MODEST_SINGLE,
    qx_multiplier: float = 1.0,
) -> SimulationResult:
    """Vectorised-ish Monte Carlo simulation of a retirement drawdown.
    Applies `strategy` to every simulated member represented by `streams`.

    qx_multiplier scales the base population mortality rate (e.g. a
    PersonalisedRuleStrategy with health_status="poor" implies a higher
    realised mortality risk than the population average -- pass its
    `.qx_multiplier` property here for a fair, health-adjusted comparison).
    """
    table = get_life_table(gender)
    qx_by_age = table["qx"]

    n_sim, n_years = streams.U.shape
    start_age = streams.start_age

    death_age = np.full(n_sim, streams.max_age, dtype=float)
    depletion_age = np.full(n_sim, np.nan)
    income_shortfall_age = np.full(n_sim, np.nan)
    bal = np.full(n_sim, float(balance))
    alive = np.ones(n_sim, dtype=bool)

    for t in range(n_years):
        age_t = start_age + t
        if age_t not in qx_by_age.index:
            break
        qx = min(qx_by_age.loc[age_t] * qx_multiplier, 1.0)

        dies_this_year = alive & (streams.U[:, t] < qx)
        death_age[dies_this_year] = age_t
        alive &= ~dies_this_year

        active = alive & (bal > 0)
        if active.any():
            idx = np.where(active)[0]
            withdrawals = np.array([strategy.withdrawal(bal[i], age_t) for i in idx])
            withdrawals = np.minimum(withdrawals, bal[idx])

            shortfall_now = withdrawals < income_floor
            newly_short = idx[shortfall_now & np.isnan(income_shortfall_age[idx])]
            income_shortfall_age[newly_short] = age_t

            bal[idx] = bal[idx] - withdrawals
            bal[idx] = bal[idx] * (1 + streams.R[idx, t])
            bal[idx] = np.maximum(bal[idx], 0)

            newly_depleted = idx[(bal[idx] <= 0) & np.isnan(depletion_age[idx])]
            depletion_age[newly_depleted] = age_t + 1

        # members already depleted (bal == 0) get income $0, definitely a shortfall
        depleted_alive = alive & (bal <= 0)
        newly_short2 = np.where(depleted_alive & np.isnan(income_shortfall_age))[0]
        income_shortfall_age[newly_short2] = age_t

        if not alive.any():
            break

    return SimulationResult(
        death_age=death_age,
        depletion_age=depletion_age,
        income_shortfall_age=income_shortfall_age,
        ending_balance=bal,
    )


# ---------------------------------------------------------------------------
# 5. Convenience wrapper for a full strategy comparison on one scenario
# ---------------------------------------------------------------------------

def compare_strategies(
    age: int,
    gender: str,
    balance: float,
    safety_margin: float = 1.2,
    health_status: str = "fair",
    consumption_level: str = "comfortable",
    n_sim: int = 2000,
    seed: int | None = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    streams = generate_random_streams(n_sim=n_sim, start_age=age, rng=rng)

    fixed = FixedDollarStrategy(initial_balance=balance, withdrawal_rate=0.04)
    dynamic = DynamicPersonalisedStrategy(gender=gender, safety_margin=safety_margin)
    rule_based = PersonalisedRuleStrategy(
        gender=gender, health_status=health_status, consumption_level=consumption_level
    )

    result_fixed = run_simulation(gender, balance, fixed, streams)
    result_dynamic = run_simulation(gender, balance, dynamic, streams)
    result_rule = run_simulation(
        gender, balance, rule_based, streams, qx_multiplier=rule_based.qx_multiplier
    )

    return {
        "fixed": result_fixed,
        "dynamic": result_dynamic,
        "rule_based": result_rule,
        "fixed_summary": result_fixed.summary(),
        "dynamic_summary": result_dynamic.summary(),
        "rule_based_summary": result_rule.summary(),
        "streams": streams,
    }


# ---------------------------------------------------------------------------
# 6. Population-level analysis — run the model across a full member dataset
# ---------------------------------------------------------------------------

def load_simulated_population(csv_path: str | Path) -> pd.DataFrame:
    """Load a batch of simulated members generated from real ABS/ASFA
    statistical distributions (produced by a teammate's data generation
    script). Expected columns: age, gender, balance (health_status and
    consumption_level are optional and not yet used by the drawdown model,
    but are retained for future personalisation)."""
    df = pd.read_csv(csv_path)
    required = {"age", "gender", "balance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Simulated population file is missing columns: {missing}")
    return df


def run_population_analysis(
    population: pd.DataFrame,
    safety_margin: float = 1.2,
    n_sim_per_member: int = 200,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Run all three withdrawal strategies for every member in a simulated
    population and return a DataFrame of per-member outcome probabilities.
    Where available, each member's own health_status and consumption_level
    columns are used to drive the rule-based personalised strategy (and its
    health-adjusted mortality); members without these columns default to
    "fair" health and a "comfortable" lifestyle target.

    n_sim_per_member is kept modest (e.g. 200) by default because this
    function runs the full simulation once per member — with a population
    of ~1,000 members, that is already up to 200,000 simulated life paths
    per strategy.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _, member in population.iterrows():
        streams = generate_random_streams(
            n_sim=n_sim_per_member, start_age=int(member["age"]), rng=rng
        )
        health_status = member.get("health_status", "fair")
        consumption_level = member.get("consumption_level", "comfortable")

        fixed = FixedDollarStrategy(initial_balance=member["balance"], withdrawal_rate=0.04)
        dynamic = DynamicPersonalisedStrategy(gender=member["gender"], safety_margin=safety_margin)
        rule_based = PersonalisedRuleStrategy(
            gender=member["gender"], health_status=health_status, consumption_level=consumption_level
        )

        res_fixed = run_simulation(member["gender"], member["balance"], fixed, streams)
        res_dynamic = run_simulation(member["gender"], member["balance"], dynamic, streams)
        res_rule = run_simulation(
            member["gender"], member["balance"], rule_based, streams,
            qx_multiplier=rule_based.qx_multiplier,
        )

        row = {
            "member_id": member.get("member_id"),
            "age": member["age"],
            "gender": member["gender"],
            "balance": member["balance"],
            "fixed_prob_ran_out": res_fixed.summary()["prob_ran_out_before_death"],
            "dynamic_prob_ran_out": res_dynamic.summary()["prob_ran_out_before_death"],
            "rule_prob_ran_out": res_rule.summary()["prob_ran_out_before_death"],
            "fixed_income_shortfall": res_fixed.summary()["prob_income_shortfall_before_death"],
            "dynamic_income_shortfall": res_dynamic.summary()["prob_income_shortfall_before_death"],
            "rule_income_shortfall": res_rule.summary()["prob_income_shortfall_before_death"],
        }
        for optional_col in ("health_status", "consumption_level"):
            if optional_col in member:
                row[optional_col] = member[optional_col]
        rows.append(row)

    return pd.DataFrame(rows)
