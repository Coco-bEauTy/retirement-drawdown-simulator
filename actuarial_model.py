"""
Actuarial Model — Personalised Retirement Drawdown Simulation (v2)
==================================================================
Monte Carlo engine comparing four ways of drawing down an account-based
pension (ABP):

  0. Minimum drawdown (status quo) — the legislated minimum rate, which is
     what most Australian retirees actually draw
  1. Fixed 4% rule — 4% of the starting balance, held constant in real terms
  2. Dynamic personalised — balance amortised over (life expectancy x safety
     margin) years
  3. Personalised rule — draws what the member needs to reach their ASFA
     lifestyle target *after* the Age Pension, capped at a sustainable rate,
     and bridges the Age Pension gap between retirement and age 67

Key modelling choices (v2)
--------------------------
* Everything is in REAL (today's) dollars. Investment returns are real and
  net of fees; ASFA targets and Age Pension thresholds are treated as indexed.
* The Age Pension is included (assets test and income test with deeming).
  ASFA's Modest and Comfortable standards assume a full or part Age Pension,
  so comparing super income alone against them overstates shortfall — this
  is why every strategy showed ~99% shortfall in v1.
* The legislated minimum drawdown can be enforced (on by default) — an ABP
  cannot legally pay less than this.
* Health status changes mortality for EVERY strategy (it is the same person),
  so all strategies are tested against exactly the same simulated lifetimes
  and market returns (Common Random Numbers).
* Fully vectorised: 1,000 members x 200 paths runs in seconds.

Data sources
------------
* Mortality (qx) and life expectancy (ex): Australian Life Tables 2020-22,
  Australian Government Actuary
* Retirement cost benchmarks: ASFA Retirement Standard, March quarter 2026
* Age Pension: payment rates from 20 September 2026, means-test thresholds
  from 1 July 2026, deeming rates from 20 September 2026 (Services Australia)
* Minimum drawdown rates: SIS Regulations, Schedule 7 (standard rates)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MAX_AGE = 110  # life table runs to 109; everyone is assumed dead by 110

# ---------------------------------------------------------------------------
# ASFA Retirement Standard (March quarter 2026), annual, real dollars
# ---------------------------------------------------------------------------
ASFA_COMFORTABLE_SINGLE = 55_923
ASFA_MODEST_SINGLE = 36_434

ASFA_ANNUAL_TABLE = {
    ("single", True, "65-84"): {"comfortable": 55_923, "modest": 36_434},
    ("single", True, "85+"):   {"comfortable": 53_656, "modest": 34_374},
    # Renters: ASFA publishes a single renter figure, reused for both tiers
    ("single", False, "65-84"): {"comfortable": 51_164, "modest": 51_164},
    ("single", False, "85+"):   {"comfortable": 51_164, "modest": 51_164},
}

# ---------------------------------------------------------------------------
# Age Pension — single person
# ---------------------------------------------------------------------------
AGE_PENSION_AGE = 67
AP_MAX_SINGLE = 1_237.70 * 26            # $/yr incl. pension & energy supplements
AP_ASSETS_THRESHOLD = {True: 333_000, False: 600_000}   # homeowner / non-homeowner
AP_ASSETS_TAPER = 3.0 * 26 / 1_000       # $3/fortnight per $1,000 -> per $ per year
AP_INCOME_FREE_AREA = 226.00 * 26        # $/yr, single
AP_INCOME_TAPER = 0.50
DEEMING_THRESHOLD = 66_800
DEEMING_LOWER = 0.0175
DEEMING_UPPER = 0.0375

# ---------------------------------------------------------------------------
# Legislated minimum drawdown (standard rates)
# ---------------------------------------------------------------------------
MIN_DRAWDOWN_BANDS = [(0, 0.04), (65, 0.05), (75, 0.06), (80, 0.07),
                      (85, 0.09), (90, 0.11), (95, 0.14)]

# ---------------------------------------------------------------------------
# Personalisation parameters
# ---------------------------------------------------------------------------
HEALTH_QX_MULT = {"good": 0.85, "fair": 1.00, "poor": 1.50}
HEALTH_HORIZON_MULT = {"good": 1.05, "fair": 1.00, "poor": 0.80}
LONGEVITY_BUFFER_YEARS = 5
RATE_FLOOR = 0.02
RATE_CAP = 0.10
BRIDGE_MAX_SHARE = 0.25      # max share of balance used to bridge to Age Pension age
DEPLETION_THRESHOLD = 1_000   # balance below this counts as "super has run out"

STRATEGIES = {
    "minimum": "Minimum drawdown (status quo)",
    "fixed": "Fixed 4% rule",
    "dynamic": "Dynamic personalised",
    "rule": "Personalised rule",
}


@dataclass
class Assumptions:
    """Economic and modelling assumptions. Returns are real and net of fees."""
    return_mean: float = 0.035
    return_sd: float = 0.09
    planning_return: float = 0.035   # rate used inside the strategies' own formulas
    include_age_pension: bool = True
    enforce_min_drawdown: bool = True
    safety_margin: float = 1.2       # dynamic strategy only


# ---------------------------------------------------------------------------
# 1. Life tables as age-indexed arrays  [gender_idx, age]
# ---------------------------------------------------------------------------

def _load_life_table(gender: str) -> pd.DataFrame:
    filename = "ALT_2020-22_Males.csv" if gender == "M" else "ALT_2020-22_Females.csv"
    df = pd.read_csv(DATA_DIR / filename, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    ex_col = next(c for c in df.columns if c.startswith("e"))  # header is mangled ("e?x")

    def clean(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

    out = pd.DataFrame({
        "Age": clean(df["Age"]).astype(int),
        "qx": clean(df["qx"]),
        "ex": clean(df[ex_col]),
    })
    return out.set_index("Age")


_TABLES = {"M": _load_life_table("M"), "F": _load_life_table("F")}
GENDER_IDX = {"M": 0, "F": 1}

QX = np.ones((2, MAX_AGE + 1))
EX = np.ones((2, MAX_AGE + 1))
for _g, _i in GENDER_IDX.items():
    _t = _TABLES[_g]
    QX[_i, _t.index.to_numpy()] = _t["qx"].to_numpy()
    EX[_i, _t.index.to_numpy()] = _t["ex"].to_numpy()
QX[:, MAX_AGE] = 1.0


def get_life_table(gender: str) -> pd.DataFrame:
    return _TABLES[gender]


def life_expectancy(gender: str, age: int) -> float:
    return float(EX[GENDER_IDX[gender], min(age, MAX_AGE)])


# ---------------------------------------------------------------------------
# 2. Vectorised building blocks
# ---------------------------------------------------------------------------

def min_drawdown_rate(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age)
    rate = np.zeros(age.shape, dtype=float)
    for start, r in MIN_DRAWDOWN_BANDS:
        rate = np.where(age >= start, r, rate)
    return rate


def age_pension(balance, age, homeowner, other_assets=0.0) -> np.ndarray:
    """Annual Age Pension for a single person: assets test and income test
    (deeming), lower result applies. The ABP balance counts as an assessable,
    deemed financial asset; the family home is exempt."""
    fin = np.maximum(np.asarray(balance, dtype=float), 0.0)
    assets = fin + other_assets
    threshold = np.where(homeowner, AP_ASSETS_THRESHOLD[True], AP_ASSETS_THRESHOLD[False])
    assets_red = np.maximum(assets - threshold, 0.0) * AP_ASSETS_TAPER

    deemed = (DEEMING_LOWER * np.minimum(fin, DEEMING_THRESHOLD)
              + DEEMING_UPPER * np.maximum(fin - DEEMING_THRESHOLD, 0.0))
    income_red = np.maximum(deemed - AP_INCOME_FREE_AREA, 0.0) * AP_INCOME_TAPER

    ap = np.maximum(AP_MAX_SINGLE - np.maximum(assets_red, income_red), 0.0)
    return np.where(np.asarray(age) >= AGE_PENSION_AGE, ap, 0.0)


def asfa_target(lifestyle, age, homeowner) -> np.ndarray:
    """ASFA annual budget for each member's lifestyle, age band and tenure."""
    age = np.asarray(age)
    older = age >= 85
    comf = np.asarray(lifestyle) == "comfortable"
    own = np.where(comf, np.where(older, 53_656, 55_923), np.where(older, 34_374, 36_434))
    return np.where(homeowner, own, 51_164).astype(float)


def asfa_modest(age, homeowner) -> np.ndarray:
    age = np.asarray(age)
    return np.where(homeowner, np.where(age >= 85, 34_374, 36_434), 51_164).astype(float)


def annuity_rate(n, r: float) -> np.ndarray:
    """Payout rate that exhausts a balance evenly over n years at return r,
    payments at the start of each year: 1 / a-due(n)."""
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    if abs(r) < 1e-9:
        return 1.0 / n
    v = 1.0 / (1.0 + r)
    return (1.0 - v) / (1.0 - v ** n)


# ---------------------------------------------------------------------------
# 3. Paths: one row per simulated life, with shared random numbers
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    age0: np.ndarray
    gender_idx: np.ndarray
    balance0: np.ndarray
    health: np.ndarray
    lifestyle: np.ndarray
    homeowner: np.ndarray
    member_idx: np.ndarray
    qx_mult: np.ndarray
    horizon_mult: np.ndarray
    U: np.ndarray = field(repr=False)   # mortality draws (n, T)
    R: np.ndarray = field(repr=False)   # real returns (n, T)

    @property
    def n(self) -> int:
        return len(self.age0)

    @property
    def T(self) -> int:
        return self.U.shape[1]


def _col(df: pd.DataFrame, name: str, default):
    return df[name].to_numpy() if name in df.columns else np.full(len(df), default)


def build_paths(members: pd.DataFrame, n_sim: int, assumptions: Assumptions,
                seed: int | None = 42) -> Paths:
    rng = np.random.default_rng(seed)
    m = members.reset_index(drop=True)
    rep = np.repeat(np.arange(len(m)), n_sim)
    age0 = m["age"].to_numpy(int)[rep]
    T = int(MAX_AGE - age0.min())
    n = len(rep)
    health = _col(m, "health_status", "fair")[rep]
    return Paths(
        age0=age0,
        gender_idx=m["gender"].map(GENDER_IDX).to_numpy(int)[rep],
        balance0=m["balance"].to_numpy(float)[rep],
        health=health,
        lifestyle=_col(m, "consumption_level", "comfortable")[rep],
        homeowner=_col(m, "homeowner", True).astype(bool)[rep],
        member_idx=rep,
        qx_mult=np.array([HEALTH_QX_MULT.get(h, 1.0) for h in health]),
        horizon_mult=np.array([HEALTH_HORIZON_MULT.get(h, 1.0) for h in health]),
        U=rng.random((n, T)),
        R=np.maximum(rng.normal(assumptions.return_mean, assumptions.return_sd, (n, T)), -0.95),
    )


def simulate_deaths(p: Paths) -> np.ndarray:
    """Age at death for every path. Independent of strategy, so every
    strategy is tested against exactly the same lifetimes."""
    death_age = np.full(p.n, MAX_AGE, dtype=int)
    alive = np.ones(p.n, dtype=bool)
    for t in range(p.T):
        age = np.minimum(p.age0 + t, MAX_AGE)
        q = np.minimum(QX[p.gender_idx, age] * p.qx_mult, 1.0)
        dies = alive & (p.U[:, t] < q)
        death_age[dies] = age[dies]
        alive &= ~dies
        if not alive.any():
            break
    return death_age


# ---------------------------------------------------------------------------
# 4. Strategies (vectorised over all paths)
# ---------------------------------------------------------------------------

def strategy_withdrawal(name: str, bal, age, p: Paths, a: Assumptions, ap_now) -> np.ndarray:
    age_c = np.minimum(age, MAX_AGE)
    if name == "minimum":
        w = min_drawdown_rate(age) * bal
    elif name == "fixed":
        w = 0.04 * p.balance0
    elif name == "dynamic":
        ex = EX[p.gender_idx, age_c]
        w = bal * annuity_rate(ex * a.safety_margin, a.planning_return)
    elif name == "rule":
        horizon = np.maximum(EX[p.gender_idx, age_c] * p.horizon_mult + LONGEVITY_BUFFER_YEARS, 1.0)
        s_rate = np.clip(annuity_rate(horizon, a.planning_return), RATE_FLOOR, RATE_CAP)
        need_from_super = np.maximum(asfa_target(p.lifestyle, age, p.homeowner) - ap_now, 0.0)
        ceiling = s_rate * bal
        if a.include_age_pension:
            # Age Pension bridge: before pension age the fund may also pay out
            # the pension the member is expected to receive from 67, so income
            # doesn't dip in the gap years and then jump.
            # The bridge may use at most BRIDGE_MAX_SHARE of the balance in
            # total, so small balances aren't exhausted before pension age.
            years_to_ap = np.maximum(AGE_PENSION_AGE - age, 1)
            expected_ap = age_pension(bal, np.full_like(age, AGE_PENSION_AGE), p.homeowner)
            bridge = np.minimum(expected_ap, BRIDGE_MAX_SHARE * bal / years_to_ap)
            ceiling = ceiling + np.where(age < AGE_PENSION_AGE, bridge, 0.0)
        w = np.minimum(need_from_super, ceiling)
    else:
        raise ValueError(f"Unknown strategy: {name}")

    if a.enforce_min_drawdown:
        w = np.maximum(w, min_drawdown_rate(age) * bal)
    return np.clip(w, 0.0, bal)


# ---------------------------------------------------------------------------
# 5. Core simulation
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    years_alive: np.ndarray
    years_meet_target: np.ndarray
    years_below_modest: np.ndarray
    coverage_sum: np.ndarray
    total_income: np.ndarray
    super_income: np.ndarray
    ap_income: np.ndarray
    lowest_income: np.ndarray
    ran_out: np.ndarray
    bequest: np.ndarray
    income_path: np.ndarray | None = None
    super_path: np.ndarray | None = None
    ap_path: np.ndarray | None = None
    balance_path: np.ndarray | None = None

    def summary(self) -> dict:
        yrs = np.maximum(self.years_alive, 1)
        tot_years = max(self.years_alive.sum(), 1)
        return {
            "target_met_share": self.years_meet_target.sum() / tot_years,
            "target_coverage": self.coverage_sum.sum() / tot_years,
            "prob_ever_below_modest": float(np.mean(self.years_below_modest > 0)),
            "prob_ran_out": float(np.mean(self.ran_out)),
            "avg_annual_income": self.total_income.sum() / tot_years,
            "avg_annual_super": self.super_income.sum() / tot_years,
            "avg_annual_ap": self.ap_income.sum() / tot_years,
            "p10_avg_income": float(np.percentile(self.total_income / yrs, 10)),
            "median_lowest_income": float(np.median(self.lowest_income)),
            "median_bequest": float(np.median(self.bequest)),
        }


def run_strategy(name: str, p: Paths, death_age: np.ndarray, a: Assumptions,
                 record: bool = False) -> StrategyResult:
    n, T = p.n, p.T
    bal = p.balance0.copy()
    years_alive = np.zeros(n, int)
    years_meet = np.zeros(n, int)
    years_below_modest = np.zeros(n, int)
    coverage = np.zeros(n)
    tot, sup, apt = np.zeros(n), np.zeros(n), np.zeros(n)
    lowest = np.full(n, np.inf)
    ran_out = np.zeros(n, bool)

    if record:
        inc_m = np.full((n, T), np.nan)
        sup_m = np.full((n, T), np.nan)
        ap_m = np.full((n, T), np.nan)
        bal_m = np.full((n, T + 1), np.nan)
        bal_m[:, 0] = bal

    for t in range(T):
        age = p.age0 + t
        alive = age < death_age          # alive at the start of this year
        if not alive.any():
            break

        ap_now = age_pension(bal, age, p.homeowner) if a.include_age_pension else np.zeros(n)
        w = np.where(alive, strategy_withdrawal(name, bal, age, p, a, ap_now), 0.0)
        ap_now = np.where(alive, ap_now, 0.0)
        income = w + ap_now

        target = asfa_target(p.lifestyle, age, p.homeowner)
        modest = asfa_modest(age, p.homeowner)

        years_alive += alive
        years_meet += alive & (income >= target - 1)
        years_below_modest += alive & (income < modest - 1)
        coverage += np.where(alive, np.minimum(income / target, 1.0), 0.0)
        tot += income
        sup += w
        apt += ap_now
        lowest = np.where(alive, np.minimum(lowest, income), lowest)

        bal = np.where(alive, np.maximum((bal - w) * (1 + p.R[:, t]), 0.0), bal)
        ran_out |= alive & (bal < DEPLETION_THRESHOLD) & (age + 1 < death_age)

        if record:
            inc_m[:, t] = np.where(alive, income, np.nan)
            sup_m[:, t] = np.where(alive, w, np.nan)
            ap_m[:, t] = np.where(alive, ap_now, np.nan)
            bal_m[:, t + 1] = np.where(age + 1 < death_age, bal, np.nan)

    res = StrategyResult(
        name=name, years_alive=years_alive, years_meet_target=years_meet,
        years_below_modest=years_below_modest, coverage_sum=coverage, total_income=tot, super_income=sup,
        ap_income=apt, lowest_income=np.where(np.isinf(lowest), 0.0, lowest),
        ran_out=ran_out, bequest=bal,
    )
    if record:
        res.income_path, res.super_path, res.ap_path, res.balance_path = inc_m, sup_m, ap_m, bal_m
    return res


# ---------------------------------------------------------------------------
# 6. Convenience wrappers
# ---------------------------------------------------------------------------

def member_frame(age: int, gender: str, balance: float, health_status: str = "fair",
                 consumption_level: str = "comfortable", homeowner: bool = True) -> pd.DataFrame:
    return pd.DataFrame([{
        "age": age, "gender": gender, "balance": balance, "health_status": health_status,
        "consumption_level": consumption_level, "homeowner": homeowner,
    }])


def compare_strategies(member: pd.DataFrame, assumptions: Assumptions, n_sim: int = 2000,
                       seed: int | None = 42, record: bool = True) -> dict:
    """All four strategies for one member, on common random numbers."""
    p = build_paths(member, n_sim, assumptions, seed)
    death_age = simulate_deaths(p)
    results = {k: run_strategy(k, p, death_age, assumptions, record) for k in STRATEGIES}
    return {"paths": p, "death_age": death_age, "results": results,
            "summary": {k: r.summary() for k, r in results.items()}}


def margin_sensitivity(member: pd.DataFrame, assumptions: Assumptions,
                       margins=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0), n_sim: int = 2000,
                       seed: int | None = 42) -> pd.DataFrame:
    """Dynamic strategy outcomes across safety margins, same random numbers."""
    p = build_paths(member, n_sim, assumptions, seed)
    death_age = simulate_deaths(p)
    rows = []
    for m in margins:
        s = run_strategy("dynamic", p, death_age, replace(assumptions, safety_margin=m)).summary()
        rows.append({"safety_margin": m, **s})
    return pd.DataFrame(rows)


def load_simulated_population(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"age", "gender", "balance"} - set(df.columns)
    if missing:
        raise ValueError(f"Simulated population file is missing columns: {missing}")
    return df


def run_population_analysis(population: pd.DataFrame, assumptions: Assumptions,
                            n_sim_per_member: int = 200, seed: int | None = 42) -> pd.DataFrame:
    """All four strategies for every member in one vectorised run. Returns
    one row per member with outcome measures for each strategy."""
    pop = population.reset_index(drop=True)
    p = build_paths(pop, n_sim_per_member, assumptions, seed)
    death_age = simulate_deaths(p)
    out = pop.copy()
    for k in STRATEGIES:
        r = run_strategy(k, p, death_age, assumptions)
        g = pd.DataFrame({
            "m": p.member_idx, "yrs": r.years_alive, "meet": r.years_meet_target,
            "inc": r.total_income, "cov": r.coverage_sum, "out": r.ran_out, "below": r.years_below_modest > 0,
            "beq": r.bequest,
        }).groupby("m")
        agg = g[["yrs", "meet", "inc", "cov"]].sum()
        out[f"{k}_target_coverage"] = (agg["cov"] / agg["yrs"].clip(lower=1)).to_numpy()
        out[f"{k}_target_met_share"] = (agg["meet"] / agg["yrs"].clip(lower=1)).to_numpy()
        out[f"{k}_avg_income"] = (agg["inc"] / agg["yrs"].clip(lower=1)).to_numpy()
        out[f"{k}_prob_ran_out"] = g["out"].mean().to_numpy()
        out[f"{k}_prob_below_modest"] = g["below"].mean().to_numpy()
        out[f"{k}_median_bequest"] = g["beq"].median().to_numpy()
    return out
