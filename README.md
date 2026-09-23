# Personalised Retirement Drawdown Dashboard

An interactive Monte Carlo simulation comparing today's default retirement drawdown behaviour against **personalised, Age-Pension-aware drawdown strategies**, built for the Actuaries Institute Innovation Challenge 2026.

🔗 **[Live demo](#)** *(add your deployed Streamlit Cloud link here)*

## The problem

Most Australians in retirement use an Account-Based Pension (ABP), withdrawing a self-selected, largely static percentage of their balance each year. This approach doesn't adapt to the member's changing life expectancy, market conditions, health, or personal risk tolerance — leaving retirees exposed to either running out of money or under-spending out of unnecessary caution.

## What this model does

- Simulates thousands of retirement lifetimes with **Monte Carlo methods**, driven by the **Australian Life Tables 2020-22** (Australian Government Actuary), with mortality adjusted for health status.
- Works in **today's dollars** throughout: real, after-fee investment returns; ASFA budgets and Age Pension thresholds treated as indexed.
- Models the **Age Pension** (assets test and income test with deeming, 20 September 2026 rates), because ASFA's Modest and Comfortable standards assume retirees receive it.
- Applies the **legislated minimum drawdown** for account-based pensions (can be switched off).
- Compares **four drawdown strategies** on exactly the same simulated futures (*Common Random Numbers*):
  1. **Minimum drawdown (status quo)** — what most retirees actually do today.
  2. **Fixed 4% rule** — the textbook benchmark.
  3. **Dynamic personalised** — the balance spread over *life expectancy × safety margin* years, allowing for expected returns.
  4. **Personalised rule** — draws what the member needs to reach their ASFA lifestyle target *after* the Age Pension, capped at a health-adjusted sustainable rate, and bridges the Age Pension gap before 67.
- Measures outcomes from the member's point of view: **target coverage** (share of the lifestyle target met each year), the **leanest year** of income, the chance income ever falls below **ASFA Modest**, the chance **super runs out**, and the **balance left at death**.
- Includes a **population view**: all four strategies across a simulated cohort of 1,000 members (fully vectorised, runs in seconds).

## Project structure

```
retirement-dashboard/
├── app.py                  # Streamlit dashboard (UI layer)
├── actuarial_model.py      # Vectorised Monte Carlo engine + all four strategies
├── .streamlit/config.toml  # Dashboard colour theme
├── data/
│   ├── ALT_2020-22_Males.csv
│   ├── ALT_2020-22_Females.csv
│   └── simulated_members.csv   # Simulated population (ABS-based)
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone <your-repo-url>
cd retirement-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

- **Mortality rates & life expectancy**: [Australian Life Tables 2020-22](https://aga.gov.au/publications/life-tables/australian-life-tables-2020-22), Australian Government Actuary
- **Retirement cost benchmarks**: [ASFA Retirement Standard](https://www.superannuation.asn.au/resources/retirement-standard), March quarter 2026
- **Age Pension**: Services Australia — payment rates and deeming rates from 20 September 2026, means-test thresholds from 1 July 2026
- **Minimum drawdown rates**: Superannuation Industry (Supervision) Regulations, Schedule 7

## Disclaimer

This is a simplified, illustrative model built for an academic innovation challenge. It does not constitute financial advice.
