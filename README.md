# Personalised Retirement Drawdown Dashboard

An interactive Monte Carlo simulation comparing a traditional **fixed 4% withdrawal rule** against a **dynamic.
## The problem

Most Australians in retirement use an Account-Based Pension (ABP), withdrawing a self-selected, largely static percentage of their balance each year. This approach doesn't adapt to the member's changing life expectancy, market conditions, health, or personal risk tolerance — leaving retirees exposed to either running out of money or under-spending out of unnecessary caution.

## What this model does

- Simulates thousands of individual retirement outcomes using **Monte Carlo methods**, driven by official **Australian Government Actuary mortality tables** (Life Tables 2020-22).
- Compares **three withdrawal strategies**:
  1. **Fixed 4% Rule** — the industry-standard benchmark: withdraws a fixed dollar amount each year.
  2. **Dynamic Personalised** — recalculates the withdrawal every year from the member's current balance, current age, and a chosen risk-tolerance ("safety margin") parameter.
  3. **Personalised Rule (need vs sustainable)** — reconciles what a member *needs* to maintain their chosen lifestyle (an ASFA Retirement Standard benchmark) against what their balance can *sustainably* support, adjusted for health status; flags members likely to need supplementary income (e.g. the Age Pension) rather than silently overspending.
- Tracks two distinct risk metrics: the probability of literally running out of money before death, and the (often more realistic) probability of income falling below the [ASFA Retirement Standard](https://www.superannuation.asn.au/resources/retirement-standard) even where the account isn't exhausted.
- Uses **Common Random Numbers**, a variance-reduction technique that tests every strategy against the *same* simulated population (same simulated lifespans, same simulated market returns) for a fair, paired comparison.
- Includes a **population-level view**: runs all three strategies across a simulated cohort of 1,000 members (generated from real ABS income/asset distributions) to show outcomes across an entire target segment, not just one hand-picked example.

## Project structure

```
retirement-dashboard/
├── app.py                  # Streamlit dashboard (UI layer)
├── actuarial_model.py       # Core Monte Carlo simulation engine + all three strategies
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

## Disclaimer

This is a simplified, illustrative model built for an academic innovation challenge. It does not constitute financial advice.
