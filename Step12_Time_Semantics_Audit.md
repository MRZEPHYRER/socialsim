# Step 12: Unified Weekly Time Scale Audit

## Step 12.1 Status

The canonical convention is **one simulation step = one week** and **52
simulation steps = one model year**. The previous monthly proposal is
abandoned.

Step 12.1 adds passive time infrastructure and diagnostics only. It does not
change age progression, demography, economics, pricing, production, inventory,
credit, dividends, public support, accounting behavior, or random draws.

The exported fields are:

```text
global_step       = technical simulation index
simulation_week   = global_step
simulation_year   = global_step / 52
```

## Weekly Migration Matrix

| Subsystem | Parameter / formula | Current value | Proposed weekly interpretation | Class | Later conversion / note |
|---|---|---:|---|---|---|
| Demography | `Person.age += 1` | `1` | Calendar age progression | DURATION | Must later become fractional years or month/week age; behavior-changing |
| Demography | mortality probability | code-defined | Annual hazard/probability converted to weekly later | ANNUAL_RATE | Do not activate conversion now |
| Demography | `BASE_BIRTH_RATE` | `0.20` | Annual rate only if confirmed annual | ANNUAL_RATE | Do not convert now |
| Demography | reproduction ages | `20..40` | Calendar-year thresholds | DURATION | Depends on age representation |
| Demography | marriage matching | each step | Weekly matching opportunity | PER-REVIEW | No automatic `/52` |
| Demography | age productivity | age curve | Productivity by calendar age | DURATION | High-risk later migration |
| Household | `TARGET_WEALTH_RESERVE_MONTHS` | `6` | Six calendar months, about 26 weeks | DURATION | Must not mean six steps |
| Household | minimum consumption | state | Weekly quantity | WEEKLY_FLOW | Retain numeric value for now |
| Household | wages / income | state | Weekly flow | WEEKLY_FLOW | Do not divide by 52 |
| Household | consumption / saving | state | Weekly flow | WEEKLY_FLOW | Identities unchanged |
| Household | transfers / public support | state | Weekly/event flow | WEEKLY_FLOW | Later calibration only |
| Firm | production capacity | state | Weekly capacity | WEEKLY_FLOW | Productivity unchanged |
| Firm | production / sales / revenue | state | Weekly flows | WEEKLY_FLOW | Retain numeric values |
| Firm | expected-demand EMA | config | Weekly memory | DURATION | Convert only to preserve another target horizon |
| Firm | inventory coverage | `15` | Prefer 15 weeks of demand | DURATION | Unresolved calibration choice |
| Firm | production review interval | `5` steps | Prefer 5 weeks | DURATION | Retain value |
| Firm | spoilage rate | config | Weekly loss unless annual intent is established | WEEKLY_FLOW | TIME SEMANTICS UNRESOLVED; do not guess |
| Firm | dividends | state | Weekly distribution | WEEKLY_FLOW | No policy change |
| Pricing | review probability | `0.20` per step | Prefer 0.20 per week | PER-REVIEW | Do not divide by 52 |
| Pricing | price experiment | `0.5% / 1%` | Per-review adjustment | PER-REVIEW | Never an annual rate |
| Pricing | evaluation window | `12` steps | Prefer 12 weeks | DURATION | Retain value |
| Credit | loan issuance / repayment | state | Weekly flows | WEEKLY_FLOW | No credit change |
| Credit | repayment rate | `0.35` per step | Temporarily documented as weekly | WEEKLY_FLOW | Flag for later Step 13 redesign |
| Credit | future interest | `0.0` | Future weekly compound rate | ANNUAL_RATE | Do not add interest |
| Accounting | statement rows | one row/step | Weekly accounting period | WEEKLY_FLOW | Time metadata added |
| Diagnostics | slopes / windows | step-based | Explicit weekly measures | DURATION | `500` means 500 weeks |

## Classification Rules

**STOCK:** cash, wealth, loan balance, and inventory. Stocks are not scaled.

**WEEKLY_FLOW:** wages, consumption, production, sales, revenue, profit,
dividends, loan issuance, principal repayment, and public cash flows retain
their current numeric values and are interpreted as one-week flows.

**ANNUAL_RATE:** mortality, fertility, and future interest require later
conversion. The planned formulas are:

```text
p_week = 1 - (1 - p_annual) ** (1 / 52)
p_week = 1 - exp(-h_annual / 52)
r_week = (1 + r_annual) ** (1 / 52) - 1
```

**DURATION:** horizons, age thresholds, inventory coverage, review windows and
cooldowns need explicit calendar units. Six calendar months is approximately
26 weeks, not six simulation steps.

**PER-REVIEW:** price experiments and exploration probabilities are conditional
on a review event and are not divided by 52 automatically.

## Passive Implementation

`time_system.py` provides `SimulationClock` and conversion helpers. Behavioral
subsystems do not call the conversion helpers. The clock is synchronized from
`World.current_step_index`, so checkpoint continuation preserves time.

Macro diagnostics, firm diagnostics, and accounting tables now carry
`global_step`, `simulation_week`, and `simulation_year`.

## Unresolved Items

1. `TARGET_WEALTH_RESERVE_MONTHS = 6` should later map to about 26 weekly
   steps, without changing it in 12.1.
2. Inventory coverage `15`, production review `5`, and price evaluation `12`
   are documented as weeks but remain calibration choices.
3. Price review probability `0.20` is documented as per-week.
4. Repayment rate `0.35` is temporarily documented as weekly and is flagged
   for later credit redesign.
5. Spoilage semantics are unresolved; no annual conversion is assumed.

## Validation

Static validation completed:

```text
python -m py_compile world.py time_system.py economy/accounting.py
```

The conversion smoke check returns week 52 / year 1.0 at step 52 and 26 steps
for six calendar months. A full Step 11F before/after trajectory regression is
still required; only the new time metadata should differ.

## Decision

**Step 12.1 is ready for acceptance**, subject to the full trajectory
regression. No demographic conversion is included. The next stage must be
designed separately after acceptance.
