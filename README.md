# SocialSim

SocialSim is an agent-based social and economic simulation. The current main model combines household demography, marriage, fertility, inheritance, labor income, food consumption, a single food-producing firm, and a provisional central-bank buffer-stock mechanism.

The project is currently focused on building a stable closed loop between population dynamics and a simple monetary food economy. Several experimental mechanisms, especially competing firms, are kept in the `test` directory and are not yet part of the main model.

## Known Limitations

- The current main model is still a single-good, single-firm economy. Inventory can only be monetized through final household consumption or the existing public buffer-stock flows. There is not yet any cross-good demand substitution, intermediate-goods market, inter-firm trade, collateral market, or asset-conversion channel that would let inventory become liquid through another sector. This is a known structural limitation, so inventory, production, price, and credit rules should not be retuned merely to force lower inventory before the multi-firm and multi-good layers are introduced.

## Current Goals

The model is being developed in stages:

1. Build a stable demographic simulation with births, deaths, households, marriage, and inheritance.
2. Connect household income and consumption to a producing firm.
3. Stabilize the firm-household money loop so wages, food sales, savings, firm cash, and food inventory remain interpretable.
4. Add a separated monetary authority that can issue and withdraw money through food-buffer operations.
5. Measure whether the model produces reasonable fertility, household inequality, firm balance-sheet behavior, money accounting, and population structure.
6. Explore future extensions such as firm competition, multi-product markets, credit, equity, and long-run growth.

## Development Documentation

The chronological performance and architecture decisions for Step 16 are recorded in [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md).

## How To Run

Run the main simulation:

```bash
python main.py
```

Useful options:

```bash
python main.py --population 5000 --steps 2000
python main.py --population 5000 --steps 2000 --no-plots
python main.py --population 5000 --steps 2000 --progress-interval 100
```

The default initial population is currently `5000`.

## Checkpoints

Step 10.9 adds full warm-start checkpoints. A checkpoint stores the complete
`World` object plus metadata, module config state, Python RNG state, and NumPy
RNG state when NumPy is available. Loading a checkpoint resumes from the saved
global step; it does not reinitialize population, households, firm state,
public-sector state, inventory, prices, loans, relationships, IDs, or random
state.

Create the formal Step 10.9 warm world:

```bash
python main.py --scenario wage_shock_1_47 --population 5000 --steps 5000 --seed 42 --progress-interval 0 --steady-window 500 --no-plots --no-analysis --diagnostics-mode compact --output-dir test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000 --save-checkpoint test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl
```

Continue from it for another `N` steps:

```bash
python main.py --load-checkpoint test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl --steps N --progress-interval 0 --no-plots --no-analysis
```

Run the checkpoint consistency smoke test:

```bash
python test/run_step10_9_checkpoint_consistency.py
```

## Project Structure

```text
socialsim/
  main.py                         Main entry point
  world.py                        Core simulation loop and history recording
  person.py                       Individual agents, aging, mortality
  household.py                    Household container
  household_manager.py            Household restructuring and lifecycle logic
  marriage.py                     Marriage and partner matching
  productivity.py                 Age-productivity curve

  fertility/
    config.py                     Fertility parameters
    fertility.py                  Birth probability and completed fertility tracking

  economy/
    config.py                     Economy, consumption, firm, food-market parameters
    consumption.py                Household consumption and budget pressure logic
    firm.py                       Single food-producing firm and food market
    income.py                     Legacy/simple income helpers
    inheritance.py                Wealth transfer and public wealth collection

  central_bank/
    config.py                     Central-bank policy parameters
    central_bank.py               Inventory purchases, release, subsidies, money accounting

  analysis/
    analyzer.py                   Combined analysis facade
    demography.py                 Population and age-structure analysis
    fertility.py                  Completed fertility analysis
    economy.py                    Macro economy and household inequality analysis
    firm.py                       Firm, food market, money, and central-bank diagnostics

  test/
    competing_food_firms/         Experimental multi-firm food market
    output/                       Generated diagnostic outputs from some test scripts
```

## Main Simulation Loop

Each step in `World.step()` roughly follows this order:

1. People age.
2. Deaths are processed.
3. Marriage and household restructuring run.
4. The firm runs production, pays wages, sells food, updates price, adjusts production, and interacts with the central bank.
5. GDP, income, consumption, saving, firm state, food-market state, and money-accounting histories are recorded.
6. Inheritance is processed. If a household dies out with no living heirs, its remaining wealth becomes public wealth.
7. Fertility runs and newborns are added.
8. Demographic histories are updated.

## Demography

Individuals are represented by `Person`.

Current demographic logic includes:

- Annual aging.
- Age-dependent mortality.
- Marriage and household formation.
- Parent-child relationships.
- Children leaving or households being restructured through the household manager.
- Births through the fertility system.

The model records:

- Population history.
- Births and deaths.
- Birth and death rates.
- Age distribution.
- Age groups: children, workers, elderly.
- Household sizes and household types.

## Fertility

Fertility is handled in `fertility/fertility.py`, with parameters in `fertility/config.py`.

The current fertility probability depends on:

- Base birth rate: `BASE_BIRTH_RATE`.
- Mother age, using an age fertility curve.
- Household economic condition, measured through free budget relative to child cost.
- Existing number of children, through a parity factor.
- Reproductive age range, currently 20 to 40.

The fertility system also tracks couples over time so completed fertility can be analyzed after both partners have died. The analysis distinguishes:

- All completed couples.
- Fertility-exposed completed couples.
- Incomplete couples.

This was added because raw birth-rate time series alone cannot tell whether families are ending up with plausible lifetime fertility.

## Household Economy

Households hold cash wealth and receive income each step.

The main food-market consumption logic is currently in `FirmSystem.household_food_purchase_units()`. A household's food demand is built from:

- Necessary consumption based on household composition.
- Discretionary spending from income above necessary consumption.
- A small wealth-based consumption component.
- A drawdown limit that controls how much of stored wealth can be used each step.

Important concepts recorded per household:

- `income_this_step`
- `consumption_this_step`
- `saving_this_step`
- `wealth`
- necessary consumption
- desired consumption
- affordable consumption
- economic pressure
- budget-constrained status
- food subsidy received

The model intentionally separates desired consumption from affordable consumption. If a household cannot afford desired consumption, it becomes budget constrained.

## Firm And Food Market

The main model currently has one food-producing firm.

The firm:

- Hires all labor through an aggregate labor pool.
- Pays wages based on labor, sales feedback, and cash feedback.
- Produces food units using labor, productivity, and production scale.
- Holds unsold food inventory.
- Sells food to households.
- Pays dividends when prior profit and cash allow it.
- Updates food price based on shortage and inventory pressure.
- Adjusts production scale based on inventory relative to demand.

Key firm state:

- Firm cash.
- Food price.
- Food output units.
- Food demand units.
- Food sales units.
- Food inventory units.
- Inventory-to-demand ratio.
- Firm inventory value.
- Firm net worth.
- Profit before dividend.
- Sales / wage-plus-dividend income.

This gives the model a closed real-goods loop: labor produces food, wages finance household food purchases, unsold food becomes inventory, and inventory affects future price and production decisions.

## Central Bank

The current central bank is a provisional monetary authority. It is separated from the producing firm.

The central bank can:

- Buy excess food inventory from the firm at a discounted price.
- Use collected public wealth before issuing new money.
- Issue new money only for the remaining part of inventory purchases.
- Hold food as a buffer stock.
- Release food to the market during shortage or high price pressure.
- Provide in-kind food subsidies to poor households.
- Recover money when released food is sold through the market.

Important central-bank parameters live in `central_bank/config.py`.

The current design is not meant to be the final banking system. It is a stable transitional mechanism that lets the model represent monetary issuance, public buffer stock, subsidy, and money withdrawal without yet adding a full credit system.

## Money Accounting

A major current achievement is that money accounting is explicitly tracked.

The model records:

- Cumulative money issued.
- Net money issued per step.
- Firm cash.
- Household cash wealth.
- Public wealth waiting to be collected.
- Central-bank public income balance.
- Expected money stock.
- Located money stock.
- Monetary accounting gap.

Households with no living members and no living heirs no longer make money disappear. Their remaining wealth is transferred to `world.public_wealth`, then collected by the central bank as public income. That public income offsets future inventory purchases, so the central bank needs to issue less new money.

The remaining accounting gap should be near floating-point error.

## GDP And Macro Economy

GDP is recorded from the firm's nominal output value:

```text
GDP = food output units * food price
```

The macro economy analysis tracks:

- GDP.
- GDP per capita.
- Total household income.
- Total consumption.
- Total saving.
- Consumption rate.
- Saving rate.
- Household wealth.
- Wealth per capita.

Because the current main economy is still single-good and labor-driven, GDP is still strongly tied to population and labor. True long-run growth is not yet implemented in the main model.

## Analysis System

`Analyzer` combines four analysis modules:

- `DemographyAnalysis`
- `FertilityAnalysis`
- `EconomyAnalysis`
- `FirmAnalysis`

The main script currently prints:

- Equilibrium report.
- Completed fertility report.
- Economy report.
- Household inequality diagnostics.
- Firm report.

When plots are enabled, it also shows:

- Population and age-structure plots.
- Completed fertility distribution.
- Macro economy curves.
- Consumption and saving rate curves.
- Household economy distributions.
- Household inequality diagnostics, including Lorenz curves.
- Firm cash, inventory, net worth, profit, price, and sales/income diagnostics.
- Monetary system plots.
- Household money-stock plots.
- Central-bank food reserve stock.

## Household Inequality Diagnostics

The analysis now includes household-level inequality reporting.

It reports:

- Wealth Gini.
- Income Gini.
- Consumption Gini.
- Saving Gini.
- Top 10% and top 20% wealth share.
- Top 10% and top 20% income share.
- Tables by wealth decile.
- Tables by income decile.
- Tables by household type.
- Top and bottom households by wealth and income.

The plot version includes:

- Lorenz curves.
- Wealth, income, consumption, saving, and pressure distributions.
- Income vs consumption scatter.
- Flow comparison by wealth decile.
- Negative-saving and budget-constrained rates by household type.

## Experimental Tests

The `test` directory contains many standalone diagnostic scripts. These were used to test mechanisms before deciding whether they should enter the main model.

Major tested areas include:

- Completed fertility by mother age at first observation.
- Completed fertility by household economic group.
- Consumption parameter sweeps.
- Dynamic food-price closed loop.
- Fertility-rate sweeps under selected consumption profiles.
- Candidate closed-loop firm reports.
- Capital-investment growth loops.
- Fixed-population capital tests.
- Warmup-then-branch A/B tests.
- Real food market tests.
- Inventory-backed money issuance tests.
- Central-bank inventory purchase loops.
- Household inequality diagnostics.
- Multi-good money issuer experiments.
- Competing food firm experiments.

Some of these tests are historical and may not represent the current main model. The most important current experimental branch is `test/competing_food_firms`.

## Competing Food Firms Experiment

The competing-firms experiment starts from the stable main model, warms it up, then splits the single food firm into several equal firms.

The design goal is to test whether competition can create:

- More realistic price dispersion.
- Firm-level cash and market-share differentiation.
- Production-scale divergence.
- Stronger market behavior than a single aggregate firm.
- A better foundation for money as a medium of exchange.

Current competing-firm assumptions:

- The original firm is split into equal firms.
- Initial firm values sum to the original firm.
- Consumers do not always choose the cheapest firm; they choose probabilistically, with price sensitivity and loyalty.
- Firms independently adjust prices and production scale.
- Central-bank money issuance is capped in this experiment.

Current status:

- The experiment can generate firm differentiation.
- Periodic inventory and market-share cycles were reduced by removing a mechanical expected-share feedback.
- Long-run price drift remains a concern.
- Recent work added a normal-cost price anchor and soft price floor, but firm cash stability still needs more work.
- This experiment is not yet ready to merge into the main model.

## Confirmed Stable Main-Model Achievements

The current main model has reached a useful provisional equilibrium for the single-good economy:

- Population can stabilize around a few thousand agents under current parameters.
- Fertility analysis produces plausible completed-fertility diagnostics.
- The firm-household food loop is closed enough to analyze production, wages, consumption, inventory, and price.
- The central bank is separated from the firm.
- Inventory purchases, discounted purchase prices, poverty food subsidies, and market release are implemented.
- Public wealth from heirless extinct households is accounted for.
- Monetary accounting can explain where money is located.
- Analysis now covers demography, fertility, macro economy, firm state, money supply, central-bank reserve, and household inequality.

## Known Limitations

The current model is intentionally incomplete.

Known limitations:

- The main economy has only one final good: food.
- The main model still has one producing firm.
- GDP is mostly population and labor driven.
- There is no full banking system.
- There is no credit creation, interest rate, loan default, or deposit system.
- There is no formal equity market in the main model.
- There is no technology growth in the main model.
- Price formation is still rule-based rather than market-clearing through explicit order books.
- Multi-good and competing-firm systems remain experimental.

## Current Research Direction

The next major modeling question is how to move from a stable single-firm, single-good system toward a richer market economy.

Possible next steps:

1. Stabilize the competing-firms test until prices, firm cash, inventory, and money supply remain stable over long runs.
2. Add firm heterogeneity only after the equal-split model is stable.
3. Separate production decisions from pricing decisions more carefully.
4. Add a true banking or credit layer only when goods-market accounting is stable.
5. Add equity claims or firm ownership so household wealth can grow through firm value, not only through cash savings.
6. Add multi-good consumption after the food-market baseline is robust.
7. Eventually add productivity growth, capital accumulation, or technology once the financial loop is clear.

## Development Notes

Most parameters are deliberately exposed in config files:

- Fertility parameters: `fertility/config.py`
- Economy and firm parameters: `economy/config.py`
- Central-bank parameters: `central_bank/config.py`
- Competing-firm experimental parameters: `test/competing_food_firms/config.py`

When adding new mechanisms, the preferred workflow is:

1. Implement the mechanism in an isolated test file or test subfolder.
2. Add text summaries and plots.
3. Run parameter sweeps when needed.
4. Compare demographic, economic, firm, monetary, and inequality diagnostics.
5. Merge into the main model only after the mechanism is stable.

This keeps the main model usable while allowing aggressive experimentation in `test`.
