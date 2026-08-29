import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


from analysis import Analyzer
import config
from model import CompetingFoodFirmMarket
from world import World


def slope(values):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.0

    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def dominant_period(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 12:
        return 0.0, 0.0

    x = np.arange(len(values), dtype=float)
    detrended = values - np.polyval(np.polyfit(x, values, 1), x)
    detrended = detrended - np.mean(detrended)
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(detrended), d=1.0)

    if len(spectrum) <= 2 or float(np.sum(spectrum[1:])) <= 0:
        return 0.0, 0.0

    idx = int(np.argmax(spectrum[1:]) + 1)
    if freqs[idx] <= 0:
        return 0.0, 0.0

    period = float(1.0 / freqs[idx])
    strength = float(spectrum[idx] / np.sum(spectrum[1:]))
    return period, strength


def turning_point_count(values, min_delta=0.0):
    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return 0

    deltas = np.diff(values)
    deltas[np.abs(deltas) < min_delta] = 0.0
    signs = np.sign(deltas)
    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def amplitude(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    return float(np.max(values) - np.min(values))


def detrended_amplitude(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return amplitude(values)

    x = np.arange(len(values), dtype=float)
    detrended = values - np.polyval(np.polyfit(x, values, 1), x)
    return amplitude(detrended)


def latest_turning_points(rows, metric, limit=5, min_delta=0.0):
    values = np.asarray([row[metric] for row in rows], dtype=float)

    if len(values) < 3:
        return []

    points = []
    deltas = np.diff(values)
    deltas[np.abs(deltas) < min_delta] = 0.0
    for idx in range(1, len(deltas)):
        before = deltas[idx - 1]
        after = deltas[idx]
        if before == 0 or after == 0 or np.sign(before) == np.sign(after):
            continue

        kind = "peak" if before > 0 and after < 0 else "trough"
        row = rows[idx]
        points.append((row["step"], kind, row[metric]))

    return points[-limit:]


def average_pairwise_correlation(series_by_firm):
    correlations = []

    for left in range(len(series_by_firm)):
        for right in range(left + 1, len(series_by_firm)):
            a = np.asarray(series_by_firm[left], dtype=float)
            b = np.asarray(series_by_firm[right], dtype=float)

            if len(a) < 3 or len(b) < 3:
                continue

            x = np.arange(len(a), dtype=float)
            a = a - np.polyval(np.polyfit(x, a, 1), x)
            b = b - np.polyval(np.polyfit(x, b, 1), x)

            if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                continue

            correlations.append(float(np.corrcoef(a, b)[0, 1]))

    if not correlations:
        return 0.0

    return float(np.mean(correlations))


def firm_series(rows, metric):
    firm_count = len(rows[-1]["firms"])
    return [
        [row["firms"][firm_id][metric] for row in rows]
        for firm_id in range(firm_count)
    ]


def row_metric(row, metric):
    if metric == "population":
        return row["population"]
    if metric == "firm_cash":
        return row["firm_cash"]
    if metric == "firm_net_worth":
        return row["firm_net_worth"]
    if metric == "price_mean":
        return row["price_mean"]
    if metric == "money_supply":
        return row["money_supply"]
    if metric == "cb_inventory":
        return row["cb_inventory"]
    if metric == "market_share_hhi":
        return row["market_share_hhi"]
    if metric == "price_std":
        return row["price_std"]
    raise KeyError(metric)


def print_segment_diagnostics(rows, boundaries):
    if not rows:
        return

    max_step = rows[-1]["step"]
    points = [0]
    for boundary in boundaries:
        if 0 < boundary < max_step:
            points.append(boundary)
    points.append(max_step)
    points = sorted(set(points))

    print("Segment Diagnostics")
    print("-------------------")
    print(
        f"{'segment':<13} {'cash sl':>10} {'nw sl':>10} {'price sl':>10} "
        f"{'money sl':>10} {'cb inv sl':>10} {'pop sl':>9} {'hhi sl':>9}"
    )
    for start, end in zip(points[:-1], points[1:]):
        segment = [
            row
            for row in rows
            if start <= row["step"] <= end
        ]
        if len(segment) < 2:
            continue
        print(
            f"{start:>4}-{end:<7} "
            f"{slope([row_metric(row, 'firm_cash') for row in segment]):>10,.2f} "
            f"{slope([row_metric(row, 'firm_net_worth') for row in segment]):>10,.2f} "
            f"{slope([row_metric(row, 'price_mean') for row in segment]):>10.6f} "
            f"{slope([row_metric(row, 'money_supply') for row in segment]):>10,.2f} "
            f"{slope([row_metric(row, 'cb_inventory') for row in segment]):>10,.2f} "
            f"{slope([row_metric(row, 'population') for row in segment]):>9.3f} "
            f"{slope([row_metric(row, 'market_share_hhi') for row in segment]):>9.6f}"
        )
    print()


def gini(values):
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]

    if len(values) == 0 or np.sum(values) <= 0:
        return 0.0

    values.sort()
    index = np.arange(1, len(values) + 1)
    return float(np.sum((2 * index - len(values) - 1) * values) / (len(values) * np.sum(values)))


def firm_snapshot(world, step):
    market = world.firm_system
    total_sales = max(1.0, sum(firm.sales_units for firm in market.firms))

    return {
        "step": step,
        "population": len(world.population),
        "households": len(world.households),
        "firm_cash": market.cash(),
        "firm_net_worth": market.net_worth(),
        "firm_inventory": market.food_inventory_units,
        "price_mean": float(np.mean([firm.price for firm in market.firms])),
        "price_min": min(firm.price for firm in market.firms),
        "price_max": max(firm.price for firm in market.firms),
        "price_std": float(np.std([firm.price for firm in market.firms])),
        "cash_gini": gini([firm.cash for firm in market.firms]),
        "market_share_hhi": sum((firm.sales_units / total_sales) ** 2 for firm in market.firms),
        "money_supply": world.cumulative_money_issued_history[-1],
        "accounting_gap": world.monetary_accounting_gap_history[-1],
        "unmet_food": world.unmet_food_demand_units_history[-1],
        "cb_inventory": world.central_bank_food_inventory_units_history[-1],
        "firms": [
            {
                "id": firm.id,
                "cash": firm.cash,
                "price": firm.price,
                "inventory": firm.inventory_units,
                "sales": firm.sales_units,
                "market_share": firm.sales_units / total_sales,
                "profit": firm.profit_before_dividend,
                "cash_flow": firm.cash_flow,
                "labor_share": firm.labor_share,
                "production_scale": firm.production_scale,
                "productivity": firm.productivity,
                "strategy_bias": firm.strategy_bias,
                "initial_strategy_bias": firm.initial_strategy_bias,
                "normal_unit_cost": firm.normal_unit_cost,
                "normal_target_price": (
                    firm.normal_unit_cost
                    *
                    config.FIRM_TARGET_MARKUP
                    *
                    (1.0 + firm.strategy_bias)
                ),
                "normal_price_floor": (
                    firm.normal_unit_cost
                    *
                    config.FIRM_TARGET_MARKUP
                    *
                    config.FIRM_NORMAL_PRICE_FLOOR_FACTOR
                ),
            }
            for firm in market.firms
        ],
    }


def print_competition_report(args, split_state, rows):
    final = rows[-1]
    tail = rows[-min(args.tail_window, len(rows)):]

    print("Competing food firms diagnostic")
    print("===============================")
    print(f"Seed: {args.seed}")
    print(f"Initial population: {args.population:,}")
    print(f"Warmup steps before split: {args.warmup_steps:,}")
    print(f"Competition steps: {args.competition_steps:,}")
    print(f"Firm count: {args.firms}")
    print(f"Price sensitivity: {args.price_sensitivity:.3f}")
    print(f"Labor reallocation rate: {args.labor_reallocation_rate:.3f}")
    print(f"Profit labor weight: {args.profit_labor_weight:.3f}")
    print(f"Strategy exploration noise: {args.strategy_exploration_noise:.4f}")
    print(f"Strategy adaptation rate: {args.strategy_adaptation_rate:.4f}")
    print(f"CB max new money issue / step: {config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP:.2f}")
    print(f"CB release rate: {config.CB_RELEASE_RATE:.3f}")
    print(f"CB purchase rate: {config.CB_PURCHASE_RATE:.3f}")
    print()

    print("Split point")
    print("-----------")
    print(f"Warmup population: {split_state['population']:,}")
    print(f"Warmup households: {split_state['households']:,}")
    print(f"Warmup firm cash: {split_state['firm_cash']:,.2f}")
    print(f"Warmup food inventory units: {split_state['food_inventory_units']:,.2f}")
    print(f"Warmup food price: {split_state['food_price']:.3f}")
    print(f"Warmup CB food inventory: {split_state['cb_food_inventory']:,.2f}")
    print()

    print("Competition outcome")
    print("-------------------")
    print(f"Final population: {final['population']:,}")
    print(f"Final firm cash total: {final['firm_cash']:,.2f}")
    print(f"Final firm net worth total: {final['firm_net_worth']:,.2f}")
    print(f"Final mean price: {final['price_mean']:.3f}")
    print(f"Final price range: {final['price_min']:.3f} - {final['price_max']:.3f}")
    print(f"Final price std: {final['price_std']:.4f}")
    print(f"Final firm cash Gini: {final['cash_gini']:.3f}")
    print(f"Final market-share HHI: {final['market_share_hhi']:.3f}")
    print(f"Final CB inventory: {final['cb_inventory']:,.2f}")
    print(f"Final money supply: {final['money_supply']:,.2f}")
    print(f"Final accounting gap: {final['accounting_gap']:.2e}")
    print()

    print("Tail slopes")
    print("-----------")
    print(f"Firm cash slope / step: {slope([row['firm_cash'] for row in tail]):,.2f}")
    print(f"Money supply slope / step: {slope([row['money_supply'] for row in tail]):,.2f}")
    print(f"Mean price slope / step: {slope([row['price_mean'] for row in tail]):.6f}")
    print(f"Price std slope / step: {slope([row['price_std'] for row in tail]):.6f}")
    print(f"Market-share HHI slope / step: {slope([row['market_share_hhi'] for row in tail]):.6f}")
    print(f"CB inventory slope / step: {slope([row['cb_inventory'] for row in tail]):,.2f}")
    print(f"Mean unmet food demand: {np.mean([row['unmet_food'] for row in tail]):,.2f}")
    period, strength = dominant_period([row["price_mean"] for row in tail])
    print(f"Mean price dominant cycle: {period:.1f} steps (strength {strength:.2f})")
    print(f"Mean price turning points: {turning_point_count([row['price_mean'] for row in tail], 0.001)}")
    print(f"Mean price amplitude: {amplitude([row['price_mean'] for row in tail]):.4f}")
    print(f"Mean price detrended amplitude: {detrended_amplitude([row['price_mean'] for row in tail]):.4f}")
    print(f"Total firm cash turning points: {turning_point_count([row['firm_cash'] for row in tail], 10_000.0)}")
    print(f"Total firm cash amplitude: {amplitude([row['firm_cash'] for row in tail]):,.2f}")
    print(f"Total firm cash detrended amplitude: {detrended_amplitude([row['firm_cash'] for row in tail]):,.2f}")
    print("Latest mean-price turning points:")
    for step, kind, value in latest_turning_points(tail, "price_mean", min_delta=0.001):
        print(f"  step {step:>5}: {kind:<6} price={value:.4f}")
    print("Latest firm-cash turning points:")
    for step, kind, value in latest_turning_points(tail, "firm_cash", min_delta=10_000.0):
        print(f"  step {step:>5}: {kind:<6} cash={value:,.2f}")
    print()
    print_segment_diagnostics(rows, args.segment_boundaries)

    share_series = firm_series(tail, "market_share")
    inventory_series = firm_series(tail, "inventory")
    print("Firm-Level Cycle Diagnostics")
    print("----------------------------")
    print(
        "Avg pairwise market-share corr: "
        f"{average_pairwise_correlation(share_series):.3f}"
    )
    print(
        "Avg pairwise inventory corr: "
        f"{average_pairwise_correlation(inventory_series):.3f}"
    )
    print(
        "Mean market-share detrended amplitude: "
        f"{np.mean([detrended_amplitude(series) for series in share_series]):.4f}"
    )
    print(
        "Mean inventory detrended amplitude: "
        f"{np.mean([detrended_amplitude(series) for series in inventory_series]):,.2f}"
    )
    print()

    tail_cashflow_by_firm = [
        np.mean([row["firms"][firm_id]["cash_flow"] for row in tail])
        for firm_id in range(len(final["firms"]))
    ]
    final_prices = [firm["price"] for firm in final["firms"]]
    final_scales = [firm["production_scale"] for firm in final["firms"]]
    price_cashflow_corr = (
        float(np.corrcoef(final_prices, tail_cashflow_by_firm)[0, 1])
        if np.std(final_prices) > 1e-12 and np.std(tail_cashflow_by_firm) > 1e-12
        else 0.0
    )
    scale_cashflow_corr = (
        float(np.corrcoef(final_scales, tail_cashflow_by_firm)[0, 1])
        if np.std(final_scales) > 1e-12 and np.std(tail_cashflow_by_firm) > 1e-12
        else 0.0
    )
    print("Firm Cash Flow Relationship")
    print("---------------------------")
    print(f"Corr(final price, tail avg cash flow): {price_cashflow_corr:.3f}")
    print(f"Corr(final scale, tail avg cash flow): {scale_cashflow_corr:.3f}")
    print(f"{'id':>3} {'tail avg cflow':>15} {'price':>7} {'scale':>7} {'share':>7} {'bias':>7} {'d_bias':>7}")
    for firm_id, avg_cash_flow in enumerate(tail_cashflow_by_firm):
        firm = final["firms"][firm_id]
        print(
            f"{firm_id:>3} "
            f"{avg_cash_flow:>15,.2f} "
            f"{firm['price']:>7.3f} "
            f"{firm['production_scale']:>7.3f} "
            f"{firm['market_share']:>7.2%} "
            f"{firm['strategy_bias']:>7.3f} "
            f"{firm['strategy_bias'] - firm['initial_strategy_bias']:>7.3f}"
        )
    print()

    print("Final firms")
    print("-----------")
    print(
        f"{'id':>3} {'cash':>12} {'price':>7} {'inventory':>12} "
        f"{'sales':>10} {'share':>7} {'profit':>10} {'cflow':>10} {'labor':>7} "
        f"{'scale':>7} {'prod':>7} {'bias':>7} {'d_bias':>7} "
        f"{'n_cost':>7} {'n_tgt':>7} {'n_floor':>8}"
    )
    for firm in sorted(final["firms"], key=lambda item: item["id"]):
        print(
            f"{firm['id']:>3} "
            f"{firm['cash']:>12,.0f} "
            f"{firm['price']:>7.3f} "
            f"{firm['inventory']:>12,.0f} "
            f"{firm['sales']:>10,.0f} "
            f"{firm['market_share']:>7.2%} "
            f"{firm['profit']:>10,.0f} "
            f"{firm['cash_flow']:>10,.0f} "
            f"{firm['labor_share']:>7.2%} "
            f"{firm['production_scale']:>7.3f} "
            f"{firm['productivity']:>7.2f} "
            f"{firm['strategy_bias']:>7.3f} "
            f"{firm['strategy_bias'] - firm['initial_strategy_bias']:>7.3f} "
            f"{firm['normal_unit_cost']:>7.3f} "
            f"{firm['normal_target_price']:>7.3f} "
            f"{firm['normal_price_floor']:>8.3f}"
        )

    print()
    print("Reading guide")
    print("-------------")
    print("The first pass is promising if accounting stays closed, unmet food demand")
    print("stays near zero, prices diverge without exploding, and market shares/cash")
    print("begin to differentiate instead of all firms staying identical.")
    print("Large smooth cycles are a feedback-risk signal. The deadband and capped")
    print("central-bank issuance are intended to damp mechanical inventory-price")
    print("oscillation and prevent persistent money-supply drift.")


def plot_competition(rows):
    steps = [row["step"] for row in rows]
    firm_count = len(rows[-1]["firms"])
    fig, ax = plt.subplots(3, 3, figsize=(17, 12))

    ax[0, 0].plot(steps, [row["firm_cash"] for row in rows], label="Total Firm Cash")
    ax[0, 0].set_title("Total Firm Cash")

    ax[0, 1].plot(steps, [row["money_supply"] for row in rows], label="Money Supply")
    ax[0, 1].set_title("Money Supply")

    ax[0, 2].plot(steps, [row["cb_inventory"] for row in rows], label="CB Inventory")
    ax[0, 2].set_title("Central Bank Food Reserve")

    mean_scales = [
        np.mean([firm["production_scale"] for firm in row["firms"]])
        for row in rows
    ]
    mean_inventories = [
        np.mean([firm["inventory"] for firm in row["firms"]])
        for row in rows
    ]
    for firm_id in range(firm_count):
        ax[1, 0].plot(
            steps,
            [
                row["firms"][firm_id]["price"] - row["price_mean"]
                for row in rows
            ],
            label=f"Firm {firm_id}",
        )
        ax[1, 1].plot(
            steps,
            [row["firms"][firm_id]["market_share"] for row in rows],
            label=f"Firm {firm_id}",
        )
        ax[1, 2].plot(
            steps,
            [
                row["firms"][firm_id]["inventory"] - mean_inventories[idx]
                for idx, row in enumerate(rows)
            ],
            label=f"Firm {firm_id}",
        )
        ax[2, 0].plot(
            steps,
            [
                row["firms"][firm_id]["production_scale"] - mean_scales[idx]
                for idx, row in enumerate(rows)
            ],
            label=f"Firm {firm_id}",
        )

    final_firms = sorted(rows[-1]["firms"], key=lambda firm: firm["production_scale"])
    ax[2, 1].bar(
        [f"F{firm['id']}" for firm in final_firms],
        [firm["production_scale"] for firm in final_firms],
    )

    ax[1, 0].axhline(0, linestyle="--", linewidth=1)
    ax[1, 0].set_title("Price Deviation From Mean")
    ax[1, 1].set_title("Market Shares")
    ax[1, 2].axhline(0, linestyle="--", linewidth=1)
    ax[1, 2].set_title("Inventory Deviation From Mean")
    ax[2, 0].axhline(0, linestyle="--", linewidth=1)
    ax[2, 0].set_title("Production Scale Deviation")
    ax[2, 1].set_title("Final Production Scale Ranking")

    ax[2, 2].plot(steps, [row["price_std"] for row in rows], label="Price Std")
    ax[2, 2].plot(steps, [row["cash_gini"] for row in rows], label="Cash Gini")
    ax[2, 2].plot(steps, [row["market_share_hhi"] for row in rows], label="HHI")
    ax[2, 2].set_title("Competition Diagnostics")

    for axis in ax.ravel():
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    plt.tight_layout()
    plt.show()


def plot_firm_cashflow_comparison(rows, tail_window=200):
    steps = [row["step"] for row in rows]
    firm_count = len(rows[-1]["firms"])
    tail = rows[-min(tail_window, len(rows)):]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    for firm_id in range(firm_count):
        ax[0, 0].plot(
            steps,
            [row["firms"][firm_id]["cash_flow"] for row in rows],
            label=f"Firm {firm_id}",
        )

    final_firms = sorted(rows[-1]["firms"], key=lambda firm: firm["id"])
    avg_cash_flows = [
        np.mean([row["firms"][firm["id"]]["cash_flow"] for row in tail])
        for firm in final_firms
    ]

    ax[0, 1].bar(
        [f"F{firm['id']}" for firm in final_firms],
        avg_cash_flows,
    )

    ax[1, 0].scatter(
        [firm["price"] for firm in final_firms],
        avg_cash_flows,
        s=[
            2000 * max(0.01, firm["market_share"])
            for firm in final_firms
        ],
    )
    for firm, cash_flow in zip(final_firms, avg_cash_flows):
        ax[1, 0].annotate(f"F{firm['id']}", (firm["price"], cash_flow))

    ax[1, 1].scatter(
        [firm["production_scale"] for firm in final_firms],
        avg_cash_flows,
        s=[
            2000 * max(0.01, firm["market_share"])
            for firm in final_firms
        ],
    )
    for firm, cash_flow in zip(final_firms, avg_cash_flows):
        ax[1, 1].annotate(f"F{firm['id']}", (firm["production_scale"], cash_flow))

    ax[0, 0].axhline(0, linestyle="--", linewidth=1)
    ax[0, 0].set_title("Firm Cash Flow Per Step")
    ax[0, 0].legend()
    ax[0, 1].axhline(0, linestyle="--", linewidth=1)
    ax[0, 1].set_title(f"Average Cash Flow, Last {len(tail)} Steps")
    ax[1, 0].axhline(0, linestyle="--", linewidth=1)
    ax[1, 0].set_title("Cash Flow vs Final Price")
    ax[1, 0].set_xlabel("Final Price")
    ax[1, 1].axhline(0, linestyle="--", linewidth=1)
    ax[1, 1].set_title("Cash Flow vs Final Production Scale")
    ax[1, 1].set_xlabel("Final Production Scale")

    plt.tight_layout()
    plt.show()


def private_cash_stock(world):
    return (
        world.firm_system.cash()
        +
        sum(household.wealth for household in world.households)
    )


def reset_world_analysis_window(world):
    for name, value in vars(world).items():
        if name.endswith("_history") and isinstance(value, list):
            value.clear()

    public_wealth = max(0.0, getattr(world, "public_wealth", 0.0))
    public_income_balance = max(
        0.0,
        getattr(world.firm_system.central_bank, "public_income_balance", 0.0),
    )
    money_supply = getattr(world.firm_system.central_bank, "money_supply", 0.0)
    located_private_money = (
        private_cash_stock(world)
        +
        public_wealth
        +
        public_income_balance
    )
    world.initial_private_money_stock = located_private_money - money_supply


def apply_runtime_config(args):
    config.CB_RELEASE_RATE = args.cb_release_rate
    config.CB_PURCHASE_RATE = args.cb_purchase_rate
    config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP = args.cb_max_new_money_issue_per_step
    config.STRATEGY_CASH_PRESSURE_WEIGHT = args.strategy_cash_pressure_weight


def run(args):
    random.seed(args.seed)
    apply_runtime_config(args)
    world = World(initial_population=args.population)
    world.steps = args.warmup_steps
    world.run(progress_interval=args.progress_interval)

    warmup_world = world
    source_firm = world.firm_system
    split_state = {
        "population": len(world.population),
        "households": len(world.households),
        "firm_cash": source_firm.cash,
        "food_inventory_units": source_firm.food_inventory_units,
        "food_price": source_firm.price,
        "cb_food_inventory": source_firm.central_bank.food_inventory_units,
    }
    world.firm_system = CompetingFoodFirmMarket(
        world,
        source_firm,
        firm_count=args.firms,
    )
    world.firm_system.price_sensitivity = args.price_sensitivity
    world.firm_system.loyalty_weight = args.loyalty_weight
    world.firm_system.price_adjustment_rate = args.price_adjustment_rate
    world.firm_system.production_adjustment_rate = args.production_adjustment_rate
    world.firm_system.strategy_exploration_noise = args.strategy_exploration_noise
    world.firm_system.labor_reallocation_rate = args.labor_reallocation_rate
    world.firm_system.profit_labor_weight = args.profit_labor_weight
    world.firm_system.strategy_adaptation_rate = args.strategy_adaptation_rate
    reset_world_analysis_window(world)

    rows = []
    for step in range(args.competition_steps):
        world.step()
        rows.append(firm_snapshot(world, step))

        if args.progress_interval and step % args.progress_interval == 0:
            final = rows[-1]
            print(
                f"Competition step {step}/{args.competition_steps} | "
                f"Population: {final['population']} | "
                f"Mean price: {final['price_mean']:.3f} | "
                f"Price std: {final['price_std']:.4f} | "
                f"HHI: {final['market_share_hhi']:.3f}"
            )

    return world, split_state, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warm up the stable single-firm model, split the firm, then test same-good competition."
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--population", type=int, default=config.INITIAL_POPULATION)
    parser.add_argument("--warmup-steps", type=int, default=config.WARMUP_STEPS)
    parser.add_argument(
        "--competition-steps",
        type=int,
        default=config.COMPETITION_STEPS,
    )
    parser.add_argument("--firms", type=int, default=config.FIRM_COUNT)
    parser.add_argument("--tail-window", type=int, default=200)
    parser.add_argument(
        "--segment-boundaries",
        type=int,
        nargs="*",
        default=[600, 1000, 1500],
        help="Competition-step boundaries for segmented slope diagnostics.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--price-sensitivity", type=float, default=config.PRICE_SENSITIVITY)
    parser.add_argument("--loyalty-weight", type=float, default=config.LOYALTY_WEIGHT)
    parser.add_argument(
        "--price-adjustment-rate",
        type=float,
        default=config.FIRM_PRICE_ADJUSTMENT_RATE,
    )
    parser.add_argument(
        "--production-adjustment-rate",
        type=float,
        default=config.FIRM_PRODUCTION_ADJUSTMENT_RATE,
    )
    parser.add_argument(
        "--strategy-exploration-noise",
        type=float,
        default=config.STRATEGY_EXPLORATION_NOISE,
    )
    parser.add_argument(
        "--labor-reallocation-rate",
        type=float,
        default=config.LABOR_REALLOCATION_RATE,
    )
    parser.add_argument(
        "--profit-labor-weight",
        type=float,
        default=config.PROFIT_LABOR_WEIGHT,
    )
    parser.add_argument(
        "--strategy-adaptation-rate",
        type=float,
        default=config.STRATEGY_ADAPTATION_RATE,
    )
    parser.add_argument(
        "--strategy-cash-pressure-weight",
        type=float,
        default=config.STRATEGY_CASH_PRESSURE_WEIGHT,
    )
    parser.add_argument("--cb-release-rate", type=float, default=config.CB_RELEASE_RATE)
    parser.add_argument("--cb-purchase-rate", type=float, default=config.CB_PURCHASE_RATE)
    parser.add_argument(
        "--cb-max-new-money-issue-per-step",
        type=float,
        default=config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP,
    )
    parser.add_argument("--skip-standard-analysis", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    world, split_state, rows = run(args)
    print_competition_report(args, split_state, rows)

    if not args.skip_standard_analysis:
        analyzer = Analyzer(world)
        print("\n======================")
        print("Post-Split Standard Analysis")
        print("======================")
        print(
            "Analysis window: competition period only "
            f"(after {args.warmup_steps:,} warmup steps)"
        )
        print()
        analyzer.equilibrium_report()
        analyzer.economy_report()
        analyzer.firm_report()

    if not args.no_plots:
        plot_competition(rows)
        plot_firm_cashflow_comparison(rows, args.tail_window)
        if not args.skip_standard_analysis:
            analyzer.plot_age_heatmap()
            analyzer.plot_population_pyramid()
            analyzer.plot_dependency_ratio()
            analyzer.plot_macro_economy()
            analyzer.plot_economy_ratio()
            analyzer.plot_household_economy_distribution()
            analyzer.plot_household_economy_diagnostics()
            analyzer.plot_household_inequality_diagnostics()
            analyzer.plot_firm_diagnostics()
            analyzer.plot_firm_balance_sheet()
            analyzer.plot_monetary_system()


if __name__ == "__main__":
    main()
