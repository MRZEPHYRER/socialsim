import argparse
import json
import os
import sys
from pathlib import Path
from world import World
from economy import config as economy_config
from analysis import Analyzer
from analysis.v2 import run_analysis_v2, write_analysis_failure
from analysis.financial_core import format_financial_core_console
from checkpoint import load_world_checkpoint
from checkpoint import save_world_checkpoint
from scenarios import apply_scenario
from scenarios import apply_runtime_scenario
from scenarios import export_manifest
from scenarios import build_manifest
from scenarios import scenario_names
from analysis.time_semantics import export_time_semantics_manifest
from analysis.step15e3 import run_step15e3_human_review
from analysis.step15 import (
    Step15AccountingViews,
    Step15AnalysisDataLoader,
    Step15AnalysisPlotter,
    Step15AnalysisQuery,
)
from analysis.step15_console import Step15AnalysisConsole
from analysis.step15_reference import (
    STEP15_CANONICAL_REFERENCE_RUN,
    resolve_step15_canonical_reference,
)


DEFAULT_OUTPUT_ROOT = os.path.join("test", "output")
DEFAULT_STEPS = 5000
DEFAULT_LEDGER_SENTINEL = "__default_ledger_path__"
REFERENCE_DEMO_RELATIVE_DIR = STEP15_CANONICAL_REFERENCE_RUN


def resolve_demo_analysis_dir():
    """Return the immutable accepted Analysis run used by ``--demo``."""
    return resolve_step15_canonical_reference(Path(__file__).resolve().parent)
FULL_DEMO_PROFILE = {
    "profile_name": "STEP15_ANALYSIS_READY_DEMO",
    "population": 5000,
    "steps": 2000,
    "seed": 42,
    "firm_count": 5,
    "scenario": "baseline",
    "diagnostics_mode": "full",
    "persist_diagnostics": True,
    "diagnostic_cadence": 1,
    "observability_mode": "FULL_DIAGNOSTIC",
    "analysis_profile": "full",
    "plot_mode": "save-only",
    # Canonical diagnostics remain enabled; these two legacy raw dumps are
    # optional and can dominate runtime/storage at N=5000.
    "no_household_diagnostics_csv": True,
    "no_ledger_csv": True,
    "progress_interval": 52,
    "runtime_overrides": {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS": 13,
        "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS": 13,
        "CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED": False,
        "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED": True,
        "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
    },
}


def _option_was_supplied(option):
    return option in sys.argv


def apply_full_demo_profile(args):
    """Resolve one centralized, explicitly non-calibrated demo profile."""
    for field, value in FULL_DEMO_PROFILE.items():
        if field in {"profile_name", "runtime_overrides"}:
            continue
        option = "--" + field.replace("_", "-")
        if field == "persist_diagnostics":
            supplied = _option_was_supplied("--persist-diagnostics") or _option_was_supplied("--diagnostic-persistence")
        elif field == "ledger_csv":
            supplied = _option_was_supplied("--ledger-csv") or _option_was_supplied("--no-ledger-csv")
        else:
            supplied = _option_was_supplied(option)
        if not supplied:
            setattr(args, field, value)
    if not _option_was_supplied("--output-dir"):
        args.output_dir = os.path.join(
            DEFAULT_OUTPUT_ROOT,
            f"main_full_n{args.population}_{args.steps}_seed{args.seed}",
        )
        args._demo_output_dir = True
    if not _option_was_supplied("--save-checkpoint"):
        args.save_checkpoint = None
        args._demo_checkpoint_default = True
    args._auto_launch_gui = bool(args.demo and not args.run_only)
    args._runtime_profile_name = FULL_DEMO_PROFILE["profile_name"]
    args._runtime_profile_overrides = dict(FULL_DEMO_PROFILE["runtime_overrides"])
    return args


def apply_runtime_profile(scenario, args):
    """Apply one named runtime profile before World construction.

    Profile selection is a configuration boundary; individual economic
    mechanisms continue to read their existing scenario override fields.
    """
    profile_name = getattr(args, "_runtime_profile_name", None)
    if not profile_name:
        return scenario
    overrides = dict(getattr(args, "_runtime_profile_overrides", {}))
    scenario["overrides"].update(overrides)
    scenario["overrides"]["RUNTIME_PROFILE"] = profile_name
    scenario["applied_parameters"]["runtime_profile"] = {
        "effective": profile_name,
        "source": "--demo",
        "overrides": overrides,
    }
    return scenario


def experiment_output_dir(args, steps):

    if args.output_dir:
        candidate = args.output_dir
        if getattr(args, "_demo_output_dir", False):
            base = candidate
            index = 1
            while os.path.exists(candidate):
                candidate = f"{base}_run{index}"
                index += 1
        return candidate

    if args.load_checkpoint:

        checkpoint_name = os.path.splitext(
            os.path.basename(args.load_checkpoint)
        )[0]
        run_name = (
            f"{checkpoint_name}_"
            f"continue_steps_{steps}"
        )

        return os.path.join(
            DEFAULT_OUTPUT_ROOT,
            "from_checkpoint",
            run_name
        )

    run_name = (
        f"seed_{args.seed}_"
        f"pop_{args.population}_"
        f"steps_{steps}"
    )

    return os.path.join(
        DEFAULT_OUTPUT_ROOT,
        args.scenario,
        run_name
    )


def resolve_output_paths(args, steps):

    output_dir = experiment_output_dir(args, steps)

    diagnostics_path = None
    firm_diagnostics_path = None
    household_diagnostics_path = None
    if not args.no_diagnostics_csv:

        diagnostics_path = (
            args.diagnostics_csv
            or os.path.join(output_dir, "diagnostics.csv")
        )
        firm_diagnostics_path = (
            args.firm_diagnostics_csv
            or os.path.join(output_dir, "firm_diagnostics.csv")
        )
    household_diagnostics_path = None
    if (
        not args.no_diagnostics_csv
        and not args.no_household_diagnostics_csv
        and getattr(args, "observability_mode", "FULL_DIAGNOSTIC")
        != "RESEARCH_FAST"
    ):
        household_diagnostics_path = os.path.join(
            output_dir,
            "household_diagnostics.csv",
        )

    manifest_path = None
    if not args.no_scenario_manifest:

        manifest_path = (
            args.scenario_manifest
            or os.path.join(output_dir, "manifest.json")
        )

    ledger_path = None
    if not args.no_ledger_csv:

        if args.ledger_csv == DEFAULT_LEDGER_SENTINEL:

            ledger_path = os.path.join(output_dir, "ledger.csv")

        else:

            ledger_path = args.ledger_csv

    return (
        output_dir,
        diagnostics_path,
        firm_diagnostics_path,
        household_diagnostics_path,
        manifest_path,
        ledger_path,
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the social simulation."
    )

    parser.add_argument(
        "--population",
        type=int,
        default=5000
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of weekly simulation steps to run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default="baseline",
        help="Experiment scenario name."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for default experiment outputs. "
            "Default: test/output/<scenario>/seed_<seed>_pop_<population>_steps_<steps>."
        )
    )
    parser.add_argument(
        "--diagnostics-csv",
        default=None,
        help="Path for per-week diagnostics CSV output. Default: <output-dir>/diagnostics.csv."
    )
    parser.add_argument(
        "--firm-diagnostics-csv",
        default=None,
        help=(
            "Path for per-firm diagnostics CSV output. "
            "Default: <output-dir>/firm_diagnostics.csv."
        )
    )
    parser.add_argument(
        "--scenario-manifest",
        default=None,
        help="Path for scenario manifest JSON output. Default: <output-dir>/manifest.json."
    )
    parser.add_argument(
        "--no-scenario-manifest",
        action="store_true",
        help="Skip writing the scenario manifest JSON."
    )
    parser.add_argument(
        "--ledger-csv",
        nargs="?",
        const=DEFAULT_LEDGER_SENTINEL,
        default=None,
        help=(
            "Optional path for detailed ledger transaction CSV output. "
            "Use --ledger-csv without a value to write <output-dir>/ledger.csv."
        )
    )
    parser.add_argument(
        "--no-diagnostics-csv",
        action="store_true",
        help="Skip writing the diagnostics CSV."
    )
    parser.add_argument(
        "--no-household-diagnostics-csv",
        action="store_true",
        help="Skip writing the large per-household diagnostics CSV."
    )
    parser.add_argument(
        "--no-ledger-csv",
        action="store_true",
        help="Skip writing the ledger CSV."
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N weekly steps. Use 0 to disable progress output."
    )
    parser.add_argument(
        "--steady-window",
        type=int,
        default=500,
        help="Rolling window length in weeks for steady-state diagnostics."
    )
    parser.add_argument(
        "--diagnostics-mode",
        choices=["full", "compact"],
        default="full",
        help="Use compact diagnostics for batch scans that only need key metrics."
    )
    parser.add_argument(
        "--persist-diagnostics", "--diagnostic-persistence",
        dest="persist_diagnostics",
        action="store_true",
        help=(
            "Append canonical macro, Firm, and accounting diagnostics during "
            "simulation execution."
        ),
    )
    parser.add_argument(
        "--diagnostic-cadence",
        type=int,
        default=1,
        help="Record canonical diagnostics every N completed steps (default: 1).",
    )
    parser.add_argument(
        "--persist-statistical-observability",
        action="store_true",
        help=(
            "Persist exact-age histograms, periodic Household cross-sections, "
            "labor events/denominators, and weekly Firm capital histories."
        ),
    )
    parser.add_argument(
        "--statistical-snapshot-cadence",
        type=int,
        default=13,
        help="Record social-Household statistical snapshots every N weeks (default: 13).",
    )
    parser.add_argument(
        "--statistical-age-cadence",
        type=int,
        default=13,
        help="Record exact-age/sex histograms every N weeks (default: 13).",
    )
    parser.add_argument(
        "--observability-mode",
        choices=["FULL_DIAGNOSTIC", "RESEARCH_FAST", "DEBUG_DETAILED"],
        default="FULL_DIAGNOSTIC",
        help=(
            "Diagnostic storage mode. FULL_DIAGNOSTIC preserves current "
            "detail; RESEARCH_FAST keeps weekly aggregates and sparse "
            "micro-panels; DEBUG_DETAILED is intended for forensic detail."
        ),
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip post-run analysis reports."
    )
    parser.add_argument(
        "--analysis",
        dest="interactive_analysis",
        action="store_true",
        help="Open the read-only interactive Step15 Analysis console without running a simulation.",
    )
    parser.add_argument(
        "--analysis-gui",
        dest="analysis_gui",
        action="store_true",
        help="Open the read-only PySide6 Step15 Analysis workbench without running a simulation.",
    )
    parser.add_argument(
        "--analysis-gui-v2",
        dest="analysis_gui_v2",
        action="store_true",
        help="Open the independent read-only Analysis GUI V2 against the frozen Step15 canonical reference.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true"
    )
    parser.add_argument(
        "--analysis-profile",
        choices=["standard", "full"],
        default="standard",
        help="Analysis v2 profile. Standard is the default acceptance-oriented profile.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["browser", "individual", "save-only"],
        default="browser",
        help=(
            "Plot display mode: browser opens one dashboard window, "
            "individual keeps legacy figure behavior, save-only writes PNGs without a GUI. "
            "Ignored when --no-plots is set."
        ),
    )
    parser.add_argument(
        "--step15e3-human-review",
        action="store_true",
        help=(
            "Write the Step 15E.3 macro/sector/final-demand/ownership "
            "human-review package without changing simulation behavior."
        ),
    )
    parser.add_argument(
        "--step15-analysis-dir",
        default=None,
        help="Read-only Step 15 Analysis mode; load a persisted final validation directory without running simulation.",
    )
    parser.add_argument("--step15-analysis-week-start", type=float, default=None)
    parser.add_argument("--step15-analysis-week-end", type=float, default=None)
    parser.add_argument("--step15-analysis-firm", default=None)
    parser.add_argument("--step15-analysis-sector", default=None)
    parser.add_argument(
        "--step15e3-human-review-output",
        default=os.path.join(
            DEFAULT_OUTPUT_ROOT,
            "step15E3_main_analysis_human_review",
        ),
        help=(
            "Output directory for the optional Step 15E.3 human-review "
            "package."
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Save the complete World checkpoint after the run."
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help=(
            "Load a complete World checkpoint and continue from its global "
            "step. When set, --steps means additional weekly steps to run."
        )
    )
    parser.add_argument(
        "--firm-count",
        type=int,
        default=None,
        help=(
            "Number of symmetric firm slices. With --load-checkpoint, values "
            ">1 split the loaded single-firm world conservatively before "
            "continuing."
        )
    )
    parser.add_argument(
        "--initial-age-phase-mode",
        choices=["distributed", "synchronized"],
        default="distributed",
        help=(
            "Fresh-population within-year age phase. 'distributed' uses an "
            "isolated deterministic RNG; 'synchronized' is the legacy audit control."
        ),
    )
    parser.add_argument(
        "--initial-firm-cash-multiplier",
        type=float,
        default=1.0,
        help=(
            "Optional fresh-run initialization multiplier for firm cash. "
            "Default 1.0; intended for scaling audits only."
        ),
    )
    parser.add_argument(
        "--person-equity-transition",
        action="store_true",
        help=(
            "Enable the conservative deterministic Person primary-equity "
            "transition. Disabled by default."
        ),
    )
    parser.add_argument(
        "--autonomous-secondary-equity",
        action="store_true",
        help=(
            "Enable the conservative deterministic autonomous secondary "
            "Legacy-share purchase screen. Disabled by default."
        ),
    )
    parser.add_argument(
        "--autonomous-secondary-equity-book-price",
        action="store_true",
        help=(
            "Enable autonomous secondary equity purchases using the locked "
            "lagged accounting book-equity reference price."
        ),
    )
    parser.add_argument(
        "--household-wealth-instrumentation",
        action="store_true",
        help=(
            "Enable optional targeted household wealth/lifecycle diagnostics. "
            "Does not change economic behavior or RNG usage."
        ),
    )
    parser.add_argument(
        "--household-employer-exposure-instrumentation",
        action="store_true",
        help=(
            "Enable diagnostics-only household worker/employer and scheduled/"
            "executed wage exposure records. Does not change behavior or RNG."
        ),
    )
    parser.add_argument(
        "--household-active-denominator-instrumentation",
        action="store_true",
        help=(
            "Enable compact all-active household-week labor/income risk-set "
            "diagnostics. Does not change behavior or RNG."
        ),
    )
    parser.add_argument(
        "--generalized-firm-operating-contracts",
        action="store_true",
        help=(
            "Enable Stage A passive generalized Firm operating contracts and "
            "shadow diagnostics. This does not change economic execution."
        ),
    )
    parser.add_argument(
        "--multisector-foundation",
        action="store_true",
        help=(
            "Enable Step 14A passive sector/good/market contracts. "
            "The canonical Food executor remains unchanged."
        ),
    )
    parser.add_argument(
        "--shadow-multigood-household-demand",
        action="store_true",
        help=(
            "Enable the passive Step 14C household Food/Service budget "
            "envelope. It does not execute Service behavior or alter Food."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the FULL_DEMO / ANALYSIS_READY_PROFILE and open its Analysis GUI.",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="With --demo, run and persist the profile without launching the GUI.",
    )

    args = parser.parse_args()
    if args.demo:
        try:
            args.step15_analysis_dir = str(resolve_demo_analysis_dir())
        except FileNotFoundError as exc:
            parser.error(str(exc))
        # ``--demo`` is a read-only presentation entry point.  The final
        # canonical run is materialized separately; opening the demo must not
        # rerun the simulation or create a fresh output directory.
        args.analysis_gui_v2 = True
        args._reference_demo = True
        return args
    if args.analysis_gui and not args.step15_analysis_dir:
        try:
            args.step15_analysis_dir = str(resolve_demo_analysis_dir())
        except FileNotFoundError as exc:
            parser.error(str(exc))
    return args


if __name__ == "__main__":

    args = parse_args()

    if args.analysis_gui_v2:
        from analysis.gui_v2 import launch_analysis_gui_v2
        raise SystemExit(launch_analysis_gui_v2(run_dir=args.step15_analysis_dir))

    if args.interactive_analysis:
        raise SystemExit(
            Step15AnalysisConsole(
                run_dir=args.step15_analysis_dir,
                output_root=DEFAULT_OUTPUT_ROOT,
            ).run()
        )

    if args.analysis_gui:
        from analysis.gui import launch_analysis_gui
        raise SystemExit(
            launch_analysis_gui(
                run_dir=args.step15_analysis_dir,
                output_root=DEFAULT_OUTPUT_ROOT,
            )
        )

    if args.step15_analysis_dir:
        loader = Step15AnalysisDataLoader(args.step15_analysis_dir)
        query = Step15AnalysisQuery(loader)
        views = Step15AccountingViews(query)
        print(json.dumps({
            "mode": "step15-analysis",
            "run_dir": str(loader.run_dir),
            "firms": loader.list_firms(),
            "sectors": loader.list_sectors(),
            "macro_rows": len(query.macro(args.step15_analysis_week_start, args.step15_analysis_week_end)),
            "firm_rows": len(query.firms(args.step15_analysis_firm, args.step15_analysis_sector, args.step15_analysis_week_start, args.step15_analysis_week_end)),
            "figures": Step15AnalysisPlotter(query).generate(
                loader.run_dir / "figures",
                selected_firm=args.step15_analysis_firm,
            ),
            "balance_sheet_available": bool(loader.list_firms()),
            "cash_bridge_available": bool(loader.list_firms()),
            "capital_bridge_available": bool(loader.list_firms()),
            "analysis_read_only": True,
        }, indent=2))
        raise SystemExit(0)

    print("======================")
    print("Social Simulation")
    print("======================")

    planned_steps = (
        args.steps
        if args.steps is not None
        else DEFAULT_STEPS
    )

    (
        output_dir,
        diagnostics_path,
        firm_diagnostics_path,
        household_diagnostics_path,
        manifest_path,
        ledger_path,
    ) = resolve_output_paths(
        args,
        planned_steps
    )

    if getattr(args, "_demo_checkpoint_default", False):
        args.save_checkpoint = os.path.join(output_dir, "world_final.pkl")

    if args.load_checkpoint:

        world, checkpoint_metadata = load_world_checkpoint(
            args.load_checkpoint
        )
        scenario = {
            "name": checkpoint_metadata["scenario"],
            "overrides": checkpoint_metadata["scenario_overrides"],
            "applied_parameters": {},
        }
        world.steps = planned_steps
        world.diagnostics_mode = args.diagnostics_mode
        world.observability_mode = args.observability_mode
        world.diagnostic_micro_snapshot_cadence = max(
            1, int(args.statistical_snapshot_cadence)
        )
        world.ledger.record_details = ledger_path is not None
        if not hasattr(world, "firm_diagnostics_rows"):

            world.firm_diagnostics_rows = []

        if not hasattr(world, "household_diagnostics_rows"):

            world.household_diagnostics_rows = []

        if not hasattr(world, "household_wealth_instrumentation_enabled"):
            world.household_wealth_instrumentation_enabled = False
        if not hasattr(world, "household_employer_exposure_instrumentation_enabled"):
            world.household_employer_exposure_instrumentation_enabled = False
        if not hasattr(world, "household_employer_exposure_rows"):
            world.household_employer_exposure_rows = []
        if not hasattr(world, "household_employer_exposure_weekly_rows"):
            world.household_employer_exposure_weekly_rows = []
        if not hasattr(world, "household_active_denominator_instrumentation_enabled"):
            world.household_active_denominator_instrumentation_enabled = False
        if not hasattr(world, "household_active_denominator_rows"):
            world.household_active_denominator_rows = []
        if not hasattr(world, "_household_employer_weekly_state"):
            world._household_employer_weekly_state = {}
        if not hasattr(world, "household_wealth_micro_trace_rows"):
            world.household_wealth_micro_trace_rows = []
        if not hasattr(world, "household_wealth_weekly_summary_rows"):
            world.household_wealth_weekly_summary_rows = []
        if not hasattr(world, "household_low_wealth_state"):
            world.household_low_wealth_state = {}
        if not hasattr(world, "household_low_wealth_history"):
            from collections import defaultdict
            world.household_low_wealth_history = defaultdict(list)
        if not hasattr(world, "household_lifecycle_transfer_events"):
            world.household_lifecycle_transfer_events = []
        if not hasattr(world, "next_lifecycle_transfer_event_id"):
            world.next_lifecycle_transfer_event_id = 0
        if not hasattr(world, "generalized_firm_operating_contracts"):
            world.generalized_firm_operating_contracts = False

        if args.firm_count is not None:

            world.split_firms(args.firm_count)

        elif not getattr(world, "firms", None):

            world.split_firms(1)

    else:

        # ======================
        # Initialize World
        # ======================

        scenario = apply_scenario(args.scenario)
        scenario = apply_runtime_profile(scenario, args)

        if args.initial_firm_cash_multiplier <= 0:
            raise ValueError("--initial-firm-cash-multiplier must be positive")
        if args.initial_firm_cash_multiplier != 1.0:
            economy_config.FIRM_INITIAL_CASH *= args.initial_firm_cash_multiplier
            scenario["overrides"]["INITIAL_FIRM_CASH_MULTIPLIER"] = (
                args.initial_firm_cash_multiplier
            )
            scenario["applied_parameters"][
                "economy.config.FIRM_INITIAL_CASH"
            ] = {
                "effective": economy_config.FIRM_INITIAL_CASH,
                "source": "--initial-firm-cash-multiplier",
                "multiplier": args.initial_firm_cash_multiplier,
            }

        world = World(
            initial_population=args.population,
            seed=args.seed,
            record_ledger_details=ledger_path is not None,
            scenario_name=scenario["name"],
            scenario_overrides=scenario["overrides"],
            diagnostics_mode=args.diagnostics_mode,
            initial_age_phase_mode=args.initial_age_phase_mode,
            observability_mode=args.observability_mode,
        )
        scenario = apply_runtime_scenario(
            scenario,
            world
        )

        if args.firm_count is not None:

            world.split_firms(args.firm_count)

        if getattr(world, "canonical_investment_enabled", False):
            world.canonical_investment_system.ensure_firms()
            world.ensure_multisector_foundation_contracts()
            print(
                "Resolved runtime profile:",
                getattr(args, "_runtime_profile_name", "legacy/default"),
            )
            print(
                "World construction: Food Firms=",
                len(getattr(world, "firms", [])),
                "Capital-good Firms=",
                len(getattr(world, "capital_good_firms", [])),
                "Sectors=",
                sorted({getattr(firm, "sector_id", "UNAVAILABLE") for firm in world.operating_firms()}),
            )

        if args.steps is not None:

            world.steps = args.steps

    world.household_wealth_instrumentation_enabled = (
        args.household_wealth_instrumentation
    )
    world.household_employer_exposure_instrumentation_enabled = (
        args.household_employer_exposure_instrumentation
    )
    world.household_active_denominator_instrumentation_enabled = (
        args.household_active_denominator_instrumentation
    )
    if args.person_equity_transition:
        world.person_equity_transition_enabled = True
        if not hasattr(world, "equity_transition_system"):
            from economy.equity_transition import GradualPersonEquityTransition
            world.equity_transition_system = GradualPersonEquityTransition(world)
        scenario["overrides"]["PERSON_EQUITY_TRANSITION_ENABLED"] = True
        scenario["applied_parameters"][
            "economy.config.PERSON_EQUITY_TRANSITION_ENABLED"
        ] = {
            "effective": True,
            "source": "--person-equity-transition",
        }
    if args.autonomous_secondary_equity or args.autonomous_secondary_equity_book_price:
        from economy.autonomous_secondary_equity import (
            AutonomousSecondaryEquityPurchaseSystem,
        )
        world.autonomous_secondary_equity_enabled = True
        world.autonomous_secondary_equity_system = (
            AutonomousSecondaryEquityPurchaseSystem(
                world,
                reference_price_mode=(
                    "lagged_book_equity"
                    if args.autonomous_secondary_equity_book_price
                    else "engineering_fixed"
                ),
            )
        )
        scenario["overrides"]["AUTONOMOUS_SECONDARY_EQUITY_ENABLED"] = True
        scenario["applied_parameters"][
            "economy.config.AUTONOMOUS_SECONDARY_EQUITY_ENABLED"
        ] = {
            "effective": True,
            "source": (
                "--autonomous-secondary-equity-book-price"
                if args.autonomous_secondary_equity_book_price
                else "--autonomous-secondary-equity"
            ),
        }
        if args.autonomous_secondary_equity_book_price:
            scenario["overrides"][
                "AUTONOMOUS_SECONDARY_EQUITY_PRICE_MODE"
            ] = "lagged_book_equity"
            scenario["applied_parameters"][
                "economy.config.AUTONOMOUS_SECONDARY_EQUITY_PRICE_MODE"
            ] = {
                "effective": "lagged_book_equity",
                "source": "--autonomous-secondary-equity-book-price",
            }
    if args.generalized_firm_operating_contracts:
        world.generalized_firm_operating_contracts = True
        scenario["overrides"]["GENERALIZED_FIRM_OPERATING_CONTRACTS"] = True
        scenario["applied_parameters"][
            "economy.config.GENERALIZED_FIRM_OPERATING_CONTRACTS"
        ] = {
            "effective": True,
            "source": "--generalized-firm-operating-contracts",
        }
    if args.multisector_foundation:
        world.multisector_foundation_enabled = True
        scenario["overrides"]["MULTISECTOR_FOUNDATION_ENABLED"] = True
        scenario["applied_parameters"][
            "economy.config.MULTISECTOR_FOUNDATION_ENABLED"
        ] = {
            "effective": True,
            "source": "--multisector-foundation",
        }
    if args.shadow_multigood_household_demand:
        world.shadow_multigood_household_demand_enabled = True
        scenario["overrides"][
            "SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED"
        ] = True
        scenario["applied_parameters"][
            "economy.config.SHADOW_MULTIGOOD_HOUSEHOLD_DEMAND_ENABLED"
        ] = {
            "effective": True,
            "source": "--shadow-multigood-household-demand",
        }
    world.ensure_multisector_foundation_contracts()
    world.ensure_household_demand_system()
    if (
        args.persist_diagnostics
        or args.persist_statistical_observability
        or args.observability_mode != "FULL_DIAGNOSTIC"
    ):
        if args.diagnostic_cadence <= 0:
            raise ValueError("--diagnostic-cadence must be positive")
        if args.statistical_snapshot_cadence <= 0:
            raise ValueError("--statistical-snapshot-cadence must be positive")
        if args.statistical_age_cadence <= 0:
            raise ValueError("--statistical-age-cadence must be positive")
        world.configure_diagnostic_persistence(
            os.path.join(output_dir, "canonical_diagnostics"),
            cadence=args.diagnostic_cadence,
            statistical_observability=(
                args.persist_statistical_observability
                or args.observability_mode != "FULL_DIAGNOSTIC"
            ),
            statistical_output_dir=os.path.join(
                output_dir,
                "statistical_observability",
            ),
            statistical_snapshot_cadence=args.statistical_snapshot_cadence,
            statistical_age_cadence=args.statistical_age_cadence,
            observability_mode=args.observability_mode,
        )
        print(
            "Canonical diagnostic persistence:",
            os.path.join(output_dir, "canonical_diagnostics"),
        )
        if args.persist_statistical_observability:
            print(
                "Statistical observability persistence:",
                os.path.join(output_dir, "statistical_observability"),
            )
    if args.household_active_denominator_instrumentation:
        world.household_active_denominator_stream_path = os.path.join(
            output_dir,
            "household_active_denominator",
            "all_active_household_week_denominator.csv",
        )

    start_step = len(world.population_history)

    print("Initial population:",
          len(world.population))

    print("Initial households:",
          len(world.households))

    print("Random seed:",
          world.seed)

    print("Scenario:",
          scenario["name"])

    print("Output directory:",
          output_dir)
    print("Firm count:",
          len(getattr(world, "firms", [])))

    if args.load_checkpoint:

        print("Loaded checkpoint:", args.load_checkpoint)
        print("Checkpoint global step:", start_step)

    # ======================
    # Run Simulation
    # ======================

    print("\nRunning simulation...")
    print("Run weekly steps:", world.steps)

    world.run(
        progress_interval=args.progress_interval
    )

    print("\nSimulation finished")

    print(
        "Final population:",
        len(world.population)
    )

    print(
        "Final households:",
        len(world.households)
    )

    print(
        "Final global step:",
        len(world.population_history)
    )

    if args.save_checkpoint:

        checkpoint_metadata = save_world_checkpoint(
            args.save_checkpoint,
            world,
            extra_metadata={
                "run_steps": world.steps,
                "output_dir": output_dir,
                "diagnostics_csv": diagnostics_path,
                "ledger_csv": ledger_path,
            },
        )
        print(
            "Checkpoint:",
            args.save_checkpoint
        )
        print(
            "Checkpoint global step:",
            checkpoint_metadata["global_step"]
        )

    if diagnostics_path:

        world.export_diagnostics_csv(
            diagnostics_path
        )

        print(
            "Diagnostics CSV:",
            diagnostics_path
        )

    if firm_diagnostics_path:

        world.export_firm_diagnostics_csv(
            firm_diagnostics_path
        )

        print(
            "Firm diagnostics CSV:",
            firm_diagnostics_path
        )

    if household_diagnostics_path:

        print(
            "Writing Household diagnostics CSV (this can be large)...",
            flush=True,
        )

        world.export_household_diagnostics_csv(
            household_diagnostics_path
        )

        print(
            "Household diagnostics CSV:",
            household_diagnostics_path
        )

    if manifest_path:

        export_manifest(
            manifest_path,
            build_manifest(
                scenario=scenario,
                seed=world.seed,
                population=world.initial_population,
                steps=len(world.population_history),
                diagnostics_csv=diagnostics_path,
                firm_diagnostics_csv=firm_diagnostics_path,
                ledger_csv=ledger_path,
                output_dir=output_dir,
                firm_count=len(getattr(world, "firms", [])),
                analysis_start_global_step=start_step,
                initial_age_phase_mode=getattr(
                    world, "initial_age_phase_mode", "legacy_synchronized"
                ),
            ),
        )

        print(
            "Scenario manifest:",
            manifest_path
        )

    if ledger_path:

        print(
            "Writing detailed ledger CSV (this can be large)...",
            flush=True,
        )

        world.ledger.export_csv(
            ledger_path
        )

        print(
            "Ledger CSV:",
            ledger_path
        )

    accounting_dir = os.path.join(output_dir, "accounting")
    accounting_paths = world.export_accounting(accounting_dir)
    print("Accounting outputs:", accounting_dir)

    demographic_paths = world.export_demographic_diagnostics(output_dir)
    print("Demographic diagnostics:", demographic_paths)

    capital_provenance_paths = world.export_capital_provenance_csv(
        os.path.join(output_dir, "capital_provenance")
    )
    print("Capital provenance diagnostics:", capital_provenance_paths)

    if args.household_wealth_instrumentation:
        household_wealth_dir = os.path.join(
            output_dir,
            "household_micro_wealth",
        )
        world.export_household_wealth_instrumentation(
            household_wealth_dir
        )
        print("Household wealth instrumentation:", household_wealth_dir)

    if args.household_employer_exposure_instrumentation:
        employer_exposure_dir = os.path.join(
            output_dir,
            "household_employer_exposure",
        )
        world.export_household_employer_exposure(
            employer_exposure_dir
        )
        print("Household employer exposure:", employer_exposure_dir)

    if args.household_active_denominator_instrumentation:
        denominator_dir = os.path.join(
            output_dir,
            "household_active_denominator",
        )
        world.export_household_active_denominator(denominator_dir)
        print("Household active denominator:", denominator_dir)

    analysis_semantics_path = export_time_semantics_manifest(
        os.path.join(output_dir, "analysis_time_semantics.json"),
        start_step,
        len(world.population_history),
        source_kind=(
            "weekly_checkpoint_continuation"
            if args.load_checkpoint
            else "fresh_weekly_simulation"
        ),
    )
    print("Analysis time semantics:", analysis_semantics_path)

    step15e3_result = None
    if args.step15e3_human_review:
        step15e3_result = run_step15e3_human_review(
            world,
            args.step15e3_human_review_output,
        )
        print(
            "Step 15E.3 human-review output:",
            step15e3_result["output_dir"],
        )
        print("Step 15E.3 verdict:", step15e3_result["verdict"])

    summary = world.diagnostics_summary()

    print(
        "Diagnostics rows:",
        summary["rows"]
    )

    if args.diagnostics_mode == "compact":

        print(
            "Compact diagnostics mode: conservation gap summary skipped."
        )

    else:

        print(
            "Invariant violations:",
            summary["invariant_violations"]
        )

        print(
            "Max abs monetary accounting gap:",
            f"{summary['max_abs_monetary_accounting_gap']:.6g}"
        )

        print(
            "Max abs money delta gap:",
            f"{summary['max_abs_money_delta_gap']:.6g}"
        )

        print(
            "Max abs ledger money net gap:",
            f"{summary['max_abs_ledger_money_net_gap']:.6g}"
        )

        print(
            "Max abs food conservation gap:",
            f"{summary['max_abs_food_conservation_gap']:.6g}"
        )

        print(
            "Max abs income spending gap:",
            f"{summary['max_abs_income_spending_gap']:.6g}"
        )

        print(
            "Max abs sales revenue split gap:",
            f"{summary['max_abs_sales_revenue_split_gap']:.6g}"
        )

    if args.no_analysis:

        print("\nAnalysis skipped")
        raise SystemExit(0)

    # ======================
    # Analyzer
    # ======================

    analysis_v2_context = None
    analysis_v2_acceptance = None
    gui_handoff_dir = None
    try:
        print(
            f"Starting Analysis v2 ({args.analysis_profile} profile)...",
            flush=True,
        )
        analysis_v2_context, analysis_v2_acceptance = run_analysis_v2(
            world,
            output_dir,
            profile=args.analysis_profile,
            plot_mode=args.plot_mode,
            no_plots=args.no_plots,
        )
        from analysis.gui.handoff import write_gui_compatibility_panels
        write_gui_compatibility_panels(analysis_v2_context, output_dir)
        gui_handoff_dir = output_dir
        print("Analysis v2 output:", os.path.join(output_dir, "analysis"))
        financial_summary = getattr(
            analysis_v2_context,
            "financial_core_summary",
            None,
        )
        if financial_summary:
            print()
            for line in format_financial_core_console(financial_summary):
                print(line)
    except Exception as exc:
        failure_path = write_analysis_failure(output_dir, exc, "analysis_v2")
        print("Analysis v2 failed after simulation completion:", failure_path)

    analyzer = Analyzer(world)
    analyzer.analysis_context = analysis_v2_context

    # Visualization is an Analysis responsibility.  The simulation domain
    # only persists authoritative state; Analysis v2 already generated the
    # requested figures above.

    # =====================================================
    # Demography Analysis
    # =====================================================

    print("\n======================")
    print("Demography Analysis")
    print("======================")

    analyzer.equilibrium_report()

    print("\n======================")
    print("Fertility Analysis")
    print("======================")

    analyzer.completed_fertility_report()

    if not args.no_plots and args.plot_mode == "individual":

        analyzer.plot_age_heatmap()

        analyzer.plot_population_pyramid()

        analyzer.plot_dependency_ratio()

        analyzer.plot_completed_fertility_distribution()

    # =====================================================
    # Economy Analysis
    # =====================================================

    print("\n======================")
    print("Economy Analysis")
    print("======================")

    analyzer.economy_report()

    analyzer.steady_state_report(
        window=args.steady_window
    )

    if not args.no_plots and args.plot_mode == "individual":

        analyzer.plot_macro_economy()

        analyzer.plot_economy_ratio()

        analyzer.plot_household_economy_distribution()

        analyzer.plot_household_economy_diagnostics()

        analyzer.plot_household_inequality_diagnostics()

    # =====================================================
    # Firm Analysis
    # =====================================================

    print("\n======================")
    print("Firm Analysis")
    print("======================")

    analyzer.firm_report()
    analyzer.accounting_report()
    analyzer.multi_firm_report()

    if not args.no_plots and args.plot_mode == "individual":

        analyzer.plot_firm_diagnostics()
        analyzer.plot_multi_firm_diagnostics()
        analyzer.plot_multi_firm_market_shares()

        analyzer.plot_firm_balance_sheet()
        analyzer.plot_accounting_diagnostics()

        analyzer.plot_monetary_system()

        analyzer.plot_household_money_stock()

        analyzer.plot_central_bank_food_reserve()

    # =====================================================
    # Finish
    # =====================================================

    print("\n======================")
    print("Analysis Finished")
    print("======================")

    if step15e3_result is not None:
        flags = step15e3_result["flags"]
        print("\nStep 15E.3 Human Review Summary")
        print("MACRO: existing diagnostics aggregated")
        print("LABOR: sector table written; inactive sectors shown explicitly")
        print("FINAL DEMAND: investment/government/external/intermediate are PASSIVE / INACTIVE")
        print(
            "OWNERSHIP:",
            "runtime views loaded; Person ownership remains passive unless a fixture is used",
        )
        print("CAPITAL / INVESTMENT STATUS: PASSIVE / INACTIVE")
        print(
            "MONEY / ACCOUNTING RECONCILIATION:",
            "PASS" if flags["money_reconciliation_pass"] and flags["accounting_reconciliation_pass"] else "CHECK FLAGS",
        )

    if getattr(args, "_auto_launch_gui", False):
        if gui_handoff_dir is None:
            print("Analysis GUI was not launched because post-run Analysis failed.")
            print("Completed run directory:", output_dir)
        else:
            print("Launching Analysis GUI for completed run:", gui_handoff_dir)
            from analysis.gui import launch_analysis_gui
            raise SystemExit(
                launch_analysis_gui(
                    run_dir=gui_handoff_dir,
                    output_root=DEFAULT_OUTPUT_ROOT,
                )
            )
