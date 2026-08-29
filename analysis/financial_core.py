"""Passive Step 13 financial-core analysis built from recorded diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


STEP13_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "output"
    / "step13_final_financial_core_closure"
    / "step13_architecture_manifest.json"
)
KNOWN_PRINCIPAL_LIMITATION = (
    "Current loan_balance mixes persistent debt and working-capital revolving "
    "utilization; the term/revolver split is deferred to generalized "
    "multi-sector corporate finance."
)


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _truth(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _sum(rows, field):
    return math.fsum(_number(row.get(field), 0.0) for row in rows)


def _last(rows, field, default=None):
    return _number(rows[-1].get(field), default) if rows else default


def _percentile(values, percentile):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def load_step13_contract(path=STEP13_CONTRACT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Step 13 architecture manifest is required: {path}"
        )
    payload = path.read_bytes()
    contract = json.loads(payload.decode("utf-8"))
    required = {
        "credit",
        "money_creation",
        "principal",
        "interest",
        "arrears",
        "distress",
        "default",
        "contractual_cure",
        "baseline_reference",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(
            "Step 13 architecture manifest is incomplete: " + ", ".join(missing)
        )
    return contract, {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_version": contract.get("manifest_version"),
        "closure_verdict": contract.get("closure_verdict"),
    }


def build_financial_core_metrics(context):
    contract, contract_identity = load_step13_contract()
    macro = context.tables.get("weekly_macro", [])
    firms = context.tables.get("weekly_firm", [])
    mature_start = context.mature_start_week
    mature_firms = [
        row for row in firms
        if (_number(row.get("step"), -1) or -1) >= mature_start
    ]
    utilization = [
        _number(row.get("principal_utilization")) for row in mature_firms
    ]
    utilization = [value for value in utilization if value is not None]
    firm_weeks = len(mature_firms)

    distress_counts = {
        state: sum(str(row.get("distress_state")) == state for row in mature_firms)
        for state in ("D0", "D1", "D2", "D3")
    }
    summary = {
        "contract_manifest_path": contract_identity["path"],
        "contract_manifest_sha256": contract_identity["sha256"],
        "contract_closure_verdict": contract_identity["closure_verdict"],
        "final_population": _last(macro, "population"),
        "final_active_households": _last(macro, "active_households"),
        "final_household_wealth": _last(macro, "household_wealth"),
        "final_outstanding_principal": _last(macro, "principal"),
        "final_interest_arrears": _last(macro, "interest_arrears"),
        "final_total_lender_exposure": _last(macro, "lender_exposure"),
        "final_revolving_exposure": _last(macro, "revolving_exposure"),
        "final_legacy_term_claim": _last(macro, "legacy_term_claim"),
        "final_credit_limit": _last(macro, "credit_limit"),
        "final_credit_headroom": _last(macro, "credit_headroom"),
        "gross_credit_issued": _sum(macro, "gross_loan_issuance"),
        "principal_repaid": _sum(macro, "principal_repayment"),
        "net_credit_money_creation": _sum(macro, "net_credit_money_creation"),
        "interest_due": _sum(macro, "interest_due"),
        "interest_paid": _sum(macro, "interest_paid"),
        "new_interest_arrears": _sum(macro, "arrears_formation"),
        "arrears_paid": _sum(macro, "arrears_payment"),
        "central_bank_interest_income": _last(
            macro, "cumulative_central_bank_interest_income"
        ),
        "credit_denied": _sum(macro, "denied_credit"),
        "default_event_count": sum(
            _truth(row.get("default_event")) for row in firms
        ),
        "active_contract_default_firm_weeks": sum(
            _truth(row.get("active_contract_default")) for row in firms
        ),
        "contractual_cure_event_count": sum(
            _truth(row.get("contract_cure")) for row in firms
        ),
        "mature_firm_weeks": firm_weeks,
        "mature_d0_share": distress_counts["D0"] / firm_weeks if firm_weeks else None,
        "mature_d1_share": distress_counts["D1"] / firm_weeks if firm_weeks else None,
        "mature_d2_share": distress_counts["D2"] / firm_weeks if firm_weeks else None,
        "mature_d3_share": distress_counts["D3"] / firm_weeks if firm_weeks else None,
        "mature_principal_utilization_median": _percentile(utilization, 0.50),
        "mature_principal_utilization_p90": _percentile(utilization, 0.90),
        "mature_principal_utilization_share_ge_90pct": (
            sum(value >= 0.90 for value in utilization) / len(utilization)
            if utilization else None
        ),
        "mature_principal_utilization_share_ge_95pct": (
            sum(value >= 0.95 for value in utilization) / len(utilization)
            if utilization else None
        ),
        "mature_principal_utilization_share_ge_100pct": (
            sum(value >= 1.00 for value in utilization) / len(utilization)
            if utilization else None
        ),
        "max_abs_money_accounting_gap": max(
            (abs(_number(row.get("monetary_accounting_gap"), 0.0)) for row in macro),
            default=0.0,
        ),
        "max_abs_money_delta_gap": max(
            (abs(_number(row.get("money_delta_gap"), 0.0)) for row in macro),
            default=0.0,
        ),
        "max_abs_goods_conservation_gap": max(
            (abs(_number(row.get("goods_conservation_gap"), 0.0)) for row in macro),
            default=0.0,
        ),
        "invariant_violations": sum(
            _truth(row.get("invariant_failed")) for row in macro
        ),
        "known_principal_limitation": KNOWN_PRINCIPAL_LIMITATION,
    }
    return summary, contract, contract_identity


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_financial_core_analysis(context):
    summary, contract, contract_identity = build_financial_core_metrics(context)
    context.financial_core_summary = summary
    context.step13_contract = contract
    context.step13_contract_identity = contract_identity

    summary_rows = [
        {"metric": key, "value": value}
        for key, value in summary.items()
    ]
    _write_csv(
        context.analysis_dir / "tables" / "financial_core_summary.csv",
        summary_rows,
    )

    firm_rows = []
    source_firms = context.tables.get("weekly_firm", [])
    firm_ids = sorted({str(row.get("firm_id")) for row in source_firms})
    for firm_id in firm_ids:
        rows = [row for row in source_firms if str(row.get("firm_id")) == firm_id]
        mature = [
            row for row in rows
            if (_number(row.get("step"), -1) or -1) >= context.mature_start_week
        ]
        values = [
            _number(row.get("principal_utilization")) for row in mature
        ]
        values = [value for value in values if value is not None]
        final = rows[-1]
        firm_rows.append({
            "firm_id": firm_id,
            "mature_observations": len(mature),
            "utilization_median": _percentile(values, 0.50),
            "utilization_p90": _percentile(values, 0.90),
            "utilization_share_ge_90pct": (
                sum(value >= 0.90 for value in values) / len(values)
                if values else None
            ),
            "utilization_share_ge_95pct": (
                sum(value >= 0.95 for value in values) / len(values)
                if values else None
            ),
            "utilization_share_ge_100pct": (
                sum(value >= 1.00 for value in values) / len(values)
                if values else None
            ),
            "final_principal": final.get("closing_principal"),
            "final_credit_limit": final.get("credit_limit"),
            "final_credit_headroom": final.get("credit_headroom"),
            "final_interest_arrears": final.get("closing_interest_arrears"),
            "final_lender_exposure": final.get("lender_exposure"),
            "final_distress_state": final.get("distress_state"),
            "final_active_contract_default": final.get("active_contract_default"),
        })
    _write_csv(
        context.analysis_dir / "tables" / "financial_core_firm_summary.csv",
        firm_rows,
    )

    semantics = {
        "category": "Financial Core",
        "step13_contract": contract_identity,
        "metric_classes": {
            "stock": [
                "principal", "interest_arrears", "lender_exposure",
                "revolving_exposure", "legacy_term_claim", "total_money",
            ],
            "flow": [
                "gross_loan_issuance", "principal_repayment",
                "net_credit_money_creation", "interest_due", "interest_paid",
                "arrears_formation", "arrears_payment",
                "central_bank_interest_income",
            ],
            "state": [
                "distress_state", "technical_interest_breach",
                "active_contract_default", "default_event", "contract_cure",
            ],
        },
        "canonical_identities": {
            "net_credit_money_creation": (
                "gross_loan_issuance - principal_repayment"
            ),
            "weekly_interest_rate": "(1 + annual_rate) ** (1 / 52) - 1",
            "current_interest_due": "opening_principal * weekly_interest_rate",
            "baseline_lender_exposure": "principal + interest_arrears",
        },
        "annualization": {
            "stocks": "year-end snapshot",
            "flows": "sum of weekly flows",
            "rates": "explicitly defined only",
        },
        "experimental_baseline": contract.get("default_feature_flags", {}),
        "known_limitations": [KNOWN_PRINCIPAL_LIMITATION],
    }
    (context.analysis_dir / "financial_core_semantics.json").write_text(
        json.dumps(semantics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def format_financial_core_console(summary):
    def display(name, key):
        value = summary.get(key)
        if isinstance(value, float):
            value = f"{value:.6g}"
        return f"{name}: {value}"

    def ratio_value(key):
        value = summary.get(key)
        return f"{value:.6g}" if isinstance(value, (int, float)) else "NA"

    return [
        "Financial Core",
        "--------------",
        display("Outstanding Principal", "final_outstanding_principal"),
        display("Interest Arrears", "final_interest_arrears"),
        display("Total Lender Exposure", "final_total_lender_exposure"),
        display("Gross Credit Issued", "gross_credit_issued"),
        display("Principal Repaid", "principal_repaid"),
        display("Net Credit Money Creation", "net_credit_money_creation"),
        display("Interest Due", "interest_due"),
        display("Interest Paid", "interest_paid"),
        display("CentralBank Interest Income", "central_bank_interest_income"),
        display("Credit Denied", "credit_denied"),
        (
            "Mature D2 / D3 Share: "
            f"{ratio_value('mature_d2_share')} / "
            f"{ratio_value('mature_d3_share')}"
        ),
        display("Default Events", "default_event_count"),
        display(
            "Active Contract Default Firm-Weeks",
            "active_contract_default_firm_weeks",
        ),
    ]


def generate_standard_financial_core_plots(context):
    """Generate exactly four compact standard-profile Step 13 figures."""

    import matplotlib.pyplot as plt

    plot_dir = context.analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    macro = context.tables.get("weekly_macro", [])
    x = [_number(row.get("simulation_week"), _number(row.get("step"), 0.0)) for row in macro]
    entries = []

    def values(field):
        return [_number(row.get(field), math.nan) for row in macro]

    def save(fig, plot_id, label, filename, required):
        path = plot_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        entries.append({
            "category": "Financial Core",
            "plot_id": plot_id,
            "plot_label": label,
            "filename": str(path.relative_to(context.analysis_dir)),
            "source_function": "analysis.financial_core.generate_standard_financial_core_plots",
            "availability": True,
            "required_fields": list(required),
            "default_visibility": True,
            "plot_role": "standard_financial_core",
        })
        return path

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for field, label in (
        ("principal", "principal"),
        ("credit_limit", "credit limit"),
        ("credit_headroom", "headroom"),
    ):
        axes[0].plot(x, values(field), label=label)
    for field, label in (
        ("gross_loan_issuance", "borrowing"),
        ("principal_repayment", "repayment"),
        ("denied_credit", "denied"),
    ):
        axes[1].plot(x, values(field), label=label)
    axes[0].set_title("Credit stock and capacity")
    axes[1].set_title("Weekly credit flows")
    axes[1].set_xlabel("simulation week")
    for axis in axes:
        axis.legend(ncol=3)
    credit_path = save(
        fig, "credit_stock_flow", "Credit stock and flow",
        "09_financial_credit_stock_flow.png",
        ("principal", "credit_limit", "gross_loan_issuance", "principal_repayment"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for field, label in (
        ("interest_due", "current due"),
        ("interest_paid", "paid"),
        ("current_interest_unpaid", "unpaid"),
    ):
        axes[0].plot(x, values(field), label=label)
    axes[1].plot(x, values("interest_arrears"), label="arrears stock")
    axes[1].plot(
        x,
        values("cumulative_central_bank_interest_income"),
        label="cumulative CB interest income",
    )
    axes[0].set_title("Weekly interest service")
    axes[1].set_title("Arrears and lender cash income")
    axes[1].set_xlabel("simulation week")
    for axis in axes:
        axis.legend(ncol=3)
    interest_path = save(
        fig, "interest_arrears", "Interest service and arrears",
        "10_financial_interest_arrears.png",
        ("interest_due", "interest_paid", "interest_arrears"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for field, label in (
        ("distress_d0_share", "D0"),
        ("distress_d1_share", "D1"),
        ("distress_d2_share", "D2"),
        ("distress_d3_share", "D3"),
    ):
        axes[0].plot(x, values(field), label=label)
    for field, label in (
        ("active_contract_default_firm_count", "active contract default"),
        ("default_event_count", "default events"),
        ("contract_cure_count", "contractual cures"),
    ):
        axes[1].plot(x, values(field), label=label)
    axes[0].set_title("Firm financial distress shares")
    axes[1].set_title("Contractual Default state")
    axes[1].set_xlabel("simulation week")
    for axis in axes:
        axis.legend(ncol=4)
    stress_path = save(
        fig, "distress_default", "Financial distress and Default",
        "11_financial_distress_default.png",
        ("distress_d3_share", "active_contract_default_firm_count"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for field, label in (
        ("gross_money_created_by_credit", "money created"),
        ("money_destroyed_by_principal_repayment", "money destroyed"),
        ("net_credit_money_creation", "net credit money"),
    ):
        axes[0].plot(x, values(field), label=label)
    for field, label in (
        ("lender_exposure", "lender exposure"),
        ("revolving_exposure", "revolving exposure"),
        ("total_money", "total money"),
    ):
        axes[1].plot(x, values(field), label=label)
    axes[0].set_title("Monetary credit flows")
    axes[1].set_title("Lender exposure and money stock")
    axes[1].set_xlabel("simulation week")
    for axis in axes:
        axis.legend(ncol=3)
    money_path = save(
        fig, "money_lender_exposure", "Money and lender exposure",
        "12_financial_money_lender_exposure.png",
        ("net_credit_money_creation", "lender_exposure", "total_money"),
    )

    return [credit_path, interest_path, stress_path, money_path], entries
