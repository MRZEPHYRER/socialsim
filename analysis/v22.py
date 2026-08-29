"""Analysis V2.2 detailed, passive figures and canonical plot manifest."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _f(row, key, default=0.0):
    try: return float(row.get(key, default) or default)
    except (TypeError, ValueError): return default


def _write_manifest(path, entries):
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_detailed_plots(context):
    """Generate useful detail plots from cached context tables in one pass."""
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = context.analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    macro = context.tables.get("weekly_macro", [])
    firms = context.tables.get("weekly_firm", [])
    households = context.tables.get("household_snapshot", [])
    x = [_f(r, "simulation_week", _f(r, "step")) for r in macro]
    by_firm = defaultdict(list)
    for r in firms:
        # Derive display-only states from already recorded cash/interest fields;
        # this does not feed back into the simulation.
        row = dict(r)
        obligation = _f(row, "total_interest_obligation")
        paid = _f(row, "interest_paid")
        unpaid = _f(row, "current_interest_unpaid")
        row["interest_service_state"] = 0 if obligation <= 1e-9 else (1 if unpaid <= 1e-9 else (2 if paid > 1e-9 else 3))
        requested = _f(row, "requested_credit")
        denied = _f(row, "denied_credit")
        payroll_shortfall = _f(row, "payroll_cash_shortfall")
        row["financing_state"] = 0 if requested <= 1e-9 else (1 if denied <= 1e-9 else (2 if payroll_shortfall <= 1e-9 else 3))
        by_firm[str(row.get("firm_id"))].append(row)
    entries = []

    def save(category, plot_id, label, filename, role="detailed_diagnostic", required=()):
        path = plot_dir / filename
        entries.append({"category": category, "plot_id": plot_id, "plot_label": label,
                        "filename": str(path.relative_to(context.analysis_dir)),
                        "source_function": "analysis.v22.generate_detailed_plots",
                        "availability": True, "required_fields": list(required),
                        "default_visibility": True, "plot_role": role})
        return path

    def line(category, pid, label, filename, series, required=()):
        path=save(category,pid,label,filename,required=required); fig,ax=plt.subplots(figsize=(9,4.5))
        for field, name in series:
            ax.plot(x, [_f(r,field) for r in macro], label=name)
        ax.set_title(label); ax.set_xlabel("simulation week"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)

    # Demography and age structure.
    line("Demography","population","Population trajectory","demography_population_trajectory.png",[("population","population")],("population",))
    line("Demography","births_deaths","Births and deaths","demography_births_deaths.png",[("births","births"),("deaths","deaths")],("births","deaths"))
    line("Demography","natural_growth","Natural growth","demography_natural_growth.png",[("births","births"),("deaths","deaths")],("births","deaths"))
    line("Demography","age_groups","Age-group trajectories","demography_age_groups.png",[("working_age_population","working age"),("elderly_population","elderly")],("working_age_population",))
    line("Demography","dependency","Dependency ratio","demography_dependency_ratio.png",[("dependency_ratio","dependency ratio")],("dependency_ratio",))
    annual = []
    p = context.output_dir / "demographic_annual_summary.csv"
    if p.exists():
        with p.open(encoding="utf-8-sig", newline="") as h: annual=list(csv.DictReader(h))
    if annual:
        path=save("Demography","age_heatmap","Age heatmap","demography_age_heatmap.png",required=("age_0_19_count","age_20_39_count","age_40_64_count","age_65_plus_count"))
        mat=np.array([[_f(r,k) for k in ("age_0_19_count","age_20_39_count","age_40_64_count","age_65_plus_count")] for r in annual]).T
        fig,ax=plt.subplots(figsize=(9,4.5)); im=ax.imshow(mat,aspect="auto",origin="lower"); ax.set_yticks(range(4),["0-19","20-39","40-64","65+"]); ax.set_xlabel("simulation year"); ax.set_title("Age structure heatmap"); fig.colorbar(im,ax=ax,label="population"); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
        path=save("Demography","population_pyramid","Population pyramid","demography_population_pyramid.png",required=("age_0_19_count","age_20_39_count","age_40_64_count","age_65_plus_count"))
        vals=[_f(annual[-1],k) for k in ("age_0_19_count","age_20_39_count","age_40_64_count","age_65_plus_count")]; fig,ax=plt.subplots(figsize=(7,5)); ax.barh(["0-19","20-39","40-64","65+"],vals); ax.set_title(f"Final population pyramid, week {int(x[-1]) if x else 0}"); ax.set_xlabel("population count"); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    # Households.
    vals={k:[_f(r,k) for r in households] for k in ("wealth","income","consumption","saving","household_size")}
    for field,label in (("wealth","Wealth distribution"),("income","Income distribution"),("consumption","Consumption distribution"),("saving","Saving distribution"),("household_size","Household size distribution")):
        path=save("Households",field,label,f"households_{field}.png",required=(field,)); fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(vals[field],bins=30); ax.set_title(label); ax.set_xlabel(field); ax.set_ylabel("households"); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    for field,label in (("wealth","Wealth Lorenz curve"),("income","Income Lorenz curve")):
        z=sorted(v for v in vals[field] if v>=0); total=sum(z) or 1; path=save("Households",field+"_lorenz",label,f"households_{field}_lorenz.png",required=(field,)); fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0]+[(i+1)/len(z) for i in range(len(z))],[0]+[sum(z[:i+1])/total for i in range(len(z))]); ax.plot([0,1],[0,1],"--"); ax.set_title(label); ax.set_xlabel("population share"); ax.set_ylabel("income share"); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    # Real economy and prices.
    line("Real Economy","production_sales_consumption","Production vs Sales vs Consumption","real_production_sales_consumption.png",[("production","production"),("sales","sales"),("consumption","consumption")],("production","sales","consumption"))
    line("Real Economy","per_capita","Per-capita production, sales, consumption","real_per_capita.png",[("production_per_capita","production pc"),("sales_per_capita","sales pc"),("consumption_per_capita","consumption pc")])
    line("Real Economy","inventory_demand","Inventory and demand","real_inventory_demand.png",[("inventory","inventory"),("unmet_demand","unmet demand")])
    line("Real Economy","saving_rate","Saving and consumption rates","real_saving_consumption_rate.png",[("household_wealth","household wealth")])
    line("Prices","planning_realized","Planning vs realized price","prices_planning_realized.png",[("planning_price","planning price"),("realized_transaction_price","transaction price")])
    # Firm grouped plots.
    def firmline(category,pid,label,filename,field,required=()):
        path=save(category,pid,label,filename,required=required); fig,ax=plt.subplots(figsize=(9,4.5))
        for fid,rs in sorted(by_firm.items()): ax.plot([_f(r,"step") for r in rs],[_f(r,field) for r in rs],label=f"Firm {fid}")
        ax.set_title(label); ax.set_xlabel("simulation week"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    for field,label in (("market_share","Market share by Firm"),("choice_probability","Choice probability by Firm"),("sales","Sales by Firm"),("production","Production by Firm"),("cash","Firm cash"),("profit","Firm profit"),("inventory","Firm inventory"),("inventory_coverage","Inventory coverage by Firm"),("closing_principal","Principal by Firm"),("principal_utilization","Principal utilization by Firm"),("price","Posted Firm prices"),("credit_headroom","Credit headroom by Firm"),("payroll_funding_ratio","Payroll funding by Firm")):
        cat="Firm Competition" if field in ("market_share","choice_probability","sales","production","price") else ("Financial Core" if field == "principal_utilization" else ("Credit" if field in ("closing_principal","credit_headroom","payroll_funding_ratio") else ("Inventory" if field in ("inventory","inventory_coverage") else "Firm Operations")))
        firmline(cat,field,label,f"{cat.lower().replace(' ','_')}_{field}.png",field,(field,))
    # State heatmaps from raw firm rows where fields exist.
    def heat(cat,pid,label,filename,field,states):
        path=save(cat,pid,label,filename,required=(field,)); ids=sorted(by_firm); matrix=[]
        for fid in ids:
            values=[]
            for row in by_firm[fid]:
                raw=row.get(field)
                values.append(states.index(raw) if raw in states else _f(row,field))
            matrix.append(values)
        fig,ax=plt.subplots(figsize=(10,3.5)); im=ax.imshow(np.array(matrix),aspect="auto",interpolation="nearest"); ax.set_yticks(range(len(ids)),[f"Firm {i}" for i in ids]); ax.set_title(label); ax.set_xlabel("time"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    heat("Credit","financing_state_heatmap","Financing state heatmap","credit_financing_state_heatmap.png","financing_state",("STATE0","STATE1","STATE2","STATE3"))
    heat("Interest","interest_state_heatmap","Interest-service state heatmap","interest_service_state_heatmap.png","interest_service_state",("I0","I1","I2","I3"))
    heat("Financial Core","distress_state_heatmap","Distress state heatmap","financial_core_distress_state_heatmap.png","distress_state",("D0","D1","D2","D3"))
    heat("Financial Core","contract_default_heatmap","Active contractual Default heatmap","financial_core_contract_default_heatmap.png","active_contract_default",(False,True))
    for field,label in (("closing_interest_arrears","Interest arrears by Firm"),("current_interest_due","Current interest due by Firm"),("interest_paid","Interest paid by Firm"),("lender_exposure","Principal plus arrears exposure")):
        firmline("Interest",field,label,f"interest_{field}.png",field,(field,))
    for field,label in (("total_money","Total money stock"),("credit_created_money","Credit-created money"),("household_money","Household money"),("firm_money","Aggregate Firm cash"),("public_or_cb_monetary_balance","Public monetary balance"),("money_location_gap","Located money gap")):
        line("Money",field,label,f"money_{field}.png",[(field,label)],(field,))
    for field,label in (("monetary_accounting_gap","Monetary accounting gap"),("money_delta_gap","Money delta gap"),("goods_conservation_gap","Goods conservation gap")):
        line("Accounting",field,label,f"accounting_{field}.png",[(field,label)],(field,))
    # Add overview entries from existing canonical 11 files.
    overview=[]
    for p in sorted(plot_dir.glob("*.png")):
        if not p.name[:1].isdigit():
            continue
        category = "Financial Core" if p.name.startswith(("09_","10_","11_","12_")) else ("Accounting" if p.name.startswith("13_") else "Overview")
        overview.append({"category":category,"plot_id":p.stem,"plot_label":p.stem.replace("_"," ").title(),"filename":str(p.relative_to(context.analysis_dir)),"source_function":"analysis.v2._plots","availability":True,"required_fields":[],"default_visibility":True,"plot_role":"overview"})
    entries=overview+entries
    _write_manifest(context.analysis_dir/"analysis_plot_manifest.json",entries)
    return [context.analysis_dir/e["filename"] for e in entries], entries
