# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve()
ENRICHED_RUN = PROJECT_ROOT / "test/output/step15_final_canonical_statistics_enriched"
LEGACY_RUN = PROJECT_ROOT / "test/output/step15_final_canonical_post_dynamic_validation"
CONSOLIDATION = PROJECT_ROOT / "test/output/step15_final_canonical_consolidation"

ANALYSIS_DATA = (
    "step15_macro_panel.csv",
    "step15_firm_panel.csv",
    "step15_demographic_panel.csv",
    "step15_demographic_events.csv",
    "step15_marriage_panel.csv",
    "step15_accounting_reconciliation.csv",
    "step15_capital_asset_ledger.csv",
    "step15_capital_provenance_events.csv",
    "step15_investment_chain_trace.csv",
    "statistical_observability/age_sex_histogram.csv",
    "statistical_observability/social_household_snapshots.csv",
    "statistical_observability/settlement_account_snapshots.csv",
    "statistical_observability/labor_events.csv",
    "statistical_observability/labor_denominators.csv",
    "statistical_observability/firm_capital_history.csv",
)

CONTRACT_DATA = (
    "authoritative_analysis_data_contract.csv",
    "analysis_availability_matrix.csv",
    "household_semantic_registry.csv",
)


def bundled_file(source_root, relative_path, destination_root):
    source = source_root / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Required Analysis GUI V2 bundle input is missing: {source}")
    destination = Path(destination_root) / Path(relative_path).parent
    return str(source), str(destination)


datas = [
    bundled_file(
        ENRICHED_RUN,
        relative_path,
        "test/output/step15_final_canonical_statistics_enriched",
    )
    for relative_path in ANALYSIS_DATA
]
datas.extend(
    bundled_file(
        CONSOLIDATION,
        relative_path,
        "test/output/step15_final_canonical_consolidation",
    )
    for relative_path in CONTRACT_DATA
)

# CanonicalDataStore keeps the frozen pre-enrichment run in its allow-list.
# A tiny real source file preserves that directory contract without bundling
# the unused historical dataset alongside the enriched reference run.
datas.append(
    bundled_file(
        LEGACY_RUN,
        "step15_marriage_panel.csv",
        "test/output/step15_final_canonical_post_dynamic_validation",
    )
)

a = Analysis(
    [str(PROJECT_ROOT / "analysis_gui_v2_launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["matplotlib.backends.backend_qtagg"],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "tkinter",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "datasets",
        "tensorflow",
        "sklearn",
        "scipy",
        "sympy",
        "pytest",
        "_pytest",
        "IPython",
        "jupyter",
        "jupyter_client",
        "jupyter_core",
        "notebook",
        "nbformat",
        "zmq",
        "pyarrow",
        "openpyxl",
        "fsspec",
        "dask",
        "xarray",
        "numba",
        "llvmlite",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SOCIALSIM_Analysis_V2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
