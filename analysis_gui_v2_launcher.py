"""Standalone entry point for the frozen SOCIALSIM Analysis GUI V2."""

from __future__ import annotations

from multiprocessing import freeze_support

from analysis.gui_v2 import launch_analysis_gui_v2


def main() -> int:
    freeze_support()
    return int(launch_analysis_gui_v2())


if __name__ == "__main__":
    raise SystemExit(main())
