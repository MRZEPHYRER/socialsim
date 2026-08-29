"""Application entry point for the independent Analysis GUI V2."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from analysis.gui_v2.main_window import AnalysisV2MainWindow
from analysis.gui_v2.localization import tr
from analysis.step15_reference import resolve_step15_analysis_reference


def launch_analysis_gui_v2(run_dir=None, auto_close_ms=None):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SOCIALSIM Analysis V2")
    try:
        window = AnalysisV2MainWindow(run_dir=run_dir or resolve_step15_analysis_reference())
    except Exception as exc:
        QMessageBox.critical(None, tr("app_title", "zh"), str(exc))
        return 2
    window.show()
    delay = auto_close_ms
    if delay is None:
        configured = os.getenv("SOCIALSIM_GUI_V2_AUTO_CLOSE_MS")
        delay = int(configured) if configured else None
    if delay is not None:
        QTimer.singleShot(int(delay), window.close)
    return app.exec()


__all__ = ["launch_analysis_gui_v2"]
