"""Application entry for the Step 15 desktop Analysis workbench."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from analysis.gui.main_window import Step15AnalysisMainWindow


def launch_analysis_gui(run_dir=None, output_root="test/output", auto_close_ms=None):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SOCIALSIM Analysis")
    app.setOrganizationName("SOCIALSIM")
    try:
        window = Step15AnalysisMainWindow(run_dir=run_dir, output_root=output_root)
    except Exception as exc:
        QMessageBox.critical(None, "SOCIALSIM Analysis", str(exc))
        return 2
    window.show()
    delay = auto_close_ms
    if delay is None:
        value = os.getenv("SOCIALSIM_GUI_AUTO_CLOSE_MS")
        delay = int(value) if value else None
    if delay is not None:
        QTimer.singleShot(int(delay), window.close)
    return app.exec()

