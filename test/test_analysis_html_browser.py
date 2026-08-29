from pathlib import Path

from analysis.plot_browser import build_html_plot_browser


def test_html_plot_browser_is_persistent_local_file(tmp_path):
    plot = tmp_path / "plots" / "macro.png"
    plot.parent.mkdir()
    plot.write_bytes(b"fixture")
    dashboard, count = build_html_plot_browser(
        [{
            "path": plot,
            "category": "Macro",
            "plot_label": "Macro overview",
        }],
        metadata={"seed": 42, "profile": "full"},
        output_path=tmp_path / "analysis_dashboard.html",
    )
    content = dashboard.read_text(encoding="utf-8")
    assert count == 1
    assert "Macro overview" in content
    assert "seed" in content and "42" in content
    assert "plots/macro.png" in content
    assert "http://" not in content and "https://" not in content
    assert "No compatible Analysis plots" in content
