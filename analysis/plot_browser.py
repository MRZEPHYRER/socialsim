"""Single-window browser for already-rendered Analysis V2 dashboards."""

import html
import json
import os
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk


class AnalysisPlotBrowser:
    def __init__(self, plot_paths, metadata=None, title="SOCIALSIM - Analysis V2 Dashboard"):
        self.entries = []
        for item in plot_paths:
            if isinstance(item, dict):
                path = Path(item.get("path") or item.get("filename", ""))
                entry = dict(item)
                entry["path"] = path
            else:
                path = Path(item)
                entry = {"path": path, "category": "Overview", "plot_label": path.stem.replace("_", " ").title()}
            if path.exists():
                self.entries.append(entry)
        self.plot_paths = [entry["path"] for entry in self.entries]
        self.metadata = metadata or {}
        self.title = title
        self.index = 0
        self.root = None
        self.image = None
        self.image_path = None
        self.started_at = None

    @property
    def labels(self):
        return [entry.get("plot_label", entry["path"].stem.replace("_", " ").title()) for entry in self.entries]

    @property
    def categories(self):
        return list(dict.fromkeys(entry.get("category", "Overview") for entry in self.entries))

    def _category_entries(self, category):
        return [entry for entry in self.entries if entry.get("category", "Overview") == category]

    def _load_image(self, path):
        image = tk.PhotoImage(file=str(path))
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = max(image.width() / canvas_width, image.height() / canvas_height)
        if scale > 1:
            factor = max(1, int(scale + 0.999))
            image = image.subsample(factor, factor)
        return image

    def _render(self):
        if not self.plot_paths:
            return
        path = self.plot_paths[self.index]
        self.image = self._load_image(path)
        self.image_path = path
        self.canvas.delete("all")
        self.canvas.create_image(
            max(1, self.canvas.winfo_width()) // 2,
            max(1, self.canvas.winfo_height()) // 2,
            image=self.image,
            anchor="center",
        )
        self.selector.current(self.index)
        self.category_selector.set(self.entries[self.index].get("category", "Overview"))
        category = self.entries[self.index].get("category", "Overview")
        category_indices = [i for i, entry in enumerate(self.entries) if entry.get("category", "Overview") == category]
        self.position.configure(text=f"{category}: {category_indices.index(self.index) + 1} / {len(category_indices)}  |  {self.index + 1} / {len(self.plot_paths)}")
        self.previous.configure(state="normal" if self.index > 0 else "disabled")
        self.next.configure(state="normal" if self.index < len(self.plot_paths) - 1 else "disabled")
        self.root.title(f"{self.title} - {self.labels[self.index]}")

    def _set_index(self, index):
        if not self.plot_paths:
            return
        self.index = max(0, min(int(index), len(self.plot_paths) - 1))
        self._render()

    def _set_category(self, category):
        indices = [i for i, entry in enumerate(self.entries) if entry.get("category", "Overview") == category]
        if indices:
            self._set_index(indices[0])

    def _move_category(self, direction):
        if not self.entries:
            return
        category = self.entries[self.index].get("category", "Overview")
        indices = [i for i, entry in enumerate(self.entries) if entry.get("category", "Overview") == category]
        if self.index in indices:
            self._set_index(indices[max(0, min(len(indices) - 1, indices.index(self.index) + direction))])

    def _key(self, event):
        if event.keysym == "Left":
            self._set_index(self.index - 1)
        elif event.keysym == "Right":
            self._set_index(self.index + 1)
        elif event.keysym == "Home":
            self._set_index(0)
        elif event.keysym == "End":
            self._set_index(len(self.plot_paths) - 1)

    def show(self, auto_close_ms=None):
        """Open exactly one Tk top-level window; raise on unavailable display."""
        if not self.plot_paths:
            return {"opened": False, "reason": "no_available_dashboards", "top_level_windows": 0}
        self.started_at = time.perf_counter()
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1280x860")
        self.root.minsize(720, 480)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(2, weight=1)
        self.previous = ttk.Button(toolbar, text="Previous", command=lambda: self._move_category(-1))
        self.previous.grid(row=0, column=0, padx=(0, 6))
        self.next = ttk.Button(toolbar, text="Next", command=lambda: self._move_category(1))
        self.next.grid(row=0, column=1, padx=(0, 12))
        self.selector = ttk.Combobox(toolbar, values=self.labels, state="readonly", width=32)
        self.category_selector = ttk.Combobox(toolbar, values=self.categories, state="readonly", width=20)
        self.category_selector.grid(row=0, column=2, sticky="w")
        self.category_selector.bind("<<ComboboxSelected>>", lambda event: self._set_category(self.category_selector.get()))
        self.selector.grid(row=0, column=3, sticky="w")
        self.selector.bind("<<ComboboxSelected>>", lambda event: self._set_index(self.selector.current()))
        self.position = ttk.Label(toolbar, text="")
        self.position.grid(row=0, column=4, padx=12)
        metadata_text = " | ".join(f"{key}: {value}" for key, value in self.metadata.items())
        ttk.Label(toolbar, text=metadata_text).grid(row=0, column=5, sticky="e")

        self.canvas = tk.Canvas(self.root, background="#202124", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda event: self._render())
        self.root.bind("<Key>", self._key)
        self._render()
        if auto_close_ms is not None:
            self.root.after(int(auto_close_ms), self.root.destroy)
        self.root.mainloop()
        return {
            "opened": True,
            "top_level_windows": 1,
            "dashboard_count": len(self.plot_paths),
            "browser_initialization_seconds": time.perf_counter() - self.started_at,
            "peak_cached_images": 1,
        }


def open_plot_browser(plot_paths, metadata=None, auto_close_ms=None):
    browser = AnalysisPlotBrowser(plot_paths, metadata=metadata)
    return browser.show(auto_close_ms=auto_close_ms)


def build_html_plot_browser(
    plot_paths,
    metadata=None,
    output_path=None,
    title="SOCIALSIM - Analysis V2 Dashboard",
):
    """Build a persistent local dashboard without starting a web server."""
    output_path = Path(output_path or "analysis_dashboard.html").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in plot_paths:
        if isinstance(item, dict):
            entry = dict(item)
            path = Path(entry.get("path") or entry.get("filename", "")).resolve()
        else:
            path = Path(item).resolve()
            entry = {
                "category": "Overview",
                "plot_label": path.stem.replace("_", " ").title(),
            }
        if not path.exists():
            continue
        entry["path"] = os.path.relpath(path, output_path.parent).replace("\\", "/")
        entry["category"] = entry.get("category", "Overview")
        entry["plot_label"] = entry.get(
            "plot_label", path.stem.replace("_", " ").title()
        )
        entries.append(entry)

    payload = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    metadata_items = "".join(
        f"<span><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</span>"
        for key, value in (metadata or {}).items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #18202a; background: #eef1f4; font-family: Segoe UI, Arial, sans-serif; }}
  header {{ min-height: 72px; padding: 14px 20px; color: #f7f9fb; background: #17212b; border-bottom: 4px solid #00a6a6; }}
  h1 {{ margin: 0 0 8px; font-size: 20px; letter-spacing: 0; }}
  .metadata {{ display: flex; flex-wrap: wrap; gap: 8px 20px; color: #dce4ea; font-size: 13px; }}
  .layout {{ display: grid; grid-template-columns: minmax(230px, 280px) 1fr; min-height: calc(100vh - 72px); }}
  aside {{ padding: 16px; background: #f8fafb; border-right: 1px solid #cbd3da; }}
  label {{ display: block; margin: 0 0 6px; font-size: 12px; font-weight: 700; color: #44515e; }}
  select {{ width: 100%; height: 38px; margin-bottom: 14px; padding: 6px 8px; border: 1px solid #98a6b3; border-radius: 4px; background: white; }}
  .nav {{ display: grid; grid-template-columns: 40px 1fr 40px; align-items: center; gap: 8px; }}
  button {{ height: 38px; border: 1px solid #687785; border-radius: 4px; background: #fff; color: #17212b; font-size: 18px; cursor: pointer; }}
  button:hover {{ border-color: #00a6a6; color: #007c7c; }}
  button:disabled {{ opacity: .35; cursor: default; }}
  #position {{ text-align: center; color: #52616e; font-size: 13px; }}
  main {{ min-width: 0; padding: 18px; }}
  .plot-shell {{ height: calc(100vh - 112px); min-height: 440px; display: flex; align-items: center; justify-content: center; background: white; border: 1px solid #cbd3da; border-radius: 6px; overflow: hidden; }}
  img {{ display: block; max-width: 100%; max-height: 100%; object-fit: contain; }}
  .empty {{ color: #7a3e00; border-left: 4px solid #e09300; padding: 12px; background: #fff8e8; }}
  @media (max-width: 760px) {{
    .layout {{ grid-template-columns: 1fr; }}
    aside {{ border-right: 0; border-bottom: 1px solid #cbd3da; }}
    .plot-shell {{ height: 65vh; min-height: 320px; }}
  }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><div class="metadata">{metadata_items}</div></header>
<div class="layout">
  <aside>
    <label for="category">Category</label><select id="category"></select>
    <label for="plot">Plot</label><select id="plot"></select>
    <div class="nav"><button id="previous" title="Previous plot">&#8592;</button><div id="position"></div><button id="next" title="Next plot">&#8594;</button></div>
  </aside>
  <main><div class="plot-shell" id="plotShell"><img id="image" alt="Selected SOCIALSIM analysis plot"></div></main>
</div>
<script>
const entries = {payload};
const category = document.getElementById('category');
const plot = document.getElementById('plot');
const image = document.getElementById('image');
const shell = document.getElementById('plotShell');
const position = document.getElementById('position');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
let visible = [];
let index = 0;
const categories = [...new Set(entries.map(item => item.category || 'Overview'))];
for (const name of categories) category.add(new Option(name, name));
function render() {{
  if (!visible.length) {{ shell.innerHTML = '<div class="empty">No available plots for this category.</div>'; return; }}
  const entry = visible[index];
  image.src = entry.path;
  image.alt = entry.plot_label;
  plot.value = String(index);
  position.textContent = `${{index + 1}} / ${{visible.length}}`;
  previous.disabled = index === 0;
  next.disabled = index === visible.length - 1;
  document.title = `${{entry.plot_label}} - {html.escape(title)}`;
}}
function setCategory(name) {{
  visible = entries.filter(item => (item.category || 'Overview') === name);
  plot.innerHTML = '';
  visible.forEach((item, i) => plot.add(new Option(item.plot_label, String(i))));
  index = 0;
  render();
}}
category.addEventListener('change', () => setCategory(category.value));
plot.addEventListener('change', () => {{ index = Number(plot.value); render(); }});
previous.addEventListener('click', () => {{ index = Math.max(0, index - 1); render(); }});
next.addEventListener('click', () => {{ index = Math.min(visible.length - 1, index + 1); render(); }});
document.addEventListener('keydown', event => {{
  if (event.key === 'ArrowLeft') previous.click();
  if (event.key === 'ArrowRight') next.click();
}});
if (categories.length) {{ category.value = categories[0]; setCategory(categories[0]); }}
else {{ shell.innerHTML = '<div class="empty">No compatible Analysis plots were found.</div>'; }}
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path, len(entries)


def open_html_plot_browser(plot_paths, metadata=None, output_path=None):
    """Open a persistent local HTML dashboard and return immediately."""
    started_at = time.perf_counter()
    path, count = build_html_plot_browser(
        plot_paths,
        metadata=metadata,
        output_path=output_path,
    )
    opened = bool(webbrowser.open(path.as_uri(), new=2))
    return {
        "opened": opened,
        "mode": "local-html-browser",
        "dashboard_count": count,
        "dashboard_path": str(path),
        "browser_initialization_seconds": time.perf_counter() - started_at,
        "server_started": False,
    }
