# Changelog

## Step 16 Performance Architecture

### Added

- `experiment_workflow.py` with serializable experiment specifications,
  deterministic branch generation, validation, manifests, summaries, and safe
  resume/skip.
- `experiment_runner.py --spec` as the standard independent-branch workflow
  entry point.
- Long-term engineering record in `docs/DEVELOPMENT_LOG.md`.

### Performance

- Retained authoritative ID indexes.
- Retained the same-week Firm labor-capacity cache.
- Added independent Windows `spawn` branch execution for separate Worlds.
- Added `FULL_DIAGNOSTIC`, `RESEARCH_FAST`, and `DEBUG_DETAILED` observability
  modes.

### Deferred / Known Limitations

- Heap-based labor allocation was rolled back after whole-model regression.
- Numeric side-car design, Numba/Cython, C++/Rust, GPU work, and
  single-World parallelization remain deferred until profiling and data layout
  justify them.
