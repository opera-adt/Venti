# ADR-0022: geepers comes from opera-adt/geepers main, never a fork

- **Status:** accepted
- **Date:** 2026-10-20
- **Source:** Talib Oliver (DISP-CAL lead), upstreaming venti-dev into opera-adt/Venti

## Decision

geepers stays a core dependency of Venti, as on venti-dev, but from
`opera-adt/geepers` `main` (unpinned), never from a personal fork. Venti uses
geepers' `euler` (Euler-pole maths) and `gps_imaging` (`msf_interpolate`) and,
for UNR snapshots, `gps_sources.unr_grid`. What the fork added and upstream
does not have lives in Venti: the ITRF2020/2014-PMM tables and the UNR plate
codes (`venti.models.plate_motion`: `plate_pole`, `plate_velocity_enu`,
`load_plate_motion_model`, `PLATE_CODES`) and
`venti.gnss.sampling.reinterpolate_nodes`. This amends ADR-0010: geepers
remains the source for UNR access, Euler and GPS Imaging, from opera-adt only.

## Context

venti-dev pinned `mgovorcin/geepers@1396741`, a fork commit not in
`opera-adt/geepers`. Official code cannot depend on a personal fork. Upstream
`main` (`09e6180` on 2026-10-20) has the same `euler` and `gps_imaging` code
as the fork, so the move is numerically neutral: on synthetic inputs the
Venti functions match the fork's bit for bit, and cal-disp's gamma and v0.5.1
goldens pass at 1e-6 with this Venti.

## Consequences

- geepers' core dependencies (dask, zarr, pandera, lxml, pyogrio, ...) are in
  every Venti install, including cal-disp's image. This relaxes the "no
  dask/zarr in the core" rule of ADR-0020 (they arrive transitively), and
  cal-disp v0.5's dependency budget (`budget.yaml`, which forbids dask and
  zarr) needs an exception or a change.
- Tracking `main` is not reproducible: two installs on different days can get
  different geepers. pixi environments are reproducible through `pixi.lock`;
  pip installs and Docker builds need a lock or pin at release time.
- Measured 2026-10-20 in pip-built environments: with zarr installed,
  `test_parallel_workers_give_identical_surfaces` (threaded netCDF reads)
  crashes intermittently; it also crashed when run alone without zarr. The
  parallel workflow's file I/O is not thread-safe with pip-wheel HDF5 builds.
  The CI `test` job therefore runs in the locked pixi `test` environment
  (conda-forge builds: 511 passed), not pip; `core-only` stays pip and runs a
  subset without the parallel workflow. A fix in Venti (serialize HDF5 I/O in
  the worker threads) is still open.
- `docs/specs.md` D10 is amended to point here.
