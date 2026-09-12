# Ridal web server architecture

> **Scope:** this document describes only the optional `server` cargo
> feature — the web server and browser GUI under `src/server/` (`ridal
> gui` / `ridal server start`). It is not an architecture overview of
> Ridal as a whole, which is primarily a GPR processing CLI/library; see
> [`README.md`](README.md) for that. A CLI-only build (`cargo build
> --no-default-features -F cli`) never links Axum, MiniJinja, or blake3,
> and none of this applies to it.

This is the narrative companion to the module docs in `src/server/` —
run `cargo doc --no-deps -F cli,server --open` for the in-code map, with
intra-doc links between every type mentioned below. This document is
where the *reasoning* lives; the module docs are where the *contract*
for each piece lives, kept next to the code it describes so it can't
drift as easily as prose.

## What this is

A read-only viewer for radargrams Ridal has already processed into its
NetCDF output format: an index page listing every discovered radargram
(grouped, with a map of each group's track), and a chunked pan/zoom
viewer per radargram, similar in spirit to a tile-based map viewer but
serving amplitude renders instead of map tiles. Two launch modes share
one implementation:

```console
ridal gui <FILE-OR-DIRECTORY>          # local, ephemeral port, opens a browser
ridal server start <FILE-OR-DIRECTORY> # persistent, 127.0.0.1:8000 by default
```

Both accept a single processed `.nc` file or a directory to scan
recursively.

## Request lifecycle

1. **Startup** (`src/server/launch.rs`) discovers every radargram under
   the given root *eagerly*, not lazily per request — the target catalog
   size is on the order of 100 files, and eager construction turns a
   broken file into a clear startup warning instead of a request-time
   surprise. This builds one shared `AppState` (`src/server/app.rs`).
2. **Discovery** (`src/server/catalog.rs`) walks the directory (or
   accepts a single file), recognising processed output via
   `io::inspect_ridal_netcdf` — a metadata-only recogniser that lives
   *outside* `src/server/` and outside the `server` feature entirely, so
   CLI-only code (`process`, `batch`) can use it too. Duplicate
   radargram IDs resolve deterministically (newest `processing_datetime`
   wins, ties broken by path order), always with a warning, never a
   silent drop.
3. **Routing** (`src/server/routes.rs`) handles every HTTP request. It
   composes the catalog and render-service layers rather than containing
   new logic itself — no NetCDF, catalog, or rendering code belongs
   here. Page HTML comes from `src/server/templates.rs` (MiniJinja,
   embedded via `include_str!`); CSS/JS/images come from
   `src/server/assets.rs` (embedded via `include_bytes!`). A deployed
   binary needs no frontend directory alongside it — everything is
   compiled in.
4. **Rendering** a chunk or overview acquires a permit from
   `AppState::render_permits` (bounded by `--n-workers`), then runs on a
   `spawn_blocking` thread against the matching radargram's
   `RenderService`. See [Render pipeline](#render-pipeline) below.
5. **Tracks** (`src/server/track.rs`) are extracted and simplified for
   both the index page's per-group map and the viewer's cursor readout.

The one rule every submodule below `app`/`routes`/`launch`/`templates`
follows: **no dependency on Axum, MiniJinja, or other HTTP/template
types.** The render pipeline in particular is plain Rust, unit-tested
against synthetic arrays with no HTTP server anywhere near the tests.

## Identity model

Four concepts that are easy to conflate and load-bearing to keep apart
(`src/identity.rs`, `src/server/catalog.rs`):

| Concept | Type | Stable across reprocessing? |
|---|---|---|
| Radargram identity | `RadargramId` | yes |
| Display label | `DisplayName` | n/a — cosmetic only |
| Group name / id | `GroupName` / `GroupId` | n/a — cosmetic only |
| Processed revision | `RevisionId` | **no** — changes every run |

`RadargramId` and `GroupId` are validated ASCII slugs (`[a-z0-9_-]`, ≤128
characters, no reserved names, never starting with `_` or `-`).
`RevisionId` is a blake3 fingerprint of `(radargram_id,
processing_datetime)`, deliberately excluding path, filesystem
timestamps, filesize, and display name — moving or renaming a processed
file does not invalidate its cache; reprocessing does.

`RadargramId` and the recognition logic live outside the `server`
feature specifically so the CLI's `process`/`batch` commands can use the
same identity types (`--radargram-id`, `--display-name`, `--group-name`,
`--group-id`) without linking anything server-only.

A radargram with no group gets its own "Ungrouped" section on the index
page — same map, same card grid, not a lesser presentation — via a
reserved id (`_none`, `NO_GROUP_ID` in `app.rs`) that can never collide
with a real slug, since real slugs can't start with `_`. The same
underscore-prefix convention is used for the metadata dialog's synthetic
entries (`__revision_id`, `__shape`, `__start_stop_datetime`). This is
an implicit convention rather than a typed one — if slug validation
rules ever change, grep for these sentinels first.

## Render pipeline

Fixed order, enforced by module structure rather than just convention:
**source amplitude → dataset view → resample → normalize → colormap →
encode.** Everything lives under `src/server/render/`.

- **Geometry** (`grid.rs`) is pure, with no I/O: `ViewerRaster` is the
  source array at 1:1 — **the viewer does not resample** — `ChunkGrid`
  divides it into addressable 256×256 chunks, and `OverviewSpec`
  describes a whole-radargram thumbnail (~512 px wide), which does
  downscale.

  The viewer raster used to be capped at 8192×4096, with anything
  larger scaled down to fit. That silently cost a 12187-trace radargram
  a third of its trace resolution *and* a third of its sample
  resolution, since one scale factor was applied to both axes. The cap
  was bounding client cost — every chunk is a decoded 256×256 bitmap in
  the browser — which is now handled by loading chunks only as the view
  reaches them (see Frontend). Removing it took the server from 320 to
  720 chunks on that file but cold render time only from 34.3 s to
  37.9 s: total work is dominated by reading the source array either
  way, and 1:1 chunks each read exactly one storage chunk instead of a
  window spanning several.
- **Reading** (`renderer.rs`, via `source.rs`) never materializes a
  full-resolution image. A chunk reads exactly its source window; an
  overview reads the source in bounded-size horizontal bands (a 64 MB
  budget by default), since an overview's *input* is the entire
  radargram regardless of how small the output is — banding this
  dropped peak memory on the largest file in the test corpus from
  268 MB to 159 MB with render time unchanged.
- **Resampling** (`resample.rs`) offers four methods, each required to
  degrade gracefully to the exact raw sample at a true 1:1 footprint —
  the same behaviour a naive box filter has, and a real bug (see below)
  when a method fails to have it:
  - `Mean` — area-weighted, NaN-aware, weight-renormalizing. The
    default. Nearest-neighbour is deliberately excluded; it preserves
    this data's high-frequency noise badly.
  - `Peak` — largest value in the footprint, unweighted. Exists because
    radar traces oscillate around zero, so averaging a downsampled
    footprint cancels signed amplitude toward zero.
  - `Lanczos` — windowed-sinc, applied as two separable 1-D passes (a
    2-D kernel would be tens of thousands of taps per output pixel at a
    typical overview downsample ratio). Still a *linear* filter, so it
    has the same cancellation problem as `Mean` on signed oscillating
    data.
  - `LanczosRectified` — Lanczos on `|amplitude|`. Removes the
    cancellation while keeping proper anti-aliasing, unlike `Peak`'s
    upward bias. **Rectification is gated per axis** on that axis
    actually downsampling (`step > 1.0`); an earlier version rectified
    unconditionally, which silently defeated any sign-dependent
    profile's stretch even at native resolution, since every value
    reaching the colormap was already non-negative regardless of the
    source's true sign. Any future nonlinear or order-sensitive
    resampling method must have this same graceful-degradation property
    or it will reproduce that bug.
- **Profiles** (`profile.rs`) are the one server-defined, non-free-form
  configuration surface — never client-defined, per the explicit
  warning against unbounded client-driven render work. Four built-ins:

  | Profile | Transform | Resampling | Notes |
  |---|---|---|---|
  | `default` | Linear | Mean | 1–99% quantile |
  | `positive` | Positive (asymmetric) | LanczosRectified | biases toward positive returns, clips negative toward black |
  | `abslog` | `log10\|A\|` | LanczosRectified | sign-agnostic by construction, so rectifying changes nothing about what it means |
  | `high-contrast` | Linear | Mean | 5–95% quantile |

  `positive`'s asymmetry is why `RenderProfile`'s amplitude transform is
  an enum rather than a boolean: it needs a different domain for
  *statistics* (percentile bounds from `|x|`) than for *display* (the
  signed value, so negatives can clip). `colormap.rs`'s
  `to_stats_domain`/`to_display_domain` keep that split explicit.
- **Statistics** (`stats.rs`) estimates percentile bounds **once per
  revision+profile**, never per chunk — per-chunk estimation would let
  adjacent chunks normalize differently and turn every chunk boundary
  into a visible seam. Sampling reads 128 runs of 16 contiguous traces
  each (a fixed seed derived from the revision ID, so limits are
  reproducible across restarts): whole, contiguous traces rather than
  scattered pixels, both because a trace carries the full vertical
  structure needed to represent the source wavelet's contribution in
  correct proportion, and because contiguous reads stay inside a
  handful of HDF5 storage chunks — measured at ~7.6× faster than
  evenly-strided single-trace reads.
- **`RenderService`** (`service.rs`) is the entry point everything above
  is reached through: resolve amplitude limits (cached separately from
  images) → check the in-memory cache → render on a miss → insert →
  return. Cache keys fold in `RevisionId` plus every profile field that
  affects pixels, so a reprocessed file or a changed profile can never
  return a stale image.

## Concurrency and resource bounds

- Rendering runs on `spawn_blocking` threads, not tokio's async workers,
  under a semaphore permit sized by `--n-workers`. The permit is
  acquired *before* spawning, so a client that disconnects while queued
  never starts a render at all — one already in flight cannot be
  cancelled, since `spawn_blocking` tasks aren't cancellable, but it
  completes into the cache rather than being wasted work.
- Permit acquisition times out (30 s) into a `503` with a `Retry-After`
  header rather than queueing without limit.
- Each radargram's `RenderService` sits behind a `Mutex` held for the
  whole read-and-render, so two chunks of the *same* radargram never
  render concurrently — deliberately not split further, since the
  `netcdf` crate serialises its C calls behind its own global lock
  regardless, so splitting would only buy concurrency *across*
  radargrams, not within one. That same `Mutex` plus the cache re-check
  on entry is also what makes concurrent requests for one chunk
  generate it only once — a queued request finds the result already
  cached rather than re-rendering, which is pinned by a test rather
  than implemented as separate single-flight machinery.
- **netcdf-c/HDF5 is not thread-safe.** A global lock serialises every
  netcdf call, which is a hard ceiling on read parallelism no amount of
  tokio tuning lifts. Tests use `#[serial_test::serial(netcdf)]` plus
  retry; a residual ~6% `Netcdf(-101)` flake is a known, documented
  symptom of this.

## What was measured, and what it settled

`scripts/bench_server.py` benchmarks the server over real HTTP against a
real catalog — cold vs. warm separately, since they differ by orders of
magnitude. On a release build, the largest radargram in the test
catalog (12187×3678):

- Chunks (the viewer's actual unit of work) cost **2–6 ms cold, 0.2 ms
  warm.** Pan and zoom are already far faster than perceptible.
- The one expensive request is a **cold overview at ~1.8 s** — this is
  index first-paint cost, not a viewer responsiveness problem.
- The warm/cold ratio is **~9000×**, which is why persisting renders
  across restarts (an on-disk cache) matters more than any further
  rendering optimisation.

Measured again when the viewer cap was removed, on the same file: a
cold overview costs 13.9 s, the full 720-chunk grid 37.7 s, and the
360 chunks an opening viewport actually requests 24.0 s (debug build,
ratios only). Per-chunk marginal cost is ~38 ms against a ~10 s fixed
cost for opening and estimating amplitude limits, so viewport culling
saves 36% here — and proportionally far more the longer the radargram,
which is what makes an uncapped raster affordable.

**This settles that multiresolution server-side `(z, x, y)` tiling is
not currently justified and should not be started speculatively.** The
viewer's horizontal-scale control is a pure client-side transform for
exactly this reason — it re-lays existing overlays rather than asking
the server for a different render, which is cheap and, per the above,
sufficient. If real usage ever shows otherwise, tiling would replace
that control rather than extend it, and should be justified by new
measurements, not assumed from first principles.

## Frontend

No build step: Leaflet is vendored (not loaded from a CDN, so the
viewer works offline — a realistic deployment for field use) via
`scripts/vendor_leaflet.sh`, and all first-party CSS/JS is embedded the
same way as the templates. **First-party assets must not live under
`assets/vendor/`** — the vendor script does `rm -rf` on that directory,
so anything placed there is silently deleted on the next Leaflet
refresh.

Page-specific JavaScript lives in `assets/index.js` / `assets/viewer.js`
(shared helpers in `assets/app.js`), not inline in the templates. Fetch
calls go through `RIDAL.fetchJson`, which surfaces the server's
structured error envelope (`{"error": {"code", "message"}}`) instead of
a bare `.then(r => r.json())` silently proceeding with a malformed
object on failure.

## Known constraints and gaps

Durable, load-bearing decisions rather than oversights:

- **No authentication of any kind.** `server start` binds loopback by
  default; binding elsewhere is an explicit flag, but deploying beyond
  localhost requires a reverse proxy that provides auth.
- **Amplitude limits are global per revision+profile**, which is what
  keeps chunks seamless — a radargram with strongly varying gain
  down-profile cannot be locally renormalised without breaking that
  guarantee. Per-region normalisation and seamlessness are mutually
  exclusive under the current design.
- **Edge chunks are returned at their true size, not padded** to
  256×256 — padding was tried and left a visible border of dead pixels.
  A client consuming the chunk API must compute the valid extent itself;
  it isn't currently advertised in a response header or manifest
  endpoint.
- **`RenderServiceConfig::source_cache_mb` is not wired to anything.**
  It's reserved for a deferred HDF5-chunk-aligned source-read cache and
  is not even exposed as a CLI flag today — always its default value.

Not yet implemented, in rough priority order given the measurements
above:

1. **On-disk render cache**, keyed on `RevisionId` so reprocessing
   invalidates it correctly. Now the highest-value remaining item — see
   [What was measured](#what-was-measured-and-what-it-settled).
2. **Digitization** — picking, editing, and persisting interpreted
   reflector layers in the viewer, with user-editable layer names and
   colours (a fixed server-side vocabulary will not survive contact
   with real interpretation work). This is the actual point of building
   a viewer at all; everything above is scaffolding for it. Needs a
   write path and a real persistence/authorship model, which nothing
   built so far requires.
3. **User-defined render profiles**, named/validated/persisted
   server-side rather than accepted as free-form per-request
   parameters — the latter is explicitly rejected, since
   `?min=…&max=…&contrast=…` on every chunk request would make each one
   a distinct, uncacheable render variant and let a client trivially
   thrash the cache.
4. A topographically corrected `DatasetView` — `DatasetView` is already
   an enum with one variant specifically so this can be added without
   reshaping the API. A significant bonus, not a blocker for
   digitization.
