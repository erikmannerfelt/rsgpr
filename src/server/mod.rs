//! Web server and browser GUI for browsing radargrams ridal has already
//! processed (#115). Everything under this module requires the `server`
//! cargo feature; a CLI-only build never links Axum or MiniJinja.
//!
//! See `ARCHITECTURE.md` at the repository root for the narrative version
//! of what follows, including the request lifecycle, the decisions most
//! likely to matter to whoever works here next, and the roadmap toward a
//! fully robust deployment. This doc is the in-code map; that one is the
//! tour with reasoning attached.
//!
//! # How a request is served
//!
//! 1. [`launch`] starts the Axum server for either launch mode (`ridal
//!    gui` opens a browser on an ephemeral port; `ridal server start`
//!    binds a fixed port and stays up) and builds one shared
//!    [`app::AppState`], discovering every radargram under the given root
//!    eagerly at startup rather than lazily per request.
//! 2. [`catalog`] does that discovery: it walks a directory (or accepts a
//!    single file), using [`crate::io::inspect_ridal_netcdf`] to recognise
//!    processed output. That function -- and the validated
//!    [`crate::identity::RadargramId`]/[`crate::identity::GroupId`] types
//!    it depends on -- live outside `server` and outside this feature
//!    entirely, so CLI-only builds (`process`, `batch`) can use the same
//!    identity types without linking anything server-only.
//! 3. [`routes`] handles every HTTP request. It composes the modules
//!    below rather than containing new logic itself -- no NetCDF,
//!    catalog, or rendering code belongs in `routes.rs`. [`templates`]
//!    renders the index and viewer pages (MiniJinja, embedded via
//!    `include_str!`); [`assets`] serves CSS, JS, and vendored Leaflet the
//!    same way (`include_bytes!`), so a deployed binary needs no separate
//!    frontend directory alongside it.
//! 4. A render request acquires a permit from
//!    [`app::AppState::render_permits`] (bounded by `--n-workers`, so a
//!    card grid requesting one overview per catalog entry cannot start
//!    hundreds of simultaneous renders), then runs on a `spawn_blocking`
//!    thread against the matching [`app::OpenRadargram`]'s
//!    [`render_service::RenderService`], which caches encoded results and
//!    drives [`crate::render`] -- the pipeline that turns a source window
//!    into encoded image bytes. That pipeline lives outside this module
//!    and outside the `server` feature, because nothing in it is about
//!    HTTP: `ridal render` on the command line and the web GUI are two
//!    callers of one renderer.
//! 5. [`track`] extracts and simplifies the physical track a radargram
//!    follows (trace-indexed, not distance-indexed -- see its module doc
//!    for why that distinction is load-bearing), used by both the index
//!    page's map and the viewer's cursor readout.
//!
//! # The one rule every submodule follows
//!
//! Rendering and dataset logic must not depend on Axum, MiniJinja, or
//! other HTTP/template types. Only [`app`], [`routes`], [`interp_routes`],
//! [`launch`], and [`templates`] are allowed to know an HTTP server exists;
//! everything else must stay testable with nothing but plain Rust and,
//! where needed, a real NetCDF file. [`crate::render`] and [`crate::source`]
//! followed that rule so completely that they no longer live here at all.

pub mod app;
pub mod assets;
pub mod auth;
pub mod auth_routes;
#[cfg(test)]
mod auth_routes_tests;
pub mod catalog;
pub mod interp_routes;
#[cfg(test)]
mod interp_routes_tests;
pub mod launch;
pub mod render_service;
pub mod routes;
pub mod templates;
pub mod track;
