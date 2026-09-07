//! Shared Axum application and typed state (#120).
//!
//! Route handlers here should primarily compose the already-tested
//! catalog (M3) and render-service (M4/M5) components -- no new NetCDF,
//! catalog, or rendering logic belongs in this module.

use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, Mutex};

use axum::routing::get;
use axum::Router;
use tokio::sync::Semaphore;

use super::catalog::{Catalog, RevisionId};
use super::render::service::{RenderService, RenderServiceConfig};
use super::source::SourceReader;
use crate::identity::RadargramId;

/// One open radargram: its render service plus the metadata needed to
/// answer dataset-detail and viewer-page requests without re-inspecting
/// the file.
pub struct OpenRadargram {
    pub service: Mutex<RenderService>,
    pub shape: (usize, usize),
}

pub struct AppState {
    pub root: PathBuf,
    /// Whether `root` itself names a single NetCDF file rather than a
    /// catalog directory, decided once here from the freshly canonicalized
    /// `root` so `resolve_absolute_path` never has to stat `root` itself --
    /// it only ever touches the filesystem via the join-then-canonicalize
    /// sequence applied to `candidate`, which is the pattern CodeQL's
    /// path-injection analysis already recognises as validated. A raw
    /// `root.is_file()` inside `resolve_absolute_path` kept getting
    /// re-flagged even after `root` was canonicalized, because that
    /// canonicalization happened in a different function than the sink --
    /// CodeQL doesn't credit a barrier it can't see next to the check it
    /// guards.
    root_is_file: bool,
    pub catalog: Catalog,
    pub radargrams: HashMap<String, OpenRadargram>,
    /// The project this catalog belongs to, when it belongs to one.
    ///
    /// `None` is the read-only case Ridal has always supported: a bare
    /// directory of `.nc` files, or a single file, with nowhere to save
    /// anything. Every write route checks this rather than assuming.
    pub project: Option<crate::project::Project>,
    /// Whether write routes are enabled. Requires a project, and can be
    /// switched off for one that has one (`--read-only`).
    pub writable: bool,
    /// Bounds how many renders may be in flight at once, across every
    /// radargram, sized from `--n-workers`.
    ///
    /// Rendering is CPU-bound and runs on `spawn_blocking` threads, whose
    /// pool tokio sizes at 512 by default -- far more than is useful for
    /// work that is already competing for cores, and enough that a card
    /// grid requesting one overview per catalog entry could start
    /// hundreds of simultaneous renders, each holding its own source
    /// band in memory. The permit is acquired *before* spawning so a
    /// client that disconnects while queued never starts one at all.
    pub render_permits: Arc<Semaphore>,
}

impl AppState {
    /// Discover the catalog under `root` and eagerly open a
    /// [`RenderService`] for every entry. Eager rather than lazy: the
    /// issue's target catalog size is ~100 files (#122/#123), and eager
    /// construction means a broken file surfaces as a clear startup
    /// warning rather than a request-time surprise.
    ///
    /// `root` is canonicalized once here, the single point every later
    /// filesystem operation on it (`Catalog::discover`, every
    /// `resolve_absolute_path` call, and the stored `self.root` used by
    /// `absolute_path()`) flows from. This is the CLI's own
    /// `ridal gui <PATH>` / `ridal server start <PATH>` argument, fixed
    /// once at process startup and never influenced by an HTTP request --
    /// `AppState::build` has exactly one production caller (`launch.rs`).
    /// Canonicalizing it anyway, rather than trusting that, is what CodeQL's
    /// path-injection analysis recognises as validating a path before use,
    /// and it's a genuine improvement on its own merits regardless: a
    /// nonexistent root now fails clearly at startup instead of silently
    /// producing an empty catalog, and every later comparison against it
    /// (this function's own containment check below) is symlink-resolved
    /// and consistent.
    /// `project` is `None` for a bare directory or single file, which is
    /// the read-only case; `writable` is ignored unless a project is present.
    pub fn build_with_project(
        root: &StdPath,
        config: &RenderServiceConfig,
        project: Option<crate::project::Project>,
        writable: bool,
    ) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("Invalid catalog root {}: {e}", root.display()))?;
        let root_is_file = root.is_file();
        let root = root.as_path();
        let catalog = Catalog::discover(root);
        let mut radargrams = HashMap::new();

        for entry in &catalog.entries {
            let absolute_path = match Self::resolve_absolute_path(root, root_is_file, entry) {
                Ok(path) => path,
                Err(e) => {
                    eprintln!(
                        "Warning: refusing unsafe catalog path {}: {e}",
                        entry.relative_path
                    );
                    continue;
                }
            };
            let reader = match SourceReader::open(&absolute_path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "Warning: could not open {} for rendering: {e}",
                        entry.relative_path
                    );
                    continue;
                }
            };
            let shape = reader.shape();
            let revision_id: RevisionId = entry.revision_id.clone();
            let service = RenderService::new(reader, revision_id, config);
            radargrams.insert(
                entry.radargram_id.as_str().to_string(),
                OpenRadargram {
                    service: Mutex::new(service),
                    shape,
                },
            );
        }

        Ok(Self {
            root: root.to_path_buf(),
            root_is_file,
            catalog,
            radargrams,
            // Writes need somewhere to go, so a catalog with no project is
            // read-only no matter what the caller asked for.
            writable: writable && project.is_some(),
            project,
            // `.max(1)`: a zero-permit semaphore would deadlock every
            // render forever. The CLI rejects `--n-workers 0` with a
            // clear message, so this only guards programmatic callers.
            render_permits: Arc::new(Semaphore::new(config.n_workers.max(1))),
        })
    }

    /// Resolves a catalog entry beneath the canonical catalog root.
    ///
    /// Rejects absolute paths, parent-directory components, and paths whose
    /// canonical form escapes the catalog root, including through symlinks.
    /// `root_is_file` is `root.is_file()`, decided once in [`Self::build`]
    /// against the freshly canonicalized root rather than re-stated here --
    /// see the field doc on [`AppState::root_is_file`].
    fn resolve_absolute_path(
        root: &StdPath,
        root_is_file: bool,
        entry: &super::catalog::CatalogEntry,
    ) -> Result<PathBuf, String> {
        if root_is_file {
            return Ok(root.to_path_buf());
        }

        let mut candidate = root.to_path_buf();
        for component in StdPath::new(&entry.relative_path).components() {
            match component {
                std::path::Component::Normal(part) => candidate.push(part),
                _ => {
                    return Err(format!(
                        "path contains a disallowed component: {}",
                        entry.relative_path
                    ));
                }
            }
        }

        let candidate = candidate
            .canonicalize()
            .map_err(|e| format!("could not resolve {}: {e}", candidate.display()))?;

        if !candidate.starts_with(root) {
            return Err(format!(
                "resolved path escapes catalog root: {}",
                candidate.display()
            ));
        }

        Ok(candidate)
    }

    /// The absolute filesystem path for a catalog entry. Never exposed to
    /// HTTP clients directly (#122: "keep filesystem paths internal") --
    /// only used server-side, e.g. to re-open a file for track reading.
    pub fn absolute_path(&self, entry: &super::catalog::CatalogEntry) -> Result<PathBuf, String> {
        Self::resolve_absolute_path(&self.root, self.root_is_file, entry)
    }

    pub fn find_entry(&self, radargram_id: &str) -> Option<&super::catalog::CatalogEntry> {
        self.catalog
            .entries
            .iter()
            .find(|e| e.radargram_id.as_str() == radargram_id)
    }

    /// All entries sharing `group`, ordered like the catalog itself.
    /// `group == NO_GROUP_ID` matches entries with no group at all,
    /// rather than a literal group id -- see [`NO_GROUP_ID`].
    pub fn entries_in_group(&self, group: &str) -> Vec<&super::catalog::CatalogEntry> {
        self.catalog
            .entries
            .iter()
            .filter(|e| {
                if group == NO_GROUP_ID {
                    e.group_id.is_none()
                } else {
                    e.group_id.as_ref().is_some_and(|g| g.as_str() == group)
                }
            })
            .collect()
    }
}

/// Reserved id for the "Ungrouped" pseudo-group on the index page and its
/// `/api/v1/groups/{id}/tracks` map. Safe by construction: `GroupId`
/// validation (`identity.rs::validate_slug`) rejects any id starting with
/// `_`, so no explicit `--group-id` or directory-derived slug can ever
/// collide with it -- the same guarantee `routes.rs`'s synthetic
/// `__revision_id`/`__shape` metadata keys rely on.
///
/// This is a presentation concept only: `CatalogEntry::group_id` itself
/// stays `None` for a genuinely ungrouped radargram. Widening that to a
/// real `GroupId` would leak into the viewer, which correctly treats
/// `None` as "no group" (no banner suffix, no sibling-track fetch) --
/// every ungrouped radargram in the catalog is not one group.
pub const NO_GROUP_ID: &str = "_none";

/// Build the complete Axum application over `state`.
pub fn build_router(state: std::sync::Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(super::routes::index_page))
        .route("/view/{radargram_id}", get(super::routes::viewer_page))
        .route("/static/leaflet.js", get(super::assets::leaflet_js))
        .route("/static/leaflet.css", get(super::assets::leaflet_css))
        .route("/static/app.css", get(super::assets::app_css))
        .route("/static/app.js", get(super::assets::app_js))
        .route("/static/index.js", get(super::assets::index_js))
        .route("/static/viewer.js", get(super::assets::viewer_js))
        .route("/static/picker.js", get(super::assets::picker_js))
        .route(
            "/static/images/marker-icon.png",
            get(super::assets::marker_icon),
        )
        .route(
            "/static/images/marker-icon-2x.png",
            get(super::assets::marker_icon_2x),
        )
        .route(
            "/static/images/marker-shadow.png",
            get(super::assets::marker_shadow),
        )
        .route("/static/images/layers.png", get(super::assets::layers_png))
        .route(
            "/static/images/layers-2x.png",
            get(super::assets::layers_2x_png),
        )
        .route("/static/images/logo.svg", get(super::assets::logo_svg))
        .route("/favicon.ico", get(super::assets::favicon))
        .route("/api/v1/health", get(super::routes::health))
        .route("/layers", get(super::routes::layers_page))
        .route("/settings", get(super::routes::settings_page))
        .route("/static/settings.js", get(super::assets::settings_js))
        .route(
            "/api/v1/project/settings",
            get(super::interp_routes::get_settings).put(super::interp_routes::put_settings),
        )
        .route("/static/layers.js", get(super::assets::layers_js))
        .route(
            "/api/v1/layers",
            get(super::interp_routes::get_layers).put(super::interp_routes::put_layers),
        )
        .route(
            "/api/v1/layers/usage",
            get(super::interp_routes::layer_usage),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/interpretations",
            get(super::interp_routes::list_interpretations),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/interpretations/{user}/level2",
            get(super::interp_routes::interpretation_level2),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/interpretations/{user}",
            get(super::interp_routes::get_interpretation)
                .put(super::interp_routes::put_interpretation)
                .delete(super::interp_routes::delete_interpretation),
        )
        .route("/api/v1/profiles", get(super::routes::list_profiles))
        .route("/api/v1/datasets", get(super::routes::list_datasets))
        .route(
            "/api/v1/datasets/{radargram_id}",
            get(super::routes::dataset_detail),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/track",
            get(super::routes::dataset_track),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/attributes",
            get(super::routes::dataset_attributes),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/axes",
            get(super::routes::dataset_axes),
        )
        .route(
            "/api/v1/groups/{group}/tracks",
            get(super::routes::group_tracks),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/views/{view}/overview",
            get(super::routes::overview_image),
        )
        .route(
            "/api/v1/datasets/{radargram_id}/views/{view}/chunks/{profile}/{x}/{y}",
            get(super::routes::chunk_image),
        )
        .with_state(state)
}

/// A radargram ID from the URL is validated the same way an explicit
/// `--radargram-id` would be, so a malformed ID (never a real dataset)
/// fails fast with a clear reason rather than a generic "not found".
pub fn validate_radargram_id(raw: &str) -> Result<RadargramId, String> {
    RadargramId::new(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tower::ServiceExt;

    fn write_test_nc(path: &StdPath, radargram_id: &str) {
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("y", 20).unwrap();
        file.add_dimension("x", 300).unwrap();
        let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
        let data: Vec<f32> = (0..(20 * 300)).map(|i| (i % 100) as f32).collect();
        var.put_values(&data, ..).unwrap();
        file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
            .unwrap();
        file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
            .unwrap();
        file.add_attribute("ridal_radargram_id", radargram_id)
            .unwrap();
    }

    /// Like `write_test_nc`, but with real track variables (crs, easting,
    /// northing, time) so `dataset_track`/`group_tracks` have something to
    /// read, and an optional group.
    fn write_test_nc_with_track(path: &StdPath, radargram_id: &str, group: Option<&str>) {
        let n_traces = 50;
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("y", 5).unwrap();
        file.add_dimension("x", n_traces).unwrap();
        let mut data_var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
        data_var
            .put_values(&vec![1.0f32; 5 * n_traces], ..)
            .unwrap();

        let mut easting_var = file.add_variable::<f64>("easting", &["x"]).unwrap();
        let easting: Vec<f64> = (0..n_traces).map(|i| 500000.0 + i as f64).collect();
        easting_var.put_values(&easting, ..).unwrap();

        let mut northing_var = file.add_variable::<f64>("northing", &["x"]).unwrap();
        northing_var
            .put_values(&vec![8_000_000.0f64; n_traces], ..)
            .unwrap();

        let mut time_var = file.add_variable::<f64>("time", &["x"]).unwrap();
        let time: Vec<f64> = (0..n_traces).map(|i| i as f64 * 0.1).collect();
        time_var.put_values(&time, ..).unwrap();

        file.add_attribute("crs", "EPSG:32633").unwrap();
        file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
            .unwrap();
        file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
            .unwrap();
        file.add_attribute("ridal_radargram_id", radargram_id)
            .unwrap();
        if let Some(group) = group {
            file.add_attribute("ridal_group_name", group).unwrap();
            file.add_attribute("ridal_group_id", group).unwrap();
        }
    }

    fn test_app(dir: &StdPath) -> Router {
        let config = RenderServiceConfig::default();
        let state =
            std::sync::Arc::new(AppState::build_with_project(dir, &config, None, false).unwrap());
        build_router(state)
    }

    /// `test_app`, but keeping the state so a test can inspect the render
    /// cache afterwards, and with a caller-chosen `n_workers`.
    fn test_app_with_state(dir: &StdPath, n_workers: usize) -> (Router, Arc<AppState>) {
        let config = RenderServiceConfig {
            n_workers,
            ..RenderServiceConfig::default()
        };
        let state =
            std::sync::Arc::new(AppState::build_with_project(dir, &config, None, false).unwrap());
        (build_router(state.clone()), state)
    }

    async fn get(app: &Router, uri: &str) -> (StatusCode, axum::body::Bytes) {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, bytes)
    }

    #[tokio::test]
    async fn health_route_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());
        let (status, body) = get(&app, "/api/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn full_route_suite_against_a_real_catalog() {
        // A single #[test] (not #[tokio::test]) driving a manually built
        // runtime, so every route in this suite shares one netcdf-serial
        // guard -- otherwise each #[tokio::test] would need its own,
        // fighting the file-per-test isolation this suite wants to test.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "route-test-a");
            let app = test_app(dir.path());

            // Catalog listing.
            let (status, body) = get(&app, "/api/v1/datasets").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["entries"].as_array().unwrap().len(), 1);
            assert_eq!(json["entries"][0]["radargram_id"], "route-test-a");

            // Dataset detail: known and unknown.
            let (status, _) = get(&app, "/api/v1/datasets/route-test-a").await;
            assert_eq!(status, StatusCode::OK);
            let (status, body) = get(&app, "/api/v1/datasets/does-not-exist").await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["code"], "dataset_not_found");

            // An invalid radargram ID (uppercase) is a 400, not a 404 --
            // structurally invalid vs. legitimately absent (#118's
            // distinction, applied here to dataset lookup too).
            let (status, body) = get(&app, "/api/v1/datasets/Not-Valid").await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["code"], "invalid_radargram_id");

            // Profiles.
            let (status, body) = get(&app, "/api/v1/profiles").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert!(json
                .as_array()
                .unwrap()
                .contains(&Value::String("default".into())));

            // Overview image.
            let (status, body) = get(
                &app,
                "/api/v1/datasets/route-test-a/views/standard/overview",
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert!(image::load_from_memory(&body).is_ok());

            // Unknown dataset view.
            let (status, body) =
                get(&app, "/api/v1/datasets/route-test-a/views/bogus/overview").await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["code"], "unknown_dataset_view");

            // Unknown profile.
            let (status, _) = get(
                &app,
                "/api/v1/datasets/route-test-a/views/standard/overview?profile=nonexistent",
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);

            // A valid chunk.
            let (status, body) = get(
                &app,
                "/api/v1/datasets/route-test-a/views/standard/chunks/default/0/0",
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            let decoded = image::load_from_memory(&body).unwrap();
            assert_eq!(
                decoded.width(),
                super::super::render::grid::CHUNK_SIZE as u32
            );

            // Structurally invalid chunk coordinate (not a number) -> 400.
            let (status, body) = get(
                &app,
                "/api/v1/datasets/route-test-a/views/standard/chunks/default/abc/0",
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["code"], "invalid_chunk_coordinate");

            // Well-formed but out-of-grid chunk coordinate -> 404, not 400.
            let (status, body) = get(
                &app,
                "/api/v1/datasets/route-test-a/views/standard/chunks/default/999/999",
            )
            .await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["code"], "image_chunk_not_found");

            // Pages: index and viewer.
            let (status, body) = get(&app, "/").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();
            assert!(html.contains("route-test-a"));

            let (status, body) = get(&app, "/view/route-test-a").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();
            assert!(html.contains("route-test-a"));

            let (status, _) = get(&app, "/view/does-not-exist").await;
            assert_eq!(status, StatusCode::NOT_FOUND);

            // Static assets are embedded, not proxied to a filesystem path.
            for asset in [
                "/static/leaflet.js",
                "/static/leaflet.css",
                "/static/app.css",
                "/static/app.js",
                "/static/index.js",
                "/static/viewer.js",
                "/static/images/logo.svg",
                "/favicon.ico",
            ] {
                let (status, body) = get(&app, asset).await;
                assert_eq!(status, StatusCode::OK, "{asset}");
                assert!(!body.is_empty(), "{asset}");
            }
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn index_page_renders_lazy_overview_thumbnails() {
        // #121 requires an ~512px overview per catalog entry, and names
        // loading="lazy" as the mechanism bounding initial render work.
        // Both were missing when M7 was first reported complete, so they
        // are pinned here rather than left to visual inspection.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "thumb-test-a");
            let app = test_app(dir.path());

            let (status, body) = get(&app, "/").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();

            assert!(
                html.contains("/api/v1/datasets/thumb-test-a/views/standard/overview"),
                "index must embed the overview image URL"
            );
            assert!(
                html.contains("loading=\"lazy\""),
                "overview images must be lazily loaded"
            );
            // The viewer's back-link targets /#card-{id}; the anchor has to
            // survive the table -> card-grid restructuring.
            assert!(
                html.contains("id=\"card-thumb-test-a\""),
                "per-entry anchor must be preserved"
            );
            assert!(
                html.contains("/static/app.css"),
                "first-party stylesheet must be linked"
            );
            assert!(
                html.contains("/static/images/logo.svg"),
                "logo must be shown beside the wordmark in the shared header"
            );
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn index_page_render_profile_switcher_propagates_to_links() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "profile-test-a");
            let app = test_app(dir.path());

            let (status, body) = get(&app, "/?profile=positive").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();
            // The chosen profile propagates to both the thumbnail source
            // and the card's own link, so opening a radargram keeps the
            // profile the index was browsing in.
            assert!(html.contains(
                "/api/v1/datasets/profile-test-a/views/standard/overview?profile=positive"
            ));
            assert!(html.contains("/view/profile-test-a?profile=positive"));
            assert!(
                html.contains("value=\"positive\" selected"),
                "the switcher must reflect the active profile"
            );
            // Path/Processed are tucked behind "More info", not shown
            // directly in the scannable part of the card.
            assert!(html.contains("More info"));

            let (status, _) = get(&app, "/?profile=nonexistent").await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn track_attributes_and_group_routes() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc_with_track(&dir.path().join("a.nc"), "track-a", Some("shared-group"));
            write_test_nc_with_track(&dir.path().join("b.nc"), "track-b", Some("shared-group"));
            write_test_nc_with_track(&dir.path().join("c.nc"), "track-c", None);
            let app = test_app(dir.path());

            // This radargram's own track.
            let (status, body) = get(&app, "/api/v1/datasets/track-a/track").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            let segments = json["segments"].as_array().unwrap();
            assert!(!segments.is_empty());
            let first_vertex = &segments[0]["vertices"][0];
            assert!(first_vertex["lon"].as_f64().is_some());
            assert!(first_vertex["lat"].as_f64().is_some());

            // Full raw attribute set, for the metadata dialog, plus the
            // curated display entries built from it.
            let (status, body) = get(&app, "/api/v1/datasets/track-a/attributes").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["raw"]["ridal_radargram_id"], "track-a");
            assert_eq!(json["raw"]["crs"], "EPSG:32633");
            let entries = json["entries"].as_array().unwrap();
            assert!(
                entries
                    .iter()
                    .any(|e| e["label"] == "CRS" && e["value"] == "EPSG:32633"),
                "{entries:?}"
            );
            // The revision checksum is server-computed, never a file
            // attribute, so it must still show up as a curated entry.
            assert!(
                entries.iter().any(|e| e["label"] == "Revision"),
                "{entries:?}"
            );
            // The fixture writes no original_filepaths attribute; the
            // dedicated field must still be present (as an empty array),
            // not missing or an error.
            assert_eq!(json["original_filepaths"].as_array().unwrap().len(), 0);

            // The `/axes` route degrades each axis to null independently
            // when the fixture never wrote distance/twtt/depth.
            let (status, body) = get(&app, "/api/v1/datasets/track-a/axes").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert!(json["distance"].is_null());
            assert!(json["twtt"].is_null());
            assert!(json["depth"].is_null());

            // Group tracks: both group members present, the ungrouped one absent.
            let (status, body) = get(&app, "/api/v1/groups/shared-group/tracks").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            let obj = json.as_object().unwrap();
            assert!(obj.contains_key("track-a"));
            assert!(obj.contains_key("track-b"));
            assert!(!obj.contains_key("track-c"));
            assert!(obj["track-a"]["track"]["segments"].is_array());

            // An empty/unknown group is an empty object, not an error.
            let (status, body) = get(&app, "/api/v1/groups/no-such-group/tracks").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json.as_object().unwrap().len(), 0);

            // The reserved "no group" sentinel matches ungrouped entries
            // specifically -- not a literal group id -- so track-c (the
            // only ungrouped member of this catalog) is the one that
            // shows up here.
            let (status, body) = get(&app, &format!("/api/v1/groups/{NO_GROUP_ID}/tracks")).await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            let obj = json.as_object().unwrap();
            assert!(obj.contains_key("track-c"));
            assert!(!obj.contains_key("track-a"));
            assert!(!obj.contains_key("track-b"));
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn index_page_gives_ungrouped_entries_a_map_like_any_group() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc_with_track(&dir.path().join("a.nc"), "grouped-a", Some("Some Group"));
            write_test_nc_with_track(&dir.path().join("b.nc"), "ungrouped-b", None);
            let app = test_app(dir.path());

            let (status, body) = get(&app, "/").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();

            assert!(html.contains(">Ungrouped<"), "{html}");
            // Same map treatment as a named group: a group-map div keyed
            // by the reserved sentinel id, which entries_in_group and
            // group_tracks both already resolve to "no group".
            assert!(
                html.contains(&format!("data-group=\"{NO_GROUP_ID}\"")),
                "{html}"
            );

            let (status, body) = get(&app, &format!("/api/v1/groups/{NO_GROUP_ID}/tracks")).await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert!(json.as_object().unwrap().contains_key("ungrouped-b"));
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn index_page_shows_ungrouped_heading_even_when_it_is_the_only_section() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "solo-ungrouped");
            let app = test_app(dir.path());

            let (status, body) = get(&app, "/").await;
            assert_eq!(status, StatusCode::OK);
            let html = String::from_utf8(body.to_vec()).unwrap();
            assert!(
                html.contains(">Ungrouped<"),
                "heading must show even with no named groups present: {html}"
            );
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn concurrent_requests_for_one_chunk_render_it_only_once() {
        // #119 requires that concurrent requests generate an item only
        // once. Nothing implements that explicitly -- it falls out of the
        // per-radargram Mutex plus the cache re-check at the top of
        // get_or_render_chunk: whichever request wins the lock renders and
        // inserts, and the ones queued behind it find the result already
        // cached. This pins that property so a future change to the
        // locking cannot quietly reintroduce duplicate rendering.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "dup-test");
            let (app, state) = test_app_with_state(dir.path(), 8);

            let uri = "/api/v1/datasets/dup-test/views/standard/chunks/default/0/0";
            let mut handles = Vec::new();
            for _ in 0..8 {
                let app = app.clone();
                handles.push(tokio::spawn(async move { get(&app, uri).await }));
            }
            let mut responses = Vec::new();
            for handle in handles {
                responses.push(handle.await.unwrap());
            }

            let first = responses[0].1.clone();
            for (status, body) in &responses {
                assert_eq!(*status, StatusCode::OK);
                assert_eq!(body, &first, "concurrent renders disagreed");
            }

            let service = state.radargrams["dup-test"].service.lock().unwrap();
            assert_eq!(
                service.cache_len(),
                1,
                "the same chunk was rendered and cached more than once"
            );
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn a_single_render_worker_still_serves_every_request() {
        // The permit semaphore is sized from --n-workers; at 1 it fully
        // serialises rendering. Everything must still be served (just
        // slower) rather than deadlocking or timing out into a 503.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            write_test_nc(&dir.path().join("a.nc"), "one-worker");
            let (app, _state) = test_app_with_state(dir.path(), 1);

            let uris = [
                "/api/v1/datasets/one-worker/views/standard/chunks/default/0/0",
                "/api/v1/datasets/one-worker/views/standard/chunks/default/1/0",
                "/api/v1/datasets/one-worker/views/standard/overview",
            ];
            let mut handles = Vec::new();
            for uri in uris {
                let app = app.clone();
                handles.push(tokio::spawn(async move { get(&app, uri).await }));
            }
            for handle in handles {
                let (status, body) = handle.await.unwrap();
                assert_eq!(status, StatusCode::OK);
                assert!(!body.is_empty());
            }
        });
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn one_unreadable_file_does_not_break_the_whole_catalog_at_startup() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("garbage.nc"), b"not a netcdf file").unwrap();
            write_test_nc(&dir.path().join("good.nc"), "still-works");

            let app = test_app(dir.path());
            let (status, body) = get(&app, "/api/v1/datasets").await;
            assert_eq!(status, StatusCode::OK);
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["entries"].as_array().unwrap().len(), 1);
            assert!(!json["warnings"].as_array().unwrap().is_empty());

            let (status, _) =
                get(&app, "/api/v1/datasets/still-works/views/standard/overview").await;
            assert_eq!(status, StatusCode::OK);
        });
    }
}
