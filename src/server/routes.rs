//! HTTP route handlers (#120). Composes catalog (M3) and render-service
//! (M4/M5) components; no NetCDF, catalog, or rendering logic here.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use super::app::{validate_radargram_id, AppState, NO_GROUP_ID};
use super::render::grid::{ChunkGrid, OverviewSpec, ViewerRaster};
use super::render::profile::{DatasetView, RenderProfile};
use super::templates;
use crate::identity::RadargramId;

/// Stable JSON error envelope (#120): `{"error": {"code", "message"}}`.
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    /// Extra response headers. Empty for almost every error -- the
    /// envelope is meant to carry the detail -- but a few statuses are
    /// incomplete without one, `503` + `Retry-After` being the case that
    /// prompted this.
    headers: Vec<(header::HeaderName, String)>,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            headers: Vec::new(),
        }
    }

    /// Attach a response header. Invalid header values are dropped rather
    /// than panicking: a malformed hint is not worth turning an error
    /// response into a different, more confusing error response.
    fn with_header(mut self, name: header::HeaderName, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    pub(super) fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, code, message)
    }

    pub(super) fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    pub(super) fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, code, message)
    }

    /// The request is well-formed but the server is not in a state that can
    /// serve it -- no project, or started read-only. Distinct from a 400:
    /// nothing about the request needs fixing.
    pub(super) fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, code, message)
    }

    /// A conditional write whose condition no longer holds.
    pub(super) fn precondition_failed(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::PRECONDITION_FAILED, code, message)
    }

    /// Temporary overload rather than a fault: the request was valid and
    /// retrying it later may well succeed.
    fn service_unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, code, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": { "code": self.code, "message": self.message }
        });
        let mut response = (self.status, Json(body)).into_response();
        for (name, value) in self.headers {
            if let Ok(value) = header::HeaderValue::from_str(&value) {
                response.headers_mut().insert(name, value);
            }
        }
        response
    }
}

/// Same error information, rendered as an HTML page for page routes
/// rather than JSON for API routes.
pub struct PageError(ApiError);

impl IntoResponse for PageError {
    fn into_response(self) -> Response {
        let env = templates::environment();
        let tmpl = env
            .get_template("error.html.jinja")
            .expect("error template is always registered");
        let html = tmpl
            .render(minijinja::context! {
                status => self.0.status.as_u16(),
                code => self.0.code,
                message => self.0.message,
            })
            .unwrap_or_else(|e| format!("<h1>Error</h1><p>{e}</p>"));
        (self.0.status, Html(html)).into_response()
    }
}

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn list_profiles() -> impl IntoResponse {
    let names: Vec<String> = RenderProfile::built_in_profiles()
        .into_iter()
        .map(|p| p.name)
        .collect();
    Json(names)
}

#[derive(serde::Serialize)]
struct DatasetSummary {
    radargram_id: String,
    effective_label: String,
    display_name: Option<String>,
    group_name: Option<String>,
    group_id: Option<String>,
    relative_path: String,
    /// The exact stored string. Kept verbatim because the revision
    /// fingerprint (#117) hashes it -- reformatting here would silently
    /// change identity.
    processing_datetime: String,
    /// A human-readable rendering of the same instant, for UI display
    /// only. The raw value carries nanosecond precision and a numeric
    /// offset, which is noise in a catalog listing and wraps badly in a
    /// narrow card.
    processing_datetime_display: String,
    revision_id: String,
    shape: (usize, usize),
    /// Picked lines stored for this radargram, across every user.
    ///
    /// `None` when the catalog is not a project, which is different from
    /// `Some(0)`: "nowhere to save picks" and "nobody has picked this yet"
    /// should not look the same on a card.
    line_count: Option<usize>,
}

/// Format an RFC3339 processing datetime for display as `YYYY-MM-DD HH:MM`.
///
/// Falls back to the input unchanged if it does not parse: a file written
/// by a future or third-party tool should still show *something* rather
/// than an empty cell or an error.
fn format_datetime_for_display(raw: &str) -> String {
    match chrono::DateTime::parse_from_rfc3339(raw) {
        Ok(dt) => dt.format("%Y-%m-%d %H:%M").to_string(),
        Err(_) => raw.to_string(),
    }
}

/// The profile a page should render with.
///
/// The request wins, then the project's configured default, then the
/// built-in one. Three pages used to hardcode the last of those, which is
/// what made a project-wide default impossible to express.
fn resolve_profile(state: &AppState, requested: Option<String>) -> String {
    requested
        .or_else(|| state.project.as_ref().and_then(|p| p.default_profile()))
        .unwrap_or_else(|| "default".to_string())
}

fn to_summary(entry: &super::catalog::CatalogEntry) -> DatasetSummary {
    summarize(entry, None)
}

/// Count the picked lines stored for `radargram`, across all users.
///
/// Returns `None` if the count cannot be established, so a card falls back
/// to saying nothing rather than claiming zero. A malformed document on
/// disk is a reason not to answer, not a reason to report "no picks".
fn count_lines(project: &crate::project::Project, radargram: &RadargramId) -> Option<usize> {
    let store = project.documents();
    let users = crate::project::interpretations::list_users(store, radargram).ok()?;
    let mut total = 0;
    for user in users {
        let user_id = crate::identity::UserId::new(user).ok()?;
        let stored = crate::project::interpretations::read(store, radargram, &user_id).ok()?;
        if let Some(stored) = stored {
            total += stored
                .document
                .features
                .iter()
                .filter(|f| matches!(f.geometry, gprinterp::Geometry::LineString(_)))
                .count();
        }
    }
    Some(total)
}

fn summarize(entry: &super::catalog::CatalogEntry, line_count: Option<usize>) -> DatasetSummary {
    DatasetSummary {
        line_count,
        radargram_id: entry.radargram_id.to_string(),
        effective_label: entry.effective_label(),
        display_name: entry.display_name.as_ref().map(|d| d.to_string()),
        group_name: entry.group_name.as_ref().map(|g| g.to_string()),
        group_id: entry.group_id.as_ref().map(|g| g.to_string()),
        relative_path: entry.relative_path.clone(),
        processing_datetime: entry.processing_datetime.clone(),
        processing_datetime_display: format_datetime_for_display(&entry.processing_datetime),
        revision_id: entry.revision_id.to_string(),
        shape: entry.shape,
    }
}

pub async fn list_datasets(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Counted here too, so `line_count` means the same thing in the API as
    // it does on a card rather than being null for a project.
    let entries: Vec<DatasetSummary> = state
        .catalog
        .entries
        .iter()
        .map(|entry| {
            summarize(
                entry,
                state
                    .project
                    .as_ref()
                    .and_then(|project| count_lines(project, &entry.radargram_id)),
            )
        })
        .collect();
    let warnings: Vec<String> = state
        .catalog
        .warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();
    Json(serde_json::json!({ "entries": entries, "warnings": warnings }))
}

pub(super) fn lookup_dataset<'a>(
    state: &'a AppState,
    radargram_id: &str,
) -> Result<&'a super::catalog::CatalogEntry, ApiError> {
    validate_radargram_id(radargram_id)
        .map_err(|e| ApiError::bad_request("invalid_radargram_id", e))?;
    state.find_entry(radargram_id).ok_or_else(|| {
        ApiError::not_found(
            "dataset_not_found",
            format!("No dataset with id '{radargram_id}'"),
        )
    })
}

pub async fn dataset_detail(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    Ok(Json(to_summary(entry)))
}

fn lookup_view(view: &str) -> Result<DatasetView, ApiError> {
    match view {
        "standard" => Ok(DatasetView::Standard),
        _ => Err(ApiError::bad_request(
            "unknown_dataset_view",
            format!("Unknown dataset view '{view}'. Supported: standard."),
        )),
    }
}

fn lookup_profile(name: &str) -> Result<RenderProfile, ApiError> {
    RenderProfile::by_name(name).ok_or_else(|| {
        ApiError::bad_request(
            "unknown_render_profile",
            format!("Unknown render profile '{name}'."),
        )
    })
}

#[derive(Deserialize)]
pub struct ProfileQuery {
    profile: Option<String>,
}

fn image_response(bytes: Vec<u8>, profile: &RenderProfile) -> Response {
    (
        [(header::CONTENT_TYPE, profile.format.content_type())],
        bytes,
    )
        .into_response()
}

/// How long a request waits for a render permit before giving up with a
/// 503. Waiting is the right default -- a burst of index thumbnails is
/// normal traffic, not an overload -- but waiting *forever* just moves an
/// unbounded queue from the thread pool into the connection table.
const RENDER_PERMIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `Retry-After` value on the `render_busy` 503, in seconds.
const RENDER_BUSY_RETRY_AFTER_SECS: u64 = 5;

/// Run one render on a blocking thread, under a permit from
/// `AppState::render_permits`.
///
/// Two things this buys, in order of importance:
///
/// 1. **Rendering leaves the async executor.** It is CPU-bound work that
///    previously ran directly on a tokio worker while holding the
///    radargram's `Mutex`, so enough concurrent image requests could
///    starve every other route -- including the HTML pages.
/// 2. **Concurrency is bounded** by `--n-workers` rather than by tokio's
///    512-thread blocking pool default.
///
/// The permit is acquired *here*, in async context, before anything is
/// spawned. That is what makes client disconnects matter: a client that
/// goes away while queued drops this future and never starts a render.
/// A render already in flight cannot be cancelled -- `spawn_blocking`
/// tasks are not cancellable -- but it finishes into the cache, so the
/// work is not thrown away.
async fn render_under_permit<F>(
    state: Arc<AppState>,
    radargram_id: String,
    render: F,
) -> Result<Vec<u8>, ApiError>
where
    F: FnOnce(&mut super::render::service::RenderService) -> Result<Vec<u8>, String>
        + Send
        + 'static,
{
    let permit = tokio::time::timeout(
        RENDER_PERMIT_TIMEOUT,
        state.render_permits.clone().acquire_owned(),
    )
    .await
    .map_err(|_| {
        // A 503 without Retry-After leaves a client guessing. The hint is
        // deliberately short relative to the timeout that produced it: by
        // the time this fires, a permit is likely to free up sooner than
        // the full wait the caller just endured.
        ApiError::service_unavailable(
            "render_busy",
            "The server is at its render concurrency limit. Retry shortly, \
             or start it with a higher --n-workers.",
        )
        .with_header(
            header::RETRY_AFTER,
            RENDER_BUSY_RETRY_AFTER_SECS.to_string(),
        )
    })?
    .map_err(|_| ApiError::internal("render_permits_closed", "Render permits were closed"))?;

    tokio::task::spawn_blocking(move || {
        // Held until the render finishes, then released for the next
        // waiter.
        let _permit = permit;
        let radargram = state.radargrams.get(&radargram_id).ok_or_else(|| {
            ApiError::internal(
                "dataset_unavailable",
                "Dataset is cataloged but its render service failed to initialize.",
            )
        })?;
        let mut service = radargram.service.lock().map_err(|_| {
            ApiError::internal(
                "render_service_poisoned",
                "Render service lock was poisoned",
            )
        })?;
        render(&mut service).map_err(|e| ApiError::internal("render_failed", e))
    })
    .await
    .map_err(|e| ApiError::internal("render_task_failed", format!("Render task failed: {e}")))?
}

pub async fn overview_image(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, view)): Path<(String, String)>,
    Query(query): Query<ProfileQuery>,
) -> Result<Response, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    let dataset_view = lookup_view(&view)?;
    let profile = lookup_profile(query.profile.as_deref().unwrap_or("default"))?;

    let radargram = state
        .radargrams
        .get(entry.radargram_id.as_str())
        .ok_or_else(|| {
            ApiError::internal(
                "dataset_unavailable",
                "Dataset is cataloged but its render service failed to initialize.",
            )
        })?;

    let (height, width) = radargram.shape;
    let spec = OverviewSpec::new(width, height, 512);
    // Owned before spawning: everything above borrows `state`.
    let radargram_id = entry.radargram_id.to_string();
    let render_profile = profile.clone();

    let bytes = render_under_permit(state.clone(), radargram_id, move |service| {
        service.get_or_render_overview(&spec, dataset_view, &render_profile)
    })
    .await?;
    Ok(image_response(bytes, &profile))
}

pub async fn chunk_image(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, view, profile_name, x_raw, y_raw)): Path<(
        String,
        String,
        String,
        String,
        String,
    )>,
) -> Result<Response, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    let dataset_view = lookup_view(&view)?;
    let profile = lookup_profile(&profile_name)?;

    // Structurally invalid coordinates (not a non-negative integer) are a
    // 400, distinct from a well-formed but out-of-grid request (404) --
    // #118's explicit distinction.
    let x: usize = x_raw.parse().map_err(|_| {
        ApiError::bad_request("invalid_chunk_coordinate", format!("Invalid x: '{x_raw}'"))
    })?;
    let y: usize = y_raw.parse().map_err(|_| {
        ApiError::bad_request("invalid_chunk_coordinate", format!("Invalid y: '{y_raw}'"))
    })?;

    let radargram = state
        .radargrams
        .get(entry.radargram_id.as_str())
        .ok_or_else(|| {
            ApiError::internal(
                "dataset_unavailable",
                "Dataset is cataloged but its render service failed to initialize.",
            )
        })?;

    let (height, width) = radargram.shape;
    let raster = ViewerRaster::new(width, height);
    let grid = ChunkGrid::new(raster);
    let chunk = grid.chunk(x, y).ok_or_else(|| {
        ApiError::not_found(
            "image_chunk_not_found",
            "The requested image chunk is outside the radargram bounds.",
        )
    })?;

    // Owned before spawning: everything above borrows `state`.
    let radargram_id = entry.radargram_id.to_string();
    let render_profile = profile.clone();

    let bytes = render_under_permit(state.clone(), radargram_id, move |service| {
        service.get_or_render_chunk(&chunk, dataset_view, &render_profile)
    })
    .await?;
    Ok(image_response(bytes, &profile))
}

/// One group's heading and members for the index page. `id` is also the
/// `data-group` value the group map's JS fetches
/// `/api/v1/groups/{id}/tracks` with, so it must stay the stable
/// [`crate::identity::GroupId`] -- never the free-form, possibly-changing
/// display name, which is `label` instead (#121 planning round: mirror the
/// radargram id/display-name split one level up).
#[derive(serde::Serialize)]
struct GroupSummary {
    id: String,
    label: String,
    entries: Vec<DatasetSummary>,
}

/// The project settings page.
///
/// Thin on purpose: one setting today, and the shape to hang the rest on
/// when multi-user and deployment settings arrive. Renders for a
/// non-project catalog too, explaining why there is nothing to configure.
pub async fn settings_page(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProfileQuery>,
) -> Result<impl IntoResponse, PageError> {
    let active_profile = resolve_profile(&state, query.profile);
    lookup_profile(&active_profile).map_err(PageError)?;

    let env = templates::environment();
    let tmpl = env
        .get_template("settings.html.jinja")
        .expect("settings template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            project => state.project.is_some(),
            writable => state.writable,
            active_profile => active_profile,
            project_name => state
                .project
                .as_ref()
                .and_then(|p| p.config().project.name.clone()),
            project_root => state
                .project
                .as_ref()
                .map(|p| p.root().display().to_string()),
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

/// The layer management page.
///
/// A page of its own rather than a panel in the viewer: the vocabulary is
/// project-scoped, editing it is a deliberate act rather than something done
/// mid-pick, and a delete needs room to say what it would affect.
///
/// Renders for a non-project catalog too, explaining why there is nothing to
/// edit -- a 404 here would be an odd answer to "show me the layers".
pub async fn layers_page(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProfileQuery>,
) -> Result<impl IntoResponse, PageError> {
    // This page has nothing to render, but it carries the profile so the
    // menu's links out of it keep the viewing preference the user arrived
    // with. An unknown profile is rejected rather than passed on, so a bad
    // value cannot propagate silently through the menu.
    let active_profile = resolve_profile(&state, query.profile);
    lookup_profile(&active_profile).map_err(PageError)?;

    let env = templates::environment();
    let tmpl = env
        .get_template("layers.html.jinja")
        .expect("layers template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            project => state.project.is_some(),
            writable => state.writable,
            active_profile => active_profile,
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

pub async fn index_page(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ProfileQuery>,
) -> Result<impl IntoResponse, PageError> {
    let active_profile = resolve_profile(&state, query.profile);
    lookup_profile(&active_profile).map_err(PageError)?;
    let profiles: Vec<String> = RenderProfile::built_in_profiles()
        .into_iter()
        .map(|p| p.name)
        .collect();

    // Counted once per entry here and reused below, rather than per card:
    // the same radargram appears in both the flat list and its group, and
    // each count is a directory read plus a JSON parse.
    let line_counts: std::collections::HashMap<String, Option<usize>> = match &state.project {
        Some(project) => state
            .catalog
            .entries
            .iter()
            .map(|e| {
                (
                    e.radargram_id.to_string(),
                    count_lines(project, &e.radargram_id),
                )
            })
            .collect(),
        None => std::collections::HashMap::new(),
    };
    let summarize_entry = |entry: &super::catalog::CatalogEntry| {
        summarize(
            entry,
            line_counts
                .get(entry.radargram_id.as_str())
                .copied()
                .flatten(),
        )
    };

    let entries: Vec<DatasetSummary> = state.catalog.entries.iter().map(&summarize_entry).collect();
    let warnings: Vec<String> = state
        .catalog
        .warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();

    // Every entry gets one map on the index page (#121): named groups,
    // and "Ungrouped" for entries with none, presented identically
    // rather than as a special case -- entries_in_group(NO_GROUP_ID)
    // already matches group_id.is_none() for exactly this reason.
    let mut group_ids: Vec<&str> = state
        .catalog
        .entries
        .iter()
        .filter_map(|e| e.group_id.as_ref().map(|g| g.as_str()))
        .collect();
    group_ids.sort_unstable();
    group_ids.dedup();
    let mut groups: Vec<GroupSummary> = group_ids
        .into_iter()
        .map(|id| {
            let label = state
                .catalog
                .group_names
                .iter()
                .find(|(gid, _)| gid.as_str() == id)
                .map(|(_, name)| name.to_string())
                .unwrap_or_else(|| id.to_string());
            GroupSummary {
                id: id.to_string(),
                label,
                entries: state
                    .entries_in_group(id)
                    .into_iter()
                    .map(&summarize_entry)
                    .collect(),
            }
        })
        .collect();
    let ungrouped_entries: Vec<DatasetSummary> = state
        .catalog
        .entries
        .iter()
        .filter(|e| e.group_id.is_none())
        .map(&summarize_entry)
        .collect();
    if !ungrouped_entries.is_empty() {
        groups.push(GroupSummary {
            id: NO_GROUP_ID.to_string(),
            label: "Ungrouped".to_string(),
            entries: ungrouped_entries,
        });
    }

    let env = templates::environment();
    let tmpl = env
        .get_template("index.html.jinja")
        .expect("index template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            entries => entries,
            warnings => warnings,
            groups => groups,
            profiles => profiles,
            active_profile => active_profile,
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

pub async fn viewer_page(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
    Query(query): Query<ProfileQuery>,
) -> Result<impl IntoResponse, PageError> {
    let entry = lookup_dataset(&state, &radargram_id).map_err(PageError)?;
    let active_profile = resolve_profile(&state, query.profile);
    lookup_profile(&active_profile).map_err(PageError)?;

    let radargram = state
        .radargrams
        .get(entry.radargram_id.as_str())
        .ok_or_else(|| {
            PageError(ApiError::internal(
                "dataset_unavailable",
                "Dataset is cataloged but its render service failed to initialize.",
            ))
        })?;
    let (height, width) = radargram.shape;
    let raster = ViewerRaster::new(width, height);
    let grid = ChunkGrid::new(raster);

    let profiles: Vec<String> = RenderProfile::built_in_profiles()
        .into_iter()
        .map(|p| p.name)
        .collect();

    let env = templates::environment();
    let tmpl = env
        .get_template("viewer.html.jinja")
        .expect("viewer template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            radargram_id => entry.radargram_id.to_string(),
            effective_label => entry.effective_label(),
            group_name => entry.group_name.as_ref().map(|g| g.to_string()),
            group_id => entry.group_id.as_ref().map(|g| g.to_string()),
            revision_id => entry.revision_id.to_string(),
            // First 7 hex characters, `git`-style, for the collapsed
            // banner row -- the full ID moves to the metadata dialog.
            revision_short => entry.revision_id.to_string().chars().take(7).collect::<String>(),
            processing_datetime => format_datetime_for_display(&entry.processing_datetime),
            shape_height => height,
            shape_width => width,
            project => state.project.is_some(),
            writable => state.writable,
            user => crate::identity::DEFAULT_USER,
            profiles => profiles,
            active_profile => active_profile,
            chunk_size => super::render::grid::CHUNK_SIZE,
            n_cols => grid.n_cols,
            n_rows => grid.n_rows,
            viewer_width => raster.width,
            viewer_height => raster.height,
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

#[derive(serde::Serialize)]
struct TrackVertexJson {
    trace_index: u32,
    lon: f64,
    lat: f64,
}

#[derive(serde::Serialize)]
struct TrackSegmentJson {
    segment_index: usize,
    trace_start: u32,
    trace_end: u32,
    n_traces: u32,
    length_m: f64,
    vertices: Vec<TrackVertexJson>,
}

#[derive(serde::Serialize)]
struct TrackJson {
    segments: Vec<TrackSegmentJson>,
}

fn track_to_json(track: &super::track::Track) -> TrackJson {
    TrackJson {
        segments: track
            .segments
            .iter()
            .map(|s| TrackSegmentJson {
                segment_index: s.segment_index,
                trace_start: s.trace_start,
                trace_end: s.trace_end,
                n_traces: s.n_traces,
                length_m: s.length_m,
                vertices: s
                    .vertices
                    .iter()
                    .map(|v| TrackVertexJson {
                        trace_index: v.trace_index,
                        lon: v.lon,
                        lat: v.lat,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// This radargram's own track (#121's cursor-sync feature).
pub async fn dataset_track(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    let path = state
        .absolute_path(entry)
        .map_err(|e| ApiError::internal("path_resolve_failed", e))?;
    let track = super::track::read_track_from_netcdf(&path)
        .map_err(|e| ApiError::internal("track_read_failed", e))?;
    Ok(Json(track_to_json(&track)))
}

/// Every radargram's track in one group, for sibling-track display on the
/// viewer map and the index page's per-group overview map. A track that
/// fails to read is silently skipped here rather than failing the whole
/// response -- one bad sibling should not break the rest, matching the
/// spirit of #122's "one bad candidate does not abort discovery."
pub async fn group_tracks(
    State(state): State<Arc<AppState>>,
    Path(group): Path<String>,
) -> impl IntoResponse {
    let mut out = serde_json::Map::new();
    for entry in state.entries_in_group(&group) {
        let Ok(path) = state.absolute_path(entry) else {
            continue;
        };
        if let Ok(track) = super::track::read_track_from_netcdf(&path) {
            out.insert(
                entry.radargram_id.to_string(),
                serde_json::json!({
                    "effective_label": entry.effective_label(),
                    "track": track_to_json(&track),
                }),
            );
        }
    }
    Json(serde_json::Value::Object(out))
}

/// Widen an `f32` NetCDF attribute to `f64` via its shortest round-trip
/// decimal string, rather than a plain numeric cast. A plain cast (`v as
/// f64`) preserves the `f32`'s exact binary value, which `f64`'s extra
/// precision then renders as noise (`0.168_f32` -> `0.16799999773502350`).
/// `f32::to_string()` already produces the shortest decimal that
/// round-trips to the same `f32`, so re-parsing it as `f64` recovers the
/// value a human actually meant.
fn f32_to_f64_exact(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(v as f64)
}

fn attribute_value_to_json(value: netcdf::AttributeValue) -> serde_json::Value {
    use netcdf::AttributeValue::*;
    match value {
        Uchar(v) => serde_json::json!(v),
        Uchars(v) => serde_json::json!(v),
        Schar(v) => serde_json::json!(v),
        Schars(v) => serde_json::json!(v),
        Ushort(v) => serde_json::json!(v),
        Ushorts(v) => serde_json::json!(v),
        Short(v) => serde_json::json!(v),
        Shorts(v) => serde_json::json!(v),
        Uint(v) => serde_json::json!(v),
        Uints(v) => serde_json::json!(v),
        Int(v) => serde_json::json!(v),
        Ints(v) => serde_json::json!(v),
        Ulonglong(v) => serde_json::json!(v),
        Ulonglongs(v) => serde_json::json!(v),
        Longlong(v) => serde_json::json!(v),
        Longlongs(v) => serde_json::json!(v),
        Float(v) => serde_json::json!(f32_to_f64_exact(v)),
        Floats(v) => {
            serde_json::json!(v.into_iter().map(f32_to_f64_exact).collect::<Vec<_>>())
        }
        Double(v) => serde_json::json!(v),
        Doubles(v) => serde_json::json!(v),
        Str(v) => serde_json::json!(v),
        Strs(v) => serde_json::json!(v),
    }
}

/// Attribute name -> display label overrides for cases the generic
/// strip-prefix/underscore-to-space/capitalize rule gets wrong (acronyms,
/// mainly).
fn label_override(name: &str) -> Option<&'static str> {
    match name {
        "crs" => Some("CRS"),
        "ridal_group_id" => Some("Group ID"),
        // The generic strip-`ridal_`-prefix rule would otherwise reduce
        // this to a bare "Version", which reads as the *radargram's* own
        // version rather than the tool that processed it.
        "ridal_version" => Some("Ridal version"),
        _ => None,
    }
}

/// `ridal_processing_datetime` -> "Processing datetime",
/// `original_filepaths` -> "Original filepaths": strip the `ridal_`
/// namespace prefix (meaningless to a human reader), replace underscores
/// with spaces, and capitalize only the first letter -- matching how the
/// rest of the dialog's prose is cased.
fn prettify_label(name: &str) -> String {
    if let Some(overridden) = label_override(name) {
        return overridden.to_string();
    }
    let stripped = name.strip_prefix("ridal_").unwrap_or(name);
    let mut words = stripped.split('_');
    let mut out = String::new();
    if let Some(first) = words.next() {
        let mut chars = first.chars();
        if let Some(c) = chars.next() {
            out.extend(c.to_uppercase());
        }
        out.push_str(chars.as_str());
    }
    for word in words {
        out.push(' ');
        out.push_str(word);
    }
    out
}

/// Round a float to 4 decimal places for display, trimming trailing zeros
/// (and a bare trailing `.`) so a whole number like `5.0` still reads as
/// `5`. Fixes the `f32`-precision-widening artifact at the point it
/// actually matters -- what a human reads -- on top of the exact-string
/// recovery `f32_to_f64_exact` already does at parse time.
fn format_rounded(v: f64) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    let rounded = (v * 10_000.0).round() / 10_000.0;
    let mut s = format!("{rounded:.4}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

fn plain_value_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Number(n) if n.is_f64() => format_rounded(n.as_f64().unwrap()),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(plain_value_string)
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Object(_) => value.to_string(),
    }
}

/// Format one attribute's value for display, appending its `*_unit`
/// sibling (if any) in parentheses rather than showing it as a separate
/// row -- "Time interval  0.3 (s)".
fn format_display_value(value: &serde_json::Value, unit: Option<&str>) -> String {
    let base = plain_value_string(value);
    match unit {
        Some(unit) => format!("{base} ({unit})"),
        None => base,
    }
}

#[derive(serde::Serialize)]
struct MetadataEntry {
    label: String,
    value: String,
}

/// Curated priority for known attribute keys: `(tier, order-within-tier)`.
/// Keys not listed fall into tier 3 ("everything else"), ordered
/// alphabetically by their prettified label -- the plan's "identity ->
/// acquisition -> processing -> everything else alphabetically".
///
/// `__revision_id`, `__start_stop_datetime` and `__shape` are synthetic
/// keys for entries this function builds itself rather than reading
/// verbatim from `raw`.
fn curated_priority(key: &str) -> (u8, usize) {
    const IDENTITY: &[&str] = &[
        "ridal_radargram_id",
        "__revision_id",
        "ridal_display_name",
        "ridal_group_name",
        "ridal_group_id",
        "__start_stop_datetime",
        "__shape",
    ];
    const ACQUISITION: &[&str] = &[
        "antenna",
        "antenna_separation",
        "frequency_steps",
        "vertical_sampling_frequency",
        "time_interval",
        "medium_velocity",
        "crs",
        "elevation_correction",
        "total_distance",
    ];
    const PROCESSING: &[&str] = &["ridal_processing_datetime", "ridal_version"];

    if let Some(i) = IDENTITY.iter().position(|&k| k == key) {
        return (0, i);
    }
    if let Some(i) = ACQUISITION.iter().position(|&k| k == key) {
        return (1, i);
    }
    if let Some(i) = PROCESSING.iter().position(|&k| k == key) {
        return (2, i);
    }
    (3, 0)
}

/// Attribute keys never shown as their own row: `processing_log` and
/// `processing_steps` get dedicated fields in the response instead (see
/// [`dataset_attributes`]); `start_datetime`/`stop_datetime` are merged
/// into one synthetic "Start/stop datetime" row;
/// `ridal_user_metadata_json` duplicates the flattened user-metadata
/// attributes already shown individually; `original_filepaths` gets its
/// own field and a dedicated `<details>` in the viewer, since it can be
/// arbitrarily long (many merged inputs, each a long path) and would
/// otherwise render as one unreadable comma-joined row; `Conventions` is
/// always `CF-1.7` -- a file has to already be Ridal-produced (and thus
/// CF-1.7) to be recognized as `Supported` at all, so the row is never
/// actionable. `*_unit` keys are consumed by their base attribute, not
/// skipped by name here.
const SKIP_FROM_ENTRIES: &[&str] = &[
    "start_datetime",
    "stop_datetime",
    "processing_log",
    "processing_steps",
    "ridal_user_metadata_json",
    "original_filepaths",
    "Conventions",
];

fn build_metadata_entries(
    raw: &serde_json::Map<String, serde_json::Value>,
    shape: (usize, usize),
    revision_id: &str,
) -> Vec<MetadataEntry> {
    struct Entry {
        key: String,
        label: String,
        value: String,
    }
    let mut entries = Vec::new();

    // `revision_id` is a server-computed fingerprint
    // (`RevisionId::fingerprint_v1`), never written as a file attribute,
    // so it is never in `raw` -- without this synthetic entry the full
    // checksum would not appear anywhere in the UI at all (only the
    // abbreviated form in the banner).
    entries.push(Entry {
        key: "__revision_id".to_string(),
        label: "Revision".to_string(),
        value: revision_id.to_string(),
    });

    if let (Some(start), Some(stop)) = (
        raw.get("start_datetime").and_then(|v| v.as_str()),
        raw.get("stop_datetime").and_then(|v| v.as_str()),
    ) {
        entries.push(Entry {
            key: "__start_stop_datetime".to_string(),
            label: "Start/stop datetime".to_string(),
            value: format!(
                "{} / {}",
                format_datetime_for_display(start),
                format_datetime_for_display(stop)
            ),
        });
    }

    entries.push(Entry {
        key: "__shape".to_string(),
        label: "Shape (samples \u{d7} traces)".to_string(),
        value: format!("{} \u{d7} {}", shape.0, shape.1),
    });

    for (key, value) in raw {
        if SKIP_FROM_ENTRIES.contains(&key.as_str()) || key.ends_with("_unit") {
            continue;
        }
        // The only other raw datetime attribute besides start/stop
        // (merged above): needs the same display formatting, not the raw
        // nanosecond-precision RFC3339 string.
        if key == "ridal_processing_datetime" {
            if let Some(raw_dt) = value.as_str() {
                entries.push(Entry {
                    key: key.clone(),
                    label: prettify_label(key),
                    value: format_datetime_for_display(raw_dt),
                });
                continue;
            }
        }
        let unit = raw.get(&format!("{key}_unit")).and_then(|v| v.as_str());
        entries.push(Entry {
            key: key.clone(),
            label: prettify_label(key),
            value: format_display_value(value, unit),
        });
    }

    entries.sort_by(|a, b| {
        curated_priority(&a.key)
            .cmp(&curated_priority(&b.key))
            .then_with(|| a.label.cmp(&b.label))
    });

    entries
        .into_iter()
        .map(|e| MetadataEntry {
            label: e.label,
            value: e.value,
        })
        .collect()
}

/// The viewer's metadata dialog: curated, human-readable `entries`
/// (prettified labels, merged units, rounded floats, merged start/stop,
/// curated order), `processing_steps`/`processing_log` as their own
/// fields (the log needs its per-step structure preserved, not squashed
/// into a single-line entry value), and the complete `raw` attribute set
/// as an escape hatch for anything the curated view doesn't surface.
pub async fn dataset_attributes(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    let path = state
        .absolute_path(entry)
        .map_err(|e| ApiError::internal("path_resolve_failed", e))?;
    let file = netcdf::open(&path)
        .map_err(|e| ApiError::internal("attributes_read_failed", format!("{e}")))?;
    let mut raw = serde_json::Map::new();
    for attr in file.attributes() {
        let name = attr.name().to_string();
        if let Ok(value) = attr.value() {
            raw.insert(name, attribute_value_to_json(value));
        }
    }

    let processing_steps: Vec<String> = raw
        .get("processing_steps")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let processing_log = raw
        .get("processing_log")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    // Its own field (not a curated entry, see `SKIP_FROM_ENTRIES`): can be
    // arbitrarily long, so the viewer gives it a dedicated `<details>`
    // rather than one comma-joined row.
    let original_filepaths: Vec<String> = raw
        .get("original_filepaths")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let entries = build_metadata_entries(&raw, entry.shape, entry.revision_id.as_str());

    Ok(Json(serde_json::json!({
        "entries": entries,
        "processing_steps": processing_steps,
        "processing_log": processing_log,
        "original_filepaths": original_filepaths,
        "raw": raw,
    })))
}

/// Distance/TWTT/depth axes for the viewer's cursor readout (item 3 of the
/// planning round). `distance`/`twtt`/`depth` are written unconditionally
/// by `export.rs`, but small hand-built test fixtures
/// (`write_test_nc`/`write_test_nc_with_track`) do not write them -- so
/// each axis degrades independently to `null` rather than failing the
/// whole response.
#[derive(serde::Serialize)]
struct AxesJson {
    distance: Option<Vec<f64>>,
    twtt: Option<Vec<f64>>,
    depth: Option<Vec<f64>>,
}

pub async fn dataset_axes(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let entry = lookup_dataset(&state, &radargram_id)?;
    let path = state
        .absolute_path(entry)
        .map_err(|e| ApiError::internal("path_resolve_failed", e))?;
    let file =
        netcdf::open(&path).map_err(|e| ApiError::internal("axes_read_failed", format!("{e}")))?;
    Ok(Json(AxesJson {
        distance: super::track::read_f64_variable(&file, "distance").ok(),
        twtt: super::track::read_f64_variable(&file, "twtt").ok(),
        depth: super::track::read_f64_variable(&file, "depth").ok(),
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        build_metadata_entries, f32_to_f64_exact, format_datetime_for_display, format_rounded,
        prettify_label, ApiError, RENDER_BUSY_RETRY_AFTER_SECS,
    };
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    #[test]
    fn service_unavailable_carries_a_retry_after_hint() {
        // A 503 without Retry-After leaves the client guessing how long
        // to back off. The envelope still carries the code and message.
        let response = ApiError::service_unavailable("render_busy", "busy")
            .with_header(
                header::RETRY_AFTER,
                RENDER_BUSY_RETRY_AFTER_SECS.to_string(),
            )
            .into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get(header::RETRY_AFTER).unwrap(),
            &RENDER_BUSY_RETRY_AFTER_SECS.to_string()
        );
    }

    #[test]
    fn errors_without_extra_headers_are_unaffected() {
        let response = ApiError::internal("boom", "went wrong").into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
    }

    #[test]
    fn display_datetime_drops_subsecond_noise() {
        // The real shape written by export.rs: chrono::Local::now()
        // to_rfc3339(), i.e. nanosecond precision plus a numeric offset.
        assert_eq!(
            format_datetime_for_display("2026-08-26T20:57:41.407887786+00:00"),
            "2026-08-26 20:57"
        );
    }

    #[test]
    fn display_datetime_handles_z_suffix_and_whole_seconds() {
        assert_eq!(
            format_datetime_for_display("2020-01-01T00:00:00Z"),
            "2020-01-01 00:00"
        );
    }

    #[test]
    fn display_datetime_falls_back_to_the_raw_value() {
        // A file from a future or third-party writer should still show
        // something rather than an empty cell.
        assert_eq!(
            format_datetime_for_display("not a datetime"),
            "not a datetime"
        );
        assert_eq!(format_datetime_for_display(""), "");
    }

    #[test]
    fn f32_widening_recovers_the_shortest_decimal() {
        // The exact reported artifact: 0.168_f32 cast plainly to f64
        // reads back as 0.16799999773502350.
        assert_eq!(f32_to_f64_exact(0.168_f32), 0.168_f64);
    }

    #[test]
    fn prettify_label_strips_ridal_prefix_and_title_cases() {
        assert_eq!(
            prettify_label("ridal_processing_datetime"),
            "Processing datetime"
        );
        assert_eq!(prettify_label("original_filepaths"), "Original filepaths");
        assert_eq!(prettify_label("crs"), "CRS");
    }

    #[test]
    fn format_rounded_trims_to_four_decimals() {
        assert_eq!(format_rounded(0.168_f32 as f64), "0.168");
        assert_eq!(format_rounded(5.0), "5");
        assert_eq!(format_rounded(1.0 / 3.0), "0.3333");
        assert_eq!(format_rounded(-0.00001), "0");
    }

    #[test]
    fn metadata_entries_merge_units_and_start_stop_and_curate_order() {
        let raw: serde_json::Map<String, serde_json::Value> = serde_json::json!({
            "ridal_radargram_id": "dronbreen-2022",
            "ridal_group_name": "Drønbreen",
            "ridal_group_id": "dronbreen",
            "ridal_processing_datetime": "2026-08-26T20:57:41.407887786+00:00",
            "ridal_version": "ridal version 0.5.2 by Erik Schytt Mannerfelt",
            "start_datetime": "2022-03-29T00:00:00Z",
            "stop_datetime": "2022-03-29T01:00:00Z",
            "time_interval": 0.3,
            "time_interval_unit": "s",
            "processing_log": "step 1 (duration: 1s):\tdid a thing",
            "processing_steps": ["step 1"],
            "ridal_user_metadata_json": "{}",
            "original_filepaths": ["a.rd3", "b.rd3"],
            "Conventions": "CF-1.7",
            "crs": "EPSG:32633",
        })
        .as_object()
        .unwrap()
        .clone();

        let entries = build_metadata_entries(&raw, (400, 1200), "0123456789abcdef");
        let by_label: std::collections::HashMap<&str, &str> = entries
            .iter()
            .map(|e| (e.label.as_str(), e.value.as_str()))
            .collect();

        assert_eq!(
            by_label["Start/stop datetime"],
            "2022-03-29 00:00 / 2022-03-29 01:00"
        );
        assert_eq!(by_label["Time interval"], "0.3 (s)");
        assert_eq!(by_label["Shape (samples \u{d7} traces)"], "400 \u{d7} 1200");
        assert_eq!(by_label["Group name"], "Drønbreen");
        assert_eq!(by_label["Group ID"], "dronbreen");
        // ridal_processing_datetime gets the same display formatting as
        // start/stop, not the raw nanosecond-precision RFC3339 string.
        assert_eq!(by_label["Processing datetime"], "2026-08-26 20:57");
        // The full revision checksum: a server-computed fingerprint, never
        // a file attribute, so it must come from the explicit parameter,
        // not from `raw`.
        assert_eq!(by_label["Revision"], "0123456789abcdef");
        // ridal_version gets a clearer label than the generic strip-prefix
        // rule would produce ("Version" reads as the radargram's own).
        assert_eq!(
            by_label["Ridal version"],
            "ridal version 0.5.2 by Erik Schytt Mannerfelt"
        );
        // Merged-unit and internal-use attributes must not also appear as
        // their own separate rows.
        assert!(!by_label.contains_key("Time interval unit"));
        assert!(!by_label.contains_key("Processing log"));
        assert!(!by_label.contains_key("Processing steps"));
        assert!(!by_label.contains_key("User metadata json"));
        // Unbounded (could be many long paths) and always-constant
        // attributes get their own handling elsewhere, not a curated row.
        assert!(!by_label.contains_key("Original filepaths"));
        assert!(!by_label.contains_key("Conventions"));

        // Identity tier (radargram ID, revision) sorts ahead of
        // acquisition (CRS).
        let id_pos = entries
            .iter()
            .position(|e| e.label == "Radargram id")
            .unwrap();
        let revision_pos = entries.iter().position(|e| e.label == "Revision").unwrap();
        let crs_pos = entries.iter().position(|e| e.label == "CRS").unwrap();
        assert!(id_pos < revision_pos);
        assert!(revision_pos < crs_pos);
    }
}
