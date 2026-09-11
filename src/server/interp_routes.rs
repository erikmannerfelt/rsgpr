//! Write routes for interpretations and layer definitions.
//!
//! Kept separate from [`super::routes`] because everything here shares one
//! set of concerns the read-only routes do not have: a project must exist,
//! writes must be enabled, and every mutation is conditional on a version so
//! two clients cannot silently overwrite each other.
//!
//! # Conditional writes
//!
//! Reads return an `ETag`. A `PUT` is expected to send it back as
//! `If-Match`, and is refused with `412 Precondition Failed` if the document
//! has moved on. `If-Match: *` requires the document to already exist, and
//! `If-None-Match: *` requires that it does not -- both standard, and both
//! more useful here than a bare overwrite.
//!
//! A `PUT` with no condition at all is accepted and overwrites, because the
//! alternative is making the simplest possible client impossible to write.
//! The browser GUI always sends one.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;

use super::app::{AppState, DownloadScope};
use super::routes::{lookup_dataset, ApiError};
use crate::identity::{RadargramId, UserId};
use crate::interp::checks;
use crate::project::store::{Expectation, StoreError, Version};
use crate::project::{interpretations, layers, Project};

/// The project, if this server has one and writes are allowed.
fn writable_project(state: &AppState) -> Result<&Project, ApiError> {
    let Some(project) = state.project.as_ref() else {
        return Err(ApiError::conflict(
            "not_a_project",
            "This catalog is not a Ridal project, so there is nowhere to save \
             interpretations. Run `ridal project init` in the directory you are \
             serving, then restart.",
        ));
    };
    if !state.writable {
        return Err(ApiError::conflict(
            "read_only",
            "This server was started read-only, so interpretations cannot be saved.",
        ));
    }
    Ok(project)
}

/// The project for reading. Reads do not require writes to be enabled.
fn readable_project(state: &AppState) -> Result<&Project, ApiError> {
    state.project.as_ref().ok_or_else(|| {
        ApiError::not_found(
            "not_a_project",
            "This catalog is not a Ridal project, so it holds no interpretations.",
        )
    })
}

/// Turn the request's conditional headers into a store [`Expectation`].
///
/// `If-Match` wins over `If-None-Match` when both are present; sending both
/// is contradictory, and honouring the stricter one is the safer reading.
fn expectation_from(headers: &HeaderMap) -> Expectation {
    if let Some(value) = headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()) {
        let value = value.trim();
        if value == "*" {
            // "must already exist"; the store has no such variant, and a
            // real version is stricter than needed but never wrong here
            // because the client only knows a version if it read one.
            return Expectation::Any;
        }
        return Expectation::Version(Version::from_header(value));
    }
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim() == "*")
        .unwrap_or(false)
    {
        return Expectation::Absent;
    }
    Expectation::Any
}

fn parse_radargram(raw: &str) -> Result<RadargramId, ApiError> {
    RadargramId::new(raw).map_err(|e| ApiError::bad_request("invalid_radargram_id", e))
}

fn parse_user(raw: &str) -> Result<UserId, ApiError> {
    UserId::new(raw).map_err(|e| ApiError::bad_request("invalid_user", e))
}

/// Who the caller is.
///
/// The single place that answers that question, so authentication replaces
/// this function body rather than being threaded through every route. It
/// will gain the request headers when sessions land -- a signed cookie is
/// read from there -- which is a mechanical change to the handful of call
/// sites below.
///
/// Until then, everyone is [`crate::identity::DEFAULT_USER`], which is the
/// single-user behaviour Ridal has always had.
pub(super) fn current_user(_state: &AppState) -> Result<UserId, ApiError> {
    UserId::new(crate::identity::DEFAULT_USER)
        .map_err(|e| ApiError::internal("invalid_default_user", e))
}

/// The user a write is allowed to target, given the one named in the path.
///
/// Interpretations are per-user by design: one person cannot modify
/// another's picks, and that is a property of the data rather than a
/// permission an administrator could grant. Without this check the path
/// parameter *is* the authorisation -- `PUT .../interpretations/alice`
/// would write Alice's document for anyone who asked.
///
/// The path keeps naming the user rather than being dropped from write
/// URLs, so reads and writes share one URL shape and adding
/// authentication changes no route at all.
fn writing_as(state: &AppState, path_user: &str) -> Result<UserId, ApiError> {
    let requested = parse_user(path_user)?;
    let current = current_user(state)?;
    if requested != current {
        return Err(ApiError::forbidden(
            "not_your_interpretation",
            format!(
                "Interpretations belong to the user who drew them; \
                 you are '{}' and cannot write '{}'.",
                current.as_str(),
                requested.as_str()
            ),
        ));
    }
    Ok(current)
}

/// Map a store conflict to `412`, everything else to `500`.
fn store_error(error: &StoreError) -> ApiError {
    match error {
        StoreError::Conflict { .. } => {
            ApiError::precondition_failed("version_conflict", error.to_string())
        }
        _ => ApiError::internal("store_failed", error.to_string()),
    }
}

fn interpretation_error(error: interpretations::InterpretationError) -> ApiError {
    match &error {
        interpretations::InterpretationError::Store(e) => store_error(e),
        interpretations::InterpretationError::KeyMismatch { .. } => {
            ApiError::bad_request("key_mismatch", error.to_string())
        }
        interpretations::InterpretationError::Malformed { .. } => {
            ApiError::internal("malformed_interpretation", error.to_string())
        }
    }
}

fn layer_error(error: layers::LayerError) -> ApiError {
    match &error {
        layers::LayerError::Store(e) => store_error(e),
        layers::LayerError::Malformed { .. } => {
            ApiError::internal("malformed_layers", error.to_string())
        }
        layers::LayerError::DuplicateId(_) | layers::LayerError::InvalidId { .. } => {
            ApiError::bad_request("invalid_layers", error.to_string())
        }
    }
}

/// `GET /api/v1/datasets/{id}/interpretations` -- who has interpreted this.
pub async fn list_interpretations(
    State(state): State<Arc<AppState>>,
    Path(radargram_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let project = readable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let users = interpretations::list_users(project.documents(), &radargram)
        .map_err(interpretation_error)?;
    Ok(Json(serde_json::json!({
        "radargram_id": radargram.as_str(),
        "users": users,
        "writable": state.writable,
    })))
}

/// `GET /api/v1/datasets/{id}/interpretations/{user}`
pub async fn get_interpretation(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, user)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let project = readable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let user = parse_user(&user)?;

    let stored = interpretations::read(project.documents(), &radargram, &user)
        .map_err(interpretation_error)?
        .ok_or_else(|| {
            ApiError::not_found(
                "interpretation_not_found",
                format!(
                    "'{}' has no interpretation of '{}'",
                    user.as_str(),
                    radargram.as_str()
                ),
            )
        })?;

    let body = serde_json::to_value(&stored.document)
        .map_err(|e| ApiError::internal("serialize_failed", e.to_string()))?;
    Ok((
        [(header::ETAG, format!("\"{}\"", stored.version))],
        Json(body),
    ))
}

/// `PUT /api/v1/datasets/{id}/interpretations/{user}`
pub async fn put_interpretation(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, user)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    let project = writable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let user = writing_as(&state, &user)?;

    // The dataset must be in this catalog. Otherwise a typo in the URL
    // silently creates an interpretation directory for a radargram that does
    // not exist, which nothing would ever read.
    if state.find_entry(radargram.as_str()).is_none() {
        return Err(ApiError::not_found(
            "dataset_not_found",
            format!("No dataset with id '{}'", radargram.as_str()),
        ));
    }

    let document: gprinterp::Document = serde_json::from_value(body)
        .map_err(|e| ApiError::bad_request("invalid_interpretation", e.to_string()))?;

    let report = gprinterp::validate(&document);
    if !report.errors.is_empty() {
        let joined: Vec<String> = report.errors.iter().map(|e| e.to_string()).collect();
        return Err(ApiError::bad_request(
            "invalid_interpretation",
            joined.join("; "),
        ));
    }

    // The overhang guardrail, enforced here because this is the boundary
    // where picks enter the store. The browser checks while drawing too, but
    // that is a convenience: anything reaching this route -- a second client,
    // a script, a replayed request -- has to pass the same rule.
    let (layer_set, _) = layers::read(project.documents()).map_err(layer_error)?;
    let violations = checks::check(&document, &|label| layer_set.allows_overhangs(label));
    if !violations.is_empty() {
        let joined: Vec<String> = violations.iter().map(|v| v.to_string()).collect();
        return Err(ApiError::bad_request("overhang", joined.join("; ")));
    }

    let existed = interpretations::read(project.documents(), &radargram, &user)
        .map_err(interpretation_error)?
        .is_some();

    let version = interpretations::write(
        project.documents(),
        &radargram,
        &user,
        &document,
        &expectation_from(&headers),
    )
    .map_err(interpretation_error)?;

    let status = if existed {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((
        status,
        [(header::ETAG, format!("\"{version}\""))],
        Json(serde_json::json!({
            "radargram_id": radargram.as_str(),
            "user": user.as_str(),
            "version": version.as_str(),
            // Warnings are reported, not enforced: the format is permissive
            // by design and a document missing a stable feature id still
            // saves correctly.
            "warnings": report.warnings.iter().map(|w| w.to_string()).collect::<Vec<_>>(),
        })),
    ))
}

/// `DELETE /api/v1/datasets/{id}/interpretations/{user}`
pub async fn delete_interpretation(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, user)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let project = writable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let user = writing_as(&state, &user)?;

    let existed = interpretations::remove(project.documents(), &radargram, &user)
        .map_err(interpretation_error)?;
    if !existed {
        return Err(ApiError::not_found(
            "interpretation_not_found",
            format!(
                "'{}' has no interpretation of '{}'",
                user.as_str(),
                radargram.as_str()
            ),
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /api/v1/layers` -- the project's layer vocabulary.
///
/// A catalog with no project returns an empty vocabulary rather than a 404:
/// the viewer asks for this on every page load, and a read-only catalog
/// having no layers is a normal answer, not a failure.
pub async fn get_layers(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let Some(project) = state.project.as_ref() else {
        return Ok((
            [(header::ETAG, String::new())],
            Json(serde_json::json!({
                "layers": [],
                "writable": false,
            })),
        ));
    };

    let (set, version) = layers::read(project.documents()).map_err(layer_error)?;
    let etag = version
        .map(|v| format!("\"{v}\""))
        .unwrap_or_else(String::new);
    Ok((
        [(header::ETAG, etag)],
        Json(serde_json::json!({
            "layers": set.layers,
            "writable": state.writable,
        })),
    ))
}

/// `GET /api/v1/datasets/{id}/interpretations/{user}/raw`
///
/// The stored gprinterp document, byte for byte, as a download. Distinct
/// from the plain `GET` above, which the viewer uses and which is not a
/// file: this one is what leaves Ridal for another tool, so it keeps
/// whatever the file actually contains -- including fields this version
/// does not model -- rather than a re-serialisation of what was parsed.
pub async fn get_interpretation_raw(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, user)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let project = readable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let user = parse_user(&user)?;

    let relative = std::path::Path::new(crate::project::INTERPRETATIONS_DIR)
        .join(radargram.as_str())
        .join(format!("{}{}", user.as_str(), interpretations::SUFFIX));
    let stored = project
        .documents()
        .read(&relative)
        .map_err(|e| store_error(&e))?
        .ok_or_else(|| {
            ApiError::not_found(
                "interpretation_not_found",
                format!(
                    "'{}' has no saved interpretation of '{}'.",
                    user.as_str(),
                    radargram.as_str()
                ),
            )
        })?;

    let filename = format!(
        "{}-{}{}",
        radargram.as_str(),
        user.as_str(),
        interpretations::SUFFIX
    );
    Ok((
        [
            (header::CONTENT_TYPE, "application/json".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
            (header::ETAG, format!("\"{}\"", stored.version)),
        ],
        stored.text,
    ))
}

/// `GET /api/v1/datasets/{id}/interpretations/{user}/level2` -- the derived
/// point product, as a download.
///
/// Derived from the *stored* interpretation rather than from anything the
/// browser holds, so what is downloaded is exactly what was saved. The
/// picker hides the link while there are unsaved picks for the same reason.
#[derive(serde::Deserialize)]
pub struct Level2Query {
    /// "auto", "per-trace", "vertices", or a distance in metres.
    #[serde(default)]
    spacing: Option<String>,
    /// "geojson" (default) or "csv".
    #[serde(default)]
    format: Option<String>,
    /// Coordinates for the GeoJSON geometry: absent for WGS84, "native" for
    /// the radargram's own projected CRS, or any CRS string PROJ accepts.
    /// Ignored for CSV, which carries both regardless.
    #[serde(default)]
    crs: Option<String>,
}

pub async fn interpretation_level2(
    State(state): State<Arc<AppState>>,
    Path((radargram_id, user)): Path<(String, String)>,
    axum::extract::Query(query): axum::extract::Query<Level2Query>,
) -> Result<impl IntoResponse, ApiError> {
    let project = readable_project(&state)?;
    let radargram = parse_radargram(&radargram_id)?;
    let user = parse_user(&user)?;

    let entry = lookup_dataset(&state, radargram.as_str())?;
    let path = state
        .absolute_path(entry)
        .map_err(|e| ApiError::internal("path_resolve_failed", e))?;

    let stored = interpretations::read(project.documents(), &radargram, &user)
        .map_err(interpretation_error)?
        .ok_or_else(|| {
            ApiError::not_found(
                "interpretation_not_found",
                format!(
                    "'{}' has no saved interpretation of '{}'. Save some picks first.",
                    user.as_str(),
                    radargram.as_str()
                ),
            )
        })?;

    let spacing = crate::cli::parse_spacing(query.spacing.as_deref().unwrap_or("auto"))
        .map_err(|e| ApiError::bad_request("invalid_spacing", e))?;

    let geometry = crate::interp::source::read_geometry(&path)
        .map_err(|e| ApiError::internal("radargram_read_failed", e))?;

    let (layer_set, _) = layers::read(project.documents()).map_err(layer_error)?;
    let allows = |label: Option<&str>| layer_set.allows_overhangs(label);

    let export =
        crate::interp::level2::export(&stored.document, &geometry, spacing, user.as_str(), &allows)
            .map_err(|e| ApiError::bad_request("level2_failed", e.to_string()))?;

    let exports = [export];
    let csv = matches!(query.format.as_deref(), Some("csv"));
    let (body, content_type, extension) = if csv {
        (
            crate::interp::writer::to_csv(&exports),
            "text/csv; charset=utf-8",
            "csv",
        )
    } else {
        let output_crs = match query.crs.as_deref() {
            None | Some("") => crate::interp::writer::OutputCrs::Wgs84,
            Some(name) => crate::interp::writer::OutputCrs::Named(name.to_string()),
        };
        (
            crate::interp::writer::to_geojson(&exports, &output_crs)
                // A CRS the projection tools cannot resolve is the caller's
                // mistake, not a server fault.
                .map_err(|e| ApiError::bad_request("invalid_crs", e))?,
            "application/geo+json",
            "geojson",
        )
    };

    // Both components are validated slugs, so the filename cannot carry a
    // quote, a newline, or a path separator into the header.
    let filename = format!(
        "{}-{}-level2.{extension}",
        radargram.as_str(),
        user.as_str()
    );
    Ok((
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body,
    ))
}

/// `GET /api/v1/project/settings`
///
/// Answers for a non-project catalog too, with `project: false` and no
/// values, so the page can explain itself rather than 404.
pub async fn get_settings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let profiles: Vec<String> = crate::server::render::profile::RenderProfile::built_in_profiles()
        .into_iter()
        .map(|p| p.name)
        .collect();
    let project = state.project.as_ref();
    Json(serde_json::json!({
        "project": project.is_some(),
        "writable": state.writable,
        "name": project.and_then(|p| p.config().project.name.clone()),
        "root": project.map(|p| p.root().display().to_string()),
        "default_profile": project.and_then(|p| p.default_profile()),
        "profiles": profiles,
        "default_xscale": project.and_then(|p| p.default_xscale()),
        "xscales": crate::server::routes::x_scale_options(),
    }))
}

#[derive(serde::Deserialize)]
pub struct SettingsUpdate {
    /// The profile to use when a request names none. `null` (or absent)
    /// clears it, restoring the built-in default.
    #[serde(default)]
    default_profile: Option<String>,
    /// Horizontal stretch to open radargrams at. `null` (or absent) clears
    /// it, restoring 1x.
    #[serde(default)]
    default_xscale: Option<f64>,
}

/// `PUT /api/v1/project/settings`
pub async fn put_settings(
    State(state): State<Arc<AppState>>,
    Json(update): Json<SettingsUpdate>,
) -> Result<impl IntoResponse, ApiError> {
    let project = writable_project(&state)?;

    // Validated here rather than in `Project`: which profiles exist is a
    // server concept, and a CLI-only build has no way to check it. Storing
    // a name nothing renders would leave every page 400ing with no obvious
    // cause.
    let profile = match update.default_profile.as_deref() {
        None | Some("") => None,
        Some(name) => {
            if crate::server::render::profile::RenderProfile::by_name(name).is_none() {
                return Err(ApiError::bad_request(
                    "unknown_profile",
                    format!("There is no render profile called '{name}'."),
                ));
            }
            Some(name)
        }
    };

    // Same reasoning as the profile above: which factors the viewer offers
    // is a server concept. Storing one it does not offer would open every
    // radargram at a stretch with no dropdown entry to change it back.
    let xscale = match update.default_xscale {
        None => None,
        Some(scale) => {
            if !crate::server::routes::is_offered_x_scale(scale) {
                return Err(ApiError::bad_request(
                    "unknown_xscale",
                    format!("The viewer does not offer a horizontal scale of {scale}."),
                ));
            }
            // 1x is the neutral value, so choosing it means "no preference"
            // and leaves the key out of the file entirely.
            if (scale - crate::server::routes::DEFAULT_X_SCALE).abs() < 1e-9 {
                None
            } else {
                Some(scale)
            }
        }
    };

    project
        .set_render_defaults(&crate::project::RenderDefaults {
            profile: profile.map(str::to_string),
            xscale,
        })
        .map_err(|e| ApiError::internal("settings_write_failed", e.to_string()))?;

    Ok(Json(serde_json::json!({
        "default_profile": project.default_profile(),
        "default_xscale": project.default_xscale(),
    })))
}

/// `GET /api/v1/groups/{group}/level2` -- every interpreted radargram in a
/// group, merged into one file.
///
/// Concatenation, not reconciliation: each point already names its own
/// radargram and revision, so the merge adds nothing and loses nothing.
///
/// Radargrams nobody has interpreted are skipped rather than erroring --
/// a group is a survey, and part of one being unpicked is its normal
/// state. If *none* are, that is worth saying rather than handing back an
/// empty file.
pub async fn group_level2(
    State(state): State<Arc<AppState>>,
    Path(group): Path<String>,
    axum::extract::Query(query): axum::extract::Query<Level2Query>,
) -> Result<impl IntoResponse, ApiError> {
    merged_level2(&state, &DownloadScope::Group(group), query)
}

/// `GET /api/v1/catalog/level2` -- every interpreted radargram the server
/// knows about, merged. The catalog-wide half of [`merged_level2`].
pub async fn catalog_level2(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<Level2Query>,
) -> Result<impl IntoResponse, ApiError> {
    merged_level2(&state, &DownloadScope::Catalog, query)
}

fn merged_level2(
    state: &AppState,
    scope: &DownloadScope,
    query: Level2Query,
) -> Result<(HeaderMap, String), ApiError> {
    let project = readable_project(state)?;
    let entries = scope.entries(state);
    if entries.is_empty() {
        return Err(ApiError::not_found(
            scope.empty_code(),
            format!("Nothing to download in {}.", scope.describe()),
        ));
    }

    // Whose picks a merged download contains. Once several people can
    // interpret one radargram this becomes a choice rather than a lookup --
    // "mine", "everyone's", or a named user -- but it resolves through the
    // same function either way.
    let user = current_user(state)?;
    let spacing = crate::cli::parse_spacing(query.spacing.as_deref().unwrap_or("auto"))
        .map_err(|e| ApiError::bad_request("invalid_spacing", e))?;
    let (layer_set, _) = layers::read(project.documents()).map_err(layer_error)?;
    let allows = |label: Option<&str>| layer_set.allows_overhangs(label);

    let mut exports = Vec::new();
    let mut skipped = Vec::new();
    for entry in &entries {
        let radargram = &entry.radargram_id;
        let Some(stored) = interpretations::read(project.documents(), radargram, &user)
            .map_err(interpretation_error)?
        else {
            skipped.push(radargram.to_string());
            continue;
        };
        let path = state
            .absolute_path(entry)
            .map_err(|e| ApiError::internal("path_resolve_failed", e))?;
        let geometry = crate::interp::source::read_geometry(&path)
            .map_err(|e| ApiError::internal("radargram_read_failed", e))?;
        let export = crate::interp::level2::export(
            &stored.document,
            &geometry,
            spacing,
            user.as_str(),
            &allows,
        )
        // Named, because in a group export "which one failed?" is the
        // first thing anyone would ask.
        .map_err(|e| {
            ApiError::bad_request("level2_failed", format!("{}: {e}", radargram.as_str()))
        })?;
        exports.push(export);
    }

    if exports.is_empty() {
        return Err(ApiError::not_found(
            "interpretation_not_found",
            format!(
                "Nothing in {} has been interpreted yet ({} radargram(s) checked).",
                scope.describe(),
                entries.len()
            ),
        ));
    }

    let csv = matches!(query.format.as_deref(), Some("csv"));
    let (body, content_type, extension) = if csv {
        (
            crate::interp::writer::to_csv(&exports),
            "text/csv; charset=utf-8",
            "csv",
        )
    } else {
        let output_crs = match query.crs.as_deref() {
            None | Some("") => crate::interp::writer::OutputCrs::Wgs84,
            Some(name) => crate::interp::writer::OutputCrs::Named(name.to_string()),
        };
        (
            crate::interp::writer::to_geojson(&exports, &output_crs)
                .map_err(|e| ApiError::bad_request("invalid_crs", e))?,
            "application/geo+json",
            "geojson",
        )
    };

    let filename = format!("{}-level2.{extension}", scope.slug());
    let mut headers = HeaderMap::new();
    let set = |headers: &mut HeaderMap, name: header::HeaderName, value: String| {
        // A header value that will not parse is dropped rather than turned
        // into a failed download; every value here is built from validated
        // slugs, so this is belt and braces.
        if let Ok(value) = value.parse() {
            headers.insert(name, value);
        }
    };
    set(&mut headers, header::CONTENT_TYPE, content_type.to_string());
    set(
        &mut headers,
        header::CONTENT_DISPOSITION,
        format!("attachment; filename=\"{filename}\""),
    );
    if !skipped.is_empty() {
        // A header rather than silence: a merged file that quietly omits
        // half a survey looks complete.
        set(
            &mut headers,
            header::WARNING,
            format!(
                "199 ridal \"not yet interpreted, omitted: {}\"",
                skipped.join(" ")
            ),
        );
    }
    Ok((headers, body))
}

/// `GET /api/v1/layers/usage` -- how many picked features use each layer.
///
/// Exists so the management page can say what deleting a layer would
/// orphan, before it is deleted rather than after. Also reports labels in
/// use that the vocabulary does not define, which is the same information
/// `ridal project info` prints.
///
/// Scans every stored interpretation. That is a directory walk plus a JSON
/// parse per document, which is fine at the scale this serves (a project has
/// tens to low hundreds of radargrams, and each document is small) and is
/// only requested when someone opens the layers page.
pub async fn layer_usage(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    let project = readable_project(&state)?;
    let (set, _) = layers::read(project.documents()).map_err(layer_error)?;

    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut undefined: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    for entry in &state.catalog.entries {
        let radargram = &entry.radargram_id;
        let users = interpretations::list_users(project.documents(), radargram)
            .map_err(interpretation_error)?;
        for user in users {
            let Ok(user_id) = UserId::new(user.as_str()) else {
                continue;
            };
            let Some(stored) = interpretations::read(project.documents(), radargram, &user_id)
                .map_err(interpretation_error)?
            else {
                continue;
            };
            for feature in &stored.document.features {
                let Some(label) = feature.label() else {
                    continue;
                };
                let target = if set.get(label).is_some() {
                    &mut counts
                } else {
                    &mut undefined
                };
                *target.entry(label.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Defined layers appear with a zero count rather than being omitted, so
    // the page can say "used by 0 features" instead of showing nothing.
    for layer in &set.layers {
        counts.entry(layer.id.clone()).or_insert(0);
    }

    Ok(Json(serde_json::json!({
        "counts": counts,
        "undefined": undefined,
    })))
}

/// `PUT /api/v1/layers` -- replace the vocabulary.
pub async fn put_layers(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    let project = writable_project(&state)?;

    // Accept either a bare array of layers or the full document. The GUI
    // only ever has the list; requiring it to reconstruct the envelope would
    // be ceremony with no benefit.
    let set: layers::LayerSet = if body.is_array() {
        let (existing, _) = layers::read(project.documents()).map_err(layer_error)?;
        let parsed: Vec<layers::Layer> = serde_json::from_value(body)
            .map_err(|e| ApiError::bad_request("invalid_layers", e.to_string()))?;
        layers::LayerSet {
            layers: parsed,
            ..existing
        }
    } else {
        serde_json::from_value(body)
            .map_err(|e| ApiError::bad_request("invalid_layers", e.to_string()))?
    };

    let version = layers::write(project.documents(), &set, &expectation_from(&headers))
        .map_err(layer_error)?;

    Ok((
        StatusCode::OK,
        [(header::ETAG, format!("\"{version}\""))],
        Json(serde_json::json!({
            "layers": set.layers,
            "version": version.as_str(),
        })),
    ))
}
