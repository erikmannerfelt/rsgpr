//! HTTP-level tests for the interpretation and layer write routes.
//!
//! Driven through the real Axum router with `ServiceExt::oneshot`, so the
//! status codes, `ETag`/`If-Match` handling and error envelopes are the ones
//! a browser will actually see -- not just the store behaviour underneath,
//! which `project::store` already covers.
//!
//! Every test here builds an `AppState`, which creates and opens a NetCDF.
//! netcdf-c is not thread-safe, so they carry the same
//! `#[serial_test::serial(netcdf)]` guard as the tests in `app.rs`; without
//! it they pass alone and flake in a full run.

use std::path::Path as StdPath;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use serde_json::Value;
use tower::ServiceExt;

use super::app::{build_router, AppState};
use super::render::service::RenderServiceConfig;
use crate::project::Project;

const RADARGRAM: &str = "line-01";

fn write_test_nc(path: &StdPath, radargram_id: &str) {
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("y", 8).unwrap();
    file.add_dimension("x", 40).unwrap();
    let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
    var.put_values(&vec![1.0f32; 8 * 40], ..).unwrap();
    file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
        .unwrap();
    file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
        .unwrap();
    file.add_attribute("ridal_radargram_id", radargram_id)
        .unwrap();
}

/// A project containing one radargram, served writable unless stated.
fn project_app(writable: bool) -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc(&dir.path().join("radargrams").join("line-01.nc"), RADARGRAM);
    let project = Project::discover(dir.path()).unwrap().unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            writable,
        )
        .unwrap(),
    );
    (dir, build_router(state))
}

/// A bare directory of radargrams: the pre-existing read-only arrangement.
fn bare_app() -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    write_test_nc(&dir.path().join("line-01.nc"), RADARGRAM);
    let state = Arc::new(
        AppState::build_with_project(dir.path(), &RenderServiceConfig::default(), None, true)
            .unwrap(),
    );
    (dir, build_router(state))
}

fn document(key: &str) -> Value {
    serde_json::json!({
        "key": key,
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[5.0, 2.0], [30.0, 3.0]]},
            "properties": {"id": "f-0001", "label": "bed"}
        }]
    })
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Option<String>, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let etag = response
        .headers()
        .get(header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, etag, body)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Option<String>, Value) {
    send(
        app,
        Request::builder().uri(uri).body(Body::empty()).unwrap(),
    )
    .await
}

async fn put(
    app: &Router,
    uri: &str,
    body: &Value,
    if_match: Option<&str>,
) -> (StatusCode, Option<String>, Value) {
    let mut builder = Request::builder()
        .method("PUT")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(value) = if_match {
        builder = builder.header(header::IF_MATCH, value);
    }
    send(app, builder.body(Body::from(body.to_string())).unwrap()).await
}

const URI: &str = "/api/v1/datasets/line-01/interpretations/default";

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_interpretation_is_created_then_updated() {
    let (_dir, app) = project_app(true);

    let (status, etag, _) = put(&app, URI, &document(RADARGRAM), None).await;
    assert_eq!(status, StatusCode::CREATED);
    let etag = etag.expect("a write must return an ETag for the next If-Match");

    let (status, get_etag, body) = get(&app, URI).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(get_etag.as_deref(), Some(etag.as_str()));
    assert_eq!(body["key"], RADARGRAM);
    assert_eq!(body["features"][0]["properties"]["label"], "bed");

    // Updating an existing document is 200, not 201.
    let (status, _, _) = put(&app, URI, &document(RADARGRAM), Some(&etag)).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_stale_if_match_is_refused_with_412() {
    // The two-tab case. The second save must not silently win.
    let (_dir, app) = project_app(true);
    let (_, first, _) = put(&app, URI, &document(RADARGRAM), None).await;
    let first = first.unwrap();

    let mut changed = document(RADARGRAM);
    changed["features"] = serde_json::json!([]);
    put(&app, URI, &changed, Some(&first)).await;

    let (status, _, body) = put(&app, URI, &document(RADARGRAM), Some(&first)).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
    assert_eq!(body["error"]["code"], "version_conflict");

    // The intervening edit survived.
    let (_, _, stored) = get(&app, URI).await;
    assert_eq!(stored["features"].as_array().unwrap().len(), 0);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn if_none_match_star_refuses_to_clobber() {
    let (_dir, app) = project_app(true);
    put(&app, URI, &document(RADARGRAM), None).await;

    let request = Request::builder()
        .method("PUT")
        .uri(URI)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::IF_NONE_MATCH, "*")
        .body(Body::from(document(RADARGRAM).to_string()))
        .unwrap();
    let (status, _, _) = send(&app, request).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_document_naming_another_radargram_is_rejected() {
    let (_dir, app) = project_app(true);
    let (status, _, body) = put(&app, URI, &document("some-other-line"), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "key_mismatch");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn writing_to_an_unknown_radargram_is_404_not_a_stray_directory() {
    let (dir, app) = project_app(true);
    let (status, _, _) = put(
        &app,
        "/api/v1/datasets/not-here/interpretations/default",
        &document("not-here"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        !dir.path().join("interpretations/not-here").exists(),
        "a typo in the URL must not create an interpretation directory"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_user_name_cannot_escape_the_interpretations_directory() {
    let (_dir, app) = project_app(true);
    // Axum's path matching already rejects most of these; the point is that
    // nothing reaches the filesystem regardless of how it is encoded.
    for user in ["..", "%2e%2e", "not_a_slug!"] {
        let uri = format!("/api/v1/datasets/line-01/interpretations/{user}");
        let (status, _, _) = put(&app, &uri, &document(RADARGRAM), None).await;
        assert!(
            status.is_client_error(),
            "user '{user}' produced {status}, expected a client error"
        );
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn deleting_removes_the_document_and_then_404s() {
    let (_dir, app) = project_app(true);
    put(&app, URI, &document(RADARGRAM), None).await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(URI)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let (status, _, _) = get(&app, URI).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn listing_reports_users_and_whether_writes_are_possible() {
    let (_dir, app) = project_app(true);
    put(&app, URI, &document(RADARGRAM), None).await;

    let (status, _, body) = get(&app, "/api/v1/datasets/line-01/interpretations").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["users"], serde_json::json!(["default"]));
    assert_eq!(body["writable"], true);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_read_only_server_refuses_writes_but_still_serves_reads() {
    let (_dir, writable) = project_app(true);
    put(&writable, URI, &document(RADARGRAM), None).await;

    let (_dir2, app) = project_app(false);
    let (status, _, body) = put(&app, URI, &document(RADARGRAM), None).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "read_only");

    let (status, _, body) = get(&app, "/api/v1/datasets/line-01/interpretations").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["writable"], false);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_catalog_that_is_not_a_project_explains_itself() {
    // The pre-existing read-only arrangement must keep working, and must say
    // why saving is unavailable rather than failing obscurely.
    let (_dir, app) = bare_app();

    let (status, _, body) = put(&app, URI, &document(RADARGRAM), None).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "not_a_project");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("ridal project init"),
        "the error should say how to fix it: {body}"
    );

    // Layers still answer, so the viewer's page load does not fail.
    let (status, _, body) = get(&app, "/api/v1/layers").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["layers"], serde_json::json!([]));
    assert_eq!(body["writable"], false);
}

fn overhanging_document() -> Value {
    serde_json::json!({
        "key": RADARGRAM,
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[5.0, 2.0], [30.0, 3.0], [20.0, 5.0]]},
            "properties": {"id": "f-0001", "label": "bed"}
        }]
    })
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_overhanging_line_is_refused_by_default() {
    // The guardrail is on for a layer nobody has opted out of -- including
    // one the vocabulary does not define at all.
    let (_dir, app) = project_app(true);
    let (status, _, body) = put(&app, URI, &overhanging_document(), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "overhang");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("f-0001"), "{message}");
    assert!(message.contains("bed"), "{message}");

    let (status, _, _) = get(&app, URI).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "a refused save must not have stored anything"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_layer_that_allows_overhangs_accepts_one() {
    let (_dir, app) = project_app(true);
    let layers = serde_json::json!([
        {"id": "bed", "name": "Bed", "allow_overhangs": true}
    ]);
    let (status, _, _) = put(&app, "/api/v1/layers", &layers, None).await;
    assert_eq!(status, StatusCode::OK);

    let (status, _, _) = put(&app, URI, &overhanging_document(), None).await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn turning_the_guardrail_off_and_on_again_changes_what_is_accepted() {
    // The setting is read at save time, not baked in at startup, so a
    // project can tighten the rule without a restart.
    let (_dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/layers",
        &serde_json::json!([{"id": "bed", "name": "Bed", "allow_overhangs": true}]),
        None,
    )
    .await;
    assert_eq!(
        put(&app, URI, &overhanging_document(), None).await.0,
        StatusCode::CREATED
    );

    put(
        &app,
        "/api/v1/layers",
        &serde_json::json!([{"id": "bed", "name": "Bed"}]),
        None,
    )
    .await;
    assert_eq!(
        put(&app, URI, &overhanging_document(), None).await.0,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_overhang_flag_round_trips_and_defaults_to_off() {
    let (_dir, app) = project_app(true);
    let layers = serde_json::json!([
        {"id": "bed", "name": "Bed"},
        {"id": "crevasse", "name": "Crevasse", "allow_overhangs": true}
    ]);
    put(&app, "/api/v1/layers", &layers, None).await;

    let (_, _, body) = get(&app, "/api/v1/layers").await;
    // Absent rather than `false`: the default is not written out.
    assert!(body["layers"][0].get("allow_overhangs").is_none());
    assert_eq!(body["layers"][1]["allow_overhangs"], true);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn layers_round_trip_through_the_api() {
    let (_dir, app) = project_app(true);

    let (status, _, body) = get(&app, "/api/v1/layers").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["layers"], serde_json::json!([]));

    // A bare array is accepted: the GUI only ever has the list.
    let layers = serde_json::json!([
        {"id": "bed", "name": "Bed", "color": "#e6194b"},
        {"id": "internal", "name": "Internal reflector", "color": "#3cb44b"}
    ]);
    let (status, etag, _) = put(&app, "/api/v1/layers", &layers, None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(etag.is_some());

    let (_, _, body) = get(&app, "/api/v1/layers").await;
    assert_eq!(body["layers"][0]["id"], "bed");
    assert_eq!(body["layers"][1]["name"], "Internal reflector");
    assert_eq!(body["writable"], true);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_layer_id_that_would_not_survive_export_is_rejected() {
    let (_dir, app) = project_app(true);
    let layers = serde_json::json!([{"id": "Bed Layer", "name": "Bed"}]);
    let (status, _, body) = put(&app, "/api/v1/layers", &layers, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_layers");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn duplicate_layer_ids_are_rejected() {
    let (_dir, app) = project_app(true);
    let layers = serde_json::json!([
        {"id": "bed", "name": "Bed"},
        {"id": "bed", "name": "Bed again"}
    ]);
    let (status, _, _) = put(&app, "/api/v1/layers", &layers, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn layer_usage_counts_features_and_flags_undefined_labels() {
    let (_dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/layers",
        &serde_json::json!([{"id": "bed", "name": "Bed"}]),
        None,
    )
    .await;

    let document = serde_json::json!({
        "key": RADARGRAM,
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[1.0, 1.0], [10.0, 2.0]]},
             "properties": {"id": "f-1", "label": "bed"}},
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[12.0, 1.0], [20.0, 2.0]]},
             "properties": {"id": "f-2", "label": "bed"}},
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[22.0, 1.0], [30.0, 2.0]]},
             "properties": {"id": "f-3", "label": "englacial"}}
        ]
    });
    put(&app, URI, &document, None).await;

    let (status, _, body) = get(&app, "/api/v1/layers/usage").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["counts"]["bed"], 2);
    // A label nobody defined is reported separately, not silently counted.
    assert_eq!(body["undefined"]["englacial"], 1);
    assert!(body["counts"].get("englacial").is_none());
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_defined_layer_nobody_uses_reports_zero_rather_than_vanishing() {
    // So the page can say "0 features" before a delete, instead of nothing.
    let (_dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/layers",
        &serde_json::json!([{"id": "unused", "name": "Unused"}]),
        None,
    )
    .await;

    let (_, _, body) = get(&app, "/api/v1/layers/usage").await;
    assert_eq!(body["counts"]["unused"], 0);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_layers_page_renders_for_a_project_and_for_a_bare_catalog() {
    let (_dir, app) = project_app(true);
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/layers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let html = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(html.contains("writable: true"), "{html}");
    assert!(html.contains("Add a layer"));

    // A catalog with no project still answers, explaining why rather than
    // 404ing on "show me the layers".
    let (_dir2, bare) = bare_app();
    let response = bare
        .oneshot(
            Request::builder()
                .uri("/layers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let html = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(html.contains("ridal project init"), "{html}");
    assert!(
        !html.contains("Add a layer"),
        "read-only page must not offer a form"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn malformed_json_is_a_client_error_with_the_standard_envelope() {
    let (_dir, app) = project_app(true);
    let (status, _, body) = put(&app, URI, &serde_json::json!({"no_key": true}), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_interpretation");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn validation_warnings_are_reported_without_refusing_the_save() {
    // The format is permissive by design: a feature with no stable id is
    // worth mentioning but must still save.
    let (_dir, app) = project_app(true);
    let document = serde_json::json!({
        "key": RADARGRAM,
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[5.0, 2.0], [30.0, 3.0]]},
            "properties": {"label": "bed"}
        }]
    });
    let (status, _, body) = put(&app, URI, &document, None).await;
    assert_eq!(status, StatusCode::CREATED);
    let warnings = body["warnings"].as_array().unwrap();
    assert!(
        warnings.iter().any(|w| w.as_str().unwrap().contains("id")),
        "{warnings:?}"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn unknown_fields_survive_a_save_and_reload_over_http() {
    let (_dir, app) = project_app(true);
    let mut document = document(RADARGRAM);
    document["from_a_future_version"] = serde_json::json!({"nested": [1, 2]});

    put(&app, URI, &document, None).await;
    let (_, _, body) = get(&app, URI).await;
    assert_eq!(
        body["from_a_future_version"],
        serde_json::json!({"nested": [1, 2]})
    );
}
