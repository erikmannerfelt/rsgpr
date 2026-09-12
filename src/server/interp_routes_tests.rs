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
    // Varying, not constant: the renderer refuses an array whose amplitude
    // percentiles collapse to a single value, because there is no contrast
    // to stretch. A flat fixture is not a realistic radargram anyway.
    let data: Vec<f32> = (0..(8 * 40)).map(|i| (i % 97) as f32).collect();
    var.put_values(&data, ..).unwrap();
    file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
        .unwrap();
    file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
        .unwrap();
    file.add_attribute("ridal_radargram_id", radargram_id)
        .unwrap();
}

/// A radargram with the coordinate variables a level 2 export needs.
///
/// `write_test_nc` deliberately writes the bare minimum the catalog
/// recognises; deriving level 2 additionally needs distance, travel time,
/// depth and positions, so those are written here rather than bloating the
/// fixture every other test uses.
fn write_test_nc_with_axes(path: &StdPath, radargram_id: &str, group: Option<&str>) {
    let (n_samples, n_traces) = (8usize, 40usize);
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("y", n_samples).unwrap();
    file.add_dimension("x", n_traces).unwrap();
    let mut data = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
    let values: Vec<f32> = (0..(n_samples * n_traces))
        .map(|i| (i % 97) as f32)
        .collect();
    data.put_values(&values, ..).unwrap();

    // 1 m between traces, running due east, so expected values are obvious.
    let mut put = |name: &str, values: Vec<f64>| {
        let mut var = file.add_variable::<f64>(name, &["x"]).unwrap();
        var.put_values(&values, ..).unwrap();
    };
    put("distance", (0..n_traces).map(|i| i as f64).collect());
    put(
        "easting",
        (0..n_traces).map(|i| 400_000.0 + i as f64).collect(),
    );
    put("northing", vec![8_700_000.0; n_traces]);
    put(
        "longitude",
        (0..n_traces).map(|i| 15.0 + i as f64 * 1e-5).collect(),
    );
    put("latitude", vec![78.0; n_traces]);
    // Track reading needs per-trace acquisition time as well as position:
    // it uses the time gaps to decide where a profile breaks.
    put(
        "time",
        (0..n_traces).map(|i| 1_677_501_559.0 + i as f64).collect(),
    );

    let mut twtt = file.add_variable::<f64>("twtt", &["y"]).unwrap();
    twtt.put_values(
        &(0..n_samples).map(|i| i as f64 * 0.4).collect::<Vec<f64>>(),
        ..,
    )
    .unwrap();
    let mut depth = file.add_variable::<f64>("depth", &["y"]).unwrap();
    depth
        .put_values(
            &(0..n_samples)
                .map(|i| i as f64 * 0.04)
                .collect::<Vec<f64>>(),
            ..,
        )
        .unwrap();

    file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
        .unwrap();
    file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
        .unwrap();
    file.add_attribute("ridal_radargram_id", radargram_id)
        .unwrap();
    file.add_attribute("crs", "EPSG:32633").unwrap();
    // Written while the file is being created. Both attributes are needed:
    // `resolve_group` treats a bare id as no group at all, since the id only
    // exists to give the name a URL-safe form.
    if let Some(group) = group {
        file.add_attribute("ridal_group_name", group).unwrap();
        file.add_attribute("ridal_group_id", group).unwrap();
    }
}

/// A writable project with two radargrams in one group, both carrying
/// coordinate axes -- the shape a merged download is about.
fn group_app() -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    let radargrams = dir.path().join("radargrams");
    for id in ["line-01", "line-02"] {
        write_test_nc_with_axes(&radargrams.join(format!("{id}.nc")), id, Some("survey"));
    }
    let project = Project::discover(dir.path()).unwrap().unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            true,
        )
        .unwrap(),
    );
    (dir, build_router(state))
}

/// A catalog holding both a group and a radargram belonging to no group.
///
/// The catalog scope has to cover both, which a fixture where everything is
/// grouped could not tell apart from "every group, merged".
fn mixed_catalog_app() -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    let radargrams = dir.path().join("radargrams");
    for id in ["line-01", "line-02"] {
        write_test_nc_with_axes(&radargrams.join(format!("{id}.nc")), id, Some("survey"));
    }
    write_test_nc_with_axes(&radargrams.join("loose-01.nc"), "loose-01", None);
    let project = Project::discover(dir.path()).unwrap().unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            true,
        )
        .unwrap(),
    );
    (dir, build_router(state))
}

/// A writable project whose radargram carries full coordinate axes.
fn project_app_with_axes() -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc_with_axes(
        &dir.path().join("radargrams").join("line-01.nc"),
        RADARGRAM,
        None,
    );
    let project = Project::discover(dir.path()).unwrap().unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            true,
        )
        .unwrap(),
    );
    (dir, build_router(state))
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

async fn page(app: &Router, uri: &str) -> (StatusCode, String) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_viewer_offers_picking_only_where_it_can_be_saved() {
    let (_dir, app) = project_app(true);
    let (status, html) = page(&app, "/view/line-01").await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("writable: true"), "config flag missing");
    assert!(html.contains(r#"id="pick-toggle""#), "no picking control");
    assert!(html.contains(r#"id="pick-save""#), "no save control");
    assert!(
        html.contains(r#"id="pick-selection""#),
        "no selection panel"
    );
    assert!(
        html.contains("/static/picker.js"),
        "picker script not loaded"
    );
    assert!(html.contains(r#"user: "default""#), "no author for saves");

    // Read-only: the toolbar says why rather than vanishing, so a missing
    // control never reads as a missing feature.
    let (_dir2, read_only) = project_app(false);
    let (_, html) = page(&read_only, "/view/line-01").await;
    assert!(html.contains("read-only"), "{html}");
    assert!(!html.contains(r#"id="pick-toggle""#));

    let (_dir3, bare) = bare_app();
    let (_, html) = page(&bare, "/view/line-01").await;
    assert!(html.contains("ridal project init"), "{html}");
    assert!(!html.contains(r#"id="pick-toggle""#));
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_level2_download_derives_from_the_saved_interpretation() {
    let (_dir, app) = project_app_with_axes();
    let (status, _, _) = put(&app, URI, &document(RADARGRAM), None).await;
    assert_eq!(status, StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("{URI}/level2?spacing=vertices&format=csv"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let disposition = response
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        disposition.contains("line-01-default-level2.csv"),
        "{disposition}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let csv = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        csv.starts_with("radargram_id,revision_id,layer,line_index,point_index,"),
        "{csv}"
    );
    assert!(csv.contains("line-01,"), "{csv}");
    assert!(csv.contains(",bed,0,0,f-0001,"), "{csv}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn downloading_before_anything_is_saved_says_so() {
    let (_dir, app) = project_app_with_axes();
    let (status, _, body) = get(&app, &format!("{URI}/level2")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Save some picks first"),
        "{body}"
    );
}

async fn raw(app: &Router, uri: &str) -> (StatusCode, Option<String>, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let disposition = response
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, disposition, bytes.to_vec())
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_group_track_merges_every_member_and_names_each() {
    let (_dir, app) = group_app();
    let (status, disposition, bytes) = raw(&app, "/api/v1/groups/survey/track.geojson").await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition.unwrap().contains("survey-tracks.geojson"));

    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = body["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["properties"]["radargram_id"].as_str().unwrap())
        .collect();
    // The merge is reversible: every feature says where it came from.
    assert!(ids.contains(&"line-01"), "{ids:?}");
    assert!(ids.contains(&"line-02"), "{ids:?}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_group_level2_merges_points_that_name_their_radargram() {
    let (_dir, app) = group_app();
    for id in ["line-01", "line-02"] {
        let uri = format!("/api/v1/datasets/{id}/interpretations/default");
        let (status, _, _) = put(&app, &uri, &document(id), None).await;
        assert_eq!(status, StatusCode::CREATED, "{id}");
    }

    let (status, disposition, bytes) = raw(
        &app,
        "/api/v1/groups/survey/level2?spacing=vertices&format=csv",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition.unwrap().contains("survey-level2.csv"));

    let csv = String::from_utf8(bytes).unwrap();
    let mut lines = csv.lines();
    // One header for the whole file, radargram first.
    assert!(lines
        .next()
        .unwrap()
        .starts_with("radargram_id,revision_id,layer,"));
    let rows: Vec<&str> = lines.collect();
    assert!(rows.iter().any(|r| r.starts_with("line-01,")), "{csv}");
    assert!(rows.iter().any(|r| r.starts_with("line-02,")), "{csv}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_group_level2_says_which_members_it_left_out() {
    // Half a survey being unpicked is normal; a merged file that quietly
    // omitted it would look complete.
    let (_dir, app) = group_app();
    let (status, _, _) = put(
        &app,
        "/api/v1/datasets/line-01/interpretations/default",
        &document("line-01"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/groups/survey/level2?format=csv")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let warning = response
        .headers()
        .get(header::WARNING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(warning.contains("line-02"), "{warning}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_group_with_nothing_interpreted_says_so() {
    let (_dir, app) = group_app();
    let (status, _, body) = get(&app, "/api/v1/groups/survey/level2").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("has been interpreted yet"),
        "{body}"
    );

    let (status, _, body) = get(&app, "/api/v1/groups/nope/track.geojson").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "group_not_found");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_catalog_track_covers_grouped_and_ungrouped_alike() {
    let (_dir, app) = mixed_catalog_app();
    let (status, disposition, bytes) = raw(&app, "/api/v1/catalog/track.geojson").await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition.unwrap().contains("catalog-tracks.geojson"));

    let body: Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = body["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["properties"]["radargram_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"line-01"), "{ids:?}");
    assert!(ids.contains(&"line-02"), "{ids:?}");
    // The one that belongs to no group is the point of this test: a catalog
    // download that only covered groups would silently drop it.
    assert!(ids.contains(&"loose-01"), "{ids:?}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_catalog_level2_merges_across_group_boundaries() {
    let (_dir, app) = mixed_catalog_app();
    for id in ["line-01", "loose-01"] {
        let uri = format!("/api/v1/datasets/{id}/interpretations/default");
        let (status, _, _) = put(&app, &uri, &document(id), None).await;
        assert_eq!(status, StatusCode::CREATED, "{id}");
    }

    let (status, disposition, bytes) =
        raw(&app, "/api/v1/catalog/level2?spacing=vertices&format=csv").await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition.unwrap().contains("catalog-level2.csv"));

    let csv = String::from_utf8(bytes).unwrap();
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert!(rows.iter().any(|r| r.starts_with("line-01,")), "{csv}");
    assert!(rows.iter().any(|r| r.starts_with("loose-01,")), "{csv}");
    // Same schema as the group and single-radargram products, so the three
    // are concatenable and tell one story.
    assert!(csv.starts_with("radargram_id,revision_id,layer,"), "{csv}");

    // line-02 was never picked, and is named rather than silently missing --
    // the same rule the group scope follows, because it is the same code.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/catalog/level2?spacing=vertices&format=csv")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let warning = response
        .headers()
        .get(header::WARNING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(warning.contains("line-02"), "{warning}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_catalog_with_nothing_interpreted_says_so() {
    let (_dir, app) = mixed_catalog_app();
    let (status, _, body) = get(&app, "/api/v1/catalog/level2").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("this catalog"),
        "{body}"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn single_level2_points_name_their_radargram_too() {
    // The same field, from the same place: it is on the point, not on the
    // file, which is what lets a merged file work at all.
    let (_dir, app) = project_app_with_axes();
    put(&app, URI, &document(RADARGRAM), None).await;
    let (_, _, bytes) = raw(&app, &format!("{URI}/level2?spacing=vertices")).await;
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["features"][0]["properties"]["radargram_id"], RADARGRAM);
    assert_eq!(body["ridal"]["sources"][0]["radargram_id"], RADARGRAM);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_radargram_downloads_byte_for_byte() {
    let (dir, app) = project_app(true);
    let (status, disposition, bytes) = raw(&app, "/api/v1/datasets/line-01/download").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        disposition.as_deref(),
        Some("attachment; filename=\"line-01.nc\"")
    );

    let source = std::fs::read(dir.path().join("radargrams").join("line-01.nc")).unwrap();
    assert_eq!(
        bytes, source,
        "the download must be the file, not a re-write"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_track_downloads_as_geojson_naming_its_radargram() {
    let (_dir, app) = project_app_with_axes();
    let (status, disposition, bytes) = raw(&app, "/api/v1/datasets/line-01/track.geojson").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        disposition.unwrap().contains("line-01-track.geojson"),
        "downloads should be named after their radargram"
    );

    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["type"], "FeatureCollection");
    let feature = &body["features"][0];
    assert_eq!(feature["geometry"]["type"], "LineString");
    // The field Erik asked for, plus enough to tell segments apart.
    assert_eq!(feature["properties"]["radargram_id"], "line-01");
    assert!(feature["properties"]["trace_start"].is_number());
    assert!(feature["properties"]["n_traces"].is_number());
    // WGS84, per RFC 7946: longitude first.
    let first = &feature["geometry"]["coordinates"][0];
    assert!(first[0].as_f64().unwrap().abs() <= 180.0);
    assert!(first[1].as_f64().unwrap().abs() <= 90.0);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn raw_picks_download_exactly_what_is_stored() {
    // Not a re-serialisation: a field this version does not model must
    // survive a round trip out to another tool.
    let (_dir, app) = project_app(true);
    let mut document = document(RADARGRAM);
    document["from_a_future_version"] = serde_json::json!({"keep": "me"});
    put(&app, URI, &document, None).await;

    let (status, disposition, bytes) = raw(&app, &format!("{URI}/raw")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition
        .unwrap()
        .contains("line-01-default.gprinterp.json"));
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["from_a_future_version"]["keep"], "me");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_image_renders_at_the_requested_width() {
    let (_dir, app) = project_app(true);
    // The fixture is 40 traces by 8 samples.
    let (status, disposition, bytes) = raw(
        &app,
        "/api/v1/datasets/line-01/views/standard/image?width=20",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        disposition.unwrap().contains("20x4.png"),
        "size in the name"
    );
    // PNG signature, then width and height from the IHDR chunk.
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    assert_eq!((width, height), (20, 4));
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_image_is_never_upscaled_past_the_source() {
    // A wider image than there are traces carries no more information, and
    // asking for one is more likely a slip than an intent.
    let (_dir, app) = project_app(true);
    let (status, disposition, _) = raw(
        &app,
        "/api/v1/datasets/line-01/views/standard/image?width=5000",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(disposition.unwrap().contains("40x8.png"));
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_image_refuses_sizes_it_cannot_produce() {
    let (_dir, app) = project_app(true);
    let base = "/api/v1/datasets/line-01/views/standard/image";
    for (query, code) in [
        ("?width=0", "invalid_width"),
        ("?width=99999", "invalid_width"),
        ("?format=tiff", "invalid_format"),
    ] {
        let (status, _, body) = get(&app, &format!("{base}{query}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(body["error"]["code"], code, "{query}");
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn dataset_downloads_work_without_a_project() {
    // A radargram, its track and an image belong to the catalog, not to an
    // interpretation, so a bare directory still serves them.
    let (_dir, app) = bare_app();
    for uri in [
        "/api/v1/datasets/line-01/download",
        "/api/v1/datasets/line-01/views/standard/image?width=10",
    ] {
        let (status, _, _) = raw(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_default_profile_round_trips_through_the_settings_api() {
    let (dir, app) = project_app(true);

    let (status, _, body) = get(&app, "/api/v1/project/settings").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["project"], true);
    assert_eq!(body["writable"], true);
    assert!(body["default_profile"].is_null(), "unset to begin with");
    assert!(
        body["profiles"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("abslog")),
        "the page needs the list to populate its select: {body}"
    );

    let (status, _, _) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_profile": "abslog"}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (_, _, body) = get(&app, "/api/v1/project/settings").await;
    assert_eq!(body["default_profile"], "abslog");

    // And on disk, so it survives a restart.
    let marker = std::fs::read_to_string(dir.path().join("ridal.toml")).unwrap();
    assert!(marker.contains("default_profile = \"abslog\""), "{marker}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_export_against_the_wrong_radargram_is_refused() {
    // The CLI refused this and the HTTP route did not, so the same inputs
    // gave an error on one path and a plausible, wrong file on the other.
    // Needs the fixture with real axes: without a CRS, `read_geometry`
    // fails first and the identity check is never reached.
    let (_dir, app) = project_app_with_axes();

    // Written straight to disk, because the API will not store a mismatch:
    // `interpretations::write` already refuses a document whose key names a
    // different radargram. So the only way in is a hand-edited or moved
    // file -- which is exactly the case the export path has to survive.
    let mut doc = document("line-01");
    doc["key"] = serde_json::json!("some-other-line");
    let dir = _dir.path().join("interpretations/line-01");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("default.gprinterp.json"),
        serde_json::to_string(&doc).unwrap(),
    )
    .unwrap();

    let (status, _, body) = get(
        &app,
        "/api/v1/datasets/line-01/interpretations/default/level2",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "radargram_mismatch");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("drawn on radargram"),
        "{body}"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_stale_delete_cannot_erase_newer_picks() {
    // Writing was version-checked and deleting was not, so a client
    // holding an old ETag could remove picks drawn after it last read.
    let (_dir, app) = project_app(true);
    let uri = "/api/v1/datasets/line-01/interpretations/default";

    let (status, etag, _) = put(&app, uri, &document("line-01"), None).await;
    assert_eq!(status, StatusCode::CREATED);
    let first = etag.expect("a write returns an ETag");

    // Someone else edits. The document must genuinely differ: the version
    // is a content hash, so saving identical bytes leaves it unchanged and
    // there would be nothing stale about the first ETag.
    let mut newer = document("line-01");
    newer["features"][0]["properties"]["id"] = serde_json::json!("f-edited");
    let (status, _, _) = put(&app, uri, &newer, Some(&first)).await;
    assert_eq!(status, StatusCode::OK);

    // The stale holder tries to delete.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(uri)
                .header(header::IF_MATCH, &first)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PRECONDITION_FAILED);

    // And the picks are still there.
    let (status, _, _) = get(&app, uri).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the document survived the stale delete"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn if_match_star_requires_the_document_to_already_exist() {
    // `If-Match: *` asserts "replace what is there". Mapping it to the
    // store's `Any` made it satisfied by an absent document too, so a
    // client asserting an existence precondition would instead create one
    // and be told 201.
    let (_dir, app) = project_app(true);
    let uri = "/api/v1/datasets/line-01/interpretations/default";

    let (status, _, body) = put(&app, uri, &document("line-01"), Some("*")).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED);
    assert_eq!(body["error"]["code"], "version_conflict");

    // Once it exists, the same header succeeds whatever the version is --
    // that is what distinguishes `*` from naming a version.
    let (status, _, _) = put(&app, uri, &document("line-01"), None).await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, _, _) = put(&app, uri, &document("line-01"), Some("*")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_write_cannot_target_another_users_interpretation() {
    // The path parameter must not be the authorisation. Without this,
    // `PUT .../interpretations/alice` writes Alice's picks for anyone who
    // asks -- harmless with one user, a hole the moment logins exist.
    let (dir, app) = project_app(true);

    let (status, _, body) = put(
        &app,
        "/api/v1/datasets/line-01/interpretations/someone-else",
        &document("line-01"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["code"], "not_your_interpretation");
    // Nothing was created for them.
    assert!(
        !dir.path()
            .join("interpretations/line-01/someone-else.gprinterp.json")
            .exists(),
        "a refused write must not leave a document behind"
    );

    // Deleting someone else's is refused the same way.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/v1/datasets/line-01/interpretations/someone-else")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    // Writing as yourself still works, so the guard is not simply refusing
    // everything.
    let (status, _, _) = put(
        &app,
        "/api/v1/datasets/line-01/interpretations/default",
        &document("line-01"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn another_users_interpretation_can_still_be_read() {
    // Per-user means "you cannot change mine", not "you cannot see mine":
    // reads keep naming the user in the path.
    let (_dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/datasets/line-01/interpretations/default",
        &document("line-01"),
        None,
    )
    .await;

    for uri in [
        "/api/v1/datasets/line-01/interpretations/default",
        "/api/v1/datasets/line-01/interpretations/default/raw",
    ] {
        let (status, _, _) = get(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_default_horizontal_scale_round_trips_and_reaches_the_viewer() {
    let (dir, app) = project_app(true);

    let (_, _, body) = get(&app, "/api/v1/project/settings").await;
    assert!(body["default_xscale"].is_null(), "unset to begin with");
    assert!(
        body["xscales"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["value"] == 2.0),
        "the page needs the offered factors to populate its select: {body}"
    );

    let (status, _, _) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_xscale": 2.0}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (_, _, body) = get(&app, "/api/v1/project/settings").await;
    assert_eq!(body["default_xscale"], 2.0);
    let marker = std::fs::read_to_string(dir.path().join("ridal.toml")).unwrap();
    assert!(marker.contains("default_xscale = 2.0"), "{marker}");

    // The point of the setting: the viewer opens already stretched, with
    // that option selected rather than 1x.
    let (status, html) = page(&app, "/view/line-01").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        html.contains(r#"<option value="2" selected>"#),
        "the viewer should preselect the project default"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn one_times_is_stored_as_no_preference_and_odd_scales_are_refused() {
    let (dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_xscale": 4.0}),
        None,
    )
    .await;

    // Back to 1x. It is the neutral value, not a preference, so the key
    // leaves the file rather than being written as 1.0.
    let (status, _, body) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_xscale": 1.0}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["default_xscale"].is_null(), "{body}");
    // Comment lines only -- `ridal project init` documents the key by name.
    let marker = std::fs::read_to_string(dir.path().join("ridal.toml")).unwrap();
    let live = marker
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| l.contains("default_xscale"))
        .count();
    assert_eq!(live, 0, "{marker}");

    // A factor the viewer does not offer would leave every radargram
    // stretched with no dropdown entry to undo it.
    let (status, _, body) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_xscale": 3.7}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unknown_xscale");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_default_profile_is_what_pages_render_with() {
    // The whole point of the setting: a page opened with no profile in its
    // address uses the project's, not the built-in one.
    let (_dir, app) = project_app(true);
    put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_profile": "abslog"}),
        None,
    )
    .await;

    for uri in ["/", "/view/line-01", "/layers", "/settings"] {
        let (status, html) = page(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        assert!(
            html.contains("?profile=abslog"),
            "{uri} did not pick up the project default"
        );
    }

    // An explicit request still wins.
    let (_, html) = page(&app, "/?profile=positive").await;
    assert!(html.contains("value=\"positive\" selected"), "{html}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_unknown_default_profile_is_refused() {
    // Storing a name nothing renders would leave every page failing with
    // no obvious cause.
    let (_dir, app) = project_app(true);
    let (status, _, body) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_profile": "nope"}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unknown_profile");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_default_profile_can_be_cleared() {
    let (_dir, app) = project_app(true);
    let settings = "/api/v1/project/settings";
    put(
        &app,
        settings,
        &serde_json::json!({"default_profile": "abslog"}),
        None,
    )
    .await;
    put(
        &app,
        settings,
        &serde_json::json!({"default_profile": null}),
        None,
    )
    .await;

    let (_, _, body) = get(&app, settings).await;
    assert!(body["default_profile"].is_null(), "{body}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn settings_are_read_only_where_writes_are() {
    let (_dir, app) = project_app(false);
    let (status, _, body) = put(
        &app,
        "/api/v1/project/settings",
        &serde_json::json!({"default_profile": "abslog"}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "read_only");

    // Reading still works, and the page says why the form is inert.
    let (_, _, body) = get(&app, "/api/v1/project/settings").await;
    assert_eq!(body["writable"], false);
    let (status, html) = page(&app, "/settings").await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("read-only"), "{html}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_bare_catalog_has_nothing_to_configure() {
    let (_dir, app) = bare_app();
    let (status, _, body) = get(&app, "/api/v1/project/settings").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["project"], false);

    let (status, html) = page(&app, "/settings").await;
    assert_eq!(status, StatusCode::OK);
    assert!(html.contains("ridal project init"), "{html}");
    assert!(!html.contains("id=\"settings-form\""), "no form to offer");
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
