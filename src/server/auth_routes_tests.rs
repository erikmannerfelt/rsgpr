//! HTTP-level tests for authentication, roles and download scopes (#131).
//!
//! Driven through the real Axum router with `ServiceExt::oneshot`, so the
//! status codes, cookies and error envelopes are the ones a browser will
//! actually see -- and so the identity middleware is in the path, which is
//! where every one of these decisions is actually made.
//!
//! Every test builds an `AppState`, which creates and opens a NetCDF.
//! netcdf-c is not thread-safe, so they carry the same
//! `#[serial_test::serial(netcdf)]` guard as the tests in `app.rs`.

use std::path::Path as StdPath;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::app::{build_router, AppState};
use super::render::service::RenderServiceConfig;
use crate::identity::UserId;
use crate::project::store::Expectation;
use crate::project::users::{self, DownloadScope, Invite, Role, User, UserSet};
use crate::project::Project;

const RADARGRAM: &str = "line-01";
const PASSWORD: &str = "correct horse battery";

fn write_test_nc(path: &StdPath, radargram_id: &str) {
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("y", 8).unwrap();
    file.add_dimension("x", 40).unwrap();
    let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
    let data: Vec<f32> = (0..(8 * 40)).map(|i| (i % 97) as f32).collect();
    var.put_values(&data, ..).unwrap();
    file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
        .unwrap();
    file.add_attribute("ridal_version", "ridal version 0.0.0 by test")
        .unwrap();
    file.add_attribute("ridal_radargram_id", radargram_id)
        .unwrap();
}

fn id(name: &str) -> UserId {
    UserId::new(name).unwrap()
}

/// An activated account. The hash is passed in so a test with several
/// accounts pays for Argon2id once rather than once per person.
fn activated(name: &str, role: Role, download: DownloadScope, hash: &str) -> User {
    let mut user = User::new(id(name), role, download);
    user.password_hash = Some(hash.to_string());
    user
}

/// A project with one radargram and the given accounts.
///
/// `users` being non-empty is what makes this an authenticated project:
/// writing the file at all is the opt-in, which is why the tests that need
/// today's unauthenticated behaviour use `interp_routes_tests` instead.
fn app_with(users: Vec<User>) -> (tempfile::TempDir, Router) {
    app_with_set(UserSet {
        users,
        ..UserSet::default()
    })
}

fn app_with_set(set: UserSet) -> (tempfile::TempDir, Router) {
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc(&dir.path().join("radargrams").join("line-01.nc"), RADARGRAM);
    let project = Project::discover(dir.path()).unwrap().unwrap();
    users::write(project.documents(), &set, &Expectation::Any).unwrap();

    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            false,
        )
        .unwrap(),
    );
    (dir, build_router(state))
}

/// The parts of a response these tests assert on.
struct Response {
    status: StatusCode,
    cookie: Option<String>,
    body: Value,
    text: String,
}

async fn send(app: &Router, request: Request<Body>) -> Response {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let cookie = response
        .headers()
        .get(header::SET_COOKIE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    Response {
        status,
        cookie,
        body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        text,
    }
}

/// The `name=value` part of a `Set-Cookie`, ready to send back as `Cookie`.
fn cookie_pair(set_cookie: &str) -> String {
    set_cookie.split(';').next().unwrap().to_string()
}

fn request(method: &str, uri: &str, session: Option<&str>) -> axum::http::request::Builder {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(session) = session {
        builder = builder.header(header::COOKIE, session);
    }
    builder
}

async fn get(app: &Router, uri: &str, session: Option<&str>) -> Response {
    send(
        app,
        request("GET", uri, session).body(Body::empty()).unwrap(),
    )
    .await
}

async fn post(app: &Router, uri: &str, body: &Value, session: Option<&str>) -> Response {
    send(
        app,
        request("POST", uri, session)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn put(app: &Router, uri: &str, body: &Value, session: Option<&str>) -> Response {
    send(
        app,
        request("PUT", uri, session)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

async fn delete(app: &Router, uri: &str, session: Option<&str>) -> Response {
    send(
        app,
        request("DELETE", uri, session).body(Body::empty()).unwrap(),
    )
    .await
}

/// Sign in and return the cookie to send back.
async fn sign_in(app: &Router, name: &str) -> String {
    let response = post(
        app,
        "/api/v1/auth/login",
        &json!({"name": name, "password": PASSWORD}),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::OK, "{}", response.text);
    cookie_pair(&response.cookie.expect("a login must set a cookie"))
}

fn document(key: &str) -> Value {
    json!({
        "key": key,
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[5.0, 2.0], [30.0, 3.0]]},
            "properties": {"id": "f-0001", "label": "bed"}
        }]
    })
}

fn interpretation_uri(user: &str) -> String {
    format!("/api/v1/datasets/{RADARGRAM}/interpretations/{user}")
}

// ---------------------------------------------------------------------------

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_invite_is_the_only_way_a_password_is_ever_set() {
    // The whole account lifecycle, end to end: an administrator exists
    // (created from the command line, simulated here by writing the file),
    // creates an account, hands over a link, and the link is what sets the
    // password. No password is ever known to two people.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);
    let admin = sign_in(&app, "erik").await;

    let created = post(
        &app,
        "/api/v1/users",
        &json!({"name": "student", "role": "picker", "download": "derived"}),
        Some(&admin),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED, "{}", created.text);
    assert_eq!(created.body["user"]["activated"], false);
    assert_eq!(created.body["user"]["invite_pending"], true);

    // The token exists exactly here and nowhere else: only its hash is
    // stored. A path rather than a URL, because Ridal sits behind a proxy
    // and must not build one from a header a client controls.
    let path = created.body["invite_path"].as_str().unwrap().to_string();
    assert!(path.starts_with("/invite/"), "{path}");
    let token = path.trim_start_matches("/invite/").to_string();

    // Until it is redeemed there is no password, and saying so is refused
    // the same way a wrong password is.
    let refused = post(
        &app,
        "/api/v1/auth/login",
        &json!({"name": "student", "password": ""}),
        None,
    )
    .await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);

    // The page is reachable by anyone holding the link.
    let page = get(&app, &path, None).await;
    assert_eq!(page.status, StatusCode::OK);

    let redeemed = post(
        &app,
        "/api/v1/auth/invite",
        &json!({"token": token, "password": "a longer passphrase"}),
        None,
    )
    .await;
    assert_eq!(redeemed.status, StatusCode::OK, "{}", redeemed.text);
    // Signed in straight away: they just proved they hold the link and
    // chose a password, and asking for it again would be ceremony.
    let session = cookie_pair(&redeemed.cookie.expect("redeeming signs you in"));
    let me = get(&app, "/api/v1/auth/me", Some(&session)).await;
    assert_eq!(me.body["user"], "student");
    assert_eq!(me.body["role"], "picker");

    // And the token is consumed.
    let again = post(
        &app,
        "/api/v1/auth/invite",
        &json!({"token": token, "password": "another passphrase"}),
        None,
    )
    .await;
    assert_eq!(again.status, StatusCode::BAD_REQUEST);
    assert!(
        again.text.contains("already have been used"),
        "{}",
        again.text
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_expired_invite_is_refused() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let mut stale = activated("student", Role::Picker, DownloadScope::All, &hash);
    stale.password_hash = None;
    stale.invite = Some(Invite {
        token_hash: blake3::hash(b"stale-token").to_hex().to_string(),
        // Long past.
        expires: 1,
    });
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Admin, DownloadScope::All, &hash),
        stale,
    ]);

    let response = post(
        &app,
        "/api/v1/auth/invite",
        &json!({"token": "stale-token", "password": "a longer passphrase"}),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert!(response.text.contains("expired"), "{}", response.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_wrong_password_and_an_unknown_name_are_refused_identically() {
    // Otherwise the login form is a way to enumerate who has an account and
    // who has not signed up yet.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);

    let wrong = post(
        &app,
        "/api/v1/auth/login",
        &json!({"name": "erik", "password": "not the password"}),
        None,
    )
    .await;
    let unknown = post(
        &app,
        "/api/v1/auth/login",
        &json!({"name": "nobody", "password": PASSWORD}),
        None,
    )
    .await;

    assert_eq!(wrong.status, StatusCode::UNAUTHORIZED);
    assert_eq!(unknown.status, StatusCode::UNAUTHORIZED);
    assert_eq!(wrong.body, unknown.body, "the two refusals must not differ");
    assert!(wrong.cookie.is_none(), "a refused login must set no cookie");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_password_hash_never_leaves_the_process() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);
    let admin = sign_in(&app, "erik").await;

    for uri in [
        "/api/v1/users",
        "/api/v1/auth/me",
        "/api/v1/project/settings",
    ] {
        let response = get(&app, uri, Some(&admin)).await;
        assert!(
            !response.text.contains("argon2"),
            "{uri}: {}",
            response.text
        );
        assert!(
            !response.text.contains("password_hash"),
            "{uri}: {}",
            response.text
        );
        assert!(
            !response.text.contains("token_hash"),
            "{uri}: {}",
            response.text
        );
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn picks_belong_to_the_person_who_drew_them_and_not_even_an_admin_may_edit_them() {
    // Per-user by design, and a property of the data rather than a
    // permission: there is deliberately no role that can overrule it.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Admin, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let student = sign_in(&app, "student").await;
    let admin = sign_in(&app, "erik").await;

    let own = put(
        &app,
        &interpretation_uri("student"),
        &document(RADARGRAM),
        Some(&student),
    )
    .await;
    assert_eq!(own.status, StatusCode::CREATED, "{}", own.text);

    // The administrator can read them...
    let read = get(&app, &interpretation_uri("student"), Some(&admin)).await;
    assert_eq!(read.status, StatusCode::OK);

    // ...and cannot write them, despite outranking everyone.
    let forged = put(
        &app,
        &interpretation_uri("student"),
        &document(RADARGRAM),
        Some(&admin),
    )
    .await;
    assert_eq!(forged.status, StatusCode::FORBIDDEN);
    assert_eq!(forged.body["error"]["code"], "not_your_interpretation");

    // Nor delete them.
    let removed = delete(&app, &interpretation_uri("student"), Some(&admin)).await;
    assert_eq!(removed.status, StatusCode::FORBIDDEN);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_path_parameter_is_no_longer_the_authorisation() {
    // Without a session the path used to be all the authorisation there
    // was: `PUT .../interpretations/erik` would write Erik's document for
    // anyone who asked.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Picker,
        DownloadScope::All,
        &hash,
    )]);

    let anonymous = put(
        &app,
        &interpretation_uri("erik"),
        &document(RADARGRAM),
        None,
    )
    .await;
    assert_eq!(anonymous.status, StatusCode::UNAUTHORIZED);
    assert_eq!(anonymous.body["error"]["code"], "authentication_required");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_forged_cookie_is_not_a_session() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);

    // Everything a real cookie has except a signature made with the key.
    let forged = format!("ridal_session=erik.1.9999999999.{}", "a".repeat(64));
    let me = get(&app, "/api/v1/auth/me", Some(&forged)).await;
    assert_eq!(me.body["authenticated"], false);
    assert_eq!(me.body["role"], "viewer");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_viewer_reads_a_picker_writes_and_an_operator_curates() {
    // The ladder, at the three places it actually bites.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("watcher", Role::Viewer, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
        activated("erik", Role::Operator, DownloadScope::All, &hash),
    ]);
    let viewer = sign_in(&app, "watcher").await;
    let picker = sign_in(&app, "student").await;
    let operator = sign_in(&app, "erik").await;

    // Reading the catalog: everyone.
    for session in [&viewer, &picker, &operator] {
        let response = get(&app, "/api/v1/datasets", Some(session)).await;
        assert_eq!(response.status, StatusCode::OK);
    }

    // Writing an interpretation: picker and above.
    let refused = put(
        &app,
        &interpretation_uri("watcher"),
        &document(RADARGRAM),
        Some(&viewer),
    )
    .await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    assert_eq!(refused.body["error"]["code"], "insufficient_role");
    assert!(refused.text.contains("picker"), "{}", refused.text);

    let allowed = put(
        &app,
        &interpretation_uri("student"),
        &document(RADARGRAM),
        Some(&picker),
    )
    .await;
    assert_eq!(allowed.status, StatusCode::CREATED, "{}", allowed.text);

    // The layer vocabulary: readable by a viewer, changed by an operator.
    // A picker *uses* layers but does not get to invent them.
    let layers = json!([{"id": "bed", "name": "Bed", "color": "#e6194b"}]);
    let read = get(&app, "/api/v1/layers", Some(&viewer)).await;
    assert_eq!(read.status, StatusCode::OK);
    assert_eq!(read.body["writable"], false);

    let refused = put(&app, "/api/v1/layers", &layers, Some(&picker)).await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    assert!(refused.text.contains("operator"), "{}", refused.text);

    let allowed = put(&app, "/api/v1/layers", &layers, Some(&operator)).await;
    assert_eq!(allowed.status, StatusCode::OK, "{}", allowed.text);
    let read = get(&app, "/api/v1/layers", Some(&operator)).await;
    assert_eq!(read.body["writable"], true);

    // The project defaults: operator, not picker.
    let refused = put(
        &app,
        "/api/v1/project/settings",
        &json!({"default_profile": "abslog"}),
        Some(&picker),
    )
    .await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    let allowed = put(
        &app,
        "/api/v1/project/settings",
        &json!({"default_profile": "abslog"}),
        Some(&operator),
    )
    .await;
    assert_eq!(allowed.status, StatusCode::OK, "{}", allowed.text);

    // Accounts: admin only, and an operator is not one.
    let refused = get(&app, "/api/v1/users", Some(&operator)).await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    assert!(refused.text.contains("admin"), "{}", refused.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_demotion_takes_effect_on_the_next_request_not_when_the_cookie_expires() {
    // What the credential version in the cookie buys: user management that
    // actually manages, rather than taking effect at some point in the next
    // fortnight.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Admin, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let admin = sign_in(&app, "erik").await;
    let student = sign_in(&app, "student").await;

    let before = get(&app, "/api/v1/auth/me", Some(&student)).await;
    assert_eq!(before.body["role"], "picker");

    let demoted = put(
        &app,
        "/api/v1/users/student",
        &json!({"role": "viewer"}),
        Some(&admin),
    )
    .await;
    assert_eq!(demoted.status, StatusCode::OK, "{}", demoted.text);

    // The same cookie, one request later.
    let after = get(&app, "/api/v1/auth/me", Some(&student)).await;
    assert_eq!(
        after.body["authenticated"], false,
        "the old cookie names a credential version the account no longer has"
    );

    let refused = put(
        &app,
        &interpretation_uri("student"),
        &document(RADARGRAM),
        Some(&student),
    )
    .await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);

    // Signing in again gets a cookie at the new version, with the new role.
    let student = sign_in(&app, "student").await;
    let now = get(&app, "/api/v1/auth/me", Some(&student)).await;
    assert_eq!(now.body["role"], "viewer");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_departed_users_picks_survive_the_account() {
    // Attributed scientific data. Removing the person must not destroy it.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (dir, app) = app_with(vec![
        activated("erik", Role::Admin, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let admin = sign_in(&app, "erik").await;
    let student = sign_in(&app, "student").await;

    put(
        &app,
        &interpretation_uri("student"),
        &document(RADARGRAM),
        Some(&student),
    )
    .await;
    // A preference, which is about the person rather than the survey.
    put(
        &app,
        "/api/v1/preferences",
        &json!({"render_profile": "abslog"}),
        Some(&student),
    )
    .await;

    let removed = delete(&app, "/api/v1/users/student", Some(&admin)).await;
    assert_eq!(removed.status, StatusCode::NO_CONTENT, "{}", removed.text);

    // The account is gone and its session with it.
    let after = get(&app, "/api/v1/auth/me", Some(&student)).await;
    assert_eq!(after.body["authenticated"], false);

    // The picks are not.
    let stored = get(&app, &interpretation_uri("student"), Some(&admin)).await;
    assert_eq!(stored.status, StatusCode::OK, "{}", stored.text);
    let listed = get(
        &app,
        &format!("/api/v1/datasets/{RADARGRAM}/interpretations"),
        Some(&admin),
    )
    .await;
    assert_eq!(listed.body["users"], json!(["student"]));

    // Their preferences are, because a preference is about a person.
    assert!(
        !dir.path().join("preferences/student.json").exists(),
        "a departed account's preferences should not linger"
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn the_last_administrator_cannot_be_demoted_or_removed() {
    // Either would lock the access settings away from everyone, with no way
    // back short of editing users.json by hand.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Admin, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let admin = sign_in(&app, "erik").await;

    let demote = put(
        &app,
        "/api/v1/users/erik",
        &json!({"role": "operator"}),
        Some(&admin),
    )
    .await;
    assert_eq!(demote.status, StatusCode::BAD_REQUEST);
    assert!(
        demote.text.contains("only administrator"),
        "{}",
        demote.text
    );

    let remove = delete(&app, "/api/v1/users/erik", Some(&admin)).await;
    assert_eq!(remove.status, StatusCode::BAD_REQUEST);

    // With a second administrator, both become possible.
    put(
        &app,
        "/api/v1/users/student",
        &json!({"role": "admin"}),
        Some(&admin),
    )
    .await;
    let demote = put(
        &app,
        "/api/v1/users/erik",
        &json!({"role": "operator"}),
        Some(&admin),
    )
    .await;
    assert_eq!(demote.status, StatusCode::OK, "{}", demote.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn download_scope_gates_the_downloads_and_not_the_viewer() {
    // The distinction the issue insists on: this stops casual bulk export
    // and states an intent. It cannot stop someone who can open the page,
    // and gating what the viewer draws with would break the viewer for
    // everyone below "all".
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("nothing", Role::Picker, DownloadScope::None, &hash),
        activated("picks", Role::Picker, DownloadScope::Picks, &hash),
        activated("derived", Role::Picker, DownloadScope::Derived, &hash),
        activated("everything", Role::Picker, DownloadScope::All, &hash),
    ]);

    let sessions = [
        ("nothing", sign_in(&app, "nothing").await),
        ("picks", sign_in(&app, "picks").await),
        ("derived", sign_in(&app, "derived").await),
        ("everything", sign_in(&app, "everything").await),
    ];

    // Something to download.
    put(
        &app,
        &interpretation_uri("picks"),
        &document(RADARGRAM),
        Some(&sessions[1].1),
    )
    .await;

    let raw = format!("/api/v1/datasets/{RADARGRAM}/interpretations/picks/raw");
    let image = format!("/api/v1/datasets/{RADARGRAM}/views/standard/image?width=20");
    let track = format!("/api/v1/datasets/{RADARGRAM}/track.geojson");
    let netcdf = format!("/api/v1/datasets/{RADARGRAM}/download");

    // What each level may take. The rows are the ladder.
    //
    // Asserted as "was this refused on permission grounds", not as "did it
    // return 200": the fixture radargram carries no coordinate variables,
    // so its track cannot be built and answers 500 even to someone allowed
    // it. Whether the scope let the request through is the question here,
    // and `track.rs` has its own tests for the rest.
    let expected = [
        // (session, raw, image, track+netcdf)
        (0usize, false, false, false),
        (1, true, false, false),
        (2, true, true, false),
        (3, true, true, true),
    ];
    for (index, picks_ok, derived_ok, all_ok) in expected {
        let (name, session) = &sessions[index];
        let permitted = |response: &Response| {
            !matches!(
                response.status,
                StatusCode::FORBIDDEN | StatusCode::UNAUTHORIZED
            )
        };

        let response = get(&app, &raw, Some(session)).await;
        assert_eq!(permitted(&response), picks_ok, "{name}: raw picks");
        if picks_ok {
            assert_eq!(response.status, StatusCode::OK, "{name}: raw picks");
        }

        let response = get(&app, &image, Some(session)).await;
        assert_eq!(permitted(&response), derived_ok, "{name}: rendered image");
        if derived_ok {
            assert_eq!(response.status, StatusCode::OK, "{name}: rendered image");
        }

        for uri in [&track, &netcdf] {
            assert_eq!(
                permitted(&get(&app, uri, Some(session)).await),
                all_ok,
                "{name}: {uri}"
            );
        }
        if all_ok {
            assert_eq!(
                get(&app, &netcdf, Some(session)).await.status,
                StatusCode::OK,
                "{name}: the radargram itself"
            );
        }

        // The viewer keeps working at every level, including "none": the
        // chunks it draws from are not a download, and gating them would
        // make the scope setting break the thing it is meant to protect.
        let chunk = format!("/api/v1/datasets/{RADARGRAM}/views/standard/chunks/default/0/0");
        assert_eq!(
            get(&app, &chunk, Some(session)).await.status,
            StatusCode::OK,
            "{name}: the viewer must keep working"
        );
        assert_eq!(
            get(&app, &format!("/view/{RADARGRAM}"), Some(session))
                .await
                .status,
            StatusCode::OK,
            "{name}: the viewer page must keep opening"
        );
    }
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_anonymous_reader_downloads_what_the_project_allows_them() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with_set(UserSet {
        anonymous_download: DownloadScope::Derived,
        users: vec![activated("erik", Role::Admin, DownloadScope::All, &hash)],
        ..UserSet::default()
    });

    // Public read is the default, so the catalog is open...
    assert_eq!(
        get(&app, "/api/v1/datasets", None).await.status,
        StatusCode::OK
    );
    // ...the rendered image is within "derived"...
    assert_eq!(
        get(
            &app,
            &format!("/api/v1/datasets/{RADARGRAM}/views/standard/image?width=20"),
            None
        )
        .await
        .status,
        StatusCode::OK
    );
    // ...and the radargram itself is not. Anonymous gets a 401 rather than
    // a 403, because for them signing in is a thing that might help.
    let refused = get(
        &app,
        &format!("/api/v1/datasets/{RADARGRAM}/download"),
        None,
    )
    .await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);

    let admin = sign_in(&app, "erik").await;
    assert_eq!(
        get(
            &app,
            &format!("/api/v1/datasets/{RADARGRAM}/download"),
            Some(&admin)
        )
        .await
        .status,
        StatusCode::OK
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn requiring_a_login_to_read_hides_everything_but_the_way_in() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with_set(UserSet {
        require_auth_to_read: true,
        users: vec![activated("erik", Role::Admin, DownloadScope::All, &hash)],
        ..UserSet::default()
    });

    // A page redirects, because someone who followed a link wants the page
    // rather than a status code.
    let redirected = get(&app, "/", None).await;
    assert_eq!(redirected.status, StatusCode::SEE_OTHER);

    // The API says so in the envelope every other failure uses.
    let refused = get(&app, "/api/v1/datasets", None).await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);
    assert_eq!(refused.body["error"]["code"], "authentication_required");

    // The way in stays open, including the login page's own assets -- a
    // login screen that 401s its stylesheet is not a login screen.
    for uri in [
        "/login",
        "/static/app.css",
        "/static/login.js",
        "/favicon.ico",
        "/api/v1/health",
    ] {
        assert_eq!(get(&app, uri, None).await.status, StatusCode::OK, "{uri}");
    }

    let session = sign_in(&app, "erik").await;
    assert_eq!(get(&app, "/", Some(&session)).await.status, StatusCode::OK);
    assert_eq!(
        get(&app, "/api/v1/datasets", Some(&session)).await.status,
        StatusCode::OK
    );
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn preferences_sit_between_the_request_and_the_project_default() {
    // The cascade, all four layers, against the value a page actually
    // renders with.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Operator, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let erik = sign_in(&app, "erik").await;
    let student = sign_in(&app, "student").await;

    let viewer_uri = format!("/view/{RADARGRAM}");
    let renders_with =
        |text: &str, profile: &str| text.contains(&format!("value=\"{profile}\" selected"));

    // Built-in, with nothing set anywhere.
    let page = get(&app, &viewer_uri, Some(&student)).await;
    assert!(renders_with(&page.text, "default"), "{}", page.text);

    // The project default reaches everyone who has not chosen.
    put(
        &app,
        "/api/v1/project/settings",
        &json!({"default_profile": "abslog"}),
        Some(&erik),
    )
    .await;
    let page = get(&app, &viewer_uri, Some(&student)).await;
    assert!(renders_with(&page.text, "abslog"), "{}", page.text);

    // One person's own preference wins over it...
    let saved = put(
        &app,
        "/api/v1/preferences",
        &json!({"render_profile": "positive", "x_scale": 2.0}),
        Some(&student),
    )
    .await;
    assert_eq!(saved.status, StatusCode::OK, "{}", saved.text);
    let page = get(&app, &viewer_uri, Some(&student)).await;
    assert!(renders_with(&page.text, "positive"), "{}", page.text);

    // ...for them alone.
    let page = get(&app, &viewer_uri, Some(&erik)).await;
    assert!(renders_with(&page.text, "abslog"), "{}", page.text);

    // And the request parameter wins over everything, for one page.
    let page = get(&app, &format!("{viewer_uri}?profile=positive"), Some(&erik)).await;
    assert!(renders_with(&page.text, "positive"), "{}", page.text);

    // `?xscale=` too, which fell out of naming the cascade.
    let page = get(&app, &format!("{viewer_uri}?xscale=4"), Some(&student)).await;
    assert!(page.text.contains("value=\"4\" selected"), "{}", page.text);

    // Clearing a preference falls back rather than storing the default,
    // which is what lets a later project change reach this person again.
    put(
        &app,
        "/api/v1/preferences",
        &json!({"render_profile": null, "x_scale": null}),
        Some(&student),
    )
    .await;
    let page = get(&app, &viewer_uri, Some(&student)).await;
    assert!(renders_with(&page.text, "abslog"), "{}", page.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_admin_setting_the_project_default_does_not_overwrite_anyone() {
    // The alternative -- a project default that stamps over personal
    // choices -- would make the operator's save button destructive in a way
    // nothing in the UI could warn about convincingly.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![
        activated("erik", Role::Operator, DownloadScope::All, &hash),
        activated("student", Role::Picker, DownloadScope::All, &hash),
    ]);
    let erik = sign_in(&app, "erik").await;
    let student = sign_in(&app, "student").await;

    put(
        &app,
        "/api/v1/preferences",
        &json!({"render_profile": "positive"}),
        Some(&student),
    )
    .await;
    put(
        &app,
        "/api/v1/project/settings",
        &json!({"default_profile": "abslog"}),
        Some(&erik),
    )
    .await;

    let mine = get(&app, "/api/v1/preferences", Some(&student)).await;
    assert_eq!(mine.body["render_profile"], "positive");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn an_anonymous_reader_has_nowhere_to_keep_a_preference() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);

    // A 401 rather than a silent no-op: the page would otherwise show a
    // saved setting that was never saved.
    let refused = put(
        &app,
        "/api/v1/preferences",
        &json!({"render_profile": "abslog"}),
        None,
    )
    .await;
    assert_eq!(refused.status, StatusCode::UNAUTHORIZED);
    assert!(refused.text.contains("?profile="), "{}", refused.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_merged_download_asks_whose_picks_when_nobody_is_signed_in() {
    // "Mine" has no meaning for an anonymous reader, and guessing would
    // hand back an empty file that looks complete.
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);

    let refused = get(&app, "/api/v1/catalog/level2", None).await;
    assert_eq!(refused.status, StatusCode::BAD_REQUEST);
    assert_eq!(refused.body["error"]["code"], "user_required");
    assert!(refused.text.contains("?user="), "{}", refused.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn signing_out_clears_the_cookie_and_the_session_with_it() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let (_dir, app) = app_with(vec![activated(
        "erik",
        Role::Admin,
        DownloadScope::All,
        &hash,
    )]);
    let session = sign_in(&app, "erik").await;
    assert_eq!(
        get(&app, "/api/v1/auth/me", Some(&session)).await.body["user"],
        "erik"
    );

    let out = post(&app, "/api/v1/auth/logout", &json!({}), Some(&session)).await;
    assert_eq!(out.status, StatusCode::OK);
    let cleared = out.cookie.expect("signing out must clear the cookie");
    assert!(cleared.contains("Max-Age=0"), "{cleared}");
    assert!(cleared.starts_with("ridal_session=;"), "{cleared}");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_session_survives_a_restart() {
    // The reason a signed cookie is the smaller option: the key is on disk,
    // so nothing about sessions has to be.
    let hash = users::hash_password(PASSWORD).unwrap();
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc(&dir.path().join("radargrams").join("line-01.nc"), RADARGRAM);

    let build = || {
        let project = Project::discover(dir.path()).unwrap().unwrap();
        let state = Arc::new(
            AppState::build_with_project(
                dir.path(),
                &RenderServiceConfig::default(),
                Some(project),
                false,
            )
            .unwrap(),
        );
        build_router(state)
    };

    let project = Project::discover(dir.path()).unwrap().unwrap();
    users::write(
        project.documents(),
        &UserSet {
            users: vec![activated("erik", Role::Admin, DownloadScope::All, &hash)],
            ..UserSet::default()
        },
        &Expectation::Any,
    )
    .unwrap();
    drop(project);

    let first = build();
    let session = sign_in(&first, "erik").await;

    // A second server over the same project, as a restart would be.
    let second = build();
    let me = get(&second, "/api/v1/auth/me", Some(&session)).await;
    assert_eq!(me.body["user"], "erik", "a restart signed everyone out");
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_project_with_no_accounts_offers_no_login_and_still_writes() {
    // The migration rule: upgrading Ridal must not lock anyone out of their
    // own data, and `ridal gui` must keep working with no login step.
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc(&dir.path().join("radargrams").join("line-01.nc"), RADARGRAM);
    let project = Project::discover(dir.path()).unwrap().unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            false,
        )
        .unwrap(),
    );
    let app = build_router(state);

    let me = get(&app, "/api/v1/auth/me", None).await;
    assert_eq!(me.body["user"], "default");
    assert_eq!(me.body["role"], "operator");
    assert_eq!(me.body["authentication_configured"], false);

    let saved = put(
        &app,
        &interpretation_uri("default"),
        &document(RADARGRAM),
        None,
    )
    .await;
    assert_eq!(saved.status, StatusCode::CREATED, "{}", saved.text);

    // No `users.json` was created by serving, only by an administrator
    // deciding to create one.
    assert!(!dir.path().join("users.json").exists());
    // And no session key either: a project that never authenticates never
    // grows one.
    assert!(!dir.path().join("session.key").exists());

    // The login page explains rather than offering a form that cannot work.
    let login = get(&app, "/login", None).await;
    assert_eq!(login.status, StatusCode::OK);
    assert!(login.text.contains("no accounts"), "{}", login.text);
    assert!(
        login.text.contains("ridal project user add"),
        "{}",
        login.text
    );
    // And the header offers no sign-in link, since there is nothing to sign
    // in to.
    let index = get(&app, "/", None).await;
    assert!(!index.text.contains(">Sign in<"), "{}", index.text);
}

#[tokio::test]
#[serial_test::serial(netcdf)]
async fn a_read_only_server_caps_even_an_administrator() {
    let hash = users::hash_password(PASSWORD).unwrap();
    let dir = tempfile::tempdir().unwrap();
    Project::init(dir.path(), Some("test")).unwrap();
    write_test_nc(&dir.path().join("radargrams").join("line-01.nc"), RADARGRAM);
    let project = Project::discover(dir.path()).unwrap().unwrap();
    users::write(
        project.documents(),
        &UserSet {
            users: vec![activated("erik", Role::Admin, DownloadScope::All, &hash)],
            ..UserSet::default()
        },
        &Expectation::Any,
    )
    .unwrap();
    let state = Arc::new(
        AppState::build_with_project(
            dir.path(),
            &RenderServiceConfig::default(),
            Some(project),
            true,
        )
        .unwrap(),
    );
    let app = build_router(state);

    let session = sign_in(&app, "erik").await;
    let me = get(&app, "/api/v1/auth/me", Some(&session)).await;
    assert_eq!(me.body["account_role"], "admin", "the account is unchanged");
    assert_eq!(me.body["role"], "viewer", "what they may do here is not");

    let refused = put(
        &app,
        &interpretation_uri("erik"),
        &document(RADARGRAM),
        Some(&session),
    )
    .await;
    assert_eq!(refused.status, StatusCode::FORBIDDEN);
    // Names the flag, because the operator of the server is the one who can
    // change it -- "you are a viewer" would be true and useless.
    assert_eq!(refused.body["error"]["code"], "read_only");

    // Reading is unaffected.
    assert_eq!(
        get(&app, "/api/v1/datasets", Some(&session)).await.status,
        StatusCode::OK
    );
}
