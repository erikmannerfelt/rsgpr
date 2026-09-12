//! Who is asking, and how the server knows (#131).
//!
//! # Sessions are a signed cookie, not a table
//!
//! The stateless option is the smaller one here, which is worth saying
//! because the instinct runs the other way. An in-memory session map loses
//! every login on restart, and avoiding that means writing sessions to disk
//! and sweeping them for expiry. A signed cookie stores nothing
//! server-side, so surviving a restart falls out of the *key* being
//! persistent rather than of persisting sessions.
//!
//! The primitive was already a dependency. `blake3::keyed_hash` is a keyed
//! MAC, and [`blake3::Hash`] provides constant-time equality -- its docs say
//! so, and it deliberately omits `Deref`/`AsRef` so the property cannot be
//! lost to an implicit conversion into a byte slice.
//!
//! # Revocation
//!
//! The wrinkle a stateless cookie brings is that it cannot be withdrawn
//! before it expires, so deleting a user or demoting them would not take
//! effect until their cookie aged out. Each account carries a credential
//! version; the cookie carries the version it was minted at, and
//! verification compares the two against `users.json`, which is a small file
//! the server reads anyway. Changing a password, a role or a download scope
//! bumps it, so user management takes effect immediately.
//!
//! # Transport
//!
//! A password over plain HTTP is cleartext on the wire, and Ridal will
//! essentially never see HTTPS: behind a TLS-terminating proxy it sees plain
//! HTTP on loopback, which is correct and safe. So the guardrail cannot ask
//! "is this connection TLS?" -- the answer is always no. It is keyed on the
//! *bind address* instead, in [`super::launch`].
//!
//! For the same reason the cookie is not marked `Secure`: that attribute
//! would stop it being sent over the loopback HTTP that both `ridal gui` and
//! every reverse-proxy deployment actually speak. `HttpOnly` and
//! `SameSite=Lax` are set, and both are meaningful regardless of transport.

use std::path::Path;
use std::sync::Arc;

use axum::extract::{FromRequestParts, Request, State};
use axum::http::{header, request::Parts, HeaderMap};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};

use super::app::AppState;
use super::routes::ApiError;
use crate::identity::UserId;
use crate::project::store::{DocumentStore, Expectation};
use crate::project::users::{self, DownloadScope, Role, UserSet};
use crate::project::Project;

/// Name of the session cookie.
pub const SESSION_COOKIE: &str = "ridal_session";

/// The key file, relative to the project root. Created on first login, so a
/// project that never authenticates never grows one.
pub const SESSION_KEY_FILE: &str = "session.key";

/// How long a session lasts. Long enough that a working week does not mean
/// five logins, short enough that a forgotten browser on a shared machine
/// does not stay signed in indefinitely.
pub const SESSION_TTL_DAYS: i64 = 14;

/// Domain separator, so a signature minted for a session can never be
/// mistaken for one minted for anything else this key might sign later.
const SESSION_DOMAIN: &str = "ridal-session-v1";

/// The key that signs session cookies.
#[derive(Clone)]
pub struct SessionKey([u8; 32]);

impl std::fmt::Debug for SessionKey {
    /// Never prints the key. A `Debug` that did would put a forgery kit for
    /// every account into any log line that formatted the app state.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionKey(<redacted>)")
    }
}

impl SessionKey {
    /// Read the project's key, creating one if there is none.
    ///
    /// Stored as hex through the document store, which gives the atomic
    /// write and the owner-only mode for free. A key file that exists but
    /// does not parse is an error rather than a reason to mint a new one:
    /// replacing it would silently sign every existing session out and hide
    /// whatever damaged the file.
    pub fn load_or_create(store: &DocumentStore) -> Result<SessionKey, String> {
        let path = Path::new(SESSION_KEY_FILE);
        if let Some(document) = store.read(path).map_err(|e| e.to_string())? {
            return users::from_hex_32(&document.text)
                .map(SessionKey)
                .ok_or_else(|| {
                    format!(
                        "{} is not a 32-byte hex key. Delete it to start a new one, \
                         which signs every existing session out.",
                        store.root().join(SESSION_KEY_FILE).display()
                    )
                });
        }

        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes)
            .map_err(|e| format!("could not read system randomness: {e}"))?;
        let text = format!("{}\n", users::to_hex(&bytes));
        match store.write_private(path, &text, &Expectation::Absent) {
            Ok(_) => Ok(SessionKey(bytes)),
            // Another process created it between the read and the write.
            // Theirs is the key now; discard the one just generated rather
            // than overwriting and invalidating their sessions.
            Err(crate::project::store::StoreError::Conflict { .. }) => {
                let document = store
                    .read(path)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| "the session key vanished as it was written".to_string())?;
                users::from_hex_32(&document.text)
                    .map(SessionKey)
                    .ok_or_else(|| "the session key was written malformed".to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn sign(&self, user: &UserId, credential_version: u64, expires: i64) -> blake3::Hash {
        let payload = format!(
            "{SESSION_DOMAIN}|{}|{credential_version}|{expires}",
            user.as_str()
        );
        blake3::keyed_hash(&self.0, payload.as_bytes())
    }

    /// A cookie value for `user`, valid until `expires`.
    pub fn mint(&self, user: &UserId, credential_version: u64, expires: i64) -> String {
        let mac = self.sign(user, credential_version, expires);
        format!(
            "{}.{credential_version}.{expires}.{}",
            user.as_str(),
            mac.to_hex()
        )
    }

    /// The user and credential version a cookie attests to, if its signature
    /// holds and it has not expired.
    ///
    /// Returns nothing rather than a reason: every way this can fail looks
    /// the same to the caller, who is either a browser with a stale cookie
    /// or someone guessing.
    pub fn verify(&self, cookie: &str, now: i64) -> Option<(UserId, u64)> {
        // The user part is a slug, which cannot contain '.', so a fixed
        // four-way split is unambiguous.
        let mut parts = cookie.splitn(4, '.');
        let user = UserId::new(parts.next()?).ok()?;
        let credential_version: u64 = parts.next()?.parse().ok()?;
        let expires: i64 = parts.next()?.parse().ok()?;
        let presented = blake3::Hash::from_hex(parts.next()?).ok()?;

        // Signature first, then expiry: an expired cookie is still evidence
        // of a real login, and checking in this order keeps both branches
        // doing the same constant-time comparison.
        let expected = self.sign(&user, credential_version, expires);
        if presented != expected || now >= expires {
            return None;
        }
        Some((user, credential_version))
    }
}

/// `Set-Cookie` for a fresh session.
pub fn session_cookie(value: &str, ttl_seconds: i64) -> String {
    format!("{SESSION_COOKIE}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={ttl_seconds}")
}

/// `Set-Cookie` that clears the session.
pub fn cleared_cookie() -> String {
    format!("{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0")
}

/// The value of one cookie from a request's `Cookie` header.
fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value.to_string())
}

/// How a caller's identity was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentitySource {
    /// The project has no user file, so it has not opted into
    /// authentication. Everyone is the local default user, which is how
    /// every Ridal before #131 behaved and how a project keeps working after
    /// an upgrade.
    Unconfigured,
    /// A signed session cookie.
    Session,
    /// Nobody signed in, on a project that does have accounts.
    Anonymous,
}

/// Why a caller's effective role is lower than their account's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleCap {
    /// `--read-only`.
    ReadOnlyServer,
    /// The catalog is not a project, so there is nowhere to write.
    NotAProject,
}

/// Who is asking, and what they may do.
///
/// Resolved once per request by [`middleware`] and read from the request
/// extensions by every handler that needs it, so a route cannot accidentally
/// resolve it twice and get two answers.
#[derive(Debug, Clone)]
pub struct Caller {
    /// `None` for an anonymous reader, who has no preferences and can write
    /// nothing.
    pub user: Option<UserId>,
    /// What the account says, before any server-wide cap.
    pub account_role: Role,
    /// What this caller may actually do here and now.
    pub role: Role,
    pub download: DownloadScope,
    pub source: IdentitySource,
    /// Set when [`Self::role`] is below [`Self::account_role`], so a refusal
    /// can explain which of the two reasons it was.
    pub cap: Option<RoleCap>,
    /// Whether the project has any accounts at all. Decides whether an
    /// anonymous caller is offered a login or told there is nothing to log
    /// in to.
    pub authentication_configured: bool,
}

impl Caller {
    pub fn is_authenticated(&self) -> bool {
        self.user.is_some()
    }

    /// What to show as the caller's name.
    pub fn display_name(&self) -> &str {
        self.user.as_ref().map_or("anonymous", |u| u.as_str())
    }

    pub fn may(&self, needed: Role) -> bool {
        self.role >= needed
    }

    pub fn may_download(&self, needed: DownloadScope) -> bool {
        self.download >= needed
    }

    /// Refuse unless the caller is at least `needed`.
    ///
    /// The single place that decides between "log in and try again" and
    /// "this will never work", so no route has to get that right on its own:
    ///
    /// - no project at all -> `409`, because nothing about the request needs
    ///   fixing and no login would help;
    /// - `--read-only` -> `403`, naming the flag, because the operator of
    ///   the server is the one who can change it;
    /// - anonymous on a project with accounts -> `401`, which is the one
    ///   case where retrying after logging in works;
    /// - signed in but not permitted -> `403`, naming both roles.
    pub fn require(&self, needed: Role, action: &str) -> Result<(), ApiError> {
        if self.may(needed) {
            return Ok(());
        }
        match self.cap {
            Some(RoleCap::NotAProject) => Err(ApiError::conflict(
                "not_a_project",
                format!(
                    "This catalog is not a Ridal project, so there is nowhere to \
                     {action}. Run `ridal project init` in the directory you are \
                     serving, then restart."
                ),
            )),
            Some(RoleCap::ReadOnlyServer) => Err(ApiError::forbidden(
                "read_only",
                format!("This server was started read-only, so you cannot {action}."),
            )),
            None if !self.is_authenticated() => Err(ApiError::unauthorized(
                "authentication_required",
                format!("Sign in to {action}."),
            )),
            None => Err(ApiError::forbidden(
                "insufficient_role",
                format!(
                    "You are '{}' ({}), and {action} needs the '{needed}' role or above.",
                    self.display_name(),
                    self.role
                ),
            )),
        }
    }

    /// Refuse unless the caller's download scope reaches `needed`.
    pub fn require_download(&self, needed: DownloadScope, what: &str) -> Result<(), ApiError> {
        if self.may_download(needed) {
            return Ok(());
        }
        if !self.is_authenticated() && self.authentication_configured {
            return Err(ApiError::unauthorized(
                "authentication_required",
                format!("Sign in to download {what}."),
            ));
        }
        Err(ApiError::forbidden(
            "download_not_permitted",
            format!(
                "Downloading {what} needs the '{needed}' download scope, and \
                 '{}' has '{}'. An administrator sets this in Access settings.",
                self.display_name(),
                self.download
            ),
        ))
    }
}

/// Resolve who is asking.
///
/// Reads `users.json` per request. That is a small JSON file parsed on each
/// call, which is the cost of revocation taking effect immediately rather
/// than when a cookie ages out -- and it is the same order of work as the
/// interpretation documents these routes already read. A cache here would
/// have to be invalidated by the very writes it exists to notice.
pub fn resolve(state: &AppState, headers: &HeaderMap, now: i64) -> Caller {
    let Some(project) = state.project.as_ref() else {
        // A bare directory of radargrams. There is no store to read and
        // nothing to write, so every caller is a viewer.
        return Caller {
            user: None,
            account_role: Role::Viewer,
            role: Role::Viewer,
            download: DownloadScope::All,
            source: IdentitySource::Unconfigured,
            cap: Some(RoleCap::NotAProject),
            authentication_configured: false,
        };
    };

    // A user file that will not parse is treated as "authentication is on
    // and nobody matches", which fails closed. Falling back to the
    // unconfigured path would turn a damaged file into an open server.
    let configured = match users::read(project.documents()) {
        Ok(Some((set, _))) => Some(set),
        Ok(None) => None,
        Err(_) => Some(UserSet::default()),
    };

    let (user, account_role, download, source) = match &configured {
        None => (
            // The migration case, and `ridal gui`'s no-login-step case: the
            // local default user, with everything an unauthenticated Ridal
            // has always allowed. Not `admin`, because the access settings
            // it would unlock are about accounts this project does not have.
            UserId::new(crate::identity::DEFAULT_USER).ok(),
            Role::Operator,
            DownloadScope::All,
            IdentitySource::Unconfigured,
        ),
        Some(set) => match authenticated_user(state, set, headers, now) {
            Some(user) => (
                Some(user.name.clone()),
                user.role,
                user.download,
                IdentitySource::Session,
            ),
            None => (
                None,
                Role::Viewer,
                set.anonymous_download,
                IdentitySource::Anonymous,
            ),
        },
    };

    let (role, cap) = if state.read_only && account_role > Role::Viewer {
        (Role::Viewer, Some(RoleCap::ReadOnlyServer))
    } else {
        (account_role, None)
    };

    Caller {
        user,
        account_role,
        role,
        download,
        source,
        cap,
        authentication_configured: configured.is_some(),
    }
}

/// The account a request's session cookie attests to, if it still holds.
fn authenticated_user<'a>(
    state: &AppState,
    set: &'a UserSet,
    headers: &HeaderMap,
    now: i64,
) -> Option<&'a users::User> {
    let cookie = cookie_value(headers, SESSION_COOKIE)?;
    let key = state.session_key().ok()?;
    let (name, version) = key.verify(&cookie, now)?;
    let user = set.get(&name)?;
    // The revocation check. A cookie minted before a password change, a
    // demotion or a deletion names a version the account no longer has.
    (user.credential_version == version).then_some(user)
}

/// Paths reachable without a session even when the project requires one to
/// read.
///
/// The login page has to render, its form has to post, an invite has to be
/// redeemable by someone with no account yet, and the page needs its own CSS
/// -- a login screen that 401s its own stylesheet is not a login screen.
fn is_public_path(path: &str) -> bool {
    path == "/login"
        || path == "/favicon.ico"
        || path == "/api/v1/health"
        || path.starts_with("/static/")
        || path.starts_with("/invite/")
        || path.starts_with("/api/v1/auth/")
}

/// Resolve the caller once and hand it to every handler.
///
/// Also enforces the one policy that has to apply to whole pages rather than
/// to single routes: a project may require a login to read at all. Doing it
/// here rather than per route means a route added later cannot forget.
pub async fn middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let caller = resolve(&state, request.headers(), now());
    let path = request.uri().path().to_string();

    if requires_login_to_read(&state) && !caller.is_authenticated() && !is_public_path(&path) {
        return if path.starts_with("/api/") {
            ApiError::unauthorized(
                "authentication_required",
                "This project requires a login to read.",
            )
            .into_response()
        } else {
            // A person who followed a link wants the page, not a status
            // code. `next` is dropped rather than round-tripped: reflecting
            // a URL out of a request and back into a redirect is how open
            // redirects happen, and the catalog is one click from the login
            // page anyway.
            Redirect::to("/login").into_response()
        };
    }

    request.extensions_mut().insert(caller);
    next.run(request).await
}

fn requires_login_to_read(state: &AppState) -> bool {
    state
        .project
        .as_ref()
        .and_then(|project| users::read(project.documents()).ok().flatten())
        .is_some_and(|(set, _)| set.require_auth_to_read)
}

/// Seconds since the epoch.
pub fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl<S: Send + Sync> FromRequestParts<S> for Caller {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<Caller>().cloned().ok_or_else(|| {
            // Only reachable if a router is built without the middleware,
            // which is a programming error rather than a request fault --
            // and one that must fail closed rather than default to a
            // permissive caller.
            ApiError::internal(
                "caller_unresolved",
                "The request reached a handler without an identity. This is a bug \
                 in how the router was assembled.",
            )
        })
    }
}

/// Load or create the signing key for a project.
///
/// Lazy, so a project that never authenticates never grows a `session.key`.
pub fn project_session_key(project: &Project) -> Result<SessionKey, String> {
    SessionKey::load_or_create(project.documents())
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::*;
    use crate::project::store::DocumentStore;

    fn key() -> SessionKey {
        SessionKey([7u8; 32])
    }

    fn user(name: &str) -> UserId {
        UserId::new(name).unwrap()
    }

    #[test]
    fn a_minted_cookie_verifies_and_names_its_user() {
        let key = key();
        let now = 1_700_000_000;
        let cookie = key.mint(&user("erik"), 3, now + 100);

        let (name, version) = key.verify(&cookie, now).expect("must verify");
        assert_eq!(name.as_str(), "erik");
        assert_eq!(version, 3);
    }

    #[test]
    fn a_cookie_signed_with_another_key_is_refused() {
        // The whole point: the cookie is self-describing, so nothing but the
        // signature stops a client writing its own.
        let now = 1_700_000_000;
        let cookie = key().mint(&user("erik"), 1, now + 100);
        let other = SessionKey([9u8; 32]);
        assert!(other.verify(&cookie, now).is_none());
    }

    #[test]
    fn tampering_with_any_field_invalidates_the_cookie() {
        let key = key();
        let now = 1_700_000_000;
        let cookie = key.mint(&user("student"), 1, now + 100);
        let mac = cookie.rsplit_once('.').unwrap().1;

        for forged in [
            format!("erik.1.{}.{mac}", now + 100),
            format!("student.2.{}.{mac}", now + 100),
            format!("student.1.{}.{mac}", now + 10_000),
            format!("student.1.{}.{}", now + 100, "0".repeat(64)),
        ] {
            assert!(key.verify(&forged, now).is_none(), "{forged}");
        }
    }

    #[test]
    fn an_expired_cookie_is_refused() {
        let key = key();
        let expires = 1_700_000_000;
        let cookie = key.mint(&user("erik"), 1, expires);
        assert!(key.verify(&cookie, expires - 1).is_some());
        assert!(key.verify(&cookie, expires).is_none());
        assert!(key.verify(&cookie, expires + 1).is_none());
    }

    #[test]
    fn a_malformed_cookie_is_refused_rather_than_panicking() {
        let key = key();
        for junk in ["", ".", "erik", "erik.1", "erik.x.1.abc", "erik.1.1.nothex"] {
            assert!(key.verify(junk, 0).is_none(), "{junk}");
        }
    }

    #[test]
    fn the_session_key_persists_so_a_restart_does_not_sign_everyone_out() {
        // The reason a stateless cookie is the smaller option: surviving a
        // restart falls out of the key being persistent.
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());

        let first = SessionKey::load_or_create(&store).unwrap();
        let cookie = first.mint(&user("erik"), 1, 1_700_000_100);

        let second = SessionKey::load_or_create(&store).unwrap();
        assert!(
            second.verify(&cookie, 1_700_000_000).is_some(),
            "a reloaded key must still verify sessions it signed"
        );
    }

    #[test]
    #[cfg(unix)]
    fn the_session_key_is_written_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        SessionKey::load_or_create(&store).unwrap();

        let mode = std::fs::metadata(dir.path().join(SESSION_KEY_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "{mode:o}");
    }

    #[test]
    fn a_damaged_key_file_is_reported_rather_than_replaced() {
        // Replacing it would sign every session out and hide whatever
        // damaged it.
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join(SESSION_KEY_FILE), "not a key").unwrap();

        let error = SessionKey::load_or_create(&store).unwrap_err();
        assert!(error.contains("32-byte hex"), "{error}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join(SESSION_KEY_FILE)).unwrap(),
            "not a key",
            "the damaged file was overwritten"
        );
    }

    #[test]
    fn a_cookie_is_found_among_others_and_never_invented() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            "theme=dark; ridal_session=abc.1.2.def; other=x"
                .parse()
                .unwrap(),
        );
        assert_eq!(
            cookie_value(&headers, SESSION_COOKIE).as_deref(),
            Some("abc.1.2.def")
        );
        assert_eq!(cookie_value(&headers, "nothing"), None);
        assert_eq!(cookie_value(&HeaderMap::new(), SESSION_COOKIE), None);
    }

    #[test]
    fn the_cleared_cookie_expires_immediately_and_carries_no_value() {
        let cleared = cleared_cookie();
        assert!(cleared.contains("Max-Age=0"), "{cleared}");
        assert!(cleared.starts_with("ridal_session=;"), "{cleared}");
        assert!(cleared.contains("HttpOnly"), "{cleared}");
    }

    #[test]
    fn a_session_cookie_is_http_only_and_same_site_but_not_secure() {
        // Not `Secure`: Ridal essentially never sees HTTPS, so the attribute
        // would stop the cookie being sent over the loopback HTTP that both
        // `ridal gui` and every reverse-proxy deployment speak.
        let cookie = session_cookie("erik.1.2.abc", 60);
        assert!(cookie.contains("HttpOnly"), "{cookie}");
        assert!(cookie.contains("SameSite=Lax"), "{cookie}");
        assert!(!cookie.contains("Secure"), "{cookie}");
        assert!(cookie.contains("Max-Age=60"), "{cookie}");
    }

    #[test]
    fn the_login_page_and_its_assets_stay_reachable_without_one() {
        for path in [
            "/login",
            "/static/app.css",
            "/static/login.js",
            "/invite/abc123",
            "/api/v1/auth/login",
            "/favicon.ico",
        ] {
            assert!(is_public_path(path), "{path}");
        }
        for path in ["/", "/view/line-01", "/api/v1/datasets", "/settings"] {
            assert!(!is_public_path(path), "{path}");
        }
    }

    #[test]
    fn a_refusal_says_which_of_the_reasons_it_was() {
        let base = Caller {
            user: None,
            account_role: Role::Viewer,
            role: Role::Viewer,
            download: DownloadScope::All,
            source: IdentitySource::Anonymous,
            cap: None,
            authentication_configured: true,
        };

        // Anonymous, on a project with accounts: retrying after a login
        // works, which is the one case that is a 401.
        let error = base.require(Role::Picker, "save picks").unwrap_err();
        assert_eq!(error.status_code(), StatusCode::UNAUTHORIZED);

        let signed_in = Caller {
            user: Some(user("student")),
            account_role: Role::Picker,
            role: Role::Picker,
            ..base.clone()
        };
        assert!(signed_in.require(Role::Picker, "save picks").is_ok());
        let error = signed_in
            .require(Role::Operator, "edit layers")
            .unwrap_err();
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
        assert!(error.message().contains("operator"), "{}", error.message());

        let read_only = Caller {
            user: Some(user("erik")),
            account_role: Role::Admin,
            role: Role::Viewer,
            cap: Some(RoleCap::ReadOnlyServer),
            ..base.clone()
        };
        let error = read_only.require(Role::Picker, "save picks").unwrap_err();
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
        assert!(error.message().contains("read-only"), "{}", error.message());

        let bare = Caller {
            cap: Some(RoleCap::NotAProject),
            authentication_configured: false,
            ..base
        };
        let error = bare.require(Role::Picker, "save picks").unwrap_err();
        assert_eq!(error.status_code(), StatusCode::CONFLICT);
        assert!(
            error.message().contains("project init"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn download_scope_refusals_distinguish_anonymous_from_restricted() {
        let anonymous = Caller {
            user: None,
            account_role: Role::Viewer,
            role: Role::Viewer,
            download: DownloadScope::None,
            source: IdentitySource::Anonymous,
            cap: None,
            authentication_configured: true,
        };
        let error = anonymous
            .require_download(DownloadScope::Derived, "level 2 points")
            .unwrap_err();
        assert_eq!(error.status_code(), StatusCode::UNAUTHORIZED);

        let restricted = Caller {
            user: Some(user("student")),
            download: DownloadScope::Picks,
            ..anonymous
        };
        assert!(restricted
            .require_download(DownloadScope::Picks, "picks")
            .is_ok());
        let error = restricted
            .require_download(DownloadScope::All, "the radargram")
            .unwrap_err();
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
        assert!(
            error.message().contains("Access settings"),
            "{}",
            error.message()
        );
    }
}
