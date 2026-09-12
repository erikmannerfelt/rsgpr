//! Signing in, invites, accounts and personal preferences (#131).
//!
//! Split from [`super::interp_routes`] for the same reason that file was
//! split from [`super::routes`]: everything here shares one set of concerns
//! the others do not have. Nothing in this module may ever put a password
//! hash or an invite token hash into a response, including into an error
//! message, which is why every account that leaves here goes through
//! [`crate::project::users::User::redacted`].
//!
//! # Creating an account
//!
//! An administrator creates the account; its owner sets the password
//! through a one-time link:
//!
//! ```text
//! admin fills in name + role
//!   -> Ridal mints a one-time token and shows a URL once
//!   -> admin sends that URL however they like
//!   -> the user opens it, chooses a password, the token is consumed
//! ```
//!
//! So no password is ever known to two people, and there is no default
//! password to forget to change. Ridal sends nothing itself: keeping SMTP
//! configuration, credentials and deliverability out of the deployment is
//! worth one manual step for a tool whose projects have a handful of users.
//!
//! A password reset is the same flow. There is one mechanism, not two.
//!
//! # Bootstrapping
//!
//! The first administrator cannot be created through here -- there is no
//! administrator to authorise it. That happens from the command line on the
//! machine itself (`ridal project user add`), which is the one place where
//! access already implies authority.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse};
use axum::Json;

use super::app::AppState;
use super::auth::{self, Caller};
use super::routes::{ApiError, PageError};
use super::templates;
use crate::identity::UserId;
use crate::project::preferences::{self, Preferences};
use crate::project::store::Expectation;
use crate::project::users::{self, DownloadScope, Role, User, UserError, UserSet};
use crate::project::Project;

fn user_error(error: UserError) -> ApiError {
    match &error {
        UserError::Store(e) => ApiError::internal("store_failed", e.to_string()),
        UserError::Malformed { .. } => ApiError::internal("malformed_users", error.to_string()),
        UserError::Duplicate(_) => ApiError::conflict("user_exists", error.to_string()),
        UserError::NotFound(_) => ApiError::not_found("user_not_found", error.to_string()),
        UserError::Rejected(_) => ApiError::bad_request("rejected", error.to_string()),
        UserError::Hash(_) => ApiError::internal("hash_failed", error.to_string()),
    }
}

/// The project, or a 409 explaining that this catalog is not one.
fn project(state: &AppState) -> Result<&Project, ApiError> {
    state.project.as_ref().ok_or_else(|| {
        ApiError::conflict(
            "not_a_project",
            "This catalog is not a Ridal project, so it has no accounts. Run \
             `ridal project init` in the directory you are serving, then \
             restart.",
        )
    })
}

/// The accounts, or a 409 for a project that has not opted into
/// authentication.
///
/// A distinct answer from "no accounts exist": a project with no user file
/// is one where everyone is the local default user, and offering to sign in
/// to it would be offering something that cannot work.
fn accounts(state: &AppState) -> Result<UserSet, ApiError> {
    let project = project(state)?;
    users::read(project.documents())
        .map_err(user_error)?
        .map(|(set, _)| set)
        .ok_or_else(|| {
            ApiError::conflict(
                "no_accounts",
                "This project has no accounts, so there is nothing to sign in \
                 to. Create one with `ridal project user add <name> --role admin`.",
            )
        })
}

fn parse_role(raw: &str) -> Result<Role, ApiError> {
    Role::parse(raw).map_err(|e| ApiError::bad_request("invalid_role", e))
}

fn parse_download(raw: &str) -> Result<DownloadScope, ApiError> {
    DownloadScope::parse(raw).map_err(|e| ApiError::bad_request("invalid_download_scope", e))
}

fn parse_name(raw: &str) -> Result<UserId, ApiError> {
    UserId::new(raw).map_err(|e| ApiError::bad_request("invalid_user", e))
}

/// `GET /login`
pub async fn login_page(
    State(state): State<Arc<AppState>>,
    caller: Caller,
) -> Result<impl IntoResponse, PageError> {
    let env = templates::environment();
    let tmpl = env
        .get_template("login.html.jinja")
        .expect("login template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            // A project with no accounts cannot be signed in to, and the
            // page says so rather than offering a form that always fails.
            authentication_configured => caller.authentication_configured,
            already_signed_in => caller.is_authenticated(),
            current_user => caller.user.as_ref().map(|u| u.as_str()),
            current_role => caller.role.as_str(),
            project => state.project.is_some(),
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

#[derive(serde::Deserialize)]
pub struct LoginBody {
    name: String,
    password: String,
}

/// `POST /api/v1/auth/login`
///
/// One refusal for every way this can fail -- no such account, wrong
/// password, or an account whose invite has not been redeemed yet. Telling
/// them apart would turn the login form into a way to enumerate who exists
/// and who has not signed up yet.
pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(body): Json<LoginBody>,
) -> Result<impl IntoResponse, ApiError> {
    password_login_allowed(&state)?;
    let set = accounts(&state)?;
    let refused = || {
        ApiError::unauthorized(
            "invalid_credentials",
            "That name and password do not match an account. If you were sent \
             an invite link, open that instead -- it is what sets your password \
             the first time.",
        )
    };

    let user = UserId::new(&body.name).ok().and_then(|name| set.get(&name));

    // The identical *response* is only half of it: an unknown name would
    // otherwise return before Argon2id ran at all, while a real one paid
    // for a verification, and that difference is measurable from outside.
    // It turns the login form back into the account enumerator the unified
    // message was meant to close. So a miss verifies against a throwaway
    // hash instead of skipping the work.
    let verified = match user {
        Some(user) if user.is_activated() => users::verify_password(user, &body.password),
        // Two ways to get here, and both must cost what a real
        // verification costs: no such name, and an account whose invite
        // has not been redeemed. The second is the subtler one --
        // `verify_password` short-circuits when there is no stored hash,
        // so it returns in microseconds and says "this person exists but
        // has not signed up yet" to anyone holding a stopwatch.
        _ => {
            burn_a_verification(&body.password);
            false
        }
    };
    if !verified {
        return Err(refused());
    }
    let user = user.expect("verified implies an account");

    Ok((
        issue_session(&state, user)?,
        Json(serde_json::json!({
            "user": user.name.as_str(),
            "role": user.role.as_str(),
            "download": user.download.as_str(),
        })),
    ))
}

/// Do the work a real verification would, and discard the answer.
///
/// For the paths where there is no stored hash to check against -- an
/// unknown name, or an account whose invite has not been redeemed -- so
/// that "wrong password" and "no such person" cost the same.
///
/// The decoy hash is computed once per process from a random passphrase
/// rather than written into the source. A constant would be a hard-coded
/// credential in a binary that ships, and one whose Argon2 parameters
/// would silently stop matching the real ones the day the defaults change,
/// which is exactly when the timings would start to differ again.
#[cfg(feature = "server")]
fn burn_a_verification(candidate: &str) {
    static DECOY: std::sync::OnceLock<Option<User>> = std::sync::OnceLock::new();
    let decoy = DECOY.get_or_init(|| {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).ok()?;
        let hash = users::hash_password(&users::to_hex(&bytes)).ok()?;
        let mut user = User::new(
            UserId::new("decoy").ok()?,
            Role::Viewer,
            DownloadScope::None,
        );
        user.password_hash = Some(hash);
        Some(user)
    });
    if let Some(decoy) = decoy {
        // The result is deliberately unused: this exists for its cost.
        let _ = users::verify_password(decoy, candidate);
    }
}

/// Refuse a password on a bind where it would travel in the clear.
///
/// Checked per request rather than once at startup. "Does this project
/// have accounts" can become true while the server is running -- an
/// administrator created from the command line takes effect on the next
/// request -- so a public `--read-only` server that started with none
/// would otherwise begin accepting cleartext logins the moment one
/// appeared, which is precisely the sequence the guard exists for.
fn password_login_allowed(state: &AppState) -> Result<(), ApiError> {
    if state.access.allow_password_login {
        return Ok(());
    }
    Err(ApiError::forbidden(
        "insecure_transport",
        "This server is bound to a network address and does not terminate \
         TLS, so a password sent to it would travel in the clear. Reach it \
         through a TLS-terminating reverse proxy, or restart it with \
         --allow-insecure-login to accept that.",
    ))
}

/// The `Set-Cookie` header that signs `user` in.
fn issue_session(
    state: &AppState,
    user: &User,
) -> Result<[(header::HeaderName, String); 1], ApiError> {
    let key = state
        .session_key()
        .map_err(|e| ApiError::internal("session_key_failed", e))?;
    let ttl = auth::SESSION_TTL_DAYS * 24 * 60 * 60;
    let value = key.mint(&user.name, user.credential_version, auth::now() + ttl);
    Ok([(header::SET_COOKIE, auth::session_cookie(&value, ttl))])
}

/// `POST /api/v1/auth/logout`
///
/// Succeeds whether or not anyone was signed in. "Sign me out" has one
/// sensible outcome, and reporting that there was nothing to do would be a
/// failure the caller cannot act on.
pub async fn logout() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::SET_COOKIE, auth::cleared_cookie())],
        Json(serde_json::json!({ "signed_out": true })),
    )
}

/// `GET /api/v1/auth/me` -- who the server thinks is calling.
///
/// Exists so a page can render its controls from one request rather than
/// inferring permission from which other requests happen to fail.
pub async fn me(caller: Caller) -> impl IntoResponse {
    Json(serde_json::json!({
        "user": caller.user.as_ref().map(|u| u.as_str()),
        "role": caller.role.as_str(),
        "account_role": caller.account_role.as_str(),
        "download": caller.download.as_str(),
        "authenticated": caller.is_authenticated(),
        "authentication_configured": caller.authentication_configured,
        "can_pick": caller.may(Role::Picker),
        "can_operate": caller.may(Role::Operator),
        "can_administer": caller.may(Role::Admin),
    }))
}

/// `GET /invite/{token}`
///
/// Deliberately says nothing about whether the token is any good. The page
/// is reachable by anyone with the URL, and a page that reported "this
/// belongs to erik, but it expired" would leak an account name to whoever
/// found the link in an old chat log. Redemption reports the outcome.
pub async fn invite_page(Path(token): Path<String>) -> Result<impl IntoResponse, PageError> {
    let env = templates::environment();
    let tmpl = env
        .get_template("invite.html.jinja")
        .expect("invite template is always registered");
    let html = tmpl
        .render(minijinja::context! {
            token => token,
            min_password_len => users::MIN_PASSWORD_LEN,
        })
        .map_err(|e| PageError(ApiError::internal("template_error", e.to_string())))?;
    Ok(Html(html))
}

#[derive(serde::Deserialize)]
pub struct RedeemBody {
    token: String,
    password: String,
}

/// `POST /api/v1/auth/invite` -- set a password with a one-time token.
///
/// Redeeming consumes the token, sets the password and bumps the credential
/// version, so any older session for that account dies with it. That last
/// part is what makes an administrator-triggered reset actually take
/// someone's access away rather than merely changing what they would type
/// next time.
///
/// Signs the person in on success. They have just proved they hold the link
/// and chosen a password; making them type it again immediately would be
/// ceremony.
pub async fn redeem_invite(
    State(state): State<Arc<AppState>>,
    Json(body): Json<RedeemBody>,
) -> Result<impl IntoResponse, ApiError> {
    password_login_allowed(&state)?;
    let project = project(&state)?;
    users::check_password(&body.password).map_err(user_error)?;

    // The token is checked *before* the hash is computed, and checked again
    // inside the update below.
    //
    // The second check is the correct one -- it runs under the store's lock,
    // so two redemptions of the same link cannot both win. This first one is
    // purely about cost: Argon2id is deliberately expensive, and without it
    // an unauthenticated caller could spend a hash of the server's CPU per
    // request by posting a password with a token that was never valid. A
    // bare existence check costs a file read and a blake3.
    let stale_link = || {
        UserError::Rejected(
            "This link is not valid. It may already have been used, or it \
             may have expired -- ask an administrator for a new one."
                .to_string(),
        )
    };
    let (set, _) = users::read(project.documents())
        .map_err(user_error)?
        .ok_or_else(|| user_error(stale_link()))?;
    if users::user_for_invite(&set, &body.token, auth::now()).is_none() {
        return Err(user_error(stale_link()));
    }
    drop(set);

    // Hashed once, outside the update, because `update` retries on a version
    // conflict and Argon2id is deliberately expensive.
    let hash = users::hash_password(&body.password).map_err(user_error)?;

    let user = users::update(project.documents(), |set| {
        let name = users::user_for_invite(set, &body.token, auth::now())
            .map(|user| user.name.clone())
            .ok_or_else(stale_link)?;
        let user = set
            .get_mut(&name)
            .ok_or_else(|| UserError::NotFound(name.to_string()))?;
        user.password_hash = Some(hash.clone());
        user.invite = None;
        user.credential_version += 1;
        Ok(user.clone())
    })
    .map_err(user_error)?;

    Ok((
        issue_session(&state, &user)?,
        Json(serde_json::json!({
            "user": user.name.as_str(),
            "role": user.role.as_str(),
        })),
    ))
}

/// The project, if the caller may administer it.
fn admin_project<'a>(
    state: &'a AppState,
    caller: &Caller,
    action: &str,
) -> Result<&'a Project, ApiError> {
    caller.require(Role::Admin, action)?;
    project(state)
}

/// `GET /api/v1/users` -- every account, redacted.
pub async fn list_users(
    State(state): State<Arc<AppState>>,
    caller: Caller,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "see the accounts")?;
    let set = users::read(project.documents())
        .map_err(user_error)?
        .map(|(set, _)| set)
        .unwrap_or_default();

    Ok(Json(serde_json::json!({
        "users": set.users.iter().map(User::redacted).collect::<Vec<_>>(),
        "require_auth_to_read": set.require_auth_to_read,
        "anonymous_download": set.anonymous_download.as_str(),
        "roles": Role::ALL.map(Role::as_str),
        "download_scopes": DownloadScope::ALL.map(DownloadScope::as_str),
        "invite_ttl_days": users::INVITE_TTL_DAYS,
    })))
}

#[derive(serde::Deserialize)]
pub struct CreateUserBody {
    name: String,
    role: String,
    #[serde(default)]
    download: Option<String>,
}

/// `POST /api/v1/users` -- create an account and mint its invite.
///
/// The token is in the response and nowhere else: only its hash is stored,
/// so this is the one moment it exists. The page shows it once and says so.
///
/// The path, not a URL. Ridal is normally behind a reverse proxy and has no
/// reliable idea what address the browser reached it on -- building one from
/// the `Host` header would mean trusting a header a client controls. The
/// page prefixes it with its own origin, which is by definition the right
/// one.
pub async fn create_user(
    State(state): State<Arc<AppState>>,
    caller: Caller,
    Json(body): Json<CreateUserBody>,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "create an account")?;
    let name = parse_name(&body.name)?;
    let role = parse_role(&body.role)?;
    let download = match body.download.as_deref() {
        Some(value) => parse_download(value)?,
        None => DownloadScope::default(),
    };

    let (token, invite) = users::mint_invite(auth::now()).map_err(user_error)?;
    let created = users::update(project.documents(), |set| {
        if set.get(&name).is_some() {
            return Err(UserError::Duplicate(name.to_string()));
        }
        let mut user = User::new(name.clone(), role, download);
        user.invite = Some(invite.clone());
        set.users.push(user.clone());
        Ok(user)
    })
    .map_err(user_error)?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "user": created.redacted(),
            "invite_path": format!("/invite/{token}"),
            "invite_expires": invite.expires,
            "invite_ttl_days": users::INVITE_TTL_DAYS,
        })),
    ))
}

#[derive(serde::Deserialize)]
pub struct UpdateUserBody {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    download: Option<String>,
}

/// `PUT /api/v1/users/{name}` -- change a role, a download scope, or both.
///
/// Either change bumps the credential version, so it reaches an already
/// signed-in person on their next request rather than when their cookie
/// happens to age out. That matters most for the change that takes
/// something away.
pub async fn update_user(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    caller: Caller,
    Json(body): Json<UpdateUserBody>,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "change an account")?;
    let name = parse_name(&name)?;
    let role = body.role.as_deref().map(parse_role).transpose()?;
    let download = body.download.as_deref().map(parse_download).transpose()?;

    let updated = users::update(project.documents(), |set| {
        // Checked before the mutable borrow, and before anything changes:
        // an administrator who demotes the last administrator locks the
        // project's access settings away from everyone, including
        // themselves, with no way back short of editing users.json by hand.
        if role.is_some_and(|role| role < Role::Admin)
            && set.get(&name).is_some_and(|user| user.role == Role::Admin)
            && !set.has_another_admin(&name)
        {
            return Err(UserError::Rejected(format!(
                "'{name}' is the only administrator. Promote someone else first, \
                 or nobody will be able to manage accounts."
            )));
        }

        let user = set
            .get_mut(&name)
            .ok_or_else(|| UserError::NotFound(name.to_string()))?;
        let mut changed = false;
        if let Some(role) = role {
            changed |= user.role != role;
            user.role = role;
        }
        if let Some(download) = download {
            changed |= user.download != download;
            user.download = download;
        }
        if changed {
            user.credential_version += 1;
        }
        Ok(user.clone())
    })
    .map_err(user_error)?;

    Ok(Json(updated.redacted()))
}

/// `POST /api/v1/users/{name}/invite` -- reissue, for a reset or a lost link.
///
/// Replaces any outstanding invite rather than adding one, so "send another
/// link" cannot leave two live tokens for one account.
///
/// Does *not* bump the credential version. The account keeps working until
/// the new link is actually redeemed, which is what someone who merely lost
/// their invite email needs; revoking on issue would sign out a person who
/// had simply forgotten their password and then never finished the reset.
pub async fn reissue_invite(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    caller: Caller,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "reset a password")?;
    let name = parse_name(&name)?;
    let (token, invite) = users::mint_invite(auth::now()).map_err(user_error)?;

    users::update(project.documents(), |set| {
        let user = set
            .get_mut(&name)
            .ok_or_else(|| UserError::NotFound(name.to_string()))?;
        user.invite = Some(invite.clone());
        Ok(())
    })
    .map_err(user_error)?;

    Ok(Json(serde_json::json!({
        "user": name.as_str(),
        "invite_path": format!("/invite/{token}"),
        "invite_expires": invite.expires,
        "invite_ttl_days": users::INVITE_TTL_DAYS,
    })))
}

/// `DELETE /api/v1/users/{name}`
///
/// **Removes the account, never the picks.** An interpretation is attributed
/// scientific data, and the person who drew it leaving the project does not
/// make it less so -- the documents stay under `interpretations/`, still
/// named after them, still exportable. Their *preferences* do go, because a
/// preference is about a person rather than about the survey.
///
/// The deletion takes effect on their next request rather than when their
/// cookie expires: session verification looks the account up, and there is
/// no longer one to find.
pub async fn delete_user(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    caller: Caller,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "remove an account")?;
    let name = parse_name(&name)?;

    users::update(project.documents(), |set| {
        let Some(user) = set.get(&name) else {
            return Err(UserError::NotFound(name.to_string()));
        };
        if user.role == Role::Admin && !set.has_another_admin(&name) {
            return Err(UserError::Rejected(format!(
                "'{name}' is the only administrator. Promote someone else first, \
                 or nobody will be able to manage accounts."
            )));
        }
        set.users.retain(|user| user.name != name);
        Ok(())
    })
    .map_err(user_error)?;

    // Best effort, and after the account is gone: failing to tidy a
    // preferences file is not a reason to report that the deletion failed
    // when it did not.
    let _ = preferences::remove(project.documents(), &name);

    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Deserialize)]
pub struct AccessBody {
    #[serde(default)]
    require_auth_to_read: Option<bool>,
    #[serde(default)]
    anonymous_download: Option<String>,
}

/// `PUT /api/v1/access` -- the project-wide access policy.
///
/// Policy, not preference: two people cannot each decide whether the
/// catalog needs a login, so there is one value and no cascade.
pub async fn put_access(
    State(state): State<Arc<AppState>>,
    caller: Caller,
    Json(body): Json<AccessBody>,
) -> Result<impl IntoResponse, ApiError> {
    let project = admin_project(&state, &caller, "change the access settings")?;
    let anonymous_download = body
        .anonymous_download
        .as_deref()
        .map(parse_download)
        .transpose()?;

    let set = users::update(project.documents(), |set| {
        if let Some(require) = body.require_auth_to_read {
            set.require_auth_to_read = require;
        }
        if let Some(scope) = anonymous_download {
            set.anonymous_download = scope;
        }
        Ok(set.clone())
    })
    .map_err(user_error)?;

    Ok(Json(serde_json::json!({
        "require_auth_to_read": set.require_auth_to_read,
        "anonymous_download": set.anonymous_download.as_str(),
    })))
}

/// `GET /api/v1/preferences` -- the caller's own.
pub async fn get_preferences(
    State(state): State<Arc<AppState>>,
    caller: Caller,
) -> Result<impl IntoResponse, ApiError> {
    let (project, user) = my_preferences_target(&state, &caller)?;
    let preferences = preferences::read(project.documents(), user)
        .map_err(|e| ApiError::internal("preferences_read_failed", e.to_string()))?;
    Ok(Json(serde_json::json!({
        "user": user.as_str(),
        "render_profile": preferences.render_profile,
        "x_scale": preferences.x_scale,
    })))
}

#[derive(serde::Deserialize)]
pub struct PreferencesBody {
    /// `null` clears the preference, falling back to the project default.
    /// Absent means the same, because the page always sends both.
    #[serde(default)]
    render_profile: Option<String>,
    #[serde(default)]
    x_scale: Option<f64>,
}

/// `PUT /api/v1/preferences`
///
/// A `viewer` may do this. Setting how you like to look at something is not
/// a permission an administrator grants -- it is the floor of what having an
/// account means.
pub async fn put_preferences(
    State(state): State<Arc<AppState>>,
    caller: Caller,
    Json(body): Json<PreferencesBody>,
) -> Result<impl IntoResponse, ApiError> {
    let (project, user) = my_preferences_target(&state, &caller)?;

    // Validated here rather than in the store, for the same reason the
    // project's defaults are: which profiles exist and which scales the
    // viewer offers are server concepts, and storing one that nothing
    // renders would leave every page failing with no obvious cause.
    let render_profile = match body.render_profile.as_deref() {
        None | Some("") => None,
        Some(name) => {
            if super::render::profile::RenderProfile::by_name(name).is_none() {
                return Err(ApiError::bad_request(
                    "unknown_profile",
                    format!("There is no render profile called '{name}'."),
                ));
            }
            Some(name.to_string())
        }
    };
    let x_scale = match body.x_scale {
        None => None,
        Some(scale) => {
            if !super::routes::is_offered_x_scale(scale) {
                return Err(ApiError::bad_request(
                    "unknown_xscale",
                    format!("The viewer does not offer a horizontal scale of {scale}."),
                ));
            }
            // 1x is the neutral value, so choosing it means "no preference"
            // and leaves the key out of the document entirely -- which is
            // what lets a later project default reach this person.
            (scale != super::routes::DEFAULT_X_SCALE).then_some(scale)
        }
    };

    let stored = Preferences {
        render_profile,
        x_scale,
    };
    preferences::write(project.documents(), user, &stored, &Expectation::Any)
        .map_err(|e| ApiError::internal("preferences_write_failed", e.to_string()))?;

    Ok(Json(serde_json::json!({
        "user": user.as_str(),
        "render_profile": stored.render_profile,
        "x_scale": stored.x_scale,
    })))
}

/// Where the caller's own preferences live.
///
/// Anonymous readers have nowhere to keep any, which is a 401 rather than a
/// silent no-op: the page would otherwise show a saved setting that was
/// never saved.
fn my_preferences_target<'a, 'b>(
    state: &'a AppState,
    caller: &'b Caller,
) -> Result<(&'a Project, &'b UserId), ApiError> {
    let project = project(state)?;
    let user = caller.user.as_ref().ok_or_else(|| {
        ApiError::unauthorized(
            "authentication_required",
            "Sign in to keep your own settings. Without an account they have \
             nowhere to live, so the project's defaults apply and `?profile=` \
             overrides them for one page.",
        )
    })?;
    Ok((project, user))
}
