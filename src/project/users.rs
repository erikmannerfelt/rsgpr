//! Accounts, roles and download scopes (#131).
//!
//! A project's answer to "who may do what". Stored as a single `users.json`
//! document through [`super::store::DocumentStore`], which brings atomic
//! temp-file-plus-rename writes, content-hash versioning and a process-wide
//! write lock at no cost -- and, for this document specifically, a
//! [`write_private`](super::store::DocumentStore::write_private) that leaves
//! it readable only by its owner.
//!
//! # Two ladders, not one
//!
//! [`Role`] is what someone may *do*; [`DownloadScope`] is what they may
//! *take away*. Both are ladders, but independent ones: a picker who may not
//! export the underlying data and a viewer who may export everything are both
//! reasonable, and folding the second into the first would multiply the
//! roles.
//!
//! # Absent is not empty
//!
//! [`read`] answers `None` for a project with no user file at all, which is
//! every project that existed before this feature. That is the migration
//! case, and it means "this project has not opted into authentication" --
//! deliberately distinct from a file holding zero users, which means an
//! administrator removed the last account and the project is locked down
//! rather than open. The distinction is the whole reason upgrading Ridal
//! does not lock anyone out of their own data.
//!
//! # What never leaves this module
//!
//! Password hashes. [`User::redacted`] is what the HTTP layer serializes,
//! and it drops the hash and the invite. Nothing in a response, including an
//! error message, may carry either.

#![cfg_attr(
    not(feature = "server"),
    allow(
        dead_code,
        reason = "authentication is reached through the server's HTTP routes; \
                  a CLI-only build still needs the types to create the first \
                  administrator and to list who exists"
    )
)]

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::store::{DocumentStore, Expectation, StoreError, Version};
use crate::identity::UserId;

/// The document holding every account, relative to the project root.
pub const USERS_FILE: &str = "users.json";

/// How long an invite link is good for. Seven days is long enough to survive
/// a weekend and a forgotten message, and short enough that a link found in
/// an old chat log has expired.
pub const INVITE_TTL_DAYS: i64 = 7;

/// The shortest password the server will accept.
///
/// A length floor rather than a composition rule: "must contain a digit and a
/// symbol" pushes people towards `Password1!` while a longer passphrase is
/// both stronger and easier to remember. Argon2id does the rest.
pub const MIN_PASSWORD_LEN: usize = 10;

/// What someone may do. Each level includes the ones below it.
///
/// The ordering is the ladder, so a permission check is `role >= required`.
/// `Ord` is derived from declaration order, which is why the variants are
/// written weakest-first and must stay that way.
///
/// `editor` is deliberately absent for the [`Operator`](Role::Operator)
/// level: pickers also edit -- their own picks -- so the word would point at
/// the wrong thing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Read the catalog and open radargrams; read the layer vocabulary; set
    /// their own preferences.
    #[default]
    Viewer,
    /// Write their own interpretation.
    Picker,
    /// Run processing; modify radargram metadata; modify the layer
    /// vocabulary; set the project-wide defaults and policy.
    Operator,
    /// Users, roles, download scopes, and the access settings.
    Admin,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::Viewer, Role::Picker, Role::Operator, Role::Admin];

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Viewer => "viewer",
            Role::Picker => "picker",
            Role::Operator => "operator",
            Role::Admin => "admin",
        }
    }

    /// Parse a role from a command-line argument or an API body.
    pub fn parse(value: &str) -> Result<Role, String> {
        Role::ALL
            .into_iter()
            .find(|r| r.as_str() == value)
            .ok_or_else(|| {
                format!(
                    "Unknown role '{value}'. Choose one of: {}.",
                    Role::ALL.map(Role::as_str).join(", ")
                )
            })
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What someone may take off the server. Also a ladder.
///
/// The order earns itself: level 1 picks are in image space -- trace and
/// sample indices, no coordinates -- so they reveal less than level 2, which
/// carries positions and depths. Anyone allowed the point product has no
/// reason to be denied the picks it was derived from.
///
/// **Not a security boundary against a determined reader.** The rendered
/// image and the track are already on the page for anyone who can open it:
/// the viewer draws a radargram from 256x256 chunks over HTTP, and the
/// catalog's maps draw the track. This gates the bulk *download* endpoints,
/// which stops casual export and states an intent; it does not stop someone
/// with read access from reassembling what they can already see. That
/// sentence belongs next to the control in the UI, not only here.
///
/// Defaults to [`All`](DownloadScope::All), which is what every Ridal server
/// did before this existed. The control is there to restrict deliberately,
/// so an upgrade must not quietly take downloads away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadScope {
    /// Nothing leaves the server.
    None,
    /// Raw picks (level 1 gprinterp).
    Picks,
    /// Level 2 points; the rendered radargram image. Usually enough to
    /// publish with, which is why this is the line a project is really
    /// deciding about.
    Derived,
    /// The radargram NetCDF; the track.
    #[default]
    All,
}

impl DownloadScope {
    pub const ALL: [DownloadScope; 4] = [
        DownloadScope::None,
        DownloadScope::Picks,
        DownloadScope::Derived,
        DownloadScope::All,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            DownloadScope::None => "none",
            DownloadScope::Picks => "picks",
            DownloadScope::Derived => "derived",
            DownloadScope::All => "all",
        }
    }

    pub fn parse(value: &str) -> Result<DownloadScope, String> {
        DownloadScope::ALL
            .into_iter()
            .find(|s| s.as_str() == value)
            .ok_or_else(|| {
                format!(
                    "Unknown download scope '{value}'. Choose one of: {}.",
                    DownloadScope::ALL.map(DownloadScope::as_str).join(", ")
                )
            })
    }
}

impl fmt::Display for DownloadScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A pending invitation to set a password.
///
/// Stateful rather than a signed stateless token, because it must be
/// single-use and revocable, and a signature cannot be consumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invite {
    /// `blake3` of the token, hex. The token itself is shown once and never
    /// stored, so a leaked `users.json` does not grant account takeover.
    pub token_hash: String,
    /// Unix seconds after which the link stops working.
    pub expires: i64,
}

impl Invite {
    pub fn is_valid_at(&self, now: i64) -> bool {
        now < self.expires
    }
}

/// One account.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct User {
    pub name: UserId,
    #[serde(default)]
    pub role: Role,
    #[serde(default)]
    pub download: DownloadScope,
    /// Argon2id, as a PHC string. `None` until an invite is redeemed, which
    /// is the state a freshly created account is in -- there is deliberately
    /// no default password to forget to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password_hash: Option<String>,
    /// Bumped whenever anything a live session depends on changes: the
    /// password, the role, the download scope. Sessions carry the version
    /// they were minted at, so a demotion or a password reset takes effect
    /// immediately instead of when the cookie happens to expire.
    #[serde(default)]
    pub credential_version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invite: Option<Invite>,
    /// RFC3339, for the access table. Cosmetic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

impl User {
    /// A new account: no password, no invite yet, version 1.
    ///
    /// Starting at 1 rather than 0 so that "never had a credential" and
    /// "credential version 0" cannot be confused in a cookie.
    pub fn new(name: UserId, role: Role, download: DownloadScope) -> Self {
        Self {
            name,
            role,
            download,
            password_hash: None,
            credential_version: 1,
            invite: None,
            created: Some(chrono::Utc::now().to_rfc3339()),
        }
    }

    /// Whether this account can be logged into at all. An account whose
    /// invite has not been redeemed has no password to check.
    pub fn is_activated(&self) -> bool {
        self.password_hash.is_some()
    }

    /// What the HTTP layer is allowed to serialize: names, roles, scopes and
    /// whether an invite is outstanding. Never the hash, never the invite
    /// token hash.
    pub fn redacted(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name.as_str(),
            "role": self.role.as_str(),
            "download": self.download.as_str(),
            "activated": self.is_activated(),
            "invite_pending": self.invite.is_some(),
            "invite_expires": self.invite.as_ref().map(|i| i.expires),
            "created": self.created,
        })
    }
}

/// Every account, plus the project-wide access policy.
///
/// Policy lives here rather than in `ridal.toml` for the same reason the
/// hashes do: that file is hand-editable, lives in the project directory and
/// is the kind of thing people commit to git. It is also one atomic write --
/// changing a role and the read policy together cannot half-apply.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UserSet {
    /// Require a login to read the catalog at all. Public read otherwise,
    /// which is the default.
    #[serde(default)]
    pub require_auth_to_read: bool,
    /// What an anonymous reader may download when reading is public.
    ///
    /// Defaults to [`DownloadScope::All`], preserving what an unauthenticated
    /// Ridal has always served. Lowering it is the deliberate act.
    #[serde(default)]
    pub anonymous_download: DownloadScope,
    #[serde(default)]
    pub users: Vec<User>,
}

impl UserSet {
    pub fn get(&self, name: &UserId) -> Option<&User> {
        self.users.iter().find(|u| &u.name == name)
    }

    pub fn get_mut(&mut self, name: &UserId) -> Option<&mut User> {
        self.users.iter_mut().find(|u| &u.name == name)
    }

    /// Whether anyone can still administer the project.
    ///
    /// Used to refuse the two edits that would lock everyone out: demoting
    /// the last administrator, and deleting them.
    pub fn has_another_admin(&self, excluding: &UserId) -> bool {
        self.users
            .iter()
            .any(|u| u.role == Role::Admin && &u.name != excluding)
    }
}

#[derive(Debug)]
pub enum UserError {
    Store(StoreError),
    /// The document on disk is not a user file.
    Malformed {
        path: PathBuf,
        message: String,
    },
    /// An account with that name already exists.
    Duplicate(String),
    /// No account with that name.
    NotFound(String),
    /// The password is too short, or the invite is expired, consumed or
    /// simply wrong. Deliberately one variant with a caller-supplied message:
    /// the HTTP layer must not tell an attacker which of those it was.
    Rejected(String),
    /// Hashing or verification failed for a reason that is not the user's
    /// fault -- a corrupt stored hash, or a parameter the crate refused.
    Hash(String),
}

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UserError::Store(e) => write!(f, "{e}"),
            UserError::Malformed { path, message } => {
                write!(f, "{} is not a valid user file: {message}", path.display())
            }
            UserError::Duplicate(name) => write!(f, "there is already a user called '{name}'"),
            UserError::NotFound(name) => write!(f, "there is no user called '{name}'"),
            UserError::Rejected(message) => f.write_str(message),
            UserError::Hash(message) => write!(f, "could not process the password: {message}"),
        }
    }
}

impl std::error::Error for UserError {}

impl From<StoreError> for UserError {
    fn from(error: StoreError) -> Self {
        UserError::Store(error)
    }
}

fn relative_path() -> PathBuf {
    PathBuf::from(USERS_FILE)
}

/// Read the user file.
///
/// `Ok(None)` means the project has no user file, which is not an error: it
/// is every project that predates authentication, and it must keep behaving
/// exactly as it did. See the module doc on why that differs from a file
/// holding no users.
pub fn read(store: &DocumentStore) -> Result<Option<(UserSet, Version)>, UserError> {
    let path = relative_path();
    let Some(document) = store.read(&path)? else {
        return Ok(None);
    };
    let set: UserSet = serde_json::from_str(&document.text).map_err(|e| UserError::Malformed {
        path: store.root().join(&path),
        message: e.to_string(),
    })?;
    Ok(Some((set, document.version)))
}

/// Whether this project has opted into authentication.
pub fn is_configured(store: &DocumentStore) -> Result<bool, UserError> {
    Ok(store.read(&relative_path())?.is_some())
}

/// Write the user file, restricted to its owner.
pub fn write(
    store: &DocumentStore,
    set: &UserSet,
    expected: &Expectation,
) -> Result<Version, UserError> {
    let text = serde_json::to_string_pretty(set)
        .map_err(|e| UserError::Hash(e.to_string()))
        .map(|mut text| {
            text.push('\n');
            text
        })?;
    Ok(store.write_private(&relative_path(), &text, expected)?)
}

/// Read, modify, write -- conditional on the version that was read.
///
/// Every mutation goes through here so none of them can be written as a
/// blind overwrite. The store's lock serialises the writes but not the
/// read-modify-write around them, so two administrators saving at once would
/// otherwise silently discard one of the changes.
///
/// Retried once on a version conflict, because the conflicting write is
/// almost always this same process's other request rather than a human's,
/// and a failed role change is a poor thing to hand back for that.
pub fn update<T>(
    store: &DocumentStore,
    change: impl Fn(&mut UserSet) -> Result<T, UserError>,
) -> Result<T, UserError> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        let (mut set, expectation) = match read(store)? {
            Some((set, version)) => (set, Expectation::Version(version)),
            None => (UserSet::default(), Expectation::Absent),
        };
        let outcome = change(&mut set)?;
        match write(store, &set, &expectation) {
            Ok(_) => return Ok(outcome),
            Err(UserError::Store(StoreError::Conflict { .. })) if attempts < 3 => continue,
            Err(e) => return Err(e),
        }
    }
}

/// 32 random bytes, hex, shown once.
///
/// Only its `blake3` hash is stored, so the URL an administrator hands over
/// is as sensitive as a password while it is valid and worthless afterwards.
pub fn mint_invite(now: i64) -> Result<(String, Invite), UserError> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|e| UserError::Hash(format!("could not read system randomness: {e}")))?;
    let token = to_hex(&bytes);
    let invite = Invite {
        token_hash: blake3::hash(token.as_bytes()).to_hex().to_string(),
        expires: now + INVITE_TTL_DAYS * 24 * 60 * 60,
    };
    Ok((token, invite))
}

/// The account a presented invite token belongs to, if the token is live.
///
/// Compared as a [`blake3::Hash`], whose equality is documented to be
/// constant-time -- and which deliberately omits `Deref`/`AsRef` so that
/// property cannot be lost to an implicit conversion into a byte slice.
pub fn user_for_invite<'a>(set: &'a UserSet, token: &str, now: i64) -> Option<&'a User> {
    let presented = blake3::hash(token.as_bytes());
    set.users.iter().find(|user| {
        user.invite.as_ref().is_some_and(|invite| {
            invite.is_valid_at(now)
                && blake3::Hash::from_hex(&invite.token_hash).is_ok_and(|stored| stored == presented)
        })
    })
}

/// Lowercase hex, without pulling in an encoding crate for 32 bytes.
fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Refuse a password that is too short to be worth hashing.
pub fn check_password(password: &str) -> Result<(), UserError> {
    // Counted in characters rather than bytes, so a passphrase in a
    // non-Latin script is not judged long for the wrong reason.
    if password.chars().count() < MIN_PASSWORD_LEN {
        return Err(UserError::Rejected(format!(
            "A password must be at least {MIN_PASSWORD_LEN} characters. A short \
             sentence is easier to remember than a short password and harder to \
             guess."
        )));
    }
    Ok(())
}

/// Hash a password with Argon2id at the crate's default parameters.
///
/// Defaults deliberately: the RustCrypto defaults track the current OWASP
/// recommendation, and a hand-tuned cost here would be a number nobody
/// revisits. `hash_password` draws its own per-password salt from the
/// operating system.
#[cfg(feature = "server")]
pub fn hash_password(password: &str) -> Result<String, UserError> {
    use argon2::password_hash::phc::PasswordHash;
    use argon2::password_hash::PasswordHasher;

    check_password(password)?;
    let hash: PasswordHash = argon2::Argon2::default()
        .hash_password(password.as_bytes())
        .map_err(|e| UserError::Hash(e.to_string()))?;
    Ok(hash.to_string())
}

/// Whether `password` matches the account's stored hash.
///
/// An account with no hash -- one whose invite has not been redeemed --
/// answers `false` rather than erroring, so the login route cannot be used to
/// tell an un-activated account apart from a wrong password.
#[cfg(feature = "server")]
pub fn verify_password(user: &User, password: &str) -> bool {
    use argon2::password_hash::phc::PasswordHash;
    use argon2::password_hash::PasswordVerifier;

    let Some(stored) = user.password_hash.as_deref() else {
        return false;
    };
    let Ok(parsed) = PasswordHash::new(stored) else {
        return false;
    };
    argon2::Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn store() -> (tempfile::TempDir, DocumentStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        (dir, store)
    }

    fn user(name: &str, role: Role) -> User {
        User::new(UserId::new(name).unwrap(), role, DownloadScope::All)
    }

    #[test]
    fn roles_form_a_ladder_weakest_first() {
        // Permission checks are written as `role >= required`, so the
        // derived ordering is load-bearing rather than incidental.
        assert!(Role::Admin > Role::Operator);
        assert!(Role::Operator > Role::Picker);
        assert!(Role::Picker > Role::Viewer);
        assert!(Role::Admin >= Role::Admin);
    }

    #[test]
    fn download_scopes_form_a_ladder_and_default_to_all() {
        assert!(DownloadScope::All > DownloadScope::Derived);
        assert!(DownloadScope::Derived > DownloadScope::Picks);
        assert!(DownloadScope::Picks > DownloadScope::None);
        // Today's behaviour: an upgrade must not take downloads away.
        assert_eq!(DownloadScope::default(), DownloadScope::All);
    }

    #[test]
    fn roles_and_scopes_round_trip_through_their_wire_names() {
        for role in Role::ALL {
            assert_eq!(Role::parse(role.as_str()).unwrap(), role);
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, format!("\"{}\"", role.as_str()));
            assert_eq!(serde_json::from_str::<Role>(&json).unwrap(), role);
        }
        for scope in DownloadScope::ALL {
            assert_eq!(DownloadScope::parse(scope.as_str()).unwrap(), scope);
            let json = serde_json::to_string(&scope).unwrap();
            assert_eq!(json, format!("\"{}\"", scope.as_str()));
            assert_eq!(serde_json::from_str::<DownloadScope>(&json).unwrap(), scope);
        }
        assert!(Role::parse("editor").is_err());
        assert!(DownloadScope::parse("everything").is_err());
    }

    #[test]
    fn a_project_with_no_user_file_reads_as_none_not_as_an_empty_set() {
        // The migration rule in one assertion: absent means "has not opted
        // into authentication", which is not the same as "has no users".
        let (_dir, store) = store();
        assert!(read(&store).unwrap().is_none());
        assert!(!is_configured(&store).unwrap());

        write(&store, &UserSet::default(), &Expectation::Absent).unwrap();
        let (set, _) = read(&store).unwrap().expect("now configured");
        assert!(set.users.is_empty());
        assert!(is_configured(&store).unwrap());
    }

    #[test]
    fn a_new_account_has_no_password_until_an_invite_is_redeemed() {
        let account = user("erik", Role::Admin);
        assert!(!account.is_activated());
        assert_eq!(account.credential_version, 1);
        assert!(account.invite.is_none());
    }

    #[test]
    #[cfg(unix)]
    fn the_user_file_is_written_readable_only_by_its_owner() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, store) = store();
        write(&store, &UserSet::default(), &Expectation::Absent).unwrap();

        let mode = std::fs::metadata(dir.path().join(USERS_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "{mode:o}");
    }

    #[test]
    fn update_applies_a_change_conditionally_on_what_it_read() {
        let (_dir, store) = store();
        update(&store, |set| {
            set.users.push(user("erik", Role::Admin));
            Ok(())
        })
        .unwrap();
        update(&store, |set| {
            set.users.push(user("student", Role::Picker));
            Ok(())
        })
        .unwrap();

        let (set, _) = read(&store).unwrap().unwrap();
        assert_eq!(set.users.len(), 2, "the second write kept the first");
        assert_eq!(set.get(&UserId::new("erik").unwrap()).unwrap().role, Role::Admin);
    }

    #[test]
    fn an_update_that_refuses_leaves_the_file_alone() {
        let (_dir, store) = store();
        update(&store, |set| {
            set.users.push(user("erik", Role::Admin));
            Ok(())
        })
        .unwrap();

        let error = update(&store, |_| {
            Err::<(), _>(UserError::Duplicate("erik".to_string()))
        })
        .unwrap_err();
        assert!(matches!(error, UserError::Duplicate(_)), "{error}");

        let (set, _) = read(&store).unwrap().unwrap();
        assert_eq!(set.users.len(), 1);
    }

    #[test]
    fn an_invite_stores_only_a_hash_and_expires() {
        let now = 1_700_000_000;
        let (token, invite) = mint_invite(now).unwrap();

        assert_eq!(token.len(), 64, "32 random bytes, hex");
        assert_ne!(invite.token_hash, token, "the token itself is never stored");
        assert!(invite.is_valid_at(now));
        assert!(invite.is_valid_at(now + INVITE_TTL_DAYS * 86_400 - 1));
        assert!(!invite.is_valid_at(now + INVITE_TTL_DAYS * 86_400));

        // Two mints never collide, which is what makes a link single-use
        // per account rather than per project.
        let (other, _) = mint_invite(now).unwrap();
        assert_ne!(token, other);
    }

    #[test]
    fn an_invite_token_finds_its_account_and_nothing_else() {
        let now = 1_700_000_000;
        let mut set = UserSet::default();
        let (token, invite) = mint_invite(now).unwrap();
        let mut account = user("erik", Role::Admin);
        account.invite = Some(invite);
        set.users.push(account);
        set.users.push(user("student", Role::Picker));

        assert_eq!(
            user_for_invite(&set, &token, now).map(|u| u.name.as_str()),
            Some("erik")
        );
        assert!(user_for_invite(&set, "not-a-token", now).is_none());
        // Expiry is checked here rather than left to the caller, so no route
        // can forget to.
        assert!(user_for_invite(&set, &token, now + INVITE_TTL_DAYS * 86_400).is_none());
        // And an account with no invite is never matched by an empty one.
        assert!(user_for_invite(&set, "", now).is_none());
    }

    #[test]
    fn a_short_password_is_refused_with_a_reason() {
        let error = check_password("short").unwrap_err();
        assert!(error.to_string().contains("at least"), "{error}");
        check_password("correct horse battery").unwrap();
    }

    #[test]
    fn the_last_administrator_is_recognisable() {
        let mut set = UserSet::default();
        let erik = UserId::new("erik").unwrap();
        set.users.push(user("erik", Role::Admin));
        set.users.push(user("student", Role::Picker));
        assert!(!set.has_another_admin(&erik));

        set.users.push(user("co-lead", Role::Admin));
        assert!(set.has_another_admin(&erik));
    }

    #[test]
    fn a_redacted_user_carries_no_secret() {
        let mut account = user("erik", Role::Admin);
        account.password_hash = Some("$argon2id$v=19$secret".to_string());
        account.invite = Some(Invite {
            token_hash: "deadbeef".to_string(),
            expires: 1,
        });

        let json = serde_json::to_string(&account.redacted()).unwrap();
        assert!(!json.contains("argon2"), "{json}");
        assert!(!json.contains("deadbeef"), "{json}");
        assert!(json.contains("\"activated\":true"), "{json}");
        assert!(json.contains("\"invite_pending\":true"), "{json}");
    }

    #[test]
    fn an_unknown_field_does_not_stop_an_older_ridal_reading_the_file() {
        // Forward compatibility, matching how ridal.toml is parsed: a file
        // written by a newer Ridal must not lock this one out of its own
        // project.
        let (_dir, store) = store();
        store
            .write(
                Path::new(USERS_FILE),
                r#"{"users":[{"name":"erik","role":"admin","future_field":1}]}"#,
                &Expectation::Absent,
            )
            .unwrap();
        let (set, _) = read(&store).unwrap().unwrap();
        assert_eq!(set.users[0].role, Role::Admin);
        // Absent keys take their defaults rather than failing the parse.
        assert_eq!(set.users[0].download, DownloadScope::All);
        assert!(!set.require_auth_to_read);
    }

    #[test]
    fn a_user_name_that_is_not_a_slug_is_refused_when_read_back() {
        // The name is a path component elsewhere (preferences/{user}.json),
        // so a hand-edited file must not be able to smuggle one in.
        let (_dir, store) = store();
        store
            .write(
                Path::new(USERS_FILE),
                r#"{"users":[{"name":"../../etc","role":"admin"}]}"#,
                &Expectation::Any,
            )
            .unwrap();
        assert!(matches!(
            read(&store),
            Err(UserError::Malformed { .. })
        ));
    }

    #[cfg(feature = "server")]
    #[test]
    fn a_hashed_password_verifies_and_a_wrong_one_does_not() {
        let mut account = user("erik", Role::Admin);
        account.password_hash = Some(hash_password("correct horse battery").unwrap());

        assert!(verify_password(&account, "correct horse battery"));
        assert!(!verify_password(&account, "correct horse batterz"));
        assert!(!verify_password(&account, ""));
    }

    #[cfg(feature = "server")]
    #[test]
    fn the_same_password_hashes_differently_every_time() {
        // Per-account salts: two people who choose the same passphrase must
        // not be visibly identical in the file.
        let first = hash_password("correct horse battery").unwrap();
        let second = hash_password("correct horse battery").unwrap();
        assert_ne!(first, second);
        assert!(first.starts_with("$argon2id$"), "{first}");
    }

    #[cfg(feature = "server")]
    #[test]
    fn an_account_with_no_password_never_verifies() {
        // The un-activated case. It must answer like a wrong password rather
        // than erroring, so a login attempt cannot be used to discover which
        // accounts are still waiting on their invite.
        let account = user("erik", Role::Admin);
        assert!(!verify_password(&account, ""));
        assert!(!verify_password(&account, "anything at all"));
    }

    #[cfg(feature = "server")]
    #[test]
    fn a_corrupt_stored_hash_fails_closed() {
        let mut account = user("erik", Role::Admin);
        account.password_hash = Some("not a PHC string".to_string());
        assert!(!verify_password(&account, "anything"));
    }

    #[cfg(feature = "server")]
    #[test]
    fn hashing_refuses_a_password_the_checker_would_refuse() {
        // Otherwise an API that forgot to call `check_password` first would
        // silently store a hash of "a".
        assert!(matches!(
            hash_password("short"),
            Err(UserError::Rejected(_))
        ));
    }
}
