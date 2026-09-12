//! What one person likes to look at a project through (#131).
//!
//! ```text
//! preferences/<user>.json
//! ```
//!
//! # Why these are not project settings
//!
//! The render profile and the horizontal scale were built as project-wide
//! settings, and that was a modelling error. They are not policy about the
//! project -- they are preferences about how one person likes to look at it.
//! Two people should be able to disagree, and neither should need permission
//! to.
//!
//! The test when adding one: *could two users reasonably want different
//! answers?* If yes it is a preference and belongs here, cascading
//! request -> user -> project -> built-in. If a disagreement would be
//! incoherent -- two people cannot each decide whether the catalog needs a
//! login -- it is policy, and belongs in [`super::users::UserSet`] or
//! `ridal.toml` instead.
//!
//! # Why a document of their own
//!
//! Not a field in `users.json`. Credentials are written rarely and read on
//! every request; preferences are the opposite. Keeping them apart means
//! changing your default profile does not rewrite the file holding
//! everyone's password hashes.
//!
//! Anonymous readers have no user layer, since there is nowhere to keep one.
//! They fall through to the project default and can still override per page
//! with `?profile=`. Preferences are one more thing logging in gets you,
//! rather than a reason to require it.

#![cfg_attr(
    not(feature = "server"),
    allow(
        dead_code,
        reason = "preferences are set through the browser; a CLI-only build \
                  still needs the types to inspect a project"
    )
)]

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::store::{DocumentStore, Expectation, StoreError, Version};
use crate::identity::UserId;

pub const SUFFIX: &str = ".json";

/// One user's viewing preferences.
///
/// Every field is optional, and absent means "no preference" rather than a
/// stored copy of the default. That is what lets an administrator change the
/// project default and have it reach exactly the people who never chose.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Preferences {
    /// Render profile. A plain string for the same reason the project's is:
    /// which profiles exist is a server concept, checked at the HTTP
    /// boundary where a bad value can be refused with a useful message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_profile: Option<String>,
    /// Horizontal stretch, as a multiplier. `None` means no preference,
    /// which falls through to the project's and then to 1x.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_scale: Option<f64>,
}

#[derive(Debug)]
pub enum PreferencesError {
    Store(StoreError),
    Malformed { path: PathBuf, message: String },
}

impl std::fmt::Display for PreferencesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreferencesError::Store(e) => write!(f, "{e}"),
            PreferencesError::Malformed { path, message } => write!(
                f,
                "{} is not a valid preferences document: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for PreferencesError {}

impl From<StoreError> for PreferencesError {
    fn from(e: StoreError) -> Self {
        PreferencesError::Store(e)
    }
}

fn path_of(user: &UserId) -> PathBuf {
    PathBuf::from(super::PREFERENCES_DIR).join(format!("{}{SUFFIX}", user.as_str()))
}

/// Read one user's preferences. A user who has never saved any reads as the
/// default -- all unset -- rather than as an absence the caller must handle.
pub fn read(store: &DocumentStore, user: &UserId) -> Result<Preferences, PreferencesError> {
    let relative = path_of(user);
    let Some(stored) = store.read(&relative)? else {
        return Ok(Preferences::default());
    };
    serde_json::from_str(&stored.text).map_err(|e| PreferencesError::Malformed {
        path: store.root().join(&relative),
        message: e.to_string(),
    })
}

/// Read one user's preferences without failing on a malformed document.
///
/// For the request path, where a hand-broken preferences file should cost
/// that person their profile choice rather than every page they open. The
/// settings page uses [`read`] instead, so the fault is visible where it can
/// be fixed.
pub fn read_lenient(store: &DocumentStore, user: &UserId) -> Preferences {
    read(store, user).unwrap_or_default()
}

/// Replace one user's preferences.
pub fn write(
    store: &DocumentStore,
    user: &UserId,
    preferences: &Preferences,
    expected: &Expectation,
) -> Result<Version, PreferencesError> {
    let mut text =
        serde_json::to_string_pretty(preferences).map_err(|e| PreferencesError::Malformed {
            path: store.root().join(path_of(user)),
            message: e.to_string(),
        })?;
    text.push('\n');
    Ok(store.write(&path_of(user), &text, expected)?)
}

/// Forget one user's preferences. Returns whether there were any.
///
/// Called when an account is deleted: a preference is about a person, and
/// keeping it would be the one piece of a departed account that lingers.
/// Their *picks* are deliberately not removed -- those are attributed
/// scientific data, and the account going away must not destroy them.
pub fn remove(store: &DocumentStore, user: &UserId) -> Result<bool, PreferencesError> {
    Ok(store.remove(&path_of(user), &Expectation::Any)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, DocumentStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        (dir, store)
    }

    fn user(name: &str) -> UserId {
        UserId::new(name).unwrap()
    }

    #[test]
    fn a_user_who_has_never_saved_reads_as_unset() {
        let (_dir, store) = store();
        let preferences = read(&store, &user("erik")).unwrap();
        assert_eq!(preferences, Preferences::default());
        assert!(preferences.render_profile.is_none());
        assert!(preferences.x_scale.is_none());
    }

    #[test]
    fn preferences_round_trip_and_are_per_user() {
        let (_dir, store) = store();
        write(
            &store,
            &user("erik"),
            &Preferences {
                render_profile: Some("abslog".to_string()),
                x_scale: Some(2.0),
            },
            &Expectation::Absent,
        )
        .unwrap();

        let erik = read(&store, &user("erik")).unwrap();
        assert_eq!(erik.render_profile.as_deref(), Some("abslog"));
        assert_eq!(erik.x_scale, Some(2.0));
        // The point of the whole module: two people can disagree.
        assert_eq!(
            read(&store, &user("student")).unwrap(),
            Preferences::default()
        );
    }

    #[test]
    fn an_unset_field_is_left_out_of_the_file_rather_than_stored_as_a_default() {
        // Absence is what lets a project default reach the people who never
        // chose. Storing the built-in value would silently opt them out.
        let (dir, store) = store();
        write(
            &store,
            &user("erik"),
            &Preferences {
                render_profile: Some("abslog".to_string()),
                x_scale: None,
            },
            &Expectation::Absent,
        )
        .unwrap();

        let text = std::fs::read_to_string(dir.path().join("preferences/erik.json")).unwrap();
        assert!(text.contains("render_profile"), "{text}");
        assert!(!text.contains("x_scale"), "{text}");
    }

    #[test]
    fn a_malformed_document_is_an_error_to_read_and_a_default_to_serve() {
        let (_dir, store) = store();
        store
            .write(&path_of(&user("erik")), "{ not json", &Expectation::Any)
            .unwrap();

        assert!(matches!(
            read(&store, &user("erik")),
            Err(PreferencesError::Malformed { .. })
        ));
        // The request path must not turn one broken file into a broken page.
        assert_eq!(read_lenient(&store, &user("erik")), Preferences::default());
    }

    #[test]
    fn removing_preferences_reports_whether_there_were_any() {
        let (_dir, store) = store();
        assert!(!remove(&store, &user("erik")).unwrap());
        write(
            &store,
            &user("erik"),
            &Preferences {
                render_profile: Some("positive".to_string()),
                x_scale: None,
            },
            &Expectation::Absent,
        )
        .unwrap();
        assert!(remove(&store, &user("erik")).unwrap());
        assert_eq!(read(&store, &user("erik")).unwrap(), Preferences::default());
    }
}
