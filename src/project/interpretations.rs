//! Reading and writing level 1 interpretations inside a project.
//!
//! One document per user per radargram:
//!
//! ```text
//! interpretations/<radargram-id>/<user>.gprinterp.json
//! ```
//!
//! Splitting by user means two people interpreting the same radargram never
//! write to the same file, so their edits cannot conflict at all -- the only
//! conflicts left are one user in two browser tabs, which the store's
//! version check already handles. It also makes "show only my picks" a file
//! selection rather than a filter, and keeps each document a plain gprinterp
//! file that can be handed to any other tool unchanged.
//!
//! Both path components are validated slug types rather than strings off the
//! wire, so a request cannot address anything outside its own directory.

#![cfg_attr(
    not(feature = "server"),
    allow(
        dead_code,
        reason = "the write half of the project API is reached through the \
                  server's HTTP routes; a CLI-only build still needs the \
                  types to read and inspect a project"
    )
)]

use std::path::PathBuf;

use gprinterp::Document;

use crate::identity::{RadargramId, UserId};
use crate::project::store::{DocumentStore, Expectation, StoreError, Version};

/// Filename suffix, chosen so the gprinterp extension stays visible.
pub const SUFFIX: &str = ".gprinterp.json";

/// The stored form of one user's interpretation of one radargram.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredInterpretation {
    pub document: Document,
    pub version: Version,
}

#[derive(Debug)]
pub enum InterpretationError {
    Store(StoreError),
    /// The file on disk is not a parseable gprinterp document.
    Malformed {
        path: PathBuf,
        message: String,
    },
    /// The document's `key` names a different radargram than the path it is
    /// being stored under.
    KeyMismatch {
        expected: String,
        found: String,
    },
}

impl std::fmt::Display for InterpretationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpretationError::Store(e) => write!(f, "{e}"),
            InterpretationError::Malformed { path, message } => write!(
                f,
                "{} is not a valid gprinterp document: {message}",
                path.display()
            ),
            InterpretationError::KeyMismatch { expected, found } => write!(
                f,
                "the interpretation names radargram '{found}', but it is being saved \
                 for '{expected}'"
            ),
        }
    }
}

impl std::error::Error for InterpretationError {}

impl From<StoreError> for InterpretationError {
    fn from(e: StoreError) -> Self {
        InterpretationError::Store(e)
    }
}

fn directory_of(radargram: &RadargramId) -> PathBuf {
    PathBuf::from(crate::project::INTERPRETATIONS_DIR).join(radargram.as_str())
}

fn path_of(radargram: &RadargramId, user: &UserId) -> PathBuf {
    directory_of(radargram).join(format!("{}{SUFFIX}", user.as_str()))
}

/// Users who have an interpretation of `radargram`.
pub fn list_users(
    store: &DocumentStore,
    radargram: &RadargramId,
) -> Result<Vec<String>, InterpretationError> {
    Ok(store.list_stems(&directory_of(radargram), SUFFIX)?)
}

/// Read one user's interpretation, or `None` if they have not made one.
pub fn read(
    store: &DocumentStore,
    radargram: &RadargramId,
    user: &UserId,
) -> Result<Option<StoredInterpretation>, InterpretationError> {
    let relative = path_of(radargram, user);
    let Some(stored) = store.read(&relative)? else {
        return Ok(None);
    };
    let document =
        Document::from_json(&stored.text).map_err(|e| InterpretationError::Malformed {
            path: store.root().join(&relative),
            message: e.to_string(),
        })?;
    Ok(Some(StoredInterpretation {
        document,
        version: stored.version,
    }))
}

/// Write one user's interpretation.
///
/// The document's `key` must name the radargram it is being stored under.
/// Storing a mismatched one would put an interpretation of line A in line
/// B's directory, where every later read would apply it to the wrong data
/// and no downstream check could notice.
pub fn write(
    store: &DocumentStore,
    radargram: &RadargramId,
    user: &UserId,
    document: &Document,
    expected: &Expectation,
) -> Result<Version, InterpretationError> {
    if document.key != radargram.as_str() {
        return Err(InterpretationError::KeyMismatch {
            expected: radargram.as_str().to_string(),
            found: document.key.clone(),
        });
    }
    let text = document
        .to_json_pretty()
        .map_err(|e| InterpretationError::Malformed {
            path: path_of(radargram, user),
            message: e.to_string(),
        })?;
    Ok(store.write(&path_of(radargram, user), &text, expected)?)
}

/// Delete one user's interpretation. Returns whether it existed.
pub fn remove(
    store: &DocumentStore,
    radargram: &RadargramId,
    user: &UserId,
    expected: &Expectation,
) -> Result<bool, InterpretationError> {
    Ok(store.remove(&path_of(radargram, user), expected)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::DEFAULT_USER;
    use crate::project::Project;

    fn project() -> (tempfile::TempDir, Project) {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        (dir, project)
    }

    fn document(key: &str) -> Document {
        serde_json::from_value(serde_json::json!({
            "key": key,
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[10.0, 200.0], [90.0, 210.0]]},
                "properties": {"id": "f-0001", "label": "bed"}
            }]
        }))
        .unwrap()
    }

    fn ids(radargram: &str, user: &str) -> (RadargramId, UserId) {
        (
            RadargramId::new(radargram).unwrap(),
            UserId::new(user).unwrap(),
        )
    }

    #[test]
    fn an_interpretation_round_trips_through_the_store() {
        let (_dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        let document = document("dronbreen-0237");

        let version = write(
            project.documents(),
            &radargram,
            &user,
            &document,
            &Expectation::Absent,
        )
        .unwrap();

        let stored = read(project.documents(), &radargram, &user)
            .unwrap()
            .unwrap();
        assert_eq!(stored.document, document);
        assert_eq!(stored.version, version);
    }

    #[test]
    fn it_lands_at_the_documented_path() {
        // The layout is part of the contract: other tools are expected to
        // read these files directly.
        let (dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        write(
            project.documents(),
            &radargram,
            &user,
            &document("dronbreen-0237"),
            &Expectation::Absent,
        )
        .unwrap();

        assert!(dir
            .path()
            .join("interpretations/dronbreen-0237/default.gprinterp.json")
            .is_file());
    }

    #[test]
    fn two_users_of_one_radargram_never_share_a_file() {
        let (_dir, project) = project();
        let (radargram, erik) = ids("dronbreen-0237", "erik");
        let student = UserId::new("student-a").unwrap();

        write(
            project.documents(),
            &radargram,
            &erik,
            &document("dronbreen-0237"),
            &Expectation::Absent,
        )
        .unwrap();
        // Absent, not Any: if these shared a file this would be a conflict.
        write(
            project.documents(),
            &radargram,
            &student,
            &document("dronbreen-0237"),
            &Expectation::Absent,
        )
        .unwrap();

        assert_eq!(
            list_users(project.documents(), &radargram).unwrap(),
            vec!["erik", "student-a"]
        );
    }

    #[test]
    fn a_document_for_a_different_radargram_is_refused() {
        let (_dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);

        let error = write(
            project.documents(),
            &radargram,
            &user,
            &document("kroppbreen-01"),
            &Expectation::Absent,
        )
        .unwrap_err();
        assert!(
            matches!(error, InterpretationError::KeyMismatch { .. }),
            "{error}"
        );
        assert!(read(project.documents(), &radargram, &user)
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_second_save_from_a_stale_tab_is_refused() {
        let (_dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        let first = write(
            project.documents(),
            &radargram,
            &user,
            &document("dronbreen-0237"),
            &Expectation::Absent,
        )
        .unwrap();

        let mut edited = document("dronbreen-0237");
        edited.features.clear();
        write(
            project.documents(),
            &radargram,
            &user,
            &edited,
            &Expectation::Version(first.clone()),
        )
        .unwrap();

        let error = write(
            project.documents(),
            &radargram,
            &user,
            &document("dronbreen-0237"),
            &Expectation::Version(first),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            InterpretationError::Store(StoreError::Conflict { .. })
        ));
    }

    #[test]
    fn unknown_fields_survive_a_save_and_reload() {
        // The gprinterp round-trip requirement has to hold through storage,
        // not just through the parser: a field this Ridal does not model
        // must still be there after the GUI saves over it.
        let (_dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        let document: Document = serde_json::from_value(serde_json::json!({
            "key": "dronbreen-0237",
            "features": [],
            "something_from_a_future_version": {"nested": [1, 2, 3]}
        }))
        .unwrap();

        write(
            project.documents(),
            &radargram,
            &user,
            &document,
            &Expectation::Absent,
        )
        .unwrap();
        let stored = read(project.documents(), &radargram, &user)
            .unwrap()
            .unwrap();

        assert_eq!(
            stored.document.extra.get("something_from_a_future_version"),
            document.extra.get("something_from_a_future_version")
        );
    }

    #[test]
    fn listing_a_radargram_nobody_has_interpreted_is_empty() {
        let (_dir, project) = project();
        let radargram = RadargramId::new("untouched").unwrap();
        assert!(list_users(project.documents(), &radargram)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn removing_reports_whether_anything_was_there() {
        let (_dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        assert!(!remove(project.documents(), &radargram, &user, &Expectation::Any).unwrap());

        write(
            project.documents(),
            &radargram,
            &user,
            &document("dronbreen-0237"),
            &Expectation::Absent,
        )
        .unwrap();
        assert!(remove(project.documents(), &radargram, &user, &Expectation::Any).unwrap());
    }

    #[test]
    fn a_corrupt_document_names_the_file_rather_than_failing_obscurely() {
        let (dir, project) = project();
        let (radargram, user) = ids("dronbreen-0237", DEFAULT_USER);
        let path = dir.path().join("interpretations/dronbreen-0237");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("default.gprinterp.json"), "{ not json").unwrap();

        let error = read(project.documents(), &radargram, &user).unwrap_err();
        assert!(
            format!("{error}").contains("default.gprinterp.json"),
            "{error}"
        );
    }
}
