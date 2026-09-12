//! A small store for text documents inside a project.
//!
//! Interpretations and layer definitions have identical persistence needs: a
//! single JSON file, written atomically, with enough concurrency control
//! that two browser tabs cannot silently overwrite each other. Ridal is
//! meant to run as a long-lived daemon eventually, so "two writers at once"
//! is a normal case rather than a corner one, and layer definitions are
//! explicitly the first of several project-scoped document kinds to come.
//! Those are the conditions under which one shared implementation is worth
//! having rather than two hand-rolled ones.
//!
//! What it deliberately does *not* do is know anything about schemas. The
//! store moves bytes and manages versions; what those bytes mean stays with
//! the caller, so adding a document kind never means touching this file.
//!
//! # Atomicity
//!
//! Every write goes to a uniquely named temporary sibling and is installed
//! with a rename, which is atomic within a filesystem. A crash mid-write
//! therefore leaves either the old document or the new one, never a
//! truncated file that still parses as JSON up to the cut. Temporary files
//! are removed even on the error paths.
//!
//! # Concurrency
//!
//! [`Version`] is a content hash, used as an HTTP `ETag`. A writer passes
//! the version it believes it is replacing; if the document has moved on
//! since, the write is refused as a conflict instead of discarding the other
//! writer's work. The compare-and-swap runs under a process-wide lock, so
//! the check and the rename cannot interleave.
//!
//! The lock is one mutex for the whole store rather than one per document.
//! These writes are small JSON files saved at human speed, while the
//! expensive concurrent work in this server is rendering, which never
//! touches the store -- so a finer-grained lock would add machinery to
//! contend for something that is not contended.

#![cfg_attr(
    not(feature = "server"),
    allow(
        dead_code,
        reason = "the write half of the project API is reached through the \
                  server's HTTP routes; a CLI-only build still needs the \
                  types to read and inspect a project"
    )
)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A document's content version, used as an HTTP `ETag`.
///
/// Derived from the bytes on disk rather than from a counter or an mtime: a
/// counter needs somewhere to live and a coarse mtime cannot distinguish two
/// writes within the same tick. A hash also makes an idempotent re-save
/// (identical content) leave the version unchanged, which is the honest
/// answer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Version(String);

impl Version {
    fn of(bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ridal-document-v1");
        hasher.update(bytes);
        Self(hasher.finalize().to_hex()[..32].to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parse a version from an HTTP header value, tolerating the quoting and
    /// the weak-comparison prefix that `ETag`/`If-Match` allow.
    pub fn from_header(raw: &str) -> Version {
        let trimmed = raw.trim();
        let trimmed = trimmed.strip_prefix("W/").unwrap_or(trimmed);
        Version(trimmed.trim_matches('"').to_string())
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A document as it exists on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub text: String,
    pub version: Version,
}

/// What a writer expects to find in place.
#[derive(Debug, Clone, PartialEq)]
pub enum Expectation {
    /// Replace whatever is there. Used by clients that cannot do better;
    /// the last writer wins.
    Any,
    /// The document must already exist, at whatever version.
    ///
    /// This is HTTP's `If-Match: *`, which asserts existence without
    /// naming a version. `Any` cannot stand in for it: `Any` is satisfied
    /// by an absent document, so a client asserting "replace the thing
    /// that is there" would instead create one.
    Present,
    /// The document must not exist yet.
    Absent,
    /// The document must currently be at this version.
    Version(Version),
}

#[derive(Debug)]
pub enum StoreError {
    /// The document changed since the version the writer expected, or exists
    /// when it was expected not to. The caller should re-read and merge
    /// rather than retrying blindly.
    Conflict {
        expected: Expectation,
        actual: Option<Version>,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The store's lock was poisoned by a panic in another thread.
    Poisoned,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Conflict { expected, actual } => {
                let actual = match actual {
                    Some(v) => format!("is at version {v}"),
                    None => "does not exist".to_string(),
                };
                match expected {
                    Expectation::Absent => write!(
                        f,
                        "the document already exists (it {actual}), but the write \
                         required it to be new"
                    ),
                    Expectation::Version(v) => write!(
                        f,
                        "the document has changed since it was loaded: the write \
                         expected version {v}, but it {actual}. Reload and reapply \
                         the change so the other edit is not discarded."
                    ),
                    Expectation::Present => write!(
                        f,
                        "the write required the document to already exist, but it \
                         {actual}"
                    ),
                    Expectation::Any => write!(f, "the document {actual} unexpectedly"),
                }
            }
            StoreError::Io { path, source } => write!(f, "{}: {source}", path.display()),
            StoreError::Poisoned => write!(f, "the document store lock was poisoned by a panic"),
        }
    }
}

impl std::error::Error for StoreError {}

/// Removes a temporary file unless it was explicitly kept.
///
/// A `Drop` guard rather than cleanup on each error path, because there are
/// several ways out of [`DocumentStore::write`] and an abandoned temporary
/// file beside a real document is exactly the kind of debris that later gets
/// mistaken for data.
struct TempFile {
    path: PathBuf,
    keep: bool,
}

impl TempFile {
    fn keep(&mut self) {
        self.keep = true;
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Stores text documents under a root directory.
///
/// JSON in every case but one: the project marker (`ridal.toml`) goes
/// through here too, for the atomic write and the process-wide write lock
/// rather than for anything JSON-specific.
#[derive(Debug)]
pub struct DocumentStore {
    root: PathBuf,
    /// Serializes the read-compare-rename sequence in [`DocumentStore::write`].
    write_lock: Mutex<()>,
}

impl DocumentStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            write_lock: Mutex::new(()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Absolute path of a document.
    ///
    /// `relative` must already be built from validated components -- slug
    /// newtypes from [`crate::identity`], not raw strings off the wire. This
    /// is the single place where that assumption is relied on, so it is
    /// asserted here rather than trusted: a component that is absolute, or
    /// that contains a parent-directory hop, would escape the project.
    fn path_of(&self, relative: &Path) -> Result<PathBuf, StoreError> {
        let escapes = relative.is_absolute()
            || relative
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)));
        if escapes {
            return Err(StoreError::Io {
                path: relative.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "document paths must be relative and must not contain '..'",
                ),
            });
        }
        Ok(self.root.join(relative))
    }

    /// Read a document, or `None` if it does not exist.
    pub fn read(&self, relative: &Path) -> Result<Option<Document>, StoreError> {
        let path = self.path_of(relative)?;
        match std::fs::read(&path) {
            Ok(bytes) => {
                let version = Version::of(&bytes);
                let text = String::from_utf8(bytes).map_err(|e| StoreError::Io {
                    path: path.clone(),
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
                })?;
                Ok(Some(Document { text, version }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(StoreError::Io { path, source }),
        }
    }

    /// The current version of a document, or `None` if absent.
    pub fn version(&self, relative: &Path) -> Result<Option<Version>, StoreError> {
        Ok(self.read(relative)?.map(|d| d.version))
    }

    /// Write a document, refusing if `expected` no longer holds.
    ///
    /// Returns the version read back *from disk* after the rename, not one
    /// computed from the in-memory text. The two agree today, but a
    /// subsequent `read` is what a client's next `If-Match` will be compared
    /// against, so the version handed out has to be the one that file
    /// actually has.
    pub fn write(
        &self,
        relative: &Path,
        text: &str,
        expected: &Expectation,
    ) -> Result<Version, StoreError> {
        let path = self.path_of(relative)?;
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Poisoned)?;

        let actual = self.version(relative)?;
        let satisfied = match (expected, &actual) {
            (Expectation::Any, _) => true,
            (Expectation::Present, Some(_)) => true,
            (Expectation::Present, None) => false,
            (Expectation::Absent, None) => true,
            (Expectation::Absent, Some(_)) => false,
            (Expectation::Version(wanted), Some(found)) => wanted == found,
            (Expectation::Version(_), None) => false,
        };
        if !satisfied {
            return Err(StoreError::Conflict {
                expected: expected.clone(),
                actual,
            });
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        // A sibling of the destination, so the rename stays within one
        // filesystem. The uniqueness suffix keeps two concurrent writers
        // from sharing a temporary file even though the lock above already
        // prevents that within one process -- a second Ridal serving the
        // same project is not prevented by anything here.
        let mut temp = TempFile {
            path: temp_path(&path),
            keep: false,
        };
        std::fs::write(&temp.path, text).map_err(|source| StoreError::Io {
            path: temp.path.clone(),
            source,
        })?;
        std::fs::rename(&temp.path, &path).map_err(|source| StoreError::Io {
            path: path.clone(),
            source,
        })?;
        temp.keep();

        let written = self.version(relative)?.ok_or_else(|| StoreError::Io {
            path: path.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "the document disappeared immediately after being written",
            ),
        })?;
        Ok(written)
    }

    /// Delete a document. Returns whether it existed.
    pub fn remove(&self, relative: &Path) -> Result<bool, StoreError> {
        let path = self.path_of(relative)?;
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Poisoned)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(source) => Err(StoreError::Io { path, source }),
        }
    }

    /// File stems of the documents directly inside `relative`, sorted.
    ///
    /// A missing directory lists as empty rather than erroring: "nobody has
    /// interpreted this radargram yet" is a normal state, not a fault.
    pub fn list_stems(&self, relative: &Path, suffix: &str) -> Result<Vec<String>, StoreError> {
        let path = self.path_of(relative)?;
        let entries = match std::fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => return Err(StoreError::Io { path, source }),
        };

        let mut stems = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(stem) = name.strip_suffix(suffix) {
                // Skip the temporary files of an interrupted write, which
                // would otherwise be listed as a user named e.g.
                // "default.a1b2c3d4.tmp".
                if !stem.is_empty() && !stem.contains(".tmp") {
                    stems.push(stem.to_string());
                }
            }
        }
        stems.sort();
        Ok(stems)
    }
}

/// A unique temporary sibling, keeping the real extension last so anything
/// that sniffs by suffix still sees the right format.
fn temp_path(path: &Path) -> PathBuf {
    let unique = Version::of(
        format!(
            "{}-{:?}-{}",
            path.display(),
            std::time::SystemTime::now(),
            std::process::id()
        )
        .as_bytes(),
    );
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("document.json");
    let (stem, extension) = name.rsplit_once('.').unwrap_or((name, "tmp"));
    path.with_file_name(format!("{stem}.{}.tmp.{extension}", &unique.as_str()[..8]))
}

#[cfg(test)]
mod tests {
    #[test]
    fn present_requires_an_existing_document() {
        // HTTP's `If-Match: *`. `Any` would accept an absent document and
        // quietly turn "replace what is there" into a create.
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        let path = Path::new("thing.json");

        let err = store
            .write(path, "{}", &Expectation::Present)
            .expect_err("must refuse when nothing is there");
        assert!(
            matches!(err, StoreError::Conflict { actual: None, .. }),
            "{err}"
        );
        assert!(
            !dir.path().join("thing.json").exists(),
            "nothing was created"
        );
        assert!(err.to_string().contains("already exist"), "{err}");

        store.write(path, "{}", &Expectation::Absent).unwrap();
        // Now that it exists, the same write is allowed whatever its version.
        store
            .write(path, "{\"v\":2}", &Expectation::Present)
            .unwrap();
        assert_eq!(store.read(path).unwrap().unwrap().text, "{\"v\":2}");
    }

    use super::*;

    fn store() -> (tempfile::TempDir, DocumentStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(dir.path().to_path_buf());
        (dir, store)
    }

    fn doc(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn a_written_document_reads_back_with_the_returned_version() {
        let (_dir, store) = store();
        let version = store
            .write(&doc("a.json"), "{\"x\":1}", &Expectation::Absent)
            .unwrap();

        let read = store.read(&doc("a.json")).unwrap().unwrap();
        assert_eq!(read.text, "{\"x\":1}");
        assert_eq!(
            read.version, version,
            "the version handed to a client must match what its next read sees"
        );
    }

    #[test]
    fn a_missing_document_reads_as_none_rather_than_erroring() {
        let (_dir, store) = store();
        assert!(store.read(&doc("nothing.json")).unwrap().is_none());
    }

    #[test]
    fn nested_directories_are_created_on_write() {
        let (_dir, store) = store();
        store
            .write(
                &doc("interpretations/line-01/default.gprinterp.json"),
                "{}",
                &Expectation::Absent,
            )
            .unwrap();
        assert!(store
            .read(&doc("interpretations/line-01/default.gprinterp.json"))
            .unwrap()
            .is_some());
    }

    #[test]
    fn a_stale_version_is_refused_so_the_other_edit_survives() {
        // The lost-update case: two tabs load the same document, both save.
        let (_dir, store) = store();
        let first = store
            .write(&doc("a.json"), "one", &Expectation::Absent)
            .unwrap();

        let second = store
            .write(&doc("a.json"), "two", &Expectation::Version(first.clone()))
            .unwrap();

        let error = store
            .write(&doc("a.json"), "three", &Expectation::Version(first))
            .unwrap_err();
        assert!(matches!(error, StoreError::Conflict { .. }), "{error}");

        // The second writer's content is intact; the stale one did not land.
        let read = store.read(&doc("a.json")).unwrap().unwrap();
        assert_eq!(read.text, "two");
        assert_eq!(read.version, second);
    }

    #[test]
    fn expecting_absent_refuses_to_clobber_an_existing_document() {
        let (_dir, store) = store();
        store
            .write(&doc("a.json"), "one", &Expectation::Absent)
            .unwrap();
        let error = store
            .write(&doc("a.json"), "two", &Expectation::Absent)
            .unwrap_err();
        assert!(matches!(
            error,
            StoreError::Conflict {
                expected: Expectation::Absent,
                actual: Some(_)
            }
        ));
    }

    #[test]
    fn expecting_a_version_refuses_when_the_document_is_gone() {
        let (_dir, store) = store();
        let version = store
            .write(&doc("a.json"), "one", &Expectation::Absent)
            .unwrap();
        store.remove(&doc("a.json")).unwrap();

        let error = store
            .write(&doc("a.json"), "two", &Expectation::Version(version))
            .unwrap_err();
        assert!(matches!(error, StoreError::Conflict { actual: None, .. }));
    }

    #[test]
    fn any_overwrites_unconditionally() {
        let (_dir, store) = store();
        store
            .write(&doc("a.json"), "one", &Expectation::Any)
            .unwrap();
        store
            .write(&doc("a.json"), "two", &Expectation::Any)
            .unwrap();
        assert_eq!(store.read(&doc("a.json")).unwrap().unwrap().text, "two");
    }

    #[test]
    fn rewriting_identical_content_keeps_the_same_version() {
        // Version is content, not a counter, so an idempotent save does not
        // invalidate another client's ETag.
        let (_dir, store) = store();
        let first = store
            .write(&doc("a.json"), "same", &Expectation::Absent)
            .unwrap();
        let second = store
            .write(&doc("a.json"), "same", &Expectation::Version(first.clone()))
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn writing_leaves_no_temporary_files_behind() {
        let (dir, store) = store();
        store
            .write(&doc("a.json"), "one", &Expectation::Absent)
            .unwrap();
        store
            .write(&doc("a.json"), "two", &Expectation::Any)
            .unwrap();

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.json".to_string()], "{names:?}");
    }

    #[test]
    fn a_failed_conflict_check_does_not_touch_the_stored_document() {
        let (dir, store) = store();
        store
            .write(&doc("a.json"), "one", &Expectation::Absent)
            .unwrap();
        let _ = store.write(&doc("a.json"), "two", &Expectation::Absent);

        assert_eq!(store.read(&doc("a.json")).unwrap().unwrap().text, "one");
        let count = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(count, 1, "a refused write left debris behind");
    }

    #[test]
    fn paths_that_escape_the_root_are_refused() {
        // Defence in depth: callers build paths from validated slugs, but
        // this is the boundary where an escape would actually happen.
        let (_dir, store) = store();
        for bad in ["../outside.json", "a/../../outside.json", "/etc/passwd"] {
            let error = store.write(&doc(bad), "x", &Expectation::Any).unwrap_err();
            assert!(matches!(error, StoreError::Io { .. }), "{bad} was allowed");
            assert!(store.read(&doc(bad)).is_err(), "{bad} was readable");
        }
    }

    #[test]
    fn listing_a_missing_directory_is_empty_not_an_error() {
        let (_dir, store) = store();
        assert!(store
            .list_stems(&doc("interpretations/nobody"), ".gprinterp.json")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn listing_returns_sorted_stems_and_ignores_other_files() {
        let (_dir, store) = store();
        for user in ["erik", "default", "student-a"] {
            store
                .write(
                    &doc(&format!("interpretations/line-01/{user}.gprinterp.json")),
                    "{}",
                    &Expectation::Absent,
                )
                .unwrap();
        }
        store
            .write(
                &doc("interpretations/line-01/notes.txt"),
                "x",
                &Expectation::Any,
            )
            .unwrap();

        let stems = store
            .list_stems(&doc("interpretations/line-01"), ".gprinterp.json")
            .unwrap();
        assert_eq!(stems, vec!["default", "erik", "student-a"]);
    }

    #[test]
    fn version_parses_quoted_and_weak_etag_headers() {
        let plain = Version::from_header("abc123");
        assert_eq!(Version::from_header("\"abc123\""), plain);
        assert_eq!(Version::from_header("W/\"abc123\""), plain);
        assert_eq!(Version::from_header("  abc123  "), plain);
    }
}
