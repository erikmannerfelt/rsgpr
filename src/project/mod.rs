//! Ridal projects: the on-disk home for everything that is *authored*
//! rather than processed.
//!
//! Until now Ridal has been strictly read-only, so a radargram's path was
//! all the state there was. Interpretations change that: picks have to be
//! saved somewhere, and so do the layer definitions they refer to. A project
//! is that somewhere.
//!
//! ```text
//! myproject/
//!   ridal.toml                              marker and settings
//!   radargrams/                             optional; roots are configurable
//!     dronbreen-0237.nc
//!   interpretations/
//!     dronbreen-0237/
//!       default.gprinterp.json              one document per user
//!   layers/
//!     layers.json                           project-scoped layer vocabulary
//!   cache/                                  derived data; safe to delete
//! ```
//!
//! # Why an explicit marker
//!
//! A directory is a project only if it contains `ridal.toml`. Pointing
//! `ridal gui` at a bare directory of `.nc` files keeps working exactly as
//! before, read-only, which means adding a write path takes nothing away
//! from the existing behaviour. It also makes "where do these picks go?"
//! answerable by looking, rather than by inference from what happens to be
//! lying around.
//!
//! # Two kinds of data, deliberately separated
//!
//! Everything under `interpretations/` and `layers/` is **authored**: a
//! person made it, nothing can regenerate it, and losing it is data loss.
//! Those go through [`store::DocumentStore`], which writes atomically and
//! refuses to silently overwrite a concurrent edit.
//!
//! `cache/` is **derived**: rendered images and anything else Ridal can
//! rebuild from a radargram. It is deliberately *not* a document store.
//! Deleting it must always be safe, it needs no versioning or conflict
//! detection, and it should not be backed up -- which is why Ridal drops a
//! `CACHEDIR.TAG` in it, the convention backup tools already understand.
//! Its location is configurable precisely because a project directory may
//! sit on a network share while a cache wants local disk, which is the
//! normal arrangement for a long-running daemon.
//!
//! # Adding another kind of data later
//!
//! Give it a store directory and a module beside [`interpretations`] and
//! [`layers`]. The document store knows nothing about schemas, so nothing in
//! `store.rs` needs to change.

pub mod interpretations;
pub mod layers;
pub mod store;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use store::DocumentStore;

/// The file whose presence makes a directory a project.
pub const MARKER: &str = "ridal.toml";

/// Store directory for interpretations, relative to the project root.
pub const INTERPRETATIONS_DIR: &str = "interpretations";
/// Store directory for layer definitions.
pub const LAYERS_DIR: &str = "layers";
/// Default location for derived data.
pub const DEFAULT_CACHE_DIR: &str = "cache";
/// Default directory scanned for radargrams when the config says nothing.
pub const DEFAULT_RADARGRAM_DIR: &str = "radargrams";

/// Contents of `ridal.toml`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub project: ProjectSection,
    #[serde(default)]
    pub radargrams: RadargramsSection,
    #[serde(default)]
    pub cache: CacheSection,
    #[serde(default)]
    pub render: RenderSection,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProjectSection {
    /// Human-facing project name. Cosmetic; no identity semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RadargramsSection {
    /// Directories scanned for processed radargrams. Relative paths resolve
    /// against the project root; absolute paths are used as given, so a
    /// project can index an archive it does not contain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roots: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderSection {
    /// Render profile used when a request does not name one.
    ///
    /// Kept as a plain string: the set of valid profiles is a server
    /// concept, and a CLI-only build has no way to check it. Validation
    /// belongs at the HTTP boundary where a bad value can be refused with a
    /// useful message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_profile: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheSection {
    /// Where derived data lives. Relative to the project root unless
    /// absolute -- an absolute path is the point, for a daemon whose project
    /// is on a network share but whose cache should be on local disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
}

/// Quote a value as a TOML basic string.
///
/// Project names are free text and paths can contain backslashes, so the
/// template cannot just wrap them in quotes and hope.
fn toml_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

/// An opened project.
///
/// The config sits behind a lock because the settings page edits it through
/// a shared `&AppState`, and a change that only reached the file would not
/// take effect until a restart -- which is not what pressing Save looks
/// like it does.
#[derive(Debug)]
pub struct Project {
    root: PathBuf,
    config: std::sync::RwLock<ProjectConfig>,
    documents: DocumentStore,
}

#[derive(Debug)]
pub enum ProjectError {
    NotAProject(PathBuf),
    AlreadyAProject(PathBuf),
    Io { path: PathBuf, message: String },
    Config { path: PathBuf, message: String },
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectError::NotAProject(path) => write!(
                f,
                "{} is not a Ridal project (no {MARKER}). Run `ridal project init` \
                 there to create one.",
                path.display()
            ),
            ProjectError::AlreadyAProject(path) => write!(
                f,
                "{} is already a Ridal project ({MARKER} exists)",
                path.display()
            ),
            ProjectError::Io { path, message } => write!(f, "{}: {message}", path.display()),
            ProjectError::Config { path, message } => {
                write!(f, "could not read {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for ProjectError {}

impl Project {
    /// Open the project rooted exactly at `root`.
    pub fn open(root: &Path) -> Result<Project, ProjectError> {
        let marker = root.join(MARKER);
        if !marker.is_file() {
            return Err(ProjectError::NotAProject(root.to_path_buf()));
        }
        let text = std::fs::read_to_string(&marker).map_err(|e| ProjectError::Io {
            path: marker.clone(),
            message: e.to_string(),
        })?;
        let config: ProjectConfig = toml::from_str(&text).map_err(|e| ProjectError::Config {
            path: marker,
            message: e.to_string(),
        })?;

        let root = root.to_path_buf();
        Ok(Project {
            documents: DocumentStore::new(root.clone()),
            root,
            config: std::sync::RwLock::new(config),
        })
    }

    /// Find the project containing `start`, searching upwards.
    ///
    /// Upwards rather than exact-match so that pointing Ridal at a
    /// subdirectory -- or at a single `.nc` file inside a project -- still
    /// finds the interpretations that belong to it. Returns `Ok(None)` when
    /// there is no project above `start`, which is the ordinary read-only
    /// case rather than an error.
    pub fn discover(start: &Path) -> Result<Option<Project>, ProjectError> {
        let start = std::fs::canonicalize(start).map_err(|e| ProjectError::Io {
            path: start.to_path_buf(),
            message: e.to_string(),
        })?;
        let mut current: Option<&Path> = if start.is_file() {
            start.parent()
        } else {
            Some(start.as_path())
        };
        while let Some(dir) = current {
            if dir.join(MARKER).is_file() {
                return Ok(Some(Project::open(dir)?));
            }
            current = dir.parent();
        }
        Ok(None)
    }

    /// Create a project at `root`, which need not exist yet.
    pub fn init(root: &Path, name: Option<&str>) -> Result<Project, ProjectError> {
        if root.join(MARKER).exists() {
            return Err(ProjectError::AlreadyAProject(root.to_path_buf()));
        }
        for dir in [
            root.to_path_buf(),
            root.join(INTERPRETATIONS_DIR),
            root.join(LAYERS_DIR),
            root.join(DEFAULT_RADARGRAM_DIR),
        ] {
            std::fs::create_dir_all(&dir).map_err(|e| ProjectError::Io {
                path: dir,
                message: e.to_string(),
            })?;
        }

        let config = ProjectConfig {
            project: ProjectSection {
                name: name.map(str::to_string),
            },
            radargrams: RadargramsSection {
                roots: vec![DEFAULT_RADARGRAM_DIR.to_string()],
            },
            cache: CacheSection::default(),
            render: RenderSection::default(),
        };
        // Written as a commented template rather than serialized, because
        // this file exists to be hand-edited: serde would emit a bare,
        // undocumented `[cache]` with no keys, which reads as debris rather
        // than as an invitation. `Project::open` parses either form.
        let name_line = match &config.project.name {
            Some(name) => format!("name = {}\n", toml_string(name)),
            None => format!("# name = {}\n", toml_string("My survey")),
        };
        let text = format!(
            "# Ridal project. Its presence is what makes this directory a project;\n\
             # `ridal gui .` here can then save interpretations.\n\
             \n\
             [project]\n\
             {name_line}\n\
             # Directories scanned for processed radargrams. Relative paths resolve\n\
             # against this file; absolute paths let a project index an archive it\n\
             # does not contain.\n\
             [radargrams]\n\
             roots = [{}]\n\
             \n\
             # Where derived data (rendered images, and anything else Ridal can\n\
             # rebuild) is kept. Safe to delete at any time. Point this at local\n\
             # disk if the project itself lives on a network share.\n\
             # [cache]\n\
             # dir = {}\n\
             \n\
             # Render profile used when a page does not ask for one. Set it\n\
             # from Project settings in the browser, or add a line here such\n\
             # as `default_profile = {}`. Left unset, Ridal uses its\n\
             # built-in \"default\" profile.\n\
             #\n\
             # A real (empty) table rather than a commented one, so that\n\
             # saving from the browser puts the key under this note instead\n\
             # of appending a second [render] elsewhere in the file.\n\
             [render]\n",
            toml_string(DEFAULT_RADARGRAM_DIR),
            toml_string("/var/cache/ridal"),
            toml_string("default"),
        );
        let marker = root.join(MARKER);
        std::fs::write(&marker, text).map_err(|e| ProjectError::Io {
            path: marker,
            message: e.to_string(),
        })?;

        let project = Project::open(root)?;
        // Created and tagged up front so the layout is complete from the
        // start, and so a project directory is safe to back up wholesale
        // before anything has been rendered into it.
        project.ensure_cache_dir()?;
        Ok(project)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A snapshot of the current settings.
    ///
    /// Cloned rather than borrowed: the config is behind a lock, and
    /// handing out a guard would make every caller hold it for as long as
    /// they held the value. It is a handful of short strings.
    pub fn config(&self) -> ProjectConfig {
        self.read_config().clone()
    }

    fn read_config(&self) -> std::sync::RwLockReadGuard<'_, ProjectConfig> {
        // A poisoned lock means a panic while writing settings. The stored
        // value is still whatever was last read from disk, which is a
        // better answer than propagating the panic to every page.
        self.config
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The render profile to use when a request does not name one.
    pub fn default_profile(&self) -> Option<String> {
        self.read_config().render.default_profile.clone()
    }

    /// Set (or clear) the default render profile, in the file and in memory.
    ///
    /// Reached through the settings page, so a CLI-only build never calls
    /// it -- same situation as the write half of the stores beside this.
    ///
    /// Edited with `toml_edit` rather than re-serialised, so the comments
    /// `ridal project init` writes survive. The file is meant to be
    /// hand-editable; a settings page that silently stripped a user's notes
    /// out of it would be a poor trade for one dropdown.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    pub fn set_default_profile(&self, profile: Option<&str>) -> Result<(), ProjectError> {
        let marker = self.root.join(MARKER);
        let text = std::fs::read_to_string(&marker).map_err(|e| ProjectError::Io {
            path: marker.clone(),
            message: e.to_string(),
        })?;
        let mut document: toml_edit::DocumentMut =
            text.parse()
                .map_err(|e: toml_edit::TomlError| ProjectError::Config {
                    path: marker.clone(),
                    message: e.to_string(),
                })?;

        match profile {
            Some(name) => {
                if !document.contains_key("render") {
                    document["render"] = toml_edit::Item::Table(toml_edit::Table::new());
                }
                document["render"]["default_profile"] = toml_edit::value(name);
            }
            None => {
                if let Some(table) = document
                    .get_mut("render")
                    .and_then(toml_edit::Item::as_table_mut)
                {
                    table.remove("default_profile");
                }
            }
        }

        let updated = document.to_string();
        // Through the document store for its atomic write and its
        // process-wide write lock, which also serialises two settings saves
        // arriving at once.
        self.documents
            .write(Path::new(MARKER), &updated, &store::Expectation::Any)
            .map_err(|e| ProjectError::Io {
                path: marker.clone(),
                message: e.to_string(),
            })?;

        // Re-parsed from what was written rather than patched in memory, so
        // the two cannot drift.
        let reparsed: ProjectConfig =
            toml::from_str(&updated).map_err(|e| ProjectError::Config {
                path: marker,
                message: e.to_string(),
            })?;
        let mut guard = self
            .config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = reparsed;
        Ok(())
    }

    /// The store holding authored documents.
    pub fn documents(&self) -> &DocumentStore {
        &self.documents
    }

    /// Absolute directories to scan for processed radargrams.
    ///
    /// Falls back to the project root itself when the config lists none, so
    /// a project whose `.nc` files sit loose at the top level still works
    /// without configuration.
    pub fn radargram_roots(&self) -> Vec<PathBuf> {
        let config = self.read_config();
        if config.radargrams.roots.is_empty() {
            return vec![self.root.clone()];
        }
        config
            .radargrams
            .roots
            .iter()
            .map(|entry| self.resolve(entry))
            .collect()
    }

    /// Where derived data belongs.
    ///
    /// Reserved now and created on demand: the on-disk render cache does not
    /// exist yet, but its location is a project-shaped decision and settling
    /// it here means adding the cache later is not also a layout change.
    pub fn cache_dir(&self) -> PathBuf {
        match &self.read_config().cache.dir {
            Some(dir) => self.resolve(dir),
            None => self.root.join(DEFAULT_CACHE_DIR),
        }
    }

    /// Create the cache directory and mark it as derived data.
    ///
    /// `CACHEDIR.TAG` is the convention backup and archiving tools already
    /// recognise (Cargo tags `target/` the same way), so a project directory
    /// can be backed up wholesale without dragging along regenerable
    /// renders.
    pub fn ensure_cache_dir(&self) -> Result<PathBuf, ProjectError> {
        let dir = self.cache_dir();
        std::fs::create_dir_all(&dir).map_err(|e| ProjectError::Io {
            path: dir.clone(),
            message: e.to_string(),
        })?;
        let tag = dir.join("CACHEDIR.TAG");
        if !tag.exists() {
            let contents = "Signature: 8a477f597d28d172789f06886806bc55\n\
                            # This file is a cache directory tag created by ridal.\n\
                            # For information about cache directory tags, see:\n\
                            #\thttps://bford.info/cachedir/\n";
            std::fs::write(&tag, contents).map_err(|e| ProjectError::Io {
                path: tag,
                message: e.to_string(),
            })?;
        }
        Ok(dir)
    }

    fn resolve(&self, entry: &str) -> PathBuf {
        let path = Path::new(entry);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_creates_a_marker_and_the_store_directories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("myproject");
        let project = Project::init(&root, Some("Drønbreen 2022")).unwrap();

        assert!(root.join(MARKER).is_file());
        assert!(root.join(INTERPRETATIONS_DIR).is_dir());
        assert!(root.join(LAYERS_DIR).is_dir());
        assert_eq!(
            project.config().project.name.as_deref(),
            Some("Drønbreen 2022")
        );
    }

    #[test]
    fn init_refuses_to_overwrite_an_existing_project() {
        let dir = tempfile::tempdir().unwrap();
        Project::init(dir.path(), None).unwrap();
        assert!(matches!(
            Project::init(dir.path(), None),
            Err(ProjectError::AlreadyAProject(_))
        ));
    }

    #[test]
    fn a_directory_without_a_marker_is_not_a_project() {
        // The existing read-only behaviour depends on this: pointing Ridal
        // at a bare directory of .nc files must not turn it into a project.
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            Project::open(dir.path()),
            Err(ProjectError::NotAProject(_))
        ));
        assert!(Project::discover(dir.path()).unwrap().is_none());
    }

    #[test]
    fn discover_walks_up_from_a_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        Project::init(dir.path(), None).unwrap();
        let nested = dir.path().join("radargrams").join("2022").join("deep");
        std::fs::create_dir_all(&nested).unwrap();

        let found = Project::discover(&nested).unwrap().unwrap();
        assert_eq!(
            std::fs::canonicalize(found.root()).unwrap(),
            std::fs::canonicalize(dir.path()).unwrap()
        );
    }

    #[test]
    fn discover_walks_up_from_a_file_inside_the_project() {
        let dir = tempfile::tempdir().unwrap();
        Project::init(dir.path(), None).unwrap();
        let file = dir.path().join("radargrams").join("line.nc");
        std::fs::write(&file, b"not really a netcdf").unwrap();

        assert!(Project::discover(&file).unwrap().is_some());
    }

    #[test]
    fn radargram_roots_resolve_relative_and_absolute_entries() {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        assert_eq!(
            project.radargram_roots(),
            vec![dir.path().join(DEFAULT_RADARGRAM_DIR)]
        );

        // An absolute root is how a project indexes an archive it does not
        // contain.
        let text = format!(
            "[radargrams]\nroots = [\"inside\", \"{}\"]\n",
            "/mnt/archive/svalbard"
        );
        std::fs::write(dir.path().join(MARKER), text).unwrap();
        let project = Project::open(dir.path()).unwrap();
        assert_eq!(
            project.radargram_roots(),
            vec![
                dir.path().join("inside"),
                PathBuf::from("/mnt/archive/svalbard")
            ]
        );
    }

    #[test]
    fn radargram_roots_fall_back_to_the_project_root() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(MARKER), "[project]\nname = \"x\"\n").unwrap();
        let project = Project::open(dir.path()).unwrap();
        assert_eq!(project.radargram_roots(), vec![dir.path().to_path_buf()]);
    }

    #[test]
    fn the_cache_directory_can_be_moved_off_the_project() {
        // The daemon case: project on a network share, cache on local disk.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(MARKER),
            "[cache]\ndir = \"/var/cache/ridal\"\n",
        )
        .unwrap();
        let project = Project::open(dir.path()).unwrap();
        assert_eq!(project.cache_dir(), PathBuf::from("/var/cache/ridal"));
    }

    #[test]
    fn the_cache_directory_is_tagged_as_derived_data() {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        let cache = project.ensure_cache_dir().unwrap();

        let tag = std::fs::read_to_string(cache.join("CACHEDIR.TAG")).unwrap();
        assert!(
            tag.starts_with("Signature: 8a477f597d28d172789f06886806bc55"),
            "the signature line is what backup tools actually match on"
        );
        // Idempotent: opening a project twice must not fail on the tag.
        project.ensure_cache_dir().unwrap();
    }

    #[test]
    fn the_default_profile_round_trips_and_keeps_the_comments() {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), Some("x")).unwrap();
        assert_eq!(project.default_profile(), None);

        let before = std::fs::read_to_string(dir.path().join(MARKER)).unwrap();
        let comment_lines = before.lines().filter(|l| l.starts_with('#')).count();
        assert!(comment_lines > 5, "the template should be commented");

        project.set_default_profile(Some("abslog")).unwrap();

        // In memory straight away -- a save that only reached the file
        // would not take effect until a restart.
        assert_eq!(project.default_profile().as_deref(), Some("abslog"));
        // And on disk, for the next process.
        assert_eq!(
            Project::open(dir.path())
                .unwrap()
                .default_profile()
                .as_deref(),
            Some("abslog")
        );

        let after = std::fs::read_to_string(dir.path().join(MARKER)).unwrap();
        assert_eq!(
            after.lines().filter(|l| l.starts_with('#')).count(),
            comment_lines,
            "editing the file must not strip the comments in it:\n{after}"
        );
        // The other settings are untouched.
        assert!(after.contains("[radargrams]"), "{after}");
    }

    #[test]
    fn clearing_the_default_profile_removes_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        project.set_default_profile(Some("abslog")).unwrap();
        project.set_default_profile(None).unwrap();

        assert_eq!(project.default_profile(), None);
        let text = std::fs::read_to_string(dir.path().join(MARKER)).unwrap();
        // Only the commented example from the template should remain.
        let live = text
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .filter(|l| l.contains("default_profile"))
            .count();
        assert_eq!(live, 0, "{text}");
    }

    #[test]
    fn a_settings_write_leaves_no_temporary_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        project.set_default_profile(Some("positive")).unwrap();

        let strays: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp"))
            .collect();
        assert!(strays.is_empty(), "{strays:?}");
    }

    #[test]
    fn an_unparseable_marker_is_reported_rather_than_ignored() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(MARKER), "this is not toml {{{").unwrap();
        assert!(matches!(
            Project::open(dir.path()),
            Err(ProjectError::Config { .. })
        ));
    }

    #[test]
    fn unknown_config_keys_are_tolerated() {
        // Forward compatibility: a project written by a newer Ridal should
        // still open in an older one rather than refusing outright.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(MARKER),
            "[project]\nname = \"x\"\n\n[future]\nsomething = 1\n",
        )
        .unwrap();
        assert!(Project::open(dir.path()).is_ok());
    }
}
