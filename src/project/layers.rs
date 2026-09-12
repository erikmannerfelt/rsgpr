//! The project's layer vocabulary: `layers/layers.json`.
//!
//! A layer is what `properties.label` on a gprinterp feature refers to --
//! "bed", "internal reflector", "englacial water". The vocabulary is
//! project-scoped and user-editable, because a fixed server-side list will
//! not survive contact with real interpretation work.
//!
//! # Why a layer has both an id and a name
//!
//! `id` is a slug and is what gets written into `properties.label`, so it is
//! also what appears in the `layer` column of a level 2 export. `name` is
//! free text for display. Separating them means renaming a layer in the GUI
//! -- "bed" to "Bed (picked 2026)" -- is a cosmetic change that does not
//! rewrite, or orphan, a single existing pick.
//!
//! # Deleting
//!
//! Removing a layer from the vocabulary does not touch the interpretations
//! that used it; their features keep their label. Such a label is reported
//! as unknown rather than being dropped, since the picks are real data and
//! the vocabulary is only a description of it.

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

use serde::{Deserialize, Serialize};

use crate::project::store::{DocumentStore, Expectation, StoreError, Version};

pub const FILE: &str = "layers.json";
pub const SCHEMA: &str = "ridal-layers";
pub const SCHEMA_VERSION: &str = "1";

/// One layer definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Stable slug. Written into gprinterp `properties.label`.
    pub id: String,
    /// Display name. Cosmetic; safe to change at any time.
    pub name: String,
    /// CSS colour used to draw the layer in the viewer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Permit lines in this layer to double back in trace.
    ///
    /// Off by default, because a reflector has one depth per position and a
    /// line that overhangs is usually a mis-click. Some layers legitimately
    /// do overhang -- a crevasse wall, a water-body outline -- which is why
    /// it is a property of the layer rather than of Ridal.
    ///
    /// Turning it on has a cost: such a layer cannot be exported at even
    /// spacing along the ground track, because that works by asking the line
    /// for its depth at a position, which is the question an overhang has
    /// two answers to. It exports as its own picked vertices instead. See
    /// [`crate::interp::checks`].
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_overhangs: bool,
    /// Anything a future version adds, preserved on rewrite.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// The whole vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSet {
    #[serde(default = "default_schema")]
    pub schema: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub layers: Vec<Layer>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_schema() -> String {
    SCHEMA.to_string()
}

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_string()
}

impl Default for LayerSet {
    fn default() -> Self {
        LayerSet {
            schema: default_schema(),
            schema_version: default_schema_version(),
            layers: Vec::new(),
            extra: serde_json::Map::new(),
        }
    }
}

impl LayerSet {
    pub fn get(&self, id: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Whether `label` names a layer that permits overhangs.
    ///
    /// An undefined or missing label answers `false`: opting out of the
    /// guardrail is a deliberate act recorded on a layer, and a label with
    /// no definition has not made it.
    pub fn allows_overhangs(&self, label: Option<&str>) -> bool {
        label
            .and_then(|id| self.get(id))
            .map(|layer| layer.allow_overhangs)
            .unwrap_or(false)
    }

    /// Ids referenced by `labels` that this vocabulary does not define.
    pub fn unknown_ids<'a>(&self, labels: impl Iterator<Item = &'a str>) -> Vec<String> {
        let mut unknown: Vec<String> = labels
            .filter(|label| self.get(label).is_none())
            .map(str::to_string)
            .collect();
        unknown.sort();
        unknown.dedup();
        unknown
    }

    /// Reject a set that cannot be used unambiguously.
    ///
    /// Duplicate ids are fatal rather than deduplicated: two definitions of
    /// "bed" means the colour a pick draws in depends on iteration order,
    /// and silently keeping one discards a real edit.
    pub fn validate(&self) -> Result<(), LayerError> {
        let mut seen: Vec<&str> = Vec::new();
        for layer in &self.layers {
            if layer.id.is_empty() {
                return Err(LayerError::InvalidId {
                    id: layer.id.clone(),
                    reason: "a layer id must not be empty".to_string(),
                });
            }
            // Same charset as the identity slugs: layer ids end up in
            // exported CSV columns, GeoJSON properties and URLs.
            if let Some(bad) = layer
                .id
                .chars()
                .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-' || *c == '_'))
            {
                return Err(LayerError::InvalidId {
                    id: layer.id.clone(),
                    reason: format!(
                        "'{bad}' is not allowed; use lowercase ASCII letters, digits, \
                         '-' and '_'. The display name is free text -- put the \
                         punctuation there instead."
                    ),
                });
            }
            if let Some(color) = &layer.color {
                if !is_hex_color(color) {
                    return Err(LayerError::InvalidColor {
                        id: layer.id.clone(),
                        color: color.clone(),
                    });
                }
            }
            if seen.contains(&layer.id.as_str()) {
                return Err(LayerError::DuplicateId(layer.id.clone()));
            }
            seen.push(&layer.id);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum LayerError {
    Store(StoreError),
    Malformed { message: String },
    DuplicateId(String),
    InvalidId { id: String, reason: String },
    InvalidColor { id: String, color: String },
}

/// Is this `#rgb`, `#rrggbb` or `#rrggbbaa`?
///
/// Colours are written into `style.background` in the browser, where CSS
/// accepts far more than a colour -- `url(http://…)` would make every
/// viewer fetch whatever the author chose. The colour input in the UI
/// constrains the widget, not the API or a hand-edited `layers.json`, so
/// the restriction has to live at the boundary that stores it.
fn is_hex_color(value: &str) -> bool {
    let Some(digits) = value.strip_prefix('#') else {
        return false;
    };
    matches!(digits.len(), 3 | 6 | 8) && digits.chars().all(|c| c.is_ascii_hexdigit())
}

impl std::fmt::Display for LayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerError::Store(e) => write!(f, "{e}"),
            LayerError::Malformed { message } => {
                write!(f, "the layer definitions are not readable: {message}")
            }
            LayerError::InvalidColor { id, color } => write!(
                f,
                "layer '{id}' has colour '{color}'; use a hex colour such as \
                 '#e6194b'. Anything else would be handed to CSS, which accepts \
                 more than colours."
            ),
            LayerError::DuplicateId(id) => {
                write!(f, "the layer id '{id}' is defined more than once")
            }
            LayerError::InvalidId { id, reason } => {
                write!(f, "the layer id '{id}' is not usable: {reason}")
            }
        }
    }
}

impl std::error::Error for LayerError {}

impl From<StoreError> for LayerError {
    fn from(e: StoreError) -> Self {
        LayerError::Store(e)
    }
}

fn path() -> PathBuf {
    PathBuf::from(crate::project::LAYERS_DIR).join(FILE)
}

/// Read the vocabulary. A project that has never defined one reads as empty
/// with no version, which is a normal state rather than an error.
pub fn read(store: &DocumentStore) -> Result<(LayerSet, Option<Version>), LayerError> {
    let Some(stored) = store.read(&path())? else {
        return Ok((LayerSet::default(), None));
    };
    let set: LayerSet = serde_json::from_str(&stored.text).map_err(|e| LayerError::Malformed {
        message: e.to_string(),
    })?;
    // Validated on the way out as well as on the way in. `write` is not
    // the only way a document gets here: `layers.json` is meant to be
    // hand-editable, and an invalid colour edited in by hand would
    // otherwise reach `style.background` in the browser, where CSS accepts
    // considerably more than a colour.
    set.validate()?;
    Ok((set, Some(stored.version)))
}

/// Replace the vocabulary.
pub fn write(
    store: &DocumentStore,
    set: &LayerSet,
    expected: &Expectation,
) -> Result<Version, LayerError> {
    set.validate()?;
    let text = serde_json::to_string_pretty(set).map_err(|e| LayerError::Malformed {
        message: e.to_string(),
    })?;
    Ok(store.write(&path(), &text, expected)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::Project;

    fn project() -> (tempfile::TempDir, Project) {
        let dir = tempfile::tempdir().unwrap();
        let project = Project::init(dir.path(), None).unwrap();
        (dir, project)
    }

    fn layer(id: &str, name: &str) -> Layer {
        Layer {
            id: id.to_string(),
            name: name.to_string(),
            color: Some("#e6194b".to_string()),
            description: None,
            allow_overhangs: false,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn a_colour_must_be_hex_because_css_accepts_more_than_colours() {
        // The value lands in `style.background` in the browser, where
        // `url(http://...)` would make every viewer fetch whatever the
        // author chose. The colour picker in the UI constrains the widget,
        // not the API or a hand-edited layers.json.
        let mut set = LayerSet::default();
        set.layers.push(Layer {
            id: "bed".to_string(),
            name: "Bed".to_string(),
            color: Some("url(http://example.invalid/x.png)".to_string()),
            description: None,
            allow_overhangs: false,
            extra: Default::default(),
        });
        let err = set.validate().expect_err("must refuse");
        assert!(matches!(err, LayerError::InvalidColor { .. }), "{err}");
        assert!(err.to_string().contains("hex colour"), "{err}");

        for good in ["#fff", "#e6194b", "#e6194bcc", "#ABCDEF"] {
            set.layers[0].color = Some(good.to_string());
            assert!(set.validate().is_ok(), "{good} should be accepted");
        }
        for bad in ["red", "#12", "#1234567", "#ggghhh", "rgb(1,2,3)", ""] {
            set.layers[0].color = Some(bad.to_string());
            assert!(set.validate().is_err(), "{bad:?} should be refused");
        }

        // No colour at all stays legal -- the viewer falls back to its own.
        set.layers[0].color = None;
        assert!(set.validate().is_ok());
    }

    #[test]
    fn a_project_without_layers_reads_as_empty() {
        let (_dir, project) = project();
        let (set, version) = read(project.documents()).unwrap();
        assert!(set.layers.is_empty());
        assert!(version.is_none());
    }

    #[test]
    fn layers_round_trip() {
        let (_dir, project) = project();
        let set = LayerSet {
            layers: vec![layer("bed", "Bed"), layer("internal", "Internal reflector")],
            ..LayerSet::default()
        };
        let version = write(project.documents(), &set, &Expectation::Absent).unwrap();

        let (read_back, read_version) = read(project.documents()).unwrap();
        assert_eq!(read_back.layers, set.layers);
        assert_eq!(read_version, Some(version));
        assert_eq!(read_back.schema, SCHEMA);
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let (_dir, project) = project();
        let set = LayerSet {
            layers: vec![layer("bed", "Bed"), layer("bed", "Bed again")],
            ..LayerSet::default()
        };
        let error = write(project.documents(), &set, &Expectation::Any).unwrap_err();
        assert!(matches!(error, LayerError::DuplicateId(id) if id == "bed"));
    }

    #[test]
    fn ids_are_restricted_but_names_are_free_text() {
        let (_dir, project) = project();
        let bad = LayerSet {
            layers: vec![layer("Bed Layer", "Bed")],
            ..LayerSet::default()
        };
        assert!(matches!(
            write(project.documents(), &bad, &Expectation::Any),
            Err(LayerError::InvalidId { .. })
        ));

        let good = LayerSet {
            layers: vec![layer("bed", "Bed (Drønbreen, picked 2026) — upper")],
            ..LayerSet::default()
        };
        assert!(write(project.documents(), &good, &Expectation::Any).is_ok());
    }

    #[test]
    fn renaming_a_layer_leaves_its_id_alone() {
        // The whole point of the id/name split: picks reference the id.
        let (_dir, project) = project();
        let mut set = LayerSet {
            layers: vec![layer("bed", "Bed")],
            ..LayerSet::default()
        };
        let version = write(project.documents(), &set, &Expectation::Absent).unwrap();

        set.layers[0].name = "Glacier bed".to_string();
        write(project.documents(), &set, &Expectation::Version(version)).unwrap();

        let (read_back, _) = read(project.documents()).unwrap();
        assert_eq!(read_back.layers[0].id, "bed");
        assert_eq!(read_back.layers[0].name, "Glacier bed");
    }

    #[test]
    fn labels_with_no_definition_are_reported_not_dropped() {
        let set = LayerSet {
            layers: vec![layer("bed", "Bed")],
            ..LayerSet::default()
        };
        let unknown = set.unknown_ids(["bed", "englacial", "englacial", "moraine"].into_iter());
        assert_eq!(unknown, vec!["englacial", "moraine"]);
    }

    #[test]
    fn unknown_fields_survive_a_rewrite() {
        let (_dir, project) = project();
        let stored = serde_json::json!({
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "layers": [{"id": "bed", "name": "Bed", "future_field": 42}],
            "future_top_level": true
        });
        project
            .documents()
            .write(
                &path(),
                &serde_json::to_string_pretty(&stored).unwrap(),
                &Expectation::Absent,
            )
            .unwrap();

        let (set, version) = read(project.documents()).unwrap();
        write(
            project.documents(),
            &set,
            &Expectation::Version(version.unwrap()),
        )
        .unwrap();

        let (again, _) = read(project.documents()).unwrap();
        assert_eq!(
            again.layers[0].extra.get("future_field"),
            Some(&serde_json::json!(42))
        );
        assert_eq!(
            again.extra.get("future_top_level"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn a_concurrent_edit_is_refused() {
        let (_dir, project) = project();
        let set = LayerSet {
            layers: vec![layer("bed", "Bed")],
            ..LayerSet::default()
        };
        let first = write(project.documents(), &set, &Expectation::Absent).unwrap();
        write(
            project.documents(),
            &set,
            &Expectation::Version(first.clone()),
        )
        .unwrap();

        // `first` is now stale only if the content changed; it did not, so
        // rewrite with a genuinely different set to move the version on.
        let changed = LayerSet {
            layers: vec![layer("bed", "Bed"), layer("moraine", "Moraine")],
            ..LayerSet::default()
        };
        write(
            project.documents(),
            &changed,
            &Expectation::Version(first.clone()),
        )
        .unwrap();

        let error = write(project.documents(), &set, &Expectation::Version(first)).unwrap_err();
        assert!(matches!(
            error,
            LayerError::Store(StoreError::Conflict { .. })
        ));
    }
}
