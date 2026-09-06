//! Geometry rules an interpretation must satisfy before it is stored.
//!
//! # Overhangs
//!
//! A picked line normally represents a reflector: one depth per position
//! along the profile. Expressed in index space, that means the line is a
//! *function of trace* -- no two vertices share a trace, and it never
//! doubles back. A line that does is an **overhang**, and it is almost
//! always a mis-click rather than an intention: the moment it exists,
//! "how deep is the bed at 400 m?" stops having one answer.
//!
//! PFA_website enforced this unconditionally. Ridal makes it a per-layer
//! setting that defaults to enforcing, because the constraint belongs to
//! what a layer *means* rather than to the tool: a bed horizon must be a
//! function of trace, while a crevasse wall or a water-body outline
//! legitimately is not.
//!
//! # The cost of allowing them
//!
//! Allowing overhangs on a layer and exporting that layer at even spacing
//! along the ground track are mutually exclusive. Even spacing works by
//! inverting distance to a trace and asking the line for its sample there,
//! which is exactly the question an overhang has two answers to. A layer
//! that permits overhangs is therefore exported as its own picked vertices
//! instead (see [`crate::interp::level2::Spacing::Vertices`]). This is not
//! a limitation to be engineered around later -- it is what the geometry
//! means.

#![cfg_attr(
    not(feature = "server"),
    allow(
        dead_code,
        reason = "the save-time guardrail runs in the server's write route; \
                  a CLI-only build still uses the same check at export time"
    )
)]

use gprinterp::{Document, Geometry, Position};

/// A rule an interpretation breaks.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// A line doubles back in trace, so it is not a function of trace.
    Overhang {
        feature_index: usize,
        feature_id: Option<String>,
        layer: Option<String>,
        /// Index of the vertex that reverses or repeats a trace.
        at_vertex: usize,
        trace: f64,
    },
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::Overhang {
                feature_index,
                feature_id,
                layer,
                at_vertex,
                trace,
            } => {
                let which = feature_id
                    .as_deref()
                    .map(|id| format!("feature '{id}'"))
                    .unwrap_or_else(|| format!("feature {feature_index}"));
                let layer = layer.as_deref().unwrap_or("<unlabelled>");
                write!(
                    f,
                    "{which} in layer '{layer}' overhangs: vertex {at_vertex} returns to \
                     trace {trace}, so the layer has two depths at that position. Split \
                     it into separate lines, or allow overhangs on this layer if that is \
                     intended."
                )
            }
        }
    }
}

/// Where a line stops being a function of trace, if it does.
///
/// A line drawn right to left is not an overhang -- it is the same
/// interpretation recorded in the opposite order, so the check runs against
/// the line's own direction, taken from its first and last vertices.
/// Vertical segments (two vertices on one trace) *are* overhangs: they are
/// the degenerate case of two depths at one position.
pub fn overhang_at(positions: &[Position]) -> Option<(usize, f64)> {
    let traces: Vec<f64> = positions.iter().filter_map(|p| p.x()).collect();
    if traces.len() < 2 {
        return None;
    }
    let descending = traces[traces.len() - 1] < traces[0];

    for (i, window) in traces.windows(2).enumerate() {
        let (previous, current) = (window[0], window[1]);
        let advances = if descending {
            current < previous
        } else {
            current > previous
        };
        if !advances {
            return Some((i + 1, current));
        }
    }
    None
}

/// Check every line in `document`, skipping layers that permit overhangs.
///
/// `allows_overhangs` is a predicate over the layer label rather than a
/// [`crate::project::layers::LayerSet`], so this stays usable from the CLI,
/// where an export may have no project and therefore no vocabulary at all.
/// A feature with no label, or one naming a layer the vocabulary does not
/// define, is checked: the guardrail is the default, and an undefined layer
/// has not opted out of anything.
pub fn check(
    document: &Document,
    allows_overhangs: &dyn Fn(Option<&str>) -> bool,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    for (feature_index, feature) in document.features.iter().enumerate() {
        let layer = feature.label();
        if allows_overhangs(layer) {
            continue;
        }
        let Geometry::LineString(positions) = &feature.geometry else {
            // Only lines carry the function-of-trace expectation. Points
            // cannot overhang, and polygons are expected to close.
            continue;
        };
        if let Some((at_vertex, trace)) = overhang_at(positions) {
            violations.push(Violation::Overhang {
                feature_index,
                feature_id: feature.id().map(str::to_string),
                layer: layer.map(str::to_string),
                at_vertex,
                trace,
            });
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(coords: &[[f64; 2]]) -> Vec<Position> {
        coords.iter().map(|c| Position(c.to_vec())).collect()
    }

    fn document(features: &[(&str, &[[f64; 2]])]) -> Document {
        let features: Vec<serde_json::Value> = features
            .iter()
            .enumerate()
            .map(|(i, (label, coords))| {
                serde_json::json!({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"id": format!("f-{i}"), "label": label}
                })
            })
            .collect();
        serde_json::from_value(serde_json::json!({"key": "line-01", "features": features})).unwrap()
    }

    fn enforce_everywhere(_: Option<&str>) -> bool {
        false
    }

    #[test]
    fn a_rising_line_is_fine() {
        assert_eq!(
            overhang_at(&line(&[[0.0, 10.0], [50.0, 12.0], [100.0, 20.0]])),
            None
        );
    }

    #[test]
    fn a_line_drawn_right_to_left_is_not_an_overhang() {
        // Same interpretation, opposite click order.
        assert_eq!(
            overhang_at(&line(&[[100.0, 20.0], [50.0, 12.0], [0.0, 10.0]])),
            None
        );
    }

    #[test]
    fn a_line_that_doubles_back_is_caught_at_the_offending_vertex() {
        let found = overhang_at(&line(&[[0.0, 10.0], [50.0, 12.0], [30.0, 14.0]]));
        assert_eq!(found, Some((2, 30.0)));
    }

    #[test]
    fn a_vertical_segment_is_an_overhang() {
        // Two depths at one trace is the degenerate case, not an exception.
        assert_eq!(
            overhang_at(&line(&[[0.0, 10.0], [50.0, 12.0], [50.0, 30.0]])),
            Some((2, 50.0))
        );
    }

    #[test]
    fn a_two_vertex_line_and_a_single_point_cannot_overhang() {
        assert_eq!(overhang_at(&line(&[[0.0, 10.0], [50.0, 12.0]])), None);
        assert_eq!(overhang_at(&line(&[[0.0, 10.0]])), None);
    }

    #[test]
    fn checking_a_document_reports_the_feature_and_its_layer() {
        let doc = document(&[
            ("bed", &[[0.0, 10.0], [100.0, 20.0]]),
            ("bed", &[[0.0, 10.0], [50.0, 12.0], [30.0, 14.0]]),
        ]);
        let violations = check(&doc, &enforce_everywhere);
        assert_eq!(violations.len(), 1);
        match &violations[0] {
            Violation::Overhang {
                feature_index,
                feature_id,
                layer,
                at_vertex,
                ..
            } => {
                assert_eq!(*feature_index, 1);
                assert_eq!(feature_id.as_deref(), Some("f-1"));
                assert_eq!(layer.as_deref(), Some("bed"));
                assert_eq!(*at_vertex, 2);
            }
        }
    }

    #[test]
    fn a_layer_that_allows_overhangs_is_skipped() {
        let doc = document(&[("crevasse", &[[0.0, 10.0], [50.0, 12.0], [30.0, 14.0]])]);
        let allows = |layer: Option<&str>| layer == Some("crevasse");
        assert!(check(&doc, &allows).is_empty());
        assert_eq!(check(&doc, &enforce_everywhere).len(), 1);
    }

    #[test]
    fn an_undefined_or_missing_layer_is_still_checked() {
        // Opting out is a deliberate act. A label the vocabulary does not
        // define has not opted out of anything.
        let doc = document(&[(
            "not_in_the_vocabulary",
            &[[0.0, 1.0], [5.0, 2.0], [3.0, 3.0]],
        )]);
        let allows = |layer: Option<&str>| layer == Some("crevasse");
        assert_eq!(check(&doc, &allows).len(), 1);

        let unlabelled: Document = serde_json::from_value(serde_json::json!({
            "key": "line-01",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0.0, 1.0], [5.0, 2.0], [3.0, 3.0]]}
            }]
        }))
        .unwrap();
        assert_eq!(check(&unlabelled, &allows).len(), 1);
    }

    #[test]
    fn non_line_geometries_are_not_subject_to_the_rule() {
        let doc: Document = serde_json::from_value(serde_json::json!({
            "key": "line-01",
            "features": [
                {"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                 "properties": {"label": "poi"}},
                {"type": "Feature",
                 "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 0.0]]]},
                 "properties": {"label": "lake"}}
            ]
        }))
        .unwrap();
        assert!(check(&doc, &enforce_everywhere).is_empty());
    }

    #[test]
    fn the_message_names_what_to_do_about_it() {
        let doc = document(&[("bed", &[[0.0, 10.0], [50.0, 12.0], [30.0, 14.0]])]);
        let message = check(&doc, &enforce_everywhere)[0].to_string();
        assert!(message.contains("f-0"), "{message}");
        assert!(message.contains("bed"), "{message}");
        assert!(
            message.contains("Split it into separate lines"),
            "{message}"
        );
    }
}
