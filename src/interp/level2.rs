//! Deriving the level 2 point product from a level 1 interpretation.
//!
//! A level 1 feature is a polyline in (trace, sample) index space. Level 2
//! resamples it to points spaced evenly **along the ground track in metres**
//! and attaches everything needed to use them outside Ridal: layer name,
//! line index, the index-space position, travel time, depth, and both
//! projected and geographic coordinates.
//!
//! Even spacing is measured in metres of along-track distance, never in
//! traces. Trace spacing is not constant -- it varies with survey speed, and
//! GPS noise perturbs it further -- so "every 10th trace" and "every 2 m"
//! are different products, and only the second is meaningful once the points
//! leave the radargram's index space. Per-trace export remains available via
//! [`Spacing::PerTrace`] for callers who want the native sampling.
//!
//! This module is pure: it takes a [`RadargramGeometry`] rather than reading
//! a NetCDF itself, so the resampling and lookup logic is testable against
//! synthetic geometry with no file I/O. The adapter that builds a
//! `RadargramGeometry` from a processed file lives with the NetCDF code.

use std::fmt;

use gprinterp::{Document, Geometry, Position};

/// The default author recorded on exported points until Ridal has real
/// multi-user support.
pub const DEFAULT_USER: &str = "default";

/// Everything about one processed radargram that level 2 derivation needs.
///
/// All per-trace vectors have length `n_traces` and all per-sample vectors
/// have length `n_samples`; [`RadargramGeometry::validate`] checks this,
/// since a mismatched axis would otherwise surface as a silently truncated
/// export.
#[derive(Debug, Clone)]
pub struct RadargramGeometry {
    pub radargram_id: String,
    pub revision_id: String,
    /// Along-track distance per trace, in metres, non-decreasing.
    pub distance: Vec<f64>,
    /// Two-way travel time per sample, in nanoseconds.
    pub twtt: Vec<f64>,
    /// Depth per sample, in metres.
    pub depth: Vec<f64>,
    /// Projected position per trace, in the units of `crs`.
    pub easting: Vec<f64>,
    pub northing: Vec<f64>,
    /// WGS84 position per trace, in degrees.
    pub longitude: Vec<f64>,
    pub latitude: Vec<f64>,
    /// The projected CRS `easting`/`northing` are expressed in.
    pub crs: String,
}

impl RadargramGeometry {
    pub fn n_traces(&self) -> usize {
        self.distance.len()
    }

    pub fn n_samples(&self) -> usize {
        self.twtt.len()
    }

    /// Check the axis lengths agree and the distance axis is usable.
    pub fn validate(&self) -> Result<(), Level2Error> {
        let n = self.n_traces();
        if n < 2 {
            return Err(Level2Error::DegenerateRadargram { n_traces: n });
        }
        for (name, len) in [
            ("easting", self.easting.len()),
            ("northing", self.northing.len()),
            ("longitude", self.longitude.len()),
            ("latitude", self.latitude.len()),
        ] {
            if len != n {
                return Err(Level2Error::AxisLengthMismatch {
                    axis: name,
                    found: len,
                    expected: n,
                });
            }
        }
        if self.depth.len() != self.n_samples() {
            return Err(Level2Error::AxisLengthMismatch {
                axis: "depth",
                found: self.depth.len(),
                expected: self.n_samples(),
            });
        }
        if self.n_samples() < 2 {
            return Err(Level2Error::DegenerateRadargram { n_traces: n });
        }
        // Non-decreasing is required to invert distance -> trace. Equal
        // consecutive values (a standstill) are fine; the inversion resolves
        // them to the first matching trace.
        if self.distance.windows(2).any(|w| w[1] < w[0]) {
            return Err(Level2Error::NonMonotoneDistance);
        }
        Ok(())
    }
}

/// How densely to sample each picked line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spacing {
    /// Evenly spaced along the ground track, in metres.
    ArcLength(f64),
    /// Evenly spaced along the ground track, at a step derived from the
    /// radargram's own median trace spacing (see [`auto_step`]).
    Auto,
    /// One point per native trace the line spans. No resampling.
    PerTrace,
}

/// One exported level 2 point.
#[derive(Debug, Clone, PartialEq)]
pub struct Level2Point {
    /// `properties.label` of the source feature: the interpreted layer.
    pub layer: String,
    /// Which physically separate line within that layer, in document order.
    pub line_index: usize,
    /// Position of this point along its line.
    pub point_index: usize,
    /// Stable feature identity from level 1, when the document carried one.
    pub feature_id: Option<String>,
    /// Fractional index-space position in the source revision.
    pub trace: f64,
    pub sample: f64,
    pub distance_m: f64,
    pub twtt_ns: f64,
    pub depth_m: f64,
    pub easting: f64,
    pub northing: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub user: String,
}

/// A level 2 export: the points plus the provenance needed to interpret
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct Level2Export {
    pub points: Vec<Level2Point>,
    pub radargram_id: String,
    pub revision_id: String,
    pub crs: String,
    /// The spacing actually used, in metres. `None` for a per-trace export.
    pub spacing_m: Option<f64>,
}

/// Why an export could not be produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Level2Error {
    AxisLengthMismatch {
        axis: &'static str,
        found: usize,
        expected: usize,
    },
    DegenerateRadargram {
        n_traces: usize,
    },
    /// The `distance` axis decreases somewhere, so distance cannot be
    /// inverted to a trace index.
    NonMonotoneDistance,
    /// A picked line doubles back on itself in trace, so it is not a
    /// function of trace and cannot be resampled onto a distance grid.
    NonMonotoneLine {
        layer: String,
        line_index: usize,
        at_vertex: usize,
    },
    /// Level 2 is defined for lines. Other geometry types are valid level 1
    /// (SPEC §4.3) and round-trip fine; they are simply not exportable yet.
    UnsupportedGeometry {
        layer: String,
        type_name: String,
    },
    /// A spacing that cannot produce points.
    InvalidSpacing(f64),
}

impl fmt::Display for Level2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Level2Error::AxisLengthMismatch {
                axis,
                found,
                expected,
            } => write!(
                f,
                "the '{axis}' axis has {found} values but the radargram has {expected} \
                 -- the file's coordinate variables disagree with its data shape"
            ),
            Level2Error::DegenerateRadargram { n_traces } => write!(
                f,
                "a radargram with {n_traces} trace(s) has no along-track extent to export along"
            ),
            Level2Error::NonMonotoneDistance => write!(
                f,
                "the 'distance' axis decreases somewhere, so a distance cannot be \
                 resolved to a single trace"
            ),
            Level2Error::NonMonotoneLine {
                layer,
                line_index,
                at_vertex,
            } => write!(
                f,
                "line {line_index} of layer '{layer}' reverses direction at vertex \
                 {at_vertex}. A layer must be a function of trace to be resampled \
                 onto an evenly spaced grid; split it into separate lines, or \
                 export with per-trace spacing."
            ),
            Level2Error::UnsupportedGeometry { layer, type_name } => write!(
                f,
                "layer '{layer}' contains a {type_name}, but level 2 export currently \
                 supports lines only"
            ),
            Level2Error::InvalidSpacing(step) => {
                write!(f, "a spacing of {step} m cannot produce points")
            }
        }
    }
}

impl std::error::Error for Level2Error {}

/// Derive the level 2 product from a level 1 document.
pub fn export(
    document: &Document,
    geometry: &RadargramGeometry,
    spacing: Spacing,
    user: &str,
) -> Result<Level2Export, Level2Error> {
    geometry.validate()?;

    let step = match spacing {
        Spacing::ArcLength(step) => {
            if !step.is_finite() || step <= 0.0 {
                return Err(Level2Error::InvalidSpacing(step));
            }
            Some(step)
        }
        Spacing::Auto => Some(auto_step(&geometry.distance)),
        Spacing::PerTrace => None,
    };

    let mut points = Vec::new();
    for (layer, features) in document.layers() {
        let layer = layer.unwrap_or("unlabeled").to_string();
        for (line_index, feature) in features.iter().enumerate() {
            let vertices = match &feature.geometry {
                Geometry::LineString(vertices) => vertices.clone(),
                other => {
                    return Err(Level2Error::UnsupportedGeometry {
                        layer,
                        type_name: other.type_name().to_string(),
                    })
                }
            };
            let line = Line::new(&vertices, &layer, line_index)?;
            let feature_id = feature.id().map(str::to_string);
            points.extend(line.sample(geometry, step, &layer, line_index, &feature_id, user));
        }
    }

    Ok(Level2Export {
        points,
        radargram_id: geometry.radargram_id.clone(),
        revision_id: geometry.revision_id.clone(),
        crs: geometry.crs.clone(),
        spacing_m: step,
    })
}

/// A picked line, normalized to be increasing in trace.
struct Line {
    /// (trace, sample), strictly increasing in trace.
    vertices: Vec<(f64, f64)>,
}

impl Line {
    fn new(positions: &[Position], layer: &str, line_index: usize) -> Result<Line, Level2Error> {
        let mut vertices: Vec<(f64, f64)> = positions
            .iter()
            .filter_map(|p| Some((p.x()?, p.y()?)))
            .collect();

        // A line drawn right-to-left is the same interpretation as one drawn
        // left-to-right, so reversing is a normalization rather than a
        // correction. A line that reverses *within* itself is not, and is
        // rejected below.
        if vertices.len() >= 2 && vertices[vertices.len() - 1].0 < vertices[0].0 {
            vertices.reverse();
        }

        for (i, w) in vertices.windows(2).enumerate() {
            if w[1].0 <= w[0].0 {
                return Err(Level2Error::NonMonotoneLine {
                    layer: layer.to_string(),
                    line_index,
                    at_vertex: i + 1,
                });
            }
        }
        Ok(Line { vertices })
    }

    fn trace_span(&self) -> Option<(f64, f64)> {
        Some((self.vertices.first()?.0, self.vertices.last()?.0))
    }

    /// Sample index at a fractional trace, along the polyline.
    fn sample_at(&self, trace: f64) -> f64 {
        let traces: Vec<f64> = self.vertices.iter().map(|v| v.0).collect();
        let samples: Vec<f64> = self.vertices.iter().map(|v| v.1).collect();
        interpolate(&traces, &samples, trace)
    }

    fn sample(
        &self,
        geometry: &RadargramGeometry,
        step: Option<f64>,
        layer: &str,
        line_index: usize,
        feature_id: &Option<String>,
        user: &str,
    ) -> Vec<Level2Point> {
        let Some((first_trace, last_trace)) = self.trace_span() else {
            return Vec::new();
        };

        let traces: Vec<f64> = match step {
            // Evenly spaced in metres: walk the distance axis, then invert
            // each target distance back to a fractional trace.
            Some(step) => {
                let d0 = interpolate_index(&geometry.distance, first_trace);
                let d1 = interpolate_index(&geometry.distance, last_trace);
                let span = d1 - d0;
                let n = if span <= 0.0 {
                    // A line drawn entirely within a standstill has no
                    // along-track extent. One point still describes it.
                    1
                } else {
                    (span / step).floor() as usize + 1
                };
                (0..n)
                    .map(|i| {
                        let target = d0 + i as f64 * step;
                        invert_axis(&geometry.distance, target).clamp(first_trace, last_trace)
                    })
                    .collect()
            }
            // Per-trace: every native trace the line spans, inclusive.
            None => {
                let lo = first_trace.ceil().max(0.0) as usize;
                let hi = (last_trace.floor() as usize).min(geometry.n_traces() - 1);
                (lo..=hi.max(lo)).map(|t| t as f64).collect()
            }
        };

        traces
            .into_iter()
            .enumerate()
            .map(|(point_index, trace)| {
                let sample = self.sample_at(trace);
                Level2Point {
                    layer: layer.to_string(),
                    line_index,
                    point_index,
                    feature_id: feature_id.clone(),
                    trace,
                    sample,
                    distance_m: interpolate_index(&geometry.distance, trace),
                    twtt_ns: interpolate_index(&geometry.twtt, sample),
                    depth_m: interpolate_index(&geometry.depth, sample),
                    easting: interpolate_index(&geometry.easting, trace),
                    northing: interpolate_index(&geometry.northing, trace),
                    longitude: interpolate_index(&geometry.longitude, trace),
                    latitude: interpolate_index(&geometry.latitude, trace),
                    user: user.to_string(),
                }
            })
            .collect()
    }
}

/// A tidy spacing derived from the radargram's own trace density.
///
/// The median *moving* step is used rather than the mean: a survey that
/// stopped for a while has a cluster of near-zero steps that drag the mean
/// down and would produce an absurdly dense export. Zero-length steps are
/// excluded outright for the same reason.
///
/// The result is snapped to a tidy ladder so that exports from similar
/// surveys share a grid and the number is presentable in a filename or a
/// legend.
pub fn auto_step(distance: &[f64]) -> f64 {
    let mut steps: Vec<f64> = distance
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0 && d.is_finite())
        .collect();
    if steps.is_empty() {
        return 1.0;
    }
    steps.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite values"));
    let median = steps[steps.len() / 2];

    const LADDER: [f64; 12] = [
        0.1, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0,
    ];
    for candidate in LADDER {
        if median <= candidate {
            return candidate;
        }
    }
    *LADDER.last().expect("ladder is not empty")
}

/// Value of a per-index axis at a fractional index, clamped at both ends.
///
/// Clamping is right here, unlike in re-anchoring: the index came from a
/// pick on *this* radargram, so a fractional index at the very last trace is
/// an edge effect of interpolation rather than a coordinate from somewhere
/// else.
fn interpolate_index(axis: &[f64], index: f64) -> f64 {
    if axis.is_empty() {
        return f64::NAN;
    }
    let xs: Vec<f64> = (0..axis.len()).map(|i| i as f64).collect();
    interpolate(&xs, axis, index.clamp(0.0, (axis.len() - 1) as f64))
}

/// Invert a non-decreasing axis: find the fractional index whose value is
/// `target`.
fn invert_axis(axis: &[f64], target: f64) -> f64 {
    if axis.len() < 2 {
        return 0.0;
    }
    let i = axis.partition_point(|v| *v < target);
    if i == 0 {
        return 0.0;
    }
    if i >= axis.len() {
        return (axis.len() - 1) as f64;
    }
    let (lo, hi) = (i - 1, i);
    let span = axis[hi] - axis[lo];
    if span <= 0.0 {
        // A standstill: many traces share this distance. The first is the
        // only defensible choice, and picking the last would silently skip
        // the stationary traces.
        return lo as f64;
    }
    lo as f64 + (target - axis[lo]) / span
}

/// Piecewise-linear interpolation over strictly increasing `xs`.
fn interpolate(xs: &[f64], ys: &[f64], at: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return f64::NAN;
    }
    if xs.len() == 1 {
        return ys[0];
    }
    let i = xs.partition_point(|v| *v <= at);
    let (lo, hi) = if i == 0 {
        (0, 1)
    } else if i >= xs.len() {
        (xs.len() - 2, xs.len() - 1)
    } else {
        (i - 1, i)
    };
    let span = xs[hi] - xs[lo];
    if span == 0.0 {
        return ys[lo];
    }
    ys[lo] + ((at - xs[lo]) / span) * (ys[hi] - ys[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 101-trace radargram, 1 m between traces, running due east.
    fn geometry() -> RadargramGeometry {
        let n = 101;
        RadargramGeometry {
            radargram_id: "test-line".into(),
            revision_id: "revabc123".into(),
            distance: (0..n).map(|i| i as f64).collect(),
            twtt: (0..50).map(|i| i as f64 * 0.4).collect(),
            depth: (0..50).map(|i| i as f64 * 0.04).collect(),
            easting: (0..n).map(|i| 400_000.0 + i as f64).collect(),
            northing: (0..n).map(|_| 8_700_000.0).collect(),
            longitude: (0..n).map(|i| 15.0 + i as f64 * 1e-5).collect(),
            latitude: (0..n).map(|_| 78.0).collect(),
            crs: "EPSG:32633".into(),
        }
    }

    fn document(coords: &[[f64; 2]]) -> Document {
        document_multi(&[("bed", coords)])
    }

    fn document_multi(lines: &[(&str, &[[f64; 2]])]) -> Document {
        let features: Vec<serde_json::Value> = lines
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
        serde_json::from_value(serde_json::json!({
            "key": "test-line",
            "features": features,
        }))
        .unwrap()
    }

    #[test]
    fn arc_length_spacing_places_points_every_step_metres() {
        // A flat pick at sample 10, spanning traces 0..100, i.e. 100 m.
        let export = export(
            &document(&[[0.0, 10.0], [100.0, 10.0]]),
            &geometry(),
            Spacing::ArcLength(10.0),
            DEFAULT_USER,
        )
        .unwrap();

        assert_eq!(export.points.len(), 11);
        for (i, point) in export.points.iter().enumerate() {
            assert!((point.distance_m - i as f64 * 10.0).abs() < 1e-9);
            assert!((point.trace - i as f64 * 10.0).abs() < 1e-9);
            assert!((point.sample - 10.0).abs() < 1e-9);
        }
        assert_eq!(export.spacing_m, Some(10.0));
    }

    #[test]
    fn points_carry_depth_travel_time_and_both_coordinate_systems() {
        let export = export(
            &document(&[[0.0, 10.0], [100.0, 10.0]]),
            &geometry(),
            Spacing::ArcLength(50.0),
            DEFAULT_USER,
        )
        .unwrap();

        let point = &export.points[1];
        assert!((point.trace - 50.0).abs() < 1e-9);
        // twtt axis is 0.4 ns per sample, depth 0.04 m per sample.
        assert!((point.twtt_ns - 4.0).abs() < 1e-9);
        assert!((point.depth_m - 0.4).abs() < 1e-9);
        assert!((point.easting - 400_050.0).abs() < 1e-9);
        assert!((point.northing - 8_700_000.0).abs() < 1e-9);
        assert!((point.latitude - 78.0).abs() < 1e-9);
        assert_eq!(point.user, "default");
        assert_eq!(point.layer, "bed");
        assert_eq!(export.crs, "EPSG:32633");
        assert_eq!(export.radargram_id, "test-line");
        assert_eq!(export.revision_id, "revabc123");
    }

    #[test]
    fn a_sloping_pick_is_interpolated_between_its_vertices() {
        // Sample rises from 10 to 20 over traces 0..100.
        let export = export(
            &document(&[[0.0, 10.0], [100.0, 20.0]]),
            &geometry(),
            Spacing::ArcLength(25.0),
            DEFAULT_USER,
        )
        .unwrap();

        let samples: Vec<f64> = export.points.iter().map(|p| p.sample).collect();
        assert_eq!(samples.len(), 5);
        for (i, sample) in samples.iter().enumerate() {
            assert!((sample - (10.0 + 2.5 * i as f64)).abs() < 1e-9, "point {i}");
        }
    }

    #[test]
    fn spacing_follows_distance_not_trace_count_when_traces_are_uneven() {
        // The point of measuring in metres: this radargram covers 100 m in
        // its first 10 traces and 10 m over the remaining 90, so evenly
        // spaced traces and evenly spaced metres are wildly different.
        let mut geom = geometry();
        geom.distance = (0..101)
            .map(|i| {
                if i <= 10 {
                    i as f64 * 10.0
                } else {
                    100.0 + (i - 10) as f64 * (10.0 / 90.0)
                }
            })
            .collect();

        let export = export(
            &document(&[[0.0, 10.0], [100.0, 10.0]]),
            &geom,
            Spacing::ArcLength(10.0),
            DEFAULT_USER,
        )
        .unwrap();

        // 110 m of track at 10 m spacing.
        assert_eq!(export.points.len(), 12);
        for (i, point) in export.points.iter().enumerate() {
            assert!(
                (point.distance_m - i as f64 * 10.0).abs() < 1e-6,
                "point {i} sits at {} m",
                point.distance_m
            );
        }
        // The first 10 points fall in the fast stretch, i.e. one per trace,
        // while the last two span the remaining 90 traces.
        assert!((export.points[1].trace - 1.0).abs() < 1e-6);
        assert!(export.points[11].trace > 90.0);
    }

    #[test]
    fn per_trace_spacing_emits_one_point_per_native_trace() {
        let export = export(
            &document(&[[10.0, 10.0], [20.0, 10.0]]),
            &geometry(),
            Spacing::PerTrace,
            DEFAULT_USER,
        )
        .unwrap();

        assert_eq!(export.points.len(), 11);
        assert_eq!(export.spacing_m, None);
        let traces: Vec<f64> = export.points.iter().map(|p| p.trace).collect();
        assert_eq!(traces, (10..=20).map(|t| t as f64).collect::<Vec<f64>>());
    }

    #[test]
    fn separate_lines_of_one_layer_get_distinct_line_indices() {
        let doc = document_multi(&[
            ("bed", &[[0.0, 10.0], [40.0, 10.0]]),
            ("bed", &[[60.0, 20.0], [100.0, 20.0]]),
            ("internal", &[[0.0, 5.0], [40.0, 5.0]]),
        ]);
        let export = export(&doc, &geometry(), Spacing::ArcLength(20.0), DEFAULT_USER).unwrap();

        let bed: Vec<(usize, &str)> = export
            .points
            .iter()
            .filter(|p| p.layer == "bed")
            .map(|p| (p.line_index, p.feature_id.as_deref().unwrap()))
            .collect();
        assert!(bed.iter().any(|(i, id)| *i == 0 && *id == "f-0"));
        assert!(bed.iter().any(|(i, id)| *i == 1 && *id == "f-1"));

        // Line index restarts per layer, so 'internal' has its own line 0.
        assert!(export
            .points
            .iter()
            .any(|p| p.layer == "internal" && p.line_index == 0));
    }

    #[test]
    fn a_line_drawn_right_to_left_is_normalized_not_rejected() {
        // Leaflet.Draw records vertices in the order clicked, and drawing
        // leftwards is not a different interpretation.
        let export = export(
            &document(&[[100.0, 20.0], [0.0, 10.0]]),
            &geometry(),
            Spacing::ArcLength(50.0),
            DEFAULT_USER,
        )
        .unwrap();

        let traces: Vec<f64> = export.points.iter().map(|p| p.trace).collect();
        assert_eq!(traces.len(), 3);
        assert!(traces.windows(2).all(|w| w[1] > w[0]), "{traces:?}");
        assert!((export.points[0].sample - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_line_that_doubles_back_is_rejected_with_the_offending_vertex() {
        // Not a function of trace, so "the sample at 40 m along" has two
        // answers. Guessing one would silently corrupt the export.
        let error = export(
            &document(&[[0.0, 10.0], [50.0, 12.0], [30.0, 14.0]]),
            &geometry(),
            Spacing::ArcLength(10.0),
            DEFAULT_USER,
        )
        .unwrap_err();

        assert_eq!(
            error,
            Level2Error::NonMonotoneLine {
                layer: "bed".into(),
                line_index: 0,
                at_vertex: 2,
            }
        );
    }

    #[test]
    fn non_line_geometries_are_reported_as_unsupported_not_skipped() {
        let doc: Document = serde_json::from_value(serde_json::json!({
            "key": "test-line",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10.0, 10.0]},
                "properties": {"label": "poi"}
            }]
        }))
        .unwrap();

        let error = export(&doc, &geometry(), Spacing::Auto, DEFAULT_USER).unwrap_err();
        assert_eq!(
            error,
            Level2Error::UnsupportedGeometry {
                layer: "poi".into(),
                type_name: "Point".into(),
            }
        );
    }

    #[test]
    fn auto_step_snaps_the_median_trace_spacing_to_a_tidy_value() {
        // The real Drønbreen figure: ~0.248 m median spacing.
        let distance: Vec<f64> = (0..1000).map(|i| i as f64 * 0.248).collect();
        assert_eq!(auto_step(&distance), 0.25);

        let sparse: Vec<f64> = (0..1000).map(|i| i as f64 * 3.0).collect();
        assert_eq!(auto_step(&sparse), 5.0);
    }

    #[test]
    fn auto_step_ignores_standstills() {
        // 500 stacked traces then 500 at 2 m. A mean would report ~1 m; the
        // median of the *moving* steps is the honest answer.
        let mut distance = vec![0.0; 500];
        for i in 0..500 {
            distance.push(i as f64 * 2.0);
        }
        assert_eq!(auto_step(&distance), 2.0);
    }

    #[test]
    fn a_standstill_line_still_produces_one_point() {
        let mut geom = geometry();
        geom.distance = vec![0.0; 101];
        let export = export(
            &document(&[[0.0, 10.0], [100.0, 10.0]]),
            &geom,
            Spacing::ArcLength(5.0),
            DEFAULT_USER,
        )
        .unwrap();
        assert_eq!(export.points.len(), 1);
    }

    #[test]
    fn mismatched_axis_lengths_are_rejected_before_any_export() {
        let mut geom = geometry();
        geom.easting.pop();
        let error = export(
            &document(&[[0.0, 10.0], [100.0, 10.0]]),
            &geom,
            Spacing::Auto,
            DEFAULT_USER,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Level2Error::AxisLengthMismatch {
                axis: "easting",
                ..
            }
        ));
    }
}
