//! Track extraction and trace-indexed simplification for the web viewer.
//!
//! PFA_website's track simplification samples vertices evenly by *distance*
//! (`format_radargrams.py`), but its viewer looks a cursor position up by
//! *trace fraction* (`digitize.js`). Those only agree when traces are
//! evenly spaced; any standstill desynchronizes them, and the error
//! accumulates along the profile -- measured up to 140 m on a real
//! Drønbreen line with an 18-trace standstill.
//!
//! This module keeps the correspondence exact by storing each retained
//! vertex's source trace index directly. Client-side (and
//! [`Track::locate_trace`], its Rust-side twin used for testing) then
//! binary-searches trace indices and interpolates, which is exact at every
//! retained vertex and monotone in trace index regardless of how unevenly
//! the vertices are spaced in distance.

use crate::coords;
use crate::gpr::GPRLocation;

/// Initial Douglas-Peucker tolerance, in the track's native projected CRS
/// units (metres for UTM). Doubled iteratively if a segment still exceeds
/// [`MAX_VERTICES_PER_SEGMENT`].
const DEFAULT_TOLERANCE_M: f64 = 1.0;
const MAX_VERTICES_PER_SEGMENT: usize = 2000;
/// Safety cap so the tolerance-doubling loop always terminates, even for a
/// pathological segment that can never be simplified under the vertex cap.
const MAX_TOLERANCE_M: f64 = 1000.0;

/// A break is inserted where consecutive traces jump more than this many
/// metres, matching PFA_website's heuristic.
const BREAK_DISTANCE_JUMP_M: f64 = 100.0;
/// ...or where the time gap exceeds this multiple of the segment's median
/// per-trace time gap.
const BREAK_TIME_GAP_MULTIPLE: f64 = 50.0;

/// One retained track vertex: a source trace index and its WGS84 position.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackVertex {
    pub trace_index: u32,
    pub lon: f64,
    pub lat: f64,
}

/// A contiguous run of traces with no large distance/time break inside it,
/// simplified to a small vertex set. Every trace in `[trace_start,
/// trace_end]` belongs to exactly one segment -- unlike PFA_website, no
/// segment is dropped for being short, so every trace has a defined
/// position.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackSegment {
    pub segment_index: usize,
    pub trace_start: u32,
    pub trace_end: u32,
    pub n_traces: u32,
    pub length_m: f64,
    pub vertices: Vec<TrackVertex>,
}

/// A full radargram track: an ordered set of segments covering every trace.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Track {
    pub segments: Vec<TrackSegment>,
}

impl Track {
    /// Build a simplified, trace-indexed track from `location`.
    ///
    /// Douglas-Peucker simplification runs in the native projected CRS
    /// (native easting/northing, already present on every [`CorPoint`] --
    /// no reprojection needed to simplify), and only the retained vertices
    /// are reprojected to WGS84 for display.
    ///
    /// [`CorPoint`]: crate::gpr::CorPoint
    pub fn from_location(location: &GPRLocation) -> Result<Track, String> {
        let n = location.cor_points.len();
        if n == 0 {
            return Ok(Track::default());
        }
        if n == 1 {
            let crs = coords::Crs::from_user_input(&location.crs)?;
            let p = &location.cor_points[0];
            let wgs84 = coords::to_wgs84(
                &[coords::Coord {
                    x: p.easting,
                    y: p.northing,
                }],
                &crs,
            )?;
            return Ok(Track {
                segments: vec![TrackSegment {
                    segment_index: 0,
                    trace_start: 0,
                    trace_end: 0,
                    n_traces: 1,
                    length_m: 0.0,
                    vertices: vec![TrackVertex {
                        trace_index: 0,
                        lon: wgs84[0].x,
                        lat: wgs84[0].y,
                    }],
                }],
            });
        }

        let distances = location.distances();
        let native: Vec<(f64, f64)> = location
            .cor_points
            .iter()
            .map(|p| (p.easting, p.northing))
            .collect();

        let breaks = find_segment_breaks(location, &distances);
        let crs = coords::Crs::from_user_input(&location.crs)?;

        let mut segments = Vec::with_capacity(breaks.len().saturating_sub(1));
        for (seg_i, w) in breaks.windows(2).enumerate() {
            let (start, end) = (w[0], w[1]); // half-open [start, end)
            let seg_points = &native[start..end];

            // Simplify in (trace_index * scale, easting, northing), not
            // plain (easting, northing). Pure 2D Douglas-Peucker is blind to
            // velocity: a standstill on an otherwise straight line adds no
            // geometric deviation, so 2D simplification collapses it away
            // -- silently reintroducing the exact class of bug this module
            // exists to fix, since trace-index interpolation between the
            // two straight-line endpoints would then imply constant speed
            // straight through the standstill. Scaling trace index by the
            // segment's average speed turns a non-uniform-speed stretch
            // into a genuine 3D deviation from the chord, so it gets a
            // retained vertex just like a directional corner would.
            let seg_len = seg_points.len();
            let seg_length_m = distances[end - 1] - distances[start];
            let speed_scale = if seg_len > 1 {
                seg_length_m / (seg_len - 1) as f64
            } else {
                1.0
            };
            let seg_points_3d: Vec<[f64; 3]> = seg_points
                .iter()
                .enumerate()
                .map(|(i, &(e, n))| [i as f64 * speed_scale, e, n])
                .collect();
            let kept_local = simplify_with_indices(
                &seg_points_3d,
                DEFAULT_TOLERANCE_M,
                MAX_VERTICES_PER_SEGMENT,
            );

            let kept_native: Vec<coords::Coord> = kept_local
                .iter()
                .map(|&i| coords::Coord {
                    x: seg_points[i].0,
                    y: seg_points[i].1,
                })
                .collect();
            let kept_wgs84 = coords::to_wgs84(&kept_native, &crs)?;

            let vertices = kept_local
                .iter()
                .zip(kept_wgs84.iter())
                .map(|(&i, c)| TrackVertex {
                    trace_index: (start + i) as u32,
                    lon: c.x,
                    lat: c.y,
                })
                .collect();

            segments.push(TrackSegment {
                segment_index: seg_i,
                trace_start: start as u32,
                trace_end: (end - 1) as u32,
                n_traces: (end - start) as u32,
                length_m: distances[end - 1] - distances[start],
                vertices,
            });
        }

        Ok(Track { segments })
    }

    /// Interpolate the WGS84 position of a (possibly fractional)
    /// `trace_index`, by linear interpolation between the nearest
    /// bracketing retained vertices in the owning segment.
    ///
    /// Returns `None` only if `trace_index` falls outside every segment's
    /// `[trace_start, trace_end]` range -- which cannot happen for a track
    /// built by [`Track::from_location`], since segments cover every trace,
    /// but is possible for a caller-constructed `Track` or an
    /// out-of-range query.
    ///
    /// Not called by production code: cursor sync happens client-side in
    /// JS against the same JSON the `/track` route serves, to avoid an
    /// HTTP round trip per `mousemove`. Kept here (and exercised directly
    /// by this module's own tests, including the real-asset regression)
    /// as the reference implementation the client-side lookup mirrors.
    #[allow(dead_code)]
    pub fn locate_trace(&self, trace_index: f64) -> Option<(f64, f64)> {
        let segment = self.segments.iter().find(|s| {
            trace_index >= s.trace_start as f64 - f64::EPSILON
                && trace_index <= s.trace_end as f64 + f64::EPSILON
        })?;
        locate_in_vertices(&segment.vertices, trace_index)
    }
}

/// Read `easting`/`northing`/`time`/`crs` back from a processed Ridal
/// NetCDF file and build a [`Track`] from them.
///
/// Reconstructs a temporary [`GPRLocation`] rather than duplicating
/// `Track::from_location`'s logic, so track-reading for the web API goes
/// through the exact same tested path as track extraction from a
/// just-processed in-memory `GPR` does.
pub fn read_track_from_netcdf(path: &std::path::Path) -> Result<Track, String> {
    let file = netcdf::open(path).map_err(|e| format!("Failed to open {path:?} as NetCDF: {e}"))?;

    let crs = read_global_str(&file, "crs")?;
    let easting = read_f64_variable(&file, "easting")?;
    let northing = read_f64_variable(&file, "northing")?;
    let time = read_f64_variable(&file, "time")?;
    if easting.len() != northing.len() || easting.len() != time.len() {
        return Err(format!(
            "Inconsistent track variable lengths in {path:?}: easting={}, northing={}, time={}",
            easting.len(),
            northing.len(),
            time.len()
        ));
    }

    let cor_points = easting
        .into_iter()
        .zip(northing)
        .zip(time)
        .enumerate()
        .map(
            |(i, ((easting, northing), time_seconds))| crate::gpr::CorPoint {
                trace_n: i as u32,
                time_seconds,
                easting,
                northing,
                altitude: 0.0,
            },
        )
        .collect();

    let location = GPRLocation {
        cor_points,
        correction: crate::gpr::LocationCorrection::None,
        crs,
    };
    Track::from_location(&location)
}

fn read_global_str(file: &netcdf::File, name: &str) -> Result<String, String> {
    let attr = file
        .attribute(name)
        .ok_or_else(|| format!("Missing global attribute '{name}'"))?;
    match attr.value() {
        Ok(netcdf::AttributeValue::Str(s)) => Ok(s),
        other => Err(format!("Attribute '{name}' is not a string: {other:?}")),
    }
}

pub(super) fn read_f64_variable(file: &netcdf::File, name: &str) -> Result<Vec<f64>, String> {
    let var = file
        .variable(name)
        .ok_or_else(|| format!("Missing variable '{name}'"))?;
    var.get_values::<f64, _>(..)
        .map_err(|e| format!("Failed to read variable '{name}': {e}"))
}

#[allow(dead_code)]
fn locate_in_vertices(vertices: &[TrackVertex], trace_index: f64) -> Option<(f64, f64)> {
    match vertices.len() {
        0 => None,
        1 => Some((vertices[0].lon, vertices[0].lat)),
        _ => {
            // First vertex index with trace_index >= the query.
            let pos = vertices.partition_point(|v| (v.trace_index as f64) < trace_index);
            let (a, b) = if pos == 0 {
                (0, 1)
            } else if pos >= vertices.len() {
                (vertices.len() - 2, vertices.len() - 1)
            } else {
                (pos - 1, pos)
            };
            let (va, vb) = (&vertices[a], &vertices[b]);
            let span = vb.trace_index as f64 - va.trace_index as f64;
            let t = if span > 0.0 {
                ((trace_index - va.trace_index as f64) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some((
                va.lon + t * (vb.lon - va.lon),
                va.lat + t * (vb.lat - va.lat),
            ))
        }
    }
}

/// Half-open segment boundaries `[b0, b1, b2, ..., n]`, so segment `i`
/// covers traces `[breaks[i], breaks[i+1])`. Always starts at 0 and ends at
/// `distances.len()`.
fn find_segment_breaks(location: &GPRLocation, distances: &ndarray::Array1<f64>) -> Vec<usize> {
    let n = location.cor_points.len();

    let mut dt = vec![0.0_f64; n];
    for (i, slot) in dt.iter_mut().enumerate().skip(1) {
        *slot = location.cor_points[i].time_seconds - location.cor_points[i - 1].time_seconds;
    }
    let mut sorted_dt: Vec<f64> = dt[1..].to_vec();
    sorted_dt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dt = sorted_dt.get(sorted_dt.len() / 2).copied().unwrap_or(0.0);

    let mut breaks = vec![0usize];
    for i in 1..n {
        let dist_jump = distances[i] - distances[i - 1];
        let is_dist_break = dist_jump > BREAK_DISTANCE_JUMP_M;
        let is_time_break = median_dt > 0.0 && dt[i] > median_dt * BREAK_TIME_GAP_MULTIPLE;
        if is_dist_break || is_time_break {
            breaks.push(i);
        }
    }
    breaks.push(n);
    breaks.dedup();
    breaks
}

/// Douglas-Peucker simplification with iterative tolerance doubling to
/// respect `max_vertices`. Returns indices into `points` of the retained
/// vertices, always including the first and last.
///
/// Points are 3D: callers pass `[trace_index * speed_scale, easting,
/// northing]` so that a non-uniform-speed stretch (a standstill, most
/// notably) shows up as a genuine geometric deviation from the chord, not
/// just a change of direction. See the comment at the call site in
/// [`Track::from_location`] for why 2D shape-only simplification is not
/// sufficient here.
fn simplify_with_indices(
    points: &[[f64; 3]],
    initial_tolerance: f64,
    max_vertices: usize,
) -> Vec<usize> {
    if points.len() <= 2 {
        return (0..points.len()).collect();
    }
    let mut tolerance = initial_tolerance;
    loop {
        let kept = douglas_peucker(points, tolerance);
        if kept.len() <= max_vertices || tolerance >= MAX_TOLERANCE_M {
            return kept;
        }
        tolerance *= 2.0;
    }
}

fn douglas_peucker(points: &[[f64; 3]], tolerance: f64) -> Vec<usize> {
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    simplify_range(points, 0, points.len() - 1, tolerance, &mut keep);
    (0..points.len()).filter(|&i| keep[i]).collect()
}

fn simplify_range(
    points: &[[f64; 3]],
    start: usize,
    end: usize,
    tolerance: f64,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }
    let (mut max_dist, mut max_idx) = (0.0_f64, start);
    for (i, &p) in points.iter().enumerate().take(end).skip(start + 1) {
        let d = perpendicular_distance(p, points[start], points[end]);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }
    if max_dist > tolerance {
        keep[max_idx] = true;
        simplify_range(points, start, max_idx, tolerance, keep);
        simplify_range(points, max_idx, end, tolerance, keep);
    }
}

/// Distance from point `p` to the line through `a` and `b`, in 3D. Reduces
/// exactly to the familiar 2D point-line distance when all inputs share a
/// constant third coordinate (used directly by this module's own tests).
fn perpendicular_distance(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if len2 == 0.0 {
        return (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt();
    }
    let cross = [
        ap[1] * ab[2] - ap[2] * ab[1],
        ap[2] * ab[0] - ap[0] * ab[2],
        ap[0] * ab[1] - ap[1] * ab[0],
    ];
    let cross_len2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    (cross_len2 / len2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpr::{CorPoint, GPRLocation, LocationCorrection};

    /// Approximate metre distance between two nearby WGS84 points, using a
    /// flat-earth approximation. Adequate for test tolerance checks over
    /// the sub-kilometre spans used here; not intended for production use.
    fn approx_lonlat_distance_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
        let mean_lat_rad = ((lat1 + lat2) / 2.0).to_radians();
        let dx = (lon2 - lon1) * mean_lat_rad.cos() * 111_320.0;
        let dy = (lat2 - lat1) * 110_540.0;
        (dx * dx + dy * dy).sqrt()
    }

    fn location_from_traces(points: Vec<(f64, f64, f64)>) -> GPRLocation {
        // (time_seconds, easting, northing)
        GPRLocation {
            cor_points: points
                .into_iter()
                .enumerate()
                .map(|(i, (t, e, n))| CorPoint {
                    trace_n: i as u32,
                    time_seconds: t,
                    easting: e,
                    northing: n,
                    altitude: 0.0,
                })
                .collect(),
            correction: LocationCorrection::None,
            crs: "EPSG:32633".to_string(),
        }
    }

    #[test]
    fn straight_line_simplifies_to_endpoints() {
        // A perfectly straight line needs only its two endpoints regardless
        // of how many traces it has.
        let points: Vec<(f64, f64, f64)> = (0..500)
            .map(|i| (i as f64 * 0.1, 500000.0 + i as f64, 8_000_000.0))
            .collect();
        let location = location_from_traces(points);

        let track = Track::from_location(&location).unwrap();
        assert_eq!(track.segments.len(), 1);
        assert_eq!(track.segments[0].vertices.len(), 2);
        assert_eq!(track.segments[0].trace_start, 0);
        assert_eq!(track.segments[0].trace_end, 499);
    }

    #[test]
    fn every_trace_has_a_defined_position() {
        let points: Vec<(f64, f64, f64)> = (0..300)
            .map(|i| {
                let t = i as f64 * 0.1;
                // A corner partway through, to force at least one interior
                // retained vertex.
                let (e, n) = if i < 150 {
                    (500000.0 + i as f64, 8_000_000.0)
                } else {
                    (500150.0, 8_000_000.0 + (i - 150) as f64)
                };
                (t, e, n)
            })
            .collect();
        let location = location_from_traces(points);
        let track = Track::from_location(&location).unwrap();

        for trace in 0..300 {
            assert!(
                track.locate_trace(trace as f64).is_some(),
                "trace {trace} has no defined position"
            );
        }
    }

    #[test]
    fn standstill_does_not_desynchronize_trace_lookup() {
        // 20 traces of travel, then a 30-trace standstill, then 20 more
        // traces of travel. The old PFA approach (sample every 5 distance
        // units, index by trace fraction) would desync badly here because
        // the standstill contributes zero distance but nonzero trace count.
        let mut points = Vec::new();
        let mut t = 0.0;
        for i in 0..20 {
            points.push((t, 500000.0 + i as f64, 8_000_000.0));
            t += 0.1;
        }
        let (stand_e, stand_n) = (500019.0, 8_000_000.0);
        for _ in 0..30 {
            points.push((t, stand_e, stand_n));
            t += 0.1;
        }
        for i in 0..20 {
            points.push((t, stand_e + i as f64, 8_000_000.0));
            t += 0.1;
        }
        let location = location_from_traces(points.clone());
        let track = Track::from_location(&location).unwrap();

        // Ground truth: reproject every native point directly and compare.
        let crs = coords::Crs::from_user_input(&location.crs).unwrap();
        let native: Vec<coords::Coord> = points
            .iter()
            .map(|&(_, e, n)| coords::Coord { x: e, y: n })
            .collect();
        let truth = coords::to_wgs84(&native, &crs).unwrap();

        // The plan's own regression bound: locate_trace() must land within
        // 2x the simplification tolerance (2 m) of the true position, for
        // every trace, including inside the standstill. This is a physical
        // distance bound, not a degrees bound -- linear interpolation
        // between two widely-spaced kept vertices happens in WGS84 lon/lat,
        // and UTM->WGS84 is nonlinear, so even an exactly straight native
        // segment picks up a small curvature error in degree terms that a
        // flat degrees-diff assertion would over- or under-state depending
        // on latitude.
        let max_allowed_error_m = 2.0 * DEFAULT_TOLERANCE_M;
        for (i, expected) in truth.iter().enumerate() {
            let (lon, lat) = track.locate_trace(i as f64).unwrap();
            let error_m = approx_lonlat_distance_m(lon, lat, expected.x, expected.y);
            assert!(
                error_m < max_allowed_error_m,
                "trace {i}: got ({lon}, {lat}), expected ({}, {}), error {error_m:.3} m",
                expected.x,
                expected.y
            );
        }
    }

    #[test]
    fn large_distance_jump_creates_a_new_segment() {
        let mut points: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, 500000.0 + i as f64, 8_000_000.0))
            .collect();
        // A 500 m jump, far beyond the 100 m break threshold.
        points.push((1.0, 500500.0, 8_000_000.0));
        points.extend((0..10).map(|i| (1.1 + i as f64 * 0.1, 500500.0 + i as f64, 8_000_000.0)));

        let location = location_from_traces(points);
        let track = Track::from_location(&location).unwrap();
        assert_eq!(track.segments.len(), 2);
        assert_eq!(track.segments[0].trace_end, 9);
        assert_eq!(track.segments[1].trace_start, 10);
    }

    #[test]
    fn short_segments_are_retained_not_dropped() {
        // Unlike PFA_website (which drops segments under 10 traces), a
        // 3-trace segment isolated by jumps on both sides must survive.
        let mut points: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1, 500000.0 + i as f64, 8_000_000.0))
            .collect();
        points.push((1.0, 501000.0, 8_000_000.0));
        points.push((1.1, 501001.0, 8_000_000.0));
        points.push((1.2, 501002.0, 8_000_000.0));
        points.push((1.3, 502000.0, 8_000_000.0));
        points.extend((0..10).map(|i| (1.4 + i as f64 * 0.1, 502000.0 + i as f64, 8_000_000.0)));

        let location = location_from_traces(points);
        let track = Track::from_location(&location).unwrap();
        assert_eq!(track.segments.len(), 3);
        assert_eq!(track.segments[1].n_traces, 3);
        assert!(track.locate_trace(10.5).is_some());
        assert!(track.locate_trace(11.0).is_some());
    }

    #[test]
    fn douglas_peucker_keeps_corner_within_tolerance() {
        // L-shaped path: straight to (10,0), straight up to (10,10). Third
        // coordinate held at 0 throughout, which exercises the pure 2D
        // shape-only behavior as a special case of the general 3D distance.
        let points: Vec<[f64; 3]> = (0..=10)
            .map(|i| [i as f64, 0.0, 0.0])
            .chain((1..=10).map(|i| [10.0, i as f64, 0.0]))
            .collect();
        let kept = super::douglas_peucker(&points, 0.5);
        assert!(kept.contains(&0));
        assert!(kept.contains(&10)); // the corner
        assert!(kept.contains(&(points.len() - 1)));
        assert!(kept.len() < points.len());
    }

    #[test]
    fn simplify_respects_max_vertices_via_tolerance_doubling() {
        // A zigzag that Douglas-Peucker at the default tolerance would keep
        // almost entirely; the vertex cap must force a coarser tolerance.
        let points: Vec<[f64; 3]> = (0..5000)
            .map(|i| [i as f64, if i % 2 == 0 { 0.0 } else { 0.5 }, 0.0])
            .collect();
        let kept = super::simplify_with_indices(&points, 1.0, 200);
        assert!(kept.len() <= 200, "kept {} vertices", kept.len());
    }

    #[test]
    fn standstill_on_a_straight_line_still_gets_a_vertex() {
        // A standstill that happens to sit on an otherwise perfectly
        // straight line is invisible to plain 2D Douglas-Peucker (zero
        // perpendicular deviation), which would silently reintroduce the
        // trace-index-vs-position bug this module exists to fix: linear
        // interpolation between the two straight-line endpoints would then
        // imply constant speed straight through the standstill. Simplifying
        // in (trace_index * speed_scale, easting, northing) instead must
        // still retain a vertex bracketing the standstill.
        // 20 traces moving at 1 unit/trace, then 30 stationary traces (all
        // at position 19.0) -- perfectly collinear in (easting), since
        // northing is constant throughout.
        let speed_scale = 19.0 / 49.0; // average speed over the whole 50-trace run
        let points: Vec<[f64; 3]> = (0..20)
            .map(|i| [i as f64 * speed_scale, i as f64, 0.0])
            .chain((20..50).map(|i| [i as f64 * speed_scale, 19.0, 0.0]))
            .collect();
        assert_eq!(points.len(), 50);

        let kept = super::douglas_peucker(&points, DEFAULT_TOLERANCE_M);
        // Some interior vertex within the standstill run [20, 49] must be
        // retained -- otherwise interpolation between trace 19 and trace 49
        // would imply the standstill traces kept moving.
        assert!(
            kept.iter().any(|&i| (20..50).contains(&i)),
            "expected a retained vertex inside the standstill, got {kept:?}"
        );
    }

    #[test]
    fn single_trace_track_has_one_vertex() {
        let location = location_from_traces(vec![(0.0, 500000.0, 8_000_000.0)]);
        let track = Track::from_location(&location).unwrap();
        assert_eq!(track.segments.len(), 1);
        assert_eq!(track.segments[0].vertices.len(), 1);
        assert!(track.locate_trace(0.0).is_some());
    }

    #[test]
    fn empty_track_has_no_segments() {
        let location = location_from_traces(vec![]);
        let track = Track::from_location(&location).unwrap();
        assert!(track.segments.is_empty());
    }

    /// Simplify `points` by picking every `stride`-th distance value,
    /// exactly reproducing PFA_website's actual bug: vertices sampled
    /// evenly in *distance*, then looked up by *trace fraction* -- i.e. as
    /// if `n_kept` vertices were evenly spaced across `n_traces` traces,
    /// rather than carrying their real trace indices as this module's
    /// [`Track`] does. See `format_radargrams.py:182` / `digitize.js:980`.
    fn pfa_style_locate(
        distances: &[f64],
        native: &[coords::Coord],
        crs: &coords::Crs,
        query_trace: usize,
    ) -> (f64, f64) {
        let n = native.len();
        let step_m = 5.0;
        let (dmin, dmax) = (distances[0], distances[n - 1]);
        let n_kept = (((dmax - dmin) / step_m).floor() as usize + 2).max(2);

        // For each of n_kept evenly-distance-spaced targets, find the
        // nearest native vertex (mirrors interp1d(distance -> trace_index)
        // in format_radargrams.py).
        let mut kept_coords: Vec<coords::Coord> = Vec::with_capacity(n_kept);
        for k in 0..n_kept {
            let target_d = dmin + (dmax - dmin) * (k as f64) / ((n_kept - 1) as f64);
            let nearest = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (**a - target_d)
                        .abs()
                        .partial_cmp(&(**b - target_d).abs())
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap();
            kept_coords.push(native[nearest]);
        }
        let kept_wgs84 = coords::to_wgs84(&kept_coords, crs).unwrap();

        // digitize.js:980 -- index by TRACE FRACTION, not trace index.
        let frac = query_trace as f64 * (n_kept as f64) / (n as f64);
        let idx = (frac.floor() as usize).min(n_kept - 1);
        (kept_wgs84[idx].x, kept_wgs84[idx].y)
    }

    /// Regression test against real acquisition data, using the exact
    /// standstill windows measured in the planning phase (see the plan
    /// artifact, section on fixtures): both subsets contain a real
    /// standstill or sustained speed variation, not a synthetic one.
    ///
    /// Confirms both halves of the fix in one place: the new trace-indexed
    /// method stays within the plan's 2m bound on real data, and the old
    /// distance-indexed (PFA-style) method demonstrably does not -- so the
    /// bug cannot silently return.
    #[test]
    #[serial_test::serial(netcdf)]
    fn real_asset_regression_new_method_accurate_old_method_fails() {
        let cases = [
            (
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/mala/dronbreen-20220329-DAT_0237_A1.rad"
                ),
                "subset(1200 1700 0 400)",
            ),
            (
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/assets/mala/dronbreen-20250327-DAT_0066_A1.rad"
                ),
                "subset(2400 2800 0 400)",
            ),
        ];

        let max_allowed_error_m = 2.0 * DEFAULT_TOLERANCE_M;
        let mut any_old_method_failure = false;

        for (input_path, subset_step) in cases {
            let params = crate::gpr::RunParams {
                filepaths: vec![std::path::PathBuf::from(input_path)],
                output_path: None,
                dem_path: None,
                cor_path: None,
                medium_velocity: 0.168,
                crs: None,
                quiet: true,
                track_path: None,
                steps: vec![subset_step.to_string()],
                no_export: true,
                render_path: None,
                render_profile: None,
                render_width: None,
                override_antenna_mhz: None,
                override_antenna_separation: None,
                user_metadata: Default::default(),
                radargram_id: None,
                display_name: None,
                group: None,
                group_id: None,
            };
            let (gpr, _) = crate::gpr::build_processed_gpr(params).unwrap();
            let location = &gpr.location;
            let n = location.cor_points.len();
            assert!(n > 100, "expected a substantial subset, got {n} traces");

            let track = Track::from_location(location).unwrap();

            let crs = coords::Crs::from_user_input(&location.crs).unwrap();
            let native: Vec<coords::Coord> = location
                .cor_points
                .iter()
                .map(|p| coords::Coord {
                    x: p.easting,
                    y: p.northing,
                })
                .collect();
            let truth = coords::to_wgs84(&native, &crs).unwrap();
            let distances = location.distances();
            let distances_vec: Vec<f64> = distances.to_vec();

            let mut new_method_max_error = 0.0_f64;
            let mut old_method_max_error = 0.0_f64;
            for i in 0..n {
                let expected = &truth[i];

                let (lon, lat) = track.locate_trace(i as f64).unwrap();
                new_method_max_error = new_method_max_error
                    .max(approx_lonlat_distance_m(lon, lat, expected.x, expected.y));

                let (plon, plat) = pfa_style_locate(&distances_vec, &native, &crs, i);
                old_method_max_error = old_method_max_error
                    .max(approx_lonlat_distance_m(plon, plat, expected.x, expected.y));
            }

            assert!(
                new_method_max_error < max_allowed_error_m,
                "{input_path}: new method max error {new_method_max_error:.2} m exceeds {max_allowed_error_m} m bound"
            );
            if old_method_max_error >= max_allowed_error_m {
                any_old_method_failure = true;
            }
        }

        assert!(
            any_old_method_failure,
            "expected the PFA-style distance-indexed method to exceed the {max_allowed_error_m} m \
             bound on at least one real fixture -- if it no longer does, the standstill windows may \
             have gone stale and should be re-measured"
        );
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn read_track_from_netcdf_matches_in_memory_extraction() {
        let dir = tempfile::tempdir().unwrap();
        let nc_path = dir.path().join("t.nc");
        let params = crate::gpr::RunParams {
            filepaths: vec![std::path::PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/mala/dronbreen-20220329-DAT_0237_A1.rad"
            ))],
            output_path: Some(nc_path.clone()),
            dem_path: None,
            cor_path: None,
            medium_velocity: 0.168,
            crs: None,
            quiet: true,
            track_path: None,
            steps: vec!["subset(0 300 0 50)".to_string()],
            no_export: false,
            render_path: None,
            render_profile: None,
            render_width: None,
            override_antenna_mhz: None,
            override_antenna_separation: None,
            user_metadata: Default::default(),
            radargram_id: Some("track-read-test".to_string()),
            display_name: None,
            group: None,
            group_id: None,
        };
        let (gpr, _) = crate::gpr::build_processed_gpr(params.clone()).unwrap();
        crate::gpr::run(params).unwrap();

        let from_memory = Track::from_location(&gpr.location).unwrap();
        let from_file = read_track_from_netcdf(&nc_path).unwrap();

        assert_eq!(from_memory.segments.len(), from_file.segments.len());
        for (a, b) in from_memory.segments.iter().zip(from_file.segments.iter()) {
            assert_eq!(a.trace_start, b.trace_start);
            assert_eq!(a.trace_end, b.trace_end);
            assert_eq!(a.vertices.len(), b.vertices.len());
            for (va, vb) in a.vertices.iter().zip(b.vertices.iter()) {
                assert_eq!(va.trace_index, vb.trace_index);
                assert!((va.lon - vb.lon).abs() < 1e-9);
                assert!((va.lat - vb.lat).abs() < 1e-9);
            }
        }
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn read_track_from_netcdf_reports_missing_variable_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_track.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            file.add_dimension("y", 2).unwrap();
            file.add_dimension("x", 2).unwrap();
            let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
            var.put_values(&[0.0f32, 0., 0., 0.], ..).unwrap();
            file.add_attribute("crs", "EPSG:32633").unwrap();
        }
        let err = read_track_from_netcdf(&path).unwrap_err();
        assert!(err.contains("easting"), "{err}");
    }
}
