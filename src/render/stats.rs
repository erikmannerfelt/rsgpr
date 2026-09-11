//! Fixed-seed sampled amplitude limits (#119).
//!
//! Sampling whole traces (not scattered pixels, not a row subset) is what
//! makes the source wavelet -- a narrow band of very high amplitude near
//! the top of the radargram, this data's main heteroscedasticity -- show up
//! in the sample at its true share of the data: every trace carries the
//! full vertical structure, so any set of complete traces reproduces the
//! row-wise mixture in correct proportion regardless of which traces are
//! drawn.

use super::colormap::to_stats_domain;
use super::profile::AmplitudeTransform;
use crate::source::AmplitudeSource;

/// Spread across the profile. 128 well-separated locations is ample for a
/// percentile dominated by vertical (not horizontal) structure; the cost
/// of sampling more is small (~0.2s per radargram, once, cached) if a
/// wider net is ever wanted.
const N_RUNS: usize = 128;
/// Contiguous traces per run -- short enough that a run stays inside a
/// handful of storage chunks (`SourceReader::sample_trace_runs`).
const TRACES_PER_RUN: usize = 16;

/// Estimate `(low, high)` amplitude limits in the *display domain* (i.e.
/// after the same `abslog` transform the colormap applies), via
/// fixed-seed sampled percentiles.
///
/// `seed` should be derived from the revision ID, not the clock, so limits
/// are reproducible across restarts and identical between the CLI and the
/// server for the same processed file.
///
/// `skip_first_samples` drops that many sample rows from the top of every
/// sampled trace before estimating percentiles -- the `positive` profile's
/// way of excluding the direct-wave band (see
/// `RenderProfile::stats_skip_first_samples`). `0` reproduces the original
/// whole-trace behavior.
pub fn sampled_amplitude_limits(
    reader: &impl AmplitudeSource,
    transform: AmplitudeTransform,
    seed: u64,
    low_pct: f32,
    high_pct: f32,
    skip_first_samples: usize,
) -> Result<(f32, f32), String> {
    let n_traces = reader.shape().1;
    if n_traces == 0 {
        return Err("cannot estimate amplitude limits: radargram has zero traces".to_string());
    }
    let step = (n_traces / N_RUNS).max(1);
    let offset = (seed as usize) % step;

    let samples = reader.sample_trace_runs(N_RUNS, TRACES_PER_RUN, offset, skip_first_samples)?;
    let mut transformed: Vec<f32> = samples
        .into_iter()
        .map(|v| to_stats_domain(v, transform))
        .filter(|v| v.is_finite())
        .collect();
    if transformed.is_empty() {
        return Err("no finite amplitude samples available for limit estimation".to_string());
    }
    transformed.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let low = percentile(&transformed, low_pct);
    let high = percentile(&transformed, high_pct);
    Ok((low, high))
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    let p = p.clamp(0.0, 1.0);
    let idx = (((sorted.len() - 1) as f32) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceReader;

    fn write_test_nc_with(path: &std::path::Path, height: usize, width: usize, values: &[f32]) {
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("y", height).unwrap();
        file.add_dimension("x", width).unwrap();
        let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
        var.put_values(values, ..).unwrap();
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn percentile_limits_bracket_uniform_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        // Every trace is the ramp 0..20 vertically, so any sample of
        // complete traces reproduces the exact same value distribution.
        let height = 20;
        let width = 200;
        let mut values = Vec::with_capacity(height * width);
        for row in 0..height {
            for _ in 0..width {
                values.push(row as f32);
            }
        }
        write_test_nc_with(&path, height, width, &values);
        let reader = SourceReader::open(&path).unwrap();

        let (low, high) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 0, 0.01, 0.99, 0)
                .unwrap();
        assert!(low >= 0.0 && low < 5.0, "low={low}");
        assert!(high > 15.0 && high <= 19.0, "high={high}");
        assert!(low < high);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn different_seeds_still_agree_closely_on_uniform_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        let height = 10;
        let width = 500;
        let mut values = Vec::with_capacity(height * width);
        for row in 0..height {
            for _ in 0..width {
                values.push(row as f32);
            }
        }
        write_test_nc_with(&path, height, width, &values);
        let reader = SourceReader::open(&path).unwrap();

        let (low_a, high_a) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 1, 0.01, 0.99, 0)
                .unwrap();
        let (low_b, high_b) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 999, 0.01, 0.99, 0)
                .unwrap();
        assert!((low_a - low_b).abs() < 1.0);
        assert!((high_a - high_b).abs() < 1.0);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn same_seed_is_fully_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc_with(&path, 5, 300, &vec![1.0; 5 * 300]);
        let reader = SourceReader::open(&path).unwrap();

        let a = sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 42, 0.01, 0.99, 0)
            .unwrap();
        let b = sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 42, 0.01, 0.99, 0)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn abslog_transform_changes_the_estimated_limits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        let height = 5;
        let width = 300;
        write_test_nc_with(&path, height, width, &vec![100.0f32; height * width]);
        let reader = SourceReader::open(&path).unwrap();

        let (low_lin, high_lin) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 7, 0.01, 0.99, 0)
                .unwrap();
        let (low_log, high_log) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::AbsLog, 7, 0.01, 0.99, 0)
                .unwrap();
        assert!((low_lin - 100.0).abs() < 1e-3);
        assert!((low_log - 2.0).abs() < 1e-3); // log10(100) == 2
        assert_ne!((low_lin, high_lin), (low_log, high_log));
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn positive_transform_estimates_limits_from_absolute_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        let height = 5;
        let width = 300;
        // All-negative data: a `Linear` estimate would report both bounds
        // negative, but `Positive` estimates from `|x|`, so both bounds
        // should come back positive.
        write_test_nc_with(&path, height, width, &vec![-100.0f32; height * width]);
        let reader = SourceReader::open(&path).unwrap();

        let (low, high) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Positive, 7, 0.01, 0.99, 0)
                .unwrap();
        assert!((low - 100.0).abs() < 1e-3, "low={low}");
        assert!((high - 100.0).abs() < 1e-3, "high={high}");
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn skip_first_samples_excludes_the_direct_wave_band() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        let height = 20;
        let width = 300;
        // A huge-amplitude "direct wave" in the first 5 rows, small
        // amplitude everywhere else -- skipping those rows should keep the
        // estimate near the small values instead of the spike.
        let mut values = Vec::with_capacity(height * width);
        for row in 0..height {
            let v = if row < 5 { 1000.0 } else { 1.0 };
            for _ in 0..width {
                values.push(v);
            }
        }
        write_test_nc_with(&path, height, width, &values);
        let reader = SourceReader::open(&path).unwrap();

        let (_, high) =
            sampled_amplitude_limits(&reader, AmplitudeTransform::Linear, 0, 0.01, 0.99, 5)
                .unwrap();
        assert!(high < 10.0, "high={high} should exclude the skipped spike");
    }
}
