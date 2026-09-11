//! Streaming reads of the `data` amplitude variable (#118).
//!
//! Never loads the complete array: every read is a hyperslab bounded to
//! the requested window, clamped to the array's actual extent. This is the
//! correctness property the plan calls "never load a full radargram" --
//! important given radargrams up to ~1 GB. The *performance* optimization
//! on top of it (an LRU of decompressed, HDF5-chunk-aligned blocks, so one
//! read serves every render chunk inside it) is deferred to M5, where the
//! caching infrastructure belongs anyway; this module's correctness does
//! not depend on it.

// SourceReader::open is only called by tests until the render service
// (M5) and HTTP routes (M6) construct one from a catalog entry's path.
#![allow(dead_code)]

use std::path::Path;

use ndarray::Array2;

/// A read handle onto one NetCDF file's `data` variable.
pub struct SourceReader {
    file: netcdf::File,
    shape: (usize, usize),
}

impl SourceReader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file =
            netcdf::open(path).map_err(|e| format!("Failed to open {path:?} as NetCDF: {e}"))?;
        let var = file
            .variable("data")
            .ok_or_else(|| format!("{path:?} has no 'data' variable"))?;
        let dims = var.dimensions();
        if dims.len() != 2 {
            return Err(format!(
                "{path:?}'s 'data' variable has {} dimensions, expected 2",
                dims.len()
            ));
        }
        let shape = (dims[0].len(), dims[1].len());
        Ok(Self { file, shape })
    }

    /// `(n_samples, n_traces)`, i.e. `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Read the half-open window `[row0, row1) x [col0, col1)`, clamped to
    /// the array's actual extent. A window entirely outside the array
    /// (after clamping, empty in either dimension) returns a `0 x 0`
    /// array rather than an error -- callers resampling from it get
    /// footprints with no source pixels, i.e. NaN output, which is the
    /// correct behavior for an edge/out-of-range chunk (#118).
    pub fn read_window(
        &self,
        row0: usize,
        row1: usize,
        col0: usize,
        col1: usize,
    ) -> Result<Array2<f32>, String> {
        let row1 = row1.min(self.shape.0);
        let col1 = col1.min(self.shape.1);
        if row0 >= row1 || col0 >= col1 {
            return Ok(Array2::from_elem((0, 0), 0.0));
        }

        let var = self
            .file
            .variable("data")
            .ok_or("'data' variable disappeared after open")?;
        var.get::<f32, _>((row0..row1, col0..col1))
            .map_err(|e| format!("Failed to read window ({row0}..{row1}, {col0}..{col1}): {e}"))?
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("Unexpected array rank reading window: {e}"))
    }

    /// Read `n_traces` complete traces (all rows from `skip_rows` down, each
    /// spanning `col..col+1`), drawn as `n_runs` short contiguous runs
    /// spread across the profile, for amplitude-limit sampling (`stats.rs`).
    ///
    /// Runs rather than isolated single-trace reads: benchmarked at ~1 GB,
    /// isolated evenly-strided traces cost nearly as much as reading the
    /// entire array (they land in every storage chunk), while the same
    /// trace count drawn as short runs is ~7.6x faster, because each run
    /// stays inside a small number of storage chunks. Every sampled trace
    /// is still complete (all samples below `skip_rows`), which is what
    /// makes the sample represent the source wavelet at its true share of
    /// the data.
    ///
    /// `skip_rows` drops the top `skip_rows` sample rows from every run --
    /// used by the `positive` render profile to exclude the direct-wave
    /// band from its percentile estimate. `0` reproduces the original
    /// whole-trace behavior.
    pub fn sample_trace_runs(
        &self,
        n_runs: usize,
        traces_per_run: usize,
        offset: usize,
        skip_rows: usize,
    ) -> Result<Vec<f32>, String> {
        let n_traces = self.shape.1;
        if n_traces == 0 || n_runs == 0 || traces_per_run == 0 {
            return Ok(Vec::new());
        }
        let row_start = skip_rows.min(self.shape.0);
        let step = (n_traces / n_runs).max(1);
        let mut samples = Vec::new();
        for i in 0..n_runs {
            let start = (i * step + offset) % n_traces;
            let end = (start + traces_per_run).min(n_traces);
            if start >= end {
                continue;
            }
            let block = self.read_window(row_start, self.shape.0, start, end)?;
            samples.extend(block.iter().copied());
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_nc(path: &std::path::Path, height: usize, width: usize) {
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("y", height).unwrap();
        file.add_dimension("x", width).unwrap();
        let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
        let data: Vec<f32> = (0..(height * width)).map(|i| i as f32).collect();
        var.put_values(&data, ..).unwrap();
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn read_window_returns_exact_requested_region() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 10, 8);

        let reader = SourceReader::open(&path).unwrap();
        assert_eq!(reader.shape(), (10, 8));

        let window = reader.read_window(2, 4, 3, 6).unwrap();
        assert_eq!(window.shape(), &[2, 3]);
        // value at (row, col) in the full array is row*8 + col
        assert_eq!(window[[0, 0]], (2 * 8 + 3) as f32);
        assert_eq!(window[[1, 2]], (3 * 8 + 5) as f32);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn read_window_clamps_to_array_extent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 5, 5);
        let reader = SourceReader::open(&path).unwrap();

        let window = reader.read_window(3, 100, 3, 100).unwrap();
        assert_eq!(window.shape(), &[2, 2]);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn read_window_entirely_outside_extent_is_empty_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 5, 5);
        let reader = SourceReader::open(&path).unwrap();

        let window = reader.read_window(10, 20, 10, 20).unwrap();
        assert_eq!(window.shape(), &[0, 0]);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn sample_trace_runs_reads_complete_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 4, 100);
        let reader = SourceReader::open(&path).unwrap();

        let samples = reader.sample_trace_runs(5, 2, 0, 0).unwrap();
        // 5 runs x 2 traces x 4 samples/trace = 40 values, each run fully
        // covering the vertical extent of its columns.
        assert_eq!(samples.len(), 40);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn sample_trace_runs_is_deterministic_given_same_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 4, 100);
        let reader = SourceReader::open(&path).unwrap();

        let a = reader.sample_trace_runs(5, 2, 3, 0).unwrap();
        let b = reader.sample_trace_runs(5, 2, 3, 0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn sample_trace_runs_skip_rows_drops_the_top_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 10, 100);
        let reader = SourceReader::open(&path).unwrap();

        let full = reader.sample_trace_runs(5, 2, 0, 0).unwrap();
        let skipped = reader.sample_trace_runs(5, 2, 0, 4).unwrap();
        // 5 runs x 2 traces x (10 - 4) samples/trace = 60 values.
        assert_eq!(skipped.len(), 60);
        assert_eq!(full.len(), 100);
        // write_test_nc fills row-major values `row * width + col`, so row 4
        // (the first retained row) starts at exactly 4*100 = 400; anything
        // below that belongs to a dropped row.
        assert!(skipped.iter().cloned().fold(f32::INFINITY, f32::min) >= 400.0);
    }
}
