use std::error::Error;
/// Functions to process GPR data
use std::path::{Path, PathBuf};

use ndarray_stats::QuantileExt;
use std::time::{Duration, SystemTime};

use ndarray::{Array1, Array2, Axis, Slice};
use rayon::prelude::*;

use crate::{dem, filters, io, tools};

const DEFAULT_ZERO_CORR_THRESHOLD_MULTIPLIER: f32 = 1.0;
const DEFAULT_EMPTY_TRACE_STRENGTH: f32 = 1.0;
const DEFAULT_DEWOW_WINDOW: u32 = 5;
const DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF: f32 = 0.3;
const DEFAULT_AUTOGAIN_N_BINS: usize = 100;
const DEFAULT_BANDPASS_LOW_CUTOFF: f32 = 0.1;
const DEFAULT_BANDPASS_HIGH_CUTOFF: f32 = 0.9;
const DEFAULT_SIGLOG_MINVAL_LOG10: f32 = -1.;

/// Metadata associated with a GPR dataset
///
/// This contains all required information except the location data and the actual data
#[derive(Debug, Clone)]
pub struct GPRMeta {
    /// The number of samples per trace (the vertical data size)
    pub samples: u32,
    /// The control unit sampling frequency (MHz)
    pub frequency: f32,
    pub frequency_steps: u32,
    /// The interval between traces (s)
    pub time_interval: f32,
    /// The name of the antenna
    pub antenna: String,
    /// The frequency of the antenna (MHz)
    pub antenna_mhz: f32,
    /// The horizontal separation between the antenna transmitter and receiver (m)
    pub antenna_separation: f32,
    /// The return time window (ns)
    pub time_window: f32,
    /// The number of traces in the data (the horizontal data size)
    pub last_trace: u32,
    /// The path to the RD3 metadata file
    pub rd3_filepath: PathBuf,
    /// The velocity of the medium (m / ns)
    pub medium_velocity: f32,
}

impl GPRMeta {
    /// Find a ".cor" file based on the location of the ".rd3" file
    ///
    /// # Arguments
    /// - `projected_crs`: The CRS to project coordinates into
    pub fn find_cor(&self, projected_crs: &str) -> Result<GPRLocation, Box<dyn Error>> {
        io::load_cor(&self.rd3_filepath.with_extension("cor"), projected_crs)
    }
}

impl std::fmt::Display for GPRMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "
GPR Metadata
------------
Filepath:\t\t{:?}
Samples (height):\t{}
Traces (width):\t\t{}
Time window:\t\t{} ns
Max depth:\t\t{:.1} m
Medium velocity:\t{} m/ns
Sampling frequency:\t{} MHz
Time between traces:\t{} s
Antenna:\t\t{}
Antenna separation:\t{} m
",
            self.rd3_filepath,
            self.samples,
            self.last_trace,
            self.time_window,
            0.5 * self.time_window * self.medium_velocity,
            self.medium_velocity,
            self.frequency,
            self.time_interval,
            self.antenna,
            self.antenna_separation,
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CorPoint {
    pub trace_n: u32,
    pub time_seconds: f64,
    pub easting: f64,
    pub northing: f64,
    pub altitude: f64,
}

impl CorPoint {
    fn txyz(&self) -> [f64; 4] {
        [
            self.time_seconds,
            self.easting,
            self.northing,
            self.altitude,
        ]
    }
}

#[derive(Debug, Clone)]
pub enum LocationCorrection {
    None,
    Dem(PathBuf),
}

#[derive(Debug, Clone)]
pub struct GPRLocation {
    pub cor_points: Vec<CorPoint>,
    pub correction: LocationCorrection,
    pub crs: String,
}

impl GPRLocation {
    /// Get a coordinate of a trace by interp- or extrapolation
    ///
    /// If the trace_n is lower than or equal to the first trace, the first coordinate is given
    /// (bfill)
    ///
    /// If the trace_n is higher than or equal to the last trace, the last coordinate is given
    /// (ffill)
    ///
    /// If the trace_n is equal to an existing coordinate, that coordinate is given
    ///
    /// If the trace_n is between two existing coordinates, the linearly interpolated coordinate is
    /// given.
    fn time_and_coord_at_trace(&self, trace_n: u32) -> (f64, f64, f64, f64) {
        let mut first_point: &CorPoint = &self.cor_points[0];
        let last_point = &self.cor_points[self.cor_points.len() - 1];

        // Return the first point if the requested trace_n is equal or lower to the first.
        if trace_n <= first_point.trace_n {
            return (
                first_point.time_seconds,
                first_point.easting,
                first_point.northing,
                first_point.altitude,
            );
        };

        if trace_n < last_point.trace_n {
            // At this time, the trace
            for point in &self.cor_points {
                if point.trace_n == trace_n {
                    return (
                        point.time_seconds,
                        point.easting,
                        point.northing,
                        point.altitude,
                    );
                };

                if trace_n < point.trace_n {
                    let v = tools::interpolate_values(
                        first_point.trace_n as f64,
                        &first_point.txyz(),
                        point.trace_n as f64,
                        &point.txyz(),
                        trace_n as f64,
                    );
                    return (v[0], v[1], v[2], v[3]);
                };
                first_point = point;
            }
        };
        (
            last_point.time_seconds,
            last_point.easting,
            last_point.northing,
            last_point.altitude,
        )
    }

    fn velocities(&self) -> Array1<f64> {
        //let mut offsets: Vec<[f64; 4]> = Vec::new();

        let mut offsets = Array2::from_elem((self.cor_points.len(), 4), 0_f64);

        for i in 1..self.cor_points.len() {
            let mut slice =
                offsets.slice_axis_mut(Axis(0), Slice::new(i as isize, Some(i as isize + 1), 1));
            slice.assign(&Array1::from_vec(vec![
                self.cor_points[i].time_seconds - self.cor_points[i - 1].time_seconds,
                self.cor_points[i].easting - self.cor_points[i - 1].easting,
                self.cor_points[i].northing - self.cor_points[i - 1].northing,
                self.cor_points[i].altitude - self.cor_points[i - 1].altitude,
            ]));
        }

        let d = offsets
            .slice_axis(Axis(1), Slice::new(1, None, 1))
            .mapv(|f| f.powi(2))
            .sum_axis(Axis(1));

        let vel = (d / offsets.column(0)).mapv(|f| if f.is_finite() { f } else { 0.0 });

        vel
    }

    pub fn distances(&self) -> Array1<f64> {
        let mut offsets = Array2::from_elem((self.cor_points.len(), 3), 0_f64);

        for i in 1..self.cor_points.len() {
            let mut slice =
                offsets.slice_axis_mut(Axis(0), Slice::new(i as isize, Some(i as isize + 1), 1));
            slice.assign(&Array1::from_vec(vec![
                self.cor_points[i].time_seconds - self.cor_points[i - 1].time_seconds,
                self.cor_points[i].easting - self.cor_points[i - 1].easting,
                self.cor_points[i].northing - self.cor_points[i - 1].northing,
                //self.cor_points[i].altitude - self.cor_points[i - 1].altitude,
            ]));
        }

        let mut dist = offsets
            .slice_axis(Axis(1), Slice::new(1, None, 1))
            .mapv(|f| f.powi(2))
            .sum_axis(Axis(1));
        dist.accumulate_axis_inplace(Axis(0), |prev, cur| *cur += prev);

        dist
    }

    fn range_fill(&self, start_trace: u32, end_trace: u32) -> GPRLocation {
        let mut new_points: Vec<CorPoint> = Vec::new();

        for i in start_trace..end_trace {
            let txyz = self.time_and_coord_at_trace(i);

            new_points.push(CorPoint {
                trace_n: i,
                time_seconds: txyz.0,
                easting: txyz.1,
                northing: txyz.2,
                altitude: txyz.3,
            })
        }

        GPRLocation {
            cor_points: new_points,
            correction: self.correction.clone(),
            crs: self.crs.clone(),
        }
    }

    fn xy_coords(&self) -> Array2<f64> {
        let mut data: Vec<f64> = Vec::new();

        for point in &self.cor_points {
            data.push(point.easting);
            data.push(point.northing);
        }

        Array2::<f64>::from_shape_vec((self.cor_points.len(), 2), data).unwrap()
    }

    pub fn get_dem_elevations(&mut self, dem_path: &Path) {
        let elev = dem::read_elevations(dem_path, self.xy_coords()).unwrap();

        for i in 0..self.cor_points.len() {
            self.cor_points[i].altitude = elev[i] as f64;
        }

        self.correction = LocationCorrection::Dem(dem_path.to_path_buf());
    }

    pub fn to_csv(&self, filepath: &Path) -> Result<(), std::io::Error> {
        let mut output = "trace_n,easting,northing,altitude\n".to_string();

        for point in &self.cor_points {
            output += &format!(
                "{},{},{},{}\n",
                point.trace_n, point.easting, point.northing, point.altitude
            );
        }

        std::fs::write(filepath, output)
    }

    pub fn length(&self) -> f64 {
        self.distances().max().unwrap().to_owned()
    }

    pub fn duration_since(&self, other: &GPRLocation) -> f64 {
        let self_times = Array1::from_iter(self.cor_points.iter().map(|p| p.time_seconds));
        let other_times = Array1::from_iter(other.cor_points.iter().map(|p| p.time_seconds));

        let self_min = self_times.min().unwrap();
        let self_max = self_times.max().unwrap();

        let other_min = other_times.min().unwrap();
        let other_max = other_times.max().unwrap();

        if self_min > other_min {
            other_max - self_min
        } else {
            other_min - self_max
        }
    }
}

impl std::fmt::Display for GPRLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let altitudes = Array1::from_iter(self.cor_points.iter().map(|point| point.altitude));
        let eastings = Array1::from_iter(self.cor_points.iter().map(|point| point.easting));
        let northings = Array1::from_iter(self.cor_points.iter().map(|point| point.northing));

        let length = self.length();
        let duration = self.cor_points[self.cor_points.len() - 1].time_seconds
            - self.cor_points[0].time_seconds;
        write!(
            f,
            "

GPR Location data
-----------------
Start time:\t\t{}
Stop time:\t\t{}
Duration:\t\t{:.1} s
Number of points:\t{}
Points per m / per s:\t{:.2},{:.2}
Track length:\t\t{:.1} m
Altitude range:\t\t{:.1}-{:.1} m.
Centroid:\t\tE {:.1} m, N: {:.1} m, Z: {:.1} m.
CRS:\t\t\t{}
",
            tools::seconds_to_rfc3339(self.cor_points[0].time_seconds),
            tools::seconds_to_rfc3339(self.cor_points[self.cor_points.len() - 1].time_seconds),
            duration,
            self.cor_points.len(),
            self.cor_points.len() as f32 / length as f32,
            self.cor_points.len() as f64 / duration,
            length,
            altitudes.min().unwrap(),
            altitudes.max().unwrap(),
            eastings.mean().unwrap(),
            northings.mean().unwrap(),
            altitudes.mean().unwrap(),
            self.crs,
        )
    }
}

/// An in-memory GPR dataset
#[allow(clippy::upper_case_acronyms)]
pub struct GPR {
    /// The data matrix, with rows of traces (originally in mV)
    pub data: Array2<f32>,
    /// Topographically corrected data matrix, if processed.
    pub topo_data: Option<Array2<f32>>,
    /// Trace X/Y/Z location data
    pub location: GPRLocation,
    /// Metadata for the GPR dataset
    pub metadata: GPRMeta,
    /// Processing log. Each line is one processing step
    pub log: Vec<String>,
    /// The horizontal component of the signal distance (m). Defaults to the antenna separation if no correction has been made.
    horizontal_signal_distance: f32,
    /// The calculated zero-point (ns). It represents the delay between the transmitter and the receiver.
    zero_point_ns: f32,
}

impl GPR {
    pub fn process(&mut self, step_name: &str) -> Result<(), Box<dyn Error>> {
        if step_name.contains("dewow") {
            let window = tools::parse_option::<u32>(step_name, 0)?.unwrap_or(DEFAULT_DEWOW_WINDOW);

            self.dewow(window);
        } else if step_name.contains("zero_corr_max_peak") {
            self.zero_corr_max_peak();
        } else if step_name.contains("zero_corr") {
            let threshold_multiplier = tools::parse_option::<f32>(step_name, 0)?;

            self.zero_corr(threshold_multiplier);
        } else if step_name.contains("equidistant_traces") {
            let step = tools::parse_option::<f32>(step_name, 0)?;
            self.make_equidistant(step);
        } else if step_name.contains("normalize_horizontal_magnitudes") {
            // Try to parse the argument as an integer. If that doesn't work, try to parse it as a
            // float and assume it's the fraction of the height
            let skip_first: isize =
                match tools::parse_option::<isize>(step_name, 0) {
                    Ok(v) => Ok(v.unwrap_or(0)),
                    Err(e) => match e.contains("Could not parse argument 0 as value") {
                        true => tools::parse_option::<f32>(step_name, 0).and_then(|fraction| {
                            match (fraction.unwrap() >= 1.0) & (fraction.unwrap() < 0.) {
                                true => Err(format!(
                                    "Invalid fraction: {:?}. Must be between 0.0 and 1.0.",
                                    fraction
                                )),
                                false => {
                                    Ok((self.height() as f32 * fraction.unwrap_or(0.)) as isize)
                                }
                            }
                        }),
                        false => Err(e),
                    },
                }?;
            self.normalize_horizontal_magnitudes(Some(skip_first));
        } else if step_name.contains("kirchhoff_migration2d") {
            self.kirchhoff_migration2d();
        } else if step_name.contains("auto_gain") {
            let n_bins =
                tools::parse_option::<usize>(step_name, 0)?.unwrap_or(DEFAULT_AUTOGAIN_N_BINS);
            self.auto_gain(n_bins);
        } else if step_name.contains("gain") {
            let factor = match tools::parse_option::<f32>(step_name, 0)? {
                Some(v) => Ok(v),
                None => Err(
                    "The gain factor must be specified when applying gain. E.g. gain(0.1)"
                        .to_string(),
                ),
            }?;
            self.gain(factor);
        } else if step_name.contains("subset") {
            let min_trace: Option<u32> = match tools::parse_option::<u32>(step_name, 0)? {
                Some(v) => Ok(Some(v)),
                None => Err("Indices must be given when subsetting, e.g. subset(0, -1, 0, 500)"),
            }?;

            let max_trace: Option<u32> = match tools::parse_option::<isize>(step_name, 1) {
                Ok(v) => Ok(v.and_then(|v2| if v2 == -1 { None } else { Some(v2 as u32) })),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                },
            }?;
            let min_sample: Option<u32> = match tools::parse_option::<u32>(step_name, 2) {
                Ok(v) => Ok(v),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                },
            }?;
            let max_sample: Option<u32> = match tools::parse_option::<isize>(step_name, 3) {
                Ok(v) => Ok(v.and_then(|v2| if v2 == -1 { None } else { Some(v2 as u32) })),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                },
            }?;
            *self = self.subset(min_trace, max_trace, min_sample, max_sample);
        } else if step_name.contains("unphase") {
            self.unphase();
        } else if step_name.contains("abslog") {
            self.abslog()
        } else if step_name.contains("siglog") {
            let minval =
                tools::parse_option::<f32>(step_name, 0)?.unwrap_or(DEFAULT_SIGLOG_MINVAL_LOG10);
            self.siglog(minval);
        } else if step_name.contains("correct_topography") {
            self.correct_topography();
        } else if step_name.contains("correct_antenna_separation") {
            self.correct_antenna_separation();
        } else if step_name.contains("remove_traces") {
            let mut traces = Vec::<usize>::new();
            for i in 0..self.width() {
                // Try to parse the i:th option as an usize.
                // If that doesn't work, it's either a range (e.g. 1-3) or it's poorly formatted
                if let Some(trace) = tools::parse_option::<usize>(step_name, i).ok().flatten() {
                    traces.push(trace);
                } else {
                    // Extract the option as a string. If i is out of bounds, this one will fail (and break the loop)
                    if let Some(token) = tools::parse_option::<String>(step_name, i).ok().flatten()
                    {
                        // Start trying to parse it as a range (e.g. 1-3), or else give helpful messages.
                        if !token.contains("-") {
                            return Err(
                                format!("Error reading 'remove_traces' argument: {token}").into()
                            );
                        }
                        let mut new_traces = Vec::<usize>::new();
                        let parts: Vec<&str> = token.split('-').collect();
                        if let (Some(start_str), Some(end_str)) = (parts.first(), parts.get(1)) {
                            if let (Ok(start), Ok(end)) =
                                (start_str.parse::<usize>(), end_str.parse::<usize>())
                            {
                                for value in start..=end {
                                    new_traces.push(value);
                                }
                            }
                        }
                        if new_traces.is_empty() {
                            return Err(
                                format!("Error reading 'remove_traces' argument: {token}").into()
                            );
                        };
                        traces.append(&mut new_traces);
                    } else {
                        break;
                    }
                }
            }
            if traces.is_empty() {
                return Err(
                    "Indices must be given when calling remove_traces, e.g. remove_traces(0 1 5)"
                        .into(),
                );
            };
            self.remove_traces(&traces, true)?;
        } else if step_name.contains("remove_empty_traces") {
            let strength =
                tools::parse_option::<f32>(step_name, 0)?.unwrap_or(DEFAULT_EMPTY_TRACE_STRENGTH);

            self.remove_empty_traces(strength)?;
        } else if step_name.contains("bandpass") {
            let low_cutoff =
                tools::parse_option(step_name, 0)?.unwrap_or(DEFAULT_BANDPASS_LOW_CUTOFF);
            let high_cutoff =
                tools::parse_option(step_name, 1)?.unwrap_or(DEFAULT_BANDPASS_HIGH_CUTOFF);
            self.bandpass(low_cutoff, high_cutoff)?;
        } else {
            return Err(format!("Step name not recognized: {}", step_name).into());
        }

        Ok(())
    }

    pub fn bandpass(&mut self, low_cutoff: f32, high_cutoff: f32) -> Result<(), String> {
        let start_time = SystemTime::now();

        if (low_cutoff < 0.) | (low_cutoff > 1.) {
            return Err(format!(
                "Normalized low cutoff needs to be in the range 0-1 (provided: {low_cutoff})"
            ));
        }
        if (high_cutoff < 0.) | (high_cutoff > 1.) {
            return Err(format!(
                "Normalized high cutoff needs to be in the range 0-1 (provided: {high_cutoff})"
            ));
        }
        if low_cutoff >= high_cutoff {
            return Err(format!("Normalized low cutoff ({low_cutoff}) needs to be smaller than the high cutoff ({high_cutoff})."));
        }

        match filters::normalized_bandpass(&mut self.data, low_cutoff, high_cutoff) {
            Ok(()) => (),
            Err(e) => return Err(format!("Error in bandpass function: {:?}", e)),
        }

        self.log_event(
            "bandpass",
            &format!(
                "Applied a normalized bandpass Butterworth filter ({:.3}-{:.3})",
                low_cutoff, high_cutoff
            ),
            start_time,
        );
        Ok(())
    }

    pub fn subset(
        &self,
        min_trace: Option<u32>,
        max_trace: Option<u32>,
        min_sample: Option<u32>,
        max_sample: Option<u32>,
    ) -> GPR {
        let start_time = SystemTime::now();
        let min_trace_ = min_trace.unwrap_or(0);
        let max_trace_ = match max_trace {
            Some(x) => x,
            None => self.width() as u32,
        };
        let min_sample_ = min_sample.unwrap_or(0);
        let max_sample_ = match max_sample {
            Some(x) => x,
            None => self.height() as u32,
        };

        let data_subset = self
            .data
            .slice(ndarray::s![
                min_sample_ as isize..max_sample_ as isize,
                min_trace_ as isize..max_trace_ as isize
            ])
            .to_owned();

        let location_subset = GPRLocation {
            cor_points: self.location.cor_points[min_trace_ as usize..max_trace_ as usize]
                .to_owned(),
            correction: self.location.correction.clone(),
            crs: self.location.crs.clone(),
        };

        let mut metadata = self.metadata.clone();

        metadata.last_trace = max_trace_;
        metadata.time_window *= (max_sample_ - min_sample_) as f32 / metadata.samples as f32;
        metadata.samples = max_sample_ - min_sample_;

        let log = self.log.clone();

        let mut new_gpr = GPR {
            data: data_subset,
            location: location_subset,
            metadata,
            log,
            topo_data: self.topo_data.clone(),
            horizontal_signal_distance: self.horizontal_signal_distance,
            zero_point_ns: self.zero_point_ns,
        };
        new_gpr.log_event(
            "subset",
            &format!(
                "Subset data from {:?} to ({}:{}, {}:{})",
                self.data.shape(),
                min_sample_,
                max_sample_,
                min_trace_,
                max_trace_
            ),
            start_time,
        );

        new_gpr
    }

    pub fn vertical_resolution_ns(&self) -> f32 {
        self.metadata.time_window / self.metadata.samples as f32
    }

    pub fn from_meta_and_loc(
        location: GPRLocation,
        metadata: GPRMeta,
    ) -> Result<GPR, Box<dyn Error>> {
        let data = io::load_rd3(&metadata.rd3_filepath, metadata.samples as usize)?;

        let location_data = match data.shape()[1] == location.cor_points.len() {
            true => location,
            false => location.range_fill(0, data.shape()[1] as u32),
        };
        let horizontal_signal_distance = metadata.antenna_separation;

        Ok(GPR {
            data,
            location: location_data,
            metadata,
            log: Vec::new(),
            topo_data: None,
            horizontal_signal_distance,
            zero_point_ns: 0.,
        })
    }

    pub fn render(&self, filepath: &Path) -> Result<(), Box<dyn Error>> {
        io::render_jpg(self, filepath)
    }

    pub fn zero_corr_max_peak(&mut self) {
        let start_time = SystemTime::now();

        let mean_trace = self.data.mean_axis(Axis(1)).unwrap();

        let threshold = 0.5 * mean_trace.std(1.0);

        let mut first_rise = 0_isize;

        for i in 1..mean_trace.shape()[0] {
            if (mean_trace[i] - mean_trace[i - 1]).abs() > threshold {
                first_rise = i as isize;
                break;
            };
        }

        if first_rise == 0 {
            return;
        };

        let mean_silent_val = mean_trace
            .slice_axis(Axis(0), Slice::new(0, Some(first_rise), 1))
            .mean()
            .unwrap();

        self.data -= mean_silent_val;

        let mut positive_peaks = Array1::<isize>::zeros(self.width());

        let mut i = 0_usize;
        for col in self.data.columns() {
            positive_peaks[i] = col.argmax().unwrap() as isize;

            i += 1;
        }

        let mut new_data = Array2::from_elem(
            (
                self.height() - positive_peaks.min().unwrap().to_owned() as usize,
                self.width(),
            ),
            0_f32,
        );

        i = 0;
        for col in self.data.columns() {
            let mut new_col = new_data.column_mut(i);

            let mut positive_data_slice = new_col.slice_axis_mut(
                Axis(0),
                Slice::new(0, Some(self.height() as isize - positive_peaks[i]), 1),
            );

            positive_data_slice += &col.slice_axis(Axis(0), Slice::new(positive_peaks[i], None, 1));
            i += 1;
        }

        self.zero_point_ns = self.metadata.time_window * (positive_peaks.mean().unwrap() as f32)
            / self.height() as f32;
        self.update_data(new_data);
        self.log_event(
            "zero_corr_max_peak",
            &format!(
                "Applied a per-trace zero-corr by removing the first {}-{} rows",
                positive_peaks.min().unwrap(),
                positive_peaks.max().unwrap()
            ),
            start_time,
        );
    }

    fn update_data(&mut self, data: Array2<f32>) {
        self.data = data;

        self.metadata.time_window *= self.height() as f32 / self.metadata.samples as f32;
        self.metadata.samples = self.height() as u32;
        self.metadata.last_trace = self.width() as u32;
    }

    pub fn correct_antenna_separation(&mut self) {
        let start_time = SystemTime::now();

        if self.horizontal_signal_distance == 0. {
            self.log_event(
                "correct_antenna_separation",
                "Skipping antenna separation correction since the antenna separation is 0 m.",
                start_time,
            );
            return;
        };

        let height_before = self.height();

        let depths = self.depths();

        if depths.max().unwrap() == &0. {
            eprintln!("correct_antenna_separation failed. Max depth after antenna correction ({} m) would be 0 m", self.horizontal_signal_distance);
            panic!("");
        }

        let resolution = self.vertical_resolution_m();
        let resampler = tools::Resampler::<f32>::new(depths, resolution);

        //resampler.resample_along_axis(&mut self.data, tools::Axis2D::Row);
        self.update_data(resampler.resample_along_axis_par(&self.data, tools::Axis2D::Row));
        //tools::groupby_average(&mut self.data, tools::Axis2D::Row, &depths, *max_diff);
        self.log_event("correct_antenna_separation", &format!("Standardized depths to {} m ({} ns) per pixel by accounting for an antenna separation of {} m (height changed from {} px to {} px).", resolution, resolution / (self.metadata.time_window / self.height() as f32), self.horizontal_signal_distance, height_before, self.height()), start_time);

        self.horizontal_signal_distance = 0.;
        self.metadata.samples = self.height() as u32;
    }

    pub fn abslog(&mut self) {
        let start_time = SystemTime::now();

        filters::abslog(&mut self.data);
        self.log_event("abslog", "Ran abslog (log10(abs(data))", start_time);
    }

    pub fn siglog(&mut self, minval_log10: f32) {
        let start_time = SystemTime::now();

        filters::siglog(&mut self.data, minval_log10);
        self.log_event(
            "siglog",
            "Ran siglog (sign-corrected log10 of absolute values) (minval: 10e{minval_log10}))",
            start_time,
        );
    }

    pub fn vertical_resolution_m(&self) -> f32 {
        let depths = self.depths();

        let mut diffs = Array1::<f32>::zeros((depths.shape()[0] - 1,));
        for i in 1..depths.shape()[0] {
            diffs[i - 1] = depths[i] - depths[i - 1];
        }
        tools::quantiles(&diffs, &[0.8], None)[0]
    }

    pub fn correct_topography(&mut self) {
        let start_time = SystemTime::now();

        let mut altitudes = Array1::<f32>::from_iter(
            self.location
                .cor_points
                .iter()
                .map(|point| point.altitude as f32),
        );
        altitudes -= *altitudes.max().unwrap();
        altitudes *= -1.;

        let max_depth = tools::return_time_to_depth(
            self.metadata.time_window,
            self.metadata.medium_velocity,
            self.metadata.antenna_separation,
        );

        let sample_per_meter = self.height() as f32 / max_depth;

        let start_indices = altitudes.mapv(|altitude| (altitude * sample_per_meter) as isize);

        let mut topo_data = Array2::<f32>::zeros((
            ((max_depth + altitudes.max().unwrap()) * sample_per_meter) as usize,
            self.width(),
        ));

        for (i, col) in self.data.columns().into_iter().enumerate() {
            topo_data
                .column_mut(i)
                .slice_axis_mut(
                    Axis(0),
                    Slice::new(
                        start_indices[i],
                        Some(self.height() as isize + start_indices[i]),
                        1,
                    ),
                )
                .assign(&col);
        }

        self.topo_data = Some(topo_data);

        self.log_event(
            "correct_topography",
            "Generated a profile that is corrected for topography (topo_data).",
            start_time,
        );
    }

    pub fn unphase(&mut self) {
        let start_time = SystemTime::now();

        let mut positive_peaks = Array1::<isize>::zeros(self.width());
        let mut negative_peaks = Array1::<isize>::zeros(self.width());

        for (i, col) in self.data.columns().into_iter().enumerate() {
            positive_peaks[i] = col.argmax().unwrap() as isize;
            negative_peaks[i] = col.argmin().unwrap() as isize;
        }

        let mean_peak_spacing = (negative_peaks - positive_peaks).mean().unwrap().abs();
        let mut new_data = self.data.mapv(|v| v.max(0.));

        self.data
            .slice_axis(Axis(0), Slice::new(mean_peak_spacing, None, 1))
            .mapv(|v| v.min(0.) * -1.)
            .assign_to(new_data.slice_axis_mut(
                Axis(0),
                Slice::new(0, Some(self.height() as isize - mean_peak_spacing), 1),
            ));

        self.update_data(new_data);

        self.log_event("unphase", &format!("Summed the positive and negative phases of the signal by shifting the negative signal component by {} rows", mean_peak_spacing), start_time);
    }

    pub fn zero_corr(&mut self, threshold_multiplier: Option<f32>) {
        let start_time = SystemTime::now();

        let mean_trace = self.data.mean_axis(Axis(1)).unwrap();

        let threshold = 0.5
            * mean_trace.std(1.0)
            * threshold_multiplier.unwrap_or(DEFAULT_ZERO_CORR_THRESHOLD_MULTIPLIER);

        let mut first_rise = 0_isize;

        for i in 1..mean_trace.shape()[0] {
            if (mean_trace[i] - mean_trace[i - 1]).abs() > threshold {
                first_rise = i as isize;
                break;
            };
        }

        if first_rise == 0 {
            return;
        };

        let mean_silent_val = mean_trace
            .slice_axis(Axis(0), Slice::new(0, Some(first_rise), 1))
            .mean()
            .unwrap();

        self.zero_point_ns = self.metadata.time_window * (first_rise as f32) / self.height() as f32;
        self.update_data(
            self.data
                .slice_axis(Axis(0), Slice::new(first_rise, None, 1))
                .to_owned(),
        );
        self.data -= mean_silent_val;

        self.log_event("zero_corr", &format!("Applied a global zero-corr by removing the first {} rows (threshold multiplier: {:?})", first_rise, threshold_multiplier), start_time);
    }

    pub fn dewow(&mut self, window: u32) {
        let start_time = SystemTime::now();

        let height = self.height() as u32;

        for i in (0..(height - window)).step_by(window as usize) {
            let mut view = self.data.slice_axis_mut(
                Axis(0),
                ndarray::Slice::new(i as isize, Some((i + window) as isize), 1_isize),
            );

            view -= view.mean().unwrap();
        }
        self.log_event(
            "dewow",
            &format!("Ran dewow with a window size of {}", window),
            start_time,
        );
    }

    pub fn normalize_horizontal_magnitudes(&mut self, skip_first: Option<isize>) {
        let start_time = SystemTime::now();
        if let Some(mean) = self
            .data
            .slice_axis(Axis(0), Slice::new(skip_first.unwrap_or(0), None, 1))
            .mean_axis(Axis(0))
        {
            self.data -= &mean;
        };
        self.log_event(
            "normalize_horizontal_magnitudes",
            &format!(
                "Normalized horizontal magnitudes, skipping {:?} of the first rows",
                skip_first
            ),
            start_time,
        );
    }

    pub fn auto_gain(&mut self, n_bins: usize) {
        let start_time = SystemTime::now();

        let step = ((self.height() / n_bins) as isize).max(1);

        let mut old_att: Option<f32> = None;
        let mut attenuations: Vec<f32> = Vec::new();

        for i in (0..(self.height() as isize - step)).step_by(step as usize) {
            let slice = self
                .data
                .slice_axis(Axis(0), Slice::new(i, Some(i + step), step));

            let new_att = slice.mapv(|a| a.abs().log10()).mean().unwrap();
            if let Some(old) = old_att {
                attenuations.push(old - new_att);
            }
            old_att = Some(new_att);
        }

        let median_att = tools::quantiles(&attenuations, &[0.5], None)[0];

        let slope = (median_att.abs() / ((self.height() as f32) / (n_bins as f32))).sqrt()
            * median_att.signum();

        self.log_event(
            "auto_gain",
            &format!("Measured gain factor using autogain from {} bins", n_bins),
            start_time,
        );
        self.gain(slope);
    }

    pub fn gain(&mut self, factor: f32) {
        let start_time = SystemTime::now();

        for i in 0..self.height() as isize {
            let mut view = self
                .data
                .slice_axis_mut(Axis(0), Slice::new(i, Some(i + 1), 1_isize));

            view *= 10_f32.powf(factor * (i as f32).sqrt());
        }
        self.log_event(
            "gain",
            &format!("Applied gain of *= 10^({factor} * sqrt(index))",),
            start_time,
        );
    }

    pub fn make_equidistant(&mut self, step: Option<f32>) {
        let start_time = SystemTime::now();
        let distances = self.location.distances().mapv(|v| v as f32);
        let max_distance = distances.max().unwrap();

        let step = step.unwrap_or({
            let velocities = self.location.velocities().mapv(|v| v as f32);

            let normal_velocity = tools::quantiles(&velocities, &[0.5], None)[0];

            let mut seconds_moving = 0_f32;
            for i in 1..self.width() {
                if velocities[i] < (0.3 * normal_velocity) {
                    continue;
                };
                seconds_moving += self.metadata.time_interval;
            }

            let nominal_data_width =
                (seconds_moving / self.metadata.time_interval).floor() as usize;

            max_distance / (nominal_data_width as f32)
        });

        if (max_distance / step).round() as usize == self.width() {
            self.log_event(
                "equidistant_traces",
                "Traces were already equidistant.",
                start_time,
            );
            return;
        };

        let resampler = tools::Resampler::<f32>::new(distances, step);
        //resampler.resample_along_axis(&mut self.data, tools::Axis2D::Col);
        self.update_data(resampler.resample_along_axis_par(&self.data, tools::Axis2D::Col));

        //let eastings = resampler.resample(&Array1::from_vec(self.location.cor_points.iter().map(|p| p.easting).collect::<Vec<f64>>()).view());
        let eastings = resampler.resample_convert::<f64>(
            &Array1::from_vec(
                self.location
                    .cor_points
                    .iter()
                    .map(|p| p.easting)
                    .collect::<Vec<f64>>(),
            )
            .view(),
        );
        let northings = resampler.resample_convert::<f64>(
            &Array1::from_vec(
                self.location
                    .cor_points
                    .iter()
                    .map(|p| p.northing)
                    .collect::<Vec<f64>>(),
            )
            .view(),
        );
        let times = resampler.resample_convert::<f64>(
            &Array1::from_vec(
                self.location
                    .cor_points
                    .iter()
                    .map(|p| p.time_seconds)
                    .collect::<Vec<f64>>(),
            )
            .view(),
        );
        let altitudes = resampler.resample_convert::<f64>(
            &Array1::from_vec(
                self.location
                    .cor_points
                    .iter()
                    .map(|p| p.altitude)
                    .collect::<Vec<f64>>(),
            )
            .view(),
        );

        let mut cor_points = Vec::<CorPoint>::new();

        for i in 0..eastings.len() {
            let cor = CorPoint {
                trace_n: i as u32,
                time_seconds: times[i],
                easting: eastings[i],
                northing: northings[i],
                altitude: altitudes[i],
            };
            cor_points.push(cor);
        }

        self.metadata.last_trace = self.data.shape()[1] as u32;
        self.location.cor_points = cor_points;
        self.log_event(
            "equidistant_traces",
            &format!("Ran equidistant traces with a spacing of {step} m"),
            start_time,
        );
        /*
        let breaks = tools::groupby_average(&mut self.data, tools::Axis2D::Col, &distances, step);

        self.metadata.last_trace = self.data.shape()[1] as u32;

        self.location.cor_points = breaks
            .iter()
            .map(|i| self.location.cor_points[*i].clone())
            .collect::<Vec<CorPoint>>();
        self.location
            .cor_points
            .insert(0, self.location.cor_points[0].clone());

        if self.width() != self.location.cor_points.len() {
            panic!(
                "Data width {} != cor points width {}. Was equidistant traces run twice?",
                self.width(),
                self.location.cor_points.len()
            )
        };

        self.log_event("equidistant_traces", "Ran equidistant traces", start_time);
        */
    }

    fn log_event(&mut self, step_name: &str, event: &str, start_time: SystemTime) {
        self.log.push(format!(
            "{} (duration: {:.2}s):\t{}",
            step_name,
            SystemTime::now()
                .duration_since(start_time)
                .unwrap()
                .as_secs_f32(),
            event
        ));
    }

    pub fn kirchhoff_migration2d(&mut self) {
        let start_time = SystemTime::now();
        let x_coords = self.location.distances().mapv(|v| v as f32);
        let mut x_diff = 0_f32;
        for i in 1..x_coords.shape()[0] {
            x_diff += x_coords[i] - x_coords[i - 1];
        }
        x_diff /= (x_coords.shape()[0] - 1) as f32;

        // The z-coords will be negative height in relation to the highest point (all values are
        // positive).
        let mut z_coords = Array1::from_iter(
            self.location
                .cor_points
                .iter()
                .map(|point| point.altitude as f32),
        );
        z_coords -= *z_coords.max().unwrap();
        z_coords *= -1.0;

        // The vertical resolution in ns
        let t_diff = self.vertical_resolution_ns();

        // The vertical resolution in m
        let z_diff = t_diff * self.metadata.medium_velocity;

        // The minimum logical resolution, assuming just one wavelength
        let logical_res =
            (self.metadata.antenna_mhz / self.metadata.medium_velocity) * (1e-9 * 1e6);

        let old_data = self.data.iter().collect::<Vec<&f32>>();

        let height = self.height();
        let width = self.width();

        let output: Vec<f32> = (0..(width * height))
            .into_par_iter()
            .map(|sample_idx| {
                let row = sample_idx / width;
                let trace_n = sample_idx - (row * width);
                let trace_top_z = z_coords[trace_n];
                let trace_x = x_coords[trace_n];

                // The expected two-way time of the sample (assuming it is straight down)
                let t_0 = 2. * row as f32 * t_diff;
                let t_0_px = row;

                // Derive the Fresnel zone
                // This is the radius in which the sample may be affected horizontally
                // Pérez-Gracia et al. (2008) Horizontal resolution in a non-destructive
                // shallow GPR survey: An experimental evaluation. NDT & E International,
                // 41(8): 611–620. doi:10.1016/j.ndteint.2008.06.002
                let fresnel_radius = 0.5 * (logical_res * 2. * z_diff * row as f32).sqrt();

                // Derive the radius in pixel space
                let fresnel_width = fresnel_radius / x_diff;

                // If the fresnel width is zero pixels, no neighbors will change the sample. Therefore, just
                // take the old value and put it in the topographically correct place.
                if fresnel_width < 0.1 {
                    return old_data[(t_0_px * width) + trace_n].to_owned();
                };

                // Derive all of the neighboring columns that may affect the sample
                let min_neighbor = (trace_n as f32 - fresnel_width)
                    .floor()
                    .clamp(0_f32, width as f32) as usize;
                let max_neighbor = (trace_n + fresnel_width.ceil() as usize + 1).clamp(0, width);

                let mut ampl = 0_f32;
                let mut n_ampl = 0_f32;

                // Pixels that are entirely within the fresnel width should be included fully. Pixels
                // outside should be excluded. Pixels that border the fresnel width should be included
                // in a weighted average. Here, these weights are created. If the neighbour-trace
                // distance is larger than the fresnel width, the weight is first assigned 1.0.
                // The ones bordering the fresnel width are given the fraction how how much is covered
                // , e.g. 24% (0.24). With a fresnel width of e.g. 1.2, there would be three weights:
                // [0.2, 1.0, 0.2]
                let n_neighbors = (max_neighbor - min_neighbor) as f32;
                let in_fresnel_weight = n_neighbors / (n_neighbors - 2.);
                let out_fresnel_weight = fresnel_width.fract();

                for neighbor_n in min_neighbor..max_neighbor {
                    // Get the vertical component of the two-way time to the neighboring sample.
                    let t_top = t_0
                        - 2. * (z_coords[neighbor_n] - trace_top_z) / self.metadata.medium_velocity;

                    // Get the travel time to the sample accounting for the x distance
                    let t_x = (t_top.powi(2)
                        + (2. * (x_coords[neighbor_n] - trace_x) / self.metadata.medium_velocity)
                            .powi(2))
                    .sqrt();
                    // The sample will be in either the pixel when rounding down or when rounding up
                    // ... so these will both be evaluated
                    // These values have pixel units, as they are normalized to pixel resolution
                    let t_1_px = (0.5 * t_x / t_diff).floor() as usize;
                    let mut t_2_px = (0.5 * t_x / t_diff).ceil() as usize;

                    if t_2_px >= height {
                        t_2_px = t_1_px;
                    };

                    // If the travel times are within the bounds of the data and the pixel displacement is not zero,
                    // ... append a weighted amplitude accounting for the displacement distance
                    if t_1_px < height {
                        let weight = match t_1_px == t_2_px {
                            true => 0_f32,
                            false => ((t_1_px as f32 - (0.5 * t_x / t_diff))
                                / (t_1_px as f32 - t_2_px as f32))
                                .abs(),
                        };

                        ampl += (x_diff / (2. * std::f32::consts::PI * t_x * self.metadata.medium_velocity).sqrt()) // Account for the horizontal distance
                        * (t_top / t_x) // Account for the vertical distance
                        * ((1. - weight) * old_data[(t_1_px * width) + neighbor_n] + weight * old_data[(t_2_px * width) + neighbor_n]) // Account for the neigbour's value
                        * if (neighbor_n == min_neighbor) | (neighbor_n == max_neighbor - 1)
                         {out_fresnel_weight} else {in_fresnel_weight};
                        n_ampl += 1.0;
                    };
                }

                if n_ampl > 0. {
                    ampl / n_ampl
                } else {
                    0.
                }
            })
            .collect();

        self.update_data(Array2::from_shape_vec((height, width), output).unwrap());
        self.log_event(
            "kirchhoff_migration2d",
            &format!(
                "Ran 2D Kirchhoff migration with a velocity of {} m/ns",
                self.metadata.medium_velocity
            ),
            start_time,
        );
    }

    /// Remove traces based on their integer index
    pub fn remove_traces(&mut self, traces: &[usize], log: bool) -> Result<(), String> {
        let start_time = SystemTime::now();
        // The width will be called multiple times, so it's better to assign it statically.
        let width = self.width();

        // Make the traces unique, and validate that they are not out of bounds
        let mut unique_traces = Vec::<usize>::new();
        for trace in traces {
            if trace >= &width {
                return Err(format!(
                    "Trace index {trace:?} is out of bounds (number of traces: {width})"
                ));
            }
            if !unique_traces.contains(trace) {
                unique_traces.push(*trace);
            }
        }

        // Make a vec of all traces to keep (faster to subset than to explicitly remove in ndarray)
        // This is done before the index trick below, because the trick below is only needed for vecs.
        let traces_to_keep: Vec<usize> =
            (0..width).filter(|i| !unique_traces.contains(i)).collect();

        // Sort them, and then subtract the amount of removals that are done before this index.
        // For example, if trace 0,1 and 2 should be removed, by the time the loop reaches 2, it's now the zeroth index.
        // Therefore, the true index is index - i, where i is the count of indices before
        unique_traces.sort_unstable();
        for i in 0..unique_traces.len() {
            // This is ugly but makes cargo clippy happy. Sorry, future developers!
            let i2 = i;
            unique_traces[i2] -= i;
        }

        // Remove each selected trace from the data, (potentially) topo. corr. data, and the location info.
        self.data = self.data.select(Axis(1), &traces_to_keep);
        if let Some(data) = self.topo_data.as_mut() {
            self.topo_data = Some(data.select(Axis(1), &traces_to_keep));
        };
        for trace in &unique_traces {
            self.location.cor_points.remove(*trace);
        }
        self.metadata.last_trace = self.width() as u32;

        if log {
            self.log_event(
                "remove_traces",
                &format!("Removed trace indices {unique_traces:?}"),
                start_time,
            );
        }

        Ok(())
    }
    // Remove all traces whose absolute mean is lower than the given "strength"
    pub fn remove_empty_traces(&mut self, strength: f32) -> Result<(), String> {
        let start_time = SystemTime::now();
        let mut traces_to_remove = Vec::<usize>::new();

        for (i, col) in self.data.columns().into_iter().enumerate() {
            if let Some(mad) = col.mapv(|v| v.abs()).mean() {
                if mad > strength {
                    continue;
                };
                traces_to_remove.push(i);
            }
        }

        let n_removed = traces_to_remove.len();
        self.remove_traces(&traces_to_remove, false)?;

        self.log_event(
            "remove_empty_traces",
            &format!("Removed {n_removed} empty traces (strength: {strength})."),
            start_time,
        );

        Ok(())
    }

    pub fn height(&self) -> usize {
        self.data.shape()[0]
    }
    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }
    pub fn export(&self, nc_filepath: &Path) -> Result<(), Box<dyn Error>> {
        io::export_netcdf(self, nc_filepath)
    }

    pub fn depths(&self) -> Array1<f32> {
        let time_windows = (Array1::<f32>::range(0., self.height() as f32, 1.)
            / self.height() as f32)
            * self.metadata.time_window;
        let corr_antenna_separation = (self.horizontal_signal_distance.powi(2)
            - (self.zero_point_ns * self.metadata.medium_velocity).powi(2))
        .max(0.)
        .sqrt();
        time_windows.mapv(|time| {
            tools::return_time_to_depth(
                time,
                self.metadata.medium_velocity,
                corr_antenna_separation,
            )
        })
    }

    pub fn merge(&mut self, other: &GPR) -> Result<(), String> {
        let start_time = SystemTime::now();
        if self.location.crs != other.location.crs {
            Err(format!(
                "CRS are different: {} vs {}",
                self.location.crs, other.location.crs
            ))
        } else if self.metadata.antenna_mhz != other.metadata.antenna_mhz {
            Err(format!(
                "Antenna frequencies are different: {} vs {}",
                self.metadata.antenna_mhz, other.metadata.antenna_mhz
            ))
        } else if self.metadata.time_window != other.metadata.time_window {
            Err(format!(
                "Time windows are different: {} vs {}",
                self.metadata.time_window, other.metadata.time_window
            ))
        } else {
            self.location
                .cor_points
                .append(other.location.cor_points.clone().as_mut());

            self.data.append(Axis(1), other.data.view()).unwrap();

            self.metadata.time_window *= self.height() as f32 / self.metadata.samples as f32;
            self.metadata.samples = self.height() as u32;
            self.metadata.last_trace = self.width() as u32;

            self.log_event(
                "merge",
                &format!("Merged {:?}", other.metadata.rd3_filepath),
                start_time,
            );

            Ok(())
        }
    }
}
pub struct RunParams {
    pub filepaths: Vec<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub only_info: bool,
    pub dem_path: Option<PathBuf>,
    pub cor_path: Option<PathBuf>,
    pub medium_velocity: f32,
    pub crs: String,
    pub quiet: bool,
    pub track_path: Option<Option<PathBuf>>,
    pub steps: Vec<String>,
    pub no_export: bool,
    pub render_path: Option<Option<PathBuf>>,
    pub merge: Option<Duration>,
}

pub fn run(params: RunParams) -> Result<Vec<GPR>, Box<dyn Error>> {
    let empty: Vec<GPR> = Vec::new();
    let mut gprs: Vec<(PathBuf, GPR)> = Vec::new();
    for filepath in &params.filepaths {
        // The given filepath may be ".rd3" or may not have an extension at all
        // Counterintuitively to the user point of view, it's the ".rad" file that should be given
        let rad_filepath = filepath.with_extension("rad");

        // Make sure that it exists
        if !rad_filepath.is_file() {
            if filepath.is_file() {
                return Err(
                    format!("File found but no '.rad' file found: {:?}", rad_filepath).into(),
                );
            }
            return Err(format!("File not found: {:?}", rad_filepath).into());
        };
        // Load the GPR metadata from the rad file
        let gpr_meta = io::load_rad(&rad_filepath, params.medium_velocity)?;

        // Load the GPR location data
        // If the "--cor" argument was used, load from there. Otherwise, try to find a ".cor" file
        let mut gpr_locations = match &params.cor_path {
            Some(fp) => io::load_cor(fp, &params.crs)?,
            None => match gpr_meta.find_cor(&params.crs) {
                Ok(v) => Ok(v),
                Err(e) => match params.filepaths.len() > 1 {
                    true => {
                        eprintln!("Error in batch mode, continuing anyway: {:?}", e);
                        continue;
                    }
                    false => Err(e),
                },
            }?,
        };

        // If a "--dem" was given, substitute elevations using said DEM
        if let Some(dem_path) = &params.dem_path {
            gpr_locations.get_dem_elevations(dem_path);
        };

        // Construct the output filepath. If one was given, use that.
        // If a path was given and it's a directory, use the file stem + ".nc" of the input
        // filename. If no output path was given, default to the directory of the input.
        let output_filepath = match &params.output_path {
            Some(p) => match p.is_dir() {
                true => p.join(filepath.file_stem().unwrap()).with_extension("nc"),
                false => {
                    if let Some(parent) = p.parent() {
                        if !parent.is_dir() {
                            return Err(format!(
                                "Output directory of path is not a directory: {:?}",
                                p
                            )
                            .into());
                        };
                    };
                    p.clone()
                }
            },
            None => filepath.with_extension("nc"),
        };

        // If the "--info" argument was given, stop here and just show info.
        if params.only_info {
            println!("{}", gpr_meta);
            println!("{}", gpr_locations);
            // If the track should be exported, do so.
            if let Some(potential_track_path) = &params.track_path {
                io::export_locations(
                    &gpr_locations,
                    potential_track_path.into(),
                    &output_filepath,
                    !params.quiet,
                )?;
            };
        } else {
            // At this point, the data should be processed.
            let gpr = match GPR::from_meta_and_loc(gpr_locations, gpr_meta) {
                Ok(g) => g,
                Err(e) => {
                    return Err(format!(
                        "Error loading GPR data from {:?}: {:?}",
                        rad_filepath.with_extension("rd3"),
                        e
                    )
                    .into())
                }
            };

            gprs.push((output_filepath, gpr));
        };
    }

    // Merge GPR profiles if the merge flag was used
    if let Some(merge) = params.merge.map(|m| m.as_secs_f64()) {
        let mut incompatible: Vec<(usize, usize)> = Vec::new();
        for _ in 0..gprs.len() {
            let mut distances: Vec<(usize, usize, f64)> = Vec::new();
            for i in 0..gprs.len() {
                for j in (0..gprs.len()).rev() {
                    if let Some(incomp) = incompatible.get(i) {
                        if (i == incomp.0) & (j == incomp.1) {
                            continue;
                        };
                    };
                    if (j >= gprs.len()) | (i >= gprs.len()) | (i == j) {
                        continue;
                    };
                    let diff = gprs[i].1.location.duration_since(&gprs[j].1.location);

                    distances.push((i, j, diff));
                }
            }
            distances.sort_by(
                |(i0, j0, _), (i1, j1, _)| match i0.partial_cmp(i1).unwrap() {
                    std::cmp::Ordering::Equal => j0.partial_cmp(j1).unwrap(),
                    o => o,
                },
            );

            if let Some(min_i) = distances.iter().map(|d| d.0).min() {
                let mut merged = 0_usize;
                for (_, j, distance) in distances.iter().filter(|d| d.0 == min_i) {
                    if distance > &merge {
                        continue;
                    };
                    let (output_fp, gpr) = gprs.remove(j - merged);
                    match gprs[min_i].1.merge(&gpr) {
                        Ok(_) => (),
                        Err(e) => {
                            eprintln!(
                                "Could not merge {:?} -> {:?}: {}",
                                output_fp, &gprs[min_i].0, e
                            );
                            gprs.insert(j - merged, (output_fp, gpr));
                            incompatible.push((min_i, j - merged));
                            continue;
                        }
                    };
                    println!("Merged {:?} -> {:?}", output_fp, &gprs[min_i].0);
                    merged += 1;
                }
            } else {
                continue;
            };
        }
    };

    for (output_filepath, mut gpr) in gprs {
        // Record the starting time to show "t+XX" times
        let start_time = SystemTime::now();
        if !params.quiet {
            println!("Processing {:?}", gpr.metadata.rd3_filepath);
        };

        // Run each step sequentially
        for (i, step) in params.steps.iter().enumerate() {
            if !params.quiet {
                println!(
                    "{}/{}, t+{:.2} s, Running step {}. ",
                    i + 1,
                    params.steps.len(),
                    SystemTime::now()
                        .duration_since(start_time)
                        .unwrap()
                        .as_secs_f32(),
                    step,
                );
            };

            // Stop if any error occurs
            match gpr.process(step) {
                Ok(_) => 0,
                Err(e) => return Err(format!("Error on step {}: {:?}", step, e).into()),
            };
        }

        // Unless the "--no-export" flag was given, export the ".nc" result
        if !params.no_export {
            if !params.quiet {
                println!("Exporting to {:?}", output_filepath);
            };
            match gpr.export(&output_filepath) {
                Ok(_) => (),
                Err(e) => return Err(format!("Error exporting data: {:?}", e).into()),
            }
        };

        // If "--render" was given, render an image of the output
        // The flag may or may not have a filepath (it can either be "-r" or "-r img.jpg")
        if let Some(potential_fp) = &params.render_path {
            // Find out the output filepath. If one was given, use that. If none was given, use
            // the output filepath with a ".jpg" extension. If a directory was given, use the
            // file stem of the output filename and a ".jpg" extension
            let render_filepath = match potential_fp {
                Some(fp) => match fp.is_dir() {
                    true => fp
                        .join(output_filepath.file_stem().unwrap())
                        .with_extension("jpg"),
                    false => fp.clone(),
                },
                None => output_filepath.with_extension("jpg"),
            };
            if !params.quiet {
                println!("Rendering image to {:?}", render_filepath);
            };
            gpr.render(&render_filepath).unwrap();
        };

        // If "--track" was given, export the track file.
        if let Some(potential_track_path) = &params.track_path {
            io::export_locations(
                &gpr.location,
                potential_track_path.into(),
                &output_filepath,
                !params.quiet,
            )?;
        };
    }

    Ok(empty)
}

pub fn all_available_steps() -> Vec<[&'static str; 2]> {
    vec![
        ["subset", "Subset the data in x (traces) and/or y (samples). Examples: Clip to the first 500 samples: subset(0 -1 0 500). Clip to the first 300 traces, subset(0 300)"],
        ["remove_traces", "Manually remove trace indices, for example in case they are visually deemed bad. Example: Remove the first two traces: remove_traces(0 1)"],
        ["remove_empty_traces", "Remove all traces that appear empty. Recommended to be run as the first filter if required!. The strength threshold (mean absolute trace value) can be tweaked. Example: 'remove_empty_traces(2)'. Default: 1."],
        ["zero_corr_max_peak", "Shift the location of the zero return time by finding the maximum row value. The peak is found for each trace individually."],
        ["zero_corr", "Shift the location of the zero return time by finding the first row where data appear. The correction can be tweaked to allow more or less data, e.g. 'zero_corr(0.9)'. Default: 1.0"],
        ["bandpass", "Apply a bandpass Butterworth filter to each trace individually. The given frequencies are normalized (0: 0Hz, 1: Nyquist). Default: bandpass(0.1 0.9)"],
        ["equidistant_traces", "Make all traces equidistant by averaging them in a fixed horizontal grid. The step size is determined from the median moving velocity. Other step sizes in m can be given, e.g. 'equidistant_traces(2.)' for 2 m. Default: auto"],
        ["normalize_horizontal_magnitudes", "Normalize the magnitudes of the traces in the horizontal axis. This removes or reduces horizontal banding. The uppermost samples of the trace can be excluded, either by sample number (integer; e.g. 'normalize_horizontal_magnitudes(300)') or by a fraction of the trace (float; e.g. 'normalize_horizontal_magnitudes(0.3)'). Default: 0.3"],
        ["dewow", "Subtract the horizontal moving average magnitude for each trace. This reduces artefacts that are consistent among every trace. The averaging window can be set, e.g. 'dewow(10)'. Default: 5"],
        ["auto_gain", "Automatically determine the best gain factor and apply it. The data are binned vertically and the mean absolute deviation of the values is used as a proxy for signal attenuation. The median attenuation in decibel volts is given to the gain filter. The amounts of bins can be given, e.g. 'auto_gain(100). Default: 100"],
        ["gain", "Multiply the magnitude as a function of depth. This is most often used to correct for signal attenuation with time/distance. Gain is applied by: '10 ^(gain * sqrt(sample_index))' where gain is the given gain factor and sample_index is the zero-based index of the sample from the top. Examples: gain(0.1). No default value."],
        ["kirchhoff_migration2d", "Migrate sample magnitudes in the horizontal and vertical distance dimension to correct hyperbolae in the data. The correction is needed because the GPR does not observe only what is directly below it, but rather in a cone that is determined by the dominant antenna frequency. Thus, without migration, each trace is the sum of a cone beneath it. Topographic Kirchhoff migration (in 2D) corrects for this in two dimensions."],
        ["abslog", "Run a log10 operation on the absolute values (log10(abs(data))), converting it to a logarithmic scale. This is useful for visualisation. Before conversion, the data are added with the 1st percentile (absolute) value in the dataset to avoid log10(0) == inf."],
        ["siglog", "Run a log10 operation on absolute values and then account for the sign. Values smaller than the set minimum magnitude are truncated to zero. E.g. with an exponent offset of 0: 1000 -> 3, -1000 -> -3, 0.001 -> 0. The argument specifies the exponent offset to apply to allow for values smaller than +-1 (e.g. 10e-1). Default: -1"],
        ["unphase", "Combine the positive and negative phases of the signal into one positive magntiude. The assumption is made that the positive magnitude of the signal comes first, followed by an offset negative component. The distance between the positive and negative peaks are found, and then the negative part is shifted accordingly."],
        ["correct_topography", "Make a copy of the data and topographically correct it. In the output, the data will be called \"topo_data\". Note that the copying means any step run after this will not be reflected in \"topo_data\". This is thus recommended to run last."],
        ["correct_antenna_separation", "Correct for the separation between the antenna transmitter and receiver. The consequence is that depths are slightly over-exaggerated at low return-times before correction. This step averages samples so that each sample represents a consistent depth interval."],
    ]
}

pub fn default_processing_profile() -> Vec<String> {
    vec![
        "remove_empty_traces".to_string(),
        format!("zero_corr_max_peak"),
        "equidistant_traces".to_string(),
        "correct_antenna_separation".to_string(),
        format!(
            "normalize_horizontal_magnitudes({})",
            DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF
        ),
        format!("dewow({})", DEFAULT_DEWOW_WINDOW),
        format!("auto_gain({})", DEFAULT_AUTOGAIN_N_BINS),
    ]
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ndarray::{Array1, Axis, Slice};

    use super::{CorPoint, GPRLocation, LocationCorrection};

    fn make_cor_points(n_points: usize, spacing: f64) -> Vec<CorPoint> {
        let eastings = Array1::range(0_f64, n_points as f64 * spacing, spacing);

        let northings = Array1::<f64>::zeros(n_points);

        let altitudes = eastings.clone();
        let seconds = eastings.clone();

        let trace_n = eastings.mapv(|v| v as u32);

        (0..n_points)
            .map(|i| CorPoint {
                trace_n: trace_n[i],
                time_seconds: seconds[i],
                easting: eastings[i],
                northing: northings[i],
                altitude: altitudes[i],
            })
            .collect::<Vec<CorPoint>>()
    }

    fn make_gpr_location(
        n_points: usize,
        spacing: Option<f64>,
        crs: Option<String>,
        correction: Option<LocationCorrection>,
    ) -> GPRLocation {
        GPRLocation {
            cor_points: make_cor_points(n_points, spacing.unwrap_or(1.)),
            correction: correction.unwrap_or(LocationCorrection::None),
            crs: crs.unwrap_or("EPSG:32633".to_string()),
        }
    }

    fn make_dummy_gpr(n_traces: usize, n_samples: usize, spacing: Option<f64>) -> super::GPR {
        let gpr_location = make_gpr_location(n_traces, spacing, None, None);
        let metadata = super::GPRMeta {
            samples: n_samples as u32,
            frequency: 5000.,
            frequency_steps: 0,
            time_interval: 0.2,
            antenna: "500MHz".to_string(),
            antenna_mhz: 500.,
            antenna_separation: 1.,
            time_window: 2000.,
            last_trace: n_traces as u32,
            rd3_filepath: std::path::PathBuf::new(),
            medium_velocity: 0.167,
        };

        let mut data = ndarray::Array2::<f32>::zeros((n_samples, n_traces));
        let new_row = ndarray::Array1::<f32>::range(0., n_traces as f32, 1.);
        for mut row in data.rows_mut() {
            row.assign(&new_row);
        }

        super::GPR {
            location: gpr_location,
            metadata,
            data,
            topo_data: None,
            zero_point_ns: 0.,
            horizontal_signal_distance: 1.,
            log: Vec::new(),
        }
    }

    #[test]
    fn test_gpr_location() {
        let mut gpr_location = make_gpr_location(10, None, None, None);
        // The first time+coordinate should be all zero
        assert_eq!(gpr_location.time_and_coord_at_trace(0), (0., 0., 0., 0.));

        // The second time+coordinate should be all one except for the northing
        assert_eq!(gpr_location.time_and_coord_at_trace(1), (1., 1., 0., 1.));
        // If the second is removed, it should still be linearly interpolated correctly
        gpr_location.cor_points.remove(1);
        // Check that interpolation works expectedly
        assert_eq!(gpr_location.time_and_coord_at_trace(1), (1., 1., 0., 1.));

        assert_eq!(
            gpr_location.time_and_coord_at_trace(100),
            gpr_location.time_and_coord_at_trace(gpr_location.cor_points.last().unwrap().trace_n)
        );

        gpr_location = gpr_location.range_fill(0, 10);

        // Check that the velocities are consistent along the track
        // The first and second velocities will still be a bit weird
        let velocities = gpr_location.velocities();
        assert_eq!(
            Some(velocities[2]),
            velocities
                .slice_axis(Axis(0), Slice::new(1, None, 1))
                .mean()
        );

        let distances = gpr_location.distances();
        assert_eq!(distances[0], 0.);
        assert_eq!(distances[9], 9.);
    }

    #[test]
    fn test_gpr_location_duration_since() {
        let gpr_location0 = make_gpr_location(10, Some(1.), None, None);

        let mut gpr_location1 = gpr_location0.clone();

        for point in gpr_location1.cor_points.iter_mut() {
            point.time_seconds += 10.;
        }

        // gpr_location0 stops at 9s. gpr_location1 starts at 10s
        // So the difference should be +1s for gpr_location0-gpr_location1
        // And -1s for gpr_location1-gpr_location0

        assert_eq!(gpr_location0.duration_since(&gpr_location1), 1.);
        assert_eq!(gpr_location1.duration_since(&gpr_location0), -1.);

        for point in gpr_location1.cor_points.iter_mut() {
            point.time_seconds -= 30.;
        }
        assert_eq!(gpr_location0.duration_since(&gpr_location1), -11.);
        assert_eq!(gpr_location1.duration_since(&gpr_location0), 11.);
    }

    fn make_test_metadata(width: Option<usize>, height: Option<usize>) -> super::GPRMeta {
        super::GPRMeta {
            samples: height.unwrap_or(1024) as u32,
            frequency: 8000.,
            frequency_steps: 1,
            time_interval: 1000.,
            antenna: "800MHz".into(),
            antenna_mhz: 800.,
            antenna_separation: 2.,
            time_window: 500.,
            last_trace: width.unwrap_or(2048) as u32,
            rd3_filepath: PathBuf::new(),
            medium_velocity: 0.168,
        }
    }

    fn make_test_gpr(width: Option<usize>, height: Option<usize>) -> super::GPR {
        let width = width.unwrap_or(2024);
        let height = height.unwrap_or(1024);

        let gpr_location = make_gpr_location(width, Some(1.), None, None);

        let meta = make_test_metadata(Some(width), Some(height));

        let antenna_separation = meta.antenna_separation;

        let mut data = ndarray::Array2::<f32>::zeros((height, width));

        for mut col in data.columns_mut() {
            col.assign(&Array1::<f32>::linspace(0., (height - 1) as f32, height));
        }

        super::GPR {
            data,
            topo_data: None,
            location: gpr_location,
            metadata: meta,
            log: Vec::new(),
            horizontal_signal_distance: antenna_separation,
            zero_point_ns: 0.,
        }
    }

    #[test]
    fn test_make_test_gpr() {
        let gpr = make_test_gpr(None, None);

        assert_eq!(gpr.data[[0, 0]], 0.);
        assert_eq!(
            gpr.data[[gpr.data.shape()[0] - 1, 0]],
            (gpr.data.shape()[0] - 1) as f32
        );
    }

    #[test]
    fn test_correct_antenna_separation() {
        let mut gpr = make_test_gpr(Some(10), Some(1024));

        gpr.horizontal_signal_distance = 30.;

        assert_eq!(gpr.data[[10, 0]], 10.);
        assert_eq!(gpr.log.len(), 0);
        gpr.correct_antenna_separation();
        assert!(gpr
            .log
            .last()
            .unwrap()
            .contains("correct_antenna_separation"));

        assert_ne!(gpr.height(), 1024);

        assert!(gpr.data[[10, 0]] > 10.);
    }

    #[test]
    fn test_equidistant_traces() {
        let width = 128;
        let mut gpr = make_test_gpr(Some(width), Some(256));

        let first = gpr.location.cor_points[0].clone();

        let n_stationary = 10;

        for (i, point) in gpr.location.cor_points.iter_mut().enumerate() {
            if i == n_stationary {
                break;
            }

            point.easting = first.easting;
            point.northing = first.northing;
            point.altitude = first.altitude;
        }
        assert_eq!(gpr.width(), width);
        gpr.make_equidistant(None);
        // Now, the N stationary points should be coerced into one
        assert_eq!(gpr.width(), width - (n_stationary - 1));
    }

    #[test]
    fn test_remove_traces() {
        let mut gpr = make_dummy_gpr(20, 30, Some(1.));

        gpr.topo_data = Some(gpr.data.clone());

        // Make sure that the dummy GPR is indeed 30x20 and it varies linearly from 0-19
        assert_eq!(gpr.width(), 20);
        assert_eq!(gpr.height(), 30);
        assert_eq!(gpr.location.cor_points.len(), 20);
        assert_eq!(gpr.data[[0, 0]], 0.);
        assert_eq!(gpr.data[[0, 19]], 19.);

        if let Some(topo_data) = &gpr.topo_data {
            assert_eq!(topo_data[[0, 19]], 19.);
        };
        // Remove indices 5 and 6, and "accidentally" duplicate one trace
        gpr.remove_traces(&[5, 5, 6], true).unwrap();

        // Validate that the dummy GPR is now shorter, and that the values are shifted correctly.
        assert_eq!(gpr.width(), 18);
        assert_eq!(gpr.height(), 30);
        assert_eq!(gpr.location.cor_points.len(), 18);
        assert_eq!(gpr.data[[0, 0]], 0.);
        assert_eq!(gpr.data[[0, 17]], 19.);
        assert_eq!(gpr.data[[0, 5]], 7.);

        if let Some(topo_data) = &gpr.topo_data {
            assert_eq!(topo_data[[0, 17]], 19.);
            assert_eq!(topo_data[[0, 5]], 7.);
        };

        // Check that the out of bounds error works expectedly
        assert_eq!(
            gpr.remove_traces(&[18], true),
            Err("Trace index 18 is out of bounds (number of traces: 18)".to_string())
        );
    }

    #[test]
    fn test_remove_empty_traces() {
        let mut gpr = make_dummy_gpr(20, 30, Some(1.));

        let empty_traces: Vec<usize> = vec![0, 10, 19];

        for (i, mut col) in gpr.data.columns_mut().into_iter().enumerate() {
            if empty_traces.iter().all(|i2| i2 != &i) {
                col.mapv_inplace(|_| 2.);
            } else {
                col.mapv_inplace(|_| 0.);
            };
        }
        gpr.remove_empty_traces(1.).unwrap();

        assert_eq!(gpr.width(), 17);
    }
}
