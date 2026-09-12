use std::error::Error;
/// Functions to process GPR data
use std::path::{Path, PathBuf};

use crate::formats::{self, FormatKind, ResolvedInput};
use serde::Serialize;
use std::time::SystemTime;

use ndarray::{Array1, Array2, Axis, Slice};
use rayon::prelude::*;

use crate::{dem, filters, io, tools, user_metadata};

const DEFAULT_ZERO_CORR_THRESHOLD_MULTIPLIER: f32 = 1.0;
const DEFAULT_EMPTY_TRACE_STRENGTH: f32 = 1.0;
const DEFAULT_DEWOW_WINDOW: u32 = 5;
const DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF: f32 = 0.3;
const DEFAULT_AUTOGAIN_N_BINS: usize = 100;
const DEFAULT_BANDPASS_LOW_CUTOFF: f32 = 0.1;
const DEFAULT_BANDPASS_HIGH_CUTOFF: f32 = 0.9;
const DEFAULT_BANDPASS_Q: f32 = 0.707;
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
    /// The path to the data file
    pub data_filepath: PathBuf,
    /// The velocity of the medium (m / ns)
    pub medium_velocity: f32,
}

impl GPRMeta {
    /// Find a ".cor" file based on the location of the ".rd3" file
    ///
    /// # Arguments
    /// - `projected_crs`: The CRS to project coordinates into
    pub fn find_cor(&self, projected_crs: Option<&String>) -> Result<GPRLocation, Box<dyn Error>> {
        let cor = crate::formats::find_neighbor_case_insensitive(&self.data_filepath, "cor")
            .unwrap_or_else(|| self.data_filepath.with_extension("cor"));
        io::load_cor(&cor, projected_crs)
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
            self.data_filepath,
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

        // Euclidean norm of the (easting, northing, altitude) step. The
        // `sqrt` is what makes this a length rather than a squared length;
        // without it every velocity is scaled by the step length itself.
        let d = offsets
            .slice_axis(Axis(1), Slice::new(1, None, 1))
            .mapv(|f| f.powi(2))
            .sum_axis(Axis(1))
            .mapv(f64::sqrt);

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

        // Per-step horizontal length, then a cumulative sum of those
        // lengths. The `sqrt` must be applied per step, before the
        // accumulation: cumulatively summing the squared steps yields a
        // quantity that is not a distance at all, and whose error grows
        // with the trace spacing rather than being a constant factor.
        // Altitude is deliberately excluded (see the commented-out term
        // above) -- this is distance along the ground track, not 3D path
        // length, which is what `velocities` measures.
        let mut dist = offsets
            .slice_axis(Axis(1), Slice::new(1, None, 1))
            .mapv(|f| f.powi(2))
            .sum_axis(Axis(1))
            .mapv(f64::sqrt);
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

    pub fn get_dem_elevations(&mut self, dem_path: &Path) -> Result<(), String> {
        let coords = self
            .cor_points
            .iter()
            .map(|cor| crate::coords::Coord {
                x: cor.easting,
                y: cor.northing,
            })
            .collect::<Vec<crate::coords::Coord>>();

        let coords_wgs84 =
            crate::coords::to_wgs84(&coords, &crate::coords::Crs::from_user_input(&self.crs)?)?;
        let elev = dem::sample_dem(dem_path, &coords_wgs84)?;

        for (i, point) in self.cor_points.iter_mut().enumerate() {
            if let Some(e) = elev.get(i) {
                point.altitude = *e as f64;
            }
        }

        self.correction = LocationCorrection::Dem(dem_path.to_path_buf());
        Ok(())
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
        let dists = self.distances();
        dists.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn duration_since(&self, other: &GPRLocation) -> f64 {
        let self_times = Array1::from_iter(self.cor_points.iter().map(|p| p.time_seconds));
        let other_times = Array1::from_iter(other.cor_points.iter().map(|p| p.time_seconds));

        let self_times_iter = self_times.iter().cloned().collect::<Vec<f64>>();
        let other_times_iter = other_times.iter().cloned().collect::<Vec<f64>>();

        let self_min = self_times_iter
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let self_max = self_times_iter
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let other_min = other_times_iter
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let other_max = other_times_iter
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

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
            altitudes.iter().cloned().fold(f64::INFINITY, f64::min),
            altitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
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
    /// The steps that the user provided
    pub steps: Vec<String>,
    /// The horizontal component of the signal distance (m). Defaults to the antenna separation if no correction has been made.
    horizontal_signal_distance: f32,
    /// The calculated zero-point (ns). It represents the delay between the transmitter and the receiver.
    zero_point_ns: f32,
    /// User-supplied metadata that should be carried through processing/export.
    pub user_metadata: user_metadata::UserMetadata,
    /// Persistent radargram identity metadata (#116). Resolved once, at the
    /// end of [`build_processed_gpr`], from CLI arguments and the output
    /// stem -- not populated on GPR instances still mid-processing.
    pub identity: RidalIdentity,
}

/// Persistent identity metadata for one processed output. See
/// [`crate::identity`] for the concepts and their precedence rules.
#[derive(Debug, Clone, Default)]
pub struct RidalIdentity {
    pub radargram_id: Option<crate::identity::RadargramId>,
    pub display_name: Option<crate::identity::DisplayName>,
    pub group_name: Option<crate::identity::GroupName>,
    pub group_id: Option<crate::identity::GroupId>,
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
        } else if step_name.contains("shift_coordinates") {
            let along_track = tools::parse_option::<f64>(step_name, 0)?
                .ok_or("Must provide an offset to shift_coordinates".to_string())?;
            let altitude = tools::parse_option::<f64>(step_name, 1)?.unwrap_or(0.);
            let cross_track = tools::parse_option::<f64>(step_name, 2)?.unwrap_or(0.);

            self.shift_coordinates(along_track, altitude, cross_track)?;
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
            *self = self.subset(min_trace, max_trace, min_sample, max_sample)?;
        } else if step_name.contains("average_traces") {
            let window: usize = tools::parse_option(step_name, 0)?
                .ok_or("Must provide an averaging window to average_traces".to_string())?;

            self.average_traces(window)?;
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
        } else if step_name.contains("bandpass_mhz") {
            let error_msg =
                "Must provide lower and upper cutoff frequencies (e.g. bandpass_mhz(50 150)";
            let low_cutoff = tools::parse_option(step_name, 0)?.ok_or(error_msg)?;
            let high_cutoff = tools::parse_option(step_name, 1)?.ok_or(error_msg)?;
            let q: f32 = tools::parse_option(step_name, 2)
                .ok()
                .flatten()
                .unwrap_or(DEFAULT_BANDPASS_Q);
            self.bandpass(low_cutoff, high_cutoff, q, false)?;
        } else if step_name.contains("bandpass") {
            let low_cutoff =
                tools::parse_option(step_name, 0)?.unwrap_or(DEFAULT_BANDPASS_LOW_CUTOFF);
            let high_cutoff =
                tools::parse_option(step_name, 1)?.unwrap_or(DEFAULT_BANDPASS_HIGH_CUTOFF);
            let q: f32 = tools::parse_option(step_name, 2)
                .ok()
                .flatten()
                .unwrap_or(DEFAULT_BANDPASS_Q);
            self.bandpass(low_cutoff, high_cutoff, q, true)?;
        } else {
            return Err(format!("Step name not recognized: {}", step_name).into());
        }

        self.steps.push(step_name.to_string());

        Ok(())
    }

    pub fn bandpass(
        &mut self,
        low_cutoff: f32,
        high_cutoff: f32,
        q: f32,
        normalized: bool,
    ) -> Result<(), String> {
        let start_time = SystemTime::now();

        if q <= 0. {
            return Err(format!("Bandpass q needs to be above 0 (provided: {q})"));
        }
        if normalized {
            if !(0. ..=1.).contains(&low_cutoff) {
                return Err(format!(
                    "Normalized low cutoff needs to be in the range 0-1 (provided: {low_cutoff})"
                ));
            }
            if !(0. ..=1.).contains(&high_cutoff) {
                return Err(format!(
                    "Normalized high cutoff needs to be in the range 0-1 (provided: {high_cutoff})"
                ));
            }
            if low_cutoff >= high_cutoff {
                return Err(format!(
                "Normalized low cutoff ({low_cutoff}) needs to be smaller than the high cutoff ({high_cutoff})."
            ));
            }
        } else if low_cutoff >= high_cutoff {
            return Err(format!(
                "Low cutoff ({low_cutoff}) needs to be smaller than the high cutoff ({high_cutoff})."
            ));
        }

        for mut col in self.data.columns_mut() {
            filters::bandpass::bandpass_hpf_then_lpf(
                &mut col,
                low_cutoff,
                high_cutoff,
                (!normalized).then_some(self.metadata.frequency),
                Some(q),
                true,
            )?;
        }

        if normalized {
            self.log_event(
                "bandpass",
                &format!(
                    "Applied a normalized bandpass Butterworth filter ({:.3}-{:.3}, q={:.3})",
                    low_cutoff, high_cutoff, q,
                ),
                start_time,
            );
        } else {
            self.log_event(
                "bandpass_mhz",
                &format!(
                    "Applied a bandpass Butterworth filter ({:.3}-{:.3} MHz, q={:.3})",
                    low_cutoff, high_cutoff, q
                ),
                start_time,
            );
        }

        Ok(())
    }

    pub fn subset(
        &self,
        min_trace: Option<u32>,
        max_trace: Option<u32>,
        min_sample: Option<u32>,
        max_sample: Option<u32>,
    ) -> Result<GPR, String> {
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

        let checks = [
            (min_trace_, self.width(), "min trace number"),
            (max_trace_, self.width(), "max trace number"),
            (min_sample_, self.height(), "min sample number"),
            (max_sample_, self.height(), "max sample number"),
        ];

        for (val, maxval, desc) in checks {
            if val > maxval as u32 {
                return Err(format!(
                    "Subset failed: {desc} ({val}) out of the dataset bounds ({maxval})"
                ));
            }
        }

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
            steps: self.steps.clone(),
            topo_data: self.topo_data.clone(),
            horizontal_signal_distance: self.horizontal_signal_distance,
            zero_point_ns: self.zero_point_ns,
            user_metadata: self.user_metadata.clone(),
            identity: self.identity.clone(),
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

        Ok(new_gpr)
    }

    pub fn vertical_resolution_ns(&self) -> f32 {
        self.metadata.time_window / self.metadata.samples as f32
    }

    pub fn from_meta_and_loc(
        location: GPRLocation,
        metadata: GPRMeta,
    ) -> Result<GPR, Box<dyn Error>> {
        let data = match metadata.data_filepath.extension().and_then(|s| s.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("rd3") => Ok(io::load_rd3(
                &metadata.data_filepath,
                metadata.samples as usize,
            )?),
            Some(ext) if ext.eq_ignore_ascii_case("dt1") => Ok(io::load_pe_dt1(
                &metadata.data_filepath,
                metadata.samples as usize,
                metadata.last_trace as usize,
            )?),
            Some(ext) if ext.eq_ignore_ascii_case("dzt") => Ok(io::load_dzt(
                &metadata.data_filepath,
                metadata.samples as usize,
            )?),
            _ => Err(format!("Unknown filetype: {:?}", metadata.data_filepath)),
        }?;

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
            steps: Vec::new(),
            topo_data: None,
            horizontal_signal_distance,
            zero_point_ns: 0.,
            user_metadata: user_metadata::UserMetadata::new(),
            identity: RidalIdentity::default(),
        })
    }

    /// Render this radargram to an image with a render profile.
    ///
    /// Goes through the same pipeline as `ridal render` and the web GUI --
    /// same profiles, same resampling, same amplitude sampling -- from the
    /// array already in memory, since `--no-export --render` is allowed and
    /// then there is no file to read back.
    ///
    /// # Always `data`, never `data_topocorr`
    ///
    /// The renderer this replaced drew the topographically corrected array
    /// whenever topographic correction had run. That made one output file
    /// have two pictures: this one, and the different one the browser drew
    /// from the exported `data` variable.
    ///
    /// One array wins instead, and it is `data`, so every way of drawing a
    /// radargram agrees. Topographic correction is a secondary product
    /// here; rendering it is a render *profile* and an explicit flag on
    /// `ridal render`, not a thing `--render` should silently switch to
    /// because an earlier step happened to run. Until that exists, a
    /// corrected section cannot be written to a JPG or PNG -- a known and
    /// accepted gap rather than an oversight.
    ///
    /// This replaced a separate implementation (`io::render_jpg`) that
    /// stretched between the 1st and 99th percentile of every tenth sample
    /// and clamped the floor to zero when `unphase` had run. Two renderers
    /// that disagree about how to draw the same data is a bad property for
    /// an instrument display, and the `unphase` case was a fixed rule
    /// applied regardless of what the amplitudes actually looked like.
    pub fn render(
        &self,
        filepath: &Path,
        profile: &crate::render::profile::RenderProfile,
        width: Option<usize>,
    ) -> Result<(), Box<dyn Error>> {
        let source = crate::source::ArraySource::new(self.data.view());
        let request = crate::render::oneshot::RenderRequest {
            profile,
            width,
            quality: None,
        };
        crate::render::oneshot::render_to_file(&source, filepath, &request)?;
        Ok(())
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
            positive_peaks[i] = col
                .into_iter()
                .enumerate()
                .max_by_key(|(_, v)| v.abs().abs() as u64)
                .map(|(idx, _)| idx as isize)
                .unwrap();

            i += 1;
        }

        let mut new_data = Array2::from_elem(
            (
                self.height()
                    - positive_peaks.iter().cloned().fold(isize::MAX, isize::min) as usize,
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
                positive_peaks.iter().cloned().fold(isize::MAX, isize::min),
                positive_peaks.iter().cloned().fold(isize::MIN, isize::max)
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
        let max_depth = depths.iter().cloned().fold(0.0f32, f32::max);

        if max_depth == 0.0 {
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
        altitudes -= altitudes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        altitudes *= -1.;

        let max_depth = tools::return_time_to_depth(
            self.metadata.time_window,
            self.metadata.medium_velocity,
            self.metadata.antenna_separation,
        );

        let sample_per_meter = self.height() as f32 / max_depth;

        let start_indices = altitudes.mapv(|altitude| (altitude * sample_per_meter) as isize);

        let mut topo_data = Array2::<f32>::zeros((
            ((max_depth + altitudes.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                * sample_per_meter) as usize,
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
            positive_peaks[i] = col
                .into_iter()
                .enumerate()
                .max_by_key(|(_, v)| v.abs().abs() as u64)
                .map(|(idx, _)| idx as isize)
                .unwrap();
            negative_peaks[i] = col
                .into_iter()
                .enumerate()
                .min_by_key(|(_, v)| v.abs().abs() as u64)
                .map(|(idx, _)| idx as isize)
                .unwrap();
        }

        let mean_peak_spacing = (negative_peaks - positive_peaks).mean().unwrap().abs();
        let mut new_data = self.data.mapv(|v| v.max(0.));

        self.data
            .slice_axis(Axis(0), Slice::new(mean_peak_spacing, None, 1))
            .mapv(|v| -v.min(0.))
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

            let new_att = slice.mapv(|a| a.abs().log10()).mean().unwrap() * 20.;
            if let Some(old) = old_att {
                attenuations.push(old - new_att);
            }
            old_att = Some(new_att);
        }

        let median_att = tools::quantiles(&attenuations, &[0.5], None)[0];

        let slope = (median_att.abs()
            / (self.vertical_resolution_ns() * (self.height() as f32) / (n_bins as f32)))
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

        let ns_per_trace = self.vertical_resolution_ns();
        for i in 0..self.height() as isize {
            let mut view = self
                .data
                .slice_axis_mut(Axis(0), Slice::new(i, Some(i + 1), 1_isize));

            view *= 10_f32.powf(factor * (i as f32) * ns_per_trace / 20.);
        }
        self.log_event(
            "gain",
            &format!("Applied gain of {factor} dB / ns (TWT)",),
            start_time,
        );
    }

    pub fn make_equidistant(&mut self, step: Option<f32>) {
        let start_time = SystemTime::now();
        let distances = self.location.distances().mapv(|v| v as f32);
        let max_distance = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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

    fn shift_coordinates(
        &mut self,
        along_track: f64,
        altitude: f64,
        cross_track: f64,
    ) -> Result<(), String> {
        let start_time = SystemTime::now();

        if self.location.cor_points.len() < 2 {
            return Err("shift_coordinates needs at least two valid coordinates".to_string());
        }

        self.location.cor_points = crate::filters::coordinates::shift_coordinates(
            &self.location.cor_points,
            along_track,
            altitude,
            cross_track,
        );

        self.log_event("shift_coordinates", &format!("Shifted coordinates {along_track} m along the track (+ is forward), {altitude} in altitude (+ is up) and {cross_track} m across the track (+ is right)."), start_time);

        Ok(())
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
        z_coords -= z_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

    pub fn average_traces(&mut self, window: usize) -> Result<(), String> {
        let start_time = SystemTime::now();

        let initial_width = self.width();
        let averaged_data = filters::average_traces(&self.data, window)?;

        if let Some(topo_data) = &self.topo_data {
            self.topo_data = Some(filters::average_traces(topo_data, window)?);
        }

        self.location.cor_points =
            filters::window_subset_vec(self.location.cor_points.clone(), window);
        self.metadata.time_interval *= window as f32;

        self.update_data(averaged_data);

        self.log_event(
            "average_traces",
            &format!("Averaged traces in a window={window}. Reduced trace number from {initial_width} to {}", self.width()),
            start_time,
        );

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

    pub fn export(&self, nc_filepath: &Path) -> Result<(), String> {
        let ds = self.export_dataset()?;
        io::export_netcdf(&ds, nc_filepath)
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
            user_metadata::merge_prefer_first(&mut self.user_metadata, &other.user_metadata);

            self.log_event(
                "merge",
                &format!("Merged {:?}", other.metadata.data_filepath),
                start_time,
            );

            Ok(())
        }
    }
}
#[derive(Debug, Clone)]
pub struct RunParams {
    pub filepaths: Vec<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub dem_path: Option<PathBuf>,
    pub cor_path: Option<PathBuf>,
    pub medium_velocity: f32,
    pub crs: Option<String>,
    pub quiet: bool,
    pub track_path: Option<Option<PathBuf>>,
    pub steps: Vec<String>,
    pub no_export: bool,
    pub render_path: Option<Option<PathBuf>>,
    /// Render profile for `render_path`: a built-in name or a TOML file.
    /// `None` means the built-in `default`.
    pub render_profile: Option<String>,
    /// Output width for `render_path`. `None` means one pixel per trace.
    pub render_width: Option<usize>,
    pub override_antenna_mhz: Option<f32>,
    pub override_antenna_separation: Option<f32>,
    pub user_metadata: user_metadata::UserMetadata,
    pub radargram_id: Option<String>,
    pub display_name: Option<String>,
    pub group: Option<String>,
    pub group_id: Option<String>,
}
#[derive(Debug, Clone)]
pub struct BatchRunParams {
    pub filepaths: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub dem_path: Option<PathBuf>,
    pub cor_path: Option<PathBuf>,
    pub medium_velocity: f32,
    pub crs: Option<String>,
    pub quiet: bool,
    pub track_dir: Option<PathBuf>,
    pub steps: Vec<String>,
    pub no_export: bool,
    pub render_dir: Option<PathBuf>,
    /// Render profile for `render_dir`, as in [`RunParams::render_profile`].
    pub render_profile: Option<String>,
    /// Output width for `render_dir`, as in [`RunParams::render_width`].
    pub render_width: Option<usize>,
    pub merge: Option<String>,
    pub override_antenna_mhz: Option<f32>,
    pub override_antenna_separation: Option<f32>,
    pub user_metadata: user_metadata::UserMetadata,
    // Batch mode derives a radargram ID and (absent) display name per output
    // from each output's own stem -- an explicit single ID or display name
    // would collide across outputs. A group applies uniformly to the whole
    // batch, since batch runs typically process one survey or campaign.
    pub group: Option<String>,
    pub group_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchProcessResult {
    pub output_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct InfoParams {
    pub filepaths: Vec<PathBuf>,
    pub dem_path: Option<PathBuf>,
    pub cor_path: Option<PathBuf>,
    pub medium_velocity: f32,
    pub crs: Option<String>,
    pub override_antenna_mhz: Option<f32>,
    pub override_antenna_separation: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamedFormat {
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetadataSummary {
    pub samples: u32,
    pub sampling_frequency_mhz: f32,
    pub frequency_steps: u32,
    pub time_interval_s: f32,
    pub antenna_name: String,
    pub antenna_mhz: f32,
    pub antenna_separation_m: f32,
    pub time_window_ns: f32,
    pub last_trace: u32,
    pub medium_velocity_m_per_ns: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CentroidSummary {
    pub easting: f64,
    pub northing: f64,
    pub altitude: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrectionSummary {
    pub kind: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocationSummary {
    pub n_points: usize,
    pub crs: String,
    pub start_time: String,
    pub stop_time: String,
    pub duration_s: f64,
    pub track_length_m: f64,
    pub altitude_min_m: f64,
    pub altitude_max_m: f64,
    pub centroid: CentroidSummary,
    pub correction: CorrectionSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct RelatedFiles {
    pub header: String,
    pub data: String,
    pub coordinates: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InfoRecord {
    pub input: String,
    pub format: NamedFormat,
    pub metadata: MetadataSummary,
    pub location: LocationSummary,
    pub related_files: RelatedFiles,
}

#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub output_path: PathBuf,
}

fn has_glob_chars(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.contains('*') || text.contains('?') || text.contains('[')
}

pub fn expand_input_paths(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut expanded = Vec::<PathBuf>::new();
    for input in inputs {
        if has_glob_chars(input) {
            let mut matches = glob::glob(&input.to_string_lossy())
                .map_err(|e| format!("Invalid glob pattern {:?}: {e}", input))?
                .filter_map(Result::ok)
                .collect::<Vec<PathBuf>>();
            matches.sort();
            if matches.is_empty() {
                return Err(format!("Glob pattern matched no files: {:?}", input));
            }
            expanded.extend(matches);
        } else {
            expanded.push(input.clone());
        }
    }
    if expanded.is_empty() {
        return Err("No input files were provided.".to_string());
    }
    Ok(expanded)
}

fn derive_output_path(
    output_path: Option<&PathBuf>,
    first_resolved: &ResolvedInput,
) -> Result<PathBuf, String> {
    let default_output = first_resolved.header.with_extension("nc");
    match output_path {
        Some(path) => {
            if path.is_dir() {
                Ok(path
                    .join(first_resolved.header.file_stem().unwrap())
                    .with_extension("nc"))
            } else {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() && !parent.is_dir() {
                        return Err(format!(
                            "Output directory of path is not a directory: {:?}",
                            path
                        ));
                    }
                }
                Ok(path.clone())
            }
        }
        None => Ok(default_output),
    }
}

fn load_meta_and_location(
    resolved: &ResolvedInput,
    medium_velocity: f32,
    crs: Option<&String>,
    cor_path: Option<&PathBuf>,
    dem_path: Option<&PathBuf>,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<(GPRMeta, GPRLocation), Box<dyn Error>> {
    let (gpr_meta, mut gpr_location) = match resolved.kind {
        FormatKind::Ramac => {
            if !resolved.header.is_file() {
                return Err(format!("File not found: {:?}", resolved.header).into());
            }
            let meta = io::load_rad(
                &resolved.header,
                medium_velocity,
                override_antenna_mhz,
                override_antenna_separation,
            )?;
            let cor_source = match cor_path {
                Some(path) => path.clone(),
                None => resolved.coordinates.clone(),
            };
            let location = io::load_cor(&cor_source, crs)?;
            (meta, location)
        }
        FormatKind::PulseEkko => {
            if cor_path.is_some() {
                return Err("The --cor option is only supported for RAMAC inputs.".into());
            }
            if !resolved.header.is_file() {
                return Err(format!("File not found: {:?}", resolved.header).into());
            }
            let meta = io::load_pe_hd(
                &resolved.header,
                medium_velocity,
                override_antenna_mhz,
                override_antenna_separation,
            )?;
            let location = io::load_pe_gp2(&resolved.coordinates, crs)?;
            (meta, location)
        }
        FormatKind::Gssi => {
            if cor_path.is_some() {
                return Err("The --cor option is only supported for RAMAC inputs.".into());
            }
            if !resolved.header.is_file() {
                return Err(format!("File not found: {:?}", resolved.header).into());
            }
            let meta = io::load_gssi_dzt(
                &resolved.header,
                medium_velocity,
                override_antenna_mhz,
                override_antenna_separation,
            )?;
            let location = io::load_gssi_dzg(&resolved.coordinates, crs)?;
            (meta, location)
        }
    };

    if let Some(dem_path) = dem_path {
        gpr_location.get_dem_elevations(dem_path)?;
    }

    Ok((gpr_meta, gpr_location))
}

fn metadata_summary(meta: &GPRMeta) -> MetadataSummary {
    MetadataSummary {
        samples: meta.samples,
        sampling_frequency_mhz: meta.frequency,
        frequency_steps: meta.frequency_steps,
        time_interval_s: meta.time_interval,
        antenna_name: meta.antenna.clone(),
        antenna_mhz: meta.antenna_mhz,
        antenna_separation_m: meta.antenna_separation,
        time_window_ns: meta.time_window,
        last_trace: meta.last_trace,
        medium_velocity_m_per_ns: meta.medium_velocity,
    }
}

fn correction_summary(location: &GPRLocation) -> CorrectionSummary {
    match &location.correction {
        LocationCorrection::None => CorrectionSummary {
            kind: "none".to_string(),
            source: None,
        },
        LocationCorrection::Dem(path) => CorrectionSummary {
            kind: "dem".to_string(),
            source: Some(path.to_string_lossy().to_string()),
        },
    }
}

fn location_summary(location: &GPRLocation) -> Result<LocationSummary, String> {
    if location.cor_points.is_empty() {
        return Err("No coordinate points were available for the input.".to_string());
    }
    let altitudes = Array1::from_iter(location.cor_points.iter().map(|point| point.altitude));
    let eastings = Array1::from_iter(location.cor_points.iter().map(|point| point.easting));
    let northings = Array1::from_iter(location.cor_points.iter().map(|point| point.northing));
    let length = location.length();
    let duration = location.cor_points[location.cor_points.len() - 1].time_seconds
        - location.cor_points[0].time_seconds;

    Ok(LocationSummary {
        n_points: location.cor_points.len(),
        crs: location.crs.clone(),
        start_time: tools::seconds_to_rfc3339(location.cor_points[0].time_seconds),
        stop_time: tools::seconds_to_rfc3339(
            location.cor_points[location.cor_points.len() - 1].time_seconds,
        ),
        duration_s: duration,
        track_length_m: length,
        altitude_min_m: altitudes.iter().cloned().fold(f64::INFINITY, f64::min),
        altitude_max_m: altitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        centroid: CentroidSummary {
            easting: eastings.mean().unwrap(),
            northing: northings.mean().unwrap(),
            altitude: altitudes.mean().unwrap(),
        },
        correction: correction_summary(location),
    })
}

fn info_record(
    resolved: &ResolvedInput,
    meta: &GPRMeta,
    location: &GPRLocation,
) -> Result<InfoRecord, String> {
    let fmt = formats::format_info(resolved.kind);
    Ok(InfoRecord {
        input: resolved.input.to_string_lossy().to_string(),
        format: NamedFormat {
            name: fmt.name,
            description: fmt.description,
        },
        metadata: metadata_summary(meta),
        location: location_summary(location)?,
        related_files: RelatedFiles {
            header: resolved.header.to_string_lossy().to_string(),
            data: resolved.data.to_string_lossy().to_string(),
            coordinates: resolved.coordinates.to_string_lossy().to_string(),
        },
    })
}

pub fn inspect(params: InfoParams) -> Result<Vec<InfoRecord>, Box<dyn Error>> {
    let expanded = expand_input_paths(&params.filepaths)?;
    let mut records = Vec::<InfoRecord>::new();

    for input in expanded {
        let resolved = formats::resolve_input(&input)?;
        let (meta, location) = load_meta_and_location(
            &resolved,
            params.medium_velocity,
            params.crs.as_ref(),
            params.cor_path.as_ref(),
            params.dem_path.as_ref(),
            params.override_antenna_mhz,
            params.override_antenna_separation,
        )?;
        records.push(info_record(&resolved, &meta, &location)?);
    }

    Ok(records)
}

pub fn build_processed_gpr(
    params: RunParams,
) -> Result<(GPR, PathBuf), Box<dyn std::error::Error>> {
    let expanded = expand_input_paths(&params.filepaths)?;
    if expanded.is_empty() {
        return Err("No input files were provided.".into());
    }
    let mut resolved_inputs = expanded
        .iter()
        .map(|input| crate::formats::resolve_input(input))
        .collect::<Result<Vec<crate::formats::ResolvedInput>, String>>()?;
    if resolved_inputs.is_empty() {
        return Err("No input files were resolved.".into());
    }

    let output_path = derive_output_path(params.output_path.as_ref(), &resolved_inputs[0])?;
    let first_resolved = resolved_inputs.remove(0);
    let (first_meta, first_location) = load_meta_and_location(
        &first_resolved,
        params.medium_velocity,
        params.crs.as_ref(),
        params.cor_path.as_ref(),
        params.dem_path.as_ref(),
        params.override_antenna_mhz,
        params.override_antenna_separation,
    )?;
    let mut gpr = GPR::from_meta_and_loc(first_location, first_meta).map_err(|e| {
        format!(
            "Error loading GPR data from {:?}: {:?}",
            first_resolved.data, e
        )
    })?;
    gpr.user_metadata = params.user_metadata.clone();

    for resolved in resolved_inputs {
        let (meta, location) = load_meta_and_location(
            &resolved,
            params.medium_velocity,
            params.crs.as_ref(),
            params.cor_path.as_ref(),
            params.dem_path.as_ref(),
            params.override_antenna_mhz,
            params.override_antenna_separation,
        )?;
        let other = GPR::from_meta_and_loc(location, meta)
            .map_err(|e| format!("Error loading GPR data from {:?}: {:?}", resolved.data, e))?;
        if !params.quiet {
            println!(
                "Merging {:?} -> {:?}",
                resolved.header, first_resolved.header
            );
        }
        gpr.merge(&other)?;
    }

    let start_time = std::time::SystemTime::now();
    if !params.quiet {
        println!("Processing {:?}", gpr.metadata.data_filepath);
    }
    for (i, step) in params.steps.iter().enumerate() {
        if !params.quiet {
            println!(
                "{}/{}, t+{:.2} s, Running step {}.",
                i + 1,
                params.steps.len(),
                std::time::SystemTime::now()
                    .duration_since(start_time)
                    .unwrap()
                    .as_secs_f32(),
                step,
            );
        }
        gpr.process(step)
            .map_err(|e| format!("Error on step {}: {:?}", step, e))?;
    }

    // Resolved once, here, from CLI arguments and the output stem. There is
    // no "inherited" tier yet because Ridal cannot currently read its own
    // processed NetCDF outputs back in as input (#123 adds metadata-only
    // inspection; full reading into GPR is explicitly future work there).
    // Passing `None` for `inherited` keeps `identity::resolve_*` already
    // shaped for that follow-up rather than requiring a redesign then.
    let output_stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("Output path has no valid file stem: {output_path:?}"))?;
    let (radargram_id, id_note) =
        crate::identity::resolve_radargram_id(params.radargram_id.as_deref(), None, output_stem)?;
    if let Some(note) = id_note {
        if !params.quiet {
            println!("{note}");
        }
    }
    let display_name = crate::identity::resolve_display_name(params.display_name.as_deref(), None);
    let group = crate::identity::resolve_group(
        params.group.as_deref(),
        params.group_id.as_deref(),
        None,
        None,
    )?;
    let (group_name, group_id) = match group {
        Some((name, id)) => (Some(name), Some(id)),
        None => (None, None),
    };
    gpr.identity = RidalIdentity {
        radargram_id: Some(radargram_id),
        display_name,
        group_name,
        group_id,
    };

    Ok((gpr, output_path))
}

pub fn run(params: RunParams) -> Result<ProcessResult, Box<dyn std::error::Error>> {
    let (gpr, output_path) = build_processed_gpr(params.clone())?;

    if !params.no_export {
        if !params.quiet {
            println!("Exporting to {:?}", output_path);
        }
        gpr.export(&output_path)
            .map_err(|e| format!("Error exporting data: {:?}", e))?;
    }

    if let Some(potential_fp) = &params.render_path {
        let render_filepath = match potential_fp {
            Some(fp) => {
                if fp.is_dir() {
                    fp.join(output_path.file_stem().unwrap())
                        .with_extension("jpg")
                } else {
                    fp.clone()
                }
            }
            None => output_path.with_extension("jpg"),
        };
        let profile = crate::render::profile::RenderProfile::resolve(
            params.render_profile.as_deref().unwrap_or("default"),
        )?;
        if !params.quiet {
            println!(
                "Rendering image to {:?} with the '{}' profile",
                render_filepath, profile.name
            );
        }
        gpr.render(&render_filepath, &profile, params.render_width)
            .map_err(|e| format!("Error writing image: {:?}", e))?;
    }

    if let Some(potential_track_path) = &params.track_path {
        crate::io::export_locations(
            &gpr.location,
            potential_track_path.into(),
            &output_path,
            !params.quiet,
        )?;
    }

    Ok(ProcessResult { output_path })
}

pub fn run_batch(params: BatchRunParams) -> Result<BatchProcessResult, String> {
    if !params.output_dir.is_dir() {
        return Err(format!(
            "output_dir must be an existing directory: {}",
            params.output_dir.display()
        ));
    }
    if let Some(render_dir) = &params.render_dir {
        if !render_dir.is_dir() {
            return Err(format!(
                "render_dir must be an existing directory: {}",
                render_dir.display()
            ));
        }
    }
    if let Some(track_dir) = &params.track_dir {
        if !track_dir.is_dir() {
            return Err(format!(
                "track_dir must be an existing directory: {}",
                track_dir.display()
            ));
        }
    }

    // Expand globs exactly like process mode does now.
    let expanded_filepaths = expand_cli_inputs(&params.filepaths)?;

    // Group inputs for batch processing.
    let groups = if let Some(merge_threshold) = params.merge.as_deref() {
        group_batch_inputs_chronologically(
            &expanded_filepaths,
            merge_threshold,
            params.medium_velocity,
            params.cor_path.clone(),
            params.dem_path.clone(),
            params.crs.clone(),
            params.override_antenna_mhz,
            params.override_antenna_separation,
        )?
    } else {
        expanded_filepaths
            .into_iter()
            .map(|p| vec![p])
            .collect::<Vec<Vec<PathBuf>>>()
    };

    let mut output_paths = Vec::<PathBuf>::new();

    for group in groups {
        if group.is_empty() {
            continue;
        }

        let first = &group[0];
        let stem = first
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Could not derive output stem from {}", first.display()))?;
        let output_nc = params.output_dir.join(format!("{stem}.nc"));

        let render_path = params
            .render_dir
            .as_ref()
            .map(|dir| Some(dir.join(format!("{stem}.jpg"))));

        let track_path = params
            .track_dir
            .as_ref()
            .map(|dir| Some(dir.join(format!("{stem}_track.csv"))));

        let run_params = RunParams {
            filepaths: group.clone(),
            output_path: Some(output_nc.clone()),
            dem_path: params.dem_path.clone(),
            cor_path: params.cor_path.clone(),
            medium_velocity: params.medium_velocity,
            crs: params.crs.clone(),
            quiet: params.quiet,
            track_path,
            steps: params.steps.clone(),
            no_export: params.no_export,
            render_path,
            render_profile: params.render_profile.clone(),
            render_width: params.render_width,
            override_antenna_mhz: params.override_antenna_mhz,
            override_antenna_separation: params.override_antenna_separation,
            user_metadata: params.user_metadata.clone(),
            radargram_id: None,
            display_name: None,
            group: params.group.clone(),
            group_id: params.group_id.clone(),
        };

        // process-mode semantics: one group -> one output, in the order we pass here.
        // For merge groups, group_batch_inputs_chronologically() already ordered them chronologically.
        let result = run(run_params).map_err(|e| e.to_string())?;
        output_paths.push(result.output_path);
    }

    Ok(BatchProcessResult { output_paths })
}

fn expand_cli_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::<PathBuf>::new();

    for input in inputs {
        let text = input.to_string_lossy();
        let looks_like_glob = text.contains('*') || text.contains('?') || text.contains('[');

        if looks_like_glob {
            let mut matched_any = false;
            let pattern =
                glob::glob(&text).map_err(|e| format!("Invalid glob pattern '{}': {e}", text))?;
            for entry in pattern {
                let path = entry.map_err(|e| format!("Error expanding glob '{}': {e}", text))?;
                matched_any = true;
                out.push(path);
            }
            if !matched_any {
                return Err(format!("Glob did not match any files: {text}"));
            }
        } else {
            out.push(input.clone());
        }
    }

    if out.is_empty() {
        return Err("No input files were provided".to_string());
    }

    Ok(out)
}

fn load_single_gpr_for_grouping(
    path: &Path,
    medium_velocity: f32,
    cor_path: Option<PathBuf>,
    dem_path: Option<PathBuf>,
    crs: Option<String>,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<GPR, String> {
    // Resolve input using the same format machinery as normal processing.
    let resolved = formats::resolve_input(path)
        .map_err(|e| format!("Failed to resolve input '{}': {e}", path.display()))?;

    let metadata = match resolved.kind {
        FormatKind::Ramac => io::load_rad(
            &resolved.header,
            medium_velocity,
            override_antenna_mhz,
            override_antenna_separation,
        )
        .map_err(|e| {
            format!(
                "Failed to load RAMAC header '{}': {e}",
                resolved.header.display()
            )
        })?,
        FormatKind::PulseEkko => io::load_pe_hd(
            &resolved.header,
            medium_velocity,
            override_antenna_mhz,
            override_antenna_separation,
        )
        .map_err(|e| {
            format!(
                "Failed to load pulseEKKO header '{}': {e}",
                resolved.header.display()
            )
        })?,
        FormatKind::Gssi => io::load_gssi_dzt(
            &resolved.header,
            medium_velocity,
            override_antenna_mhz,
            override_antenna_separation,
        )
        .map_err(|e| {
            format!(
                "Failed to load GSSI header '{}': {e}",
                resolved.header.display()
            )
        })?,
    };

    let mut location = if let Some(cor_override) = &cor_path {
        io::load_cor(cor_override, crs.as_ref()).map_err(|e| {
            format!(
                "Failed to load overridden .cor '{}': {e}",
                cor_override.display()
            )
        })?
    } else {
        match resolved.kind {
            FormatKind::Ramac => metadata.find_cor(crs.as_ref()).map_err(|e| {
                format!(
                    "Failed to find/load .cor for '{}': {e}",
                    metadata.data_filepath.display()
                )
            })?,
            FormatKind::PulseEkko => {
                let gp2 = &resolved.coordinates;
                io::load_pe_gp2(gp2, crs.as_ref())
                    .map_err(|e| format!("Failed to load .gp2 '{}': {e}", gp2.display()))?
            }
            FormatKind::Gssi => {
                let dzg = &resolved.coordinates;
                io::load_gssi_dzg(dzg, crs.as_ref())
                    .map_err(|e| format!("Failed to load .dzg '{}': {e}", dzg.display()))?
            }
        }
    };

    if let Some(dem_path) = &dem_path {
        location.get_dem_elevations(dem_path)?;
    }

    let mut gpr = GPR::from_meta_and_loc(location, metadata)
        .map_err(|e| format!("Failed to build GPR for '{}': {e}", path.display()))?;
    gpr.user_metadata = user_metadata::UserMetadata::new();
    Ok(gpr)
}

#[allow(clippy::too_many_arguments)]
fn group_batch_inputs_chronologically(
    filepaths: &[PathBuf],
    merge_threshold: &str,
    medium_velocity: f32,
    cor_path: Option<PathBuf>,
    dem_path: Option<PathBuf>,
    crs: Option<String>,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<Vec<Vec<PathBuf>>, String> {
    let threshold = parse_duration::parse(merge_threshold)
        .map_err(|e| format!("Failed to parse merge duration '{merge_threshold}': {e}"))?
        .as_secs_f64();

    let mut loaded = filepaths
        .iter()
        .map(|path| {
            let gpr = load_single_gpr_for_grouping(
                path,
                medium_velocity,
                cor_path.clone(),
                dem_path.clone(),
                crs.clone(),
                override_antenna_mhz,
                override_antenna_separation,
            )?;
            let start_time = gpr
                .location
                .cor_points
                .first()
                .map(|p| p.time_seconds)
                .ok_or_else(|| format!("No location points found in '{}'", path.display()))?;
            Ok::<_, String>((path.clone(), start_time, gpr))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Chronological grouping for batch --merge
    loaded.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut groups = Vec::<Vec<PathBuf>>::new();
    let mut current = Vec::<PathBuf>::new();
    let mut current_gpr: Option<GPR> = None;

    for (path, _start, gpr) in loaded {
        match current_gpr.as_mut() {
            None => {
                current.push(path);
                current_gpr = Some(gpr);
            }
            Some(existing) => {
                let gap = existing.location.duration_since(&gpr.location).abs();
                let compatible = existing.merge(&gpr).is_ok();

                if (gap <= threshold) && compatible {
                    // We already merged into 'existing' above to test compatibility, so keep this path in group.
                    current.push(path);
                } else {
                    groups.push(current);
                    current = vec![path];
                    current_gpr = Some(gpr);
                }
            }
        }
    }

    if !current.is_empty() {
        groups.push(current);
    }

    Ok(groups)
}
// src/steps.rs
const STEPS_MD: &str = include_str!("../steps.md");

pub fn all_available_steps() -> Vec<(String, String)> {
    // Split on “## ” headings
    STEPS_MD
        .split("\n## ")
        .skip(1) // text before first heading
        .filter_map(|section| {
            let mut lines = section.lines();

            // first line after "## " is the name (up to end of line)
            let name = lines.next()?.trim().to_string();

            // rest is the description
            let description = lines.collect::<Vec<_>>().join("\n").trim().to_string();

            Some((name, description))
        })
        .collect()
}
pub fn validate_steps(steps: &[String]) -> Result<(), String> {
    let allowed_steps = all_available_steps()
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<String>>();
    for step in steps {
        if !allowed_steps.iter().any(|allowed| step.contains(allowed)) {
            return Err(format!("Unrecognized step: {step}"));
        }
    }
    Ok(())
}

pub fn default_processing_profile() -> Vec<String> {
    vec![
        "remove_empty_traces".to_string(),
        "zero_corr_max_peak".to_string(),
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
pub mod tests {
    use std::{path::PathBuf, str::FromStr};

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
        let cor_points = make_cor_points(n_points, spacing.unwrap_or(1.));
        let crs = match crs {
            Some(c) => c,
            None => {
                let first_coord = crate::coords::Coord {
                    x: cor_points[0].easting,
                    y: cor_points[0].northing,
                };
                crate::coords::UtmCrs::optimal_crs(&first_coord).to_epsg_str()
            }
        };
        GPRLocation {
            cor_points,
            correction: correction.unwrap_or(LocationCorrection::None),
            crs,
        }
    }

    pub fn make_dummy_gpr(n_traces: usize, n_samples: usize, spacing: Option<f64>) -> super::GPR {
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
            data_filepath: std::path::PathBuf::from_str("filepath.rd3").unwrap(),
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
            steps: Vec::new(),
            zero_point_ns: 0.,
            horizontal_signal_distance: 1.,
            log: Vec::new(),
            user_metadata: crate::user_metadata::UserMetadata::new(),
            // A real radargram_id is required for export() to succeed
            // (#116); test helpers give it one directly rather than routing
            // through build_processed_gpr()'s CLI-argument resolution.
            identity: super::RidalIdentity {
                radargram_id: Some(crate::identity::RadargramId::new("test-radargram").unwrap()),
                display_name: None,
                group_name: None,
                group_id: None,
            },
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

    /// `distances` and `velocities` must take the square root of the summed
    /// squared components.
    ///
    /// The unit-spacing fixture used by `test_gpr_location` cannot detect a
    /// missing `sqrt`, because a step of length one is its own square. This
    /// uses a 3-4-5 step, where the true per-step length (5) and the squared
    /// length (25) differ, so only a correct implementation passes.
    #[test]
    fn test_distances_and_velocities_are_lengths_not_squared_lengths() {
        // Steps of (3, 4) in (easting, northing) -- length exactly 5 -- with
        // altitude held constant so the 3D step `velocities` measures is the
        // same 5, and one second per step so speed equals step length.
        let cor_points: Vec<CorPoint> = (0..5_u32)
            .map(|i| CorPoint {
                trace_n: i,
                time_seconds: i as f64,
                easting: 3. * i as f64,
                northing: 4. * i as f64,
                altitude: 0.,
            })
            .collect();
        let location = GPRLocation {
            cor_points,
            correction: LocationCorrection::None,
            crs: "EPSG:32633".to_string(),
        };

        let distances = location.distances();
        assert_eq!(distances[0], 0.);
        // Cumulative length, not a cumulative sum of squares (which would
        // give 25, 50, 75, 100).
        assert_eq!(distances.to_vec(), vec![0., 5., 10., 15., 20.]);
        assert_eq!(location.length(), 20.);

        // The first entry is 0 by construction (no preceding point); every
        // later step covers 5 m in 1 s.
        let velocities = location.velocities();
        assert_eq!(velocities.to_vec(), vec![0., 5., 5., 5., 5.]);
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
            data_filepath: PathBuf::new(),
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
            steps: Vec::new(),
            log: Vec::new(),
            horizontal_signal_distance: antenna_separation,
            zero_point_ns: 0.,
            user_metadata: crate::user_metadata::UserMetadata::new(),
            identity: super::RidalIdentity::default(),
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

        // The stationary run collapses to a single trace: the whole cluster
        // sits at distance 0, so only the first target-grid position can
        // draw from it. This is the property the test is really about.
        assert!(gpr.location.cor_points[1].easting > first.easting);

        // The exact resulting width follows from the auto-step heuristic:
        // the track covers 127 m (the 10 stacked traces contribute nothing,
        // then 117 unit steps plus the 10 m jump out of the cluster), and
        // 118 traces are classified as moving, giving a step of
        // 127/118 ~= 1.076 m and 120 positions on the resampled grid.
        //
        // This previously read `width - (n_stationary - 1)` == 119, which
        // held only because `distances`/`velocities` both returned squared
        // lengths: the bogus max_distance (217) and the bogus step
        // (217/118) shared the same squared units, so their ratio landed
        // near the right trace count for this unit-spaced fixture. That
        // cancellation is why the fixture could not detect the missing
        // `sqrt` -- see
        // `test_distances_and_velocities_are_lengths_not_squared_lengths`.
        assert_eq!(gpr.width(), 120);
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
