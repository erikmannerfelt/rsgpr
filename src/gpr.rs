use std::path::{Path,PathBuf};
use std::error::Error;

use ndarray_stats::QuantileExt;
use std::time::SystemTime;

use ndarray::{Array2, Axis, Slice, Array1};
use smartcore::linalg::BaseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use rayon::prelude::*;

use crate::{io, dem, tools};


const DEFAULT_ZERO_CORR_THRESHOLD_MULTIPLIER: f32 = 1.0;
const DEFAULT_DEWOW_WINDOW: u32 = 5;
const DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF: f32 = 0.3;
const DEFAULT_AUTOGAIN_N_BINS: usize = 100;

#[derive(Debug, Clone)]
pub struct GPRMeta {
    pub samples: u32,
    pub frequency: f32,
    pub frequency_steps: u32,
    pub time_interval: f32,
    pub antenna: String,
    pub antenna_mhz: f32,
    pub antenna_separation: f32,
    pub time_window: f32,
    pub last_trace: u32,
    pub rd3_filepath: PathBuf,
    pub medium_velocity: f32,
}

impl GPRMeta {

    pub fn find_cor(&self, projected_crs: &str) -> Result<GPRLocation, Box<dyn Error>> {
        io::load_cor(&self.rd3_filepath.with_extension("cor"), projected_crs)
    }

}

impl std::fmt::Display for GPRMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result  {

        write!(f, "
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
        [self.time_seconds, self.easting, self.northing, self.altitude]
    }
}

#[derive(Debug, Clone)]
pub enum LocationCorrection {
    NONE,
    DEM(PathBuf),
}

#[derive(Debug)]
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
            return (first_point.time_seconds, first_point.easting, first_point.northing, first_point.altitude);
        };

        if trace_n < last_point.trace_n {
            // At this time, the trace
            for point in &self.cor_points {
                if point.trace_n == trace_n {
                    return (point.time_seconds, point.easting, point.northing, point.altitude);
                };

                if trace_n < point.trace_n {
                    let v = tools::interpolate_values(first_point.trace_n as f64, &first_point.txyz(), point.trace_n as f64, &point.txyz(), trace_n as f64);
                    return (v[0], v[1], v[2], v[3]);
                    
                };
                first_point = &point;
            };
        };
        (last_point.time_seconds, last_point.easting, last_point.northing, last_point.altitude)
    }

    fn velocities(&self) -> Array1<f64> {

        
        //let mut offsets: Vec<[f64; 4]> = Vec::new();

        let mut offsets = Array2::from_elem((self.cor_points.len(), 4), 0_f64);

        for i in 1..self.cor_points.len() {

            let mut slice = offsets.slice_axis_mut(Axis(0), Slice::new(i as isize, Some(i as isize + 1), 1));
            slice.assign(&Array1::from_vec(vec![
                     self.cor_points[i].time_seconds - self.cor_points[i - 1].time_seconds, 
                     self.cor_points[i].easting - self.cor_points[i - 1].easting, 
                     self.cor_points[i].northing - self.cor_points[i - 1].northing, 
                     self.cor_points[i].altitude - self.cor_points[i - 1].altitude,
            ]));
        };


        let d = offsets.slice_axis(Axis(1), Slice::new(1, None, 1)).mapv(|f| f.powi(2)).sum_axis(Axis(1));


        let vel = (d / offsets.column(0)).mapv(|f| if f.is_finite() {f} else {0.0});

        vel
    }

    pub fn distances(&self) -> Array1<f64> {
        let mut offsets = Array2::from_elem((self.cor_points.len(), 3), 0_f64);

        for i in 1..self.cor_points.len() {

            let mut slice = offsets.slice_axis_mut(Axis(0), Slice::new(i as isize, Some(i as isize + 1), 1));
            slice.assign(&Array1::from_vec(vec![
                     self.cor_points[i].time_seconds - self.cor_points[i - 1].time_seconds, 
                     self.cor_points[i].easting - self.cor_points[i - 1].easting, 
                     self.cor_points[i].northing - self.cor_points[i - 1].northing, 
                     //self.cor_points[i].altitude - self.cor_points[i - 1].altitude,
            ]));
        };


        let mut dist = offsets.slice_axis(Axis(1), Slice::new(1, None, 1)).mapv(|f| f.powi(2)).sum_axis(Axis(1));
        dist.accumulate_axis_inplace(Axis(0), | prev, cur | *cur += prev);

        dist
    }


    fn range_fill(&self, start_trace: u32, end_trace: u32) -> GPRLocation {

        let mut new_points: Vec<CorPoint> = Vec::new();


        for i in start_trace..end_trace {
                let txyz = self.time_and_coord_at_trace(i);

                new_points.push(CorPoint{trace_n: i, time_seconds: txyz.0, easting: txyz.1, northing: txyz.2, altitude: txyz.3})
        };

        
        GPRLocation { cor_points: new_points , correction: self.correction.clone(), crs: self.crs.clone()}
    }

    fn xy_coords(&self) -> Array2<f64> {

        let mut data: Vec<f64> = Vec::new();

        for point in &self.cor_points {
            data.push(point.easting);
            data.push(point.northing);
        };

        Array2::<f64>::from_shape_vec((self.cor_points.len(), 2), data).unwrap()
    }

    pub fn get_dem_elevations(&mut self, dem_path: &Path)  {
        let elev = dem::read_elevations(dem_path, self.xy_coords()).unwrap();

        for i in 0..self.cor_points.len() {
            self.cor_points[i].altitude = elev[i] as f64;
        };

        self.correction = LocationCorrection::DEM(dem_path.to_path_buf());
    }

    pub fn to_csv(&self, filepath: &Path) -> Result<(), std::io::Error> {

        let mut output = "trace_n,easting,northing,altitude\n".to_string();

        for point in &self.cor_points {
            output += &format!("{},{},{},{}\n", point.trace_n, point.easting, point.northing, point.altitude);
        };

        std::fs::write(filepath, output)
    }

    pub fn length(&self) -> f64 {
        self.distances().max().unwrap().to_owned()
    }
}


impl std::fmt::Display for GPRLocation {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result  {
        let altitudes = Array1::from_iter(self.cor_points.iter().map(|point| point.altitude));
        let eastings = Array1::from_iter(self.cor_points.iter().map(|point| point.easting));
        let northings = Array1::from_iter(self.cor_points.iter().map(|point| point.northing));

        let length = self.length();
        let duration = self.cor_points[self.cor_points.len() - 1].time_seconds - self.cor_points[0].time_seconds;
        write!(f, "

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



pub struct GPR {
    pub data: ndarray::Array2<f32>,
    pub location: GPRLocation,
    pub metadata: GPRMeta,
    pub log: Vec<String>,
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
            let mean_velocity = tools::parse_option::<f32>(step_name, 0)?;
            self.make_equidistant(mean_velocity);

        } else if step_name.contains("normalize_horizontal_magnitudes") {

            // Try to parse the argument as an integer. If that doesn't work, try to parse it as a
            // float and assume it's the fraction of the height
            let skip_first: isize = match tools::parse_option::<isize>(step_name, 0) {
                Ok(v) => Ok(v.unwrap_or(0)),
                Err(e) => match e.contains("Could not parse argument 0 as value") {
                    true => tools::parse_option::<f32>(step_name, 0).and_then(|fraction| match (fraction.unwrap() >= 1.0) & (fraction.unwrap() < 0.) {
                        true => Err(format!("Invalid fraction: {:?}. Must be between 0.0 and 1.0.", fraction)),
                        false => Ok((self.height() as f32 * fraction.unwrap_or(0.)) as isize),

                    }),
                    false => Err(e)
                }
            }?;
            self.normalize_horizontal_magnitudes(Some(skip_first));
        } else if step_name.contains("kirchhoff_migration2d") {
            self.kirchhoff_migration2d();
        } else if step_name.contains("auto_gain") {
            let n_bins = tools::parse_option::<usize>(step_name, 0)?.unwrap_or(DEFAULT_AUTOGAIN_N_BINS);
            self.auto_gain(n_bins);
        } else if step_name.contains("gain") {

            let linear = match tools::parse_option::<f32>(step_name, 0)? {
                Some(v) => Ok(v),
                None => Err("The linear gain factor must be specified when applying gain. E.g. gain(0.1)".to_string()),

            }?;
            self.gain(linear);
        } else if step_name.contains("subset") {

            let min_trace: Option<u32> = match tools::parse_option::<u32>(step_name, 0)? {
                Some(v) => Ok(Some(v)),
                None => Err("Indices must be given when subsetting, e.g. subset(0, -1, 0, 500)")
            }?;

            let max_trace: Option<u32> = match tools::parse_option::<isize>(step_name, 1) {
                Ok(v) => Ok(v.and_then(|v2| if v2 == -1 {None} else {Some(v2 as u32)})),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                }
            }?;
            let min_sample: Option<u32> = match tools::parse_option::<u32>(step_name, 2) {
                Ok(v) => Ok(v),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                }
            }?;
            let max_sample: Option<u32> = match tools::parse_option::<isize>(step_name, 3) {
                Ok(v) => Ok(v.and_then(|v2| if v2 == -1 {None} else {Some(v2 as u32)})),
                Err(e) => match e.contains("out of bounds") {
                    true => Ok(None),
                    false => Err(e),
                }
            }?;
            *self = self.subset(min_trace, max_trace, min_sample, max_sample);
        } else if step_name.contains("unphase") {
            self.unphase();
        } else {
            return Err(format!("Step name not recognized: {}", step_name).into());
        }

        Ok(())
    }

    pub fn subset(&self, min_trace: Option<u32>, max_trace: Option<u32>, min_sample: Option<u32>, max_sample: Option<u32>) -> GPR {

        let start_time = SystemTime::now();
        let min_trace_ = match min_trace {Some(x) => x, None => 0};
        let max_trace_ = match max_trace {Some(x) => x, None => self.width() as u32};
        let min_sample_ = match min_sample {Some(x) => x, None => 0};
        let max_sample_ = match max_sample {Some(x) => x, None => self.height() as u32};

        let data_subset = self.data.slice(ndarray::s![min_sample_ as isize..max_sample_ as isize, min_trace_ as isize..max_trace_ as isize]).to_owned();


        let location_subset = GPRLocation{cor_points: self.location.cor_points[min_trace_ as usize..max_trace_ as usize].to_owned(), correction: self.location.correction.clone(), crs: self.location.crs.clone()};

        let mut metadata = self.metadata.clone();

        metadata.last_trace = max_trace_;
        metadata.time_window = metadata.time_window * ((max_sample_ - min_sample_) as f32 / metadata.samples as f32);
        metadata.samples = max_sample_ - min_sample_;

        let log = self.log.clone();

        let mut new_gpr = GPR{data: data_subset, location: location_subset, metadata, log};
        new_gpr.log_event("subset", &format!("Subset data from {:?} to ({}:{}, {}:{})", self.data.shape(), min_sample_, max_sample_, min_trace_, max_trace_), start_time);

        new_gpr

    }

    pub fn vertical_resolution_ns(&self) -> f32 {
        self.metadata.time_window / self.metadata.samples as f32
    }

    pub fn from_meta_and_loc(location: GPRLocation, metadata: GPRMeta) -> Result<GPR, Box<dyn Error>> {

        let data = io::load_rd3(&metadata.rd3_filepath, metadata.samples as usize)?;

        let location_data = match data.shape()[1] == location.cor_points.len() {
            true => location,
            false => location.range_fill(0, data.shape()[1] as u32)

        };

        Ok(GPR {data, location: location_data, metadata, log: Vec::new()})
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
                break
            };
        };

        if first_rise == 0 {
            return
        };

        let mean_silent_val = mean_trace.slice_axis(Axis(0), Slice::new(0, Some(first_rise), 1)).mean().unwrap();

        self.data -= mean_silent_val;

        let mut positive_peaks = Array1::<isize>::zeros(self.width());

        let mut i = 0_usize;
        for col in self.data.columns() {

            positive_peaks[i] = col.argmax().unwrap() as isize;

            i += 1;
        };

        let mut new_data = Array2::from_elem((self.height() - positive_peaks.min().unwrap().to_owned() as usize, self.width()), 0_f32);

        i = 0;
        for col in self.data.columns() {

            let mut new_col = new_data.column_mut(i);

            let mut positive_data_slice = new_col.slice_axis_mut(Axis(0), Slice::new(0, Some(self.height() as isize - positive_peaks[i]), 1));

            positive_data_slice += &col.slice_axis(Axis(0), Slice::new(positive_peaks[i], None, 1));
            i += 1;

        };

        self.update_data(new_data);
        self.log_event("zero_corr_max_peak", &format!("Applied a per-trace zero-corr by removing the first {}-{} rows", positive_peaks.min().unwrap(), positive_peaks.max().unwrap()), start_time);
    }

    fn update_data(&mut self, data: Array2::<f32>) {

        self.data = data;

        self.metadata.time_window = self.metadata.time_window * (self.height() as f32 / self.metadata.samples as f32);
        self.metadata.samples = self.height() as u32;
        self.metadata.last_trace = self.width() as u32;


    }

    pub fn unphase(&mut self) {

        let start_time = SystemTime::now();

        let mut positive_peaks = Array1::<isize>::zeros(self.width());
        let mut negative_peaks = Array1::<isize>::zeros(self.width());

        let mut i = 0_usize;
        for col in self.data.columns() {

            positive_peaks[i] = col.argmax().unwrap() as isize;
            negative_peaks[i] = col.argmin().unwrap() as isize;

            i += 1;
        };

        let mean_peak_spacing = (negative_peaks - positive_peaks).mean().unwrap().abs();
        let mut new_data = self.data.mapv(|v| v.max(0.));

        self.data.slice_axis(Axis(0), Slice::new(mean_peak_spacing, None, 1)).mapv(|v| v.min(0.) * -1.).assign_to(new_data.slice_axis_mut(Axis(0), Slice::new(0, Some(self.height() as isize - mean_peak_spacing), 1)));

        self.update_data(new_data);

        self.log_event("unphase", &format!("Summed the positive and negative phases of the signal by shifting the negative signal component by {} rows", mean_peak_spacing), start_time);
    }

    pub fn zero_corr(&mut self, threshold_multiplier: Option<f32>) {
        let start_time = SystemTime::now();

        let mean_trace = self.data.mean_axis(Axis(1)).unwrap();

        let threshold = 0.5 * mean_trace.std(1.0) * threshold_multiplier.unwrap_or(DEFAULT_ZERO_CORR_THRESHOLD_MULTIPLIER);

        let mut first_rise = 0_isize;

        for i in 1..mean_trace.shape()[0] {
            if (mean_trace[i] - mean_trace[i - 1]).abs() > threshold {
                first_rise = i as isize;
                break
            };
        };

        if first_rise == 0 {
            return
        };

        let mean_silent_val = mean_trace.slice_axis(Axis(0), Slice::new(0, Some(first_rise), 1)).mean().unwrap();


        self.update_data(self.data.slice_axis(Axis(0), Slice::new(first_rise, None, 1)).to_owned());
        self.data -= mean_silent_val;

        self.log_event("zero_corr", &format!("Applied a global zero-corr by removing the first {} rows (threshold multiplier: {:?})", first_rise, threshold_multiplier), start_time);
    }

    pub fn dewow(&mut self, window: u32)  {
        let start_time = SystemTime::now();

        let height = self.height() as u32;

        for i in (0..(height - window)).step_by(window as usize) {

            let mut view = self.data.slice_axis_mut(Axis(0), ndarray::Slice::new(i as isize, Some((i + window) as isize), 1_isize));

            view -= view.mean().unwrap();
        };
        self.log_event("dewow", &format!("Ran dewow with a window size of {}", window), start_time);
    }

    pub fn normalize_horizontal_magnitudes(&mut self, skip_first: Option<isize>) {
        let start_time = SystemTime::now();
        if let Some(mean) = self.data.slice_axis(Axis(0), Slice::new(skip_first.unwrap_or(0), None, 1)).mean_axis(Axis(0)) {
            self.data -= &mean;
        };
        self.log_event("normalize_horizontal_magnitudes", &format!("Normalized horizontal magnitudes, skipping {:?} of the first rows", skip_first), start_time);
    }

    pub fn auto_gain(&mut self, n_bins: usize) {
        let start_time = SystemTime::now();
        

        let step = (self.height() / n_bins) as isize;

        let mut step_mids: Vec<f32> = Vec::new();
        let mut stds: Vec<f32> = Vec::new();

        for i in (0..(self.height() as isize - step)).step_by(step as usize) {

            let slice = self.data.slice_axis(Axis(0), Slice::new(i, Some(i + step), step));

            step_mids.push((i as f32 + (i + step) as f32) / 2.0);

            stds.push(slice.mapv(|a| a.abs()).mean().unwrap());
        };

        let xs = DenseMatrix::from_2d_array(&[&step_mids]).transpose();

        let lr = LinearRegression::fit(&xs, &stds, LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR)).unwrap();


        self.gain(-lr.coefficients().get(0, 0));
        self.log_event("auto_gain", &format!("Applied autogain from {} bins", n_bins), start_time);
    }

    pub fn gain(&mut self, linear: f32) {
        let start_time = SystemTime::now();

        for i in 0..self.height() as isize {

            let mut view = self.data.slice_axis_mut(Axis(0), Slice::new(i, Some(i + 1), 1_isize));
view *= (i as f32) * linear;
        };
        self.log_event("gain", &format!("Applied linear gain of *= {} * index", linear), start_time);
    }

    pub fn make_equidistant(&mut self, mean_velocity: Option<f32>) {
        let start_time = SystemTime::now();

        let velocities = self.location.velocities().mapv(|v| v as f32);

        let normal_velocity = mean_velocity.unwrap_or(tools::quantiles(&velocities, &[0.5])[0]);

        let mut seconds_moving = 0_f32;
        for i in 1..self.width() {
            if velocities[i] < (0.3 * normal_velocity) {
                continue
            };
            seconds_moving += self.metadata.time_interval;
        };

        let nominal_data_width = (seconds_moving / self.metadata.time_interval).floor() as usize;

        let distances = self.location.distances().mapv(|v| v as f32);
        let old_width = self.width() as usize;

        let step = distances.max().unwrap() / (nominal_data_width as f32);

        let mut j = 0_usize;
        let mut k = 0_usize;
        let mut new_coords: Vec<CorPoint> = Vec::new();
        for i in 0..nominal_data_width {
            if i >= j {
                j = i + 1;
            };
            new_coords.push(self.location.cor_points[j]);
            if j >= (old_width - 1) {
                break
            };
            for l in j..old_width {
                if ((distances[k] / step) as usize > i) | (k >= old_width - 1) {
                    break
                };
                k = l;
            };

            let old_data_slice = if k > j {
                self.data.slice_axis(Axis(1), Slice::new(j as isize, Some(k as isize + 1) , 1)).mean_axis(Axis(1)).unwrap()
            } else {
                self.data.column(j as usize).to_owned()
            };
            let mut new_data_slice = self.data.column_mut(i);

            new_data_slice.assign(&old_data_slice);

            j = k + 1;
        };

        self.update_data(self.data.slice_axis(Axis(1), Slice::new(0, Some(nominal_data_width as isize), 1)).to_owned());
        self.location.cor_points = new_coords;

        if self.width() != self.location.cor_points.len() {

            panic!("Data width {} != cor points width {}. Was equidistant traces run twice?", self.width(), self.location.cor_points.len())

        };

        self.log_event("equidistant_traces", "Ran equidistant traces", start_time);
    }


    fn log_event(&mut self, step_name: &str, event: &str, start_time: SystemTime) {
        self.log.push(format!("{} (duration: {:.2}s):\t{}", step_name, SystemTime::now().duration_since(start_time).unwrap().as_secs_f32(), event));
    }

    pub fn kirchhoff_migration2d(&mut self) {

        let start_time = SystemTime::now();
        let x_coords = self.location.distances().mapv(|v| v as f32);
        let mut x_diff = 0_f32;
        for i in 1..x_coords.shape()[0] {
            x_diff += x_coords[i] - x_coords[i - 1];
        };
        x_diff /= (x_coords.shape()[0] - 1) as f32;

        // The z-coords will be negative height in relation to the highest point (all values are
        // positive).
        let mut z_coords = Array1::from_iter(self.location.cor_points.iter().map(|point| point.altitude as f32));
        z_coords -= *z_coords.max().unwrap();
        z_coords *= -1.0;

        // The vertical resolution in ns
        let t_diff = self.vertical_resolution_ns();

        // The vertical resolution in m
        let z_diff = t_diff * self.metadata.medium_velocity;

        // The minimum logical resolution, assuming just one wavelength
        let logical_res = (self.metadata.antenna_mhz / self.metadata.medium_velocity) * (1e-9 * 1e6);

        let old_data = self.data.iter().collect::<Vec<&f32>>();

        let height = self.height();
        let width = self.width();

        let output: Vec<f32> = (0..(width * height)).into_par_iter().map(|sample_idx| {
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
            let min_neighbor = (trace_n as f32 - fresnel_width).floor().clamp(0_f32, width as f32) as usize;
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
                let t_top = t_0 - 2. * (z_coords[neighbor_n] - trace_top_z) / self.metadata.medium_velocity;

                // Get the travel time to the sample accounting for the x distance
                let t_x = (t_top.powi(2) + (2. * (x_coords[neighbor_n] - trace_x) / self.metadata.medium_velocity).powi(2)).sqrt();
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

                    let weight = match t_1_px == t_2_px {true => 0_f32, false => ((t_1_px as f32 - (0.5 * t_x / t_diff)) / (t_1_px as f32 - t_2_px as f32)).abs()};

                    ampl += 
                        (x_diff / (2. * std::f32::consts::PI * t_x * self.metadata.medium_velocity).sqrt()) // Account for the horizontal distance
                        * (t_top / t_x) // Account for the vertical distance
                        * ((1. - weight) * old_data[(t_1_px * width) + neighbor_n] + weight * old_data[(t_2_px * width) + neighbor_n]) // Account for the neigbour's value
                        * if (neighbor_n == min_neighbor) | (neighbor_n == max_neighbor - 1)
                         {out_fresnel_weight} else {in_fresnel_weight}
                    ;
                    n_ampl += 1.0;
                };

            };

            if n_ampl > 0. {
                ampl / n_ampl
            } else {
                0.
            }

        }).collect();

        self.update_data(Array2::from_shape_vec((height, width), output).unwrap());
        self.log_event("kirchhoff_migration2d", &format!("Ran 2D Kirchhoff migration with a velocity of {} m/ns", self.metadata.medium_velocity), start_time);
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
}


pub fn all_available_steps() -> Vec<[&'static str; 2]> {

    vec![
        ["zero_corr_max_peak", "Shift the location of the zero return time by finding the maximum row value. The peak is found for each trace individually."],
        ["zero_corr", "Shift the location of the zero return time by finding the first row where data appear. The correction can be tweaked to allow more or less data, e.g. 'zero_corr(0.9)'. Default: 1.0"],
        ["equidistant_traces", "Make all traces equidistant by averaging them in a fixed horizontal grid. The gridsize is determined from the median moving velocity. Other velocities in m/s can be given, e.g. 'equidistant_traces(2.)'. Default: auto"],
        ["normalize_horizontal_magnitudes", "Normalize the magnitudes of the traces in the horizontal axis. This removes or reduces horizontal banding. The uppermost samples of the trace can be excluded, either by sample number (integer; e.g. 'normalize_horizontal_magnitudes(300)') or by a fraction of the trace (float; e.g. 'normalize_horizontal_magnitudes(0.3)'). Default: 0.3"],
        ["dewow", "Subtract the horizontal moving average magnitude for each trace. This reduces artefacts that are consistent among every trace. The averaging window can be set, e.g. 'dewow(10)'. Default: 5"],
        ["auto_gain", "Automatically determine the best linear gain and apply it. The data are binned vertically and the standard deviation of the values is used as a proxy for signal attenuation. A linear model is fit to the standard deviations vs. depth, and the subsequent linear coefficient is given to the gain filter. Note that this will show up as having run auto_gain and then gain in the log. The amounts of bins can be given, e.g. 'auto_gain(100). Default: 100"],
        ["gain", "Linearly multiply the magnitude as a function of depth. This is most often used to correct for signal attenuation with time/distance. Gain is applied by: 'gain * sample_index' where gain is the given gain and sample_index is the zero-based index of the sample from the top. Example: gain(0.1). No default value."],
        ["kirchhoff_migration2d", "Migrate sample magnitudes in the horizontal and vertical distance dimension to correct hyperbolae in the data. The correction is needed because the GPR does not observe only what is directly below it, but rather in a cone that is determined by the dominant antenna frequency. Thus, without migration, each trace is the mean of a cone beneath it. Topographic Kirchhoff migration (in 2D) corrects for this in two dimensions."],
        ["unphase", "Combine the positive and negative phases of the signal into one positive magntiude. The assumption is made that the positive magnitude of the signal comes first, followed by an offset negative component. The distance between the positive and negative peaks are found, and then the negative part is shifted accordingly."],
    ]

}

pub fn default_processing_profile() -> Vec<String> {
    vec![
        format!("zero_corr_max_peak"),
        "equidistant_traces".to_string(),
        format!("normalize_horizontal_magnitudes({})", DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF),
        "unphase".to_string(),
        "kirchhoff_migration2d".to_string(),
        format!("normalize_horizontal_magnitudes({})", DEFAULT_NORMALIZE_HORIZONTAL_MAGNITUDES_CUTOFF),
        format!("dewow({})", DEFAULT_DEWOW_WINDOW),
        format!("auto_gain({})", DEFAULT_AUTOGAIN_N_BINS),
    ]
}
