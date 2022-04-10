use std::ops::Mul;
use std::path::{Path,PathBuf};
use std::collections::HashMap;
use std::error::Error;

use nshare::MutNdarray2;
use image::{GrayImage};
use ndarray_stats::{QuantileExt, Quantile1dExt};
//use show_image::{ImageView, ImageInfo, create_window};

use ndarray::{Array2, Axis, Slice, Array1, ArrayView1, ArrayBase, array};
use smartcore::linalg::BaseMatrix;
use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use rayon::prelude::*;

fn main()-> Result<(), Box<dyn std::error::Error>> {

    let gpr_meta = load_rad(Path::new("/media/hdd/Erik/Data/GPR/2022/AG325/glacier_radar/Slakbreen/20220330/DAT_0251_A1.rad")).unwrap();

    let gpr_locations = gpr_meta.find_cor(33).unwrap();

    let mut gpr = GPR::from_meta_and_loc(gpr_locations, gpr_meta).unwrap().subset(Some(0), None, Some(0), None);

    let start = std::time::SystemTime::now();
    println!("Width: {} Height: {}\nt+0ms Running zero-corr", gpr.width(), gpr.height());
    gpr.zero_corr(None);

    println!("t+{:?} Normalizing horizontal magnitudes...", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.normalize_horizontal_magnitudes();
    println!("t+{:?} Making traces equidistant", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.make_equidistant();
    println!("t+{:?} Running dewow", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.dewow(5);
    println!("t+{:?} Automatically finding gain", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.auto_gain(50);

    println!("t+{:?} Migrating", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.kirchoff_migration2d(0.168);

    println!("t+{:?} Saving ", std::time::SystemTime::now().duration_since(start).unwrap());
    gpr.render(&Path::new("img.jpg"))?;

    Ok(())
}


#[derive(Debug, Clone)]
struct GPRMeta {
    samples: u32,
    frequency: f32,
    frequency_steps: u32,
    time_interval: f32,
    antenna: String,
    antenna_mhz: f32,
    antenna_separation: f32,
    time_window: f32,
    last_trace: u32,
    rd3_filepath: PathBuf,
}

impl GPRMeta {

    pub fn find_cor(&self, utm_zone: u8) -> Result<GPRLocation, Box<dyn Error>> {
        load_cor(&self.rd3_filepath.with_extension("cor"), utm_zone)
    }


}


fn load_rad(filepath: &Path) -> Result<GPRMeta, Box<dyn Error>> {

    let content = std::fs::read_to_string(filepath)?;

    let data: HashMap<&str, &str> = content.lines().filter_map(|s| s.split_once(":")).collect();

    let rd3_filepath = filepath.with_extension("rd3");

    if !rd3_filepath.is_file() {
        return Err(format!("File not found: {:?}", rd3_filepath).into());
    };

    let antenna = data.get("ANTENNAS").ok_or("No 'ANTENNAS' key in metadata")?.trim().to_string();

    Ok(GPRMeta{
        samples: data.get("SAMPLES").ok_or("No 'SAMPLES' key in metadata")?.trim().parse()?,
        frequency: data.get("FREQUENCY").ok_or("No 'FREQUENCY' key in metadata")?.trim().parse()?,
        frequency_steps: data.get("FREQUENCY STEPS").ok_or("No 'FREQUENCY STEPS' key in metadata")?.trim().parse()?,
        time_interval: data.get("TIME INTERVAL").ok_or("No 'TIME INTERVAL' key in metadata")?.replace(" ", "").parse()?,
        antenna_mhz: antenna.split("MHz").collect::<Vec<&str>>()[0].trim().parse()?,
        antenna,
        antenna_separation: data.get("ANTENNA SEPARATION").ok_or("No 'ANTENNA SEPARATION' key in metadata")?.trim().parse()?,
        time_window: data.get("TIMEWINDOW").ok_or("No 'TIMEWINDOW' key in metadata")?.trim().parse()?,
        last_trace: data.get("LAST TRACE").ok_or("No 'LAST TRACE' key in metadata")?.trim().parse()?,
        rd3_filepath,
    })
}

#[derive(Debug)]
struct GPRLocation {
    cor_points: Vec<CorPoint>
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
                    let v = interpolate_values(first_point.trace_n as f64, &first_point.txyz(), point.trace_n as f64, &point.txyz(), trace_n as f64);
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

    fn distances(&self) -> Array1<f64> {
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

        
        GPRLocation { cor_points: new_points }
    }
}

fn interpolate_values(x0: f64, y0: &[f64], x1: f64, y1: &[f64], x: f64) -> Vec<f64> {

    let mut output: Vec<f64> = Vec::new();

    for i in 0..y0.len() {
        output.push(interpolate_between_known((x0, y0[i]), (x1, y1[i]), x))
    };

    output
}

fn interpolate_coordinate(time0: f64, coord0: (f64, f64, f64), time1: f64, coord1: (f64, f64, f64), time: f64) -> (f64, f64, f64) {

    let easting = interpolate_between_known((time0, coord0.0), (time1, coord1.0), time);
    let northing = interpolate_between_known((time0, coord0.1), (time1, coord1.1), time);
    let altitude = interpolate_between_known((time0, coord0.2), (time1, coord1.2), time);

    (easting, northing, altitude)
}

/// Interpolate linearly between two known points
/// https://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_between_two_known_points
fn interpolate_between_known(known_xy0: (f64, f64), known_xy1: (f64, f64), x: f64) -> f64 {

    (known_xy0.1 * (known_xy1.0 - x) + known_xy1.1 * (x - known_xy0.0)) / (known_xy1.0 - known_xy0.0)

}

#[derive(Debug, Copy, Clone)]
struct CorPoint {
    trace_n: u32,
    time_seconds: f64,
    easting: f64,
    northing: f64,
    altitude: f64,
}

impl CorPoint {

    fn xyz(&self) -> (f64, f64, f64) {
        (self.easting, self.northing, self.altitude)
    }

    fn txyz(&self) -> [f64; 4] {
        [self.time_seconds, self.easting, self.northing, self.altitude]
    }
}

fn load_cor(filepath: &Path, utm_zone: u8) -> Result<GPRLocation, Box<dyn Error>> {

    let content = std::fs::read_to_string(filepath)?;

    let mut points: Vec<CorPoint>  = Vec::new();

    for line in content.lines() {
        
        let data: Vec<&str> = line.splitn(10, "\t").collect();

        if data.len() < 10 {
            continue
        };

        let mut latitude: f64 = data[3].parse()?;
        let mut longitude: f64 = data[5].parse()?;

        if data[4].trim() == "S" {
            latitude *= -1.;
        };

        if data[6].trim() == "W" {
            longitude *= -1.;
        };

        let (northing, easting, _) = utm::to_utm_wgs84(latitude, longitude, utm_zone);

        let datetime = chrono::DateTime::parse_from_rfc3339(&format!("{}T{}+00:00", data[1], data[2]))?.timestamp() as f64;


        points.push(CorPoint{
            trace_n: (data[0].parse::<i64>()? - 1) as u32,  // The data is 1-indexed
            time_seconds: datetime,
            easting,
            northing,
            altitude: data[7].parse()?,
        });
    };

    Ok(GPRLocation{cor_points: points})
}

struct GPR {
    pub data: ndarray::Array2<f32>,
    pub location: GPRLocation,
    pub metadata: GPRMeta,
}


fn load_rd3(filepath: &Path, height: u32) -> Result<Array2<f32>, Box<dyn std::error::Error>> {

    let bytes = std::fs::read(filepath)?;

    let mut data: Vec<f32> = Vec::new();

    for byte_pair in bytes.chunks_exact(2) {
        let short = i16::from_le_bytes([byte_pair[0], byte_pair[1]]);
        data.push(short as f32);
    };

    let width: usize = data.len() / (height as usize);

    Ok(ndarray::Array2::from_shape_vec((width, height as usize), data).unwrap().reversed_axes())
}


impl GPR {

    fn subset(&self, min_trace: Option<u32>, max_trace: Option<u32>, min_sample: Option<u32>, max_sample: Option<u32>) -> GPR {

        let min_trace_ = match min_trace {Some(x) => x, None => 0};
        let max_trace_ = match max_trace {Some(x) => x, None => self.width() as u32};
        let min_sample_ = match min_sample {Some(x) => x, None => 0};
        let max_sample_ = match max_sample {Some(x) => x, None => self.height() as u32};

        let data_subset = self.data.slice(ndarray::s![min_sample_ as isize..max_sample_ as isize, min_trace_ as isize..max_trace_ as isize]).to_owned();


        let location_subset = GPRLocation{cor_points: self.location.cor_points[min_trace_ as usize..max_trace_ as usize].to_owned()};

        let mut metadata = self.metadata.clone();

        metadata.last_trace = max_trace_;
        metadata.time_window = metadata.time_window * ((max_sample_ - min_sample_) as f32 / metadata.samples as f32);
        metadata.samples = max_sample_ - min_sample_;


        GPR{data: data_subset, location: location_subset, metadata}

    }

    fn vertical_resolution_ns(&self) -> f32 {
        self.metadata.time_window / self.metadata.samples as f32
    }

    fn from_meta_and_loc(location: GPRLocation, metadata: GPRMeta) -> Result<GPR, Box<dyn Error>> {

        let data = load_rd3(&metadata.rd3_filepath, metadata.samples)?;

        let location_data = match data.shape()[1] == location.cor_points.len() {
            true => location,
            false => location.range_fill(0, data.shape()[1] as u32)

        };

        Ok(GPR {data, location: location_data, metadata})
    }


    fn render(&self, filepath: &Path) -> Result<(), Box<dyn Error>> {

        //let minval: f32 = data.quantile_mut(noisy_float::NoisyFloat::new(0.01_f64), &ndarray_stats::interpolate::Nearest).unwrap();
        //
        let vals = self.data.iter().map(|f| f.to_owned()).collect::<Vec<f32>>();

        let mut data = Array1::from_vec(vals);
        //
        
        
        //let minval: f32 = data.quantile_axis_skipnan_mut(Axis(0), noisy_float::NoisyFloat::new(0.01_f64), &ndarray_stats::interpolate::Nearest).unwrap().first().unwrap().to_owned();
        //let maxval: f32 = data.quantile_axis_skipnan_mut(Axis(0), noisy_float::NoisyFloat::new(0.99_f64), &ndarray_stats::interpolate::Nearest).unwrap().first().unwrap().to_owned();
        let min_max = quantiles(&data, &[0.01, 0.99]);
        //let data: Array2<u8> = (255.0 * (&self.data - minval) / (maxval - minval)).mapv(|elem| elem as u8);

        let mut image = GrayImage::new(self.width() as u32, self.height() as u32);

        let mut vals = image.mut_ndarray2();

        let logit99 = (0.99_f32 / (1.0_f32 - 0.99_f32)).log(std::f32::consts::E);

        // Scale the values to logit and convert them to u8
        vals.assign(&self.data.mapv(|f| {
            (
                255.0 * {
            
                let val_norm = ((f - min_max[0]) / (min_max[1] - min_max[0])).clamp(0.0, 1.0);

                0.5 + (val_norm / (1.0_f32 - val_norm)).log(std::f32::consts::E) / logit99
                }

            ) as u8

        }));
        
        //vals.assign(&data.mapv(|elem| elem as u8));


        image.save(filepath)?;

        Ok(())
    }

    fn zero_corr(&mut self, threshold_multiplier: Option<f32>) {

        let mean_trace = self.data.mean_axis(Axis(1)).unwrap();

        let threshold = 0.5 * mean_trace.std(1.0) * threshold_multiplier.unwrap_or(1.0);

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


        self.data = self.data.slice_axis(Axis(0), Slice::new(first_rise, None, 1)).to_owned();
        self.data -= mean_silent_val;

        self.metadata.time_window = self.metadata.time_window * (self.height() as f32 / self.metadata.samples as f32);
        self.metadata.samples = self.height() as u32;
    }

    fn dewow(&mut self, window: u32)  {

        let height = self.height() as u32;

        for i in (0..(height - window)).step_by(window as usize) {

            let mut view = self.data.slice_axis_mut(Axis(0), ndarray::Slice::new(i as isize, Some((i + window) as isize), 1_isize));

            view -= view.mean().unwrap();
        };
    }

    fn normalize_horizontal_magnitudes(&mut self) {
        if let Some(mean) = self.data.mean_axis(Axis(0)) {
            self.data -= &mean;
        };
    }

    fn auto_gain(&mut self, n_bins: usize) {
        

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
    }

    fn gain(&mut self, linear: f32) {

        for i in 0..self.height() as isize {

            let mut view = self.data.slice_axis_mut(Axis(0), Slice::new(i, Some(i + 1), 1_isize));
view *= (i as f32) * linear;
        };
    }

    fn make_equidistant(&mut self) {

        let velocities = self.location.velocities().mapv(|v| v as f32);

        let normal_velocity = quantiles(&velocities, &[0.5])[0];

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

        self.data = self.data.slice_axis(Axis(1), Slice::new(0, Some(nominal_data_width as isize), 1)).to_owned();
        self.metadata.last_trace = nominal_data_width as u32;
        self.location = GPRLocation{cor_points: new_coords};
    }

    fn kirchoff_migration2d(&mut self, velocity_m_per_ns: f32) {

        let x_coords = self.location.distances().mapv(|v| v as f32);

        let x_diff = x_coords[1] - x_coords[0];

        let mut z_coords = Array1::from_iter(self.location.cor_points.iter().map(|point| point.altitude as f32));
        z_coords -= *z_coords.max().unwrap();
        z_coords *= -1.0;

        // The vertical resolution in ns
        let t_diff = self.vertical_resolution_ns();

        // The vertical resolution in m
        let z_diff = t_diff * velocity_m_per_ns;

        // The maximum depth at which to measure
        let max_depth = (self.height() as f32 * z_diff * 0.5) * 0.9 + z_coords.max().unwrap();

        // The minimum logical resolution, assuming just one wavelength
        let logical_res = (self.metadata.antenna_mhz / velocity_m_per_ns) * (1e-9 * 1e6);

        let old_data = self.data.iter().collect::<Vec<&f32>>();

        let old_height = self.height();
        let new_height = (max_depth / z_diff).ceil() as usize;
        //let mut output: Vec<f32> = (0..(self.width() * new_height)).map(|_| 0_f32).collect();
        let width = self.width();

        let output: Vec<f32> = (0..(self.width() * new_height)).into_par_iter().map(|sample_idx| {
            let row = sample_idx / width;
            let trace_n = sample_idx - (row * width);

            let trace_x = x_coords[trace_n];
            let trace_top_z = z_coords[trace_n];

            let sample_z = row as f32 * z_diff;

            if sample_z < trace_top_z {
                return 0.
            };


            // The expected two-way time of the sample (assuming it is straight down)
            let t_0 = 2. * (sample_z - trace_top_z) / velocity_m_per_ns;
            //let t_0 = t_diff * row as f32;
            let t_0_px = (t_0 / t_diff) as usize;


            // If the expected two-way time is larger than the time window, there's no use in
            // continuing
            if t_0 > self.metadata.time_window {
                return 0.
            };

            // Derive the Fresnel zone
            // This is the radius in which the sample may be affected horizontally
            // Pérez-Gracia et al. (2008) Horizontal resolution in a non-destructive
            // shallow GPR survey: An experimental evaluation. NDT & E International,
            // 41(8): 611–620. doi:10.1016/j.ndteint.2008.06.002
            let fresnel_radius = 0.5 * (logical_res * 2. * (sample_z - trace_top_z)).sqrt();

            // Derive the radius in pixel space
            let fresnel_width = (fresnel_radius / x_diff).round() as usize;

            // If the fresnel width is zero pixels, no neighbors will change the sample. Therefore, just
            // take the old value and put it in the topographically correct place.
            if fresnel_width == 0 {
                if row < old_height {
                    return old_data[(t_0_px * width) + trace_n].to_owned();
                };
            };

            // Derive all of the neighboring columns that may affect the sample
            let min_neighbor = if fresnel_width < trace_n {trace_n - fresnel_width} else {trace_n};
            let max_neighbor = (trace_n + fresnel_width).clamp(0, width);

            let mut ampl = [0_f32; 512];
            let mut n_ampl = 0;

            for neighbor_n in min_neighbor..max_neighbor {

                // If the "neighbor" is the middle sample, no geometric assumptions need to be
                // made.
                if neighbor_n == trace_n {
                    if t_0_px < old_height {
                        ampl[neighbor_n - min_neighbor] = old_data[(t_0_px * width) + trace_n].to_owned();
                        n_ampl += 1;
                    };
                    continue;
                };


                // Get the vertical component of the two-way time to the neighboring sample.
                let t_top = t_0 - 2. * (z_coords[neighbor_n] - trace_top_z) / velocity_m_per_ns;

                // Get the travel time to the sample accounting for the x distance
                let t_x = (t_top.powi(2) + 4. * (x_coords[neighbor_n] - trace_x).powi(2)  / velocity_m_per_ns.powi(2)).sqrt() / t_diff;
                // The sample will be in either the pixel when rounding down or when rounding up
                // ... so these will both be evaluated
                // These values have pixel units, as they are normalized to pixel resolution
                let t_1 = t_x.floor() as usize;
                let mut t_2 = t_x.ceil() as usize;

                if t_2 >= old_height {
                    t_2 = t_1;
                };

                // If the travel times are within the bounds of the data and the pixel displacement is not zero,
                // ... append a weighted amplitude accounting for the displacement distance
                if t_1 < old_height {

                    let weight = match t_1 == t_2 {true => 0_f32, false => ((t_1 as f32 - t_x) / (t_1 as f32 - t_2 as f32)).abs()};
                    ampl[neighbor_n - min_neighbor] = 
                        (x_diff / (2. * std::f32::consts::PI * t_x * velocity_m_per_ns)) // Account for the horizontal distance
                        * (t_top / t_x) // Account for the vertical distance
                        * (1. - weight) * old_data[(t_1 * width) + neighbor_n] + weight *
                        * old_data[(t_2 * width) + neighbor_n] // Account for the neigbour's value
                    ;
                    n_ampl += 1;
                };

            };


            if n_ampl > 0 {
                ampl.iter().sum::<f32>() / n_ampl as f32
            } else {
                0.
            }

        }).collect();

        self.data = Array2::from_shape_vec((new_height, width), output).unwrap();

    }

    fn height(&self) -> usize {
        self.data.shape()[0]
    }
    fn width(&self) -> usize {

        self.data.shape()[1]

    }
}


fn quantiles<'a, I>(values: I, quantiles: &[f32]) -> Vec<f32> where
    I: IntoIterator<Item = &'a f32>, {


    let mut values_arr = Array1::from_iter(values.into_iter().map(|f| f.to_owned()));
    let mut output: Vec<f32> = Vec::new();

    for quantile in quantiles {

        output.push(
            values_arr.quantile_axis_skipnan_mut(Axis(0), noisy_float::NoisyFloat::new(quantile.to_owned() as f64), &ndarray_stats::interpolate::Midpoint).unwrap().first().unwrap().to_owned().to_owned()
        );

    };


    output


}

#[cfg(test)]
mod tests {

    #[test]
    fn test_interpolate_between_known() {

        let known_xy0 = (0_f64, 0_f64);
        let known_xy1 = (5_f64, 10_f64);

        assert_eq!(super::interpolate_between_known(known_xy0, known_xy1, 2.5), 5.0)

    }

    #[test]
    fn test_interpolate_coordinate() {

        let coord0 = (0_f64, 0_f64, 0_f64);
        let time0 = 0_f64;

        let coord1 = (5_f64, 10_f64, 15_f64);
        let time1 = 1_f64;

        assert_eq!(super::interpolate_coordinate(time0, coord0, time1, coord1, 0.5), (2.5, 5.0, 7.5))
    }
    #[test]
    fn test_interpolate_values() {

        let coord0 = vec![0_f64, 0_f64, 0_f64];
        let time0 = 0_f64;

        let coord1 = vec![5_f64, 10_f64, 15_f64];
        let time1 = 1_f64;

        assert_eq!(super::interpolate_values(time0, &coord0, time1, &coord1, 0.5), vec![2.5, 5.0, 7.5])
    }
}
