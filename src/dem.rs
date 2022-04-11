
use std::error::Error;
use std::path::Path;
use gdal::raster::ResampleAlg;
use ndarray_stats::QuantileExt;

use ndarray::{Array2, Array1, Axis};
pub fn read_elevations(dem_path: &Path, xy_coords: Array2<f64>) -> Result<Array1<f32>, Box<dyn Error>> {

    let raster = gdal::Dataset::open(dem_path)?;

    let band = raster.rasterband(1)?;

    let transform = raster.geo_transform()?;

    let mut px_coords = Array2::<f32>::from_elem((0, 2), 0.);

    for coord in xy_coords.rows() {
        px_coords.push_row(project_geo_to_px(coord[0], coord[1], &transform).view()).unwrap();
    };

    let mut px_diffs = Vec::<f32>::new();
    for i in 1..px_coords.shape()[0] {

        px_diffs.push((px_coords.get((i, 0)).unwrap() - px_coords.get((i - 1, 0)).unwrap()).powi(2) + ((px_coords.get((i, 0)).unwrap() - px_coords.get((i - 1, 0)).unwrap()).powi(2)).sqrt());
    };

    let med_diff = Array1::from_vec(px_diffs).quantile_axis_skipnan_mut(Axis(0), noisy_float::NoisyFloat::new(0.5_f64), &ndarray_stats::interpolate::Midpoint).unwrap().first().unwrap().to_owned();

    let upscale = ((1. / med_diff) as usize).clamp(1, 10);

    let min_xy = px_coords.map_axis(Axis(0), |a| a.min().unwrap().to_owned() as usize);
    let max_xy = px_coords.map_axis(Axis(0), |a| a.max().unwrap().to_owned().ceil() as usize);

    let upper_left = (min_xy[0], min_xy[1]);

    let window_size = (max_xy[0] - min_xy[0] + 1, max_xy[1] - min_xy[1] + 1);

    let upscaled_window_size = (window_size.0 * upscale, window_size.1 * upscale);

    let dem = band.read_as_array::<f32>((upper_left.0 as isize, upper_left.1 as isize), window_size, upscaled_window_size, Some(ResampleAlg::Bilinear)).unwrap();

    let mut elevations = Vec::<f32>::new();

    for coord in px_coords.rows() {
        let y_upscaled = ((coord[1] - upper_left.1 as f32) * upscale as f32) as usize;
        let x_upscaled = ((coord[0] - upper_left.0 as f32) * upscale as f32) as usize;

        elevations.push(dem.get((y_upscaled, x_upscaled)).unwrap().to_owned());
    };



    Ok(Array1::from_vec(elevations))
}

fn project_geo_to_px(easting: f64, northing: f64, transform: &[f64; 6]) -> Array1<f32> {

    if (transform[2] != 0.) | (transform[4] != 0.) {
        panic!("DEM transform has unsupported rotational components! {:?}", transform);
    };

    let xy: Vec<f32> = vec![((easting - transform[0]) / transform[1]) as f32, ((transform[3] - northing) / -transform[5]) as f32];

    Array1::from_vec(xy)
}
