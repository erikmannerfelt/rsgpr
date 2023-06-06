/// Tools to read elevation from Digital Elevation Models (DEMs)
use gdal::raster::ResampleAlg;
use ndarray_stats::QuantileExt;
use std::error::Error;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};

/// Read elevation values from a DEM from the specified coordinates
/// # Arguments
///  - `dem_path`: The filepath to the DEM
///  - `xy_coords`: Coordinates in the shape of [[east, north], [east, ...] ...]
/// # Returns
/// Elevation values of the given coordinates if the process succeeded.
/// TODO: Add nodata handling
/// TODO: Add a test dataset and associated tests
pub fn read_elevations(
    dem_path: &Path,
    xy_coords: Array2<f64>,
) -> Result<Array1<f32>, Box<dyn Error>> {
    let raster = gdal::Dataset::open(dem_path)?;

    // The DEM is almost always in band 1, so this is assumed
    let band = raster.rasterband(1)?;

    // The transform allows transformations between geographic and pixel space
    let transform = raster.geo_transform()?;

    // Initialize an empty array to push transformed coordinates to.
    // TODO: Try to pre-initialize the full-size array
    let mut px_coords = Array2::<f32>::from_elem((0, 2), 0.);

    // Loop over each coordinate and project them to pixel space
    // TODO: Try to do this in one closure for simplicity and easier debugging.
    for coord in xy_coords.rows() {
        px_coords.push_row(project_geo_to_px(coord[0], coord[1], &transform).view())?;
    }

    // Measure the median step size in pixels of the coordinate sequence.
    // This will decide whether upscaling of the DEM should be done.
    // If no upscaling is done on a coarse DEM, the entire coordinate sequence might be in very few DEM pixels,
    // which gives a blocky appearance to the elevation track.
    let med_diff = {
        // First, measure the differences pixels (read the coordinate distances in pixels)
        let mut px_diffs = Vec::<f32>::new();
        for i in 1..px_coords.shape()[0] {
            px_diffs.push(
                (px_coords.get((i, 0)).unwrap() - px_coords.get((i - 1, 0)).unwrap()).powi(2)
                    + ((px_coords.get((i, 0)).unwrap() - px_coords.get((i - 1, 0)).unwrap())
                        .powi(2))
                    .sqrt(),
            );
        }
        // Then, in a slightly convoluted syntax, measure the median pixel difference
        Array1::from_vec(px_diffs)
            .quantile_axis_skipnan_mut(
                Axis(0),
                noisy_float::NoisyFloat::new(0.5_f64),
                &ndarray_stats::interpolate::Midpoint,
            )
            .unwrap()
            .first()
            .unwrap()
            .to_owned()
    };

    // Here, the potential DEM upscaling is determined. It is clamped from 1 (no scaling) to 10
    let upscale = ((1. / med_diff) as usize).clamp(1, 10);

    // Get the pixel bounds of the coordinate track to crop the DEM
    let min_xy = px_coords.map_axis(Axis(0), |a| a.min().unwrap().to_owned() as usize);
    let max_xy = px_coords.map_axis(Axis(0), |a| a.max().unwrap().to_owned().ceil() as usize);

    let upper_left = (min_xy[0], min_xy[1]);

    // The window size is the "(width, height)" of the part to extract from the DEM
    let window_size = (max_xy[0] - min_xy[0] + 1, max_xy[1] - min_xy[1] + 1);

    // The upscaled window size is the expected size of the read DEM, which may be larger
    // than the original window size in the case of upscaling. If upscale == 1, this is identical
    // It basically tells GDAL: "Read this window of `window_size` and (potentially) upscale it to `upscaled_window_size`"
    let upscaled_window_size = (window_size.0 * upscale, window_size.1 * upscale);

    // Read the DEM elevations as a 2D array
    let dem = band
        .read_as_array::<f32>(
            (upper_left.0 as isize, upper_left.1 as isize),
            window_size,
            upscaled_window_size,
            Some(ResampleAlg::Bilinear),
        )
        .unwrap();

    // Initialize an output elevation vec, to be filled below
    let mut elevations = Vec::<f32>::new();

    // Transform the coordinates from normal pixel space to upscaled pixel space, and then read from the DEM
    for coord in px_coords.rows() {
        let y_upscaled = ((coord[1] - upper_left.1 as f32) * upscale as f32) as usize;
        let x_upscaled = ((coord[0] - upper_left.0 as f32) * upscale as f32) as usize;

        elevations.push(dem.get((y_upscaled, x_upscaled)).unwrap().to_owned());
    }

    Ok(Array1::from_vec(elevations))
}

/// Transform a geographical coordinate to the pixel-space using a transform
///
/// The affine transform object consists of six components:
/// [x_offset, x_resolution, unsupported_rotation, y_offset, unsupported_rotation, -y_resolution]
///
/// The pixel space is assumed to be zero at the top left.
///
/// # Arguments
///  - `easting`: The easting coordinate of the point
///  - `northing`: The northing coordinate of the point
///  - `transform`: The affine transform.
/// # Returns
/// The projected (x, y) coordinate where x is horizontal and y is vertical
/// # Panics
/// If rotational components exist in the affine transform. These are extremely unusual
/// and are also not supported in other large libraries.
fn project_geo_to_px(easting: f64, northing: f64, transform: &[f64; 6]) -> Array1<f32> {
    if (transform[2] != 0.) | (transform[4] != 0.) {
        panic!(
            "DEM transform has unsupported rotational components! {:?}",
            transform
        );
    };

    let xy: Vec<f32> = vec![
        ((easting - transform[0]) / transform[1]) as f32,
        ((transform[3] - northing) / -transform[5]) as f32,
    ];

    Array1::from_vec(xy)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_project_geo_to_px() {
        use super::project_geo_to_px;

        let transform0: [f64; 6] = [5000., 5., 0., 10000., 0., -10.];

        let point0 = (5000., 5000.);
        assert_eq!(
            project_geo_to_px(point0.0, point0.1, &transform0).into_raw_vec(),
            vec![0., 500.]
        );

        let point1 = (6000., 7500.);
        assert_eq!(
            project_geo_to_px(point1.0, point1.1, &transform0).into_raw_vec(),
            vec![200., 250.]
        );
    }
}
