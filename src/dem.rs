/// Tools to read elevation from Digital Elevation Models (DEMs)
use gdal::raster::ResampleAlg;
use ndarray_stats::QuantileExt;
use smartcore::linalg::basic::arrays::ArrayView1;
use std::error::Error;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};

use crate::coords::{Coord, Crs};

pub fn sample_dem(dem_path: &Path, coords_wgs84: &Vec<Coord>) -> Result<Vec<f32>, String> {
    use std::io::Write;

    let args = vec![
        "-xml",
        "-b",
        "1",
        "-wgs84",
        "-r",
        "bilinear",
        dem_path.to_str().unwrap(),
    ];
    let mut child = std::process::Command::new("gdallocationinfo")
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let mut values = Vec::<String>::new();
    for coord in coords_wgs84 {
        values.push(format!("{} {}", coord.x, coord.y));
    }
    // println!("{values:?}");
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all((values.join("\n") + "\n").as_bytes())
            .unwrap();
    }

    let output = child.wait_with_output().unwrap();
    let parsed = String::from_utf8_lossy(&output.stdout);

    let mut elevations = Vec::<f32>::new();
    for line in parsed.lines().map(|s| s.trim()) {
        if line.contains("<Value>") {
            elevations.push(
                line.replace("<Value>", "")
                    .replace("</Value>", "")
                    .parse()
                    .unwrap(), // .map_err(|e| Err(format!("Error parsing <Value>: {e:?}").to_string()))?,
            );
        } else if line.contains("<Alert>") {
            let error = line.replace("<Alert>", "").replace("</Alert>", "");
            let coord = coords_wgs84[elevations.len()];

            return Err(format!(
                "Error parsing coord (lon: {:.3}, lat: {:.3}): {}",
                coord.x, coord.y, error
            ));
        }
        // println!("{line:?}");
    }

    if elevations.len() != coords_wgs84.len() {
        if !output.stderr.is_empty() {
            return Err(format!(
                "DEM sampling error: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        return Err(format!("Shape error. Length of sampled elevations ({}) does not align with length of coordinates ({})", elevations.len(), coords_wgs84.len()));
    }

    Ok(elevations)
}

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
        .read_as::<f32>(
            (upper_left.0 as isize, upper_left.1 as isize),
            window_size,
            upscaled_window_size,
            Some(ResampleAlg::Bilinear),
        )
        .unwrap()
        .to_array()
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

    use std::{any::Any, path::Path};

    use crate::coords::{Coord, Crs, UtmCrs};

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

    #[test]
    fn test_read_elevations() {
        // let coords = vec![Coord {
        //     y: 77.8252,
        //     x: 17.275,
        // }];

        let coords_elevs = vec![
            (
                Coord {
                    x: 553802.,
                    y: 8639550.,
                },
                Ok(422.0352_f32),
            ),
            (
                Coord { x: 0., y: 8639550. },
                Err("Location is off this file".to_string()),
            ),
        ];
        let working_coords = coords_elevs
            .iter()
            .filter(|(_c, e)| e.is_ok())
            .map(|(c, _e)| *c)
            .collect::<Vec<Coord>>();
        let all_coords = coords_elevs
            .iter()
            .map(|(c, _e)| *c)
            .collect::<Vec<Coord>>();
        let crs = Crs::Utm(UtmCrs {
            zone: 33,
            north: true,
        });

        let dem_path = Path::new("assets/test_dem_dtm20_mettebreen.tif");

        println!("Sampling DEM");
        let coords_wgs84 = crate::coords::to_wgs84(&working_coords, &crs).unwrap();
        super::sample_dem(dem_path, &coords_wgs84).unwrap();

        let coords_wgs84 = crate::coords::to_wgs84(&all_coords, &crs).unwrap();
        super::sample_dem(dem_path, &coords_wgs84).err().unwrap();

        for (coord, expected) in coords_elevs {
            let coord_wgs84 = crate::coords::to_wgs84(&[coord], &crs).unwrap();

            let result = super::sample_dem(dem_path, &coord_wgs84);

            if let Ok(expected_elevation) = expected {
                assert_eq!(Ok(vec![expected_elevation]), result);
            } else if let Err(expected_err_str) = expected {
                if let Err(err_str) = result {
                    assert!(
                        err_str.contains(&expected_err_str),
                        "{} != {}",
                        err_str,
                        expected_err_str
                    );
                } else {
                    panic!("Should have been an error but wasn't: {result:?}");
                }
            }
        }
        let wrong_path = Path::new("assets/test_dem_dtm20_mettebreen.tiffffff");
        assert!(super::sample_dem(wrong_path, &coords_wgs84)
            .err()
            .unwrap()
            .contains("No such file or directory"));
    }
}
