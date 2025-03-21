/// Functions to handle input and output (I/O) of GPR data files.
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::{gpr, tools};

/// Load and parse a Malå metadata file (.rad)
///
/// # Arguments
/// - `filepath`: The filepath of the input metadata file
/// - `medium_velocity`: The velocity of the portrayed medium to assign the GPR data
///
/// # Returns
/// A gpr::GPRMeta instance.
///
/// # Errors
/// - The file could not be read
/// - The contents could not be parsed correctly
/// - The associated ".rd3" file does not exist.
pub fn load_rad(filepath: &Path, medium_velocity: f32) -> Result<gpr::GPRMeta, Box<dyn Error>> {
    let content = std::fs::read_to_string(filepath)?;

    // Collect all rows into a hashmap, assuming a "KEY:VALUE" structure.
    let data: HashMap<&str, &str> = content.lines().filter_map(|s| s.split_once(':')).collect();

    let rd3_filepath = filepath.with_extension("rd3");
    if !rd3_filepath.is_file() {
        return Err(format!("File not found: {rd3_filepath:?}").into());
    };

    // Extract and parse all required metadata into a new GPRMeta object.
    let antenna = data
        .get("ANTENNAS")
        .ok_or("No 'ANTENNAS' key in metadata")?
        .trim()
        .to_string();

    Ok(gpr::GPRMeta {
        samples: data
            .get("SAMPLES")
            .ok_or("No 'SAMPLES' key in metadata")?
            .trim()
            .parse()?,
        frequency: data
            .get("FREQUENCY")
            .ok_or("No 'FREQUENCY' key in metadata")?
            .trim()
            .parse()?,
        frequency_steps: data
            .get("FREQUENCY STEPS")
            .ok_or("No 'FREQUENCY STEPS' key in metadata")?
            .trim()
            .parse()?,
        time_interval: data
            .get("TIME INTERVAL")
            .ok_or("No 'TIME INTERVAL' key in metadata")?
            .replace(' ', "")
            .parse()?,
        antenna_mhz: antenna.split("MHz").collect::<Vec<&str>>()[0]
            .trim()
            .parse()?,
        antenna,
        antenna_separation: data
            .get("ANTENNA SEPARATION")
            .ok_or("No 'ANTENNA SEPARATION' key in metadata")?
            .trim()
            .parse()?,
        time_window: data
            .get("TIMEWINDOW")
            .ok_or("No 'TIMEWINDOW' key in metadata")?
            .trim()
            .parse()?,
        last_trace: data
            .get("LAST TRACE")
            .ok_or("No 'LAST TRACE' key in metadata")?
            .trim()
            .parse()?,
        rd3_filepath,
        medium_velocity,
    })
}

/// Load and parse a Malå ".cor" location file
///
/// # Arguments
/// - `filepath`: The path to the file to read.
/// - `projected_crs`: Any projected CRS understood by PROJ to project the coordinates into
///
/// # Returns
/// The parsed location points in a GPRLocation object.
///
/// # Errors
/// - The file could not be found/read
/// - `projected_crs` is not understood by PROJ
/// - The contents of the file could not be parsed.
pub fn load_cor(
    filepath: &Path,
    projected_crs: Option<&String>,
) -> Result<gpr::GPRLocation, Box<dyn Error>> {
    let content = std::fs::read_to_string(filepath)?;

    // Create a new empty points vec
    let mut coords = Vec::<crate::coords::Coord>::new();
    let mut points: Vec<gpr::CorPoint> = Vec::new();
    // Loop over the lines of the file and parse CorPoints from it
    for line in content.lines() {
        // Split the line into ten separate columns.
        let data: Vec<&str> = line.splitn(10, '\t').collect();

        // If the line could not be split in ten columns, it is probably wrong.
        if data.len() < 10 {
            continue;
        };

        let mut latitude: f64 = data[3].parse()?;
        let mut longitude: f64 = data[5].parse()?;

        // Invert the sign of the latitude if it's on the southern hemisphere
        if data[4].trim() == "S" {
            latitude *= -1.;
        };

        // Invert the sign of the longitude if it's west of the prime meridian
        if data[6].trim() == "W" {
            longitude *= -1.;
        };

        coords.push(crate::coords::Coord {
            x: longitude,
            y: latitude,
        });

        // Parse the date and time columns into datetime, then convert to seconds after UNIX epoch.
        let datetime =
            chrono::DateTime::parse_from_rfc3339(&format!("{}T{}+00:00", data[1], data[2]))?
                .timestamp() as f64;

        // Coordinates are 0 right now. That's fixed right below
        points.push(gpr::CorPoint {
            trace_n: (data[0].parse::<i64>()? - 1) as u32, // The ".cor"-files are 1-indexed whereas this is 0-indexed
            time_seconds: datetime,
            easting: 0.,
            northing: 0.,
            altitude: data[7].parse()?,
        });
    }

    if points.is_empty() {
        return Err(format!("Could not parse location data from: {:?}", filepath).into());
    }

    let projected_crs = match projected_crs {
        Some(s) => s.to_string(),
        None => crate::coords::UtmCrs::optimal_crs(&coords[0]).to_epsg_str(),
    };
    for (i, coord) in crate::coords::from_wgs84(
        &coords,
        &crate::coords::Crs::from_user_input(&projected_crs)?,
    )?
    .iter()
    .enumerate()
    {
        points[i].easting = coord.x;
        points[i].northing = coord.y;
    }

    if !points.is_empty() {
        Ok(gpr::GPRLocation {
            cor_points: points,
            correction: gpr::LocationCorrection::None,
            crs: projected_crs.to_string(),
        })
    } else {
        Err(format!("Could not parse location data from: {:?}", filepath).into())
    }
}

/// Load a Malå data (.rd3) file
///
/// # Arguments
/// - `filepath`: The path of the file to read.
/// - `height`: The expected height of the data. The width is parsed automatically.
///
/// # Returns
/// A 2D array of 32 bit floating point values in the shape (height, width).
///
/// # Errors
/// - The file cannot be read
/// - The length does not work with the expected shape
pub fn load_rd3(filepath: &Path, height: usize) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(filepath)?;

    let mut data: Vec<f32> = Vec::new();

    // It's 50V (50000mV) in RGPR https://github.com/emanuelhuber/RGPR/blob/d78ff7745c83488111f9e63047680a30da8f825d/R/readMala.R#L8
    let bits_to_millivolt = 50000. / i16::MAX as f32;

    // The values are read as 16 bit little endian signed integers, and are converted to millivolts
    for byte_pair in bytes.chunks_exact(2) {
        let value = i16::from_le_bytes([byte_pair[0], byte_pair[1]]);
        data.push(value as f32 * bits_to_millivolt);
    }

    let width: usize = data.len() / height;

    Ok(ndarray::Array2::from_shape_vec((width, height), data)?.reversed_axes())
}

/// Export a GPR profile and its metadata to a NetCDF (".nc") file.
///
/// It will overwrite any file that already exists with the same filename.
///
/// # Arguments
/// - `gpr`: The GPR object to export
/// - `nc_filepath`: The filepath of the output NetCDF file
///
/// # Errors
/// - If the file already exists and cannot be removed.
/// - If a dimension, attribute or variable could not be created in the NetCDF file
/// - If data could not be written to the file
pub fn export_netcdf(gpr: &gpr::GPR, nc_filepath: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Remove any previously existing file. If this is not added, netcdf will throw a useless
    // error!
    if nc_filepath.is_file() {
        std::fs::remove_file(nc_filepath)?;
    };
    // Create a new NetCDF file
    let mut file = netcdf::create(nc_filepath)?;

    // Add the x/y dimensions for the data
    file.add_dimension("x", gpr.width())?;
    file.add_dimension("y", gpr.height())?;

    // Add global attributes to the file
    file.add_attribute(
        "start-datetime",
        chrono::DateTime::from_timestamp(gpr.location.cor_points[0].time_seconds as i64, 0)
            .unwrap()
            .to_rfc3339(),
    )?;
    file.add_attribute(
        "stop-datetime",
        chrono::DateTime::from_timestamp(
            gpr.location.cor_points[gpr.location.cor_points.len() - 1].time_seconds as i64,
            0,
        )
        .unwrap()
        .to_rfc3339(),
    )?;
    file.add_attribute("processing-datetime", chrono::Local::now().to_rfc3339())?;
    file.add_attribute("antenna", gpr.metadata.antenna.clone())?;
    file.add_attribute("antenna-separation", gpr.metadata.antenna_separation)?;
    file.add_attribute("frequency-steps", gpr.metadata.frequency_steps)?;
    file.add_attribute("vertical-sampling-frequency", gpr.metadata.frequency)?;
    if gpr.metadata.time_interval.is_finite() {
        file.add_attribute("time-interval", gpr.metadata.time_interval)?;
    }

    file.add_attribute("processing-log", gpr.log.join("\n"))?;
    file.add_attribute(
        "original-filename",
        gpr.metadata
            .rd3_filepath
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    )?;

    file.add_attribute("medium-velocity", gpr.metadata.medium_velocity)?;
    file.add_attribute("medium-velocity-unit", "m / ns")?;

    file.add_attribute(
        "elevation-correction",
        match gpr.location.correction.clone() {
            gpr::LocationCorrection::None => "None".to_string(),
            gpr::LocationCorrection::Dem(fp) => format!(
                "DEM-corrected: {:?}",
                fp.as_path().file_name().unwrap().to_str().unwrap()
            ),
        },
    )?;

    file.add_attribute("crs", gpr.location.crs.clone())?;
    let distance_vec = gpr.location.distances().into_raw_vec();
    file.add_attribute("total-distance", distance_vec[distance_vec.len() - 1])?;
    file.add_attribute("total-distance-unit", "m")?;

    file.add_attribute(
        "program-version",
        format!(
            "{} version {}, © {}",
            crate::PROGRAM_NAME,
            crate::PROGRAM_VERSION,
            crate::PROGRAM_AUTHORS
        ),
    )?;

    // Add the data to the file
    {
        let mut data = file.add_variable::<f32>("data", &["y", "x"])?;
        data.set_compression(5, true)?;
        data.set_chunking(&[1024, 1024])?;

        data.put_values(
            &gpr.data.iter().map(|v| v.to_owned()).collect::<Vec<f32>>(),
            ..,
        )?;

        // The default coordinates are distance for x and return time for y
        data.put_attribute("coordinates", "distance return-time")?;
        data.put_attribute("unit", "mV")?;
    }

    if let Some(topo_data) = &gpr.topo_data {
        // Add the data to the file
        let height = topo_data.shape()[0];
        file.add_dimension("y2", height)?;
        let mut data2 = file.add_variable::<f32>("data_topographically_corrected", &["y2", "x"])?;
        data2.set_compression(5, true)?;
        data2.set_chunking(&[1024, 1024])?;

        data2.put_values(
            &topo_data.iter().map(|v| v.to_owned()).collect::<Vec<f32>>(),
            ..,
        )?;

        // The default coordinates are distance for x and return time for y
        data2.put_attribute("coordinates", "distance elevation")?;
        data2.put_attribute("unit", "mV")?;
    };

    // Add the distance variable to the x dimension
    let mut ds = file.add_variable::<f32>("distance", &["x"])?;
    ds.put_values(&distance_vec, ..)?;
    ds.put_attribute("unit", "m")?;

    // Add the time variable to the x dimension
    let mut time = file.add_variable::<f64>("time", &["x"])?;
    time.put_values(
        &gpr.location
            .cor_points
            .iter()
            .map(|point| point.time_seconds)
            .collect::<Vec<f64>>(),
        ..,
    )?;
    time.put_attribute("unit", "s")?;

    // Add the easting variable to the x dimension
    let mut easting = file.add_variable::<f64>("easting", &["x"])?;
    easting.put_values(
        &gpr.location
            .cor_points
            .iter()
            .map(|point| point.easting)
            .collect::<Vec<f64>>(),
        ..,
    )?;
    easting.put_attribute("unit", "m")?;

    // Add the northing variable to the x dimension
    let mut northing = file.add_variable::<f64>("northing", &["x"])?;
    northing.put_values(
        &gpr.location
            .cor_points
            .iter()
            .map(|point| point.northing)
            .collect::<Vec<f64>>(),
        ..,
    )?;
    northing.put_attribute("unit", "m")?;

    // Add the elevation variable to the x dimension
    let mut elevation = file.add_variable::<f64>("elevation", &["x"])?;
    elevation.put_values(
        &gpr.location
            .cor_points
            .iter()
            .map(|point| point.altitude)
            .collect::<Vec<f64>>(),
        ..,
    )?;
    elevation.put_attribute("unit", "m a.s.l.")?;

    // Add the two-way return time variable to the y dimension
    let return_time_arr = (Array1::range(
        0_f32,
        gpr.metadata.time_window,
        gpr.vertical_resolution_ns(),
    )
    .slice_axis(
        ndarray::Axis(0),
        ndarray::Slice::new(0, Some(gpr.height() as isize), 1),
    ))
    .to_owned()
    .into_raw_vec();
    let mut return_time = file.add_variable::<f32>("return-time", &["y"])?;
    return_time.put_values(&return_time_arr, ..)?;
    return_time.put_attribute("unit", "ns")?;

    // Add the depth variable to the y dimension
    let mut depth = file.add_variable::<f32>("depth", &["y"])?;
    depth.put_values(&gpr.depths().into_raw_vec(), ..)?;

    depth.put_attribute("unit", "m")?;

    Ok(())
}

/// Render an image of the processed GPR data.
///
/// # Arguments
/// - `gpr`: The GPR data to render
/// - `filepath`: The output filepath of the image
///
/// # Errors
/// - The file could not be written.
/// - The extension is not understood.
pub fn render_jpg(gpr: &gpr::GPR, filepath: &Path) -> Result<(), Box<dyn Error>> {
    let data_to_render = match &gpr.topo_data {
        Some(d) => d,
        None => &gpr.data,
    };

    let data = data_to_render.iter().collect::<Vec<&f32>>();

    // Get quick and dirty quantiles by only looking at a 10th of the data
    let q = tools::quantiles(&data, &[0.01, 0.99], Some(10));
    let mut minval = q[0];
    let maxval = q[1];

    // If unphase has been run, there are no (valid) negative numbers, so it should instead start at 0
    let unphase_run = gpr.log.iter().any(|s| s.contains("unphase"));
    if unphase_run {
        minval = &0.;
    };

    //let logit99 = (0.99_f32 / (1.0_f32 - 0.99_f32)).log(std::f32::consts::E);

    // Render the pixels into a grayscale image
    let pixels: Vec<u8> = data
        .into_par_iter()
        .map(|f| {
            (255.0 * {
                let mut val_norm = ((f - minval) / (maxval - minval)).clamp(0.0, 1.0);
                if unphase_run {
                    val_norm = 0.5 * val_norm + 0.5;
                };

                //0.5 + (val_norm / (1.0_f32 - val_norm)).log(std::f32::consts::E) / logit99
                val_norm
            }) as u8
        })
        .collect();

    image::save_buffer(
        filepath,
        &pixels,
        data_to_render.shape()[1] as u32,
        data_to_render.shape()[0] as u32,
        image::ColorType::L8,
    )?;

    Ok(())
}

/// Export a "track" file.
///
/// It has its own associated function because the logic may happen in two different places in the
/// main() function.
///
/// # Arguments
/// - `gpr_locations`: The GPRLocation object to export
/// - `potential_track_path`: The output path of the track file or a directory (if provided)
/// - `output_filepath`: The output filepath to derive a track filepath from in case `potential_track_path` was not provided.
/// - `verbose`: Print progress?
///
/// # Returns
/// The exit code of the function
pub fn export_locations(
    gpr_locations: &gpr::GPRLocation,
    potential_track_path: Option<&PathBuf>,
    output_filepath: &Path,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    // Determine the output filepath. If one was given, use that. If none was given, use the
    // parent and file stem + "_track.csv" of the output filepath. If a directory was given,
    // use the directory + the file stem of the output filepath + "_track.csv".
    let track_path: PathBuf = match potential_track_path {
        // Here is in case a filepath or directory was given
        Some(fp) => match fp.is_dir() {
            // In case the filepath points to a directory
            true => fp
                .join(
                    output_filepath
                        .file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        + "_track",
                )
                .with_extension("csv"),
            // In case it is not a directory (and thereby assumed to be a normal filepath)
            false => fp.clone(),
        },
        // Here is if no filepath was given
        None => output_filepath
            .with_file_name(
                output_filepath
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
                    + "_track",
            )
            .with_extension("csv"),
    };
    if verbose {
        println!("Exporting track to {:?}", track_path);
    };

    Ok(gpr_locations.to_csv(&track_path)?)
}

#[cfg(test)]
mod tests {

    use super::{load_cor, load_rad};

    /// Fake some data. One point is in the northern hemisphere and one is in the southern
    fn fake_cor_text() -> String {
        [
            "1\t2022-01-01\t00:00:01\t78.0\tN\t16.0\tE\t100.0\tM\t1",
            "10\t2022-01-01\t00:01:00\t78.0\tS\t16.0\tW\t100.0\tM\t1",
            "11\t2022-01", // This simulates an unfinished line that should be skipped
        ]
        .join("\r\n")
    }

    #[test]
    fn test_load_cor() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cor_path = temp_dir.path().join("hello.cor");

        std::fs::write(&cor_path, fake_cor_text()).unwrap();

        // Load it and "convert" (or rather don't convert) the CRS to WGS84
        let locations = load_cor(&cor_path, Some(&"EPSG:4326".to_string())).unwrap();

        // Check that the trace number is now zero based, and that the other fields were read
        // correctly
        assert_eq!(locations.cor_points[0].trace_n, 0);
        assert_eq!(locations.cor_points[0].easting, 16.0);
        assert_eq!(locations.cor_points[0].northing, 78.0);
        assert_eq!(locations.cor_points[0].altitude, 100.0);
        assert_eq!(
            locations.cor_points[0].time_seconds,
            chrono::DateTime::parse_from_rfc3339("2022-01-01T00:00:01+00:00")
                .unwrap()
                .timestamp() as f64
        );

        // Check that the second point has inverted signs (since it's 78*S, 16*W)
        assert_eq!(locations.cor_points[1].easting, -16.0);
        assert_eq!(locations.cor_points[1].northing, -78.0);

        // Load the data again but convert it to WGS84 UTM Zone 33N
        let locations = load_cor(&cor_path, Some(&"EPSG:32633".to_string())).unwrap();

        // Check that the coordinates are within reason
        assert!(
            (locations.cor_points[0].easting > 500_000_f64)
                & (locations.cor_points[0].easting < 600_000_f64)
        );
        assert!(
            (locations.cor_points[0].northing > 8_000_000_f64)
                & (locations.cor_points[0].easting < 9_000_000_f64)
        );
        assert!(
            (locations.cor_points[1].northing < 0_f64)
                & (locations.cor_points[1].northing > -9_000_000_f64)
        );
    }

    #[test]
    fn test_load_rad() {
        // Fake a .rad metadata file
        let temp_dir = tempfile::tempdir().unwrap();
        let rad_path = temp_dir.path().join("hello.rad");
        let rd3_path = rad_path.with_extension("rd3");
        let rad_text = [
            "SAMPLES:2024",
            "FREQUENCY:                 1000.",
            "FREQUENCY STEPS: 20",
            "TIME INTERVAL: 0.1",
            "ANTENNAS: 100 MHz unshielded",
            "ANTENNA SEPARATION: 0.5",
            "TIMEWINDOW:2000",
            "LAST TRACE: 40",
        ]
        .join("\r\n");

        std::fs::write(&rad_path, rad_text).unwrap();

        // The rd3 file needs to exist, but it doesn't need to contain anything
        std::fs::write(&rd3_path, "").unwrap();

        let gpr_meta = load_rad(&rad_path, 0.1).unwrap();

        // Check that the correct values were parsed
        assert_eq!(gpr_meta.samples, 2024);
        assert_eq!(gpr_meta.frequency, 1000.);
        assert_eq!(gpr_meta.frequency_steps, 20);
        assert_eq!(gpr_meta.time_interval, 0.1);
        assert_eq!(gpr_meta.antenna_mhz, 100.);
        assert_eq!(gpr_meta.antenna_separation, 0.5);
        assert_eq!(gpr_meta.time_window, 2000.);
        assert_eq!(gpr_meta.last_trace, 40);
        assert_eq!(gpr_meta.rd3_filepath, rd3_path);
    }

    #[test]
    fn test_export_locations() {
        use super::export_locations;
        let temp_dir = tempfile::tempdir().unwrap();
        let cor_path = temp_dir.path().join("hello.cor");

        std::fs::write(&cor_path, fake_cor_text()).unwrap();

        // Load it and "convert" (or rather don't convert) the CRS to WGS84
        let locations = load_cor(&cor_path, Some(&"EPSG:4326".to_string())).unwrap();

        let out_dir = temp_dir.path().to_path_buf();
        let out_path = out_dir.join("track.csv");

        // The GPR filepath will be used in case no explicit filepath was given
        let dummy_gpr_output_path = out_dir.join("gpr.nc");
        let expected_default_path = out_dir.join("gpr_track.csv");

        for alternative in [
            Some(&out_path), // In case of a target filepath
            Some(&out_dir),  // In case of a target directory
            None,            // In case of a default name beside the GPR file
        ] {
            export_locations(&locations, alternative, &dummy_gpr_output_path, false).unwrap();

            let expected_path = match alternative {
                Some(p) if p == &out_path => &out_path,
                _ => &expected_default_path,
            };
            assert!(expected_path.is_file());

            let content = std::fs::read_to_string(expected_path)
                .unwrap()
                .split("\n")
                .map(|s| s.to_string())
                .collect::<Vec<String>>();

            assert_eq!(content[0], "trace_n,easting,northing,altitude");

            let line0: Vec<&str> = content[1].split(",").collect();

            // The cor file says 1 but rsgpr is zero-indexed, hence 0
            assert_eq!(line0[0], "0");
            assert_eq!(line0[1], "16");
            assert_eq!(line0[2], "78");
            assert_eq!(line0[3], "100");

            let line1: Vec<&str> = content[2].split(",").collect();
            assert_eq!(line1[2], "-78");

            std::fs::remove_file(expected_path).unwrap();
        }
    }
}
