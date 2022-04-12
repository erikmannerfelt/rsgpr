/// Functions to handle input and output (I/O) of files.
use std::path::Path;
use std::error::Error;
use std::collections::HashMap;
use ndarray::{Array2,  Array1};
use rayon::prelude::*;

use crate::gpr;

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
    let data: HashMap<&str, &str> = content.lines().filter_map(|s| s.split_once(":")).collect();

    let rd3_filepath = filepath.with_extension("rd3");
    if !rd3_filepath.is_file() {
        return Err(format!("File not found: {:?}", rd3_filepath).into());
    };

    // Extract and parse all required metadata into a new GPRMeta object.
    let antenna = data.get("ANTENNAS").ok_or("No 'ANTENNAS' key in metadata")?.trim().to_string();

    Ok(gpr::GPRMeta{
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
        medium_velocity
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
pub fn load_cor(filepath: &Path, projected_crs: &str) -> Result<gpr::GPRLocation, Box<dyn Error>> {

    let content = std::fs::read_to_string(filepath)?;

    let transformer = proj::Proj::new_known_crs("EPSG:4326", projected_crs, None)?;

    // Create a new empty points vec
    let mut points: Vec<gpr::CorPoint>  = Vec::new();
    // Loop over the lines of the file and parse CorPoints from it
    for line in content.lines() {
        
        // Split the line into ten separate columns.
        let data: Vec<&str> = line.splitn(10, "\t").collect();

        // If the line could not be split in ten columns, it is probably wrong.
        if data.len() < 10 {
            continue
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

        // Project the coordinate to eastings/northings
        let (easting, northing) = transformer.convert((longitude, latitude))?;

        // Parse the date and time columns into datetime, then convert to seconds after UNIX epoch.
        let datetime = chrono::DateTime::parse_from_rfc3339(&format!("{}T{}+00:00", data[1], data[2]))?.timestamp() as f64;


        points.push(gpr::CorPoint{
            trace_n: (data[0].parse::<i64>()? - 1) as u32,  // The ".cor"-files are 1-indexed whereas this is 0-indexed
            time_seconds: datetime,
            easting,
            northing,
            altitude: data[7].parse()?,
        });
    };

    if points.len() > 0 {
        Ok(gpr::GPRLocation{cor_points: points, correction: gpr::LocationCorrection::NONE, crs: projected_crs.to_string()})
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
    let bits_to_millivolt = 50000. /  i16::MAX as f32;

    // The values are read as 16 bit little endian signed integers, and are converted to millivolts
    for byte_pair in bytes.chunks_exact(2) {
        let value = i16::from_le_bytes([byte_pair[0], byte_pair[1]]);
        data.push(value as f32 * bits_to_millivolt);
    };

    let width: usize = data.len() / height;

    Ok(ndarray::Array2::from_shape_vec((width, height), data)?.reversed_axes())
}


/// Export a GPR profile and its metadata to a NetCDF (".nc") file.
///
/// It will remove any file that already exists with the same filename.
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
        file.add_attribute("start-datetime", chrono::DateTime::<chrono::Utc>::from_utc(chrono::NaiveDateTime::from_timestamp(gpr.location.cor_points[0].time_seconds as i64, 0), chrono::Utc).to_rfc3339())?;
        file.add_attribute("stop-datetime", chrono::DateTime::<chrono::Utc>::from_utc(chrono::NaiveDateTime::from_timestamp(gpr.location.cor_points[gpr.location.cor_points.len() - 1].time_seconds as i64, 0), chrono::Utc).to_rfc3339())?;
        file.add_attribute("processing-datetime", chrono::Local::now().to_rfc3339())?;
        file.add_attribute("antenna", gpr.metadata.antenna.clone())?;
        file.add_attribute("antenna-separation", gpr.metadata.antenna_separation)?;
        file.add_attribute("frequency-steps", gpr.metadata.frequency_steps)?;
        file.add_attribute("vertical-sampling-frequency", gpr.metadata.frequency.clone())?;

        file.add_attribute("processing-log", gpr.log.join("\n"))?;
        file.add_attribute("original-filename", gpr.metadata.rd3_filepath.file_name().unwrap().to_str().unwrap())?;

        file.add_attribute("medium-velocity", gpr.metadata.medium_velocity)?;
        file.add_attribute("medium-velocity-unit", "m / ns")?;

        file.add_attribute("elevation-correction",  match gpr.location.correction.clone() {
            gpr::LocationCorrection::NONE => "None".to_string(),
            gpr::LocationCorrection::DEM(fp) => format!("DEM-corrected: {:?}", fp.as_path().file_name().unwrap().to_str().unwrap())
        })?;

        file.add_attribute("crs", gpr.location.crs.clone())?;
        let distance_vec = gpr.location.distances().into_raw_vec();
        file.add_attribute("total-distance",distance_vec[distance_vec.len() - 1])?;
        file.add_attribute("total-distance-unit", "m")?;

        file.add_attribute("program-version", format!("{} version {}, © {}", crate::PROGRAM_NAME, crate::PROGRAM_VERSION, crate::PROGRAM_AUTHORS))?;

        // Add the data to the file
        let mut data = file.add_variable::<f32>("data", &["y","x"])?;
        data.put_values(&gpr.data.iter().map(|v| v.to_owned()).collect::<Vec<f32>>(), None, Some(&[gpr.height(), gpr.width()]))?;

        // The default coordinates are distance for x and return time for y
        data.add_attribute("coordinates", "distance return-time")?;
        data.add_attribute("unit", "mV")?;

        // Add the distance variable to the x dimension
        let mut ds = file.add_variable::<f32>("distance", &["x"])?;
        ds.put_values(&distance_vec, Some(&[0]), None)?;
        ds.add_attribute("unit", "m")?;

        // Add the time variable to the x dimension
        let mut time = file.add_variable::<f64>("time", &["x"])?;
        time.put_values(&gpr.location.cor_points.iter().map(|point| point.time_seconds).collect::<Vec<f64>>(), Some(&[0]), None)?;
        time.add_attribute("unit", "s")?;

        // Add the easting variable to the x dimension
        let mut easting = file.add_variable::<f64>("easting", &["x"])?;
        easting.put_values(&gpr.location.cor_points.iter().map(|point| point.easting).collect::<Vec<f64>>(), Some(&[0]), None)?;
        easting.add_attribute("unit", "m")?;

        // Add the northing variable to the x dimension
        let mut northing = file.add_variable::<f64>("northing", &["x"])?;
        northing.put_values(&gpr.location.cor_points.iter().map(|point| point.northing).collect::<Vec<f64>>(), Some(&[0]), None)?;
        northing.add_attribute("unit", "m")?;

        // Add the elevation variable to the x dimension
        let mut elevation = file.add_variable::<f64>("elevation", &["x"])?;
        elevation.put_values(&gpr.location.cor_points.iter().map(|point| point.altitude).collect::<Vec<f64>>(), Some(&[0]), None)?;
        elevation.add_attribute("unit", "m a.s.l.")?;

        // Add the two-way return time variable to the y dimension
        let return_time_arr = (Array1::range(0_f32, gpr.metadata.time_window, gpr.vertical_resolution_ns())).into_raw_vec();
        let mut return_time = file.add_variable::<f32>("return-time", &["y"])?;
        return_time.put_values(&return_time_arr, Some(&[0]), None)?;
        return_time.add_attribute("unit", "ns")?;

        // Add the depth variable to the y dimension
        let mut depth = file.add_variable::<f32>("depth", &["y"])?;
        depth.put_values(&(return_time_arr.iter().map(|t| t * 0.168 * 0.5).collect::<Vec<f32>>()), Some(&[0]), None)?;

        depth.add_attribute("unit", "m")?;

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
        let data = gpr.data.iter().collect::<Vec<&f32>>();

        // Get quick and dirty quantiles by only looking at 10th of the data
        let mut vals = data.iter().step_by(10).collect::<Vec<&&f32>>();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let minval = *vals[(vals.len() as f32 * 0.01) as usize];
        let maxval = *vals[(vals.len() as f32 * 0.99) as usize];

        let logit99 = (0.99_f32 / (1.0_f32 - 0.99_f32)).log(std::f32::consts::E);

        let pixels: Vec<u8> = data.into_par_iter().map(|f| {
                (
                    255.0 * {
                        let val_norm = ((f - minval) / (maxval - minval)).clamp(0.0, 1.0);

                        0.5 + (val_norm / (1.0_f32 - val_norm)).log(std::f32::consts::E) / logit99
                    }

                ) as u8
        }).collect();

        image::save_buffer(
            filepath,
            &pixels,
            gpr.width() as u32,
            gpr.height() as u32,
            image::ColorType::L8,
        )?;

        Ok(())

}
