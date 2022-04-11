use std::path::Path;
use std::error::Error;
use std::collections::HashMap;
use ndarray::{Array2,  Array1};
use nshare::MutNdarray2;
use image::GrayImage;

use crate::{tools,gpr};

pub fn load_rad(filepath: &Path, medium_velocity: f32) -> Result<gpr::GPRMeta, Box<dyn Error>> {

    let content = std::fs::read_to_string(filepath)?;

    let data: HashMap<&str, &str> = content.lines().filter_map(|s| s.split_once(":")).collect();

    let rd3_filepath = filepath.with_extension("rd3");

    if !rd3_filepath.is_file() {
        return Err(format!("File not found: {:?}", rd3_filepath).into());
    };

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

pub fn load_cor(filepath: &Path, projected_crs: &str) -> Result<gpr::GPRLocation, Box<dyn Error>> {

    let content = std::fs::read_to_string(filepath)?;

    let mut points: Vec<gpr::CorPoint>  = Vec::new();

    let transformer = proj::Proj::new_known_crs("EPSG:4326", projected_crs, None)?;

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

        let (easting, northing) = transformer.convert((longitude, latitude))?;
        //let (northing, easting, _) = utm::to_utm_wgs84(latitude, longitude, utm_zone);

        let datetime = chrono::DateTime::parse_from_rfc3339(&format!("{}T{}+00:00", data[1], data[2]))?.timestamp() as f64;


        points.push(gpr::CorPoint{
            trace_n: (data[0].parse::<i64>()? - 1) as u32,  // The data is 1-indexed
            time_seconds: datetime,
            easting,
            northing,
            altitude: data[7].parse()?,
        });
    };

    Ok(gpr::GPRLocation{cor_points: points, correction: gpr::LocationCorrection::NONE, crs: projected_crs.to_string()})
}

pub fn load_rd3(filepath: &Path, height: u32) -> Result<Array2<f32>, Box<dyn std::error::Error>> {

    let bytes = std::fs::read(filepath)?;

    let mut data: Vec<f32> = Vec::new();

    for byte_pair in bytes.chunks_exact(2) {
        let short = i16::from_le_bytes([byte_pair[0], byte_pair[1]]);
        data.push(short as f32);
    };

    let width: usize = data.len() / (height as usize);

    Ok(ndarray::Array2::from_shape_vec((width, height as usize), data).unwrap().reversed_axes())
}


pub fn export_netcdf(gpr: &gpr::GPR, nc_filepath: &Path) -> Result<(), Box<dyn std::error::Error>> {

        if nc_filepath.is_file() {
            std::fs::remove_file(nc_filepath)?;

        };
        let mut file = netcdf::create(nc_filepath)?;

        file.add_dimension("x", gpr.width())?;

        file.add_dimension("y", gpr.height())?;

        
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

        //let utm33n_wkt = r#"PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32633"]]"#;

        //file.add_attribute("spatial_ref", utm33n_wkt)?;
        file.add_attribute("crs", gpr.location.crs.clone())?;
        let distance_vec = gpr.location.distances().into_raw_vec();
        file.add_attribute("total-distance",distance_vec[distance_vec.len() - 1])?;
        file.add_attribute("total-distance-unit", "m")?;

        file.add_attribute("program-version", format!("{} version {}, © {}", crate::PROGRAM_NAME, crate::PROGRAM_VERSION, crate::PROGRAM_AUTHORS))?;

        let mut data = file.add_variable::<f32>("data", &["y","x"])?;
        data.put_values(&gpr.data.clone().reversed_axes().into_raw_vec(), Some(&[0, 0]), None)?;
        data.add_attribute("coordinates", "distance return-time")?;

        let mut ds = file.add_variable::<f32>("distance", &["x"])?;
        ds.put_values(&distance_vec, Some(&[0]), None)?;
        ds.add_attribute("unit", "m")?;

        
        let mut time = file.add_variable::<f64>("time", &["x"])?;
        time.put_values(&gpr.location.cor_points.iter().map(|point| point.time_seconds).collect::<Vec<f64>>(), Some(&[0]), None)?;
        time.add_attribute("unit", "s")?;

        let mut easting = file.add_variable::<f64>("easting", &["x"])?;
        easting.put_values(&gpr.location.cor_points.iter().map(|point| point.easting).collect::<Vec<f64>>(), Some(&[0]), None)?;
        easting.add_attribute("unit", "m")?;

        let mut northing = file.add_variable::<f64>("northing", &["x"])?;
        northing.put_values(&gpr.location.cor_points.iter().map(|point| point.northing).collect::<Vec<f64>>(), Some(&[0]), None)?;
        northing.add_attribute("unit", "m")?;

        let mut elevation = file.add_variable::<f64>("elevation", &["x"])?;
        elevation.put_values(&gpr.location.cor_points.iter().map(|point| point.altitude).collect::<Vec<f64>>(), Some(&[0]), None)?;
        elevation.add_attribute("unit", "m a.s.l.")?;

        let return_time_arr = (Array1::range(0_f32, gpr.metadata.time_window, gpr.vertical_resolution_ns())).into_raw_vec();
        let mut return_time = file.add_variable::<f32>("return-time", &["y"])?;
        return_time.put_values(&return_time_arr, Some(&[0]), None)?;
        return_time.add_attribute("unit", "ns")?;

        let mut depth = file.add_variable::<f32>("depth", &["y"])?;
        depth.put_values(&(return_time_arr.iter().map(|t| t * 0.168 * 0.5).collect::<Vec<f32>>()), Some(&[0]), None)?;

        depth.add_attribute("unit", "m")?;

        Ok(())

}


pub fn render_jpg(gpr: &gpr::GPR, filepath: &Path) -> Result<(), Box<dyn Error>> {

        let vals = gpr.data.iter().map(|f| f.to_owned()).collect::<Vec<f32>>();

        let data = Array1::from_vec(vals);
        let min_max = tools::quantiles(&data, &[0.01, 0.99]);

        let mut image = GrayImage::new(gpr.width() as u32, gpr.height() as u32);

        let mut vals = image.mut_ndarray2();

        let logit99 = (0.99_f32 / (1.0_f32 - 0.99_f32)).log(std::f32::consts::E);

        // Scale the values to logit and convert them to u8
        vals.assign(&gpr.data.mapv(|f| {
            (
                255.0 * {
            
                let val_norm = ((f - min_max[0]) / (min_max[1] - min_max[0])).clamp(0.0, 1.0);

                0.5 + (val_norm / (1.0_f32 - val_norm)).log(std::f32::consts::E) / logit99
                }

            ) as u8

        }));

        image.save(filepath)?;

        Ok(())

}
