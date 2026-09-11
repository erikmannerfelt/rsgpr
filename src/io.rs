/// Functions to handle input and output (I/O) of GPR data files.
use ndarray::Array2;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::export::ExportAttr;
use crate::{formats, gpr};

/// Load and parse a Malå metadata file (.rad)
///
/// # Arguments
/// - `filepath`: The filepath of the input metadata file
/// - `medium_velocity`: The velocity of the portrayed medium to assign the GPR data
/// - `override_antenna_mhz`: Optional antenna frequency override (will not read from metadata).
/// - `override_antenna_separation`: Optional antenna separation override (will not read from metadata).
///
/// # Returns
/// A gpr::GPRMeta instance.
///
/// # Errors
/// - The file could not be read
/// - The contents could not be parsed correctly
/// - The associated ".rd3" file does not exist.
pub fn load_rad(
    filepath: &Path,
    medium_velocity: f32,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<gpr::GPRMeta, Box<dyn Error>> {
    let bytes = std::fs::read(Path::new(filepath))?; // read as raw bytes
    let content = String::from_utf8_lossy(&bytes); // &str with invalid bytes replaced

    // Collect all rows into a hashmap, assuming a "KEY:VALUE" structure.
    let data: HashMap<&str, &str> = content.lines().filter_map(|s| s.split_once(':')).collect();

    let rd3_filepath = formats::find_neighbor_case_insensitive(filepath, "rd3")
        .unwrap_or_else(|| filepath.with_extension("rd3"));
    if !rd3_filepath.is_file() {
        return Err(format!("File not found: {rd3_filepath:?}").into());
    };

    // Extract and parse all required metadata into a new GPRMeta object.
    let antenna = data
        .get("ANTENNAS")
        .ok_or("No 'ANTENNAS' key in metadata")?
        .trim()
        .to_string();

    let antenna_mhz = match override_antenna_mhz {
        Some(v) => v,
        None => antenna.split("MHz").collect::<Vec<&str>>()[0]
            .trim()
            .parse::<f32>()
            .map_err(|e| {
                format!("Could not read frequency from the antenna field ({e:?}). Try using the antenna MHz override")
            })?
    };

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
        antenna_mhz,
        antenna,
        antenna_separation: match override_antenna_separation {
            Some(v) => v,
            None => data
                .get("ANTENNA SEPARATION")
                .ok_or("No 'ANTENNA SEPARATION' key in metadata")?
                .trim()
                .parse()?,
        },
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
        data_filepath: rd3_filepath,
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
        let data: Vec<&str> = line.split_whitespace().collect();

        // If the line could not be split in ten columns, it is probably wrong.
        if data.len() < 10 {
            continue;
        };

        let Ok(mut latitude) = data[3].parse::<f64>() else {
            continue;
        };
        let Ok(mut longitude) = data[5].parse::<f64>() else {
            continue;
        };

        // Invert the sign of the latitude if it's on the southern hemisphere
        if data[4].trim() == "S" {
            latitude *= -1.;
        };

        // Invert the sign of the longitude if it's west of the prime meridian
        if data[6].trim() == "W" {
            longitude *= -1.;
        };

        // Ugly fix for 9:00:00 -> 09:00:00
        let mut time_str = data[2].to_string();
        if time_str.len() == 7 {
            time_str = "0".to_string() + &time_str;
        }
        // Parse the date and time columns into datetime, then convert to seconds after UNIX epoch.
        // In some odd cases, the time information is wrong. Those lines should b eskipped
        let Ok(datetime_obj) =
            chrono::DateTime::parse_from_rfc3339(&format!("{}T{}+00:00", data[1], time_str))
        else {
            continue;
        };
        let datetime = datetime_obj.timestamp() as f64;

        let Ok(altitude) = data[7].parse::<f64>() else {
            continue;
        };

        // The ".cor"-files are 1-indexed whereas this is 0-indexed
        let Ok(trace_n) = data[0].parse::<i64>().map(|v| v - 1) else {
            continue;
        };

        // If the trace number in the corfile is 0, then this will overflow
        if trace_n < 0 {
            continue;
        };

        coords.push(crate::coords::Coord {
            x: longitude,
            y: latitude,
        });

        // Coordinates are 0 right now. That's fixed right below
        points.push(gpr::CorPoint {
            trace_n: trace_n as u32,
            time_seconds: datetime,
            easting: 0.,
            northing: 0.,
            altitude,
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
    for byte_pair in bytes.as_chunks::<2>().0 {
        let value = i16::from_le_bytes(*byte_pair);
        data.push(value as f32 * bits_to_millivolt);
    }

    let width: usize = data.len() / height;

    Ok(ndarray::Array2::from_shape_vec((width, height), data)?.reversed_axes())
}

pub fn load_pe_dt1(
    filepath: &Path,
    height: usize,
    width: usize,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(filepath)?;

    const TRACE_HEADER_BYTES: usize = 25 * 4 + 28; // 128

    // Based on one header. Should probably be set from the header itself.
    // Also, it's a bit unclear if it should be halved or not...
    let bits_to_millivolt = 104.12 / i16::MAX as f32;

    let bytes_per_trace = TRACE_HEADER_BYTES + height * 2;
    let expected_len = width * bytes_per_trace;

    if bytes.len() < expected_len {
        return Err(format!(
            "File too short: got {} bytes, expected at least {} bytes",
            bytes.len(),
            expected_len
        )
        .into());
    }

    let mut data: Vec<f32> = Vec::with_capacity(height * width);
    let mut offset: usize = 0;

    for _ in 0..width {
        offset += TRACE_HEADER_BYTES;

        let end = offset + height * 2;
        let slice = &bytes[offset..end];

        for j in 0..height {
            let k = j * 2;
            let v = i16::from_le_bytes([slice[k], slice[k + 1]]);
            data.push(v as f32 * bits_to_millivolt);
        }

        offset = end;
    }

    Ok(Array2::from_shape_vec((width, height), data)?.reversed_axes())
}

pub fn load_pe_hd(
    filepath: &Path,
    medium_velocity: f32,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<gpr::GPRMeta, Box<dyn Error>> {
    let content = std::fs::read_to_string(filepath)?;

    // Collect all rows into a hashmap, assuming a "KEY:VALUE" structure.
    let mut data = HashMap::<&str, &str>::new();
    for (key, value) in content.lines().filter_map(|s| s.split_once('=')) {
        data.insert(key.trim(), value.trim());
    }
    let samples: u32 = data
        .get("NUMBER OF PTS/TRC")
        .ok_or("No 'NUMBER OF PTS/TRC' key in metadata")?
        .trim()
        .parse()?;
    let time_window: f32 = data
        .get("TOTAL TIME WINDOW")
        .ok_or("No 'TOTAL TIME WINDOW' key in metadata")?
        .trim()
        .parse()?;

    let frequency = 1000. * (samples as f32) / time_window;

    let dt1_filepath = formats::find_neighbor_case_insensitive(filepath, "dt1")
        .unwrap_or_else(|| filepath.with_extension("dt1"));
    if !dt1_filepath.is_file() {
        return Err(format!("File not found: {dt1_filepath:?}").into());
    };

    let antenna_mhz = match override_antenna_mhz {
        Some(v) => v,
        None => data
            .get("NOMINAL FREQUENCY")
            .ok_or("No 'NOMINAL FREQUENCY' key in metadata")?
            .replace(' ', "")
            .parse()
            .map_err(|e| {
                format!("Could not read frequency from the 'NOMINAL FREQUENCY' field ({e:?}). Try using the antenna MHz override")
            })?
    };

    Ok(gpr::GPRMeta {
        samples,
        frequency,
        frequency_steps: 0,
        time_interval: data
            .get("TRACE INTERVAL (s)")
            .ok_or("No 'TRACE INTERVAL (s)' key in metadata")?
            .replace(' ', "")
            .parse()?,
        antenna_mhz,
        antenna: data
            .get("NOMINAL FREQUENCY")
            .ok_or("No 'NOMINAL FREQUENCY' key in metadata")?
            .replace(' ', "")
            .parse::<String>()?
            + " MHz",
        antenna_separation: match override_antenna_separation {
            Some(v) => v,
            None => data
                .get("ANTENNA SEPARATION")
                .ok_or("No 'ANTENNA SEPARATION' key in metadata")?
                .trim()
                .parse()?,
        },
        time_window,
        last_trace: data
            .get("NUMBER OF TRACES")
            .ok_or("No 'NUMBER OF TRACES' key in metadata")?
            .trim()
            .parse()?,
        data_filepath: dt1_filepath,
        medium_velocity,
    })
}

fn gssi_date_to_iso(date: &str) -> Result<String, Box<dyn Error>> {
    if date.len() != 6 {
        return Err(format!("Invalid GSSI date: {date}").into());
    }

    let day = &date[0..2];
    let month = &date[2..4];
    let year = &date[4..6];
    let year = format!("20{year}");
    Ok(format!("{year}-{month}-{day}"))
}

type GssiHeader = (u16, u16, u16, f32, f32, f32, f32, u16, String);

fn read_gssi_header(bytes: &[u8]) -> Result<GssiHeader, Box<dyn Error>> {
    if bytes.len() < 128 {
        return Err("GSSI header is too short".into());
    }

    let u16_at = |offset: usize| u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
    let i16_at = |offset: usize| i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
    let f32_at = |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    let data = u16_at(2);
    let nsamp = u16_at(4);
    let bits = u16_at(6);
    let _zero = i16_at(8);
    let sps = f32_at(10);
    let spm = f32_at(14);
    let _mpm = f32_at(18);
    let position = f32_at(22);
    let range = f32_at(26);
    let _npass = u16_at(30);
    let _create = &bytes[32..36];
    let _modify = &bytes[36..40];
    let _rgain = u16_at(40);
    let _nrgain = u16_at(42);
    let _text = u16_at(44);
    let _ntext = u16_at(46);
    let _proc = u16_at(48);
    let _nproc = u16_at(50);
    let nchan = u16_at(52);
    let antname = String::from_utf8_lossy(&bytes[98..112])
        .trim_matches('\0')
        .trim()
        .to_string();

    Ok((data, nsamp, bits, sps, spm, position, range, nchan, antname))
}

fn gssi_data_offset(bytes: &[u8], bits: u16, data: u16, nchan: u16) -> usize {
    let bytes_per_sample = (bits as usize / 8).max(1);
    let offset = if data < 1024 {
        1024 * data as usize
    } else {
        1024 * nchan as usize
    };
    if offset < bytes.len() {
        offset
    } else {
        32768 * bytes_per_sample
    }
}

fn gssi_bits_to_millivolt(bits: u16) -> f32 {
    // RGPR's readDZT.R uses bits2volt(Vmax = 50, nbits = bits), i.e.
    // abs(Vmax - Vmin) / 2^nbits with Vmin = -Vmax, then ridal exports mV.
    100000.0 / 2f32.powi(bits as i32)
}

pub(crate) fn gssi_antenna_mhz(antname: &str) -> Option<f32> {
    // RGPR readDZT.R antenna map.
    match antname.trim() {
        "3200" | "3200MLF" => None,
        "500MHz" => Some(500.),
        "3207" | "3207AP" => Some(100.),
        "5106" | "5106A" | "50200HS" => Some(200.),
        "50300" => Some(300.),
        "350" | "350HS" => Some(350.),
        "50270" | "50270S" => Some(270.),
        "50400" | "50400S" => Some(400.),
        "800" | "D50800" => Some(800.),
        "3101" | "3101A" => Some(900.),
        "51600" | "51600S" => Some(1600.),
        "62000" | "62000-003" => Some(2000.),
        "62300" | "62300XT" => Some(2300.),
        "52600" | "52600S" => Some(2600.),
        other => other
            .split_whitespace()
            .next()
            .and_then(|token| token.trim_end_matches("MHz").parse::<f32>().ok()),
    }
}

pub fn load_gssi_dzt(
    filepath: &Path,
    medium_velocity: f32,
    override_antenna_mhz: Option<f32>,
    override_antenna_separation: Option<f32>,
) -> Result<gpr::GPRMeta, Box<dyn Error>> {
    let bytes = std::fs::read(filepath)?;
    let (data, nsamp, bits, sps, _spm, position, range, nchan, antenna) = read_gssi_header(&bytes)?;
    let time_window = range - position;
    let frequency = 1000. * (nsamp as f32) / time_window;
    let antenna_mhz = match override_antenna_mhz {
        Some(v) => v,
        None => gssi_antenna_mhz(&antenna).ok_or_else(|| {
            format!(
                "Could not read frequency from the GSSI antenna field ({antenna:?}). Try using the antenna MHz override"
            )
        })?,
    };

    let data_offset = gssi_data_offset(&bytes, bits, data, nchan);
    if data_offset >= bytes.len() {
        return Err(format!("File too short: no data found in {:?}", filepath).into());
    }
    let bytes_per_sample = (bits as usize / 8).max(1);
    let samples_per_trace = nsamp as usize;
    let bytes_per_trace = samples_per_trace * bytes_per_sample;
    if bytes_per_trace == 0 {
        return Err("Invalid GSSI header: zero sample size".into());
    }
    let trace_bytes = bytes.len() - data_offset;
    let last_trace = (trace_bytes / bytes_per_trace) as u32;

    Ok(gpr::GPRMeta {
        samples: nsamp as u32,
        frequency,
        frequency_steps: 0,
        time_interval: 1.0 / sps,
        antenna_mhz,
        antenna,
        antenna_separation: override_antenna_separation.unwrap_or(0.0),
        time_window,
        last_trace,
        data_filepath: filepath.to_path_buf(),
        medium_velocity,
    })
}

pub fn load_dzt(filepath: &Path, height: usize) -> Result<Array2<f32>, Box<dyn Error>> {
    let bytes = std::fs::read(filepath)?;
    let (data, nsamp, bits, _sps, _spm, _position, _range, nchan, _antenna) =
        read_gssi_header(&bytes)?;
    let data_offset = gssi_data_offset(&bytes, bits, data, nchan);
    if data_offset >= bytes.len() {
        return Err(format!("File too short: no data found in {:?}", filepath).into());
    }

    let bytes_per_sample = (bits as usize / 8).max(1);
    let samples_per_trace = height.max(nsamp as usize);
    let bytes_per_trace = samples_per_trace * bytes_per_sample;
    let payload = &bytes[data_offset..];
    if payload.len() < bytes_per_trace {
        return Err(format!("File too short: {:?}", filepath).into());
    }

    let mut data: Vec<f32> = Vec::with_capacity(payload.len() / bytes_per_sample);
    let scale = gssi_bits_to_millivolt(bits);
    match bits {
        8 => {
            for byte in payload.iter() {
                data.push((*byte as i16 - 128) as f32 * scale);
            }
        }
        16 => {
            for chunk in payload.as_chunks::<2>().0 {
                let value = u16::from_le_bytes(*chunk) as i32 - 32768;
                data.push(value as f32 * scale);
            }
        }
        32 => {
            for chunk in payload.as_chunks::<4>().0 {
                let value = i32::from_le_bytes(*chunk);
                data.push(value as f32 * scale);
            }
        }
        other => return Err(format!("Unsupported GSSI sample width: {other} bits").into()),
    }

    let width = data.len() / height;
    Ok(Array2::from_shape_vec((width, height), data)?.reversed_axes())
}

pub fn load_gssi_dzg(
    filepath: &Path,
    projected_crs: Option<&String>,
) -> Result<gpr::GPRLocation, Box<dyn Error>> {
    let content = std::fs::read_to_string(filepath)?;
    let mut date_str: Option<String> = None;
    let mut current_scan: Option<u32> = None;
    let mut coords = Vec::<crate::coords::Coord>::new();
    let mut points: Vec<gpr::CorPoint> = Vec::new();

    for line in content.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts.first().copied() {
            Some("$GSSIS") => {
                current_scan = parts.get(1).and_then(|s| s.parse::<u32>().ok());
            }
            Some("$GPRMC") => {
                if let Some(date) = parts.get(9) {
                    date_str = Some(gssi_date_to_iso(date)?);
                }
            }
            Some("$GPGGA") => {
                let Some(scan) = current_scan else {
                    continue;
                };
                let Some(date) = &date_str else {
                    continue;
                };
                let (datetime, coord, altitude) = read_gga(line, date)?;
                coords.push(coord);
                points.push(gpr::CorPoint {
                    trace_n: scan,
                    time_seconds: datetime,
                    easting: 0.,
                    northing: 0.,
                    altitude,
                });
                current_scan = None;
            }
            _ => {}
        }
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

    Ok(gpr::GPRLocation {
        cor_points: points,
        correction: gpr::LocationCorrection::None,
        crs: projected_crs.to_string(),
    })
}

fn read_gga(gga_str: &str, date: &str) -> Result<(f64, crate::coords::Coord, f64), Box<dyn Error>> {
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut date = date.to_string();
    for (i, month) in months.iter().enumerate() {
        date = date.replace(month, &format!("{:02}", (i + 1)));
    }

    let parts: Vec<&str> = gga_str.split(",").collect();

    let lat_str = parts.get(2).unwrap();
    let mut lat = lat_str[..2].parse::<f64>()? + (lat_str[2..].parse::<f64>()? / 60.);

    if parts.get(3) == Some(&"S") {
        lat *= -1.;
    }

    let lon_str = parts.get(4).unwrap();
    let mut lon = lon_str[..3].parse::<f64>()? + (lon_str[3..].parse::<f64>()? / 60.);

    if parts.get(5) == Some(&"W") {
        lon *= -1.;
    }

    let coord = crate::coords::Coord { x: lon, y: lat };

    let elev = parts.get(9).unwrap().parse::<f64>()?;

    let time_str = parts.get(1).unwrap();
    let hr = time_str[..2].to_string();
    let min = time_str[2..4].to_string();
    let sec = time_str[4..].to_string();

    let datetime =
        chrono::DateTime::parse_from_rfc3339(&format!("{}T{}:{}:{}+00:00", date, hr, min, sec))?
            .timestamp() as f64;

    Ok((datetime, coord, elev))
}

pub fn load_pe_gp2(
    filepath: &Path,
    projected_crs: Option<&String>,
) -> Result<gpr::GPRLocation, Box<dyn Error>> {
    let content = std::fs::read_to_string(filepath)?;

    let mut date_str: Option<&str> = None;

    // Create a new empty points vec
    let mut coords = Vec::<crate::coords::Coord>::new();
    let mut points: Vec<gpr::CorPoint> = Vec::new();
    // Loop over the lines of the file and parse CorPoints from it
    for line in content.lines() {
        if line.starts_with(";") | line.starts_with("traces") {
            if line.contains("Date=") {
                date_str = Some(line.split_once("=").unwrap().1.split_once(" ").unwrap().0);
            }
            continue;
        };

        let data: Vec<&str> = line.splitn(5, ",").collect();

        let trace_n = (data[0].parse::<i64>()? - 1) as u32; // The ".cor"-files are 1-indexed whereas this is 0-indexed

        if points.last().map(|p| p.trace_n == trace_n) == Some(true) {
            continue;
        }

        let (datetime, coord, altitude) = read_gga(data[4], date_str.unwrap())?;

        coords.push(coord);

        // Coordinates are 0 right now. That's fixed right below
        points.push(gpr::CorPoint {
            trace_n,
            time_seconds: datetime,
            easting: 0.,
            northing: 0.,
            altitude,
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

impl From<ExportAttr> for netcdf::AttributeValue {
    fn from(val: ExportAttr) -> Self {
        match val {
            ExportAttr::String(s) => netcdf::AttributeValue::Str(s),
            ExportAttr::Strings(s) => netcdf::AttributeValue::Strs(s),
            ExportAttr::F64(v) => netcdf::AttributeValue::Double(v),
            ExportAttr::F32(v) => netcdf::AttributeValue::Float(v),
            ExportAttr::U8(v) => netcdf::AttributeValue::Uchar(v),
            ExportAttr::I64(v) => netcdf::AttributeValue::Longlong(v),
        }
    }
}
/// Common functionality for writing NetCDF variables
fn write_nc_variable_common<T>(
    v: &mut netcdf::VariableMut,
    name: &str,
    data: &[T],
    attrs: Option<&BTreeMap<String, ExportAttr>>,
) -> Result<(), String>
where
    T: netcdf::NcTypeDescriptor,
{
    v.put_values(data, ..)
        .map_err(|e| format!("NetCDF export error when adding variable '{name}' data: {e}"))?;

    if let Some(attrs) = attrs {
        for (k, attr) in attrs {
            v.put_attribute(k, attr.to_owned()).map_err(|e| {
                format!("NetCDF export error when setting variable '{name}' attribute '{k}': {e}")
            })?;
        }
    };

    Ok(())
}

/// Add a variable without compression/chunking
fn add_nc_variable<T>(
    file: &mut netcdf::FileMut,
    name: &str,
    dims: &[&str],
    data: &[T],
    attrs: Option<&BTreeMap<String, ExportAttr>>,
) -> Result<(), String>
where
    T: netcdf::NcTypeDescriptor,
{
    let mut v = file
        .add_variable::<T>(name, dims)
        .map_err(|e| format!("NetCDF export error when adding variable '{name}': {e}"))?;

    write_nc_variable_common(&mut v, name, data, attrs)
}

/// Add a 2D variable with compression/chunking
fn add_nc_variable_compressed_2d<T>(
    file: &mut netcdf::FileMut,
    name: &str,
    dims: &[&str],
    data: &[T],
    shape: (usize, usize), // (ny, nx) in the same order as `dims`
    attrs: Option<&BTreeMap<String, ExportAttr>>,
) -> Result<(), String>
where
    T: netcdf::NcTypeDescriptor,
{
    let (ny, nx) = shape;

    let mut v = file
        .add_variable::<T>(name, dims)
        .map_err(|e| format!("NetCDF export error when adding variable '{name}': {e}"))?;

    v.set_compression(5, true)
        .map_err(|e| format!("NetCDF export error when setting '{name}' compression: {e}"))?;

    // 256 matches the web viewer's render chunk size (see #115 / #118), so a
    // render chunk decompresses exactly one HDF5 chunk instead of a fraction
    // of a larger one. Measured on a ~1 GB synthetic radargram: 5x lower
    // latency for a single chunk (8.4 -> 1.7 ms) and 3x for a full viewer
    // sweep (6.9 -> 2.2 s), for no change in file size -- radar amplitudes
    // compress ~1.17x regardless of chunk size, so there is no space/speed
    // tradeoff here to weigh against.
    for chunking in [256_usize, 128, 64, 32, 16, 8] {
        if ny < chunking || nx < chunking {
            continue;
        }
        v.set_chunking(&[chunking, chunking])
            .map_err(|e| format!("NetCDF export error when chunking '{name}': {e}"))?;
        break;
    }

    write_nc_variable_common(&mut v, name, data, attrs)?;

    Ok(())
}

/// Add an attribute to a NetCDF file
fn add_nc_attribute<T>(file: &mut netcdf::FileMut, name: &str, data: T) -> Result<(), String>
where
    T: Into<netcdf::AttributeValue>,
{
    file.add_attribute(name, data)
        .map_err(|e| format!("NetCDF export error when adding '{name}' attribute: {e}"))?;
    Ok(())
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
pub fn export_netcdf(
    ds: &crate::export::ExportDataset<'_>,
    nc_filepath: &Path,
) -> Result<(), String> {
    // Remove existing file (same reason as before)
    if nc_filepath.is_file() {
        std::fs::remove_file(nc_filepath).map_err(|e| {
            format!("NetCDF export error when removing old file with same name: {e}")
        })?;
    }

    // Create new file
    let mut file = netcdf::create(nc_filepath)
        .map_err(|e| format!("NetCDF export error when creating NetCDF file: {e}"))?;

    // ---- Dimensions ----
    for (name, len) in &ds.dims {
        file.add_dimension(name, *len)
            .map_err(|e| format!("NetCDF export error when adding dimension {name}: {e}"))?;
    }

    // ---- Global attributes from dataset ----
    for (k, v) in &ds.attrs {
        add_nc_attribute(&mut file, k, v.to_owned())?;
    }

    // ---- Coordinates (1D) ----
    for (name, var) in &ds.coords {
        match &var.data {
            crate::export::ExportArray::U32Owned1D(v) => {
                add_nc_variable::<u32>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    v,
                    Some(&var.attrs),
                )?;
            }
            crate::export::ExportArray::F32Owned1D(v) => {
                // collect unit attr if present
                add_nc_variable::<f32>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    v,
                    Some(&var.attrs),
                )?;
            }
            crate::export::ExportArray::F64Owned1D(v) => {
                add_nc_variable::<f64>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    v,
                    Some(&var.attrs),
                )?;
            }
            crate::export::ExportArray::U8Scalar(v) => {
                add_nc_variable::<u8>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    &[*v],
                    Some(&var.attrs),
                )?;
            }
            crate::export::ExportArray::F32Borrowed2D(_) => {
                // coords are expected to be 1D; ignore
                continue;
            }
        }
    }

    // ---- Data variables ----
    for (name, var) in &ds.data_vars {
        match &var.data {
            crate::export::ExportArray::F32Borrowed2D(arr2d) => {
                // Flatten and write compressed/chunked 2D
                let ny = ds.dims[var.dims[0].as_str()];
                let nx = ds.dims[var.dims[1].as_str()];
                let flat: Vec<f32> = arr2d.iter().copied().collect();

                add_nc_variable_compressed_2d::<f32>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    &flat,
                    (ny, nx),
                    Some(&var.attrs),
                )?;
            }
            crate::export::ExportArray::U8Scalar(v) => {
                add_nc_variable::<u8>(
                    &mut file,
                    name,
                    &var.dims.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    &[*v],
                    Some(&var.attrs),
                )?;
            }
            // data variables are expected to be 2D here; ignore other shapes
            _ => continue,
        }
    }

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

/// Result of inspecting a `.nc` candidate for Ridal recognition (#123).
///
/// A plain `is_ridal_nc() -> bool` is deliberately avoided: callers need
/// metadata for `Supported` files and need to distinguish an ordinary
/// non-Ridal NetCDF file (`NotRidal`) from an I/O or NetCDF-reading failure,
/// which `inspect_ridal_netcdf` reports as `Err` rather than as a variant
/// here.
///
/// Only consumed by catalog discovery under the `server` feature (#122);
/// the `cfg_attr` below reflects that honestly rather than blanket-allowing
/// dead code for CLI-only builds.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub enum RidalNetcdfKind {
    NotRidal,
    Supported(RidalNetcdfMetadata),
}

/// Metadata read from a supported Ridal-produced NetCDF file, without
/// loading the amplitude array.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub struct RidalNetcdfMetadata {
    pub radargram_id: crate::identity::RadargramId,
    pub display_name: Option<crate::identity::DisplayName>,
    pub group_name: Option<crate::identity::GroupName>,
    pub group_id: Option<crate::identity::GroupId>,
    /// Kept as the raw RFC3339 string rather than parsed: the fingerprint in
    /// #117 hashes this exact string, so re-serializing a parsed value could
    /// silently change revision identity by changing formatting.
    pub processing_datetime: String,
    pub ridal_version: String,
    /// `(n_samples, n_traces)`, i.e. `(rows, columns)` of the `data` variable.
    pub shape: (usize, usize),
}

/// Read a single global attribute as a string, or `None` if absent or not a
/// string-valued attribute.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
fn read_global_str_attr(file: &netcdf::File, name: &str) -> Option<String> {
    let attr = file.attribute(name)?;
    match attr.value() {
        Ok(netcdf::AttributeValue::Str(s)) => Some(s),
        _ => None,
    }
}

/// Read a global attribute as a string, trying `primary` first and falling
/// back to `legacy` if absent. Supports files written before the
/// `ridal_*` attribute rename (#116); the value is otherwise identical.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
fn read_global_str(file: &netcdf::File, primary: &str, legacy: &str) -> Option<String> {
    read_global_str_attr(file, primary).or_else(|| read_global_str_attr(file, legacy))
}

/// Inspect `path` for Ridal recognition without loading the amplitude array.
///
/// Recognition requires `ridal_version`, `ridal_processing_datetime` and a
/// valid `ridal_radargram_id` to all be present; anything else is reported
/// as `NotRidal` rather than an error. No legacy-unprefixed-name fallback
/// is needed for these two: `ridal_radargram_id` became a mandatory,
/// always-written attribute in the exact same change that renamed
/// `program_version`/`processing_datetime` to their `ridal_` forms (#116),
/// so a file old enough to have the unprefixed names is always also old
/// enough to lack `ridal_radargram_id` -- and gets rejected on that check
/// regardless. (The group name/id legacy fallback below is a different
/// case: that split landed well after `ridal_radargram_id` was already
/// mandatory, so files genuinely exist with a valid id and only the old
/// unsplit `ridal_group` attribute.)
///
/// Errors are reserved for failures to open or read the file at all, kept
/// distinct from `NotRidal` so catalog discovery (#122) can report them
/// separately rather than silently skipping unreadable candidates.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
pub fn inspect_ridal_netcdf(path: &Path) -> Result<RidalNetcdfKind, String> {
    let file = netcdf::open(path).map_err(|e| format!("Failed to open {path:?} as NetCDF: {e}"))?;

    let ridal_version = read_global_str_attr(&file, "ridal_version");
    let processing_datetime = read_global_str_attr(&file, "ridal_processing_datetime");
    let (ridal_version, processing_datetime) = match (ridal_version, processing_datetime) {
        (Some(v), Some(d)) => (v, d),
        _ => return Ok(RidalNetcdfKind::NotRidal),
    };

    let radargram_id = match read_global_str_attr(&file, "ridal_radargram_id") {
        Some(raw) => match crate::identity::RadargramId::new(&raw) {
            Ok(id) => id,
            Err(_) => return Ok(RidalNetcdfKind::NotRidal),
        },
        None => return Ok(RidalNetcdfKind::NotRidal),
    };

    let display_name = read_global_str_attr(&file, "ridal_display_name")
        .and_then(crate::identity::DisplayName::from_input);
    // Legacy fallback: files from before the group name/id split (#116
    // extended one level up) wrote a single "ridal_group" attribute that
    // was itself a validated slug, used directly as both display heading
    // and id. Reading it as the *name* here and deriving the id from it
    // below reproduces that value unchanged for such files, since
    // sanitizing an already-valid slug is a no-op.
    let group_name = read_global_str(&file, "ridal_group_name", "ridal_group")
        .and_then(crate::identity::GroupName::from_input);
    let group_id = read_global_str_attr(&file, "ridal_group_id")
        .and_then(|raw| crate::identity::GroupId::new(&raw).ok())
        .or_else(|| {
            group_name
                .as_ref()
                .and_then(|name| crate::identity::GroupId::from_fallback(name.as_str()).ok())
        });

    let Some(data_var) = file.variable("data") else {
        return Ok(RidalNetcdfKind::NotRidal);
    };
    let dims = data_var.dimensions();
    if dims.len() != 2 {
        return Ok(RidalNetcdfKind::NotRidal);
    }
    let shape = (dims[0].len(), dims[1].len());

    Ok(RidalNetcdfKind::Supported(RidalNetcdfMetadata {
        radargram_id,
        display_name,
        group_name,
        group_id,
        processing_datetime,
        ridal_version,
        shape,
    }))
}

#[cfg(test)]
mod tests {

    use std::{path::PathBuf, str::FromStr};

    use super::{gssi_antenna_mhz, load_cor, load_dzt, load_gssi_dzg, load_gssi_dzt, load_rad};

    fn make_gssi_dzt(bytes_per_sample: usize) -> Vec<u8> {
        let samples = 4usize;
        let traces = 2usize;
        let data_offset = 128 * 1024;
        let mut bytes = vec![0u8; data_offset + samples * traces * bytes_per_sample];
        bytes[2..4].copy_from_slice(&128u16.to_le_bytes());
        bytes[4..6].copy_from_slice(&(samples as u16).to_le_bytes());
        bytes[6..8].copy_from_slice(&((bytes_per_sample * 8) as u16).to_le_bytes());
        bytes[10..14].copy_from_slice(&12f32.to_le_bytes());
        bytes[14..18].copy_from_slice(&4f32.to_le_bytes());
        bytes[22..26].copy_from_slice(&(-60f32).to_le_bytes());
        bytes[26..30].copy_from_slice(&600f32.to_le_bytes());
        bytes[52..54].copy_from_slice(&1u16.to_le_bytes());
        let ant = b"5106";
        bytes[98..98 + ant.len()].copy_from_slice(ant);

        let mut offset = data_offset;
        for trace in 0..traces {
            for sample in 0..samples {
                let value = (trace * 10 + sample + 1) as i32;
                bytes[offset..offset + bytes_per_sample]
                    .copy_from_slice(&value.to_le_bytes()[..bytes_per_sample]);
                offset += bytes_per_sample;
            }
        }

        bytes
    }

    #[test]
    fn test_gssi_antenna_map() {
        assert_eq!(gssi_antenna_mhz("5106"), Some(200.));
        assert_eq!(gssi_antenna_mhz("3101A"), Some(900.));
        assert_eq!(gssi_antenna_mhz("50300"), Some(300.));
    }

    /// Fake some data. One point is in the northern hemisphere and one is in the southern
    fn fake_cor_text() -> String {
        [
            "1\t2022-01-01\t00:00:01\t78.0\tN\t16.0\tE\t100.0\tM\t1",
            "10\t2022-01-01\t9:01:00\t78.0\tS\t16.0\tW\t100.0\tM\t1",
            "0\t2022-01-01\t00:01:00\t78.0\tS\t16.0\tW\t100.0\tM\t1", // Trace starts at 0 (bad)
            "11\t2022-01", // This simulates an unfinished line that should be skipped
            "000000\tN\t17.433201666667\tE\t332.20\tM\t2.00", // Another bad line that should be skipped
            "9673\t2011-05-07\t18:95\t79.89\tN\t23.88\tE\t722.1317\tM\t0.62", // Bad time
            "14897\t2010-05-05\t1.:00:\t79.793\tN\t23.32\tE\t692.8199\tM\t0.58", // Another bad time
            "21584\t2010-05-05\t12:04:58   79.78905884333\tN 23.23301804333 E M 2        0.58.0592", // Bad elevation and mixed whitespace/tab
        ]
        .join("\r\n")
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Added 2026-02-17 because gdal is hard to install in CI
    fn test_load_cor() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cor_path = temp_dir.path().join("hello.cor");

        std::fs::write(&cor_path, fake_cor_text()).unwrap();

        // Load it and "convert" (or rather don't convert) the CRS to WGS84
        let locations = load_cor(&cor_path, Some(&"EPSG:4326".to_string())).unwrap();

        println!("{locations:?}");
        assert_eq!(locations.cor_points.len(), 2);

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

        let gpr_meta = load_rad(&rad_path, 0.1, None, None).unwrap();

        // Check that the correct values were parsed
        assert_eq!(gpr_meta.samples, 2024);
        assert_eq!(gpr_meta.frequency, 1000.);
        assert_eq!(gpr_meta.frequency_steps, 20);
        assert_eq!(gpr_meta.time_interval, 0.1);
        assert_eq!(gpr_meta.antenna_mhz, 100.);
        assert_eq!(gpr_meta.antenna_separation, 0.5);
        assert_eq!(gpr_meta.time_window, 2000.);
        assert_eq!(gpr_meta.last_trace, 40);
        assert_eq!(gpr_meta.data_filepath, rd3_path);

        // Test overriding the antenna frequency
        let gpr_meta = load_rad(&rad_path, 0.1, Some(200.), Some(1.25)).unwrap();
        assert_eq!(gpr_meta.antenna_mhz, 200.);
        assert_eq!(gpr_meta.antenna_separation, 1.25);
    }

    #[test]
    fn test_load_rad_bad_antenna_mhz() {
        // Fake a .rad metadata file
        let temp_dir = tempfile::tempdir().unwrap();
        let rad_path = temp_dir.path().join("hello.rad");
        let rd3_path = rad_path.with_extension("rd3");
        let rad_text = [
            "SAMPLES:2024",
            "FREQUENCY:                 1000.",
            "FREQUENCY STEPS: 20",
            "TIME INTERVAL: 0.1",
            "ANTENNAS: onehundredmegaherzz unshielded",
            "ANTENNA SEPARATION: 0.5",
            "TIMEWINDOW:2000",
            "LAST TRACE: 40",
        ]
        .join("\r\n");

        std::fs::write(&rad_path, rad_text).unwrap();

        // The rd3 file needs to exist, but it doesn't need to contain anything
        std::fs::write(&rd3_path, "").unwrap();

        // This should return an error
        let gpr_meta_fail = load_rad(&rad_path, 0.1, None, None);
        assert!(gpr_meta_fail.is_err());

        let err_msg = gpr_meta_fail.unwrap_err().to_string();
        assert!(
            err_msg.contains("frequency from the antenna field"),
            "Got:     {err_msg:?}\nExpected 'Could not read frequency from the antenna field'",
        );
        assert!(load_rad(&rad_path, 0.1, None, None).is_err());

        let gpr_meta = load_rad(&rad_path, 0.1, Some(100.), Some(2.5)).unwrap();
        assert_eq!(gpr_meta.antenna_mhz, 100.);
        assert_eq!(gpr_meta.antenna_separation, 2.5);
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Added 2026-02-17 because gdal is hard to install in CI
    fn test_load_pe_hd() {
        // Fake a .rad metadata file
        let temp_dir = tempfile::tempdir().unwrap();
        let rad_path = temp_dir.path().join("hello.hd");
        let rd3_path = rad_path.with_extension("dt1");
        let hd_text = [
            "1234",
            "200MHz_lines - pulseEKKO v1.8.1423",
            "2025-Apr-04",
            "NUMBER OF TRACES   = 9896",
            "NUMBER OF PTS/TRC  = 1625",
            "TIMEZERO AT POINT  = 163.5",
            "TOTAL TIME WINDOW  = 650",
            "STARTING POSITION  = 0",
            "FINAL POSITION     = 9895",
            "STEP SIZE USED     = 1",
            "POSITION UNITS     = m",
            "NOMINAL FREQUENCY  = 200",
            "ANTENNA SEPARATION = 1",
            "PULSER VOLTAGE (V) = 250",
            "NUMBER OF STACKS   = 1024",
            "SURVEY MODE        = Reflection",
            "STACKING TYPE      = F1, P1024, DynaQ OFF",
            "ELEVATION DATA ENTERED : MAX = 704.945 MIN = 625.49",
            "X Y Z POSITIONS ADDED - LatLong",
            "TRIGGER MODE       = Free",
            "DATA TYPE          = I*2",
            "AMPLITUDE WINDOW (mV)= 104.12",
            "TRACE INTERVAL (s) = 0.2",
            "TRACEHEADERDEF_26  = ORIENA",
            "GPR SERIAL#        = 006785670042",
            "RX SERIAL#         = 009030322610",
            "DVL SERIAL#        = 0087-0052-3004",
            "TX SERIAL#         = 002431701007",
        ]
        .join("\r\n");

        std::fs::write(&rad_path, hd_text).unwrap();

        // The rd3 file needs to exist, but it doesn't need to contain anything
        std::fs::write(&rd3_path, "").unwrap();

        let gpr_meta = crate::io::load_pe_hd(&rad_path, 0.1, None, None).unwrap();

        // Check that the correct values were parsed
        assert_eq!(gpr_meta.samples, 1625);
        assert_eq!(gpr_meta.frequency, 1000. * 1625. / 650.);
        // assert_eq!(gpr_meta.frequency_steps, 20);
        assert_eq!(gpr_meta.time_interval, 0.2);
        assert_eq!(gpr_meta.antenna_mhz, 200.);
        assert_eq!(gpr_meta.antenna_separation, 1.);
        assert_eq!(gpr_meta.time_window, 650.);
        assert_eq!(gpr_meta.last_trace, 9896);
        assert_eq!(gpr_meta.data_filepath, rd3_path);

        // Test overriding the antenna frequency
        let gpr_meta = crate::io::load_pe_hd(&rad_path, 0.1, Some(300.), Some(1.75)).unwrap();
        assert_eq!(gpr_meta.antenna_mhz, 300.);
        assert_eq!(gpr_meta.antenna_separation, 1.75);
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Added 2026-02-17 because gdal is hard to install in CI
    fn test_load_pe_gp2() {
        let temp_dir = tempfile::tempdir().unwrap();
        let gp2_path = temp_dir.path().join("hello.gp2");

        let gp2_text = [
            ";GPS@@@",
            ";Ver=1.1.0",
            ";DIP=2009-00152-00",
            ";Date=2025-Apr-04 02:08:52",
            ";----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
            "traces,odo_tick,pos(m),time_elapsed(s),GPS",
            "1,0,0.000000,0.028076,\"$GPGGA,130857.30,7719.1908439,N,01522.6497456,E,2,42,0.8,625.490,M,31.466,M,5.2,0123*40\"",
            "1,0,0.000000,0.131520,\"$GPGGA,130857.40,7719.1908439,N,01522.6497254,E,2,42,0.8,625.495,M,31.466,M,3.4,0123*46\"",
            "1,0,0.000000,0.227752,\"$GPGGA,130857.50,7719.1908439,N,01522.6497254,E,2,42,0.8,625.497,M,31.466,M,3.4,0123*45\"",
            "1,0,0.000000,0.331571,\"$GPGGA,130857.60,7719.1908439,N,01522.6497075,E,2,42,0.8,625.501,M,31.466,M,3.6,0123*4B\"",
            "2,0,0.000000,0.427717,\"$GPGGA,130857.70,7719.1908438,N,01522.6497080,E,2,42,0.8,625.502,M,31.466,M,3.6,0123*42\"",
            "2,0,0.000000,0.531579,\"$GPGGA,130857.80,7719.1908438,N,01522.6496916,E,2,42,0.8,625.505,M,31.466,M,3.8,0123*43\"",
            "3,0,0.000000,0.627810,\"$GPGGA,130857.90,7719.1908437,N,01522.6496922,E,2,42,0.8,625.507,M,31.466,M,3.8,0123*48\"",
            "3,0,0.000000,0.746427,\"$GPGGA,130858.00,7719.1908436,N,01522.6496784,E,2,42,0.8,625.509,M,31.466,M,4.0,0123*4C\"",
            "4,0,0.000000,0.827951,\"$GPGGA,130858.10,7719.1908435,N,01522.6496785,E,2,42,0.8,625.510,M,31.466,M,4.0,0123*47\"",
            "4,0,0.000000,0.931560,\"$GPGGA,130858.20,7719.1908434,N,01522.6496653,E,2,42,0.8,625.513,M,31.466,M,4.2,0123*4E\"",
            "5,0,0.000000,1.027760,\"$GPGGA,130858.30,7719.1908435,N,01522.6496658,E,2,42,0.8,625.515,M,31.466,M,4.2,0123*43\"",
            "5,0,0.000000,1.131538,\"$GPGGA,130858.40,7719.1908431,N,01522.6496540,E,2,42,0.8,625.516,M,31.466,M,4.4,0123*4F\"",
            "6,0,0.000000,1.227757,\"$GPGGA,130858.50,7719.1908433,N,01522.6496541,E,2,42,0.8,625.518,M,31.466,M,4.4,0123*43\"",
            "6,0,0.000000,1.331490,\"$GPGGA,130858.60,7719.1908428,N,01522.6496436,E,2,42,0.8,625.519,M,31.466,M,4.6,0123*48\"",
            "7,0,0.000000,1.427735,\"$GPGGA,130858.70,7719.1908427,N,01522.6496441,E,2,42,0.8,625.519,M,31.466,M,4.6,0123*46\"",
            "7,0,0.000000,1.531530,\"$GPGGA,130858.80,7719.1908423,N,01522.6496353,E,2,42,0.8,625.518,M,31.466,M,4.8,0123*46\"",
            "8,0,0.000000,1.627638,\"$GPGGA,130858.90,7719.1908423,N,01522.6496350,E,2,42,0.8,625.519,M,31.466,M,4.8,0123*45\"",
            "8,0,0.000000,1.735229,\"$GPGGA,130859.00,7719.1908420,N,01522.6496265,E,2,42,0.8,625.519,M,31.466,M,5.0,0123*40\"",
            "9,0,0.000000,1.827934,\"$GPGGA,130859.10,7719.1908422,N,01522.6496267,E,2,42,0.8,625.522,M,31.466,M,5.0,0123*49\"",
            "9,0,0.000000,1.931559,\"$GPGGA,130859.20,7719.1908419,N,01522.6496187,E,2,42,0.8,625.521,M,31.466,M,5.2,0123*4E\"",
        ]
        .join("\r\n");
        std::fs::write(&gp2_path, gp2_text).unwrap();

        let locations = crate::io::load_pe_gp2(&gp2_path, Some(&"EPSG:4326".to_string())).unwrap();

        assert_eq!(locations.cor_points.len(), 9);
        assert!(locations.cor_points.first().unwrap().northing > 77.);
    }

    #[test]
    fn test_load_gssi_dzt() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dzt_path = temp_dir.path().join("track.DZT");
        std::fs::write(&dzt_path, make_gssi_dzt(4)).unwrap();

        let meta = load_gssi_dzt(&dzt_path, 0.1, Some(200.), Some(1.5)).unwrap();
        assert_eq!(meta.samples, 4);
        assert_eq!(meta.last_trace, 2);
        assert_eq!(meta.time_interval, 1.0 / 12.0);
        assert_eq!(meta.antenna_mhz, 200.);
        assert_eq!(meta.antenna, "5106");
        assert_eq!(meta.antenna_separation, 1.5);

        let data = load_dzt(&dzt_path, 4).unwrap();
        assert_eq!(data.shape(), &[4, 2]);
        assert!(data[[0, 0]] > 0.0);
        assert!(data[[0, 1]] > 0.0);
        assert!(data[[0, 0]] < 1.0);
        assert!(data[[0, 1]] < 1.0);
    }

    #[test]
    fn test_load_gssi_dzg() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dzg_path = temp_dir.path().join("track.DZG");
        let dzg_text = [
            "$GSSIS,0,-1",
            "$GPRMC,140541,A,7901.4763,N,01332.0122,E,0.0,357.5,140424,8.8,E,A*16",
            "$GPGGA,140541,7901.4763,N,01332.0122,E,1,12,0.7,753.0,M,35.3,M,,*4C",
            "$GSSIS,3,-1",
            "$GPRMC,140543,A,7901.4764,N,01332.0123,E,0.0,357.5,140424,8.8,E,A*15",
            "$GPGGA,140543,7901.4764,N,01332.0123,E,1,12,0.7,752.9,M,35.3,M,,*4F",
        ]
        .join("\n");
        std::fs::write(&dzg_path, dzg_text).unwrap();

        let locations = load_gssi_dzg(&dzg_path, Some(&"EPSG:4326".to_string())).unwrap();
        assert_eq!(locations.cor_points.len(), 2);
        assert_eq!(locations.cor_points[0].trace_n, 0);
        assert_eq!(locations.cor_points[1].trace_n, 3);
        assert!((locations.cor_points[0].altitude - 753.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Added 2026-02-17 because gdal is hard to install in CI
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

            // The cor file says 1 but ridal is zero-indexed, hence 0
            assert_eq!(line0[0], "0");
            assert_eq!(line0[1], "16");
            assert_eq!(line0[2], "78");
            assert_eq!(line0[3], "100");

            let line1: Vec<&str> = content[2].split(",").collect();
            assert_eq!(line1[2], "-78");

            std::fs::remove_file(expected_path).unwrap();
        }
    }

    #[test]
    // #[ignore] // Added 2026-03-13 because it randomly fails sometimes. Unclear why
    // 2026-08-26: the "randomly fails" was very likely netcdf-c/HDF5 not being
    // thread-safe for concurrent open/create across tests -- see the
    // `#[serial_test::serial(netcdf)]` added here and on inspect_ridal_netcdf's
    // tests below, which introduced enough concurrent netcdf::create/open calls
    // to make the same underlying race reproduce on every run instead of
    // occasionally. Keeping the retry as a second line of defense.
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_save_netcdf() {
        let mut gpr = crate::gpr::tests::make_dummy_gpr(100, 10, Some(1.));

        let mut gpr2 = crate::gpr::tests::make_dummy_gpr(100, 10, Some(1.));
        gpr2.metadata.data_filepath = PathBuf::from_str("other_filepath.rd3").unwrap();

        gpr.merge(&gpr2).unwrap();
        gpr.process("subset(0 50)").unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let nc_path = temp_dir.path().join("data.nc");

        gpr.export(&nc_path).unwrap();

        assert!(nc_path.is_file());

        std::thread::sleep(std::time::Duration::from_millis(200));
        let out = netcdf::open(&nc_path)
            .map_err(|e| format!("Error reading NetCDF: {e:?}"))
            .unwrap();

        let expected_attrs = vec![
            (
                "processing_steps",
                netcdf::AttributeValue::Strs(vec!["subset(0 50)".to_string()]),
            ),
            (
                "processing_log",
                netcdf::AttributeValue::Str(
                    "merge (duration: 0.00s):\tMerged \"other_filepath.rd3\"\nsubset (duration: 0.00s):\tSubset data from [10, 200] to (0:10, 0:50)"
                        .to_string(),
                ),
            ),
            ("total_distance", netcdf::AttributeValue::Double(49.)),
            (
                "original_filepaths",
                netcdf::AttributeValue::Strs(vec![
                    "filepath.rd3".to_string(),
                    "other_filepath.rd3".to_string(),
                ]),
            ),
        ];

        let grid_mapping = out.variable("projected_crs").unwrap();
        assert_eq!(
            grid_mapping
                .attribute("grid_mapping_name")
                .unwrap()
                .value()
                .unwrap(),
            netcdf::AttributeValue::Str("transverse_mercator".into())
        );
        assert_eq!(
            grid_mapping
                .attribute("false_easting")
                .unwrap()
                .value()
                .unwrap(),
            netcdf::AttributeValue::Double(500000.0)
        );

        // Load the data and check that it's identical
        let mut data = ndarray::Array2::<f32>::zeros((gpr.height(), gpr.width()));
        out.variable("data")
            .unwrap()
            .get_into(data.view_mut(), ..)
            .unwrap();
        assert_eq!((data - gpr.data).mapv(|v| v.abs()).sum(), 0.);

        for (key, expected) in expected_attrs {
            assert_eq!(
                out.attribute(key)
                    .ok_or(format!("Cannot find attribute {key}"))
                    .unwrap()
                    .value()
                    .unwrap(),
                expected
            );
        }
    }

    fn export_dummy_ridal_nc(path: &std::path::Path) {
        let gpr = crate::gpr::tests::make_dummy_gpr(20, 10, Some(1.));
        gpr.export(path).unwrap();
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_supported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("supported.nc");
        export_dummy_ridal_nc(&path);

        match super::inspect_ridal_netcdf(&path).unwrap() {
            super::RidalNetcdfKind::Supported(meta) => {
                assert_eq!(meta.radargram_id.as_str(), "test-radargram");
                assert_eq!(meta.display_name, None);
                assert_eq!(meta.group_name, None);
                assert_eq!(meta.group_id, None);
                assert_eq!(meta.shape, (10, 20)); // (n_samples, n_traces)
                assert!(meta.ridal_version.contains("ridal version"));
            }
            other => panic!("expected Supported, got {other:?}"),
        }
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_unrelated_file_is_not_ridal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unrelated.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            file.add_dimension("x", 3).unwrap();
            let mut var = file.add_variable::<f32>("temperature", &["x"]).unwrap();
            var.put_values(&[1.0f32, 2.0, 3.0], ..).unwrap();
        }

        assert_eq!(
            super::inspect_ridal_netcdf(&path).unwrap(),
            super::RidalNetcdfKind::NotRidal
        );
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_rejects_unprefixed_legacy_attrs() {
        // Unprefixed processing_datetime/program_version, with no
        // ridal_processing_datetime/ridal_version at all: not recognized.
        // A prior version of this function fell back to these unprefixed
        // names, on the theory that a file might predate the ridal_*
        // rename (#116) while still having ridal_radargram_id -- but that
        // combination can never occur, since ridal_radargram_id became
        // mandatory in the exact same change that introduced the rename.
        // Any file with only the unprefixed names necessarily also lacks
        // ridal_radargram_id, and is rejected on that check regardless.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            file.add_dimension("y", 2).unwrap();
            file.add_dimension("x", 2).unwrap();
            let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
            var.put_values(&[0.0f32, 0., 0., 0.], ..).unwrap();
            file.add_attribute("processing_datetime", "2020-01-01T00:00:00Z")
                .unwrap();
            file.add_attribute("program_version", "ridal version 0.1.0 by test")
                .unwrap();
            file.add_attribute("ridal_radargram_id", "legacy-radargram")
                .unwrap();
        }

        assert_eq!(
            super::inspect_ridal_netcdf(&path).unwrap(),
            super::RidalNetcdfKind::NotRidal
        );
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_missing_radargram_id_is_not_ridal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_id.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            file.add_dimension("y", 2).unwrap();
            file.add_dimension("x", 2).unwrap();
            let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
            var.put_values(&[0.0f32, 0., 0., 0.], ..).unwrap();
            file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
                .unwrap();
            file.add_attribute("ridal_version", "ridal version 0.1.0 by test")
                .unwrap();
            // Deliberately no ridal_radargram_id.
        }

        assert_eq!(
            super::inspect_ridal_netcdf(&path).unwrap(),
            super::RidalNetcdfKind::NotRidal
        );
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_malformed_id_is_not_ridal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_id.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            file.add_dimension("y", 2).unwrap();
            file.add_dimension("x", 2).unwrap();
            let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
            var.put_values(&[0.0f32, 0., 0., 0.], ..).unwrap();
            file.add_attribute("ridal_processing_datetime", "2020-01-01T00:00:00Z")
                .unwrap();
            file.add_attribute("ridal_version", "ridal version 0.1.0 by test")
                .unwrap();
            file.add_attribute("ridal_radargram_id", "Not A Valid ID!")
                .unwrap();
        }

        assert_eq!(
            super::inspect_ridal_netcdf(&path).unwrap(),
            super::RidalNetcdfKind::NotRidal
        );
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_unreadable_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.nc");
        std::fs::write(&path, b"this is not a netcdf file").unwrap();

        let result = super::inspect_ridal_netcdf(&path);
        assert!(result.is_err(), "expected an error, got {result:?}");
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn test_inspect_ridal_netcdf_nonexistent_file_is_an_error() {
        let result = super::inspect_ridal_netcdf(std::path::Path::new("/no/such/file.nc"));
        assert!(result.is_err());
    }
}
