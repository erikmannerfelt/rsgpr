/// Tools to read elevation from Digital Elevation Models (DEMs)
use std::path::Path;

use crate::coords::Coord;

fn get_gdal_version() -> Result<String, String> {
    let child = std::process::Command::new("gdalinfo")
        .arg("--version")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.to_string().contains("No such file or directory") {
                format!("GDAL (gdalinfo) cannot be found / is not installed: {e}")
            } else {
                format!("Call error when spawning process: {e}")
            }
        })?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Call failed: {e}"))?;

    if output.status.success() {
        let mut version = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_string()
            .replace("GDAL ", "");

        if let Some((first, _)) = version.split_once(",") {
            version = first.trim().to_string();
        }
        Ok(version)
    } else if output.stderr.is_empty() {
        Err("Unknown error getting GDAL version.".to_string())
    } else {
        Err(format!(
            "Error getting GDAL version: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn supports_interpolation() -> Result<bool, String> {
    let version = get_gdal_version()?;

    let mut parts = version.split(".");
    let error_msg = format!("Unrecognized version format: {version}");

    if let Some(major) = parts.next() {
        if major.parse::<usize>().map_err(|_| error_msg.clone())? < 3 {
            return Ok(false);
        }
    }

    if let Some(minor) = parts.next() {
        let minor = minor.parse::<usize>().map_err(|_| error_msg.clone())?;

        return Ok(minor >= 10);
    }

    Err(error_msg)
}

pub fn sample_dem(dem_path: &Path, coords_wgs84: &Vec<Coord>) -> Result<Vec<f32>, String> {
    use std::io::Write;

    if coords_wgs84.is_empty() {
        return Err("Coords vec is empty.".into());
    }

    let mut args = vec![
        "-xml",
        "-b",
        "1",
        "-wgs84",
        dem_path.to_str().ok_or("Empty DEM path given")?,
    ];
    if supports_interpolation()? {
        args.push("-r");
        args.push("bilinear");
    } else {
        eprintln!("GDAL version lower than 3.10. Falling back on nearest neighbor sampling.");
    }
    let mut child = std::process::Command::new("gdallocationinfo")
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Call error when spawning process: {e}"))?;

    let mut values = Vec::<String>::new();
    for coord in coords_wgs84 {
        values.push(format!("{} {}", coord.x, coord.y));
    }
    child
        .stdin
        .take()
        .ok_or("Call error: stdin could not be bound".to_string())?
        .write_all((values.join("\n") + "\n").as_bytes())
        .map_err(|e| format!("Call error writing to stdin: {e}"))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Call process error: {e}"))?;
    let parsed = String::from_utf8_lossy(&output.stdout);

    let mut elevations = Vec::<f32>::new();
    for line in parsed.lines().map(|s| s.trim()) {
        if line.contains("<Value>") {
            elevations.push(
                line.replace("<Value>", "")
                    .replace("</Value>", "")
                    .parse()
                    .map_err(|e| format!("Error parsing <Value>: {e}"))?,
            );
        } else if line.contains("<Alert>") {
            let error = line.replace("<Alert>", "").replace("</Alert>", "");
            let coord = coords_wgs84[elevations.len()];

            return Err(format!(
                "Error parsing coord (lon: {:.3}, lat: {:.3}): {}",
                coord.x, coord.y, error
            ));
        }
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

#[cfg(test)]
mod tests {

    use std::path::{Path, PathBuf};

    use crate::coords::{Coord, Crs, UtmCrs};

    fn get_dem_path() -> PathBuf {
        Path::new("assets/test_dem_dtm20_mettebreen.tif").to_path_buf()
    }

    fn make_test_coords() -> Vec<(Coord, Result<f32, String>)> {
        vec![
            (
                Coord {
                    x: 553802.,
                    y: 8639550.,
                },
                Ok(422.0352_f32),
            ),
            (
                Coord {
                    x: 553820.,
                    y: 8639550.,
                },
                Ok(423.3629_f32),
            ),
            (
                Coord { x: 0., y: 8639550. },
                Err("Location is off this file".to_string()),
            ),
        ]
    }

    #[test]
    #[serial_test::serial]
    fn test_read_elevations() {
        let coords_elevs = make_test_coords();
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

        let dem_path = get_dem_path();

        println!("Sampling DEM");
        let coords_wgs84 = crate::coords::to_wgs84(&working_coords, &crs).unwrap();
        super::sample_dem(&dem_path, &coords_wgs84).unwrap();

        let coords_wgs84 = crate::coords::to_wgs84(&all_coords, &crs).unwrap();
        super::sample_dem(&dem_path, &coords_wgs84).err().unwrap();

        for (coord, expected) in coords_elevs {
            let coord_wgs84 = crate::coords::to_wgs84(&[coord], &crs).unwrap();

            let result = super::sample_dem(&dem_path, &coord_wgs84);

            if let Ok(expected_elevation) = expected {
                if !super::supports_interpolation().unwrap() {
                    eprintln!(
                        "Skipping elevation Ok assertion because interpolation is not supported."
                    );
                    continue;
                }
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
        let wrong_path = dem_path.with_extension("tiffffff");
        assert!(super::sample_dem(&wrong_path, &coords_wgs84)
            .err()
            .unwrap()
            .contains("No such file or directory"));
    }

    #[test]
    #[serial_test::serial]
    fn test_no_gdal_failure() {
        let crs = Crs::Utm(UtmCrs {
            zone: 33,
            north: true,
        });

        let working_coords: Vec<Coord> = make_test_coords().iter().map(|(c, _)| *c).collect();

        let dem_path = get_dem_path();

        println!("Sampling DEM");
        let coords_wgs84 = crate::coords::to_wgs84(&working_coords, &crs).unwrap();
        // super::sample_dem(&dem_path, &coords_wgs84, None).unwrap();
        // let original_path = std::env::var("PATH").unwrap();
        // let temp_path = "/some/empty/directory";
        // std::env::set_var("PATH", temp_path);

        temp_env::with_vars(vec![("PATH", Option::<&str>::None)], || {
            if super::get_gdal_version().is_ok() {
                eprintln!("WARNING: Could not properly unset the GDAL location. Skipping test.");
                return;
            };
            let res = super::sample_dem(&dem_path, &coords_wgs84);
            assert!(res
                .err()
                .unwrap()
                .contains("GDAL (gdalinfo) cannot be found / is not installed"));
        });

        // Restore the original PATH
        // std::env::set_var("PATH", original_path);
    }
}
