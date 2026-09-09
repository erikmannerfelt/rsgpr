/// Tools to read elevation from Digital Elevation Models (DEMs)
use std::path::Path;

use crate::coords::Coord;
use std::io::Write;

/// Describe a failure to launch one of GDAL's command-line tools.
///
/// GDAL is an external dependency, so "not installed" is an ordinary thing
/// for a user to hit rather than a bug. The bare io error is
/// `No such file or directory (os error 2)`, which does not say *which*
/// file -- easily read as the DEM being missing when it is GDAL itself.
///
/// Matches on [`std::io::ErrorKind`] rather than the message text, which is
/// platform- and locale-dependent.
fn spawn_error(program: &str, error: &std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        format!("GDAL ({program}) cannot be found / is not installed: {error}")
    } else {
        format!("Call error when spawning process: {error}")
    }
}

fn run_gdallocationinfo(
    dem_path: &Path,
    coords_wgs84: &[Coord],
    use_bilinear: bool,
) -> Result<std::process::Output, String> {
    let dem_str = dem_path.to_str().ok_or("Empty DEM path given")?;

    let mut args = vec!["-xml", "-b", "1", "-wgs84", dem_str];

    if use_bilinear {
        args.push("-r");
        args.push("bilinear");
    }

    let mut child = std::process::Command::new("gdallocationinfo")
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| spawn_error("gdallocationinfo", &e))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or("Call error: stdin could not be bound".to_string())?;

        let mut buf = String::new();
        for coord in coords_wgs84 {
            buf.push_str(&format!("{} {}\n", coord.x, coord.y));
        }

        stdin
            .write_all(buf.as_bytes())
            .map_err(|e| format!("Call error writing to stdin: {e}"))?;
    }

    child
        .wait_with_output()
        .map_err(|e| format!("Call process error: {e}"))
}

/// Helper: does stdout contain any XML result tags?
fn has_result_tags(stdout: &[u8]) -> bool {
    let s = String::from_utf8_lossy(stdout);
    s.contains("<Value>") || s.contains("<Alert>")
}

fn parse_elevations_from_output(
    output: &std::process::Output,
    coords_wgs84: &[Coord],
) -> Result<Vec<f32>, String> {
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
        let stderr_str = String::from_utf8_lossy(&output.stderr);

        if !stderr_str.is_empty() {
            return Err(format!("DEM sampling error: {}", stderr_str));
        }

        // Empty output but no stderr - gdallocationinfo may have failed silently
        if elevations.is_empty() {
            return Err("DEM sampling failed. gdallocationinfo returned no data. Check that GDAL is properly installed and in PATH.".to_string());
        }

        return Err(format!(
            "Shape error. Length of sampled elevations ({}) does not align with length of coordinates ({})",
            elevations.len(),
            coords_wgs84.len()
        ));
    }

    Ok(elevations)
}

pub fn sample_dem(dem_path: &Path, coords_wgs84: &[Coord]) -> Result<Vec<f32>, String> {
    if coords_wgs84.is_empty() {
        return Err("Coords vec is empty.".into());
    }

    // First: try with bilinear
    let first_output = run_gdallocationinfo(dem_path, coords_wgs84, true)?;

    if !has_result_tags(&first_output.stdout) {
        // No <Value> or <Alert> at all → likely unsupported -r bilinear /
        // invalid combo → fall back and retry without -r.
        eprintln!(
            "gdallocationinfo output failed (no output) with '-r bilinear'. \
             Falling back to nearest neighbor sampling."
        );

        let second_output = run_gdallocationinfo(dem_path, coords_wgs84, false)?;
        return parse_elevations_from_output(&second_output, coords_wgs84);
    }
    parse_elevations_from_output(&first_output, coords_wgs84)
}

#[cfg(test)]
mod tests {

    use std::path::{Path, PathBuf};

    use crate::coords::{Coord, Crs, UtmCrs};

    /// The PATH-manipulating test below can only run on some platforms and
    /// skips itself if GDAL is still reachable. This pins the same rule
    /// directly, everywhere.
    #[test]
    fn a_missing_gdal_is_reported_by_program_name() {
        let missing = std::io::Error::from(std::io::ErrorKind::NotFound);
        let message = super::spawn_error("gdallocationinfo", &missing);
        assert!(
            message.contains("GDAL (gdallocationinfo) cannot be found / is not installed"),
            "{message}"
        );

        // Anything else is not a missing install, and saying so would send
        // the reader off installing software they already have.
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let message = super::spawn_error("gdallocationinfo", &denied);
        assert!(!message.contains("is not installed"), "{message}");
        assert!(
            message.contains("Call error when spawning process"),
            "{message}"
        );
    }

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

    pub fn get_gdal_version() -> Result<String, String> {
        let child = std::process::Command::new("gdalinfo")
            .arg("--version")
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| super::spawn_error("gdalinfo", &e))?;

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

    pub fn supports_interpolation() -> Result<bool, String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Use the same DEM your tests use; you can generalize if needed
        let dem_path = crate::dem::tests::get_dem_path();
        let dem_str = dem_path
            .to_str()
            .ok_or("Empty DEM path given in supports_interpolation")?;

        // Single arbitrary coordinate; if the flag is supported, we should get XML with <Value>/<Alert>.
        let mut child = Command::new("gdallocationinfo")
            .args(["-xml", "-b", "1", "-wgs84", "-r", "bilinear", dem_str])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Error spawning gdallocationinfo: {e}"))?;

        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or("stdin could not be bound in supports_interpolation".to_string())?;
            // Some coordinate; doesn't matter as long as it's a valid numeric pair.
            stdin
                .write_all(b"0 0\n")
                .map_err(|e| format!("Error writing to stdin in supports_interpolation: {e}"))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Error waiting for gdallocationinfo: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // If the option is supported, we should see some XML structure; if not, we'll typically
        // just get usage/help on stderr and nothing useful on stdout.
        Ok(stdout.contains("<Value>") || stdout.contains("<Alert>"))
    }

    #[test]
    #[cfg(not(target_os = "windows"))] // Added 2026-04-12 because gdal stopped working properly in CI
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

        let supports_interpolation = supports_interpolation().unwrap();

        println!("Sampling DEM");
        let coords_wgs84 = crate::coords::to_wgs84(&working_coords, &crs).unwrap();
        super::sample_dem(&dem_path, &coords_wgs84).unwrap();

        let coords_wgs84 = crate::coords::to_wgs84(&all_coords, &crs).unwrap();
        super::sample_dem(&dem_path, &coords_wgs84).err().unwrap();

        for (coord, expected) in coords_elevs {
            let coord_wgs84 = crate::coords::to_wgs84(&[coord], &crs).unwrap();

            let result = super::sample_dem(&dem_path, &coord_wgs84);

            // The tests validate on bilinearly interpolated coordinates. This will fail if it's nearest
            if supports_interpolation {
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
        }
        let wrong_path = dem_path.with_extension("tiffffff");
        assert!(super::sample_dem(&wrong_path, &coords_wgs84)
            .err()
            .unwrap()
            .contains("No such file or directory"));
    }

    #[test]
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
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
            if get_gdal_version().is_ok() {
                eprintln!("WARNING: Could not properly unset the GDAL location. Skipping test.");
                return;
            };
            let res = super::sample_dem(&dem_path, &coords_wgs84);
            // Names the program Ridal actually tried to run. `sample_dem`
            // spawns `gdallocationinfo`, not `gdalinfo`, and saying so is
            // the whole point: "No such file or directory" alone reads as
            // the DEM being missing.
            assert!(
                res.as_ref()
                    .err()
                    .unwrap()
                    .contains("GDAL (gdallocationinfo) cannot be found / is not installed"),
                "Error {:?} should name the missing GDAL program",
                res
            );
        });

        // Restore the original PATH
        // std::env::set_var("PATH", original_path);
    }
}
