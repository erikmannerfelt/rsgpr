/// Tools to read elevation from Digital Elevation Models (DEMs)
use std::path::Path;

use crate::coords::Coord;

pub fn sample_dem(dem_path: &Path, coords_wgs84: &Vec<Coord>) -> Result<Vec<f32>, String> {
    use std::io::Write;

    if coords_wgs84.is_empty() {
        return Err("Coords vec is empty.".into());
    }

    let args = vec![
        "-xml",
        "-b",
        "1",
        "-wgs84",
        "-r",
        "bilinear",
        dem_path.to_str().ok_or("Empty DEM path given")?,
    ];
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

    use std::path::Path;

    use crate::coords::{Coord, Crs, UtmCrs};

    #[test]
    fn test_read_elevations() {
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
