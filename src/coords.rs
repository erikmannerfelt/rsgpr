#[derive(Debug, Copy, Clone)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    fn to_geomorph_coord(&self) -> geomorph::Coord {
        geomorph::Coord {
            lat: self.y,
            lon: self.x,
        }
    }

    fn to_geomorph_utm(&self, crs: &UtmCrs) -> geomorph::Utm {
        let band = match crs.north {
            true => 'N',
            false => 'S',
        };
        geomorph::Utm {
            easting: self.x,
            northing: self.y,
            north: crs.north,
            zone: crs.zone as i32,
            band,
            ups: false,
        }
    }

    fn to_wgs84(&self, crs: &UtmCrs) -> Self {
        let crd: geomorph::Coord = self.to_geomorph_utm(crs).into();
        Self {
            x: crd.lon,
            y: crd.lat,
        }
    }

    fn from_wgs84(&self, crs: &UtmCrs) -> Self {
        let (mut northing, easting, _) = utm::to_utm_wgs84(self.y, self.x, crs.zone as u8);

        // Edge case exceptions since the utm crate doesn't care about N/S
        if !crs.north & (self.y > 0.) {
            northing += 10000000.;
        } else if crs.north & (self.y < 0.) {
            northing -= 10000000.;
        }

        Self {
            x: easting,
            y: northing,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct UtmCrs {
    pub zone: usize,
    pub north: bool,
}

impl UtmCrs {
    pub fn optimal_crs(coord: &Coord) -> Self {
        let utm: geomorph::Utm = coord.to_geomorph_coord().into();
        Self {
            zone: utm.zone as usize,
            north: utm.north,
        }
    }

    pub fn to_epsg_str(&self) -> String {
        let mut epsg = "EPSG:32".to_string();

        if self.north {
            epsg += "6";
        } else {
            epsg += "8";
        }
        epsg += &format!("{}", self.zone);
        epsg
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Crs {
    Utm(UtmCrs),
    Proj(String),
}

impl Crs {
    pub fn from_user_input(text: &str) -> Result<Self, String> {
        let utm_result = parse_crs_utm(text);
        if let Ok(utm) = utm_result {
            return Ok(Self::Utm(utm));
        }
        let proj_result = proj_parse_crs(text);

        if let Ok(proj_str) = proj_result {
            if proj_str.contains("+proj=utm")
                & proj_str.contains("+zone=")
                & proj_str.contains("+datum=WGS84")
            {
                let utm_zone: usize = proj_str
                    .split("+zone=")
                    .last()
                    .unwrap()
                    .split(" ")
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                return Ok(Crs::Utm(UtmCrs {
                    zone: utm_zone,
                    north: !proj_str.contains("+south"),
                }));
            }

            return Ok(Crs::Proj(proj_str.to_string()));
        }

        Err(format!(
            "Could not read CRS.\nInternal error: {}.\nProj error: {}",
            utm_result.err().unwrap(),
            proj_result.err().unwrap()
        ))
    }
}

fn parse_crs_utm(text: &str) -> Result<UtmCrs, String> {
    let parts = text
        .to_lowercase()
        .trim()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    if let Some(first) = parts.first() {
        // Try EPSG:32XXX format
        if first.contains("epsg") {
            let code = first.replace(":", "").replace("epsg", "");
            if !code.starts_with("32") | (code.len() != 5) {
                return Err(format!("EPSG code is not a WGS84 UTM zone: {text}"));
            }

            let (north, start) = match code.chars().nth(2) {
                Some('6') => Ok((true, "326")),
                Some('8') => Ok((false, "328")),
                _ => Err(format!("EPSG code is not a WGS84 UTM zone: {text}")),
            }?;

            if let Ok(zone) = code.replace(start, "").parse::<usize>() {
                return Ok(UtmCrs { zone, north });
            }
        };

        // Try "WGS84 UTM Zone 33 N" format
        if ["wgs84", "wgs1984"].iter().any(|s| first.starts_with(s)) {
            if let Some(mut zone_number) = parts.get(3).map(|v| v.to_string()) {
                let mut north_south: Option<bool> = None;
                if zone_number.contains("n") {
                    north_south = Some(true);
                    zone_number = zone_number.replace("n", "");
                } else if zone_number.contains("s") {
                    north_south = Some(false);
                    zone_number = zone_number.replace("s", "");
                }

                if north_south.is_none() {
                    if let Some(n_s) = parts.get(4) {
                        if n_s.contains("n") {
                            north_south = Some(true);
                        } else if n_s.contains("s") {
                            north_south = Some(false);
                        }
                    }
                }
                if let Ok(number) = zone_number.parse::<usize>() {
                    if let Some(north) = north_south {
                        return Ok(UtmCrs {
                            zone: number,
                            north,
                        });
                    } else {
                        return Err(format!("UTM zone letter not provided or invalid: {text}"));
                    }
                }
            }
        } else {
            return Err(format!("CRS parse error. No 'WGS84' string in {text}"));
        }
    } else {
        return Err(format!("CRS parse error. No whitespaces in {text}"));
    }

    Err(format!("CRS parse error: {text}"))
}

fn proj_parse_crs(text: &str) -> Result<String, String> {
    use std::io::BufRead;
    let mut child = std::process::Command::new("projinfo")
        .arg(text)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let stdout = child.stdout.take().expect("Failed to open stdout");
    let reader = std::io::BufReader::new(stdout);

    let mut output = String::new();
    // Read output line by line
    let mut next = false;
    for line in reader.lines() {
        let line = line.unwrap();
        // Check if the line contains a PROJ.4 definition
        if next {
            output.push_str(line.trim());
            break;
        };
        if line.starts_with("PROJ.4 string:") {
            // Extract and print the PROJ.4 string
            next = true;
        }
    }

    // Ensure the command completes successfully
    let _ = child.wait().unwrap();

    match next {
        false => Err("Could not find proj string for given CRS.".into()),
        true => match output.is_empty() {
            true => Err("Proj string was empty.".into()),
            false => Ok(output),
        },
    }
}
fn proj_convert_to_wgs84(x: &[f64], y: &[f64], crs: &str) -> Result<Vec<Coord>, String> {
    proj_convert_crs(x, y, crs, "+init=epsg:4326")
}
fn proj_convert_from_wgs84(x: &[f64], y: &[f64], crs: &str) -> Result<Vec<Coord>, String> {
    proj_convert_crs(x, y, "+init=epsg:4326", crs)
}

fn proj_convert_crs(
    x: &[f64],
    y: &[f64],
    src_crs: &str,
    dst_crs: &str,
) -> Result<Vec<Coord>, String> {
    let mut new_coords = Vec::<Coord>::new();

    use std::io::BufRead;
    use std::io::Write;
    let proj_conv_str = format!("{src_crs} +to {dst_crs} -f %.4f")
        .split(" ")
        .map(|s| s.to_string())
        .collect::<Vec<String>>();
    let mut child = std::process::Command::new("cs2cs")
        .args(proj_conv_str)
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();

    let mut values = Vec::<String>::new();
    for i in 0..x.len() {
        values.push(format!("{} {}", x[i], y[i]));
    }

    stdin
        .write_all((values.join("\n") + "\n").as_bytes())
        .unwrap();
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let reader = std::io::BufReader::new(stdout);
    for line in reader.lines() {
        let line = line.unwrap();

        let values: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        new_coords.push(Coord {
            x: values[0],
            y: values[1],
        });

        if new_coords.len() == x.len() {
            break;
        };
    }

    child.kill().unwrap();
    Ok(new_coords)
}

pub fn to_wgs84(coords: &[Coord], crs: &Crs) -> Result<Vec<Coord>, String> {
    let mut new_coords = Vec::<Coord>::new();
    match crs {
        Crs::Utm(utm) => {
            for coord in coords {
                new_coords.push(coord.to_wgs84(&utm));
            }
        }
        Crs::Proj(proj_str) => {
            let mut eastings = Vec::<f64>::new();
            let mut northings = eastings.clone();

            for coord in coords {
                eastings.push(coord.x);
                northings.push(coord.y);
            }
            new_coords.append(&mut proj_convert_to_wgs84(&eastings, &northings, proj_str)?);
        }
    }

    Ok(new_coords)
}

pub fn from_wgs84(coords: &[Coord], crs: &Crs) -> Result<Vec<Coord>, String> {
    let mut new_coords = Vec::<Coord>::new();
    match crs {
        Crs::Utm(utm) => {
            for coord in coords {
                new_coords.push(coord.from_wgs84(utm));
            }
        }
        Crs::Proj(proj_str) => {
            let mut eastings = Vec::<f64>::new();
            let mut northings = eastings.clone();

            for coord in coords {
                eastings.push(coord.x);
                northings.push(coord.y);
            }
            new_coords.append(&mut proj_convert_from_wgs84(
                &eastings, &northings, proj_str,
            )?);
        }
    }

    Ok(new_coords)
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::{Coord, Crs, UtmCrs};

    fn coords_approx_eq(first: &Coord, second: &Coord, precision: f64) -> bool {
        let xdiff = first.x - second.x;
        let ydiff = first.y - second.y;
        (xdiff.powi(2) + ydiff.powi(2)).sqrt() < precision
    }

    fn make_test_cases() -> Vec<(String, Crs)> {
        vec![
            (
                "EPSG:32633".into(),
                Crs::Utm(UtmCrs {
                    zone: 33,
                    north: true,
                }),
            ),
            (
                "WGS84 UTM Zone 33S".into(),
                Crs::Utm(UtmCrs {
                    zone: 33,
                    north: false,
                }),
            ),
            (
                "EPSG:3006".into(),
                Crs::Proj("+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs".into()),

            )
        ]
    }

    #[test]
    fn test_crs_from_user() {
        for (crs_str, expected) in make_test_cases() {
            let _parsed_proj = super::proj_parse_crs(&crs_str).unwrap();

            let parsed = super::Crs::from_user_input(&crs_str).unwrap();

            println!("Expected: {:?}", expected);
            println!("Received: {:?}", parsed);

            assert_eq!(parsed.type_id(), expected.type_id());
            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_parse_utm() {
        for (crs_str, expected_any) in make_test_cases() {
            if let Crs::Utm(expected) = expected_any {
                let res = super::parse_crs_utm(&crs_str);

                assert_eq!(res, Ok(expected));
            }
        }
        let failures = vec![
            ("EPSG:3006", "EPSG code is not a WGS84"),
            ("ETRS89 UTM Zone 33N", "No 'WGS84' string in"),
            ("WGS84UTMZone33N", "CRS parse error"),
            (
                "WGS84 UTM Zone 33 X",
                "UTM zone letter not provided or invalid",
            ),
        ];

        for (failure, expected_err) in failures {
            if let Err(err_str) = super::parse_crs_utm(&failure) {
                if !err_str.contains(expected_err) {
                    panic!("Expected '{expected_err}' in '{err_str}'");
                }
            } else {
                panic!("Should have failed on {failure}")
            }
        }
    }

    #[test]
    fn test_optimal_crs() {
        let crs = Crs::Utm(UtmCrs::optimal_crs(&Coord { y: 78., x: 15. }));

        match crs {
            Crs::Utm(utm) => {
                assert_eq!(utm.zone, 33);
                assert_eq!(utm.north, true);
            }
            _ => panic!(),
        };
    }

    #[test]
    fn test_crs_convert() {
        let coords = vec![
            Coord { x: 15., y: 78. },
            Coord { x: 0., y: 1. },
            Coord { x: 15., y: -78. },
        ];
        for (crs_str, _) in make_test_cases() {
            println!("Converting with {crs_str}");
            let parsed = super::Crs::from_user_input(&crs_str).unwrap();
            let conv = super::from_wgs84(&coords, &parsed).unwrap();
            let conv_back = super::to_wgs84(&conv, &parsed).unwrap();

            for i in 0..conv_back.len() {
                println!("{:?} -> {:?} -> {:?}", coords[i], conv[i], conv_back[i]);

                assert!(coords_approx_eq(&coords[i], &conv_back[i], 0.01));
            }
        }
    }
}
