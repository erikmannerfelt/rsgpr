use std::collections::BTreeMap;
#[derive(Debug, Copy, Clone)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    fn to_geomorph_coord(self) -> geomorph::Coord {
        geomorph::Coord {
            lat: self.y,
            lon: self.x,
        }
    }

    fn to_geomorph_utm(self, crs: &UtmCrs) -> geomorph::Utm {
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

    fn to_wgs84(self, crs: &UtmCrs) -> Self {
        let crd: geomorph::Coord = self.to_geomorph_utm(crs).into();
        Self {
            x: crd.lon,
            y: crd.lat,
        }
    }

    fn conv_from_wgs84(self, crs: &UtmCrs) -> Self {
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
                    .ok_or("Expected '+zone=' in proj4 string")?
                    .split(" ")
                    .next()
                    .ok_or("Expected whitespace after '+zone=<..>' in proj4 string")?
                    .parse()
                    .map_err(|e| format!("Could not parse '+zone=' value in proj4 string: {e}"))?;
                return Ok(Crs::Utm(UtmCrs {
                    zone: utm_zone,
                    north: !proj_str.contains("+south"),
                }));
            }

            return Ok(Crs::Proj(proj_str.to_string()));
        }

        Err(format!(
            "Could not read CRS.\nInternal error: {}.\nProj error: {}",
            utm_result.err().unwrap_or("None".into()),
            proj_result.err().unwrap_or("None".into())
        ))
    }
}

fn parse_crs_utm(text: &str) -> Result<UtmCrs, String> {
    let parts = text
        .to_lowercase()
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
    let child = std::process::Command::new("projinfo")
        .arg(text)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.to_string().contains("No such file or directory") {
                format!("PROJ (projinfo) cannot be found / is not installed: {e}")
            } else {
                format!("Call error when spawning process: {e}")
            }
        })?;

    let result = child
        .wait_with_output()
        .map_err(|e| format!("Call process error: {e}"))?;
    let parsed = String::from_utf8_lossy(&result.stdout);

    let mut output = String::new();
    // Read output line by line
    let mut next = false;
    for line in parsed.lines() {
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

    use std::io::Write;
    let proj_conv_str = format!("{src_crs} +to {dst_crs} -f %.4f")
        .split(" ")
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let mut values = Vec::<String>::new();
    for i in 0..x.len() {
        values.push(format!("{} {}", x[i], y[i]));
    }
    let mut child = std::process::Command::new("cs2cs")
        .args(proj_conv_str)
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Call error when spawning process: {e}"))?;

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

    for line in parsed.lines() {
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

    Ok(new_coords)
}

pub fn to_wgs84(coords: &[Coord], crs: &Crs) -> Result<Vec<Coord>, String> {
    let mut new_coords = Vec::<Coord>::new();
    match crs {
        Crs::Utm(utm) => {
            for coord in coords {
                new_coords.push(coord.to_wgs84(utm));
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
                new_coords.push(coord.conv_from_wgs84(utm));
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
#[derive(Clone, Debug)]
pub struct GridMappingSpec {
    pub variable_name: String,
    pub attrs: BTreeMap<String, crate::export::ExportAttr>,
}

fn parse_proj_params(proj_str: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for token in proj_str.split_whitespace() {
        if !token.starts_with('+') {
            continue;
        }
        let token = &token[1..];
        if let Some((k, v)) = token.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        } else {
            // flag-style parameter such as +south
            out.insert(token.to_string(), String::new());
        }
    }
    out
}

fn param_f64(params: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    params.get(key)?.parse::<f64>().ok()
}

fn insert_attr_f64(
    attrs: &mut BTreeMap<String, crate::export::ExportAttr>,
    params: &BTreeMap<String, String>,
    proj_key: &str,
    cf_key: &str,
) {
    if let Some(v) = param_f64(params, proj_key) {
        attrs.insert(cf_key.to_string(), v.into());
    }
}

fn insert_known_ellipsoid_attrs(
    attrs: &mut BTreeMap<String, crate::export::ExportAttr>,
    params: &BTreeMap<String, String>,
) {
    // Prefer explicit numeric axes if present
    if let Some(a) = param_f64(params, "a") {
        attrs.insert("semi_major_axis".into(), a.into());

        if let Some(rf) = param_f64(params, "rf") {
            attrs.insert("inverse_flattening".into(), rf.into());
            return;
        }

        if let Some(b) = param_f64(params, "b") {
            if (a - b).abs() > f64::EPSILON {
                let inv_f = a / (a - b);
                attrs.insert("inverse_flattening".into(), inv_f.into());
            } else {
                attrs.insert("inverse_flattening".into(), 0.0f64.into());
            }
            return;
        }
    }

    // Common named ellipsoids / datums
    if let Some(ellps) = params.get("ellps") {
        match ellps.as_str() {
            "WGS84" => {
                attrs.insert("semi_major_axis".into(), 6378137.0f64.into());
                attrs.insert("inverse_flattening".into(), 298.257223563f64.into());
            }
            "GRS80" => {
                attrs.insert("semi_major_axis".into(), 6378137.0f64.into());
                attrs.insert("inverse_flattening".into(), 298.257222101f64.into());
            }
            _ => {}
        }
        return;
    }

    if Some(&"WGS84".to_string()) == params.get("datum") {
        attrs.insert("semi_major_axis".into(), 6378137.0f64.into());
        attrs.insert("inverse_flattening".into(), 298.257223563f64.into());
    }
}

fn utm_grid_mapping_attrs(utm: &UtmCrs) -> BTreeMap<String, crate::export::ExportAttr> {
    let mut attrs = BTreeMap::new();

    let lon0 = (utm.zone as f64) * 6.0 - 183.0;
    let false_northing = if utm.north { 0.0 } else { 10_000_000.0 };

    attrs.insert("grid_mapping_name".into(), "transverse_mercator".into());
    attrs.insert("scale_factor_at_central_meridian".into(), 0.9996f64.into());
    attrs.insert("longitude_of_central_meridian".into(), lon0.into());
    attrs.insert("latitude_of_projection_origin".into(), 0.0f64.into());
    attrs.insert("false_easting".into(), 500_000.0f64.into());
    attrs.insert("false_northing".into(), false_northing.into());
    attrs.insert("semi_major_axis".into(), 6378137.0f64.into());
    attrs.insert("inverse_flattening".into(), 298.257223563f64.into());

    attrs
}

fn projinfo_to_wkt(definition: &str) -> Result<String, String> {
    let output = std::process::Command::new("projinfo")
        .args(["-o", "WKT2:2019", "--single-line", definition])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.to_string().contains("No such file or directory") {
                format!("PROJ (projinfo) cannot be found / is not installed: {e}")
            } else {
                format!("Call error when spawning projinfo: {e}")
            }
        })?
        .wait_with_output()
        .map_err(|e| format!("Call process error: {e}"))?;

    let text = String::from_utf8_lossy(&output.stdout);

    let mut take_next_nonempty = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Common case: header line, WKT on the next line
        if take_next_nonempty {
            return Ok(trimmed.to_string());
        }

        // Exact header emitted by projinfo for this output mode
        if trimmed == "WKT2:2019 string:" {
            take_next_nonempty = true;
            continue;
        }

        // Defensive fallback in case some projinfo version puts the WKT on the same line
        if let Some(rest) = trimmed.strip_prefix("WKT2:2019 string:") {
            let rest = rest.trim();
            if !rest.is_empty() {
                return Ok(rest.to_string());
            }
            take_next_nonempty = true;
        }
    }

    Err("Could not extract WKT2:2019 from projinfo output.".into())
}

fn proj_grid_mapping_attrs(
    crs_str: &str,
) -> Result<Option<BTreeMap<String, crate::export::ExportAttr>>, String> {
    let proj_str = proj_parse_crs(crs_str)?;
    let params = parse_proj_params(&proj_str);

    // If the PROJ string is really just WGS84 UTM, normalize to the UTM branch
    if params.get("proj").map(|s| s.as_str()) == Some("utm") {
        if let Some(zone_s) = params.get("zone") {
            if let Ok(zone) = zone_s.parse::<usize>() {
                let utm = UtmCrs {
                    zone,
                    north: !params.contains_key("south"),
                };
                let mut attrs = utm_grid_mapping_attrs(&utm);
                if let Ok(wkt) = projinfo_to_wkt(crs_str) {
                    attrs.insert("crs_wkt".into(), wkt.into());
                }
                return Ok(Some(attrs));
            }
        }
    }

    let proj_name = match params.get("proj").map(|s| s.as_str()) {
        Some("tmerc") => "transverse_mercator",
        Some("merc") => "mercator",
        Some("laea") => "lambert_azimuthal_equal_area",
        Some("aeqd") => "azimuthal_equidistant",
        Some("ortho") => "orthographic",
        Some("stere") => "stereographic",
        Some("geos") => "vertical_perspective",
        _ => return Ok(None),
    };

    let mut attrs = BTreeMap::new();
    attrs.insert("grid_mapping_name".into(), proj_name.into());

    match proj_name {
        "transverse_mercator" => {
            insert_attr_f64(
                &mut attrs,
                &params,
                "k_0",
                "scale_factor_at_central_meridian",
            );
            if !attrs.contains_key("scale_factor_at_central_meridian") {
                insert_attr_f64(&mut attrs, &params, "k", "scale_factor_at_central_meridian");
            }
            insert_attr_f64(
                &mut attrs,
                &params,
                "lon_0",
                "longitude_of_central_meridian",
            );
            insert_attr_f64(
                &mut attrs,
                &params,
                "lat_0",
                "latitude_of_projection_origin",
            );
            insert_attr_f64(&mut attrs, &params, "x_0", "false_easting");
            insert_attr_f64(&mut attrs, &params, "y_0", "false_northing");
        }
        "mercator" => {
            insert_attr_f64(
                &mut attrs,
                &params,
                "lon_0",
                "longitude_of_projection_origin",
            );
            insert_attr_f64(&mut attrs, &params, "lat_ts", "standard_parallel");
            if !attrs.contains_key("standard_parallel") {
                insert_attr_f64(
                    &mut attrs,
                    &params,
                    "k_0",
                    "scale_factor_at_projection_origin",
                );
                if !attrs.contains_key("scale_factor_at_projection_origin") {
                    insert_attr_f64(
                        &mut attrs,
                        &params,
                        "k",
                        "scale_factor_at_projection_origin",
                    );
                }
            }
            insert_attr_f64(&mut attrs, &params, "x_0", "false_easting");
            insert_attr_f64(&mut attrs, &params, "y_0", "false_northing");
        }
        "lambert_azimuthal_equal_area" | "azimuthal_equidistant" | "orthographic" => {
            insert_attr_f64(
                &mut attrs,
                &params,
                "lon_0",
                "longitude_of_projection_origin",
            );
            insert_attr_f64(
                &mut attrs,
                &params,
                "lat_0",
                "latitude_of_projection_origin",
            );
            insert_attr_f64(&mut attrs, &params, "x_0", "false_easting");
            insert_attr_f64(&mut attrs, &params, "y_0", "false_northing");
        }
        "stereographic" => {
            insert_attr_f64(
                &mut attrs,
                &params,
                "lon_0",
                "longitude_of_projection_origin",
            );
            insert_attr_f64(
                &mut attrs,
                &params,
                "lat_0",
                "latitude_of_projection_origin",
            );
            insert_attr_f64(
                &mut attrs,
                &params,
                "k_0",
                "scale_factor_at_projection_origin",
            );
            if !attrs.contains_key("scale_factor_at_projection_origin") {
                insert_attr_f64(
                    &mut attrs,
                    &params,
                    "k",
                    "scale_factor_at_projection_origin",
                );
            }
            insert_attr_f64(&mut attrs, &params, "x_0", "false_easting");
            insert_attr_f64(&mut attrs, &params, "y_0", "false_northing");
        }
        "vertical_perspective" => {
            insert_attr_f64(
                &mut attrs,
                &params,
                "lon_0",
                "longitude_of_projection_origin",
            );
            insert_attr_f64(
                &mut attrs,
                &params,
                "lat_0",
                "latitude_of_projection_origin",
            );
            insert_attr_f64(&mut attrs, &params, "h", "perspective_point_height");
            insert_attr_f64(&mut attrs, &params, "x_0", "false_easting");
            insert_attr_f64(&mut attrs, &params, "y_0", "false_northing");
        }
        _ => {}
    }

    insert_known_ellipsoid_attrs(&mut attrs, &params);

    if let Ok(wkt) = projinfo_to_wkt(crs_str) {
        attrs.insert("crs_wkt".into(), wkt.into());
    }

    Ok(Some(attrs))
}

pub fn build_grid_mapping_from_crs(crs_str: &str) -> Result<Option<GridMappingSpec>, String> {
    let crs = Crs::from_user_input(crs_str)?;
    let attrs = match crs {
        Crs::Utm(utm) => {
            let mut attrs = utm_grid_mapping_attrs(&utm);

            // Optional enhancement: if PROJ is available, also attach crs_wkt using the EPSG code
            if let Ok(wkt) = projinfo_to_wkt(crs_str) {
                attrs.insert("crs_wkt".into(), wkt.into());
            }

            Some(attrs)
        }
        Crs::Proj(_) => proj_grid_mapping_attrs(crs_str)?,
    };

    Ok(attrs.map(|attrs| GridMappingSpec {
        variable_name: "projected_crs".into(),
        attrs,
    }))
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
    #[serial_test::serial]
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
    #[serial_test::serial]
    #[cfg(not(target_os = "windows"))] // Added 2026-03-13 because the path unsetting logic doesn't work on Windows
    fn test_crs_noproj() {
        // This test simulates machines without PROJ installed. UTM CRSes should work but not others.
        temp_env::with_vars(vec![("PATH", Option::<&str>::None)], || {
            // This "complex" CRS should fail
            let res = super::Crs::from_user_input("EPSG:3006");

            if let Err(msg) = res {
                assert!(msg.contains("PROJ (projinfo) cannot be found / is not installed"))
            } else {
                eprintln!("WARNING: Could not properly unset the PROJ location. Skipping test.");
                return;
            }
            // A UTM CRS should still work.
            let parsed = super::Crs::from_user_input("EPSG:32633").unwrap();

            if let Crs::Utm(crs) = &parsed {
                assert_eq!(crs.zone, 33);
                assert_eq!(crs.north, true);
            } else {
                panic!("Wrong type of parsed CRS: {parsed:?}");
            }
            let coords = vec![
                Coord { x: 15., y: 78. },
                Coord { x: 0., y: 1. },
                Coord { x: 15., y: -78. },
            ];
            let conv = super::from_wgs84(&coords, &parsed).unwrap();
            let conv_back = super::to_wgs84(&conv, &parsed).unwrap();

            for i in 0..conv_back.len() {
                println!("{:?} -> {:?} -> {:?}", coords[i], conv[i], conv_back[i]);

                assert!(coords_approx_eq(&coords[i], &conv_back[i], 0.01));
            }
        });
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
    #[serial_test::serial]
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
    #[test]
    // Runs `projinfo`, so it must not overlap the tests that unset `PATH`
    // process-wide to simulate a machine without PROJ or GDAL
    // (`test_crs_noproj` here, `dem::tests::test_no_gdal_failure`). Those
    // are `serial`, but that only excludes each other -- an unmarked test
    // still runs alongside them and finds no `projinfo` on `PATH`. That is
    // the "random failure" the cfg above blames on macOS: it is a race, not
    // a platform.
    #[serial_test::parallel]
    #[cfg(not(any(target_os = "windows", target_os = "macos")))] // Added windows 2026-04-6, macos 2026-04-13 (random failure)
    fn test_projinfo_to_wkt() {
        let retval = super::projinfo_to_wkt("EPSG:32633").unwrap();

        println!("{}", retval);
        assert!(retval.starts_with("PROJCRS[\"WGS 84"));
    }
}
