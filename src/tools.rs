/// Miscellaneous functions that are used in other parts of the program
use std::str::FromStr;
use core::ops::{Add, Mul, Sub, Div};

/// Interpolate an arbitrary amount of independent values between two known points
///
/// # Arguments
/// - `x0`: The first known explanatory variable
/// - `y0`: The first known independent variables
/// - `x1`: The second known explanatory variable
/// - `y1`: The second known independent variables
/// - `x`: The explanatory point at which to interpolate the independent variables
///
/// # Returns
/// The interpolated independent (y) values.
///
/// # Examples
/// ```
/// assert_eq!(interpolate_values(0_f32, &[0., 5.], 1., &[-1., 10.], 0.5), &[-0.5, 7.5]);
///
/// ```
///
/// # Panics
/// - The first slice of independent values is longer than the second: `y0.len()` > `y1.len()`
pub fn interpolate_values<T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy>(x0: T, y0: &[T], x1: T, y1: &[T], x: T) -> Vec<T> {
    (0..y0.len()).map(|i| interpolate_between_known((x0, y0[i]), (x1, y1[i]), x)).collect::<Vec<T>>()
}

/// Interpolate linearly between two known points
///
/// https://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_between_two_known_points
///
/// # Arguments
/// - `known_xy0`: The first known point as (explanatory, independent)
/// - `known_xy1`: The second known point as (explanatory, independent)
/// - `x`: The explanatory point at which to interpolate the independent variables
///
/// # Returns
/// The interpolated independent (y) value.
pub fn interpolate_between_known<T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy>(known_xy0: (T, T), known_xy1: (T, T), x: T) -> T {

    (known_xy0.1 * (known_xy1.0 - x) + known_xy1.1 * (x - known_xy0.0)) / (known_xy1.0 - known_xy0.0)

}

/// Derive the quantiles of an iterator of values
///
/// # Arguments
/// - `values`: An iterator of values
/// - `quantiles`: The quantiles to derive
/// - `downsample`: Downsample the data to increase performance.
pub fn quantiles<'a, T: 'a + PartialOrd + Copy, I, const L: usize>(values: I, quantiles: &[f32; L], downsample: Option<usize>) -> [T; L] where I: IntoIterator<Item = &'a T> {

    let mut vals: Vec<&T> = values.into_iter().step_by(downsample.unwrap_or(1)).collect();
    vals.sort_by(| a, b | a.partial_cmp(b).unwrap());

    let mut output = [*vals[0]; L];

    for (i, quantile) in quantiles.iter().enumerate() {
        output[i] = *vals[(vals.len() as f32 * quantile) as usize];
    };

    output
}


/// Convert numbers of seconds since UNIX epoch into an RFC3339 datetime string in UTC
///
/// # Arguments
/// - `seconds`: The number of seconds since UNIX epoch
///
/// # Examples
///
///
/// # Returns
/// A string representation of the datetime
pub fn seconds_to_rfc3339(seconds: f64) -> String {
    chrono::DateTime::<chrono::Utc>::from_utc(chrono::NaiveDateTime::from_timestamp(seconds as i64, (seconds.fract() * 1e9) as u32), chrono::Utc).to_rfc3339()
}


/// Parse the options (arguments) of a user-supplied step
///
/// # Arguments
/// - `string`: The string to parse
/// - `argument_index`: The expected index of the argument
///
/// # Examples
/// ```
/// assert_eq!(parse_option::<u32>("dewow(5)", 0), Ok(Some(5)));
/// assert_eq!(parse_option::<f32>("some_fancy_step(1 2.0)", 1), Ok(Some(2.0));
/// assert_eq!(parse_option::<f32>("some_fancy_step", 1), Ok(None));
///
/// ```
///
/// # Returns
/// - Ok(Some(x)) where x is the successfully parsed argument
/// - Ok(None) if there is no argument in the string
/// - Err(e) if the argument could not be parsed
pub fn parse_option<T: FromStr>(string: &str, argument_index: usize) -> Result<Option<T>, String> {
    match string.split_once("(") {
        None => Ok(None),
        Some((_, first_part)) => {
            match first_part.split_once(")") {
                Some((within_parentheses, _)) => {

                    // Replace has to be run twice, as it may be an odd number of whitespaces:
                    // "_-_-_" => "_-_" => "_"
                    let removed_consecutive_whitespace = within_parentheses.replace("  ", " ").replace("  ", " ");
                
                    let arguments = removed_consecutive_whitespace.split(" ").collect::<Vec<&str>>();

                    match arguments.get(argument_index) {
                        Some(s) => match s.trim().parse::<T>() {
                            Ok(v) => Ok(Some(v)),
                            Err(_) => Err(format!("Could not parse argument {} as value in string {}: {}", argument_index, string, s))
                        },
                        None => Err(format!("Argument {} out of bounds in string: {}", argument_index, string))
                    }
                },
                None => Err(format!("String: {} has opening parenthesis but not closing", string))
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_interpolate_between_known() {

        let known_xy0 = (0_f64, 0_f64);
        let known_xy1 = (5_f64, 10_f64);

        assert_eq!(super::interpolate_between_known(known_xy0, known_xy1, 2.5), 5.0)

    }

    #[test]
    fn test_interpolate_values() {

        let coord0 = vec![0_f64, 0_f64, 0_f64];
        let time0 = 0_f64;

        let coord1 = vec![5_f64, 10_f64, 15_f64];
        let time1 = 1_f64;

        assert_eq!(super::interpolate_values(time0, &coord0, time1, &coord1, 0.5), vec![2.5, 5.0, 7.5])
    }

    #[test]
    fn test_parse_option() {

        assert_eq!(super::parse_option::<u32>("dewow", 0), Ok(None));
        assert_eq!(super::parse_option::<u32>("dewow(1)", 0), Ok(Some(1_u32)));
        assert_eq!(super::parse_option::<f32>("dewow(1 2.0)", 1), Ok(Some(2_f32)));
        assert_eq!(super::parse_option::<i64>("dewow(1  -2)", 1), Ok(Some(-2_i64)));
        assert_eq!(super::parse_option::<i64>("kirchoff_migration2d    (1    -2)    ", 1), Ok(Some(-2_i64)));


        assert!(super::parse_option::<f32>("dewow(", 0).unwrap_err().contains("opening parenthesis but not closing"));
        assert!(super::parse_option::<f32>("dewow(1)", 1).unwrap_err().contains("Argument 1 out of bounds"));

        assert!(super::parse_option::<f32>("dewow(1,1.0)", 1).unwrap_err().contains("Argument 1 out of bounds"));

        assert!(super::parse_option::<f32>("dewow(1 1,1)", 1).unwrap_err().contains("Could not parse argument 1"));
    }

    #[test]
    fn test_quantiles() {

        let values = vec![4, 1, 2, 3, 0];

        assert_eq!(super::quantiles(&values, &[0.1, 0.5, 0.9], None), [0, 2, 4]);
        assert_eq!(super::quantiles(&values, &[0.1, 0.5, 0.9], Some(2)), [0, 2, 4]);
    }

    #[test]
    fn test_seconds_to_rfc3339 () {

        let seconds = 1_600_000_000_f64;

        assert_eq!(super::seconds_to_rfc3339(seconds), "2020-09-13T12:26:40+00:00");

    }
}
