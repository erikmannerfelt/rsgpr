use ndarray::{Axis, Array1};
use ndarray_stats::QuantileExt;
use std::str::FromStr;
use core::ops::{Add, Mul, Sub, Div};

/// Interpolate an arbitrary amount of independent values between two known points
///
/// # Arguments
/// - 
///
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
/// - `known_xy0`: The first known coordinate as (explanatory, independent)
/// - `known_xy1`: The second known coordinate as (explanatory, independent)
/// - `x`: The explanatory point at which to interpolate the independent point
///
/// # Returns
/// The interpolated independent (y) coordinate.
pub fn interpolate_between_known<T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy>(known_xy0: (T, T), known_xy1: (T, T), x: T) -> T {

    (known_xy0.1 * (known_xy1.0 - x) + known_xy1.1 * (x - known_xy0.0)) / (known_xy1.0 - known_xy0.0)

}

pub fn quantiles<'a, I>(values: I, quantiles: &[f32]) -> Vec<f32> where
    I: IntoIterator<Item = &'a f32>, {


    let mut values_arr = Array1::from_iter(values.into_iter().map(|f| f.to_owned()));
    let mut output: Vec<f32> = Vec::new();

    for quantile in quantiles {

        output.push(
            values_arr.quantile_axis_skipnan_mut(Axis(0), noisy_float::NoisyFloat::new(quantile.to_owned() as f64), &ndarray_stats::interpolate::Midpoint).unwrap().first().unwrap().to_owned().to_owned()
        );
    };

    output
}

pub fn seconds_to_rfc3339(seconds: f64) -> String {
    chrono::DateTime::<chrono::Utc>::from_utc(chrono::NaiveDateTime::from_timestamp(seconds as i64, (seconds.fract() * 1e9) as u32), chrono::Utc).to_rfc3339()
}


pub fn parse_option<T: FromStr>(string: &str, argument_index: usize) -> Result<Option<T>, String> {
    match string.split_once("(") {
        None => Ok(None),
        Some((_, first_part)) => {
            match first_part.split_once(")") {
                Some((within_parentheses, _)) => {

                    // Replace has to be run twice, as it may be an odd numbe of whitespaces:
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
}
