use core::ops::{Add, Div, Mul, Sub};
use ndarray::{Array1, Array2, Axis, Slice};
use ndarray_stats::QuantileExt;
/// Miscellaneous functions that are used in other parts of the program
use std::str::FromStr;

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
pub fn interpolate_values<
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
>(
    x0: T,
    y0: &[T],
    x1: T,
    y1: &[T],
    x: T,
) -> Vec<T> {
    (0..y0.len())
        .map(|i| interpolate_between_known((x0, y0[i]), (x1, y1[i]), x))
        .collect::<Vec<T>>()
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
pub fn interpolate_between_known<
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Copy,
>(
    known_xy0: (T, T),
    known_xy1: (T, T),
    x: T,
) -> T {
    (known_xy0.1 * (known_xy1.0 - x) + known_xy1.1 * (x - known_xy0.0))
        / (known_xy1.0 - known_xy0.0)
}

/// Derive the quantiles of an iterator of values
///
/// # Arguments
/// - `values`: An iterator of values
/// - `quantiles`: The quantiles to derive
/// - `downsample`: Downsample the data to increase performance.
pub fn quantiles<'a, T: 'a + PartialOrd + Copy, I, const L: usize>(
    values: I,
    quantiles: &[f32; L],
    downsample: Option<usize>,
) -> [T; L]
where
    I: IntoIterator<Item = &'a T>,
{
    let mut vals: Vec<&T> = values
        .into_iter()
        .step_by(downsample.unwrap_or(1))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut output = [*vals[0]; L];

    for (i, quantile) in quantiles.iter().enumerate() {
        output[i] = *vals[(vals.len() as f32 * quantile) as usize];
    }

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
    chrono::DateTime::<chrono::Utc>::from_utc(
        chrono::NaiveDateTime::from_timestamp(seconds as i64, (seconds.fract() * 1e9) as u32),
        chrono::Utc,
    )
    .to_rfc3339()
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
                    let removed_consecutive_whitespace =
                        within_parentheses.replace("  ", " ").replace("  ", " ");

                    let arguments = removed_consecutive_whitespace
                        .split(" ")
                        .collect::<Vec<&str>>();

                    match arguments.get(argument_index) {
                        Some(s) => match s.trim().parse::<T>() {
                            Ok(v) => Ok(Some(v)),
                            Err(_) => Err(format!(
                                "Could not parse argument {} as value in string {}: {}",
                                argument_index, string, s
                            )),
                        },
                        None => Err(format!(
                            "Argument {} out of bounds in string: {}",
                            argument_index, string
                        )),
                    }
                }
                None => Err(format!(
                    "String: {} has opening parenthesis but not closing",
                    string
                )),
            }
        }
    }
}

pub enum Axis2D {
    Row,
    Col,
}

/// Average the data by binning them along one axis in a specified dimension
///
/// The python-equivalent would be:
///
/// DATA.groupby((X_VALUES / BIN_SIZE).astype(int)).mean(axis=AXIS)
///
/// # Arguments
/// - `data`: The data to modify inplace
/// - `axis`: The axis (row-wise or column-wise0 to average the values
/// - `x_values`: The values to bin, whose bins are later used to average the data in the specified
/// axis
/// - `bin_size`: The step size to bin the `x_values` in
///
/// # Returns
/// The upper breaks + 1 of the slices of the old data, e.g. [1, 3] for the bins [0, 1, 1]
pub fn groupby_average(
    data: &mut Array2<f32>,
    axis: Axis2D,
    x_values: &Array1<f32>,
    bin_size: f32,
) -> Array1<usize> {
    let bins = x_values.mapv(|value| (value / bin_size) as usize);

    let nd_axis = match axis {
        Axis2D::Row => Axis(0),
        Axis2D::Col => Axis(1),
    };

    let new_size = *bins.max().unwrap() + 1;

    let mut breaks = Array1::<usize>::from_elem((new_size,), bins.shape()[0] - 1);
    let mut last_highest = 0_usize;
    let mut i = 0_usize;
    for j in 0..bins.shape()[0] {
        if bins[j] > last_highest {
            breaks[i] = j;
            last_highest = bins[j];
            i += 1;
        };
    }
    let mut unique_breaks = breaks.into_raw_vec();
    unique_breaks.dedup();
    let breaks = Array1::from_vec(unique_breaks);

    for i in 0..breaks.shape()[0] {
        let lower = match i == 0 {
            true => 0,
            false => breaks[i - 1],
        };

        let upper = breaks[i];

        let old_data_slice = match (upper - lower) > 1 {
            true => data
                .slice_axis(nd_axis, Slice::new(lower as isize, Some(upper as isize), 1))
                .mean_axis(nd_axis)
                .unwrap(),
            false => match axis {
                Axis2D::Row => data.row(lower as usize).to_owned(),
                Axis2D::Col => data.column(lower as usize).to_owned(),
            },
        };
        let mut new_data_slice = match axis {
            Axis2D::Col => data.column_mut(i),
            Axis2D::Row => data.row_mut(i),
        };
        new_data_slice.assign(&old_data_slice);
    }
    data.slice_axis_inplace(
        nd_axis,
        Slice::new(0, Some(breaks.shape()[0] as isize + 1), 1),
    );

    breaks
}

/// Convert the two-way return time to depth
///
/// two_way_travel_distance = two_way_return_time * velocity
/// two_way_travel_distance² = 2² * (one_way_depth² + antenna_separation²)
/// two_way_travel_distance² = 2² * one_way_depth² + 2² * antenna_separation²
/// one_way_depth = √((two_way_return_time * velocity)² - 2² * antenna_separation²) / 2
///
/// # Arguments
/// - `return_time`: The two-way return time in nanoseconds
/// - `velocity`: The wave velocity in m/ns
/// - `antenna_separation`: The separation between the transmitter and the receiver
///
/// # Returns
/// The depth in m corresponding to the return time, or 0. if the return time is smaller than
/// theoretically possible given the antenna separation.
///
///
pub fn return_time_to_depth(return_time: f32, velocity: f32, antenna_separation: f32) -> f32 {
    let two_way_distance = return_time * velocity;
    match two_way_distance > (2. * antenna_separation) {
        true => ((two_way_distance).powi(2) - 4. * antenna_separation.powi(2)).sqrt() / 2.,
        false => 0.,
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    #[test]
    fn test_interpolate_between_known() {
        let known_xy0 = (0_f64, 0_f64);
        let known_xy1 = (5_f64, 10_f64);

        assert_eq!(
            super::interpolate_between_known(known_xy0, known_xy1, 2.5),
            5.0
        )
    }

    #[test]
    fn test_interpolate_values() {
        let coord0 = vec![0_f64, 0_f64, 0_f64];
        let time0 = 0_f64;

        let coord1 = vec![5_f64, 10_f64, 15_f64];
        let time1 = 1_f64;

        assert_eq!(
            super::interpolate_values(time0, &coord0, time1, &coord1, 0.5),
            vec![2.5, 5.0, 7.5]
        )
    }

    #[test]
    fn test_parse_option() {
        assert_eq!(super::parse_option::<u32>("dewow", 0), Ok(None));
        assert_eq!(super::parse_option::<u32>("dewow(1)", 0), Ok(Some(1_u32)));
        assert_eq!(
            super::parse_option::<f32>("dewow(1 2.0)", 1),
            Ok(Some(2_f32))
        );
        assert_eq!(
            super::parse_option::<i64>("dewow(1  -2)", 1),
            Ok(Some(-2_i64))
        );
        assert_eq!(
            super::parse_option::<i64>("kirchoff_migration2d    (1    -2)    ", 1),
            Ok(Some(-2_i64))
        );

        assert!(super::parse_option::<f32>("dewow(", 0)
            .unwrap_err()
            .contains("opening parenthesis but not closing"));
        assert!(super::parse_option::<f32>("dewow(1)", 1)
            .unwrap_err()
            .contains("Argument 1 out of bounds"));

        assert!(super::parse_option::<f32>("dewow(1,1.0)", 1)
            .unwrap_err()
            .contains("Argument 1 out of bounds"));

        assert!(super::parse_option::<f32>("dewow(1 1,1)", 1)
            .unwrap_err()
            .contains("Could not parse argument 1"));
    }

    #[test]
    fn test_quantiles() {
        let values = vec![4, 1, 2, 3, 0];

        assert_eq!(super::quantiles(&values, &[0.1, 0.5, 0.9], None), [0, 2, 4]);
        assert_eq!(
            super::quantiles(&values, &[0.1, 0.5, 0.9], Some(2)),
            [0, 2, 4]
        );
    }

    #[test]
    fn test_seconds_to_rfc3339() {
        let seconds = 1_600_000_000_f64;

        assert_eq!(
            super::seconds_to_rfc3339(seconds),
            "2020-09-13T12:26:40+00:00"
        );
    }
    #[test]
    fn test_groupby_average() {
        let test_data = Array1::<f32>::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();

        let xs = Array1::<f32>::from_vec(vec![0., 1., 1., 2., 3.]);

        let mut test_data0 = test_data.clone();
        super::groupby_average(&mut test_data0, super::Axis2D::Row, &xs, 1.);

        assert_eq!(test_data0.shape(), &[4_usize, 5_usize]);
        let expected = (test_data.get((1, 0)).unwrap() + test_data.get((2, 0)).unwrap()) / 2.;
        assert_eq!(test_data0.get((1, 0)), Some(&expected));

        let mut test_data1 = test_data.clone();
        super::groupby_average(&mut test_data1, super::Axis2D::Col, &(xs * 2.), 2.);
        assert_eq!(test_data1.shape(), &[5_usize, 4_usize]);
        let expected = (test_data.get((0, 1)).unwrap() + test_data.get((0, 2)).unwrap()) / 2.;
        assert_eq!(test_data1.get((0, 1)), Some(&expected));
    }
    #[test]
    fn test_return_time_to_depth() {
        // The depth without antenna distance should be the time * velocity / 2
        let depth = super::return_time_to_depth(200., 0.1, 0.);
        assert_eq!(depth, 10.);

        let depth_2m_antenna = super::return_time_to_depth(200., 0.1, 2.);
        // The depth with antenna distance should be smaller than the one without
        assert!(depth_2m_antenna < depth);
        // It should equal to the equation in the function docstring
        assert_eq!(
            depth_2m_antenna,
            ((200_f32 * 0.1).powi(2) - 2.0_f32.powi(2) * 2.0_f32.powi(2)).sqrt() / 2.
        );

        // If the return time is tiny and the antenna separation is too large, 0. should be
        // returned.
        assert_eq!(super::return_time_to_depth(2., 0.1, 2.), 0.);


        // A weird NAN error came up with these settings which should not occur
        let depth = super::return_time_to_depth(5.9624557, 0.168, 1.0);
        assert!(depth.is_finite(), "{}", depth);
    }
}
