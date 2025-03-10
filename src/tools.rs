use core::ops::{Add, Div, Mul, Sub};
use enterpolation::Generator;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_stats::QuantileExt;
use num::Float;
use rayon::prelude::*;
/// Miscellaneous functions that are used in other parts of the program
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

/// Parse a provided step list (or filepath to a step list)
///
/// # Arguments
/// - `steps`: An unformatted list of steps, or a filepath
///
/// # Returns
/// Formatted steps either from the string itself or from the parsed file.
pub fn parse_step_list(steps: &str) -> Result<Vec<String>, String> {
    let filepath = Path::new(steps);
    if filepath.is_file() {
        match crate::tools::read_text(&filepath.to_path_buf()) {
            Ok(s) => Ok(s),
            Err(e) => Err(format!("Tried to read step file but failed: {e:?}")),
        }
    } else {
        Ok(steps.split(',').map(|s| s.trim().to_string()).collect())
    }
}

/// Read a text file and return all lines as a vec
///
/// # Arguments
/// - `filepath`: The filepath of the text file
///
/// # Returns
/// Each line as a trimmed String
pub fn read_text(filepath: &PathBuf) -> Result<Vec<String>, std::io::Error> {
    let content = std::fs::read_to_string(filepath)?;

    let mut lines = Vec::<String>::new();

    for line in content.lines() {
        lines.push(line.trim().to_owned());
    }

    Ok(lines)
}

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
/// <https://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_between_two_known_points>
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

fn interpolate_vec<T: Float + Copy + Sub<Output = T> + std::fmt::Debug>(
    x_old: &[T],
    y_old: &[T],
    x_new: &[T],
) -> Vec<T> {
    if x_old.len() != y_old.len() {
        panic!("Interpolation failed. x_old and y_old must have the same length");
    }

    let model = enterpolation::linear::Linear::builder()
        .elements(y_old)
        .knots(x_old)
        .build()
        .unwrap();
    model.sample(x_new.iter().map(|v| v.to_owned())).collect()
}

fn interpolate_ndarray<T: Float + std::fmt::Debug>(
    x_old: &ArrayView1<T>,
    y_old: &ArrayView1<T>,
    x_new: &ArrayView1<T>,
) -> Array1<T> {
    Array1::<T>::from_vec(interpolate_vec(
        &x_old.to_vec(),
        &y_old.to_vec(),
        &x_new.to_vec(),
    ))
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
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut output = [*vals[0]; L];

    for (i, quantile) in quantiles.iter().enumerate() {
        output[i] = *vals[((vals.len() as f32 * quantile) as usize).min(vals.len() - 1)];
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
    chrono::DateTime::from_timestamp(seconds as i64, (seconds.fract() * 1e9) as u32)
        .unwrap()
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
/// assert_eq!(parse_option::<f32>("some_fancy_step(1 2.0)", 1), Ok(Some(2.0)));
/// assert_eq!(parse_option::<f32>("some_fancy_step", 1), Ok(None));
///
/// ```
///
/// # Returns
/// - Ok(Some(x)) where x is the successfully parsed argument
/// - Ok(None) if there is no argument in the string
/// - Err(e) if the argument could not be parsed
pub fn parse_option<T: FromStr>(string: &str, argument_index: usize) -> Result<Option<T>, String> {
    match string.split_once('(') {
        None => Ok(None),
        Some((_, first_part)) => {
            match first_part.split_once(')') {
                Some((within_parentheses, _)) => {
                    // Replace has to be run twice, as it may be an odd number of whitespaces:
                    // "_-_-_" => "_-_" => "_"
                    let removed_consecutive_whitespace =
                        within_parentheses.replace("  ", " ").replace("  ", " ");

                    let arguments = removed_consecutive_whitespace
                        .split(' ')
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

fn digitize<F: Float>(values: &[F], bins: &[F]) -> Vec<usize> {
    let mut bins = bins.iter().collect::<Vec<&F>>();
    bins.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut indices = Vec::<usize>::new();

    for value in values {
        // Initialize the upper bin index by the "out of bounds" value
        let mut upper_bin = 0;

        if value >= bins[bins.len() - 1] {
            upper_bin = bins.len();
        } else if value > bins[0] {
            for bin in &bins {
                if bin > &value {
                    break;
                }
                upper_bin += 1;
            }
        }
        indices.push(upper_bin);
    }

    indices
}

pub struct Resampler<F: Float> {
    pub x_values: Array1<F>,
    pub target_x_values: Array1<F>,
    pub digitized: Vec<usize>,
    slope_indices: Vec<Vec<usize>>,
    intercept_indices: Vec<Vec<usize>>,
    _debug: bool,
}

fn equally_spaced_from_sparse<F: Float>(sparse: &Array1<F>, resolution: F) -> Array1<F> {
    Array1::<F>::range(
        *sparse.min().unwrap(),
        *sparse.max().unwrap() + resolution,
        resolution,
    )
}

impl<F: Float + std::fmt::Display + std::iter::Sum + Send + Sync + std::fmt::Debug> Resampler<F> {
    fn _new(x_values: Array1<F>, target_x_values: Array1<F>, debug: bool) -> Resampler<F> {
        //let target_x_values = Array1::<F>::range(*x_values.min().unwrap(), x_values.max().unwrap().clone() + resolution, resolution);

        let digitized = digitize(
            x_values.as_slice().unwrap(),
            target_x_values.as_slice().unwrap(),
        );

        let mut slope_indices = Vec::<Vec<usize>>::new();
        let mut intercept_indices = slope_indices.clone();
        for i in 0..target_x_values.len() {
            let mut indices_between = Vec::<usize>::new();
            let mut potential_outside_behind = Vec::<(usize, usize)>::new();
            let mut potential_outside_ahead = Vec::<(usize, usize)>::new();
            let mut indices_outside = Vec::<usize>::new();
            for (j, k) in digitized.iter().enumerate() {
                match k.cmp(&i) {
                    std::cmp::Ordering::Less => {
                        potential_outside_behind.push((i - k, j));
                    }
                    std::cmp::Ordering::Greater => {
                        potential_outside_ahead.push((*k - i, j));
                    }
                    std::cmp::Ordering::Equal => {
                        indices_between.push(j);
                    }
                };
            }

            if indices_between.len() < 2 {
                potential_outside_behind.sort_by(|a, b| a.0.cmp(&b.0));
                potential_outside_ahead.sort_by(|a, b| a.0.cmp(&b.0));

                if let Some(point_behind) = potential_outside_behind.first() {
                    for index in potential_outside_behind
                        .iter()
                        .filter_map(|(distance, index)| {
                            (distance == &point_behind.0).then_some(index)
                        })
                    {
                        indices_outside.push(*index);
                    }
                }
                if let Some(point_ahead) = potential_outside_ahead.first() {
                    for index in potential_outside_ahead
                        .iter()
                        .filter_map(|(distance, index)| {
                            (distance == &point_ahead.0).then_some(index)
                        })
                    {
                        indices_outside.push(*index);
                    }
                }
            };
            let mut all_indices = indices_outside.clone();
            all_indices.append(indices_between.clone().as_mut());
            let indices_for_intercept = match indices_between.is_empty() {
                true => all_indices.clone(),
                false => indices_between.clone(),
            };
            let indices_for_slope = match indices_between.len() < 2 {
                true => all_indices.clone(),
                false => indices_between,
            };

            intercept_indices.push(indices_for_intercept);
            slope_indices.push(indices_for_slope);
        }

        Resampler {
            x_values,
            target_x_values,
            digitized,
            slope_indices,
            intercept_indices,
            _debug: debug,
        }
    }

    pub fn new(x_values: Array1<F>, resolution: F) -> Self {
        let target_x_values = equally_spaced_from_sparse::<F>(&x_values, resolution);
        Resampler::_new(x_values, target_x_values, false)
    }

    /*
    pub fn new_with_target(x_values: Array1<F>, target_x_values: Array1<F>) -> Self {
        Resampler::_new(x_values, target_x_values, false)
    }
    */

    fn _new_debug(x_values: Array1<F>, resolution: F) -> Self {
        let target_x_values = equally_spaced_from_sparse::<F>(&x_values, resolution);
        Self::_new(x_values, target_x_values, true)
    }
    fn _resample<F2: Float + std::fmt::Display + std::iter::Sum + Send + std::fmt::Debug>(
        &self,
        x_values: &Array1<F2>,
        target_x_values: &Array1<F2>,
        y_values: &ArrayView1<F2>,
    ) -> Array1<F2> {
        interpolate_ndarray::<F2>(&x_values.view(), y_values, &target_x_values.view())
    }
    pub fn resample_convert<
        F2: Float + std::fmt::Display + std::iter::Sum + Send + std::fmt::Debug,
    >(
        &self,
        y_values: &ArrayView1<F2>,
    ) -> Array1<F2> {
        let target_x_values: Array1<F2> = self.target_x_values.mapv(|v| F2::from(v).unwrap());
        let x_values = self.x_values.mapv(|v| F2::from(v).unwrap());

        self._resample(&x_values, &target_x_values, y_values)
    }
    pub fn resample(&self, y_values: &ArrayView1<F>) -> Array1<F> {
        self._resample(&self.x_values, &self.target_x_values, y_values)
    }

    /*
    pub fn resample_along_axis(&self, data: &Array2<F>, axis: Axis2D) -> Array2<F> {
        let nd_axis = match axis {
            Axis2D::Row => ndarray::Axis(0),
            Axis2D::Col => ndarray::Axis(1),
        };
        let slice = ndarray::Slice::new(0, Some(self.target_x_values.len() as isize), 1);

        let length = match axis {
            Axis2D::Row => data.shape()[0],
            Axis2D::Col => data.shape()[1],
        };

        let mut buffer = Array1::<F>::zeros(length);

        let mut data2 = data.clone();

        let axis_values = match axis {
            Axis2D::Row => data2.columns_mut(),
            Axis2D::Col => data2.rows_mut(),
        };

        for mut arr in axis_values {
            let resampled = self.resample(&arr.view());

            let mut slice = buffer.slice_axis_mut(ndarray::Axis(0), slice);
            slice.assign(&resampled);
            arr.assign(&buffer);
        }
        data2.slice_axis_inplace(nd_axis, slice);
        data2
    }
    */

    pub fn resample_along_axis_par(&self, data: &Array2<F>, axis: Axis2D) -> Array2<F> {
        let length = match axis {
            Axis2D::Row => data.shape()[1],
            Axis2D::Col => data.shape()[0],
        };

        let out_shape = match axis {
            Axis2D::Row => (self.target_x_values.len(), data.shape()[1]),
            Axis2D::Col => (data.shape()[0], self.target_x_values.len()),
        };

        let output: Vec<Array1<F>> = (0..length)
            .into_par_iter()
            .map(|i| {
                let y_vals = match axis {
                    Axis2D::Row => data.column(i),
                    Axis2D::Col => data.row(i),
                };
                self.resample(&y_vals)
            })
            .collect();

        let mut out = Array2::<F>::zeros(out_shape);

        let iterator = match axis {
            Axis2D::Row => out.columns_mut(),
            Axis2D::Col => out.rows_mut(),
        };
        for (i, mut slice) in iterator.into_iter().enumerate() {
            slice.assign(&output[i]);
        }

        out
    }
}

impl std::convert::From<Resampler<f32>> for Resampler<f64> {
    fn from(resampler: Resampler<f32>) -> Resampler<f64> {
        let x_values = resampler.x_values.mapv(f64::from);
        let target_x_values = resampler.target_x_values.mapv(f64::from);
        Resampler {
            x_values,
            target_x_values,
            digitized: resampler.digitized.clone(),
            slope_indices: resampler.slope_indices.clone(),
            intercept_indices: resampler.intercept_indices.clone(),
            _debug: resampler._debug,
        }
    }
}

//}

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
    fn test_interpolate() {
        let mut tests = Vec::<[Vec<f64>; 4]>::new();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let x_new = vec![1.5, 2.5, 0., 4.0]; // Includes values for extrapolation
        tests.push([x, y, x_new.clone(), x_new.clone()]);

        let x = vec![1.0, 3.0, 5.0];
        let y = vec![1.0, 3.0, 5.0];
        let x_new = vec![2.0, 4.0, 0.0, 6.0]; // Includes values for extrapolation

        tests.push([x, y, x_new.clone(), x_new.clone()]);

        let x = vec![0., 2., 5.];
        let y = vec![0., 4., 1.];
        let x_new = vec![-1., 0., 1., 2., 3., 4., 5., 6.];
        let y_test = vec![-2., 0., 2., 4., 3., 2., 1., 0.];

        tests.push([x, y, x_new, y_test]);

        for test_case in tests {
            let y_new = super::interpolate_vec(&test_case[0], &test_case[1], &test_case[2])
                .iter()
                .map(|v| (v * 10_f64).round() / 10.)
                .collect::<Vec<f64>>();
            assert_eq!(y_new, test_case[3]);

            let y_new_arr = super::interpolate_ndarray(
                &Array1::from_vec(test_case[0].clone()).view(),
                &Array1::from_vec(test_case[1].clone()).view(),
                &Array1::from_vec(test_case[2].clone()).view(),
            )
            .mapv(|v| (v * 10_f64).round() / 10.);

            assert_eq!(y_new_arr, Array1::from_vec(test_case[3].clone()));
        }
    }

    #[test]
    #[should_panic(expected = "x_old and y_old must have the same length")]
    fn test_interpolate_panic() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let x_new = vec![1.5, 2.5];
        let _ = super::interpolate_vec(&x, &y, &x_new);
    }
    /*
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
    */
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

    #[test]
    fn test_digitize() {
        let bins = [0., 1., 2., 3.];
        let values = [-0.5, 1., 0.5, 0.8, 0.9, 2.3, 3.4];

        let expected = [0, 2, 1, 1, 1, 3, 4];

        assert_eq!(super::digitize(&values, &bins), expected);
    }

    #[test]
    fn test_resample() {
        let x_values = Array1::from_vec(vec![0., 0.01, 0.5, 0.99, 1.99, 2.05, 2.05, 5.05]);
        let y_values = Array1::from_vec(vec![1., 1., 2., 3., 4., 4., 4., 5.]);

        for i in 0..x_values.len() {
            let xval = &x_values[i];
            let yval = &y_values[i];
            println!("{i}: x={xval}, y={yval}");
        }

        let resolution = 1.;

        let mut resampler = super::Resampler::<f32>::_new_debug(x_values, resolution);
        println!("{:?}", resampler.target_x_values);

        let new_ys = resampler.resample(&y_values.view());
        let new_ys_f64 = resampler.resample_convert::<f64>(&y_values.mapv(|v| f64::from(v)).view());

        assert_eq!(new_ys_f64[0] as f32, new_ys[0]);

        assert_eq!(new_ys[0], 1.);
        assert!((new_ys[1] - 3.).abs() < 1e-1);
        assert!((new_ys[2] > 3.5) & (new_ys[2] < 5.));

        assert!(
            new_ys
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                < &5.5
        );

        resampler._debug = false;
        let y_matrix_manyrows =
            ndarray::Array2::from_shape_fn((50, y_values.len()), |(_, j)| y_values[j]);
        let resampled_colwise =
            resampler.resample_along_axis_par(&y_matrix_manyrows, super::Axis2D::Col);
        assert_eq!(
            resampled_colwise.shape(),
            [50, resampler.target_x_values.len()]
        );

        assert!(resampled_colwise.iter().all(|v| !v.is_nan()));

        let y_matrix_manycols =
            ndarray::Array2::from_shape_fn((y_values.len(), 50), |(i, _)| y_values[i]);
        let resampled_rowwise =
            resampler.resample_along_axis_par(&y_matrix_manycols, super::Axis2D::Row);
        assert_eq!(
            resampled_rowwise.shape(),
            [resampler.target_x_values.len(), 50]
        );
        assert!(resampled_rowwise.iter().all(|v| !v.is_nan()));
    }
}
