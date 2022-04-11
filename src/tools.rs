
use ndarray::{Axis, Array1};
use ndarray_stats::QuantileExt;

pub fn interpolate_values(x0: f64, y0: &[f64], x1: f64, y1: &[f64], x: f64) -> Vec<f64> {

    let mut output: Vec<f64> = Vec::new();

    for i in 0..y0.len() {
        output.push(interpolate_between_known((x0, y0[i]), (x1, y1[i]), x))
    };

    output
}

/// Interpolate linearly between two known points
/// https://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_between_two_known_points
pub fn interpolate_between_known(known_xy0: (f64, f64), known_xy1: (f64, f64), x: f64) -> f64 {

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
}
