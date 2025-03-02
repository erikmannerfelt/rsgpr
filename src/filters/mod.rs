use biquad::Biquad;
use ndarray::{Array, Array2};
use num::Float;

use crate::tools;

pub fn abslog<T: Float>(data: &mut Array2<T>) {
    data.mapv_inplace(|v| v.abs());

    let mut minval = T::one();

    let subsampling = ((data.shape()[0] * data.shape()[1]) as f32 * 0.1).max(100.) as usize;
    for quantile in [0.01, 0.05, 0.5, 0.9] {
        let new_min = tools::quantiles(
            data.iter().filter(|v| v >= &&T::zero()),
            &[quantile],
            Some(subsampling),
        )[0];

        if !new_min.is_zero() {
            minval = new_min;
            break;
        }
    }

    data.mapv_inplace(|v| (v + minval).log10());
}

pub fn siglog<T: Float, D: ndarray::Dimension>(data: &mut Array<T, D>, minval_log10: T) {
    data.mapv_inplace(|v| (v.abs().log10() - minval_log10).max(T::zero()) * v.signum());
}

pub trait ButterworthBandpass<T: Float> {
    fn butter_coef_norm(
        low_cutoff: T,
        high_cutoff: T,
    ) -> Result<biquad::Coefficients<T>, biquad::Errors>;
}

impl ButterworthBandpass<f32> for f32 {
    fn butter_coef_norm(
        low_cutoff: f32,
        high_cutoff: f32,
    ) -> Result<biquad::Coefficients<f32>, biquad::Errors> {
        let center_frequency = (low_cutoff + high_cutoff) / 2.;
        let bandwidth = high_cutoff - low_cutoff;
        let q_factor = center_frequency / bandwidth;

        biquad::Coefficients::from_normalized_params(
            biquad::Type::BandPass,
            center_frequency,
            q_factor,
        )
    }
}

pub fn normalized_bandpass<T: Float + ButterworthBandpass<T>>(
    data: &mut Array2<T>,
    low_cutoff: T,
    high_cutoff: T,
) -> Result<(), biquad::Errors> {
    for mut col in data.columns_mut() {
        let coefs = T::butter_coef_norm(low_cutoff, high_cutoff)?;

        let mut filt = biquad::DirectForm2Transposed::new(coefs);

        col.mapv_inplace(|v| filt.run(v));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use biquad::Biquad;
    use ndarray::{Array2, AssignElem};
    use ndarray_stats::DeviationExt;

    #[test]
    fn test_abslog() {
        let mut data = Array2::<f32>::from_shape_vec(
            (10, 10),
            (1..101).into_iter().map(|v| v as f32).collect::<Vec<f32>>(),
        )
        .unwrap();

        super::abslog(&mut data);

        let new_minval = *data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let new_maxval = *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        println!("Max: {new_maxval}, min: {new_minval}");
        assert!(new_minval > 0.2);
        assert!(new_minval < 1.);

        assert!(new_maxval < 2.1);
        assert!(new_maxval > 1.9);
    }

    #[test]
    fn test_bandpass() {
        use super::ButterworthBandpass;
        // Generate filter coefficients
        let coefs = f32::butter_coef_norm(0.05, 0.9).unwrap();

        // let data: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1000.];
        let data: Vec<f32> = (0..10000)
            .map(|v| {
                if v % 3000 == 0 {
                    1000.
                } else if v % 3 == 0 {
                    0.
                } else if v % 2 == 0 {
                    -1.
                } else {
                    1.
                }
            })
            .collect();
        // Create filter instance
        let mut filt = biquad::DirectForm2Transposed::new(coefs);

        let filtered_data = data.iter().map(|v| filt.run(*v)).collect::<Vec<f32>>();
        // let filtered_data = super::apply_bandpass_1d(&data, &mut filt);

        assert_eq!(filtered_data.len(), data.len());

        assert!(filtered_data.iter().any(|&x| x != 0.0));

        assert!(
            filtered_data
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
                < &300.
        );
    }

    #[test]
    fn test_siglog() {
        let arr = ndarray::arr1(&[1000_f32, -1000_f32, 0_f32]);

        let mut arr0 = arr.clone();
        super::siglog(&mut arr0, 0.);
        assert_eq!(arr0, ndarray::arr1(&[3., -3., 0.]));

        let mut arr1 = arr.clone();
        arr1[2].assign_elem(0.0001);
        super::siglog(&mut arr1, 0.);
        assert_eq!(arr1, ndarray::arr1(&[3., -3., 0.]));

        let mut arr2 = arr.clone();
        arr2[2].assign_elem(0.1);
        super::siglog(&mut arr2, -2.);
        assert_eq!(arr2, ndarray::arr1(&[5., -5., 1.]));
    }
}
