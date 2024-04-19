use ndarray::Array2;
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

#[cfg(test)]
mod tests {
    use ndarray::Array2;

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

        assert!(new_minval > 0.4);
        assert!(new_minval < 1.);
        let new_maxval = *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(new_maxval < 2.1);
        assert!(new_maxval > 1.9);
    }
}
