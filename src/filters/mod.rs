use crate::tools;
use ndarray::{Array, Array2};
use num::Float;

pub mod bandpass;

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

/// Normalized band‑pass wrapper that applies the **new** RBJ/W3C constant‑peak
/// biquad per trace (column).
///
/// `low_cutoff` and `high_cutoff` are normalized to Nyquist in (0,1).
pub fn normalized_bandpass<T: Float>(
    data: &mut Array2<T>,
    low_cutoff: T,
    high_cutoff: T,
) -> Result<(), String> {
    for mut col in data.columns_mut() {
        // Work on an owned 1D array, apply the new filter, then write back.
        let mut tmp = col.to_owned();
        bandpass::bandpass_constant_peak(&mut tmp, low_cutoff, high_cutoff, None, None)
            .map_err(|e| e.to_string())?;
        col.assign(&tmp);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, AssignElem};

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
        assert!(new_minval > 0.2);
        assert!(new_minval < 1.);
        assert!(new_maxval < 2.1);
        assert!(new_maxval > 1.9);
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
