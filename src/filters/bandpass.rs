use ndarray::Array1;
use num::Float;

/// Design and run a constant‑peak (0 dB at center) band‑pass biquad using RBJ/W3C formulas,
/// applying it **in place** (Direct Form II Transposed).
///
/// Modes:
/// - **Absolute‑units mode**: pass `Some(center_frequency)` and `Some(sample_rate)`.
///   All frequencies (low/high/center/sample_rate) must use the **same unit**.
/// - **Normalized mode**: pass `center_frequency=None`. Then `low_cutoff` and
///   `high_cutoff` are interpreted as normalized to Nyquist in `(0, 1)`, i.e., `1` = Nyquist.
///   `sample_rate` is ignored in this mode.
///
/// # Arguments
/// * `data`               – Signal to be filtered (modified in place).
/// * `low_cutoff`         – Low edge of the passband (normalized or absolute).
/// * `high_cutoff`        – High edge of the passband (normalized or absolute).
/// * `center_frequency`   – `Some(f0)` (absolute‑units mode) or `None` (normalized mode).
/// * `sample_rate`        – `Some(fs)` when `center_frequency` is `Some(_ )`, else ignored.
///
/// # Returns
/// Ok(()) on success; Err(&str) on invalid parameters.
///
/// # Implementation notes
/// - Coefficients follow the **RBJ/W3C Audio EQ Cookbook** band‑pass (constant 0 dB peak)
///   form and the BW↔Q relation in octaves, using bilinear transform design.
/// - Filter is applied in **Direct‑Form II Transposed** for good numerical behavior.
///
/// (RBJ/W3C “Audio EQ Cookbook”)
pub fn bandpass_constant_peak<T: Float>(
    data: &mut Array1<T>,
    low_cutoff: T,
    high_cutoff: T,
    center_frequency: Option<T>,
    sample_rate: Option<T>,
) -> Result<(), &'static str> {
    // --- Parameter checks and coefficient design ---
    let (b0, b1, b2, a1, a2) =
        design_biquad_coeffs::<T>(low_cutoff, high_cutoff, center_frequency, sample_rate)?;

    // --- Apply in-place (DF2-Transposed) ---
    let mut z1 = T::zero();
    let mut z2 = T::zero();

    // Indexed loop to avoid borrow issues.
    for i in 0..data.len() {
        let x = data[i];
        let y = b0 * x + z1;
        // b1 is zero for constant-peak form, but keep it explicit for clarity.
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        data[i] = y;
    }

    Ok(())
}

/// Internal: compute normalized biquad coefficients (a0 normalized to 1) for the
/// RBJ/W3C constant‑peak band‑pass.
///
/// Returns (b0, b1, b2, a1, a2) with `a0 = 1`.
fn design_biquad_coeffs<T: Float>(
    low_cutoff: T,
    high_cutoff: T,
    center_frequency: Option<T>,
    sample_rate: Option<T>,
) -> Result<(T, T, T, T, T), &'static str> {
    // Basic input validation (finite, positive, low < high)
    if !(low_cutoff.is_finite() && high_cutoff.is_finite()) {
        return Err("cutoffs must be finite");
    }
    let zero = T::zero();
    if low_cutoff <= zero || high_cutoff <= zero {
        return Err("cutoffs must be positive");
    }
    let f1 = low_cutoff.min(high_cutoff);
    let f2 = low_cutoff.max(high_cutoff);
    if f1 >= f2 {
        return Err("low_cutoff must be < high_cutoff");
    }

    // Constants
    let two = T::from(2.0).unwrap();
    let one = T::from(1.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let ln2_over_2 = T::from(core::f64::consts::LN_2 / 2.0).unwrap();
    let eps = T::from(1.0e-7).unwrap();

    // Determine mode (absolute‑units vs normalized), and set (f0, fs).
    let (f0, fs) = match (center_frequency, sample_rate) {
        (Some(f0), Some(fs)) => {
            if !(f1 < f0 && f0 < f2) {
                return Err("center_frequency must lie strictly between the cutoffs");
            }
            if fs <= zero {
                return Err("sample_rate must be positive");
            }
            if f2 >= fs * T::from(0.5).unwrap() {
                return Err("high_cutoff must be < Nyquist");
            }
            (f0, fs)
        }
        (None, _) => {
            // Normalized mode: Nyquist = 1.0 => choose Fs = 2.0 so that Nyquist = 1.
            if !(f2 < one) {
                return Err("in normalized mode, high_cutoff must be < 1.0 (Nyquist)");
            }
            let f0 = (f1 * f2).sqrt();
            (f0, two)
        }
        (Some(_), None) => {
            return Err("when center_frequency is Some(_), sample_rate must also be provided");
        }
    };

    // Digital radian frequency
    let w0 = two * pi * (f0 / fs);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();

    // Guard: center too close to DC or Nyquist leads to poor numerical behavior
    if sin_w0.abs() < eps || w0 <= T::zero() || w0 >= pi {
        return Err("center frequency too close to DC or Nyquist for stable design");
    }

    // Bandwidth in octaves and alpha (cookbook BW case):
    // BW_oct = log2(f2/f1)
    // alpha  = sin(w0) * sinh( (ln2/2) * BW_oct * (w0 / sin(w0)) )
    let bw_oct = (f2 / f1).log2();
    let k = ln2_over_2 * bw_oct * (w0 / sin_w0);
    let alpha = sin_w0 * k.sinh();

    // Constant 0 dB peak variant (normalized by a0):
    // b0= alpha/a0, b1= 0, b2= -alpha/a0; a0=1+alpha, a1= -2cos(w0)/a0, a2= (1-alpha)/a0
    let a0 = one + alpha;
    let b0 = alpha / a0;
    let b1 = T::zero(); // explicitly zero in this variant
    let b2 = -alpha / a0;
    let a1 = (-two * cos_w0) / a0;
    let a2 = (one - alpha) / a0;

    Ok((b0, b1, b2, a1, a2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{s, Array1};

    // ---------- helpers ----------
    fn sine_fs(freq: f32, fs: f32, n: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            let t = i as f32 / fs;
            v[i] = (2.0 * std::f32::consts::PI * freq * t).sin();
        }
        v
    }
    fn sine_norm(freq_norm: f32, n: usize) -> Array1<f32> {
        // Fs = 2 => Nyquist = 1.0 (ω = π f_norm per sample)
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            v[i] = (std::f32::consts::PI * freq_norm * i as f32).sin();
        }
        v
    }
    fn step(n: usize, value: f32) -> Array1<f32> {
        Array1::from_elem(n, value)
    }
    fn impulse(n: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(n);
        if n > 0 {
            v[0] = 1.0;
        }
        v
    }
    fn rms(x: &Array1<f32>) -> f32 {
        // Skip transient to approximate steady-state
        let skip = (x.len() as f32 * 0.2) as usize;
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for &xi in x.iter().skip(skip) {
            acc += xi * xi;
            cnt += 1;
        }
        (acc / cnt as f32).sqrt()
    }
    fn rel_db(a: f32, b: f32) -> f32 {
        20.0 * (a / b).log10()
    }

    // ---------- 1) Normalized mode basic band behavior ----------
    #[test]
    fn bandpass_normalized_mode() {
        let n = 16_384;
        let low = 0.25_f32;
        let high = 0.50_f32;
        let x_low = sine_norm(0.10, n);
        let x_mid = sine_norm((low * high).sqrt(), n);
        let x_high = sine_norm(0.75, n);

        let mut y_low = x_low.clone();
        let mut y_mid = x_mid.clone();
        let mut y_high = x_high.clone();

        bandpass_constant_peak(&mut y_low, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y_mid, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y_high, low, high, None, None).unwrap();

        let r_in_mid = rms(&x_mid);
        let r_low = rms(&y_low);
        let r_mid = rms(&y_mid);
        let r_high = rms(&y_high);

        assert_relative_eq!(r_mid / r_in_mid, 1.0, max_relative = 0.15);

        let att_low_rel_db = 20.0 * (r_low / r_mid).log10();
        let att_high_rel_db = 20.0 * (r_high / r_mid).log10();
        assert!(
            att_low_rel_db <= -12.0,
            "low-side attenuation too small: {att_low_rel_db:.2} dB"
        );
        assert!(
            att_high_rel_db <= -12.0,
            "high-side attenuation too small: {att_high_rel_db:.2} dB"
        );
    }

    // ---------- 2) Absolute-units mode (e.g., MHz) ----------
    #[test]
    fn bandpass_absolute_units_mhz() {
        let n = 16_384;
        let fs_mhz = 1000.0;
        let low = 50.0;
        let high = 150.0;
        let center = 100.0;

        let x_low = sine_fs(20.0, fs_mhz, n);
        let x_mid = sine_fs(center, fs_mhz, n);
        let x_high = sine_fs(400.0, fs_mhz, n);

        let mut y_low = x_low.clone();
        let mut y_mid = x_mid.clone();
        let mut y_high = x_high.clone();

        bandpass_constant_peak(&mut y_low, low, high, Some(center), Some(fs_mhz)).unwrap();
        bandpass_constant_peak(&mut y_mid, low, high, Some(center), Some(fs_mhz)).unwrap();
        bandpass_constant_peak(&mut y_high, low, high, Some(center), Some(fs_mhz)).unwrap();

        let r_in_mid = rms(&x_mid);
        let r_low = rms(&y_low);
        let r_mid = rms(&y_mid);
        let r_high = rms(&y_high);

        assert_relative_eq!(r_mid / r_in_mid, 1.0, max_relative = 0.15);
        assert!(20.0 * (r_low / r_mid).log10() <= -12.0);
        assert!(20.0 * (r_high / r_mid).log10() <= -12.0);
    }

    // ---------- 3) Unity at center across several normalized bands ----------
    #[test]
    fn unity_at_center_multiple_bands() {
        let n = 16384;
        let bands = [
            (0.10_f32, 0.25_f32),
            (0.20_f32, 0.40_f32),
            (0.30_f32, 0.55_f32),
        ];
        for (low, high) in bands {
            let f0 = (low * high).sqrt();
            let x = sine_norm(f0, n);
            let mut y = x.clone();
            bandpass_constant_peak(&mut y, low, high, None, None).unwrap();
            let g = rms(&y) / rms(&x);
            assert_relative_eq!(g, 1.0, max_relative = 0.15);
        }
    }

    // ---------- 4) −3 dB at band edges (normalized) ----------
    #[test]
    fn minus_3db_at_edges_normalized() {
        let n = 16384;
        let low = 0.20_f32;
        let high = 0.40_f32;
        let f0 = (low * high).sqrt();

        let x_c = sine_norm(f0, n);
        let x_l = sine_norm(low, n);
        let x_h = sine_norm(high, n);

        let mut y_c = x_c.clone();
        let mut y_l = x_l.clone();
        let mut y_h = x_h.clone();

        bandpass_constant_peak(&mut y_c, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y_l, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y_h, low, high, None, None).unwrap();

        let rc = rms(&y_c);
        let rl = rms(&y_l);
        let rh = rms(&y_h);
        let gl = rl / rc;
        let gh = rh / rc;

        assert!(
            (0.60..=0.82).contains(&gl),
            "left edge gain {gl} not near 0.707 (−3 dB)"
        );
        assert!(
            (0.60..=0.82).contains(&gh),
            "right edge gain {gh} not near 0.707 (−3 dB)"
        );
    }

    // ---------- 5) DC and near‑Nyquist rejection ----------
    #[test]
    fn rejects_dc_and_near_nyquist() {
        let n = 16384;
        let low = 0.15_f32;
        let high = 0.45_f32;

        let x_dc = step(n, 1.0);
        let x_nyq = sine_norm(0.98, n);

        let mut y_dc = x_dc.clone();
        let mut y_nyq = x_nyq.clone();

        bandpass_constant_peak(&mut y_dc, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y_nyq, low, high, None, None).unwrap();

        let r_dc = rms(&y_dc);
        let r_nyq = rms(&y_nyq);
        assert!(r_dc <= 1e-3, "DC not sufficiently rejected (rms={r_dc})");

        let x_mid = sine_norm((low * high).sqrt(), n);
        let mut y_mid = x_mid.clone();
        bandpass_constant_peak(&mut y_mid, low, high, None, None).unwrap();
        let r_mid = rms(&y_mid);
        let att_nyq_db = rel_db(r_nyq, r_mid);
        assert!(
            att_nyq_db <= -12.0,
            "near-Nyquist attenuation too small: {att_nyq_db:.2} dB"
        );
    }

    // ---------- 6) Linearity: scaling and superposition ----------
    #[test]
    fn linearity_scaling_and_superposition() {
        let n = 8192;
        let low = 0.20_f32;
        let high = 0.40_f32;

        let x1 = sine_norm((low * high).sqrt(), n); // in-band
        let x2 = sine_norm(0.90, n); // out-of-band

        let alpha = 3.5_f32;

        // y1
        let mut y1 = x1.clone();
        bandpass_constant_peak(&mut y1, low, high, None, None).unwrap();

        // Scaling: F{αx} = αF{x}
        let mut y1_scaled = x1.mapv(|v| alpha * v);
        bandpass_constant_peak(&mut y1_scaled, low, high, None, None).unwrap();
        let scaled_ref = y1.mapv(|v| alpha * v);
        let err_scale = (&y1_scaled - &scaled_ref).mapv(|e| e.abs()).sum() / y1_scaled.len() as f32;
        assert!(
            err_scale < 1e-5,
            "violates scaling property (avg abs error {err_scale})"
        );

        // Superposition: F{x1 + x2} = F{x1} + F{x2}
        let mut y1_again = x1.clone();
        let mut y2 = x2.clone();
        bandpass_constant_peak(&mut y1_again, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y2, low, high, None, None).unwrap();

        let mut y_sum = (&x1 + &x2).to_owned();
        bandpass_constant_peak(&mut y_sum, low, high, None, None).unwrap();

        let err_super = (&y_sum - (&y1_again + &y2)).mapv(|e| e.abs()).sum() / y_sum.len() as f32;
        assert!(
            err_super < 1e-5,
            "violates superposition (avg abs error {err_super})"
        );
    }

    // ---------- 7) Stability: no NaNs/Infs & bounded output ----------
    #[test]
    fn stable_no_nans_infs() {
        let n = 10_000;
        let low = 0.10_f32;
        let high = 0.30_f32;

        // Random-like mixture of tones; stay within [-1, 1]
        let x = {
            let s1 = sine_norm(0.05, n);
            let s2 = sine_norm(0.20, n);
            let s3 = sine_norm(0.70, n);
            (s1 + s2 + s3).mapv(|v| 0.33 * v)
        };

        let mut y = x.clone();
        bandpass_constant_peak(&mut y, low, high, None, None).unwrap();

        for &v in y.iter() {
            assert!(v.is_finite(), "non-finite sample encountered");
        }

        let peak = y.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 5.0, "unexpectedly large peak {peak}");
    }

    // ---------- 8) Equivalence: absolute-units vs normalized (log-symmetric band) ----------
    #[test]
    fn absolute_vs_normalized_equivalence() {
        let n = 16384;
        let fs_mhz = 1000.0; // Nyquist 500 MHz

        // Pick a log-symmetric band around 100 MHz so that sqrt(low*high) = 100 MHz.
        let low_mhz = 80.0;
        let high_mhz = 125.0;
        let center_mhz = 100.0;

        let nyq = fs_mhz * 0.5;
        let low_n = low_mhz / nyq;
        let high_n = high_mhz / nyq;

        // Probe input: a mixture
        let x = {
            let a = sine_fs(center_mhz, fs_mhz, n);
            let b = sine_fs(0.90 * low_mhz, fs_mhz, n);
            let c = sine_fs(0.90 * high_mhz, fs_mhz, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };

        let mut y_abs = x.clone();
        bandpass_constant_peak(
            &mut y_abs,
            low_mhz,
            high_mhz,
            Some(center_mhz),
            Some(fs_mhz),
        )
        .unwrap();

        // Build an equivalent normalized input
        let x_norm = {
            let a = sine_norm(center_mhz / nyq, n);
            let b = sine_norm(0.90 * low_mhz / nyq, n);
            let c = sine_norm(0.90 * high_mhz / nyq, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };

        let mut y_norm = x_norm.clone();
        bandpass_constant_peak(&mut y_norm, low_n, high_n, None, None).unwrap();

        assert_relative_eq!(rms(&y_abs), rms(&y_norm), max_relative = 0.05);
    }

    // ---------- 9) Parameter validation ----------
    #[test]
    fn parameter_validation() {
        let x = Array1::<f32>::zeros(32);

        // Normalized mode: require 0 < low < high < 1
        assert!(bandpass_constant_peak(&mut x.clone(), 0.0, 0.5, None, None).is_err());
        assert!(bandpass_constant_peak(&mut x.clone(), 0.5, 1.0, None, None).is_err());

        // Absolute mode requires center & fs, and high < Nyquist
        assert!(bandpass_constant_peak(&mut x.clone(), 50.0, 150.0, Some(100.0), None).is_err());
        assert!(
            bandpass_constant_peak(&mut x.clone(), 50.0, 150.0, Some(40.0), Some(1000.0)).is_err()
        ); // center not between cutoffs
        assert!(
            bandpass_constant_peak(&mut x.clone(), 50.0, 600.0, Some(100.0), Some(1000.0)).is_err()
        ); // high >= Nyq
    }

    // ---------- 10) DC gain ~ 0 via step ----------
    #[test]
    fn dc_gain_near_zero_via_step() {
        let n = 16384;
        let mut y = step(n, 1.0);
        bandpass_constant_peak(&mut y, 0.2, 0.4, None, None).unwrap();
        let half = n / 2;
        let tail_rms = {
            let tail = y.slice(s![half..]).to_owned();
            rms(&tail)
        };
        assert!(
            tail_rms < 1e-3,
            "step steady-state not near zero (rms={tail_rms})"
        );
    }

    // ---------- 11) Finite impulse-response energy (sanity) ----------
    #[test]
    fn impulse_finite_energy() {
        let n = 4096;
        let mut h = impulse(n);
        bandpass_constant_peak(&mut h, 0.2, 0.4, None, None).unwrap();
        let energy: f32 = h.iter().map(|v| v * v).sum();
        assert!(energy.is_finite() && energy > 0.0 && energy < 1000.0);
    }
    #[test]
    fn cascade_improves_skirts() {
        let n = 16_384;
        let low = 0.25_f32;
        let high = 0.50_f32;
        let f0 = (low * high).sqrt();

        let x_center = {
            // tone exactly at geometric center
            let mut v = Array1::<f32>::zeros(n);
            for i in 0..n {
                v[i] = (std::f32::consts::PI * f0 * i as f32).sin();
            }
            v
        };
        let x_low = {
            let mut v = Array1::<f32>::zeros(n);
            for i in 0..n {
                v[i] = (std::f32::consts::PI * 0.10 * i as f32).sin();
            }
            v
        };

        // 1 section
        let mut y1_c = x_center.clone();
        bandpass_constant_peak(&mut y1_c, low, high, None, None).unwrap();
        let mut y1_l = x_low.clone();
        bandpass_constant_peak(&mut y1_l, low, high, None, None).unwrap();

        // 2 sections (run same section twice)
        let mut y2_c = x_center.clone();
        bandpass_constant_peak(&mut y2_c, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y2_c, low, high, None, None).unwrap();

        let mut y2_l = x_low.clone();
        bandpass_constant_peak(&mut y2_l, low, high, None, None).unwrap();
        bandpass_constant_peak(&mut y2_l, low, high, None, None).unwrap();

        // Center remains ~unity
        let g1 = rms(&y1_c) / rms(&x_center);
        let g2 = rms(&y2_c) / rms(&x_center);
        assert_relative_eq!(g1, 1.0, max_relative = 0.15);
        assert_relative_eq!(g2, 1.0, max_relative = 0.15);

        // Out-of-band attenuation improves (more negative dB)
        let att1 = 20.0 * (rms(&y1_l) / rms(&y1_c)).log10();
        let att2 = 20.0 * (rms(&y2_l) / rms(&y2_c)).log10();
        assert!(
            att2 < att1 - 5.0,
            "cascading did not clearly improve attenuation"
        );
    }
}
