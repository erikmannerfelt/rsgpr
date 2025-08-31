use ndarray::{ArrayBase, DataMut, Ix1};
use num::Float;
use num_complex::Complex;

/// Apply a band‑pass by running a **high‑pass at low_cutoff** then a **low‑pass at high_cutoff**.
/// Two biquads total (4th‑order overall), steeper skirts than a single RBJ band‑pass,
/// with only two passes over the buffer.
///
/// Modes:
/// - Absolute units: pass `Some(fs)`; cutoffs in same unit; require 0 < low < high < fs/2.
/// - Normalized: pass `None` for `sample_rate`, and treat cutoffs in (0, 1) where 1 = Nyquist.
///
/// `q`: section damping (default ~0.707). `normalize_at_center`: if true, apply one constant gain so
/// the response is ~0 dB at f0 = sqrt(low*high).
///
/// Coefficients follow RBJ/W3C cookbook HPF/LPF; filter is DF‑II‑Transposed for good numerics.
pub fn bandpass_hpf_then_lpf<T: Float, S: DataMut<Elem = T>>(
    data: &mut ArrayBase<S, Ix1>,
    low_cutoff: T,
    high_cutoff: T,
    sample_rate: Option<T>,
    q: Option<T>,
    normalize_at_center: bool,
) -> Result<(), &'static str> {
    // ----- Validate -----
    if !(low_cutoff.is_finite() && high_cutoff.is_finite()) {
        return Err("cutoffs must be finite");
    }
    let zero = T::zero();
    if low_cutoff <= zero || high_cutoff <= zero {
        return Err("cutoffs must be positive");
    }
    if low_cutoff >= high_cutoff {
        return Err("low_cutoff must be < high_cutoff");
    }

    let one = T::one();
    let two = T::from(2.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let q = q.unwrap_or_else(|| T::from(1.0 / 2f64.sqrt()).unwrap()); // ~0.7071

    // Normalized vs absolute
    let fs = match sample_rate {
        Some(fs) => {
            if fs <= zero {
                return Err("sample_rate must be positive");
            }
            let nyq = fs * T::from(0.5).unwrap();
            if high_cutoff >= nyq {
                return Err("high_cutoff must be < Nyquist");
            }
            fs
        }
        None => {
            if high_cutoff >= one {
                return Err("in normalized mode, high_cutoff must be < 1.0 (Nyquist)");
            }
            two // Fs = 2 => Nyquist = 1
        }
    };

    // ----- Design HPF(low) and LPF(high) -----
    let (b0_h, b1_h, b2_h, a1_h, a2_h) = design_hpf_rbj::<T>(low_cutoff, fs, q)?;
    let (b0_l, b1_l, b2_l, a1_l, a2_l) = design_lpf_rbj::<T>(high_cutoff, fs, q)?;

    // Optional: normalize ~0 dB at geometric center
    let gain = if normalize_at_center {
        let f0 = (low_cutoff * high_cutoff).sqrt();
        let w = two * pi * (f0 / fs);
        let h_hpf = biquad_h_ejw::<T>(w, b0_h, b1_h, b2_h, a1_h, a2_h);
        let h_lpf = biquad_h_ejw::<T>(w, b0_l, b1_l, b2_l, a1_l, a2_l);
        let mag = complex_abs(h_hpf * h_lpf);
        if mag > T::from(1e-12).unwrap() {
            one / mag
        } else {
            one
        }
    } else {
        one
    };

    // ----- Run HPF then LPF (DF2‑T) -----
    apply_biquad_df2t_in_place(data, b0_h, b1_h, b2_h, a1_h, a2_h);
    apply_biquad_df2t_in_place(data, b0_l, b1_l, b2_l, a1_l, a2_l);
    if gain != one {
        for i in 0..data.len() {
            data[i] = data[i] * gain;
        }
    }

    Ok(())
}

/// RBJ/W3C HPF biquad with a0 normalized to 1 (case: Q).
/// Returns (b0, b1, b2, a1, a2).
fn design_hpf_rbj<T: Float>(f_c: T, fs: T, q: T) -> Result<(T, T, T, T, T), &'static str> {
    let zero = T::zero();
    if !(f_c > zero && fs > zero && q > zero) {
        return Err("invalid params");
    }
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let w0 = two * pi * (f_c / fs);
    let sw = w0.sin();
    let cw = w0.cos();
    let alpha = sw / (two * q);
    let b0 = (one + cw) / two;
    let b1 = -(one + cw);
    let b2 = (one + cw) / two;
    let a0 = one + alpha;
    let a1 = -two * cw;
    let a2 = one - alpha;
    Ok((b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0))
}

/// RBJ/W3C LPF biquad with a0 normalized to 1 (case: Q).
/// Returns (b0, b1, b2, a1, a2).
fn design_lpf_rbj<T: Float>(f_c: T, fs: T, q: T) -> Result<(T, T, T, T, T), &'static str> {
    let zero = T::zero();
    if !(f_c > zero && fs > zero && q > zero) {
        return Err("invalid params");
    }
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let pi = T::from(core::f64::consts::PI).unwrap();
    let w0 = two * pi * (f_c / fs);
    let sw = w0.sin();
    let cw = w0.cos();
    let alpha = sw / (two * q);
    let b0 = (one - cw) / two;
    let b1 = one - cw;
    let b2 = (one - cw) / two;
    let a0 = one + alpha;
    let a1 = -two * cw;
    let a2 = one - alpha;
    Ok((b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0))
}

/// Apply one biquad in‑place, Direct‑Form II Transposed, with a0 assumed 1.
fn apply_biquad_df2t_in_place<T: Float, S: DataMut<Elem = T>>(
    data: &mut ArrayBase<S, Ix1>,
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
) {
    let mut z1 = T::zero();
    let mut z2 = T::zero();
    for i in 0..data.len() {
        let x = data[i];
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        data[i] = y;
    }
}

/// H(e^{jw}) of a normalized (a0=1) biquad at angular frequency w.
fn biquad_h_ejw<T: Float>(w: T, b0: T, b1: T, b2: T, a1: T, a2: T) -> Complex<T> {
    // e^{-jw} = cos(w) - j sin(w); e^{-j2w} likewise
    let two = T::from(2.0).unwrap();
    let ejw = Complex::new(w.cos(), -(w.sin()));
    let ej2w = Complex::new((two * w).cos(), -((two * w).sin()));
    let zero = T::zero();
    let one = T::one();
    let num = Complex::new(b0, zero) + ejw * b1 + ej2w * b2;
    let den = Complex::new(one, zero) + ejw * a1 + ej2w * a2;
    num / den
}

/// Magnitude of a complex number for generic Float `T`.
fn complex_abs<T: Float>(c: Complex<T>) -> T {
    (c.re * c.re + c.im * c.im).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{s, Array1};

    // ----------- helpers -----------
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

    // 1) Normalized mode basic band behavior
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
        bandpass_hpf_then_lpf(&mut y_low, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_mid, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_high, low, high, None, None, true).unwrap();
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

    // 2) Absolute-units mode (e.g., MHz)
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
        bandpass_hpf_then_lpf(&mut y_low, low, high, Some(fs_mhz), None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_mid, low, high, Some(fs_mhz), None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_high, low, high, Some(fs_mhz), None, true).unwrap();
        let r_in_mid = rms(&x_mid);
        let r_low = rms(&y_low);
        let r_mid = rms(&y_mid);
        let r_high = rms(&y_high);
        assert_relative_eq!(r_mid / r_in_mid, 1.0, max_relative = 0.15);
        assert!(20.0 * (r_low / r_mid).log10() <= -12.0);
        assert!(20.0 * (r_high / r_mid).log10() <= -12.0);
    }

    // 3) Unity at center across several normalized bands
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
            bandpass_hpf_then_lpf(&mut y, low, high, None, None, true).unwrap();
            let g = rms(&y) / rms(&x);
            assert_relative_eq!(g, 1.0, max_relative = 0.15);
        }
    }

    // 4) Edges are clearly below center (not asserting exact −3 dB)
    #[test]
    fn edges_below_center_normalized() {
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
        bandpass_hpf_then_lpf(&mut y_c, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_l, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_h, low, high, None, None, true).unwrap();
        let rc = rms(&y_c);
        let rl = rms(&y_l);
        let rh = rms(&y_h);
        let gl = rl / rc;
        let gh = rh / rc;
        // For HPF→LPF product, edges are clearly below center; do not rely on exact -3 dB.
        assert!(gl < 0.9 && gl > 0.45, "left edge gain {gl} unexpected");
        assert!(gh < 0.9 && gh > 0.45, "right edge gain {gh} unexpected");
    }

    // 5) DC and near‑Nyquist rejection
    #[test]
    fn rejects_dc_and_near_nyquist() {
        let n = 16384;
        let low = 0.15_f32;
        let high = 0.45_f32;
        let x_dc = step(n, 1.0);
        let x_nyq = sine_norm(0.98, n);
        let mut y_dc = x_dc.clone();
        let mut y_nyq = x_nyq.clone();
        bandpass_hpf_then_lpf(&mut y_dc, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y_nyq, low, high, None, None, true).unwrap();
        let r_dc = rms(&y_dc);
        let r_nyq = rms(&y_nyq);
        assert!(r_dc <= 1e-3, "DC not sufficiently rejected (rms={r_dc})");
        let x_mid = sine_norm((low * high).sqrt(), n);
        let mut y_mid = x_mid.clone();
        bandpass_hpf_then_lpf(&mut y_mid, low, high, None, None, true).unwrap();
        let r_mid = rms(&y_mid);
        let att_nyq_db = rel_db(r_nyq, r_mid);
        assert!(
            att_nyq_db <= -12.0,
            "near-Nyquist attenuation too small: {att_nyq_db:.2} dB"
        );
    }

    // 6) Linearity: scaling and superposition
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
        bandpass_hpf_then_lpf(&mut y1, low, high, None, None, true).unwrap();
        // Scaling: F{αx} = αF{x}
        let mut y1_scaled = x1.mapv(|v| alpha * v);
        bandpass_hpf_then_lpf(&mut y1_scaled, low, high, None, None, true).unwrap();
        let scaled_ref = y1.mapv(|v| alpha * v);
        let err_scale = (&y1_scaled - &scaled_ref).mapv(|e| e.abs()).sum() / y1_scaled.len() as f32;
        assert!(
            err_scale < 1e-5,
            "violates scaling property (avg abs error {err_scale})"
        );
        // Superposition: F{x1 + x2} = F{x1} + F{x2}
        let mut y1_again = x1.clone();
        let mut y2 = x2.clone();
        bandpass_hpf_then_lpf(&mut y1_again, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y2, low, high, None, None, true).unwrap();
        let mut y_sum = (&x1 + &x2).to_owned();
        bandpass_hpf_then_lpf(&mut y_sum, low, high, None, None, true).unwrap();
        let err_super = (&y_sum - (&y1_again + &y2)).mapv(|e| e.abs()).sum() / y_sum.len() as f32;
        assert!(
            err_super < 1e-5,
            "violates superposition (avg abs error {err_super})"
        );
    }

    // 7) Stability: no NaNs/Infs & bounded output
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
        bandpass_hpf_then_lpf(&mut y, low, high, None, None, true).unwrap();
        for &v in y.iter() {
            assert!(v.is_finite(), "non-finite sample encountered");
        }
        let peak = y.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 5.0, "unexpectedly large peak {peak}");
    }

    // 8) Equivalence: absolute-units vs normalized (log-symmetric band)
    #[test]
    fn absolute_vs_normalized_equivalence() {
        let n = 16384;
        let fs_mhz = 1000.0; // Nyquist 500 MHz
                             // Pick a log-symmetric band around 100 MHz so that sqrt(low*high) = 100 MHz.
        let low_mhz = 80.0;
        let high_mhz = 125.0;
        let nyq = fs_mhz * 0.5;
        let low_n = low_mhz / nyq;
        let high_n = high_mhz / nyq;
        // Probe input: a mixture
        let x = {
            let a = sine_fs(100.0, fs_mhz, n);
            let b = sine_fs(0.90 * low_mhz, fs_mhz, n);
            let c = sine_fs(0.90 * high_mhz, fs_mhz, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };
        let mut y_abs = x.clone();
        bandpass_hpf_then_lpf(&mut y_abs, low_mhz, high_mhz, Some(fs_mhz), None, true).unwrap();
        // Build an equivalent normalized input
        let x_norm = {
            let a = sine_norm(100.0 / nyq, n);
            let b = sine_norm(0.90 * low_mhz / nyq, n);
            let c = sine_norm(0.90 * high_mhz / nyq, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };
        let mut y_norm = x_norm.clone();
        bandpass_hpf_then_lpf(&mut y_norm, low_n, high_n, None, None, true).unwrap();
        assert_relative_eq!(rms(&y_abs), rms(&y_norm), max_relative = 0.05);
    }

    // 9) Parameter validation
    #[test]
    fn parameter_validation() {
        let x = Array1::<f32>::zeros(32);
        // Normalized mode: require 0 < low < high < 1
        assert!(bandpass_hpf_then_lpf(&mut x.clone(), 0.0, 0.5, None, None, true).is_err());
        assert!(bandpass_hpf_then_lpf(&mut x.clone(), 0.5, 1.0, None, None, true).is_err());
        assert!(bandpass_hpf_then_lpf(&mut x.clone(), 0.3, 0.2, None, None, true).is_err());
        // Invalid Q
        assert!(bandpass_hpf_then_lpf(&mut x.clone(), 0.2, 0.4, None, Some(0.0), true).is_err());
        // Absolute mode requires fs>0 and high < Nyquist
        assert!(bandpass_hpf_then_lpf(&mut x.clone(), 50.0, 150.0, Some(0.0), None, true).is_err());
        assert!(
            bandpass_hpf_then_lpf(&mut x.clone(), 50.0, 600.0, Some(1000.0), None, true).is_err()
        ); // high >= Nyq
    }

    // 10) DC gain ~ 0 via step
    #[test]
    fn dc_gain_near_zero_via_step() {
        let n = 16384;
        let mut y = step(n, 1.0);
        bandpass_hpf_then_lpf(&mut y, 0.2, 0.4, None, None, true).unwrap();
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

    // 11) Finite impulse-response energy (sanity)
    #[test]
    fn impulse_finite_energy() {
        let n = 4096;
        let mut h = impulse(n);
        bandpass_hpf_then_lpf(&mut h, 0.2, 0.4, None, None, true).unwrap();
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
        // 1 pass
        let mut y1_c = x_center.clone();
        bandpass_hpf_then_lpf(&mut y1_c, low, high, None, None, true).unwrap();
        let mut y1_l = x_low.clone();
        bandpass_hpf_then_lpf(&mut y1_l, low, high, None, None, true).unwrap();
        // 2 passes (8th order overall)
        let mut y2_c = x_center.clone();
        bandpass_hpf_then_lpf(&mut y2_c, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y2_c, low, high, None, None, true).unwrap();
        let mut y2_l = x_low.clone();
        bandpass_hpf_then_lpf(&mut y2_l, low, high, None, None, true).unwrap();
        bandpass_hpf_then_lpf(&mut y2_l, low, high, None, None, true).unwrap();
        // Center remains ~unity
        let g1 = rms(&y1_c) / rms(&x_center);
        let g2 = rms(&y2_c) / rms(&x_center);
        assert_relative_eq!(g1, 1.0, max_relative = 0.15);
        assert_relative_eq!(g2, 1.0, max_relative = 0.20);
        // Out-of-band attenuation improves (more negative dB)
        let att1 = 20.0 * (rms(&y1_l) / rms(&y1_c)).log10();
        let att2 = 20.0 * (rms(&y2_l) / rms(&y2_c)).log10();
        assert!(
            att2 < att1 - 5.0,
            "cascading did not clearly improve attenuation"
        );
    }
}
