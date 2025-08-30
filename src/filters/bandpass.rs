use ndarray::Array1;

/// Design and run a constant-peak (0 dB at center) band-pass biquad using RBJ/W3C formulas.
/// - If `center_frequency` is `Some(f0)` you must also pass `sample_rate = Some(fs)`.
///   All frequencies (low/high/center/sample_rate) must be in the **same unit** (e.g., MHz for GPR).
/// - If `center_frequency` is `None`, `low_cutoff` & `high_cutoff` are interpreted as
///   **normalized to Nyquist** in (0,1), i.e., `1.0` = Nyquist. `sample_rate` is ignored.
///
/// Returns a new `Array1<f32>` with the filtered samples.
///
/// References:
///   - W3C “Audio EQ Cookbook” (adapted from RBJ): band-pass (constant 0 dB peak), digital BW–Q relation
///     and bilinear-transform prewarping terms.  https://www.w3.org/TR/audio-eq-cookbook/
///   - DF2-Transposed is a standard form for SOS IIRs with good numerical behavior.
pub fn bandpass_constant_peak(
    data: &Array1<f32>,
    low_cutoff: f32,
    high_cutoff: f32,
    center_frequency: Option<f32>,
    sample_rate: Option<f32>,
) -> Result<Array1<f32>, &'static str> {
    if !(low_cutoff.is_finite() && high_cutoff.is_finite()) {
        return Err("cutoffs must be finite");
    }
    if low_cutoff <= 0.0 || high_cutoff <= 0.0 {
        return Err("cutoffs must be positive");
    }

    let f1 = low_cutoff.min(high_cutoff);
    let f2 = low_cutoff.max(high_cutoff);
    if f1 >= f2 {
        return Err("low_cutoff must be < high_cutoff");
    }

    // Determine mode and set (f0, Fs) consistently.
    let (f0, fs, normalized_mode) = match (center_frequency, sample_rate) {
        (Some(f0), Some(fs)) => {
            if !(f1 < f0 && f0 < f2) {
                return Err("center_frequency must lie strictly between the cutoffs");
            }
            if fs <= 0.0 {
                return Err("sample_rate must be positive");
            }
            if f2 >= fs * 0.5 {
                return Err("high_cutoff must be < Nyquist");
            }
            (f0, fs, false)
        }
        (None, _) => {
            // Normalized mode: 1.0 == Nyquist -> use Fs = 2 so Nyquist = 1
            if !(f2 < 1.0) {
                return Err("in normalized mode, high_cutoff must be < 1.0 (Nyquist)");
            }
            let f0 = (f1 * f2).sqrt();
            (f0, 2.0_f32, true)
        }
        (Some(_), None) => {
            return Err("when center_frequency is Some(_), sample_rate must also be provided");
        }
    };

    // RBJ/W3C: digital design (BLT) with prewarping-aware bandwidth relation.           [1](https://e2e.ti.com/support/audio-group/audio/f/audio-forum/389579/aic3254-coefficients-b0-b1-b2-a0-a1-a2-calculation-for-bandpass-digital-biquad-filters)
    let w0 = 2.0 * std::f32::consts::PI * (f0 / fs);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    if sin_w0.abs() < 1e-7 || w0 <= 0.0 || w0 >= std::f32::consts::PI {
        return Err("center frequency too close to DC or Nyquist for stable design");
    }

    // Bandwidth in octaves and alpha (digital) per cookbook (BW case).
    //   BW_oct = log2(f2/f1)
    //   alpha  = sin(w0) * sinh( (ln2/2) * BW_oct * (w0 / sin(w0)) )
    let bw_oct = (f2 / f1).log2();
    let ln2_over_2 = std::f32::consts::LN_2 * 0.5;
    let k = ln2_over_2 * bw_oct * (w0 / sin_w0);
    let alpha = sin_w0 * k.sinh();

    // Constant 0 dB peak variant: b0=alpha, b1=0, b2=-alpha; a0=1+alpha, a1=-2cos(w0), a2=1-alpha.  [1](https://e2e.ti.com/support/audio-group/audio/f/audio-forum/389579/aic3254-coefficients-b0-b1-b2-a0-a1-a2-calculation-for-bandpass-digital-biquad-filters)
    let a0 = 1.0 + alpha;
    let b0 = alpha / a0;
    let b1 = 0.0;
    let b2 = -alpha / a0;
    let a1 = (-2.0 * cos_w0) / a0;
    let a2 = (1.0 - alpha) / a0;

    // Direct Form 2 Transposed (normalized: a0=1).
    let mut z1 = 0.0_f32;
    let mut z2 = 0.0_f32;

    let mut out = Array1::<f32>::zeros(data.len());
    for (i, &x) in data.iter().enumerate() {
        // y[n] = b0*x[n] + z1
        let y = b0 * x + z1;
        // z1' = b1*x[n] - a1*y[n] + z2
        z1 = b1 * x - a1 * y + z2;
        // z2' = b2*x[n] - a2*y[n]
        z2 = b2 * x - a2 * y;

        out[i] = y;
    }

    // Optional sanity guard in normalized mode: enforce high_cutoff < 1.0
    if normalized_mode && !(f2 < 1.0) {
        return Err("normalized mode requires high_cutoff < 1.0 (Nyquist)");
    }

    Ok(out)
}

mod test {
    use super::*;
    use approx::assert_relative_eq;

    fn sine_fs(freq: f32, fs: f32, n: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            let t = i as f32 / fs;
            v[i] = (2.0 * std::f32::consts::PI * freq * t).sin();
        }
        v
    }

    fn sine_norm(freq_norm: f32, n: usize) -> Array1<f32> {
        // Fs=2 => Nyquist=1, omega = π f_norm per sample
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            v[i] = (std::f32::consts::PI * freq_norm * i as f32).sin();
        }
        v
    }

    fn rms(x: &Array1<f32>) -> f32 {
        let n = x.len();
        let start = (0.2 * n as f32) as usize; // drop first 20% to avoid transient
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for &xi in x.iter().skip(start) {
            acc += xi * xi;
            cnt += 1;
        }
        (acc / cnt as f32).sqrt()
    }

    #[test]
    fn bandpass_normalized_mode() {
        let n = 16_384;
        let low = 0.25_f32;
        let high = 0.50_f32;

        let x_low = sine_norm(0.10, n);
        let x_mid = sine_norm((low * high).sqrt(), n);
        let x_high = sine_norm(0.75, n);

        let y_low = bandpass_constant_peak(&x_low, low, high, None, None).unwrap();
        let y_mid = bandpass_constant_peak(&x_mid, low, high, None, None).unwrap();
        let y_high = bandpass_constant_peak(&x_high, low, high, None, None).unwrap();

        let r_in_mid = rms(&x_mid);
        let r_low = rms(&y_low);
        let r_mid = rms(&y_mid);
        let r_high = rms(&y_high);

        // Expect near-unity at center (allow ~±0.15 rel. due to finite window & discretization).
        assert_relative_eq!(r_mid / r_in_mid, 1.0, max_relative = 0.15);

        // Out-of-band attenuation relative to center.
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

    #[test]
    fn bandpass_absolute_units_mhz() {
        let n = 16_384;
        let fs_mhz = 1000.0; // 1000 MHz sampling => Nyquist 500 MHz
        let low = 50.0;
        let high = 150.0;
        let center = 100.0;

        let x_low = sine_fs(20.0, fs_mhz, n);
        let x_mid = sine_fs(center, fs_mhz, n);
        let x_high = sine_fs(400.0, fs_mhz, n);

        let y_low = bandpass_constant_peak(&x_low, low, high, Some(center), Some(fs_mhz)).unwrap();
        let y_mid = bandpass_constant_peak(&x_mid, low, high, Some(center), Some(fs_mhz)).unwrap();
        let y_high =
            bandpass_constant_peak(&x_high, low, high, Some(center), Some(fs_mhz)).unwrap();

        let r_in_mid = rms(&x_mid);
        let r_low = rms(&y_low);
        let r_mid = rms(&y_mid);
        let r_high = rms(&y_high);

        assert_relative_eq!(r_mid / r_in_mid, 1.0, max_relative = 0.15);
        let att_low_rel_db = 20.0 * (r_low / r_mid).log10();
        let att_high_rel_db = 20.0 * (r_high / r_mid).log10();
        assert!(att_low_rel_db <= -12.0);
        assert!(att_high_rel_db <= -12.0);
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{s, Array1};

    // ---------- Shared helpers ----------
    fn sine_fs(freq: f32, fs: f32, n: usize) -> Array1<f32> {
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            let t = i as f32 / fs;
            v[i] = (2.0 * std::f32::consts::PI * freq * t).sin();
        }
        v
    }

    fn sine_norm(freq_norm: f32, n: usize) -> Array1<f32> {
        // Fs = 2 -> Nyquist = 1.0 (ω = π f_norm per sample)
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

    // ---------- 1) Unity at center across several bands ----------
    #[test]
    fn unity_at_center_multiple_bands() {
        let n = 16384;

        // Choose a few normalized bands, all with geometric-center centers
        let bands = [
            (0.10_f32, 0.25_f32),
            (0.20_f32, 0.40_f32),
            (0.30_f32, 0.55_f32),
        ];

        for (low, high) in bands {
            let f0 = (low * high).sqrt();
            let x = sine_norm(f0, n);
            let y = bandpass_constant_peak(&x, low, high, None, None).unwrap();

            // RMS(y)/RMS(x) ~ 1.0 at center
            let g = rms(&y) / rms(&x);
            assert_relative_eq!(g, 1.0, max_relative = 0.15);
        }
    }

    // ---------- 2) −3 dB at band edges ----------
    // RBJ/W3C defines BW in octaves between the -3 dB frequencies for BPF.        [1](https://e2e.ti.com/support/audio-group/audio/f/audio-forum/389579/aic3254-coefficients-b0-b1-b2-a0-a1-a2-calculation-for-bandpass-digital-biquad-filters)
    #[test]
    fn minus_3db_at_edges_normalized() {
        let n = 16384;
        let low = 0.20_f32;
        let high = 0.40_f32;
        let f0 = (low * high).sqrt();

        // Probes at f0 and the edges
        let x_c = sine_norm(f0, n);
        let x_l = sine_norm(low, n);
        let x_h = sine_norm(high, n);

        let y_c = bandpass_constant_peak(&x_c, low, high, None, None).unwrap();
        let y_l = bandpass_constant_peak(&x_l, low, high, None, None).unwrap();
        let y_h = bandpass_constant_peak(&x_h, low, high, None, None).unwrap();

        let rc = rms(&y_c);
        let rl = rms(&y_l);
        let rh = rms(&y_h);

        // Edge gains should be close to -3 dB (≈ 0.707) relative to center.
        let gl = rl / rc;
        let gh = rh / rc;

        // Allow a bit of tolerance for discretization & windowing
        assert!(
            (0.60..=0.82).contains(&gl),
            "left edge gain {gl} not near 0.707 (-3 dB)"
        );
        assert!(
            (0.60..=0.82).contains(&gh),
            "right edge gain {gh} not near 0.707 (-3 dB)"
        );
    }

    // ---------- 3) DC and near‑Nyquist rejection ----------
    #[test]
    fn rejects_dc_and_near_nyquist() {
        let n = 16384;
        let low = 0.15_f32;
        let high = 0.45_f32;

        // DC via step; Nyquist-ish tone at 0.98 (avoid exact Nyquist)
        let x_dc = step(n, 1.0);
        let x_nyq = sine_norm(0.98, n);

        let y_dc = bandpass_constant_peak(&x_dc, low, high, None, None).unwrap();
        let y_nyq = bandpass_constant_peak(&x_nyq, low, high, None, None).unwrap();

        // Expect very small DC steady-state and strong attenuation near Nyquist.
        let r_dc = rms(&y_dc);
        let r_nyq = rms(&y_nyq);

        assert!(r_dc <= 1e-3, "DC not sufficiently rejected (rms={r_dc})");

        // Measure relative to in-band reference
        let x_mid = sine_norm((low * high).sqrt(), n);
        let y_mid = bandpass_constant_peak(&x_mid, low, high, None, None).unwrap();
        let r_mid = rms(&y_mid);

        let att_nyq_db = rel_db(r_nyq, r_mid);
        assert!(
            att_nyq_db <= -12.0,
            "near-Nyquist attenuation too small: {att_nyq_db:.2} dB"
        );
    }

    // ---------- 4) Linearity: scaling and superposition ----------
    #[test]
    fn linearity_scaling_and_superposition() {
        let n = 8192;
        let low = 0.20_f32;
        let high = 0.40_f32;

        let x1 = sine_norm((low * high).sqrt(), n); // in-band
        let x2 = sine_norm(0.90, n); // out-of-band
        let alpha = 3.5_f32;

        let y1 = bandpass_constant_peak(&x1, low, high, None, None).unwrap();
        let y2 = bandpass_constant_peak(&x2, low, high, None, None).unwrap();

        // Scaling: F{αx} = αF{x}
        let y1_scaled =
            bandpass_constant_peak(&(x1.mapv(|v| alpha * v)), low, high, None, None).unwrap();
        let err_scale = (&y1_scaled - y1.mapv(|v| alpha * v))
            .mapv(|e| e.abs())
            .sum()
            / y1.len() as f32;
        assert!(
            err_scale < 1e-5,
            "violates scaling property (avg abs error {err_scale})"
        );

        // Superposition: F{x1 + x2} = F{x1} + F{x2}
        let y_sum =
            bandpass_constant_peak(&(x1.clone() + x2.clone()), low, high, None, None).unwrap();
        let err_super = (&y_sum - (y1 + y2)).mapv(|e| e.abs()).sum() / y_sum.len() as f32;
        assert!(
            err_super < 1e-5,
            "violates superposition (avg abs error {err_super})"
        );
    }

    // ---------- 5) Stability: no NaNs/Infs & bounded output ----------
    #[test]
    fn stable_no_nans_infs() {
        let n = 10000;
        let low = 0.10_f32;
        let high = 0.30_f32;

        // Random-like mixture of tones; stay within [-1,1]
        let x = {
            let s1 = sine_norm(0.05, n);
            let s2 = sine_norm(0.20, n);
            let s3 = sine_norm(0.70, n);
            (s1 + s2 + s3).mapv(|v| 0.33 * v)
        };

        let y = bandpass_constant_peak(&x, low, high, None, None).unwrap();
        for &v in y.iter() {
            assert!(v.is_finite(), "non-finite sample encountered");
        }
        // Very loose bound (amplification at center is unity)
        let peak = y.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 5.0, "unexpectedly large peak {peak}");
    }

    // ---------- 6) Equivalence: absolute-units vs normalized (log-symmetric band) ----------
    // If we pick low/high so that center = sqrt(low*high), the absolute-units mode and the
    // normalized mode are equivalent after converting by fs/2.                         [1](https://e2e.ti.com/support/audio-group/audio/f/audio-forum/389579/aic3254-coefficients-b0-b1-b2-a0-a1-a2-calculation-for-bandpass-digital-biquad-filters)
    #[test]
    fn absolute_vs_normalized_equivalence() {
        let n = 16384;
        let fs_mhz = 1000.0; // Nyquist 500 MHz

        // Pick a log-symmetric band around 100 MHz so that sqrt(low*high) = 100 MHz.
        let low_mhz = 80.0;
        let high_mhz = 125.0;
        let center_mhz = 100.0;

        // Convert to normalized
        let nyq = fs_mhz * 0.5;
        let low_n = low_mhz / nyq;
        let high_n = high_mhz / nyq;

        // Probe input: a mixture to avoid trivial equality
        let x = {
            let a = sine_fs(center_mhz, fs_mhz, n);
            let b = sine_fs(0.90 * low_mhz, fs_mhz, n);
            let c = sine_fs(0.90 * high_mhz, fs_mhz, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };

        let y_abs =
            bandpass_constant_peak(&x, low_mhz, high_mhz, Some(center_mhz), Some(fs_mhz)).unwrap();

        // Build an equivalent normalized input (resample notionally by mapping frequencies)
        // Since our generator depends on Fs explicitly, we regenerate the same composite
        // signal under Fs=2 with the corresponding normalized frequencies.
        let x_norm = {
            let a = sine_norm(center_mhz / nyq, n);
            let b = sine_norm(0.90 * low_mhz / nyq, n);
            let c = sine_norm(0.90 * high_mhz / nyq, n);
            (a + b + c).mapv(|v| (1.0 / 3.0) * v)
        };
        let y_norm = bandpass_constant_peak(&x_norm, low_n, high_n, None, None).unwrap();

        // Compare RMS—should be extremely close
        assert_relative_eq!(rms(&y_abs), rms(&y_norm), max_relative = 0.05);
    }

    // ---------- 7) Cascading sections increases rejection (while center stays 0 dB) ----------
    #[test]
    fn cascade_improves_skirts() {
        let n = 16384;
        let low = 0.25_f32;
        let high = 0.50_f32;
        let f0 = (low * high).sqrt();

        let x_center = sine_norm(f0, n);
        let x_low = sine_norm(0.10, n);

        // 1 section
        let y1_c = bandpass_constant_peak(&x_center, low, high, None, None).unwrap();
        let y1_l = bandpass_constant_peak(&x_low, low, high, None, None).unwrap();

        // 2 sections (run twice)
        let y2_c = {
            let t = bandpass_constant_peak(&x_center, low, high, None, None).unwrap();
            bandpass_constant_peak(&t, low, high, None, None).unwrap()
        };
        let y2_l = {
            let t = bandpass_constant_peak(&x_low, low, high, None, None).unwrap();
            bandpass_constant_peak(&t, low, high, None, None).unwrap()
        };

        // Center remains ~unity
        let g1 = rms(&y1_c) / rms(&x_center);
        let g2 = rms(&y2_c) / rms(&x_center);
        assert_relative_eq!(g1, 1.0, max_relative = 0.15);
        assert_relative_eq!(g2, 1.0, max_relative = 0.15);

        // Out-of-band attenuation improves (more negative dB)
        let att1 = rel_db(rms(&y1_l), rms(&y1_c));
        let att2 = rel_db(rms(&y2_l), rms(&y2_c));
        assert!(
            att2 < att1 - 5.0,
            "cascading did not clearly improve attenuation"
        );
    }

    // ---------- 8) Parameter validation ----------
    #[test]
    fn parameter_validation() {
        let x = Array1::<f32>::zeros(32);
        // Normalized mode: require 0<low<high<1
        assert!(bandpass_constant_peak(&x, 0.0, 0.5, None, None).is_err());
        assert!(bandpass_constant_peak(&x, 0.5, 1.0, None, None).is_err());

        // Absolute mode requires center & fs, and high < Nyquist
        assert!(bandpass_constant_peak(&x, 50.0, 150.0, Some(100.0), None).is_err());
        assert!(bandpass_constant_peak(&x, 50.0, 150.0, Some(40.0), Some(1000.0)).is_err());
        assert!(bandpass_constant_peak(&x, 50.0, 600.0, Some(100.0), Some(1000.0)).is_err());
    }

    // ---------- 9) DC gain ~ 0 via impulse/step reasoning ----------
    // For a band-pass, H(e^{j0}) ≈ 0. The step response should settle near 0.
    #[test]
    fn dc_gain_near_zero_via_step() {
        let n = 16384;
        let y = bandpass_constant_peak(&step(n, 1.0), 0.2, 0.4, None, None).unwrap();
        let tail_rms = {
            let half = n / 2;
            rms(&y.slice(s![half..]).to_owned())
        };
        assert!(
            tail_rms < 1e-3,
            "step steady-state not near zero (rms={tail_rms})"
        );
    }

    // ---------- 10) Finite impulse response energy (sanity) ----------
    #[test]
    fn impulse_finite_energy() {
        let n = 4096;
        let h = bandpass_constant_peak(&impulse(n), 0.2, 0.4, None, None).unwrap();
        let energy: f32 = h.iter().map(|v| v * v).sum();
        assert!(energy.is_finite() && energy > 0.0 && energy < 1000.0);
    }
}
