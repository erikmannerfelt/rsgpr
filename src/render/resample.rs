//! Area-weighted mean resampling from a source window to an output grid
//! (#118).
//!
//! Not nearest-neighbor: the issue is explicit that nearest-neighbor
//! preserves high-frequency noise poorly for this data, so downsampling
//! must integrate over each output pixel's full source footprint. Ignores
//! NaN values, renormalizes weights over the remaining valid contributions,
//! and produces NaN where a footprint contains no valid values -- which
//! also happens to be exactly the right behavior for footprints that fall
//! partially or fully outside the source extent (an edge or out-of-range
//! chunk), since those samples don't exist any more than a NaN does.

use ndarray::{Array2, ArrayView2};

use super::grid::SourceWindow;
use super::profile::ResamplingMethod;

fn overlap_1d(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

/// Resample `source` over `window` (a possibly non-integer, possibly
/// partially out-of-bounds sub-region of `source`) into an
/// `out_height x out_width` grid, using area-weighted mean.
///
/// `source` is the *complete* source array; `window` selects the region of
/// it to read, in source row/col coordinates. This keeps the function
/// usable both against a fully in-memory array and, once the streaming
/// reader lands, against a source-aligned sub-array the caller has already
/// windowed -- the footprint math is identical either way as long as
/// `window` is expressed in the coordinates of whatever `source` covers.
pub fn resample_area_weighted_mean(
    source: ArrayView2<f32>,
    window: &SourceWindow,
    out_width: usize,
    out_height: usize,
) -> Array2<f32> {
    resample(
        source,
        window,
        out_width,
        out_height,
        ResamplingMethod::Mean,
    )
}

/// Resample `source` over `window` into an `out_height x out_width` grid
/// using `method`.
///
/// [`ResamplingMethod::Peak`] exists because the mean is the wrong
/// reducer for a profile that reads *signed* amplitude asymmetrically.
/// Radar traces oscillate about zero, so averaging a footprint of many
/// source samples largely cancels them out; the `positive` profile then
/// clips the collapsed result below its black level and the whole image
/// goes black. That is exactly what happened to `positive` overviews,
/// which downsample ~5x in each axis. Taking the largest value in the
/// footprint instead keeps "the strongest positive return here", which is
/// what that profile is trying to show in the first place. At a 1:1
/// footprint (a single source sample, i.e. the viewer's own chunks when
/// the raster is not downsampled) `Peak` and `Mean` agree exactly, so
/// this changes nothing about the full-resolution view.
pub fn resample(
    source: ArrayView2<f32>,
    window: &SourceWindow,
    out_width: usize,
    out_height: usize,
    method: ResamplingMethod,
) -> Array2<f32> {
    let mut out = Array2::from_elem((out_height, out_width), f32::NAN);
    if out_width == 0 || out_height == 0 {
        return out;
    }

    if matches!(
        method,
        ResamplingMethod::Lanczos | ResamplingMethod::LanczosRectified
    ) {
        // Needs a different neighbourhood entirely -- a kernel radius
        // rather than the box footprint the loop below walks -- and a
        // separable two-pass structure to stay affordable.
        let rectify = method == ResamplingMethod::LanczosRectified;
        return resample_lanczos(source, window, out_width, out_height, rectify);
    }

    let (src_h, src_w) = (source.shape()[0], source.shape()[1]);
    let row_step = (window.row1 - window.row0) / out_height as f64;
    let col_step = (window.col1 - window.col0) / out_width as f64;

    for oy in 0..out_height {
        let r0 = window.row0 + oy as f64 * row_step;
        let r1 = window.row0 + (oy + 1) as f64 * row_step;
        let ir0 = r0.floor().max(0.0) as usize;
        let ir1 = ((r1.ceil().max(0.0)) as usize).min(src_h);
        if ir0 >= ir1 {
            continue;
        }

        for ox in 0..out_width {
            let c0 = window.col0 + ox as f64 * col_step;
            let c1 = window.col0 + (ox + 1) as f64 * col_step;
            let ic0 = c0.floor().max(0.0) as usize;
            let ic1 = ((c1.ceil().max(0.0)) as usize).min(src_w);
            if ic0 >= ic1 {
                continue;
            }

            let mut weighted_sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            let mut peak = f32::NEG_INFINITY;
            let mut any_valid = false;
            for row in ir0..ir1 {
                let row_w = overlap_1d(row as f64, (row + 1) as f64, r0, r1);
                if row_w <= 0.0 {
                    continue;
                }
                for col in ic0..ic1 {
                    let col_w = overlap_1d(col as f64, (col + 1) as f64, c0, c1);
                    if col_w <= 0.0 {
                        continue;
                    }
                    let v = source[[row, col]];
                    if v.is_finite() {
                        // Peak is deliberately unweighted: a footprint's
                        // strongest return is its strongest return
                        // regardless of what fraction of the pixel it
                        // covers. Weighting would scale peaks down near
                        // footprint edges and reintroduce the very
                        // dimming this method exists to avoid.
                        any_valid = true;
                        peak = peak.max(v);
                        let w = row_w * col_w;
                        weighted_sum += v as f64 * w;
                        weight_sum += w;
                    }
                }
            }

            out[[oy, ox]] = match method {
                ResamplingMethod::Mean if weight_sum > 0.0 => (weighted_sum / weight_sum) as f32,
                ResamplingMethod::Peak if any_valid => peak,
                // No valid contributions: stays NaN, the caller's "no
                // data here" signal (rendered as the pad color).
                _ => f32::NAN,
            };
        }
    }

    out
}

/// Lanczos kernel order. `a = 3` is the usual choice: enough lobes to
/// preserve detail, few enough that the ringing the negative lobes cause
/// stays modest.
const LANCZOS_A: f64 = 3.0;

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn lanczos_weight(x: f64) -> f64 {
    if x.abs() >= LANCZOS_A {
        0.0
    } else {
        sinc(x) * sinc(x / LANCZOS_A)
    }
}

/// Kernel taps along one axis for an output sample centred at `center`
/// (in source coordinates), returning the first source index and the
/// weights for the contiguous run from there.
///
/// The kernel is *widened* by the downsampling ratio: filtering has to
/// happen at the output's band limit, not the input's, or the result is
/// aliased regardless of how good the kernel is. At a 1:1 ratio the taps
/// land exactly on integers, where `sinc` is zero everywhere except the
/// centre, so this reduces to an identity — the same graceful-degradation
/// property `Peak` has.
fn lanczos_axis_weights(center: f64, scale: f64, src_len: usize) -> (usize, Vec<f64>) {
    let s = scale.max(1.0);
    let radius = LANCZOS_A * s;
    let lo = (center - radius).floor().max(0.0) as usize;
    let hi = ((center + radius).ceil().max(0.0) as usize).min(src_len);
    if lo >= hi {
        return (0, Vec::new());
    }
    let weights = (lo..hi)
        .map(|i| lanczos_weight((i as f64 + 0.5 - center) / s))
        .collect();
    (lo, weights)
}

/// Separable two-pass Lanczos: rows first into an intermediate, then
/// columns.
///
/// Separability is not an optimisation detail here, it is what makes the
/// filter usable at all. An overview downsamples ~24x, so the kernel
/// spans `2 * 3 * 24` source pixels per axis; applied as a 2-D kernel
/// that is ~21k taps *per output pixel*, or billions of operations for
/// one thumbnail. Two passes make it ~290 taps per pixel instead.
///
/// NaN (the "no valid source data" signal) is excluded from both passes
/// and the weights renormalised over what remains, so a footprint that
/// straddles the edge of the data is not dragged toward zero. Because
/// Lanczos weights are signed, a footprint whose surviving taps nearly
/// cancel would divide by ~0; those produce NaN rather than a wild value.
///
/// `rectify` takes `|v|` before filtering, which is what
/// [`ResamplingMethod::LanczosRectified`] needs: it removes the
/// cancellation that makes a linear filter useless on signed oscillating
/// traces, without giving up proper anti-aliasing. Gated **per axis** on
/// that axis's own step being a real downsample (`> 1.0`), not applied
/// unconditionally: at a 1:1 (or upsampling) step the kernel's only
/// significant contribution is a single source sample (see
/// `lanczos_axis_weights`'s doc), so there is nothing to cancel and
/// rectifying would only destroy that sample's sign for no reason. This
/// is the same graceful-degradation-to-identity property `Peak` and
/// plain `Lanczos` already have; a first version of this rectified
/// everything unconditionally, which silently defeated the `positive`
/// and `abslog` profiles' asymmetric stretch at native resolution -- the
/// display value reaching the colormap was always non-negative,
/// regardless of the source's true sign, even where no averaging was
/// happening at all.
fn resample_lanczos(
    source: ArrayView2<f32>,
    window: &SourceWindow,
    out_width: usize,
    out_height: usize,
    rectify: bool,
) -> Array2<f32> {
    let (src_h, src_w) = (source.shape()[0], source.shape()[1]);
    let row_step = (window.row1 - window.row0) / out_height as f64;
    let col_step = (window.col1 - window.col0) / out_width as f64;
    let rectify_row = rectify && row_step > 1.0;
    let rectify_col = rectify && col_step > 1.0;

    // Pass 1: vertical. Full source width is kept so pass 2 has every
    // column available to filter across.
    let mut vertical = Array2::from_elem((out_height, src_w), f32::NAN);
    for oy in 0..out_height {
        let center = window.row0 + (oy as f64 + 0.5) * row_step;
        let (start, weights) = lanczos_axis_weights(center, row_step, src_h);
        if weights.is_empty() {
            continue;
        }
        for col in 0..src_w {
            let mut acc = 0.0_f64;
            let mut wsum = 0.0_f64;
            for (k, &w) in weights.iter().enumerate() {
                let v = source[[start + k, col]];
                if v.is_finite() {
                    let v = if rectify_row { v.abs() } else { v };
                    acc += v as f64 * w;
                    wsum += w;
                }
            }
            if wsum.abs() > 1e-6 {
                vertical[[oy, col]] = (acc / wsum) as f32;
            }
        }
    }

    // Pass 2: horizontal, over the already row-filtered intermediate.
    // Rectifies independently of pass 1's decision -- idempotent if pass 1
    // already did it (`abs(abs(x)) == abs(x)`), and necessary on its own
    // when only the column axis is genuinely downsampling.
    let mut out = Array2::from_elem((out_height, out_width), f32::NAN);
    for ox in 0..out_width {
        let center = window.col0 + (ox as f64 + 0.5) * col_step;
        let (start, weights) = lanczos_axis_weights(center, col_step, src_w);
        if weights.is_empty() {
            continue;
        }
        for oy in 0..out_height {
            let mut acc = 0.0_f64;
            let mut wsum = 0.0_f64;
            for (k, &w) in weights.iter().enumerate() {
                let v = vertical[[oy, start + k]];
                if v.is_finite() {
                    let v = if rectify_col { v.abs() } else { v };
                    acc += v as f64 * w;
                    wsum += w;
                }
            }
            if wsum.abs() > 1e-6 {
                out[[oy, ox]] = (acc / wsum) as f32;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn full_window(source: &Array2<f32>) -> SourceWindow {
        SourceWindow {
            row0: 0.0,
            row1: source.shape()[0] as f64,
            col0: 0.0,
            col1: source.shape()[1] as f64,
        }
    }

    #[test]
    fn identity_resampling_reproduces_source_exactly() {
        let source = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 3, 2);
        assert_eq!(out, source);
    }

    #[test]
    fn integer_2x2_downsample_averages_exactly() {
        #[rustfmt::skip]
        let source = array![
            [1.0f32, 3.0, 1.0, 3.0],
            [1.0,    3.0, 1.0, 3.0],
            [5.0,    7.0, 5.0, 7.0],
            [5.0,    7.0, 5.0, 7.0],
        ];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 2, 2);
        // Each 2x2 block averages to (1+3+1+3)/4=2 and (5+7+5+7)/4=6.
        assert_eq!(out, array![[2.0f32, 2.0], [6.0, 6.0]]);
    }

    #[test]
    fn nan_pixels_are_excluded_and_weights_renormalized() {
        let source = array![[1.0f32, f32::NAN], [3.0, 5.0]];
        let window = full_window(&source);
        // Downsample the whole 2x2 block to a single pixel: mean of the
        // three valid values (1, 3, 5) = 3, NOT (1+0+3+5)/4.
        let out = resample_area_weighted_mean(source.view(), &window, 1, 1);
        assert!((out[[0, 0]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn footprint_with_no_valid_values_is_nan() {
        let source = array![[f32::NAN, f32::NAN], [f32::NAN, f32::NAN]];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 1, 1);
        assert!(out[[0, 0]].is_nan());
    }

    #[test]
    fn non_integer_ratio_upsamples_with_shared_source_pixel_weight() {
        // 2x2 -> 3x3 is a non-integer ratio (2/3). Every output pixel must
        // still be finite (full coverage, no gaps), and the corners must
        // equal the corresponding exact source corner value, since a
        // corner output pixel's footprint touches only one source pixel
        // heavily.
        let source = array![[1.0f32, 2.0], [3.0, 4.0]];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 3, 3);
        for v in out.iter() {
            assert!(v.is_finite());
        }
        assert!((out[[0, 0]] - 1.0).abs() < 0.3);
        assert!((out[[2, 2]] - 4.0).abs() < 0.3);
    }

    #[test]
    fn window_partially_outside_source_extent_treated_like_nan() {
        let source = array![[1.0f32, 2.0], [3.0, 4.0]];
        // Window extends one row/col beyond the 2x2 source -- as an edge
        // chunk's window would when the raster doesn't divide evenly.
        let window = SourceWindow {
            row0: 0.0,
            row1: 3.0,
            col0: 0.0,
            col1: 3.0,
        };
        let out = resample_area_weighted_mean(source.view(), &window, 3, 3);
        // The bottom-right output pixel's footprint is entirely outside
        // the source (rows/cols 2..3 don't exist) -> NaN.
        assert!(out[[2, 2]].is_nan());
        // The top-left pixel is entirely inside -> finite, close to source[0,0].
        assert!(out[[0, 0]].is_finite());
    }

    #[test]
    fn zero_sized_output_does_not_panic() {
        let source = array![[1.0f32]];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 0, 0);
        assert_eq!(out.shape(), &[0, 0]);
    }

    #[test]
    fn asymmetric_source_is_not_transposed() {
        // 1 row x 3 cols: averaging down to 1x1 must reflect all three
        // columns, not silently read only one row/col due to a swapped
        // index somewhere.
        let source = array![[10.0f32, 20.0, 30.0]];
        let window = full_window(&source);
        let out = resample_area_weighted_mean(source.view(), &window, 1, 1);
        assert!((out[[0, 0]] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn peak_survives_the_cancellation_that_collapses_the_mean() {
        // An oscillating trace, exactly the shape of real radar data: the
        // mean of a large footprint collapses to ~0, which the `positive`
        // profile's black level then clips away entirely. Peak keeps the
        // strongest positive return, which is what that profile is for.
        let source = array![[5.0f32, -5.0, 4.0, -4.0, 6.0, -6.0, 3.0, -3.0]];
        let window = full_window(&source);

        let mean = resample(source.view(), &window, 1, 1, ResamplingMethod::Mean);
        let peak = resample(source.view(), &window, 1, 1, ResamplingMethod::Peak);

        assert!(mean[[0, 0]].abs() < 0.5, "mean collapsed: {}", mean[[0, 0]]);
        assert!((peak[[0, 0]] - 6.0).abs() < 1e-6, "peak: {}", peak[[0, 0]]);
    }

    #[test]
    fn peak_and_mean_agree_at_a_one_to_one_footprint() {
        // The viewer's own chunks resample 1:1 whenever the raster is not
        // downsampled, so switching the `positive` profile to Peak must
        // leave the full-resolution view untouched.
        let source = array![[5.0f32, -5.0, 4.0], [-4.0, 6.0, -6.0]];
        let window = full_window(&source);

        let mean = resample(source.view(), &window, 3, 2, ResamplingMethod::Mean);
        let peak = resample(source.view(), &window, 3, 2, ResamplingMethod::Peak);
        for (m, p) in mean.iter().zip(peak.iter()) {
            assert!((m - p).abs() < 1e-6, "{m} vs {p}");
        }
    }

    #[test]
    fn lanczos_is_an_identity_at_an_aligned_one_to_one_footprint() {
        // Same graceful-degradation property the other methods have, and
        // the reason adding Lanczos cannot disturb the full-resolution
        // viewer: at a 1:1 ratio every kernel tap lands on an integer,
        // where sinc is zero except at the centre.
        let source = array![[5.0f32, -5.0, 4.0], [-4.0, 6.0, -6.0]];
        let window = full_window(&source);

        let out = resample(source.view(), &window, 3, 2, ResamplingMethod::Lanczos);
        for (a, b) in out.iter().zip(source.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn lanczos_rectified_is_an_identity_at_an_aligned_one_to_one_footprint() {
        // Regression guard: a first version of LanczosRectified rectified
        // every sample unconditionally, so this test would have returned
        // |source| here instead of source, silently defeating `positive`
        // and `abslog`'s sign-dependent stretch at native resolution --
        // not just in downsampled overviews.
        let source = array![[5.0f32, -5.0, 4.0], [-4.0, 6.0, -6.0]];
        let window = full_window(&source);

        let out = resample(
            source.view(),
            &window,
            3,
            2,
            ResamplingMethod::LanczosRectified,
        );
        for (a, b) in out.iter().zip(source.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b} (sign not preserved)");
        }
    }

    #[test]
    fn lanczos_rectified_avoids_cancellation_when_columns_downsample() {
        // The column axis genuinely downsamples (8 -> 1); rectification
        // must still kick in there even though nothing about the row axis
        // does, since a profile's resampling choice applies uniformly.
        let source = array![[5.0f32, -5.0, 4.0, -4.0, 6.0, -6.0, 3.0, -3.0]];
        let window = full_window(&source);

        let plain = resample(source.view(), &window, 1, 1, ResamplingMethod::Lanczos);
        let rectified = resample(
            source.view(),
            &window,
            1,
            1,
            ResamplingMethod::LanczosRectified,
        );
        assert!(
            plain[[0, 0]].abs() < 1.0,
            "plain Lanczos should cancel toward zero: {}",
            plain[[0, 0]]
        );
        assert!(
            rectified[[0, 0]] > 3.0,
            "rectified should retain real magnitude: {}",
            rectified[[0, 0]]
        );
    }

    #[test]
    fn lanczos_rectified_avoids_cancellation_when_rows_downsample() {
        // Same as the column case, transposed: the row axis's own gate
        // must independently trigger when it is the one downsampling.
        let source =
            Array2::from_shape_vec((8, 1), vec![5.0f32, -5.0, 4.0, -4.0, 6.0, -6.0, 3.0, -3.0])
                .unwrap();
        let window = full_window(&source);

        let plain = resample(source.view(), &window, 1, 1, ResamplingMethod::Lanczos);
        let rectified = resample(
            source.view(),
            &window,
            1,
            1,
            ResamplingMethod::LanczosRectified,
        );
        assert!(
            plain[[0, 0]].abs() < 1.0,
            "plain Lanczos should cancel toward zero: {}",
            plain[[0, 0]]
        );
        assert!(
            rectified[[0, 0]] > 3.0,
            "rectified should retain real magnitude: {}",
            rectified[[0, 0]]
        );
    }

    #[test]
    fn lanczos_rectified_preserves_sign_on_the_identity_axis_while_the_other_downsamples() {
        // Anisotropic case: rows stay 1:1 (must keep their sign exactly)
        // while columns downsample 8x on one of the two rows (must
        // rectify there). Catches a gating bug that ties both axes'
        // decisions together instead of applying each independently.
        #[rustfmt::skip]
        let source = array![
            [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [5.0,   -5.0, 4.0, -4.0, 6.0, -6.0, 3.0, -3.0],
        ];
        let window = full_window(&source);

        let out = resample(
            source.view(),
            &window,
            1,
            2,
            ResamplingMethod::LanczosRectified,
        );
        // Row 0 is constant, so both a rectified and unrectified column
        // pass agree here; it isn't the discriminating row. Assert
        // row 1's genuinely oscillating columns still rectified.
        assert!(
            out[[1, 0]] > 3.0,
            "row 1 should rectify away its column cancellation: {}",
            out[[1, 0]]
        );
        assert!(
            (out[[0, 0]] - 1.0).abs() < 1e-5,
            "row 0 should be untouched: {}",
            out[[0, 0]]
        );
    }

    #[test]
    fn lanczos_downsamples_a_smooth_ramp_without_distorting_it() {
        // A linear ramp is where a correct filter should be essentially
        // exact: an output pixel takes the ramp's value at its centre.
        // Catches sign errors, off-by-one tap placement and missing
        // weight normalisation at once.
        //
        // Asserted only where the kernel fits entirely inside the source.
        // There is no edge extension, so near the borders the kernel is
        // truncated and renormalised over the surviving (asymmetric) taps,
        // which is approximate by construction -- measured at ~0.2% of
        // range on this ramp, small but not exact. Testing the interior
        // keeps the assertion tight enough to be worth something.
        let width = 64;
        let out_width = 32;
        let scale = width as f64 / out_width as f64;
        let radius = LANCZOS_A * scale;
        let source = Array2::from_shape_fn((1, width), |(_, c)| c as f32);
        let window = full_window(&source);

        let out = resample(
            source.view(),
            &window,
            out_width,
            1,
            ResamplingMethod::Lanczos,
        );
        let mut checked = 0;
        for ox in 0..out_width {
            let center = (ox as f64 + 0.5) * scale;
            if center - radius < 0.0 || center + radius > width as f64 {
                continue;
            }
            // Source pixel i is centred at i + 0.5 and holds value i, so
            // the ramp's value at source coordinate x is x - 0.5.
            let expected = (center - 0.5) as f32;
            let got = out[[0, ox]];
            assert!(
                (got - expected).abs() < 0.01,
                "output {ox}: got {got}, expected about {expected}"
            );
            checked += 1;
        }
        assert!(checked > 10, "only {checked} interior pixels were checked");
    }

    #[test]
    fn lanczos_excludes_nan_and_renormalises() {
        // Same contract as the other methods: NaN means "no data here",
        // not "zero", so it must not drag the result toward zero.
        let source = array![[2.0f32, f32::NAN, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]];
        let window = full_window(&source);
        let out = resample(source.view(), &window, 2, 1, ResamplingMethod::Lanczos);
        for v in out.iter() {
            assert!((v - 2.0).abs() < 0.2, "constant input became {v}");
        }
    }

    #[test]
    fn lanczos_reports_no_data_for_an_all_nan_footprint() {
        let source = array![[f32::NAN, f32::NAN, f32::NAN, f32::NAN]];
        let window = full_window(&source);
        let out = resample(source.view(), &window, 1, 1, ResamplingMethod::Lanczos);
        assert!(out[[0, 0]].is_nan());
    }

    #[test]
    fn peak_reports_no_data_for_an_all_nan_footprint() {
        let source = array![[f32::NAN, f32::NAN]];
        let window = full_window(&source);
        let out = resample(source.view(), &window, 1, 1, ResamplingMethod::Peak);
        assert!(out[[0, 0]].is_nan());
    }
}
