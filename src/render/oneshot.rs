//! Render a whole processed radargram to one image file.
//!
//! The command-line counterpart to the web GUI's image download, and what
//! `ridal process --render` produces. Deliberately *not* a second renderer:
//! it resolves a profile, estimates amplitude limits, and calls the same
//! [`crate::render::renderer::Renderer`] the server does, so a picture
//! drawn here and one drawn in the browser are the same picture.
//!
//! Ridal used to have two: `io::render_jpg` stretched between the 1st and
//! 99th percentile of every tenth sample with a special case for `unphase`,
//! while the server applied a render profile. Two implementations of "draw
//! a radargram" that disagree is a bad property for an instrument display.

use std::path::Path;

use crate::render::grid::OverviewSpec;
use crate::render::profile::{AmplitudeLimits, ImageFormat, RenderProfile};
use crate::render::{colormap, renderer::Renderer, stats};
use crate::source::{AmplitudeSource, SourceReader};

/// Fixed so the same file rendered twice gives byte-identical output.
///
/// Amplitude limits come from a sampled subset of traces, so an arbitrary
/// seed would make the stretch -- and therefore every pixel -- vary run to
/// run. The server folds the render variant into its seed to the same end.
const SAMPLE_SEED: u64 = 0x5249_4441_4c00_0001;

/// What to draw and how, resolved from CLI arguments.
pub struct RenderRequest<'a> {
    pub profile: &'a RenderProfile,
    /// Output width in pixels. `None` means one pixel per trace.
    ///
    /// Never upsamples: a width beyond the trace count is clamped, because
    /// a wider image than the data cannot show more than the data.
    pub width: Option<usize>,
    /// JPEG quality override. Ignored when the output is PNG.
    pub quality: Option<u8>,
}

/// Pick the encoding from the output path's extension, falling back to the
/// profile's own format when there is nothing to go on.
///
/// The extension wins because it is the more specific instruction: someone
/// who wrote `out.png` asked for a PNG regardless of what the profile
/// prefers, and silently writing JPEG bytes to a `.png` file would produce
/// something most tools open and some reject.
fn format_for(path: &Path, profile: &RenderProfile, quality: Option<u8>) -> ImageFormat {
    let default_quality = match profile.format {
        ImageFormat::Jpeg { quality } => quality,
        ImageFormat::Png => 85,
    };
    let quality = quality.unwrap_or(default_quality);
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => ImageFormat::Png,
        Some("jpg") | Some("jpeg") => ImageFormat::Jpeg { quality },
        _ => match profile.format {
            ImageFormat::Png => ImageFormat::Png,
            ImageFormat::Jpeg { .. } => ImageFormat::Jpeg { quality },
        },
    }
}

/// JPEG cannot address more than this in either dimension. PNG can, so the
/// limit is checked against the chosen encoding rather than applied to
/// every render -- a 90000-trace radargram is a perfectly good PNG.
const MAX_JPEG_DIMENSION: usize = 65535;

/// Render a processed `.nc` to `output`, returning the image's dimensions.
///
/// The `ridal render` path. `process --render` uses [`render_to_file`]
/// directly with the array it already has in memory.
pub fn render_path_to_file(
    input: &Path,
    output: &Path,
    request: &RenderRequest,
) -> Result<(usize, usize), String> {
    let reader = SourceReader::open(input)?;
    render_to_file(&reader, output, request)
}

/// Render any [`AmplitudeSource`] to `output`, returning the image's
/// dimensions.
pub fn render_to_file(
    source: &impl AmplitudeSource,
    output: &Path,
    request: &RenderRequest,
) -> Result<(usize, usize), String> {
    let (source_height, source_width) = source.shape();

    let width = request
        .width
        .unwrap_or(source_width)
        .clamp(1, source_width.max(1));
    let spec = OverviewSpec::new(source_width, source_height, width);

    let format = format_for(output, request.profile, request.quality);
    if let ImageFormat::Jpeg { .. } = format {
        for (axis, value) in [("wide", spec.width), ("tall", spec.height)] {
            if value > MAX_JPEG_DIMENSION {
                return Err(format!(
                    "Image too {axis} for JPEG ({value} px, max {MAX_JPEG_DIMENSION}). \
                     Write a .png instead, or pass --width."
                ));
            }
        }
    }

    // Estimated once for the whole image, exactly as the server does per
    // revision+profile. An explicit-limits profile skips the sampling pass
    // entirely rather than estimating and discarding.
    let sampled = match request.profile.limits {
        AmplitudeLimits::Percentile { low, high } => Some(stats::sampled_amplitude_limits(
            source,
            request.profile.transform,
            SAMPLE_SEED,
            low,
            high,
            request.profile.stats_skip_first_samples,
        )?),
        AmplitudeLimits::Explicit { .. } => None,
    };
    let limits = colormap::resolve_limits(&request.profile.limits, sampled)?;

    // The profile carries a format of its own, which `format_for` may have
    // overridden from the output extension. The renderer must be told the
    // resolved one, not the profile's.
    let profile = RenderProfile {
        format,
        ..request.profile.clone()
    };
    let bytes = Renderer::new(source).render_overview(&spec, &profile, limits)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Could not create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(output, &bytes)
        .map_err(|e| format!("Could not write {}: {e}", output.display()))?;
    Ok((spec.width, spec.height))
}

/// Default output path for an input, when none was given: the input with
/// the profile's own extension.
pub fn sidecar_path(input: &Path, profile: &RenderProfile) -> std::path::PathBuf {
    let extension = match profile.format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg { .. } => "jpg",
    };
    input.with_extension(extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> RenderProfile {
        RenderProfile::default_profile()
    }

    #[test]
    fn the_output_extension_decides_the_encoding() {
        // Someone who wrote `out.png` asked for a PNG, whatever the
        // profile's own format says.
        assert_eq!(
            format_for(Path::new("out.png"), &profile(), None),
            ImageFormat::Png
        );
        assert!(matches!(
            format_for(Path::new("out.jpg"), &profile(), None),
            ImageFormat::Jpeg { .. }
        ));
        assert!(matches!(
            format_for(Path::new("OUT.JPEG"), &profile(), None),
            ImageFormat::Jpeg { .. }
        ));
    }

    #[test]
    fn an_unknown_extension_falls_back_to_the_profile() {
        let mut p = profile();
        p.format = ImageFormat::Png;
        assert_eq!(format_for(Path::new("out.dat"), &p, None), ImageFormat::Png);
        assert_eq!(format_for(Path::new("out"), &p, None), ImageFormat::Png);
    }

    #[test]
    fn quality_overrides_the_profiles_own() {
        let mut p = profile();
        p.format = ImageFormat::Jpeg { quality: 70 };
        assert_eq!(
            format_for(Path::new("out.jpg"), &p, Some(95)),
            ImageFormat::Jpeg { quality: 95 }
        );
        // And without an override, the profile's value survives.
        assert_eq!(
            format_for(Path::new("out.jpg"), &p, None),
            ImageFormat::Jpeg { quality: 70 }
        );
    }

    #[test]
    fn the_sidecar_follows_the_profiles_format() {
        let mut p = profile();
        p.format = ImageFormat::Png;
        assert_eq!(
            sidecar_path(Path::new("/data/line.nc"), &p),
            Path::new("/data/line.png")
        );
        p.format = ImageFormat::Jpeg { quality: 85 };
        assert_eq!(
            sidecar_path(Path::new("/data/line.nc"), &p),
            Path::new("/data/line.jpg")
        );
    }
}
