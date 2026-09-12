//! Typed visualization configuration (#119, first cut).
//!
//! Only settings the v1 renderer actually supports -- no free-form
//! client-defined profiles (#120 explicitly warns against unbounded
//! client-driven render work). Every field here is part of the render
//! variant identity once that lands in M5: changing any of them must
//! produce a new render ID.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResamplingMethod {
    /// Area-weighted mean over source-pixel footprints (#118).
    /// Nearest-neighbor is deliberately excluded per the issue's own
    /// guidance for this data.
    Mean,
    /// Largest source value in each footprint. For a profile that reads
    /// *signed* amplitude asymmetrically (see
    /// [`AmplitudeTransform::Positive`]), the mean is the wrong reducer:
    /// radar traces oscillate about zero, so averaging a large footprint
    /// cancels them toward zero and the asymmetric stretch then clips the
    /// result to black. Identical to `Mean` at a 1:1 footprint. See
    /// `resample::resample` for the full rationale.
    ///
    /// Superseded by [`LanczosRectified`](Self::LanczosRectified) for
    /// `positive` (biases every footprint upward into speckle, which the
    /// rectified windowed-sinc filter does not); not currently used by any
    /// built-in profile, kept as a documented option.
    Peak,
    /// Windowed-sinc (Lanczos-3) filtering, applied separably.
    ///
    /// The principled anti-aliasing choice: `Mean` is a box filter and
    /// blurs, `Peak` is an order statistic that biases every footprint
    /// upward and turns noise into speckle, while this preserves the
    /// shape of the signal around a feature rather than only its average
    /// or its maximum. Also reduces to an identity at an aligned 1:1
    /// footprint, where the kernel taps land on integers and `sinc` is
    /// zero at all of them but the centre.
    ///
    /// Plain (unrectified) Lanczos is still a *linear* filter, so on
    /// signed oscillating amplitude it cancels toward zero exactly the way
    /// `Mean` does -- measured, not assumed: `positive` overviews under
    /// this came out nearly black. Not currently used by any built-in
    /// profile as a result; kept as a documented option for a profile
    /// whose amplitude is not signed and asymmetric.
    Lanczos,
    /// Lanczos applied to `|amplitude|` rather than signed amplitude.
    ///
    /// Removes the cancellation plain `Lanczos` suffers on signed
    /// oscillating traces while keeping proper anti-aliasing, which `Peak`
    /// (an order statistic that biases every footprint upward into
    /// speckle) does not provide. Compared against both on real data (see
    /// `PHASE1_LOG.md`) and read best of the three; used by
    /// [`RenderProfile::positive_profile`] and
    /// [`RenderProfile::abslog_profile`].
    ///
    /// Safe for any profile whose display value is already sign-agnostic
    /// (`positive`'s asymmetric stretch, `abslog`'s `log10|amplitude|`).
    /// For a plain signed-linear profile it would change what the image
    /// means rather than just how it is anti-aliased.
    LanczosRectified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    /// Chosen over WebP for v1: `image` 0.25's WebP encoder
    /// (`image-webp` 0.2.4) is lossless-only in every released version:
    /// its own README states "only supports lossless encoding", and its
    /// `EncoderParams` has no `use_lossy` field. JPEG needs no new
    /// dependency -- `image` 0.24 already encodes it.
    Jpeg {
        quality: u8,
    },
    Png,
}

impl ImageFormat {
    pub fn content_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg { .. } => "image/jpeg",
            ImageFormat::Png => "image/png",
        }
    }
}

/// How raw amplitude is mapped into the domain that limits and
/// normalization operate in, both for percentile estimation (`stats.rs`)
/// and for the pixel value that actually gets normalized (`colormap.rs`).
///
/// `AbsLog` uses the same domain (`log10|x|`) for both purposes. `Positive`
/// does not: PFA_website's `normalize()` estimates percentile bounds from
/// `|x|` but stretches the *signed* value against those bounds, which is
/// what pushes negative returns toward black while favoring positive ones.
/// That asymmetry is why `Positive` needs its own variant rather than
/// reusing a single "domain" transform for both stats and display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmplitudeTransform {
    /// Plain linear amplitude, no transform.
    Linear,
    /// `log10(|amplitude|)`, matching svalbard_radar's "Absolute" display
    /// mode.
    AbsLog,
    /// Percentile bounds from `|amplitude|`, signed amplitude displayed --
    /// PFA_website's (mis-named there) "abslog" stretch.
    Positive,
}

/// How amplitude limits are determined for normalization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AmplitudeLimits {
    /// User-specified; requires no radargram-wide estimate.
    Explicit { min: f32, max: f32 },
    /// Estimated once per revision+profile via sampled percentiles (#119),
    /// then reused for every chunk -- never estimated independently per
    /// chunk, which would make adjacent chunks normalize differently and
    /// produce visible seams.
    Percentile { low: f32, high: f32 },
}

/// Dataset view: the standard (trace, sample) view is the only one
/// implemented in v1. Topographic correction is an explicitly deferred
/// future view (#118).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetView {
    Standard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderProfile {
    pub name: String,
    pub view: DatasetView,
    pub transform: AmplitudeTransform,
    pub limits: AmplitudeLimits,
    pub resampling: ResamplingMethod,
    pub format: ImageFormat,
    /// Post-normalization contrast multiplier: `1.0` (the default profile's
    /// value) leaves the plain min-max stretch untouched.
    pub contrast: f32,
    /// Post-normalization black-level offset, subtracted before the
    /// contrast multiplier: `0.0` (the default profile's value) leaves the
    /// plain min-max stretch untouched.
    pub black_level: f32,
    /// Sample rows dropped from the top of every trace before estimating
    /// percentile limits (`stats.rs`), excluding the direct-wave band from
    /// the estimate. `0` (the default profile's value) samples whole
    /// traces, unchanged from before this field existed.
    pub stats_skip_first_samples: usize,
}

impl RenderProfile {
    /// `default`: 1-99% quantile, linear amplitude.
    pub fn default_profile() -> Self {
        Self {
            name: "default".to_string(),
            view: DatasetView::Standard,
            transform: AmplitudeTransform::Linear,
            limits: AmplitudeLimits::Percentile {
                low: 0.01,
                high: 0.99,
            },
            resampling: ResamplingMethod::Mean,
            format: ImageFormat::Jpeg { quality: 85 },
            contrast: 1.0,
            black_level: 0.0,
            stats_skip_first_samples: 0,
        }
    }

    /// `abslog`: `log10|amplitude|`, same percentile window as default.
    pub fn abslog_profile() -> Self {
        Self {
            name: "abslog".to_string(),
            transform: AmplitudeTransform::AbsLog,
            // Unlike `default`, rectifying before filtering doesn't change
            // what this profile's image means: its display value is
            // already `log10|amplitude|`, sign-agnostic by construction.
            // Compared against Mean/Peak/Lanczos on real data and read
            // best of the four.
            resampling: ResamplingMethod::LanczosRectified,
            ..Self::default_profile()
        }
    }

    /// `positive`: PFA_website's asymmetric stretch (mis-named "abslog"
    /// there, and not a log transform at all -- see `AmplitudeTransform`).
    /// Biases the display heavily toward positive returns, clipping
    /// negative ones toward black; deliberately not the default, useful in
    /// specific cases rather than as a general-purpose view.
    pub fn positive_profile() -> Self {
        Self {
            name: "positive".to_string(),
            transform: AmplitudeTransform::Positive,
            contrast: 0.9,
            black_level: 0.1,
            // Excludes the direct-wave band (the source wavelet's own very
            // high amplitude near the top of the radargram) from the
            // percentile estimate, matching PFA_website's
            // `normalize()`, which skips the first 50 sample rows.
            stats_skip_first_samples: 50,
            // Settled after comparing Peak, Lanczos and LanczosRectified
            // against real data (see PHASE1_LOG.md): plain `Mean` or
            // `Lanczos` cancel signed, oscillating amplitude toward zero
            // over a downsampled footprint, which this profile's black
            // level then clips to black -- the reason `positive` overviews
            // once rendered almost entirely dark. `Peak` fixed that but
            // biases every footprint upward into speckle. `LanczosRectified`
            // (filtering `|amplitude|`) removes the cancellation while
            // keeping proper anti-aliasing, and read best of the three on
            // real data.
            resampling: ResamplingMethod::LanczosRectified,
            ..Self::default_profile()
        }
    }

    /// `high-contrast`: 5-95% quantile, linear amplitude.
    pub fn high_contrast_profile() -> Self {
        Self {
            name: "high-contrast".to_string(),
            limits: AmplitudeLimits::Percentile {
                low: 0.05,
                high: 0.95,
            },
            ..Self::default_profile()
        }
    }

    /// The server-defined profiles offered in v1 (#121's dropdown).
    pub fn built_in_profiles() -> Vec<Self> {
        vec![
            Self::default_profile(),
            Self::positive_profile(),
            Self::abslog_profile(),
            Self::high_contrast_profile(),
        ]
    }

    pub fn by_name(name: &str) -> Option<Self> {
        Self::built_in_profiles()
            .into_iter()
            .find(|p| p.name == name)
    }

    /// Resolve what a user typed after `--render-profile`: either the name
    /// of a built-in profile, or a path to a TOML file describing one.
    ///
    /// The two are told apart by *looking at the filesystem*, not by
    /// pattern-matching the string. A name that happens to contain a dot
    /// or a slash is not automatically a path, and a file called `default`
    /// in the working directory does not shadow the built-in of that name
    /// -- built-ins win, so a project cannot silently change what
    /// `--render-profile default` means by adding a file.
    ///
    /// Anything that is neither is an error naming both possibilities,
    /// since "no such profile" and "no such file" are the same mistake
    /// from the user's side and they will not know which one they made.
    pub fn resolve(spec: &str) -> Result<Self, String> {
        if let Some(profile) = Self::by_name(spec) {
            return Ok(profile);
        }
        let path = std::path::Path::new(spec);
        if path.is_file() {
            return Self::from_toml_file(path);
        }
        let names: Vec<String> = Self::built_in_profiles()
            .into_iter()
            .map(|p| p.name)
            .collect();
        Err(format!(
            "'{spec}' is neither a built-in render profile ({}) nor a readable file.",
            names.join(", ")
        ))
    }

    /// Read a profile from a TOML file.
    ///
    /// Every field is required. Profiles that inherit from a built-in and
    /// override a field or two are the obvious next step and deliberately
    /// not guessed at here -- whether that is a `base = "default"` key, a
    /// separate `--render-profile-override`, or serde defaults changes
    /// what a file means, and getting it wrong later would silently
    /// re-interpret files people had already written.
    pub fn from_toml_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Could not read render profile {}: {e}", path.display()))?;
        toml::from_str(&text)
            .map_err(|e| format!("Could not parse render profile {}: {e}", path.display()))
    }

    /// A stable string identifying everything about this profile that
    /// affects rendered pixels, for folding into the render variant ID
    /// (M5). Deliberately excludes nothing display-affecting and includes
    /// nothing identity-affecting (e.g. no display name).
    pub fn cache_key_fragment(&self) -> String {
        let limits = match self.limits {
            AmplitudeLimits::Explicit { min, max } => format!("explicit:{min}:{max}"),
            AmplitudeLimits::Percentile { low, high } => format!("pct:{low}:{high}"),
        };
        let format = match self.format {
            ImageFormat::Jpeg { quality } => format!("jpeg:{quality}"),
            ImageFormat::Png => "png".to_string(),
        };
        format!(
            "{:?}|{:?}|{}|{:?}|{}|{}|{}|{}",
            self.view,
            self.transform,
            limits,
            self.resampling,
            format,
            self.contrast,
            self.black_level,
            self.stats_skip_first_samples
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_built_in_profile_round_trips_through_toml() {
        // Also the answer to "what does a profile file look like": every
        // built-in serialises to a complete, re-readable one.
        for original in RenderProfile::built_in_profiles() {
            let text = toml::to_string(&original).unwrap();
            let parsed: RenderProfile = toml::from_str(&text).unwrap();
            assert_eq!(parsed, original, "{}:\n{text}", original.name);
        }
    }

    #[test]
    // Changes the process-wide working directory, so it must not overlap
    // any test that reads a relative path. Same hazard as the tests that
    // unset PATH.
    #[serial_test::serial(current_dir)]
    fn resolve_prefers_a_built_in_name_over_a_file_of_that_name() {
        let dir = tempfile::tempdir().unwrap();
        let decoy = dir.path().join("default");
        std::fs::write(&decoy, "name = 'not this one'").unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.path()).unwrap();
        let resolved = RenderProfile::resolve("default");
        std::env::set_current_dir(previous).unwrap();
        // A file cannot silently redefine what a built-in name means.
        assert_eq!(resolved.unwrap().name, "default");
    }

    #[test]
    fn resolve_reads_a_profile_from_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mine.toml");
        let mut source = RenderProfile::default_profile();
        source.name = "mine".to_string();
        source.contrast = 1.75;
        std::fs::write(&path, toml::to_string(&source).unwrap()).unwrap();

        let resolved = RenderProfile::resolve(path.to_str().unwrap()).unwrap();
        assert_eq!(resolved, source);
    }

    #[test]
    fn resolve_says_both_things_it_looked_for() {
        // "No such profile" and "no such file" are the same mistake from
        // the user's side, and they will not know which one they made.
        let error = RenderProfile::resolve("noideawhatthisis").unwrap_err();
        assert!(error.contains("built-in render profile"), "{error}");
        assert!(error.contains("readable file"), "{error}");
        assert!(
            error.contains("abslog"),
            "should list the real ones: {error}"
        );
    }

    #[test]
    fn built_in_profiles_have_distinct_names_and_cache_keys() {
        let profiles = RenderProfile::built_in_profiles();
        let names: std::collections::HashSet<&str> =
            profiles.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names.len(), profiles.len(), "profile names must be unique");

        let keys: std::collections::HashSet<String> =
            profiles.iter().map(|p| p.cache_key_fragment()).collect();
        assert_eq!(
            keys.len(),
            profiles.len(),
            "distinct profiles must produce distinct cache keys"
        );
    }

    #[test]
    fn by_name_finds_built_ins_and_rejects_unknown() {
        assert!(RenderProfile::by_name("default").is_some());
        assert!(RenderProfile::by_name("positive").is_some());
        assert!(RenderProfile::by_name("abslog").is_some());
        assert!(RenderProfile::by_name("high-contrast").is_some());
        assert!(RenderProfile::by_name("nonexistent").is_none());
        // The comparison profiles this settled from no longer exist.
        assert!(RenderProfile::by_name("positive-lanczos").is_none());
        assert!(RenderProfile::by_name("positive-lanczos-rect").is_none());
        assert!(RenderProfile::by_name("default-lanczos").is_none());
        assert!(RenderProfile::by_name("default-lanczos-rect").is_none());
    }

    #[test]
    fn built_in_resampling_methods_are_the_settled_choice() {
        // Pins the outcome of the Peak/Lanczos/LanczosRectified comparison
        // (PHASE1_LOG.md) so a change here is a deliberate edit, not a
        // silent side effect of something else.
        let expected: &[(&str, ResamplingMethod)] = &[
            ("default", ResamplingMethod::Mean),
            ("positive", ResamplingMethod::LanczosRectified),
            ("abslog", ResamplingMethod::LanczosRectified),
            ("high-contrast", ResamplingMethod::Mean),
        ];
        for (name, method) in expected {
            let profile = RenderProfile::by_name(name).unwrap();
            assert_eq!(
                profile.resampling, *method,
                "profile '{name}' resampling method changed"
            );
        }
    }

    #[test]
    fn positive_profile_is_not_the_default() {
        assert_ne!(
            RenderProfile::positive_profile(),
            RenderProfile::default_profile()
        );
        assert_eq!(RenderProfile::default_profile().contrast, 1.0);
        assert_eq!(RenderProfile::default_profile().black_level, 0.0);
        assert_eq!(RenderProfile::default_profile().stats_skip_first_samples, 0);
    }

    #[test]
    fn identical_profiles_produce_identical_cache_keys() {
        let a = RenderProfile::default_profile();
        let b = RenderProfile::default_profile();
        assert_eq!(a.cache_key_fragment(), b.cache_key_fragment());
    }

    #[test]
    fn changing_quality_changes_the_cache_key() {
        let mut a = RenderProfile::default_profile();
        let b = a.clone();
        a.format = ImageFormat::Jpeg { quality: 50 };
        assert_ne!(a.cache_key_fragment(), b.cache_key_fragment());
    }
}
