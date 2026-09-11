//! Render identities, byte-bounded cache, and the service that ties
//! `SourceReader` + `Renderer` + cache together (#119).
//!
//! Required behavior, in order: construct the render-object key -> look
//! for an encoded result in the in-memory cache -> generate it on a miss
//! -> insert -> return. Amplitude limits are resolved once per
//! `RenderVariantId` and cached separately from the encoded images
//! themselves, which is what keeps adjacent chunks normalized identically
//! (the seam problem M4's tests already guard against).

// Cache introspection (RenderService::cache_len/cache_bytes,
// ByteBoundedCache::len/current_bytes, RenderObjectKey::as_str) exists for
// API completeness and for eventual cache metrics, and is currently only
// reached from tests. This allowance came with the file when it moved out
// of `render/mod.rs`, which carried it for the same reason.
#![allow(dead_code)]

use std::collections::HashMap;

use crate::render::colormap;
use crate::render::grid::{Chunk, OverviewSpec, CHUNK_SIZE};
use crate::render::profile::{AmplitudeLimits, DatasetView, RenderProfile};
use crate::render::renderer::Renderer;
use crate::render::stats::sampled_amplitude_limits;
use crate::server::catalog::RevisionId;
use crate::source::SourceReader;

/// Bumped whenever a change to the resampling implementation would change
/// rendered pixels for existing content, so cached renders from a previous
/// version become unreachable rather than silently stale.
const RESAMPLER_VERSION: u32 = 1;
/// Bumped whenever a change to the renderer/encoder pipeline would change
/// rendered pixels or bytes for existing content.
const RENDERER_VERSION: u32 = 1;

fn blake3_hex32(parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    for p in parts {
        hasher.update(p);
    }
    // First 16 bytes (32 hex chars): matches RevisionId's precedent in
    // catalog.rs -- collision-resistant among one user's radargrams and
    // profiles, not cryptographically unforgeable, and shorter than the
    // full 32-byte digest in already-long chunk/overview URLs.
    hasher.finalize().to_hex()[..32].to_string()
}

/// Identifies the selected dataset view and render profile for one
/// revision: everything that affects rendered pixels except which
/// specific chunk or overview is being requested.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderVariantId(String);

impl RenderVariantId {
    pub fn compute(revision_id: &RevisionId, view: DatasetView, profile: &RenderProfile) -> Self {
        Self(blake3_hex32(&[
            b"ridal-render-variant-v1",
            revision_id.as_str().as_bytes(),
            format!("{view:?}").as_bytes(),
            profile.cache_key_fragment().as_bytes(),
            &RESAMPLER_VERSION.to_le_bytes(),
            &RENDERER_VERSION.to_le_bytes(),
        ]))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identifies one concrete overview or image chunk within a render
/// variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RenderObjectDescriptor {
    Chunk { x: usize, y: usize, size: usize },
    Overview { width: usize, height: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderObjectKey(String);

impl RenderObjectKey {
    pub fn compute(variant: &RenderVariantId, descriptor: &RenderObjectDescriptor) -> Self {
        let desc = match descriptor {
            RenderObjectDescriptor::Chunk { x, y, size } => format!("chunk:{x}:{y}:{size}"),
            RenderObjectDescriptor::Overview { width, height } => {
                format!("overview:{width}:{height}")
            }
        };
        Self(blake3_hex32(&[
            variant.as_str().as_bytes(),
            desc.as_bytes(),
        ]))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// An in-memory LRU bounded by total encoded byte size, not item count
/// (#119: "bound the cache by encoded byte size rather than item count").
pub struct ByteBoundedCache {
    cache: lru::LruCache<RenderObjectKey, Vec<u8>>,
    current_bytes: usize,
    max_bytes: usize,
}

impl ByteBoundedCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            cache: lru::LruCache::unbounded(),
            current_bytes: 0,
            max_bytes,
        }
    }

    pub fn get(&mut self, key: &RenderObjectKey) -> Option<Vec<u8>> {
        self.cache.get(key).cloned()
    }

    /// Insert `value`, evicting least-recently-used entries until the
    /// cache is back under `max_bytes`. A single entry larger than the
    /// entire budget is still inserted (nothing else to evict) rather than
    /// silently refused -- correctness over strict enforcement, matching
    /// "cache write failures should not turn a successful render into an
    /// HTTP failure" in spirit (there's nothing to fail here yet, since
    /// this is memory-only; the same posture carries over once a fallible
    /// disk tier is added).
    pub fn insert(&mut self, key: RenderObjectKey, value: Vec<u8>) {
        let size = value.len();
        if let Some(old) = self.cache.put(key.clone(), value) {
            self.current_bytes -= old.len();
        }
        self.current_bytes += size;
        while self.current_bytes > self.max_bytes {
            match self.cache.peek_lru() {
                Some((lru_key, _)) if lru_key == &key && self.cache.len() == 1 => break,
                _ => {}
            }
            match self.cache.pop_lru() {
                Some((_, evicted)) => self.current_bytes -= evicted.len(),
                None => break,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }
}

/// Configuration the CLI (`ridal gui` / `ridal server start`) parses its
/// `--cache-memory-mb` / `--n-workers` flags into. Defined here, alongside
/// the service it configures, rather than in `cli.rs`, since the CLI only
/// needs to parse and pass these through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderServiceConfig {
    pub cache_memory_mb: usize,
    /// Reserved for `source.rs`'s deferred HDF5-chunk-aligned read cache;
    /// unused until that lands, and **not currently exposed as a CLI flag
    /// at all** -- always its `Default` value. Do not describe this as an
    /// inert `--source-cache-mb` flag: no such flag is parsed, so passing
    /// one is a hard CLI error, not a silent no-op.
    pub source_cache_mb: usize,
    pub n_workers: usize,
}

impl Default for RenderServiceConfig {
    fn default() -> Self {
        Self {
            cache_memory_mb: 256,
            source_cache_mb: 256,
            n_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

/// Coordinates render-object key construction, cache lookup, rendering on
/// a miss, and cache insertion for one revision's `data` variable.
///
/// Concurrency policy (the read/render split from the plan): this type's
/// own methods take `&mut self`, so callers serialize access naturally --
/// which is exactly right for the read side, since netcdf-c is not
/// thread-safe for concurrent access (confirmed the hard way in M2/M3).
/// Bounding *rendering* concurrency across multiple `RenderService`
/// instances (one per open radargram) via `--n-workers` and dispatching
/// CPU-heavy work off the async executor via `spawn_blocking` is the HTTP
/// layer's job (M6), where concurrent callers first exist; nothing here
/// should be read as already implementing that.
pub struct RenderService {
    reader: SourceReader,
    revision_id: RevisionId,
    cache: ByteBoundedCache,
    limits_cache: HashMap<RenderVariantId, (f32, f32)>,
}

impl RenderService {
    pub fn new(
        reader: SourceReader,
        revision_id: RevisionId,
        config: &RenderServiceConfig,
    ) -> Self {
        Self {
            reader,
            revision_id,
            cache: ByteBoundedCache::new(config.cache_memory_mb * 1024 * 1024),
            limits_cache: HashMap::new(),
        }
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache.current_bytes()
    }

    /// Resolve a variant's amplitude limits, computing and caching them on
    /// first use. Never recomputed per chunk (#119) -- every chunk and the
    /// overview for one variant share the same call's result via
    /// `limits_cache`.
    fn resolve_limits(
        &mut self,
        variant: &RenderVariantId,
        profile: &RenderProfile,
    ) -> Result<(f32, f32), String> {
        if let Some(&limits) = self.limits_cache.get(variant) {
            return Ok(limits);
        }
        let sampled = match profile.limits {
            AmplitudeLimits::Percentile { low, high } => {
                let seed = seed_from_variant(variant);
                Some(sampled_amplitude_limits(
                    &self.reader,
                    profile.transform,
                    seed,
                    low,
                    high,
                    profile.stats_skip_first_samples,
                )?)
            }
            AmplitudeLimits::Explicit { .. } => None,
        };
        let limits = colormap::resolve_limits(&profile.limits, sampled)?;
        self.limits_cache.insert(variant.clone(), limits);
        Ok(limits)
    }

    pub fn get_or_render_chunk(
        &mut self,
        chunk: &Chunk,
        view: DatasetView,
        profile: &RenderProfile,
    ) -> Result<Vec<u8>, String> {
        let variant = RenderVariantId::compute(&self.revision_id, view, profile);
        let key = RenderObjectKey::compute(
            &variant,
            &RenderObjectDescriptor::Chunk {
                x: chunk.x,
                y: chunk.y,
                size: CHUNK_SIZE,
            },
        );
        if let Some(bytes) = self.cache.get(&key) {
            return Ok(bytes);
        }
        let limits = self.resolve_limits(&variant, profile)?;
        let bytes = Renderer::new(&self.reader).render_chunk(chunk, profile, limits)?;
        self.cache.insert(key, bytes.clone());
        Ok(bytes)
    }

    pub fn get_or_render_overview(
        &mut self,
        spec: &OverviewSpec,
        view: DatasetView,
        profile: &RenderProfile,
    ) -> Result<Vec<u8>, String> {
        let variant = RenderVariantId::compute(&self.revision_id, view, profile);
        let key = RenderObjectKey::compute(
            &variant,
            &RenderObjectDescriptor::Overview {
                width: spec.width,
                height: spec.height,
            },
        );
        if let Some(bytes) = self.cache.get(&key) {
            return Ok(bytes);
        }
        let limits = self.resolve_limits(&variant, profile)?;
        let bytes = Renderer::new(&self.reader).render_overview(spec, profile, limits)?;
        self.cache.insert(key, bytes.clone());
        Ok(bytes)
    }
}

/// Derive a sampling seed from a variant ID rather than the clock, so
/// amplitude-limit sampling is reproducible across restarts and identical
/// between repeated calls for the same variant (#119).
fn seed_from_variant(variant: &RenderVariantId) -> u64 {
    u64::from_str_radix(&variant.as_str()[..16], 16).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::RadargramId;
    use crate::render::grid::ViewerRaster;

    fn write_test_nc(path: &std::path::Path, height: usize, width: usize) {
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("y", height).unwrap();
        file.add_dimension("x", width).unwrap();
        let mut var = file.add_variable::<f32>("data", &["y", "x"]).unwrap();
        let data: Vec<f32> = (0..(height * width)).map(|i| (i % 1000) as f32).collect();
        var.put_values(&data, ..).unwrap();
    }

    fn test_revision_id() -> RevisionId {
        RevisionId::fingerprint_v1(
            &RadargramId::new("test-service").unwrap(),
            "2020-01-01T00:00:00Z",
        )
    }

    #[test]
    fn variant_id_is_deterministic_and_changes_with_inputs() {
        let rev_a = test_revision_id();
        let rev_b =
            RevisionId::fingerprint_v1(&RadargramId::new("other").unwrap(), "2020-01-01T00:00:00Z");
        let profile = RenderProfile::default_profile();

        let a1 = RenderVariantId::compute(&rev_a, DatasetView::Standard, &profile);
        let a2 = RenderVariantId::compute(&rev_a, DatasetView::Standard, &profile);
        assert_eq!(a1, a2);

        let b = RenderVariantId::compute(&rev_b, DatasetView::Standard, &profile);
        assert_ne!(a1, b, "different revision must produce a different variant");

        let other_profile = RenderProfile::abslog_profile();
        let c = RenderVariantId::compute(&rev_a, DatasetView::Standard, &other_profile);
        assert_ne!(a1, c, "different profile must produce a different variant");
    }

    #[test]
    fn object_key_distinguishes_chunks_and_overviews() {
        let variant = RenderVariantId::compute(
            &test_revision_id(),
            DatasetView::Standard,
            &RenderProfile::default_profile(),
        );
        let chunk_key = RenderObjectKey::compute(
            &variant,
            &RenderObjectDescriptor::Chunk {
                x: 0,
                y: 0,
                size: 256,
            },
        );
        let other_chunk_key = RenderObjectKey::compute(
            &variant,
            &RenderObjectDescriptor::Chunk {
                x: 1,
                y: 0,
                size: 256,
            },
        );
        let overview_key = RenderObjectKey::compute(
            &variant,
            &RenderObjectDescriptor::Overview {
                width: 512,
                height: 400,
            },
        );
        assert_ne!(chunk_key, other_chunk_key);
        assert_ne!(chunk_key, overview_key);
    }

    #[test]
    fn byte_bounded_cache_evicts_lru_when_over_budget() {
        let mut cache = ByteBoundedCache::new(10);
        let k = |s: &str| RenderObjectKey(s.to_string());
        cache.insert(k("a"), vec![0u8; 4]);
        cache.insert(k("b"), vec![0u8; 4]);
        assert_eq!(cache.current_bytes(), 8);
        // Touch "a" so "b" becomes the least-recently-used.
        assert!(cache.get(&k("a")).is_some());
        cache.insert(k("c"), vec![0u8; 4]); // now 12 bytes, over budget of 10
        assert!(cache.current_bytes() <= 10);
        assert!(
            cache.get(&k("b")).is_none(),
            "b should have been evicted, not a"
        );
        assert!(cache.get(&k("a")).is_some());
    }

    #[test]
    fn byte_bounded_cache_bounds_by_bytes_not_item_count() {
        let mut cache = ByteBoundedCache::new(1000);
        for i in 0..50 {
            cache.insert(RenderObjectKey(format!("k{i}")), vec![0u8; 5]);
        }
        // 50 * 5 = 250 bytes, well under the 1000-byte budget: nothing
        // evicted despite 50 items existing.
        assert_eq!(cache.len(), 50);
        assert_eq!(cache.current_bytes(), 250);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn service_caches_chunk_renders_across_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 300, 300);

        let reader = SourceReader::open(&path).unwrap();
        let config = RenderServiceConfig::default();
        let mut service = RenderService::new(reader, test_revision_id(), &config);

        let raster = ViewerRaster::new(300, 300);
        let grid = raster.grid();
        let chunk = grid.chunk(0, 0).unwrap();
        let profile = RenderProfile::default_profile();

        assert_eq!(service.cache_len(), 0);
        let first = service
            .get_or_render_chunk(&chunk, DatasetView::Standard, &profile)
            .unwrap();
        assert_eq!(service.cache_len(), 1);

        let second = service
            .get_or_render_chunk(&chunk, DatasetView::Standard, &profile)
            .unwrap();
        assert_eq!(
            service.cache_len(),
            1,
            "second call must be a cache hit, not a new entry"
        );
        assert_eq!(first, second);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn service_reuses_sampled_limits_across_chunks_in_one_variant() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 300, 600);

        let reader = SourceReader::open(&path).unwrap();
        let config = RenderServiceConfig::default();
        let mut service = RenderService::new(reader, test_revision_id(), &config);
        let raster = ViewerRaster::new(600, 300);
        let grid = raster.grid();
        let profile = RenderProfile::default_profile(); // percentile limits -> must be sampled

        service
            .get_or_render_chunk(&grid.chunk(0, 0).unwrap(), DatasetView::Standard, &profile)
            .unwrap();
        assert_eq!(service.limits_cache.len(), 1);

        service
            .get_or_render_chunk(&grid.chunk(1, 0).unwrap(), DatasetView::Standard, &profile)
            .unwrap();
        // A second chunk under the SAME variant must not add a second
        // limits entry -- it reuses the one computed for chunk (0,0).
        assert_eq!(service.limits_cache.len(), 1);
    }

    #[test]
    #[test_retry::retry]
    #[serial_test::serial(netcdf)]
    fn service_gives_distinct_cache_entries_per_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.nc");
        write_test_nc(&path, 300, 300);

        let reader = SourceReader::open(&path).unwrap();
        let config = RenderServiceConfig::default();
        let mut service = RenderService::new(reader, test_revision_id(), &config);
        let raster = ViewerRaster::new(300, 300);
        let grid = raster.grid();
        let chunk = grid.chunk(0, 0).unwrap();

        service
            .get_or_render_chunk(
                &chunk,
                DatasetView::Standard,
                &RenderProfile::default_profile(),
            )
            .unwrap();
        service
            .get_or_render_chunk(
                &chunk,
                DatasetView::Standard,
                &RenderProfile::abslog_profile(),
            )
            .unwrap();
        assert_eq!(service.cache_len(), 2);
    }
}
