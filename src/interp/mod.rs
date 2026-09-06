//! Interpretation support: reading level 1 documents and deriving level 2
//! products from them.
//!
//! Two levels, with one source of truth:
//!
//! - **Level 1** is a [gprinterp] document. It stores picked features in the
//!   trace/sample index space of one processed radargram, plus the axis
//!   metadata needed to move them onto a different processing revision. It
//!   is what gets edited and persisted, and it is the only thing that is
//!   authoritative.
//! - **Level 2** is a derived convenience product: geographic points with
//!   depth, travel time and provenance attached, for loading into a GIS or a
//!   analysis script. It is regenerated from level 1 on demand and is never
//!   edited in place.
//!
//! The asymmetry matters. Level 2 embeds values (depth, position) that
//! depend on how the radargram was processed, so a level 2 export is only
//! meaningful alongside the radargram and revision it names. Level 1 stays
//! valid across reprocessing, which is the entire reason it exists.
//!
//! [gprinterp]: https://github.com/erikmannerfelt/gprinterp

pub mod checks;
pub mod level2;
pub mod source;
pub mod writer;
