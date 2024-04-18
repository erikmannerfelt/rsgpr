//! # rsgpr  --- "rust GPR" or "really simple GPR" or "`rsgpr` simplifies GPR"
//! A Ground Penetrating Radar (GPR) processing tool written in rust.
//!
//! **This is a WIP and may still have significant bugs.**
//!
//! The main aims of `rsgpr` are:
//! - **Ease of use**: A command line interface to process data or batches of data in one command.
//! - **Transparency**: All code is (or will be) thoroughly documented to show exactly how the data are modified.
//! - **Low memory usage and high speed**: While data are processed in-memory, they are usually no larger than an image (say 4000x2000 px). The functions of `rsgpr` avoid copying as much as possible, to keep memory usage to a minimum. Wherever possible, it is also multithreaded for fast processing times.
//! - **Reliability**: All functions will be tested in CI, meaning no crash or invalid behaviour should occur.
//!
//! Thank you, creators of [rgpr](https://github.com/emanuelhuber/RGPR) for the inspiration for this project.
use clap::Parser;

mod cli;
mod dem;
mod filters;
mod gpr;
mod io;
mod tools;

const PROGRAM_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");
const PROGRAM_AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

fn main() {
    let args = cli::Args::parse();
    let return_code = cli::main(args);

    std::process::exit(return_code);
}
