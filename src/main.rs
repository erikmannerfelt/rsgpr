use clap::Parser;

mod dem;
mod io;
mod tools;
mod gpr;
mod cli;

const PROGRAM_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROGRAM_NAME: &str = env!("CARGO_PKG_NAME");
const PROGRAM_AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

fn main() {

    let args = cli::Args::parse();
    let return_code = cli::main(args);

    std::process::exit(return_code);

}





