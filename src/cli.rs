use crate::{gpr, tools};
/// Functions to handle the command line interface (CLI)
use clap::Parser;
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(group(
        clap::ArgGroup::new("step_choice")
        .required(false)
        .args(&["steps", "default", "default_with_topo"]),
    ))
]
#[clap(group(
        clap::ArgGroup::new("exit_choice")
        .required(false)
        .args(&["show_default", "info", "show_all_steps", "output"]),
    ))
]
pub struct Args {
    /// Filepath of the Malå rd3 file or a glob pattern of many files
    #[clap(short, long)]
    filepath: Option<String>,

    /// Velocity of the medium in m/ns. Defaults to the typical velocity of ice.
    #[clap(short, long, default_value = "0.168")]
    velocity: f32,

    /// Only show metadata for the file
    #[clap(short, long)]
    info: bool,

    /// Load a separate ".cor" file. If not given, it will be searched for automatically
    #[clap(short, long)]
    cor: Option<PathBuf>,

    /// Correct elevation values with a DEM
    #[clap(short, long)]
    dem: Option<PathBuf>,

    /// Which coordinate reference system to project coordinates in.
    #[clap(long, default_value = "EPSG:32633")]
    crs: String,

    /// Export the location track to a comma separated values (CSV) file. Defaults to the output filename location and stem +
    /// "_track.csv"
    #[clap(short, long)]
    track: Option<Option<PathBuf>>,

    /// Process with the default profile. See "--show-default" to list the profile.
    #[clap(long)]
    default: bool,

    /// Process with the default profile plus topographic correction. See "--show-default" to list the profile.
    #[clap(long)]
    default_with_topo: bool,

    /// Show the default profile and exit
    #[clap(long)]
    show_default: bool,

    /// Show the available steps
    #[clap(long)]
    show_all_steps: bool,

    /// Processing steps to run, separated by commas. Can be a filepath to a newline separated step file.
    #[clap(long)]
    steps: Option<String>,

    /// Output filename or directory. Defaults to the input filename with a ".nc" extension
    #[clap(short, long)]
    output: Option<PathBuf>,

    /// Suppress progress messages
    #[clap(short, long)]
    quiet: bool,

    /// Render an image of the profile and save it to the specified path. Defaults to a jpg in the
    /// directory of the output filepath
    #[clap(short, long)]
    render: Option<Option<PathBuf>>,

    /// Don't export a nc file
    #[clap(long)]
    no_export: bool,

    /// Merge profiles closer in time than the given threshold when in batch mode (e.g. "10 min")
    #[clap(long)]
    merge: Option<String>,
}

enum ParsedArgs {
    Params(Box<gpr::RunParams>),
    Error(String),
    Done,
}

impl Args {
    fn parse(&self) -> ParsedArgs {
        // If the user only wants to show the available steps, stop here.
        if self.show_all_steps {
            println!("Name\t\tDescription");

            for line in gpr::all_available_steps() {
                println!("{}\n{}\n{}\n", line[0], "-".repeat(line[0].len()), line[1]);
            }
            return ParsedArgs::Done;
        };

        // If the user only wants to show the default profile, stop here.
        if self.show_default {
            for line in gpr::default_processing_profile() {
                println!("{}", line);
            }
            return ParsedArgs::Done;
        };

        let merge: Option<Duration> = match &self.merge {
            Some(merge_string) => match parse_duration::parse(merge_string) {
                Ok(d) => Some(d),
                Err(e) => {
                    return ParsedArgs::Error(format!("Error parsing --merge string: {:?}", e))
                }
            },
            None => None,
        };

        let filepaths =
            match &self.filepath {
                Some(fp) => glob::glob(fp)
                    .unwrap()
                    .map(|v| v.unwrap())
                    .collect::<Vec<PathBuf>>(),
                None => return ParsedArgs::Error(
                    "No filepath given.\nUse the help text (\"-h\" or \"--help\") for assistance."
                        .to_string(),
                ),
            };

        // The profile (the list of steps) is the default profile if "--default" was given, or a
        // list of "--steps a,b,c". If none were given, raise an error
        let steps: Vec<String> = match self.info {
            true => Vec::new(),
            false => match self.default_with_topo {
                true => {
                    let mut profile = gpr::default_processing_profile();
                    profile.push("correct_topography".to_string());
                    profile
                }
                false => match self.default {
                    true => gpr::default_processing_profile(),
                    false => match &self.steps {
                        Some(steps) => match tools::parse_step_list(steps) {
                            Ok(s) => s,
                            Err(e) => return ParsedArgs::Error(e),
                        },
                        None => {
                            println!("No processing steps specified. Saving raw data.");
                            vec![]
                        }
                    },
                },
            },
        };
        // Fetch all allowed steps and validate that they exist.
        // It's not a perfect validation, because "dewoww" will pass, but another validation is
        // done later to make sure it's exact. This check is only to run into fewer errors
        // mid-processing (and rather have them before beginning)
        let allowed_steps = gpr::all_available_steps()
            .iter()
            .map(|s| s[0])
            .collect::<Vec<&str>>();
        for step in &steps {
            if !allowed_steps.iter().any(|allowed| step.contains(allowed)) {
                return ParsedArgs::Error(format!("Unrecognized step: {}", step));
            };
        }
        ParsedArgs::Params(Box::new(gpr::RunParams {
            filepaths,
            output_path: self.output.clone(),
            only_info: self.info,
            dem_path: self.dem.clone(),
            cor_path: self.cor.clone(),
            medium_velocity: self.velocity,
            crs: self.crs.clone(),
            quiet: self.quiet,
            track_path: self.track.clone(),
            steps,
            no_export: self.no_export,
            render_path: self.render.clone(),
            merge,
        }))

        //gpr::run(RunParams::from_args(filepaths, steps, merge, self)).unwrap();
    }
}

/// Run the main CLI functionality based on the given arguments
///
/// # Arguments
/// - `arguments`: The Args object containing the parsed arguments.
///
/// # Returns
/// The appropriate exit code.
pub fn main(arguments: Args) -> i32 {
    match arguments.parse() {
        ParsedArgs::Params(params) => match gpr::run(*params) {
            Ok(_) => 0,
            Err(e) => error(&format!("{e:?}"), 1),
        },
        ParsedArgs::Error(message) => error(&message, 1),
        ParsedArgs::Done => 0,
    }
}

/// Print an error to /dev/stderr and return an exit code
///
/// At the moment, it's quite barebones, but this allows for better handling later.
///
/// # Arguments
/// - `message`: The message to print to /dev/stderr
/// - `code`: The exit code
///
/// # Returns
/// The same exit code that was provided
fn error(message: &str, code: i32) -> i32 {
    eprintln!("{}", message);
    code
}
