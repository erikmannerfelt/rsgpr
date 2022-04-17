use crate::{gpr, io};
/// Functions to handle the command line interface (CLI)
use clap::Parser;
use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Filepath of the Malå rd3 file
    #[clap(short, long)]
    filepath: Option<PathBuf>,

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

    /// Show the default profile and exit
    #[clap(long)]
    show_default: bool,

    /// Show the available steps
    #[clap(long)]
    show_all_steps: bool,

    /// Processing steps to run.
    #[clap(long)]
    steps: Option<String>,

    /// Output filename or directory. Defaults to the input filename with a ".nc" extension
    #[clap(short, long)]
    output: Option<PathBuf>,

    /// Surpress progress messages
    #[clap(short, long)]
    quiet: bool,

    /// Render an image of the profile and save it to the specified path. Defaults to a jpg in the
    /// directory of the output filepath
    #[clap(short, long)]
    render: Option<Option<PathBuf>>,

    /// Don't export a nc file
    #[clap(long)]
    no_export: bool,
}

/// Run the main CLI functionality based on the given arguments
///
/// # Arguments
/// - `arguments`: The Args object containing the parsed arguments.
///
/// # Returns
/// The appropriate exit code.
pub fn main(arguments: Args) -> i32 {
    // If the user only wants to show the available steps, stop here.
    if arguments.show_all_steps {
        println!("Name\t\tDescription");

        for line in gpr::all_available_steps() {
            println!("{}\n{}\n{}\n", line[0], "-".repeat(line[0].len()), line[1]);
        }
        return 0;
    };

    // If the user only wants to show the default profile, stop here.
    if arguments.show_default {
        for line in gpr::default_processing_profile() {
            println!("{}", line);
        }
        return 0;
    };

    // If a filepath was given, the file should be processed.
    // If none was given, throw an error
    if let Some(input_filepath) = arguments.filepath {
        // The given filepath may be ".rd3" or may not have an extension at all
        // Counterintuitively to the user point of view, it's the ".rad" file that should be given
        let rad_filepath = input_filepath.with_extension("rad");

        // Make sure that it exists
        if !rad_filepath.is_file() {
            if input_filepath.is_file() {
                return error(
                    &format!("File found but no '.rad' file found: {:?}", rad_filepath),
                    1,
                );
            }
            return error(&format!("File not found: {:?}", rad_filepath), 1);
        };
        // Load the GPR metadata from the rad file
        let gpr_meta = match io::load_rad(&rad_filepath, arguments.velocity) {
            Ok(x) => x,
            Err(e) => return error(&format!("Error fetching GPR metadata: {:?}", e), 1),
        };

        // Load the GPR location data
        // If the "--cor" argument was used, load from there. Otherwise, try to find a ".cor" file
        let mut gpr_locations = match &arguments.cor {
            Some(fp) => io::load_cor(&fp, &arguments.crs).unwrap(),
            None => gpr_meta.find_cor(&arguments.crs).unwrap(),
        };

        // If a "--dem" was given, substitute elevations using said DEM
        if let Some(dem_path) = &arguments.dem {
            gpr_locations.get_dem_elevations(&dem_path);
        };

        // Construct the output filepath. If one was given, use that.
        // If a path was given and it's a directory, use the file stem + ".nc" of the input
        // filename. If no output path was given, default to the directory of the input.
        let output_filepath = match &arguments.output {
            Some(p) => match p.is_dir() {
                true => p
                    .join(input_filepath.file_stem().unwrap())
                    .with_extension("nc"),
                false => {
                    if let Some(parent) = p.parent() {
                        if !parent.is_dir() {
                            return error(
                                &format!("Output directory of path is not a directory: {:?}", p),
                                1,
                            );
                        };
                    };
                    p.clone()
                }
            },
            None => input_filepath.with_extension("nc"),
        };

        // If the "--info" argument was given, stop here and just show info.
        if arguments.info {
            println!("{}", gpr_meta);
            println!("{}", gpr_locations);
            // If the track should be exported, do so.
            if let Some(potential_track_path) = &arguments.track {
                return export_locations(
                    gpr_locations,
                    potential_track_path,
                    &output_filepath,
                    !arguments.quiet,
                );
            };

            return 0;
        };

        // At this point, the data should be processed.
        let mut gpr = match gpr::GPR::from_meta_and_loc(gpr_locations, gpr_meta) {
            Ok(g) => g,
            Err(e) => {
                return error(
                    &format!(
                        "Error loading GPR data from {:?}: {:?}",
                        rad_filepath.with_extension("rd3"),
                        e
                    ),
                    1,
                )
            }
        };

        // The profile (the list of steps) is the default profile if "--default" was given, or a
        // list of "--steps a,b,c". If none were given, raise an error
        let profile: Vec<String> = match arguments.default {
            true => gpr::default_processing_profile(),
            false => match &arguments.steps {
                Some(steps) => steps.split(",").map(|s| s.trim().to_string()).collect(),
                None => {
                    return error(
                        &format!("No steps specified. Choose a profile or what steps to run"),
                        1,
                    )
                }
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
        for step in &profile {
            if !allowed_steps.iter().any(|allowed| step.contains(allowed)) {
                return error(&format!("Unrecognized step: {}", step), 1);
            };
        }

        // Record the starting time to show "t+XX" times
        let start_time = SystemTime::now();
        if !arguments.quiet {
            println!("Processing {:?}", input_filepath.with_extension("rd3"));
        };

        // Run each step sequentially
        for (i, step) in profile.iter().enumerate() {
            if !arguments.quiet {
                println!(
                    "{}/{}, t+{:.2} s, Running step {}. ",
                    i + 1,
                    profile.len(),
                    SystemTime::now()
                        .duration_since(start_time)
                        .unwrap()
                        .as_secs_f32(),
                    step,
                );
            };

            // Stop if any error occurs
            match gpr.process(step) {
                Ok(_) => 0,
                Err(e) => return error(&format!("Error on step {}: {:?}", step, e), 1),
            };
        }

        // Unless the "--no-export" flag was given, export the ".nc" result
        if !arguments.no_export {
            if !arguments.quiet {
                println!("Exporting to {:?}", output_filepath);
            };
            match gpr.export(&output_filepath) {
                Ok(_) => (),
                Err(e) => return error(&format!("Error exporting data: {:?}", e), 1),
            }
        };

        // If "--render" was given, render an image of the output
        // The flag may or may not have a filepath (it can either be "-r" or "-r img.jpg")
        if let Some(potential_fp) = arguments.render {
            // Find out the output filepath. If one was given, use that. If none was given, use
            // the output filepath with a ".jpg" extension. If a directory was given, use the
            // file stem of the output filename and a ".jpg" extension
            let render_filepath = match potential_fp {
                Some(fp) => match fp.is_dir() {
                    true => fp
                        .join(output_filepath.file_stem().unwrap())
                        .with_extension("jpg"),
                    false => fp.clone(),
                },
                None => output_filepath.with_extension("jpg"),
            };
            if !arguments.quiet {
                println!("Rendering image to {:?}", render_filepath);
            };
            gpr.render(&render_filepath).unwrap();
        };

        // If "--track" was given, export the track file.
        if let Some(potential_track_path) = &arguments.track {
            match export_locations(
                gpr.location,
                potential_track_path,
                &output_filepath,
                !arguments.quiet,
            ) {
                0 => (),
                i => return i,
            }
        };

        return 0;
    } else {
        // This is only reached if no filepath was given
        eprintln!("Filepath needs to be provided (-f or --filepath)");
        return 1;
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

/// Export a "track" file.
///
/// It has its own associated function because the logic may happen in two different places in the
/// main() function.
///
/// # Arguments
/// - `gpr_locations`: The GPRLocation object to export
/// - `potential_track_path`: The output path of the track file (if any)
/// - `output_filepath`: The output filepath to derive a track filepath from in case
/// `potential_track_path` was not provided
/// - `verbose`: Print progress?
///
/// # Returns
/// The exit code of the function
fn export_locations(
    gpr_locations: gpr::GPRLocation,
    potential_track_path: &Option<PathBuf>,
    output_filepath: &Path,
    verbose: bool,
) -> i32 {
    // Determine the output filepath. If one was given, use that. If none was given, use the
    // parent and file stem + "_track.csv" of the output filepath. If a directory was given,
    // use the directory + the file stem of the output filepath + "_track.csv".
    let track_path: PathBuf = match potential_track_path {
        Some(fp) => match fp.is_dir() {
            true => fp
                .join(
                    output_filepath
                        .file_stem()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string()
                        + "_track",
                )
                .with_extension("csv"),
            false => fp.clone().to_path_buf(),
        },
        None => output_filepath
            .with_file_name(
                output_filepath
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
                    + "_track",
            )
            .with_extension("csv"),
    };
    if verbose {
        println!("Exporting track to {:?}", track_path);
    };

    match gpr_locations.to_csv(&track_path) {
        Ok(_) => 0,
        Err(e) => error(&format!("Error saving track {:?}: {:?}", track_path, e), 1),
    }
}
