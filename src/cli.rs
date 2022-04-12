use clap::Parser;
use std::{path::{Path, PathBuf}, time::SystemTime};
use crate::{gpr, io};
use std::str::FromStr;


#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    /// Filepath of the Malå rd3 file
    #[clap(short, long)]
    filepath: Option<PathBuf>,

    /// Velocity of the medium in m/ns. Defaults to the typical velocity of ice.
    #[clap(short, long, default_value="0.168")]
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
    #[clap(long, default_value="EPSG:32633")]
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


pub fn main(arguments: Args) -> i32 {

    if arguments.show_all_steps {
        println!("Name\t\tDescription");

        for line in gpr::all_available_steps() {
            println!("{}\n{}\n{}\n",  line[0],"-".repeat(line[0].len()), line[1]);
        };
        return 0

    };

    if arguments.show_default {
        for line in gpr::default_processing_profile() {
            println!("{}", line);
        };
        return 0
    };

    if let Some(input_filepath) = arguments.filepath {


        let rad_filepath = input_filepath.with_extension("rad");

        if !rad_filepath.is_file() {
            if input_filepath.is_file() {
                eprintln!("File found but no '.rad' file found: {:?}", rad_filepath);
            } else {
                eprintln!("File not found: {:?}", rad_filepath);
            }
            return 1
        };
        let gpr_meta = match io::load_rad(match input_filepath.extension() {
            Some(ext) if ext == std::ffi::OsString::from_str("rad").unwrap() => &input_filepath,
            Some(ext) if ext == std::ffi::OsString::from_str("rd3").unwrap() => &rad_filepath,
            None => &rad_filepath,
            Some(_) => {eprintln!("Filepath not understood: {:?} \nSupported files: ['.rd3']", input_filepath); return 1},
        }, arguments.velocity) {
            Ok(x) => x,
            Err(e) => {eprintln!("Uncaught error fetching GPR metadata: {:?}", e); return 1}
        };

        let mut gpr_locations = match &arguments.cor {
            Some(fp) => io::load_cor(&fp, &arguments.crs).unwrap(),
            None => gpr_meta.find_cor(&arguments.crs).unwrap(),
        };

        if let Some(dem_path) = &arguments.dem {
            gpr_locations.get_dem_elevations(&dem_path);
        };

        let output_filepath = match &arguments.output {
            Some(p) => match p.is_dir() {
                true => p.join(input_filepath.file_stem().unwrap()).with_extension("nc"),
                false => p.clone()
            },
            None => input_filepath.with_extension("nc"),
        };

        if arguments.info {
            println!("{}", gpr_meta);
            println!("{}", gpr_locations);
            if let Some(potential_track_path) = &arguments.track {
                return export_locations(gpr_locations, potential_track_path, &output_filepath, !arguments.quiet)
            };

            return 0
        };

        let mut gpr = gpr::GPR::from_meta_and_loc(gpr_locations, gpr_meta).unwrap();

        let profile: Vec<String> = match arguments.default {
            true => gpr::default_processing_profile(),
            false => match &arguments.steps {
                Some(steps) => steps.split(",").map(|s| s.trim().to_string()).collect(),
                None => {eprintln!("No steps specified. Choose a profile or what steps to run"); return 1}
            },
        };

        let allowed_steps = gpr::all_available_steps().iter().map(|s| s[0]).collect::<Vec<&str>>();

        for step in &profile {
            if !allowed_steps.iter().any(|allowed| step.contains(allowed)) {
                eprintln!("Unrecognized step: {}", step);
                return 1;
            };
        };

        let start_time = SystemTime::now();

        if !arguments.quiet {
            println!("Processing {:?}", input_filepath.with_extension("rd3"));
        };

        for (i, step) in profile.iter().enumerate() {
            if !arguments.quiet {
                println!("{}/{}, t+{:.2} s, Running step {}. ",i + 1, profile.len(), SystemTime::now().duration_since(start_time).unwrap().as_secs_f32(), step, );
            };

            match gpr.process(step) {
                Ok(_) => 0,
                Err(e) => {eprintln!("Error on step {}: {:?}", step, e); return 1}
            };

            assert_eq!(gpr.width(), gpr.location.cor_points.len());


        };

        if !arguments.no_export {
            if !arguments.quiet {
                println!("Exporting to {:?}", output_filepath);
            };
            gpr.export(&output_filepath).unwrap();
        };

        match arguments.render {
            Some(potential_fp) => {
                let render_filepath = match potential_fp {
                    Some(fp) => match fp.is_dir() {
                        true => fp.join(output_filepath.file_stem().unwrap()).with_extension("jpg"),
                        false => fp.clone()
                    },
                    None => output_filepath.with_extension("jpg"),
                };
                if !arguments.quiet {
                    println!("Rendering image to {:?}", render_filepath);
                };
                gpr.render(&render_filepath).unwrap();
            },
            None => (),
        };
        if let Some(potential_track_path) = &arguments.track {

            match export_locations(gpr.location, potential_track_path, &output_filepath, !arguments.quiet) {
                0 => (),
                i => return i
            }
        };

        return 0





    } else {

        eprintln!("Filepath needs to be provided (-f or --filepath)");
        return 1;

    }

}

fn export_locations(gpr_locations: gpr::GPRLocation, potential_track_path: &Option<PathBuf>, output_filepath: &Path, verbose: bool) -> i32 {

    let track_path: PathBuf = match potential_track_path {
        Some(fp) => match fp.is_dir() {
            true => fp.join(output_filepath.file_stem().unwrap().to_str().unwrap().to_string() + "_track").with_extension("csv"),
            false => fp.clone().to_path_buf()
        },
        None => output_filepath.with_file_name(output_filepath.file_stem().unwrap().to_str().unwrap().to_string() + "_track").with_extension("csv"),
    };
    if verbose {
        println!("Exporting track to {:?}", track_path);
    };

    match gpr_locations.to_csv(&track_path) {
        Ok(_) => 0,
        Err(e) => {eprintln!("Error saving track {:?}: {:?}", track_path, e); return 1}
    }

}
