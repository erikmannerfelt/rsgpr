use crate::{formats, gpr, tools, user_metadata};
use clap::{ArgGroup, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Process one or more GPR profiles into one final output
    Process(ProcessArgs),
    /// Batch-process one or more GPR profiles into many outputs
    BatchProcess(BatchProcessArgs),
    /// Show metadata/location information for one or more GPR profiles
    Info(InfoArgs),
    /// Inspect available processing steps
    Steps(StepsArgs),
    /// Inspect supported formats
    Formats(FormatsArgs),
    /// Work with interpretations (picked layers) of processed radargrams
    Interp(InterpArgs),
    /// Open a local browser GUI for one radargram or a directory of them
    #[cfg(feature = "server")]
    Gui(GuiArgs),
    /// Run the web server explicitly (for remote or persistent deployment)
    #[cfg(feature = "server")]
    Server(ServerArgs),
}

#[derive(Debug, clap::Args)]
pub struct InterpArgs {
    #[command(subcommand)]
    pub command: InterpCommand,
}

#[derive(Debug, Subcommand)]
pub enum InterpCommand {
    /// Derive the level 2 point product from a level 1 interpretation
    Export(InterpExportArgs),
}

#[derive(Debug, clap::Args)]
pub struct InterpExportArgs {
    /// The processed radargram (.nc) the interpretation was drawn on.
    pub radargram: PathBuf,

    /// The level 1 interpretation (a gprinterp JSON document).
    pub interpretation: PathBuf,

    /// Where to write the level 2 product. The format follows the
    /// extension: ".geojson"/".json" for GeoJSON, ".csv" for CSV.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Point spacing along the ground track. A distance in metres ("5",
    /// "2.5"), "auto" to derive one from the radargram's own trace spacing,
    /// or "per-trace" for one point per native trace.
    ///
    /// Spacing is always measured in metres along the track, never in
    /// traces: trace spacing varies with survey speed, so a fixed trace
    /// stride produces unevenly spaced ground positions.
    #[arg(long, default_value = "auto")]
    pub spacing: String,

    /// CRS for the output geometry. WGS84 by default, which is what RFC 7946
    /// requires of GeoJSON. Accepts "native" for the radargram's own
    /// projected CRS, or any CRS string PROJ understands.
    ///
    /// Note that projected GeoJSON is not portable: readers that follow
    /// RFC 7946 will interpret the coordinates as degrees. Native
    /// easting/northing are always present as properties regardless.
    #[arg(long)]
    pub crs: Option<String>,

    /// The author recorded on every exported point. Ridal has no
    /// multi-user support yet, so this is a label rather than an identity.
    #[arg(long, default_value = crate::interp::level2::DEFAULT_USER)]
    pub user: String,
}

#[cfg(feature = "server")]
#[derive(Debug, clap::Args)]
pub struct GuiArgs {
    /// A single processed .nc file, or a directory to scan recursively.
    pub path: PathBuf,

    /// In-memory cache budget for encoded chunk/overview images, in MB.
    #[arg(long)]
    pub cache_memory_mb: Option<usize>,

    /// Number of worker threads for CPU-heavy rendering.
    #[arg(long)]
    pub n_workers: Option<usize>,
}

#[cfg(feature = "server")]
#[derive(Debug, clap::Args)]
pub struct ServerArgs {
    #[command(subcommand)]
    pub command: ServerCommand,
}

#[cfg(feature = "server")]
#[derive(Debug, Subcommand)]
pub enum ServerCommand {
    /// Start the HTTP server
    Start(ServerStartArgs),
}

#[cfg(feature = "server")]
#[derive(Debug, clap::Args)]
pub struct ServerStartArgs {
    /// A single processed .nc file, or a directory to scan recursively.
    pub path: PathBuf,

    /// Bind address. Loopback by default; binding elsewhere is explicit
    /// because this milestone implements no authentication (#120).
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port. A stable default rather than an OS-assigned ephemeral
    /// port, since this mode is for persistent/remote deployment.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Open a browser after starting (off by default in this mode).
    #[arg(long)]
    pub open_browser: bool,

    /// In-memory cache budget for encoded chunk/overview images, in MB.
    #[arg(long)]
    pub cache_memory_mb: Option<usize>,

    /// Number of worker threads for CPU-heavy rendering.
    #[arg(long)]
    pub n_workers: Option<usize>,
}

#[derive(Debug, clap::Args)]
#[command(group(
    ArgGroup::new("step_choice")
        .required(false)
        .args(["steps", "default", "default_with_topo"]),
))]
pub struct ProcessArgs {
    /// Input header/data path(s). Explicit paths are preferred, but glob patterns are also expanded.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Velocity of the medium in m/ns. Defaults to the typical velocity of ice.
    #[arg(short, long, default_value = "0.168")]
    pub velocity: f32,

    /// Load a separate ".cor" file (RAMAC only). If not given, it will be searched for automatically.
    #[arg(short, long)]
    pub cor: Option<PathBuf>,

    /// Correct elevation values with a DEM
    #[arg(short, long)]
    pub dem: Option<PathBuf>,

    /// Which coordinate reference system to project coordinates in.
    #[arg(long)]
    pub crs: Option<String>,

    /// Export the location track to CSV. If no value is given, a sidecar path is derived from the output path.
    #[arg(short, long)]
    pub track: Option<Option<PathBuf>>,

    /// Process with the default profile.
    #[arg(long)]
    pub default: bool,

    /// Process with the default profile plus topographic correction.
    #[arg(long = "default-with-topo")]
    pub default_with_topo: bool,

    /// Processing steps to run, separated by commas. Can also be a filepath to a newline-separated step file.
    #[arg(long)]
    pub steps: Option<String>,

    /// Output filename or directory. Defaults to the first input with a ".nc" extension.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Suppress progress messages
    #[arg(short, long)]
    pub quiet: bool,

    /// Render an image of the profile and save it to the specified path. If no path is given, a JPG sidecar is used.
    #[arg(short, long)]
    pub render: Option<Option<PathBuf>>,

    /// Don't export an nc file
    #[arg(long)]
    pub no_export: bool,

    /// Override the antenna center frequency (in MHz) from file metadata
    #[arg(long)]
    pub override_antenna_mhz: Option<f32>,

    /// Override the antenna separation (in m) from file metadata
    #[arg(long)]
    pub override_antenna_separation: Option<f32>,

    /// Add user metadata as key=value. Repeatable.
    #[arg(long = "metadata", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub metadata: Vec<String>,

    /// Stable, unique identifier for this radargram (lowercase ASCII, digits, '-', '_').
    /// Defaults to the output file stem if not given.
    #[arg(long = "radargram-id")]
    pub radargram_id: Option<String>,

    /// Human-readable display label. Purely cosmetic: has no identity semantics.
    #[arg(long = "display-name")]
    pub display_name: Option<String>,

    /// Human-readable name of the group this radargram belongs to (survey,
    /// campaign, location; Unicode is fine), for catalog grouping. A stable
    /// URL/filesystem-safe id is derived from this automatically unless
    /// --group-id overrides it. `--group` is a supported alias.
    #[arg(long = "group-name", alias = "group")]
    pub group: Option<String>,

    /// Explicit override for the group's id, when the id automatically
    /// derived from --group-name is not the one wanted.
    #[arg(long = "group-id")]
    pub group_id: Option<String>,
}

#[derive(Debug, clap::Args)]
#[command(group(
    ArgGroup::new("step_choice")
        .required(false)
        .args(["steps", "default", "default_with_topo"]),
))]
pub struct BatchProcessArgs {
    /// Input header/data path(s). Explicit paths are preferred, but glob patterns are also expanded.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Output directory. Must already exist.
    #[arg(short, long, required = true)]
    pub output: PathBuf,

    /// Velocity of the medium in m/ns. Defaults to the typical velocity of ice.
    #[arg(short, long, default_value = "0.168")]
    pub velocity: f32,

    /// Load a separate ".cor" file (RAMAC only). If not given, it will be searched for automatically.
    #[arg(short, long)]
    pub cor: Option<PathBuf>,

    /// Correct elevation values with a DEM
    #[arg(short, long)]
    pub dem: Option<PathBuf>,

    /// Which coordinate reference system to project coordinates in.
    #[arg(long)]
    pub crs: Option<String>,

    /// Export location tracks to CSV in the given directory.
    #[arg(short, long)]
    pub track: Option<Option<PathBuf>>,

    /// Process with the default profile.
    #[arg(long)]
    pub default: bool,

    /// Process with the default profile plus topographic correction.
    #[arg(long = "default-with-topo")]
    pub default_with_topo: bool,

    /// Processing steps to run, separated by commas. Can also be a filepath to a newline-separated step file.
    #[arg(long)]
    pub steps: Option<String>,

    /// Suppress progress messages
    #[arg(short, long)]
    pub quiet: bool,

    /// Render images into the given directory.
    #[arg(short, long)]
    pub render: Option<Option<PathBuf>>,

    /// Don't export nc files
    #[arg(long)]
    pub no_export: bool,

    /// Merge neighboring chronological profiles that are closer than the given threshold
    /// (e.g. "10 min"). Incompatible neighbors remain separate outputs.
    #[arg(long)]
    pub merge: Option<String>,

    /// Override the antenna center frequency (in MHz) from file metadata
    #[arg(long)]
    pub override_antenna_mhz: Option<f32>,

    /// Override the antenna separation (in m) from file metadata
    #[arg(long)]
    pub override_antenna_separation: Option<f32>,

    /// Add user metadata as key=value. Repeatable.
    #[arg(long = "metadata", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub metadata: Vec<String>,

    /// Human-readable name of the group all outputs in this batch belong to
    /// (survey, campaign, location; Unicode is fine), for catalog grouping.
    /// Applied uniformly; radargram IDs and display names are still derived
    /// per-output since an explicit single value would collide. `--group`
    /// is a supported alias.
    #[arg(long = "group-name", alias = "group")]
    pub group: Option<String>,

    /// Explicit override for the group's id, when the id automatically
    /// derived from --group-name is not the one wanted.
    #[arg(long = "group-id")]
    pub group_id: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct InfoArgs {
    /// Input header/data path(s). Explicit paths are preferred, but glob patterns are also expanded.
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,

    /// Emit JSON instead of human-readable text
    #[arg(long)]
    pub json: bool,

    /// Velocity of the medium in m/ns. Defaults to the typical velocity of ice.
    #[arg(short, long, default_value = "0.168")]
    pub velocity: f32,

    /// Load a separate ".cor" file (RAMAC only). If not given, it will be searched for automatically.
    #[arg(short, long)]
    pub cor: Option<PathBuf>,

    /// Correct elevation values with a DEM
    #[arg(short, long)]
    pub dem: Option<PathBuf>,

    /// Which coordinate reference system to project coordinates in.
    #[arg(long)]
    pub crs: Option<String>,

    /// Override the antenna center frequency (in MHz) from file metadata
    #[arg(long)]
    pub override_antenna_mhz: Option<f32>,

    /// Override the antenna separation (in m) from file metadata
    #[arg(long)]
    pub override_antenna_separation: Option<f32>,
}

#[derive(Debug, clap::Args)]
#[command(group(
    ArgGroup::new("steps_mode")
        .required(false)
        .args(["describe_all", "describe", "default"]),
))]
pub struct StepsArgs {
    /// Show descriptions for all steps
    #[arg(long = "describe-all")]
    pub describe_all: bool,

    /// Show the description for one step
    #[arg(long)]
    pub describe: Option<String>,

    /// Show the default processing pipeline
    #[arg(long)]
    pub default: bool,

    /// Emit JSON instead of human-readable text
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, clap::Args)]
pub struct FormatsArgs {
    /// Emit JSON instead of human-readable text
    #[arg(long)]
    pub json: bool,
}

fn resolve_steps(
    default: bool,
    default_with_topo: bool,
    steps: Option<&str>,
) -> Result<Vec<String>, String> {
    let resolved_steps = if default_with_topo {
        let mut profile = gpr::default_processing_profile();
        profile.push("correct_topography".to_string());
        profile
    } else if default {
        gpr::default_processing_profile()
    } else if let Some(step_text) = steps {
        tools::parse_step_list(step_text)?
    } else {
        vec![]
    };

    gpr::validate_steps(&resolved_steps)?;
    Ok(resolved_steps)
}

fn optional_existing_dir(
    value: &Option<Option<PathBuf>>,
    label: &str,
) -> Result<Option<PathBuf>, String> {
    match value {
        None => Ok(None),
        Some(None) => Ok(None),
        Some(Some(path)) => {
            if !path.is_dir() {
                Err(format!(
                    "{label} must be an existing directory in batch mode: {}",
                    path.display()
                ))
            } else {
                Ok(Some(path.clone()))
            }
        }
    }
}

#[cfg(feature = "cli")]
#[allow(dead_code)]
pub fn main(arguments: Args) -> i32 {
    match run(arguments) {
        Ok(()) => 0,
        Err(message) => error(&message, 1),
    }
}

pub fn run(arguments: Args) -> Result<(), String> {
    match arguments.command {
        Commands::Process(args) => process_command(&args),
        Commands::BatchProcess(args) => batch_process_command(&args),
        Commands::Info(args) => info_command(args),
        Commands::Steps(args) => steps_command(args),
        Commands::Formats(args) => formats_command(args),
        Commands::Interp(args) => match args.command {
            InterpCommand::Export(args) => interp_export_command(&args),
        },
        #[cfg(feature = "server")]
        Commands::Gui(args) => gui_command(args),
        #[cfg(feature = "server")]
        Commands::Server(args) => server_command(args),
    }
}

#[cfg(feature = "server")]
fn render_service_config(
    cache_memory_mb: Option<usize>,
    n_workers: Option<usize>,
) -> Result<crate::server::render::service::RenderServiceConfig, String> {
    // Rejected rather than silently clamped: n_workers sizes the render
    // permit semaphore, and zero permits would leave every image request
    // waiting until it times out into a 503. A user who typed 0 meant
    // something, and it is not that.
    if n_workers == Some(0) {
        return Err("--n-workers must be at least 1".to_string());
    }
    let default = crate::server::render::service::RenderServiceConfig::default();
    Ok(crate::server::render::service::RenderServiceConfig {
        cache_memory_mb: cache_memory_mb.unwrap_or(default.cache_memory_mb),
        n_workers: n_workers.unwrap_or(default.n_workers),
        ..default
    })
}

#[cfg(feature = "server")]
fn gui_command(args: GuiArgs) -> Result<(), String> {
    let config = render_service_config(args.cache_memory_mb, args.n_workers)?;
    crate::server::launch::run_gui(&args.path, config)
}

#[cfg(feature = "server")]
fn server_command(args: ServerArgs) -> Result<(), String> {
    match args.command {
        ServerCommand::Start(start_args) => {
            let host: std::net::IpAddr = start_args
                .host
                .parse()
                .map_err(|e| format!("Invalid --host '{}': {e}", start_args.host))?;
            let config = render_service_config(start_args.cache_memory_mb, start_args.n_workers)?;
            crate::server::launch::run_server_start(
                &start_args.path,
                host,
                start_args.port,
                start_args.open_browser,
                config,
            )
        }
    }
}

fn process_command(args: &ProcessArgs) -> Result<(), String> {
    let resolved_steps =
        resolve_steps(args.default, args.default_with_topo, args.steps.as_deref())?;
    let user_metadata = user_metadata::parse_cli_metadata(&args.metadata)?;

    let params = gpr::RunParams {
        filepaths: args.inputs.clone(),
        output_path: args.output.clone(),
        dem_path: args.dem.clone(),
        cor_path: args.cor.clone(),
        medium_velocity: args.velocity,
        crs: args.crs.clone(),
        quiet: args.quiet,
        track_path: args.track.clone(),
        steps: resolved_steps,
        no_export: args.no_export,
        render_path: args.render.clone(),
        override_antenna_mhz: args.override_antenna_mhz,
        override_antenna_separation: args.override_antenna_separation,
        user_metadata,
        radargram_id: args.radargram_id.clone(),
        display_name: args.display_name.clone(),
        group: args.group.clone(),
        group_id: args.group_id.clone(),
    };

    let result = gpr::run(params).map_err(|e| e.to_string())?;
    if !args.quiet {
        println!("{}", result.output_path.display());
    }
    Ok(())
}
fn batch_process_command(args: &BatchProcessArgs) -> Result<(), String> {
    if !args.output.is_dir() {
        return Err(format!(
            "output must be an existing directory in batch mode: {}",
            args.output.display()
        ));
    }

    let render_dir = optional_existing_dir(&args.render, "render")?;
    let track_dir = optional_existing_dir(&args.track, "track")?;

    let resolved_steps =
        resolve_steps(args.default, args.default_with_topo, args.steps.as_deref())?;
    let user_metadata = user_metadata::parse_cli_metadata(&args.metadata)?;

    let params = gpr::BatchRunParams {
        filepaths: args.inputs.clone(),
        output_dir: args.output.clone(),
        dem_path: args.dem.clone(),
        cor_path: args.cor.clone(),
        medium_velocity: args.velocity,
        crs: args.crs.clone(),
        quiet: args.quiet,
        track_dir,
        steps: resolved_steps,
        no_export: args.no_export,
        render_dir,
        merge: args.merge.clone(),
        override_antenna_mhz: args.override_antenna_mhz,
        override_antenna_separation: args.override_antenna_separation,
        user_metadata,
        group: args.group.clone(),
        group_id: args.group_id.clone(),
    };

    let result = gpr::run_batch(params)?;
    if !args.quiet {
        for path in result.output_paths {
            println!("{}", path.display());
        }
    }
    Ok(())
}
fn info_command(args: InfoArgs) -> Result<(), String> {
    let params = gpr::InfoParams {
        filepaths: args.inputs,
        dem_path: args.dem,
        cor_path: args.cor,
        medium_velocity: args.velocity,
        crs: args.crs,
        override_antenna_mhz: args.override_antenna_mhz,
        override_antenna_separation: args.override_antenna_separation,
    };
    let records = gpr::inspect(params).map_err(|e| format!("{e:?}"))?;
    if args.json {
        if records.len() == 1 {
            println!(
                "{}",
                serde_json::to_string_pretty(&records[0]).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&records).map_err(|e| e.to_string())?
            );
        }
    } else {
        for (i, record) in records.iter().enumerate() {
            if i > 0 {
                println!();
            }
            print_info_record(record);
        }
    }
    Ok(())
}

fn steps_command(args: StepsArgs) -> Result<(), String> {
    let all_steps = gpr::all_available_steps();
    if args.json {
        if args.default {
            println!(
                "{}",
                serde_json::to_string_pretty(&gpr::default_processing_profile())
                    .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        if let Some(step_name) = args.describe {
            let mapping = step_mapping(Some(step_name), &all_steps)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&mapping).map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        if args.describe_all {
            let mapping = step_mapping(None, &all_steps)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&mapping).map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        let names = all_steps
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<String>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&names).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    if args.default {
        for step in gpr::default_processing_profile() {
            println!("{step}");
        }
        return Ok(());
    }
    if let Some(step_name) = args.describe {
        let mapping = step_mapping(Some(step_name), &all_steps)?;
        for (name, description) in mapping {
            println!("{name}\n{}\n{description}\n", "-".repeat(name.len()));
        }
        return Ok(());
    }
    if args.describe_all {
        for (name, description) in all_steps {
            println!("{name}\n{}\n{description}\n", "-".repeat(name.len()));
        }
        return Ok(());
    }

    for (name, _) in all_steps {
        println!("{name}");
    }
    Ok(())
}

fn formats_command(args: FormatsArgs) -> Result<(), String> {
    let all_formats = formats::all_formats();
    if args.json {
        let payload = serde_json::json!({ "formats": all_formats });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    for fmt in all_formats {
        println!("{}", fmt.name);
        println!("{}", "-".repeat(fmt.name.len()));
        println!("{}", fmt.description);
        println!("Read:  {}", fmt.capabilities.read);
        println!("Write: {}", fmt.capabilities.write);
        println!(
            "Files:  header={} data={} coordinates={}",
            fmt.files.header, fmt.files.data, fmt.files.coordinates
        );
        println!();
    }
    Ok(())
}

// TODO: Might not be used anywhere (2026-03-28)
#[allow(dead_code)]
fn choose_steps(
    default: bool,
    default_with_topo: bool,
    steps: Option<&str>,
) -> Result<Vec<String>, String> {
    if default_with_topo {
        let mut profile = gpr::default_processing_profile();
        profile.push("correct_topography".to_string());
        return Ok(profile);
    }
    if default {
        return Ok(gpr::default_processing_profile());
    }
    match steps {
        Some(step_text) => tools::parse_step_list(step_text),
        None => Ok(vec![]),
    }
}

fn step_mapping(
    only_name: Option<String>,
    all_steps: &[(String, String)],
) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::<String, String>::new();
    for (name, description) in all_steps {
        if let Some(target) = &only_name {
            if name != target {
                continue;
            }
        }
        out.insert(name.clone(), description.clone());
    }
    if let Some(target) = only_name {
        if out.is_empty() {
            return Err(format!("Unknown step: {target}"));
        }
    }
    Ok(out)
}

fn print_info_record(record: &gpr::InfoRecord) {
    println!("Input:\t\t{}", record.input);
    println!(
        "Format:\t\t{} ({})",
        record.format.name, record.format.description
    );
    println!("Header:\t\t{}", record.related_files.header);
    println!("Data:\t\t{}", record.related_files.data);
    println!("Coordinates:\t{}", record.related_files.coordinates);
    println!();
    println!("Metadata");
    println!("--------");
    println!("Samples:\t{}", record.metadata.samples);
    println!("Traces:\t\t{}", record.metadata.last_trace);
    println!("Time window:\t{} ns", record.metadata.time_window_ns);
    println!(
        "Velocity:\t{} m/ns",
        record.metadata.medium_velocity_m_per_ns
    );
    println!(
        "Sampling freq:\t{} MHz",
        record.metadata.sampling_frequency_mhz
    );
    println!("Antenna:\t{}", record.metadata.antenna_name);
    println!("Antenna MHz:\t{}", record.metadata.antenna_mhz);
    println!("Antenna sep:\t{} m", record.metadata.antenna_separation_m);
    println!();
    println!("Location");
    println!("--------");
    println!("Points:\t\t{}", record.location.n_points);
    println!("CRS:\t\t{}", record.location.crs);
    println!("Start:\t\t{}", record.location.start_time);
    println!("Stop:\t\t{}", record.location.stop_time);
    println!("Duration:\t{:.3} s", record.location.duration_s);
    println!("Track length:\t{:.3} m", record.location.track_length_m);
    println!(
        "Altitude:\t{:.3} - {:.3} m",
        record.location.altitude_min_m, record.location.altitude_max_m
    );
    println!(
        "Centroid:\tE {:.3}, N {:.3}, Z {:.3}",
        record.location.centroid.easting,
        record.location.centroid.northing,
        record.location.centroid.altitude
    );
    println!(
        "Correction:\t{}{}",
        record.location.correction.kind,
        record
            .location
            .correction
            .source
            .as_ref()
            .map(|s| format!(" ({s})"))
            .unwrap_or_default()
    );
}

#[cfg(feature = "cli")]
#[allow(dead_code)]
fn error(message: &str, code: i32) -> i32 {
    eprintln!("{message}");
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[cfg(feature = "server")]
    #[test]
    fn n_workers_zero_is_rejected_not_silently_clamped() {
        // n_workers sizes the render permit semaphore; zero permits would
        // leave every image request waiting until it times out into a 503,
        // which is a confusing way to learn about a typo.
        let err = super::render_service_config(None, Some(0)).unwrap_err();
        assert!(err.contains("--n-workers"), "{err}");

        assert!(super::render_service_config(None, Some(1)).is_ok());
        assert_eq!(
            super::render_service_config(None, None).unwrap().n_workers,
            crate::server::render::service::RenderServiceConfig::default().n_workers,
            "an omitted flag must keep the default, not become an error"
        );
    }

    #[test]
    fn test_parse_process_command() {
        let args = Args::parse_from([
            "ridal",
            "process",
            "a.rad",
            "b.rad",
            "--default",
            "-o",
            "out.nc",
        ]);
        match args.command {
            Commands::Process(process) => {
                assert_eq!(
                    process.inputs,
                    vec![PathBuf::from("a.rad"), PathBuf::from("b.rad")]
                );
                assert!(process.default);
                assert_eq!(process.output, Some(PathBuf::from("out.nc")));
            }
            _ => panic!("Expected process command"),
        }
    }

    #[test]
    fn test_parse_info_command() {
        let args = Args::parse_from(["ridal", "info", "line01.rad", "--json"]);
        match args.command {
            Commands::Info(info) => {
                assert_eq!(info.inputs, vec![PathBuf::from("line01.rad")]);
                assert!(info.json);
            }
            _ => panic!("Expected info command"),
        }
    }

    #[test]
    fn test_parse_override_antenna_separation() {
        let args = Args::parse_from([
            "ridal",
            "process",
            "line01.DZT",
            "--override-antenna-separation",
            "1.25",
        ]);
        match args.command {
            Commands::Process(process) => {
                assert_eq!(process.override_antenna_separation, Some(1.25));
            }
            _ => panic!("Expected process command"),
        }
    }

    #[test]
    fn test_choose_steps_default_with_topo() {
        let steps = choose_steps(false, true, None).unwrap();
        assert!(steps.iter().any(|step| step == "correct_topography"));
    }
}

/// Parse the `--spacing` value.
///
/// Accepts a bare number of metres, "auto", or "per-trace". A bare number is
/// metres rather than traces by design: see [`InterpExportArgs::spacing`].
fn parse_spacing(text: &str) -> Result<crate::interp::level2::Spacing, String> {
    use crate::interp::level2::Spacing;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(Spacing::Auto),
        "per-trace" | "per_trace" | "pertrace" => Ok(Spacing::PerTrace),
        other => {
            // Tolerate a trailing "m" so `--spacing 5m` does not fail on
            // something that obviously means five metres.
            let numeric = other.strip_suffix('m').unwrap_or(other);
            let step: f64 = numeric.parse().map_err(|_| {
                format!(
                    "Could not read --spacing '{text}'. Expected a distance in metres \
                     (e.g. '5' or '2.5'), 'auto', or 'per-trace'."
                )
            })?;
            if !step.is_finite() || step <= 0.0 {
                return Err(format!("--spacing must be greater than zero, got '{text}'"));
            }
            Ok(Spacing::ArcLength(step))
        }
    }
}

fn interp_export_command(args: &InterpExportArgs) -> Result<(), String> {
    let spacing = parse_spacing(&args.spacing)?;

    let text = std::fs::read_to_string(&args.interpretation)
        .map_err(|e| format!("Could not read {:?}: {e}", args.interpretation))?;
    let document = gprinterp::Document::from_json(&text).map_err(|e| {
        format!(
            "Could not parse {:?} as gprinterp: {e}",
            args.interpretation
        )
    })?;

    // Validation warnings are surfaced but not fatal: the format is
    // deliberately permissive, and a document missing a stable feature id
    // still exports correctly.
    let report = gprinterp::validate(&document);
    if !report.errors.is_empty() {
        // All of them, not just the first: a hand-written document usually
        // has several problems at once, and fixing them one round trip at a
        // time is needless.
        let errors: Vec<String> = report.errors.iter().map(|e| format!("  - {e}")).collect();
        return Err(format!(
            "{:?} is not a valid gprinterp document:\n{}",
            args.interpretation,
            errors.join("\n")
        ));
    }
    for warning in &report.warnings {
        eprintln!("warning: {warning}");
    }

    let geometry = crate::interp::source::read_geometry(&args.radargram)?;

    // The interpretation names a radargram; exporting it against a different
    // one silently produces plausible, wrong depths. Refusing is the only
    // safe default, since nothing downstream can detect the mistake.
    if document.key != geometry.radargram_id {
        return Err(format!(
            "{:?} was drawn on radargram '{}', but {:?} is '{}'. \
             Export it against the radargram it was drawn on.",
            args.interpretation, document.key, args.radargram, geometry.radargram_id
        ));
    }
    if let Some(revision) = document
        .source
        .as_ref()
        .and_then(|s| s.revision_id.as_deref())
    {
        if revision != geometry.revision_id {
            eprintln!(
                "warning: the interpretation was drawn on revision {revision}, but {:?} is \
                 revision {}. The radargram has been reprocessed since, so trace and sample \
                 indices may no longer line up.",
                args.radargram, geometry.revision_id
            );
        }
    }

    let export = crate::interp::level2::export(&document, &geometry, spacing, &args.user)
        .map_err(|e| format!("{e}"))?;

    let output_crs = match &args.crs {
        None => crate::interp::writer::OutputCrs::Wgs84,
        Some(name) => crate::interp::writer::OutputCrs::Named(name.clone()),
    };

    let extension = args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let serialized = match extension.as_str() {
        "csv" => crate::interp::writer::to_csv(&export),
        "geojson" | "json" => crate::interp::writer::to_geojson(&export, &output_crs)?,
        other => {
            return Err(format!(
                "Cannot tell what format to write from the extension '{other}'. \
                 Use '.geojson' or '.csv'."
            ))
        }
    };
    if extension == "csv" && args.crs.is_some() {
        eprintln!(
            "warning: --crs is ignored for CSV output, which always carries both native \
             easting/northing and WGS84 longitude/latitude as columns."
        );
    }

    std::fs::write(&args.output, serialized)
        .map_err(|e| format!("Could not write {:?}: {e}", args.output))?;

    let spacing_note = match export.spacing_m {
        Some(step) => format!("{step} m spacing"),
        None => "per-trace spacing".to_string(),
    };
    println!(
        "Wrote {} point(s) from {} layer(s) at {spacing_note} to {:?}",
        export.points.len(),
        document.layers().len(),
        args.output
    );
    Ok(())
}
