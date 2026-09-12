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
    /// Create and inspect Ridal projects
    Project(ProjectArgs),
    /// Open a local browser GUI for one radargram or a directory of them
    #[cfg(feature = "server")]
    Gui(GuiArgs),
    /// Run the web server explicitly (for remote or persistent deployment)
    #[cfg(feature = "server")]
    Server(ServerArgs),
}

#[derive(Debug, clap::Args)]
pub struct ProjectArgs {
    #[command(subcommand)]
    pub command: ProjectCommand,
}

#[derive(Debug, Subcommand)]
pub enum ProjectCommand {
    /// Create a project so interpretations have somewhere to live
    Init(ProjectInitArgs),
    /// Show what a project contains
    Info(ProjectInfoArgs),
    /// Manage who may use the project's server
    User(ProjectUserArgs),
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserArgs {
    #[command(subcommand)]
    pub command: ProjectUserCommand,
}

/// Account management from the command line.
///
/// This exists because of the chicken-and-egg at the start: a project's
/// first administrator cannot be created through the browser, since there is
/// no administrator to authorise it. It happens on the machine itself, which
/// is the one place where access already implies authority.
///
/// Deliberately no `set-password`. A password is set by its owner through a
/// one-time link, so it is never known to two people and there is no default
/// to forget to change; a command that took one would undo that.
#[derive(Debug, Subcommand)]
pub enum ProjectUserCommand {
    /// Create an account and print a one-time invite link
    Add(ProjectUserAddArgs),
    /// List the accounts and what each may do
    List(ProjectUserListArgs),
    /// Change someone's role or download scope
    Set(ProjectUserSetArgs),
    /// Issue a fresh invite link, for a password reset or a lost one
    Reset(ProjectUserResetArgs),
    /// Remove an account. Their interpretations are kept.
    Remove(ProjectUserRemoveArgs),
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserAddArgs {
    /// The account name. Lowercase letters, digits, '-' and '_'; it is used
    /// as a filename inside the project.
    pub name: String,

    /// What they may do: viewer, picker, operator or admin. Each level
    /// includes the ones below it.
    #[arg(long, default_value = "picker")]
    pub role: String,

    /// What they may download: none, picks, derived or all.
    #[arg(long, default_value = "all")]
    pub download: String,

    /// A path inside the project. The project is found by searching upwards.
    #[arg(long, default_value = ".")]
    pub path: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserListArgs {
    /// A path inside the project. The project is found by searching upwards.
    #[arg(default_value = ".")]
    pub path: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserSetArgs {
    pub name: String,

    /// New role: viewer, picker, operator or admin.
    #[arg(long)]
    pub role: Option<String>,

    /// New download scope: none, picks, derived or all.
    #[arg(long)]
    pub download: Option<String>,

    /// A path inside the project. The project is found by searching upwards.
    #[arg(long, default_value = ".")]
    pub path: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserResetArgs {
    pub name: String,

    /// A path inside the project. The project is found by searching upwards.
    #[arg(long, default_value = ".")]
    pub path: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProjectUserRemoveArgs {
    pub name: String,

    /// A path inside the project. The project is found by searching upwards.
    #[arg(long, default_value = ".")]
    pub path: PathBuf,
}

#[derive(Debug, clap::Args)]
pub struct ProjectInitArgs {
    /// Directory to create the project in. Created if it does not exist.
    #[arg(default_value = ".")]
    pub path: PathBuf,

    /// Human-facing project name. Cosmetic.
    #[arg(long)]
    pub name: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct ProjectInfoArgs {
    /// A path inside the project. The project is found by searching upwards.
    #[arg(default_value = ".")]
    pub path: PathBuf,
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
    /// "per-trace" for one point per native trace, or "vertices" for the
    /// picked vertices exactly as drawn.
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

    /// Serve a project without accepting any writes.
    #[arg(long)]
    pub read_only: bool,
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

    /// Serve a project without accepting any writes. Caps every caller at
    /// the "viewer" role, whatever their account says.
    #[arg(long)]
    pub read_only: bool,

    /// Accept password logins while bound to a non-loopback address.
    ///
    /// Ridal does not terminate TLS, so a password sent to a non-loopback
    /// address travels in the clear unless something in front of it is
    /// doing so. Use this only when you know what that something is; the
    /// supported arrangement is to bind loopback behind a TLS-terminating
    /// reverse proxy.
    #[arg(long)]
    pub allow_insecure_login: bool,

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
        Commands::Project(args) => match args.command {
            ProjectCommand::Init(args) => project_init_command(&args),
            ProjectCommand::Info(args) => project_info_command(&args),
            ProjectCommand::User(args) => match args.command {
                ProjectUserCommand::Add(args) => project_user_add_command(&args),
                ProjectUserCommand::List(args) => project_user_list_command(&args),
                ProjectUserCommand::Set(args) => project_user_set_command(&args),
                ProjectUserCommand::Reset(args) => project_user_reset_command(&args),
                ProjectUserCommand::Remove(args) => project_user_remove_command(&args),
            },
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
    crate::server::launch::run_gui(&args.path, args.read_only, config)
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
                start_args.read_only,
                start_args.allow_insecure_login,
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

    /// A project to run `ridal project user` against.
    fn project_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        crate::project::Project::init(dir.path(), Some("test")).unwrap();
        dir
    }

    fn accounts(dir: &tempfile::TempDir) -> crate::project::users::UserSet {
        let project = crate::project::Project::open(dir.path()).unwrap();
        crate::project::users::read(project.documents())
            .unwrap()
            .map(|(set, _)| set)
            .unwrap_or_default()
    }

    fn add(dir: &tempfile::TempDir, name: &str, role: &str) -> Result<(), String> {
        super::project_user_add_command(&ProjectUserAddArgs {
            name: name.to_string(),
            role: role.to_string(),
            download: "all".to_string(),
            path: dir.path().to_path_buf(),
        })
    }

    #[test]
    fn adding_the_first_administrator_is_what_turns_authentication_on() {
        // The bootstrap, and the only way out of the chicken-and-egg: there
        // is no administrator to authorise creating the first one, so it
        // happens on the machine itself.
        let dir = project_dir();
        assert!(!dir.path().join("users.json").exists());

        add(&dir, "erik", "admin").unwrap();

        let set = accounts(&dir);
        assert_eq!(set.users.len(), 1);
        assert_eq!(set.users[0].role, crate::project::users::Role::Admin);
        // Created without a password, with an invite outstanding. There is
        // deliberately no default password to forget to change.
        assert!(!set.users[0].is_activated());
        assert!(set.users[0].invite.is_some());
    }

    #[test]
    fn a_duplicate_name_is_refused_rather_than_replacing_the_account() {
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        let error = add(&dir, "erik", "picker").unwrap_err();
        assert!(error.contains("already"), "{error}");
        assert_eq!(
            accounts(&dir).users[0].role,
            crate::project::users::Role::Admin,
            "the existing account must be untouched"
        );
    }

    #[test]
    fn an_unknown_role_or_scope_lists_the_ones_that_exist() {
        let dir = project_dir();
        let error = add(&dir, "erik", "editor").unwrap_err();
        // `editor` is the one someone will reach for, and the message has to
        // point at `operator` rather than just saying no.
        assert!(error.contains("operator"), "{error}");

        let error = super::project_user_add_command(&ProjectUserAddArgs {
            name: "erik".to_string(),
            role: "picker".to_string(),
            download: "everything".to_string(),
            path: dir.path().to_path_buf(),
        })
        .unwrap_err();
        assert!(error.contains("derived"), "{error}");
    }

    #[test]
    fn a_reset_replaces_the_outstanding_invite_rather_than_adding_one() {
        // Two live tokens for one account would make "single use" a lie.
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        let first = accounts(&dir).users[0].invite.clone().unwrap();

        super::project_user_reset_command(&ProjectUserResetArgs {
            name: "erik".to_string(),
            path: dir.path().to_path_buf(),
        })
        .unwrap();

        let second = accounts(&dir).users[0].invite.clone().unwrap();
        assert_ne!(first.token_hash, second.token_hash);
    }

    #[test]
    fn changing_a_role_signs_their_sessions_out() {
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        add(&dir, "student", "picker").unwrap();
        let before = accounts(&dir).users[1].credential_version;

        super::project_user_set_command(&ProjectUserSetArgs {
            name: "student".to_string(),
            role: Some("viewer".to_string()),
            download: None,
            path: dir.path().to_path_buf(),
        })
        .unwrap();

        let after = &accounts(&dir).users[1];
        assert_eq!(after.role, crate::project::users::Role::Viewer);
        assert!(
            after.credential_version > before,
            "a demotion must reach an open session"
        );
    }

    #[test]
    fn setting_nothing_is_refused_rather_than_silently_doing_nothing() {
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        let error = super::project_user_set_command(&ProjectUserSetArgs {
            name: "erik".to_string(),
            role: None,
            download: None,
            path: dir.path().to_path_buf(),
        })
        .unwrap_err();
        assert!(error.contains("--role"), "{error}");
    }

    #[test]
    fn the_last_administrator_cannot_be_demoted_or_removed_from_the_command_line_either() {
        // The same guard the HTTP route applies. Without it here, the
        // command line would be a way around the check rather than the
        // place it is most likely to be needed.
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        add(&dir, "student", "picker").unwrap();

        let error = super::project_user_set_command(&ProjectUserSetArgs {
            name: "erik".to_string(),
            role: Some("operator".to_string()),
            download: None,
            path: dir.path().to_path_buf(),
        })
        .unwrap_err();
        assert!(error.contains("only administrator"), "{error}");

        let error = super::project_user_remove_command(&ProjectUserRemoveArgs {
            name: "erik".to_string(),
            path: dir.path().to_path_buf(),
        })
        .unwrap_err();
        assert!(error.contains("only administrator"), "{error}");
    }

    #[test]
    fn removing_an_account_keeps_the_interpretations_it_authored() {
        // Attributed scientific data. The person leaving does not unmake it.
        let dir = project_dir();
        add(&dir, "erik", "admin").unwrap();
        add(&dir, "student", "picker").unwrap();

        let project = crate::project::Project::open(dir.path()).unwrap();
        let radargram = crate::identity::RadargramId::new("line-01").unwrap();
        let user = crate::identity::UserId::new("student").unwrap();
        project
            .documents()
            .write(
                std::path::Path::new("interpretations/line-01/student.gprinterp.json"),
                r#"{"key":"line-01","features":[]}"#,
                &crate::project::store::Expectation::Any,
            )
            .unwrap();

        super::project_user_remove_command(&ProjectUserRemoveArgs {
            name: "student".to_string(),
            path: dir.path().to_path_buf(),
        })
        .unwrap();

        assert!(accounts(&dir).get(&user).is_none());
        assert_eq!(
            crate::project::interpretations::list_users(project.documents(), &radargram).unwrap(),
            vec!["student".to_string()],
            "the picks must outlive the account"
        );
    }

    #[test]
    fn a_user_command_outside_a_project_says_how_to_make_one() {
        let dir = tempfile::tempdir().unwrap();
        let error = add(&dir, "erik", "admin").unwrap_err();
        assert!(error.contains("ridal project init"), "{error}");
    }

    #[test]
    fn user_subcommands_parse() {
        let args = Args::parse_from(["ridal", "project", "user", "add", "erik", "--role", "admin"]);
        match args.command {
            Commands::Project(project) => match project.command {
                ProjectCommand::User(user) => match user.command {
                    ProjectUserCommand::Add(add) => {
                        assert_eq!(add.name, "erik");
                        assert_eq!(add.role, "admin");
                        // The default that matters: a new account can
                        // download everything unless someone decides
                        // otherwise, which is what every Ridal did before.
                        assert_eq!(add.download, "all");
                    }
                    other => panic!("{other:?}"),
                },
                other => panic!("{other:?}"),
            },
            _ => panic!("expected a project command"),
        }
    }

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
pub fn parse_spacing(text: &str) -> Result<crate::interp::level2::Spacing, String> {
    use crate::interp::level2::Spacing;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(Spacing::Auto),
        "per-trace" | "per_trace" | "pertrace" => Ok(Spacing::PerTrace),
        "vertices" | "vertex" => Ok(Spacing::Vertices),
        other => {
            // Tolerate a trailing "m" so `--spacing 5m` does not fail on
            // something that obviously means five metres.
            let numeric = other.strip_suffix('m').unwrap_or(other);
            let step: f64 = numeric.parse().map_err(|_| {
                format!(
                    "Could not read --spacing '{text}'. Expected a distance in metres \
                     (e.g. '5' or '2.5'), 'auto', 'per-trace', or 'vertices'."
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

    // Shared with the HTTP download routes, which previously skipped this
    // and produced a plausible, wrong file from the same inputs.
    let identity = crate::interp::checks::check_identity(&document, &geometry)
        .map_err(|e| format!("{:?}: {e}", args.interpretation))?;
    if let Some(warning) = identity.warning {
        eprintln!("warning: {warning}");
    }

    // Overhang permission is a property of the project's layer vocabulary.
    // An export from outside a project has no vocabulary, so nothing has
    // opted out and the guardrail applies everywhere -- the safe default.
    let layer_set =
        match crate::project::Project::discover(&args.radargram).map_err(|e| e.to_string())? {
            Some(project) => {
                crate::project::layers::read(project.documents())
                    .map_err(|e| e.to_string())?
                    .0
            }
            None => crate::project::layers::LayerSet::default(),
        };
    let allows_overhangs = |label: Option<&str>| layer_set.allows_overhangs(label);

    let export =
        crate::interp::level2::export(&document, &geometry, spacing, &args.user, &allows_overhangs)
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
    // A slice of one: the writers take several so a group download can
    // concatenate them, and a single export is that with one element.
    let exports = [export];
    let serialized = match extension.as_str() {
        "csv" => crate::interp::writer::to_csv(&exports),
        "geojson" | "json" => crate::interp::writer::to_geojson(&exports, &output_crs)?,
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

    let export = &exports[0];
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

fn project_init_command(args: &ProjectInitArgs) -> Result<(), String> {
    let project = crate::project::Project::init(&args.path, args.name.as_deref())
        .map_err(|e| e.to_string())?;
    println!("Created project at {}", project.root().display());
    println!(
        "  {} names it; interpretations go in {}/, layer definitions in {}/",
        crate::project::MARKER,
        crate::project::INTERPRETATIONS_DIR,
        crate::project::LAYERS_DIR,
    );
    println!(
        "  Put processed radargrams in {}/ (or point [radargrams] roots elsewhere).",
        crate::project::DEFAULT_RADARGRAM_DIR
    );
    Ok(())
}

fn project_info_command(args: &ProjectInfoArgs) -> Result<(), String> {
    let Some(project) = crate::project::Project::discover(&args.path).map_err(|e| e.to_string())?
    else {
        return Err(format!(
            "No Ridal project at or above {}. Run `ridal project init` to create one.",
            args.path.display()
        ));
    };

    println!("Project: {}", project.root().display());
    if let Some(name) = &project.config().project.name {
        println!("Name: {name}");
    }
    for root in project.radargram_roots() {
        println!("Radargram root: {}", root.display());
    }
    println!("Cache: {}", project.cache_dir().display());
    // Both halves of the project layer of the preference cascade, so
    // "why does it open like that?" is answerable without the browser.
    println!(
        "Default render profile: {}",
        project
            .default_profile()
            .unwrap_or_else(|| "(unset, Ridal's built-in default)".to_string())
    );
    println!(
        "Default horizontal scale: {}",
        match project.default_xscale() {
            Some(scale) => format!("{scale}x"),
            None => "(unset, 1x)".to_string(),
        }
    );

    let (layers, _) =
        crate::project::layers::read(project.documents()).map_err(|e| e.to_string())?;
    println!("Layers: {}", layers.layers.len());
    for layer in &layers.layers {
        println!("  {} ({})", layer.id, layer.name);
    }

    // Listed from the interpretations directory rather than from the
    // catalog: an interpretation whose radargram is missing is exactly the
    // thing worth noticing, and inspecting must not need the server feature.
    let interpretations = project.root().join(crate::project::INTERPRETATIONS_DIR);
    let mut total = 0usize;
    if let Ok(entries) = std::fs::read_dir(&interpretations) {
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        for name in names {
            let Ok(radargram) = crate::identity::RadargramId::new(&name) else {
                continue;
            };
            let users =
                crate::project::interpretations::list_users(project.documents(), &radargram)
                    .map_err(|e| e.to_string())?;
            if !users.is_empty() {
                println!("Interpretations: {name} <- {}", users.join(", "));
                total += users.len();
            }
            for user in &users {
                let Ok(user_id) = crate::identity::UserId::new(user.as_str()) else {
                    continue;
                };
                let Some(stored) = crate::project::interpretations::read(
                    project.documents(),
                    &radargram,
                    &user_id,
                )
                .map_err(|e| e.to_string())?
                else {
                    continue;
                };
                // Labels with no definition are worth surfacing: the picks
                // are real, the vocabulary just does not describe them, and
                // the viewer will draw them with no colour.
                let labels = stored.document.features.iter().filter_map(|f| f.label());
                let unknown = layers.unknown_ids(labels);
                if !unknown.is_empty() {
                    println!(
                        "  warning: {name}/{user} uses undefined layer(s): {}",
                        unknown.join(", ")
                    );
                }
            }
        }
    }
    if total == 0 {
        println!("Interpretations: none yet");
    }

    // Said here too, because "who can reach this" is the first question
    // anyone asks about a project they are about to serve, and the answer
    // for a project with no accounts is "everyone who can reach the port".
    match crate::project::users::read(project.documents()).map_err(|e| e.to_string())? {
        Some((set, _)) => {
            println!("Accounts: {}", set.users.len());
            for user in &set.users {
                let state = if user.is_activated() {
                    "active"
                } else if user.invite.is_some() {
                    "invited"
                } else {
                    "no password, no invite"
                };
                println!(
                    "  {} ({}, downloads: {}, {state})",
                    user.name, user.role, user.download
                );
            }
            println!(
                "Public read: {}",
                if set.require_auth_to_read {
                    "no, a login is required"
                } else {
                    "yes"
                }
            );
        }
        None => println!(
            "Accounts: none -- everyone is '{}'. Create the first with \
             `ridal project user add <name> --role admin`.",
            crate::identity::DEFAULT_USER
        ),
    }
    Ok(())
}

/// Open the project containing `path`, or say how to make one.
fn open_project(path: &std::path::Path) -> Result<crate::project::Project, String> {
    crate::project::Project::discover(path)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| {
            format!(
                "No Ridal project at or above {}. Run `ridal project init` to create one.",
                path.display()
            )
        })
}

fn parse_role(value: &str) -> Result<crate::project::users::Role, String> {
    crate::project::users::Role::parse(value)
}

fn parse_download(value: &str) -> Result<crate::project::users::DownloadScope, String> {
    crate::project::users::DownloadScope::parse(value)
}

/// Print an invite link the way it can actually be used.
///
/// The path, plus an example of what to prefix it with. This command has no
/// idea what address the server will be reached on -- it may not even be
/// running -- so inventing a hostname would be inventing one, and a link
/// that looks authoritative and is wrong is worse than one that is
/// obviously a fragment.
fn print_invite(name: &str, token: &str, expires: i64) {
    let days = crate::project::users::INVITE_TTL_DAYS;
    println!("Invite link for '{name}' (valid {days} days, single use):");
    println!("  /invite/{token}");
    println!("Prefix it with the address the server is reached on, e.g.");
    println!("  http://localhost:8000/invite/{token}");
    if let Some(when) = chrono::DateTime::from_timestamp(expires, 0) {
        println!("Expires {}", when.format("%Y-%m-%d %H:%M UTC"));
    }
    println!();
    println!(
        "This is the only time it is shown -- only its hash is stored. It is as \
         sensitive as a password until it is used or expires, so send it the way \
         you would send one."
    );
}

fn project_user_add_command(args: &ProjectUserAddArgs) -> Result<(), String> {
    let project = open_project(&args.path)?;
    let name = crate::identity::UserId::new(args.name.clone())?;
    let role = parse_role(&args.role)?;
    let download = parse_download(&args.download)?;

    let existing =
        crate::project::users::is_configured(project.documents()).map_err(|e| e.to_string())?;
    let (token, invite) = crate::project::users::mint_invite(chrono::Utc::now().timestamp())
        .map_err(|e| e.to_string())?;

    crate::project::users::update(project.documents(), |set| {
        if set.get(&name).is_some() {
            return Err(crate::project::users::UserError::Duplicate(
                name.to_string(),
            ));
        }
        let mut user = crate::project::users::User::new(name.clone(), role, download);
        user.invite = Some(invite.clone());
        set.users.push(user);
        Ok(())
    })
    .map_err(|e| e.to_string())?;

    println!("Created '{name}' as {role} (downloads: {download}).");
    println!();
    print_invite(name.as_str(), &token, invite.expires);

    // The moment a project stops being open, which is a bigger change than
    // "one account exists" and is worth saying out loud once.
    if !existing {
        println!();
        println!(
            "This project now requires authentication. Anyone who was writing as \
             '{}' will need an account; their existing interpretations are \
             untouched and still stored under that name. Restart the server for \
             the change to take effect.",
            crate::identity::DEFAULT_USER
        );
    }
    Ok(())
}

fn project_user_list_command(args: &ProjectUserListArgs) -> Result<(), String> {
    let project = open_project(&args.path)?;
    let Some((set, _)) =
        crate::project::users::read(project.documents()).map_err(|e| e.to_string())?
    else {
        println!(
            "This project has no accounts, so everyone using its server is '{}'.",
            crate::identity::DEFAULT_USER
        );
        println!("Create the first with `ridal project user add <name> --role admin`.");
        return Ok(());
    };

    if set.users.is_empty() {
        println!("No accounts. Nobody can sign in, including to create one.");
        println!("Add one with `ridal project user add <name> --role admin`.");
    }
    for user in &set.users {
        let state = if user.is_activated() {
            if user.invite.is_some() {
                "active, reset pending"
            } else {
                "active"
            }
        } else if user.invite.is_some() {
            "invited, not yet activated"
        } else {
            "no password and no invite -- issue one with `ridal project user reset`"
        };
        println!(
            "{}\t{}\tdownloads: {}\t{state}",
            user.name, user.role, user.download
        );
    }
    println!();
    println!(
        "Public read: {}",
        if set.require_auth_to_read {
            "no, a login is required"
        } else {
            "yes"
        }
    );
    println!("Anonymous downloads: {}", set.anonymous_download);
    Ok(())
}

fn project_user_set_command(args: &ProjectUserSetArgs) -> Result<(), String> {
    if args.role.is_none() && args.download.is_none() {
        return Err("Nothing to change. Pass --role, --download, or both.".to_string());
    }
    let project = open_project(&args.path)?;
    let name = crate::identity::UserId::new(args.name.clone())?;
    let role = args.role.as_deref().map(parse_role).transpose()?;
    let download = args.download.as_deref().map(parse_download).transpose()?;

    let updated = crate::project::users::update(project.documents(), |set| {
        // The same guard the HTTP route applies: demoting the last
        // administrator locks the access settings away from everyone.
        if role.is_some_and(|role| role < crate::project::users::Role::Admin)
            && set
                .get(&name)
                .is_some_and(|user| user.role == crate::project::users::Role::Admin)
            && !set.has_another_admin(&name)
        {
            return Err(crate::project::users::UserError::Rejected(format!(
                "'{name}' is the only administrator. Promote someone else first."
            )));
        }
        let user = set
            .get_mut(&name)
            .ok_or_else(|| crate::project::users::UserError::NotFound(name.to_string()))?;
        let mut changed = false;
        if let Some(role) = role {
            changed |= user.role != role;
            user.role = role;
        }
        if let Some(download) = download {
            changed |= user.download != download;
            user.download = download;
        }
        // Bumped so the change reaches an already signed-in person on their
        // next request rather than when their cookie ages out.
        if changed {
            user.credential_version += 1;
        }
        Ok(user.clone())
    })
    .map_err(|e| e.to_string())?;

    println!(
        "'{}' is now {} (downloads: {}).",
        updated.name, updated.role, updated.download
    );
    if updated.credential_version > 1 {
        println!("Any session they had open has been signed out.");
    }
    Ok(())
}

fn project_user_reset_command(args: &ProjectUserResetArgs) -> Result<(), String> {
    let project = open_project(&args.path)?;
    let name = crate::identity::UserId::new(args.name.clone())?;
    let (token, invite) = crate::project::users::mint_invite(chrono::Utc::now().timestamp())
        .map_err(|e| e.to_string())?;

    crate::project::users::update(project.documents(), |set| {
        let user = set
            .get_mut(&name)
            .ok_or_else(|| crate::project::users::UserError::NotFound(name.to_string()))?;
        // Replaces any outstanding invite rather than adding one, so
        // "send another link" cannot leave two live tokens for one account.
        user.invite = Some(invite.clone());
        Ok(())
    })
    .map_err(|e| e.to_string())?;

    print_invite(name.as_str(), &token, invite.expires);
    println!();
    println!(
        "Their current password keeps working until this link is used. \
         Redeeming it sets a new one and signs out any session they had open."
    );
    Ok(())
}

fn project_user_remove_command(args: &ProjectUserRemoveArgs) -> Result<(), String> {
    let project = open_project(&args.path)?;
    let name = crate::identity::UserId::new(args.name.clone())?;

    crate::project::users::update(project.documents(), |set| {
        let Some(user) = set.get(&name) else {
            return Err(crate::project::users::UserError::NotFound(name.to_string()));
        };
        if user.role == crate::project::users::Role::Admin && !set.has_another_admin(&name) {
            return Err(crate::project::users::UserError::Rejected(format!(
                "'{name}' is the only administrator. Promote someone else first."
            )));
        }
        set.users.retain(|user| user.name != name);
        Ok(())
    })
    .map_err(|e| e.to_string())?;

    let _ = crate::project::preferences::remove(project.documents(), &name);

    println!("Removed the account '{name}'.");
    // Worth stating rather than leaving to be discovered: a departed user's
    // picks are attributed scientific data and the account going away does
    // not unmake them.
    println!(
        "Their interpretations are kept, still stored under '{name}'. Only the \
         account and their personal settings are gone."
    );
    Ok(())
}
