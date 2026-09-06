//! `ridal gui` and `ridal server start` launch modes (#120). Both use the
//! same [`crate::server::app::build_router`] application; only bind
//! behavior, port selection, and browser-opening differ between them.

use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

use super::app::AppState;
use super::render::service::RenderServiceConfig;

pub struct LaunchOptions {
    pub host: IpAddr,
    /// `0` requests an OS-assigned ephemeral port.
    pub port: u16,
    pub open_browser: bool,
    /// Serve a project without accepting writes.
    pub read_only: bool,
    /// Permit writes while bound to a non-loopback address.
    pub allow_remote_writes: bool,
}

async fn serve(
    root: &Path,
    options: LaunchOptions,
    config: RenderServiceConfig,
) -> Result<(), String> {
    // A project is found by searching upwards, so pointing Ridal at a
    // subdirectory or at a single file inside a project still saves
    // interpretations to the right place.
    let project = crate::project::Project::discover(root).map_err(|e| e.to_string())?;
    let writable = project.is_some() && !options.read_only;

    check_write_safety(options.host, writable, options.allow_remote_writes)?;

    let state = Arc::new(AppState::build_with_project(
        root, &config, project, writable,
    )?);
    if !state.catalog.warnings.is_empty() {
        for w in &state.catalog.warnings {
            eprintln!("Warning: {}", w.message);
        }
    }
    println!(
        "Discovered {} radargram(s) under {}",
        state.catalog.entries.len(),
        root.display()
    );
    match (state.project.as_ref(), state.writable) {
        (Some(project), true) => {
            println!("Project {} (writable)", project.root().display());
            // Any configured radargram root outside the served tree is not
            // scanned yet. Saying so is better than a config key that looks
            // honoured and is not.
            for extra in project.radargram_roots() {
                if !extra.starts_with(project.root()) {
                    eprintln!(
                        "Warning: radargram root {} is outside the project and is not \
                         scanned yet; only the project tree is indexed.",
                        extra.display()
                    );
                }
            }
        }
        (Some(project), false) => {
            println!("Project {} (read-only)", project.root().display())
        }
        (None, _) => println!("No project here; interpretations cannot be saved."),
    }

    let router = super::app::build_router(state);
    let addr = SocketAddr::new(options.host, options.port);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("Failed to bind {addr}: {e}"))?;
    let bound_addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to read bound address: {e}"))?;
    let url = format!("http://{bound_addr}");
    println!("Serving on {url}");

    if options.open_browser {
        println!("{url}");
        if let Err(e) = webbrowser::open(&url) {
            eprintln!("Warning: could not open a browser automatically: {e}");
        }
    }

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("Server error: {e}"))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    println!("Shutting down.");
}

/// `ridal gui`: local convenience mode. Binds loopback only, selects an
/// available port, and opens a browser -- a failure to open the browser
/// is a warning, never a reason to stop the server (#120).
pub fn run_gui(root: &Path, read_only: bool, config: RenderServiceConfig) -> Result<(), String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to start async runtime: {e}"))?;
    runtime.block_on(serve(
        root,
        LaunchOptions {
            host: IpAddr::from([127, 0, 0, 1]),
            port: 0,
            open_browser: true,
            read_only,
            // Always loopback, so the remote-write question cannot arise.
            allow_remote_writes: false,
        },
        config,
    ))
}

/// `ridal server start`: deployment-oriented mode. Loopback by default;
/// remote binding is explicit, and this milestone deliberately implements
/// no authentication -- documented as future work (#120), not silently
/// assumed safe.
#[allow(clippy::too_many_arguments)]
pub fn run_server_start(
    root: &Path,
    host: IpAddr,
    port: u16,
    open_browser: bool,
    read_only: bool,
    allow_remote_writes: bool,
    config: RenderServiceConfig,
) -> Result<(), String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to start async runtime: {e}"))?;
    runtime.block_on(serve(
        root,
        LaunchOptions {
            host,
            port,
            open_browser,
            read_only,
            allow_remote_writes,
        },
        config,
    ))
}

/// Refuse to accept writes from a network the server cannot authenticate.
///
/// An interim guardrail, not a security model. Ridal has no authentication,
/// so a writable server bound to a non-loopback address is editable by
/// anyone who can reach it -- and that is precisely the systemd deployment
/// case, where the mistake is easiest to make and hardest to notice. One
/// explicit flag turns it from an accident into a decision.
///
/// **This function should be deleted when real authentication lands**, not
/// extended. It is not a permission system and must not grow into one.
fn check_write_safety(
    host: IpAddr,
    writable: bool,
    allow_remote_writes: bool,
) -> Result<(), String> {
    if !writable || host.is_loopback() || allow_remote_writes {
        return Ok(());
    }
    Err(format!(
        "Refusing to accept writes on {host}: Ridal has no authentication yet, so \
         anyone who can reach this address could modify interpretations. Re-run \
         with --allow-remote-writes to accept that, with --read-only to serve the \
         project without writes, or bind loopback and put a reverse proxy that \
         authenticates in front."
    ))
}

#[cfg(test)]
mod tests {
    use super::check_write_safety;
    use std::net::IpAddr;

    fn ip(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    #[test]
    fn loopback_may_accept_writes() {
        assert!(check_write_safety(ip("127.0.0.1"), true, false).is_ok());
        assert!(check_write_safety(ip("::1"), true, false).is_ok());
    }

    #[test]
    fn a_remote_bind_refuses_writes_by_default() {
        // The systemd case: `--host 0.0.0.0` on a machine other people can
        // reach, with no authentication anywhere in the stack.
        let error = check_write_safety(ip("0.0.0.0"), true, false).unwrap_err();
        assert!(error.contains("--allow-remote-writes"), "{error}");
        assert!(error.contains("--read-only"), "{error}");
        assert!(check_write_safety(ip("192.168.1.10"), true, false).is_err());
    }

    #[test]
    fn a_remote_bind_is_fine_once_it_is_a_decision() {
        assert!(check_write_safety(ip("0.0.0.0"), true, true).is_ok());
    }

    #[test]
    fn a_read_only_server_may_bind_anywhere() {
        // Nothing to protect: this is the behaviour Ridal already had.
        assert!(check_write_safety(ip("0.0.0.0"), false, false).is_ok());
    }
}
