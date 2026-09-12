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
    /// Serve a project without accepting writes: cap every caller at
    /// `viewer`, whatever their account says.
    pub read_only: bool,
    /// Accept password logins while bound to a non-loopback address.
    pub allow_insecure_login: bool,
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
    let accounts = match project.as_ref() {
        Some(project) => {
            crate::project::users::is_configured(project.documents()).map_err(|e| e.to_string())?
        }
        None => false,
    };
    let writable = project.is_some() && !options.read_only;

    check_bind_safety(
        options.host,
        writable,
        accounts,
        options.allow_insecure_login,
    )?;

    let state = Arc::new(AppState::build_with_project(
        root,
        &config,
        project,
        options.read_only,
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
    match (state.project.as_ref(), !state.read_only) {
        (Some(project), true) => {
            if accounts {
                println!("Project {} (authenticated)", project.root().display());
            } else {
                println!(
                    "Project {} (writable, no accounts -- everyone is '{}')",
                    project.root().display(),
                    crate::identity::DEFAULT_USER
                );
            }
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
            // Always loopback, so neither bind question can arise.
            allow_insecure_login: false,
        },
        config,
    ))
}

/// `ridal server start`: deployment-oriented mode. Loopback by default;
/// remote binding is explicit, and what a remote bind may serve is decided
/// by [`check_bind_safety`].
#[allow(clippy::too_many_arguments)]
pub fn run_server_start(
    root: &Path,
    host: IpAddr,
    port: u16,
    open_browser: bool,
    read_only: bool,
    allow_insecure_login: bool,
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
            allow_insecure_login,
        },
        config,
    ))
}

/// What a bind address may serve.
///
/// Two questions, both keyed on the bind address rather than on the
/// connection, because **Ridal will essentially never see HTTPS**: behind a
/// TLS-terminating proxy it sees plain HTTP on loopback, which is correct
/// and safe, so "is this connection TLS?" always answers no and is useless
/// as a guardrail. A loopback bind means either it is genuinely local, or
/// there is a proxy in front; a non-loopback bind is a decision about what
/// the operator has put on the network.
///
/// 1. **Unauthenticated writes.** A project with no accounts treats everyone
///    as the local default user, which is how Ridal behaved before #131 and
///    is exactly right on loopback. Exposed to a network it means anyone who
///    can reach the port can rewrite the picks. This used to be waved
///    through with `--allow-remote-writes`; there is now a better answer
///    than a flag, so the refusal names it -- create an administrator, and
///    the writes become authenticated rather than merely permitted.
///
/// 2. **Passwords in cleartext.** Once there *are* accounts, a login on a
///    non-loopback bind puts a password on the wire in the clear. That one
///    does still need a flag, because the operator may legitimately have a
///    TLS proxy that Ridal cannot see. `--allow-insecure-login` reads as "I
///    have put this on the network and I accept what is in front of it".
fn check_bind_safety(
    host: IpAddr,
    writable: bool,
    accounts: bool,
    allow_insecure_login: bool,
) -> Result<(), String> {
    if host.is_loopback() {
        return Ok(());
    }
    if writable && !accounts {
        return Err(format!(
            "Refusing to accept writes on {host}: this project has no accounts, so \
             everyone who can reach this address would be the '{}' user and could \
             modify interpretations. Create an administrator with `ridal project \
             user add <name> --role admin`, or start with --read-only, or bind \
             loopback behind a reverse proxy.",
            crate::identity::DEFAULT_USER
        ));
    }
    if accounts && !allow_insecure_login {
        return Err(format!(
            "Refusing to accept password logins on {host}: Ridal does not terminate \
             TLS, so a password sent to this address travels in the clear unless \
             something in front of it is doing so. Bind loopback behind a \
             TLS-terminating reverse proxy -- the supported way to serve this \
             remotely -- or re-run with --allow-insecure-login to accept what is \
             in front of this address."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_bind_safety;
    use std::net::IpAddr;

    fn ip(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    #[test]
    fn loopback_may_do_anything() {
        // Either it is genuinely local, or there is a TLS proxy in front.
        // This covers `ridal gui` and every sane remote deployment.
        for accounts in [false, true] {
            assert!(check_bind_safety(ip("127.0.0.1"), true, accounts, false).is_ok());
            assert!(check_bind_safety(ip("::1"), true, accounts, false).is_ok());
        }
    }

    #[test]
    fn a_remote_bind_with_no_accounts_refuses_writes_and_says_how_to_fix_it() {
        // The systemd case: `--host 0.0.0.0` on a machine other people can
        // reach, with nothing authenticating anywhere in the stack. The
        // answer is no longer a flag that waves it through -- it is to
        // create an administrator, so the writes become authenticated.
        let error = check_bind_safety(ip("0.0.0.0"), true, false, false).unwrap_err();
        assert!(error.contains("ridal project user add"), "{error}");
        assert!(error.contains("--read-only"), "{error}");
        assert!(check_bind_safety(ip("192.168.1.10"), true, false, false).is_err());

        // And the flag that used to exist cannot buy its way past it.
        assert!(check_bind_safety(ip("0.0.0.0"), true, false, true).is_err());
    }

    #[test]
    fn a_remote_bind_with_accounts_refuses_cleartext_passwords_unless_told_to() {
        let error = check_bind_safety(ip("0.0.0.0"), true, true, false).unwrap_err();
        assert!(error.contains("--allow-insecure-login"), "{error}");
        assert!(error.contains("reverse proxy"), "{error}");

        // The flag reads as "I accept what is in front of this address",
        // which is the honest shape: Ridal cannot see the TLS proxy that
        // makes this fine.
        assert!(check_bind_safety(ip("0.0.0.0"), true, true, true).is_ok());
    }

    #[test]
    fn a_read_only_server_with_no_accounts_may_bind_anywhere() {
        // Nothing to protect and nothing to log in to: this is the
        // public-catalog arrangement Ridal already had.
        assert!(check_bind_safety(ip("0.0.0.0"), false, false, false).is_ok());
    }

    #[test]
    fn a_read_only_server_with_accounts_still_guards_the_password() {
        // Read-only caps what a session can *do*, but signing in still
        // sends a password, and that is what this guard is about.
        assert!(check_bind_safety(ip("0.0.0.0"), false, true, false).is_err());
        assert!(check_bind_safety(ip("0.0.0.0"), false, true, true).is_ok());
    }
}
