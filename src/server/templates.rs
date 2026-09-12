//! MiniJinja environment with embedded templates (#120).
//!
//! Templates are embedded via `include_str!` at compile time, not read
//! from disk at runtime -- production must not require a separate
//! frontend/template directory alongside the binary.

use minijinja::Environment;

pub fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.add_template("base.html.jinja", include_str!("templates/base.html.jinja"))
        .expect("base template must parse");
    env.add_template(
        "index.html.jinja",
        include_str!("templates/index.html.jinja"),
    )
    .expect("index template must parse");
    env.add_template(
        "viewer.html.jinja",
        include_str!("templates/viewer.html.jinja"),
    )
    .expect("viewer template must parse");
    env.add_template(
        "layers.html.jinja",
        include_str!("templates/layers.html.jinja"),
    )
    .expect("layers template must parse");
    env.add_template(
        "settings.html.jinja",
        include_str!("templates/settings.html.jinja"),
    )
    .expect("settings template must parse");
    env.add_template(
        "login.html.jinja",
        include_str!("templates/login.html.jinja"),
    )
    .expect("login template must parse");
    env.add_template(
        "invite.html.jinja",
        include_str!("templates/invite.html.jinja"),
    )
    .expect("invite template must parse");
    env.add_template(
        "error.html.jinja",
        include_str!("templates/error.html.jinja"),
    )
    .expect("error template must parse");
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_templates_load_without_error() {
        // environment() itself panics on a template error, so simply
        // constructing it is the test; this also confirms each named
        // template is actually registered under the name routes.rs uses.
        let env = environment();
        for name in [
            "base.html.jinja",
            "index.html.jinja",
            "viewer.html.jinja",
            "layers.html.jinja",
            "settings.html.jinja",
            "login.html.jinja",
            "invite.html.jinja",
            "error.html.jinja",
        ] {
            assert!(env.get_template(name).is_ok(), "missing template {name}");
        }
    }

    #[test]
    fn error_template_renders_with_expected_context() {
        let env = environment();
        let tmpl = env.get_template("error.html.jinja").unwrap();
        let out = tmpl
            .render(minijinja::context! {
                status => 404,
                code => "dataset_not_found",
                message => "no such dataset",
            })
            .unwrap();
        assert!(out.contains("404"));
        assert!(out.contains("dataset_not_found"));
        assert!(out.contains("no such dataset"));
    }
}
