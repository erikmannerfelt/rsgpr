/* Signing in, signing out, and redeeming an invite.
 *
 * First-party, embedded via assets.rs, loaded after app.js. Deliberately
 * NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does `rm -rf` on
 * that directory.
 *
 * One file for three forms because they are the same interaction: post
 * credentials, let the server set a cookie, land on the catalog. Only one
 * of the three is ever on the page.
 *
 * Wrapped in an IIFE so it declares nothing globally; `assets.rs` has a
 * test that fails if any two scripts on a page collide.
 */
(function () {
  "use strict";

  const loginForm = document.getElementById("login-form");
  const logoutForm = document.getElementById("logout-form");
  const inviteForm = document.getElementById("invite-form");

  /* Shared plumbing for whichever form is present. `errorBox` and `status`
   * differ by page, so they are looked up rather than assumed. */
  const box = (id) => document.getElementById(id);

  const show = (errorBox, message) => {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.hidden = false;
  };
  const clear = (errorBox) => {
    if (!errorBox) return;
    errorBox.hidden = true;
    errorBox.textContent = "";
  };
  const setStatus = (element, message) => {
    if (element) element.textContent = message;
  };

  /* Post JSON and return the parsed body, or throw with the server's own
   * message. Deliberately not RIDAL.fetchJson: these responses matter for
   * their Set-Cookie header as much as their body, and a failed sign-in is
   * an expected outcome to render rather than an error to log. */
  async function post(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const parsed = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        parsed?.error?.message || `Request failed (${response.status}).`,
      );
    }
    return parsed;
  }

  if (loginForm) {
    const errorBox = box("login-error");
    const status = box("login-status");
    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clear(errorBox);
      setStatus(status, "Signing in…");
      try {
        await post("/api/v1/auth/login", {
          name: box("login-name").value.trim(),
          password: box("login-password").value,
        });
        // Replace rather than assign: the login page should not sit in the
        // history behind the catalog, where Back would return to a form
        // that is now pointless.
        window.location.replace("/");
      } catch (error) {
        show(errorBox, error.message);
        setStatus(status, "");
        box("login-password").value = "";
        box("login-password").focus();
      }
    });
  }

  if (logoutForm) {
    const status = box("login-status");
    logoutForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setStatus(status, "Signing out…");
      try {
        await post("/api/v1/auth/logout", {});
      } catch {
        // Signing out has one sensible outcome, and the cookie is cleared
        // by the response either way. Reloading shows whatever is true.
      }
      window.location.replace("/login");
    });
  }

  if (inviteForm) {
    const errorBox = box("invite-error");
    const status = box("invite-status");
    inviteForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clear(errorBox);

      const password = box("invite-password").value;
      // Checked here rather than left to the server: a mistyped
      // confirmation is not something the server can see, and a round trip
      // to be told so would consume nothing but time.
      if (password !== box("invite-confirm").value) {
        show(errorBox, "The two passwords do not match.");
        return;
      }

      setStatus(status, "Setting your password…");
      try {
        await post("/api/v1/auth/invite", {
          token: inviteForm.dataset.token,
          password,
        });
        window.location.replace("/");
      } catch (error) {
        show(errorBox, error.message);
        setStatus(status, "");
      }
    });
  }
})();
