/* The project settings page.
 *
 * First-party, embedded via assets.rs, loaded after app.js. Deliberately
 * NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does `rm -rf` on
 * that directory.
 *
 * Wrapped in an IIFE so it declares nothing globally; `assets.rs` has a
 * test that fails if any two scripts on a page collide.
 */
(function () {
  "use strict";

  const form = document.getElementById("settings-form");
  if (!form) return; // Not a project: the page is an explanation, not a form.

  const select = document.getElementById("default-profile");
  const xscale = document.getElementById("default-xscale");
  const status = document.getElementById("settings-status");
  const errorBox = document.getElementById("settings-error");

  let writable = false;

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = false;
  };
  const clearError = () => {
    errorBox.hidden = true;
    errorBox.textContent = "";
  };
  const setStatus = (message) => {
    if (status) status.textContent = message;
  };

  async function load() {
    clearError();
    let settings;
    try {
      settings = await RIDAL.fetchJson("/api/v1/project/settings");
    } catch (error) {
      showError(`Could not load settings: ${error.message}`);
      return;
    }
    writable = Boolean(settings.writable);

    // The empty option is a real choice, not a placeholder: it clears the
    // setting rather than storing the name of the built-in profile, so a
    // project that never chose one stays that way in the file.
    const options = [new Option("Ridal default", "")];
    for (const name of settings.profiles || []) {
      options.push(new Option(name, name));
    }
    select.replaceChildren(...options);
    select.value = settings.default_profile || "";
    select.disabled = !writable;

    // Offered factors come from the server, so a stored default always has
    // an entry to select. No empty option here: unlike a profile name, 1x
    // *is* the neutral value, so "no preference" and "1x" are the same
    // choice and offering both would be a distinction without a difference.
    xscale.replaceChildren(
      ...(settings.xscales || []).map((s) => new Option(s.label, s.text)),
    );
    xscale.value = String(settings.default_xscale || 1);
    xscale.disabled = !writable;
    setStatus("");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!writable) return;
    clearError();
    setStatus("Saving…");

    try {
      const response = await fetch("/api/v1/project/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          default_profile: select.value || null,
          default_xscale: Number(xscale.value) || null,
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        showError(body?.error?.message || `Could not save (${response.status}).`);
        setStatus("");
        return;
      }
      const saved = await response.json();
      select.value = saved.default_profile || "";
      // 1x is stored as absent, so read it back the same way it was sent.
      xscale.value = String(saved.default_xscale || 1);
      // Naming the file is the point: the change lands somewhere the user
      // can go and look at, which is not obvious from a dropdown.
      setStatus("Saved to ridal.toml");
    } catch (error) {
      showError(`Could not save: ${error.message}`);
      setStatus("");
    }
  });

  load();
})();
