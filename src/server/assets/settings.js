/* The settings page: my preferences, the project defaults, and access.
 *
 * First-party, embedded via assets.rs, loaded after app.js. Deliberately
 * NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does `rm -rf` on
 * that directory.
 *
 * Three sections in one file because they share one load: the settings
 * endpoint already answers what every one of them needs to render, and
 * splitting them would mean three requests to draw one page.
 *
 * Wrapped in an IIFE so it declares nothing globally; `assets.rs` has a
 * test that fails if any two scripts on a page collide.
 */
(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);

  const projectForm = byId("settings-form");
  const myForm = byId("my-settings-form");
  const accessSection = byId("access-section");
  if (!projectForm && !myForm && !accessSection) return; // Not a project.

  const errorBox = byId("settings-error");

  let canEditProject = false;
  let canEditAccess = false;
  let xscales = [];
  let profiles = [];

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = false;
  };
  const clearError = () => {
    errorBox.hidden = true;
    errorBox.textContent = "";
  };
  const setStatus = (id, message) => {
    const element = byId(id);
    if (element) element.textContent = message;
  };

  /* Send JSON and surface the server's own message on failure. */
  async function send(method, url, body) {
    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (response.status === 204) return null;
    const parsed = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        parsed?.error?.message || `Request failed (${response.status}).`,
      );
    }
    return parsed;
  }

  /* Fill a <select> with profile names. `emptyLabel` names the "unset"
   * choice, which is a real option rather than a placeholder: it clears
   * the setting instead of storing the name of the fallback, which is what
   * lets a later change to the layer below reach whoever never chose. */
  function fillProfiles(select, emptyLabel, selected) {
    if (!select) return;
    const options = [new Option(emptyLabel, "")];
    for (const name of profiles) options.push(new Option(name, name));
    select.replaceChildren(...options);
    select.value = selected || "";
  }

  /* Offered scales come from the server, so a stored value always has an
   * entry to select. No empty option: unlike a profile name, 1x *is* the
   * neutral value, so "no preference" and "1x" are the same choice and
   * offering both would be a distinction without a difference. */
  function fillScales(select, selected) {
    if (!select) return;
    select.replaceChildren(...xscales.map((s) => new Option(s.label, s.text)));
    select.value = String(selected || 1);
  }

  function fillNames(select, names, selected) {
    if (!select) return;
    select.replaceChildren(...names.map((name) => new Option(name, name)));
    if (selected) select.value = selected;
  }

  async function load() {
    clearError();
    let settings;
    try {
      settings = await RIDAL.fetchJson("/api/v1/project/settings");
    } catch (error) {
      showError(`Could not load settings: ${error.message}`);
      return;
    }
    canEditProject = Boolean(settings.can_edit_project);
    canEditAccess = Boolean(settings.can_edit_access);
    profiles = settings.profiles || [];
    xscales = settings.xscales || [];

    fillProfiles(byId("my-profile"), "Project default", settings.my_profile);
    fillScales(byId("my-xscale"), settings.my_xscale);

    fillProfiles(
      byId("default-profile"),
      "Ridal default",
      settings.default_profile,
    );
    fillScales(byId("default-xscale"), settings.default_xscale);
    const projectProfile = byId("default-profile");
    const projectScale = byId("default-xscale");
    if (projectProfile) projectProfile.disabled = !canEditProject;
    if (projectScale) projectScale.disabled = !canEditProject;

    setStatus("settings-status", "");
    setStatus("my-settings-status", "");

    if (canEditAccess) await loadAccess();
  }

  if (myForm) {
    myForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearError();
      setStatus("my-settings-status", "Saving…");
      try {
        const saved = await send("PUT", "/api/v1/preferences", {
          render_profile: byId("my-profile").value || null,
          x_scale: Number(byId("my-xscale").value) || null,
        });
        byId("my-profile").value = saved.render_profile || "";
        // 1x is stored as absent, so read it back the way it was sent.
        byId("my-xscale").value = String(saved.x_scale || 1);
        setStatus("my-settings-status", "Saved");
      } catch (error) {
        showError(error.message);
        setStatus("my-settings-status", "");
      }
    });
  }

  if (projectForm) {
    projectForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!canEditProject) return;
      clearError();
      setStatus("settings-status", "Saving…");
      try {
        const saved = await send("PUT", "/api/v1/project/settings", {
          default_profile: byId("default-profile").value || null,
          default_xscale: Number(byId("default-xscale").value) || null,
        });
        byId("default-profile").value = saved.default_profile || "";
        byId("default-xscale").value = String(saved.default_xscale || 1);
        // Naming the file is the point: the change lands somewhere the user
        // can go and look at, which is not obvious from a dropdown.
        setStatus("settings-status", "Saved to ridal.toml");
      } catch (error) {
        showError(error.message);
        setStatus("settings-status", "");
      }
    });
  }

  /* ---- Access ------------------------------------------------------- */

  let access = null;

  async function loadAccess() {
    try {
      access = await RIDAL.fetchJson("/api/v1/users");
    } catch (error) {
      showError(`Could not load accounts: ${error.message}`);
      return;
    }

    const addForm = byId("add-user");
    if (addForm) {
      fillNames(addForm.elements.role, access.roles || [], "picker");
      fillNames(
        addForm.elements.download,
        access.download_scopes || [],
        "derived",
      );
    }
    fillNames(
      byId("anonymous-download"),
      access.download_scopes || [],
      access.anonymous_download,
    );
    byId("require-auth").checked = Boolean(access.require_auth_to_read);

    renderUsers();
  }

  function renderUsers() {
    const body = byId("users-table").querySelector("tbody");
    body.replaceChildren();
    for (const user of access.users || []) {
      body.appendChild(userRow(user));
    }
  }

  function userRow(user) {
    const row = document.createElement("tr");

    const name = document.createElement("td");
    name.textContent = user.name;
    row.appendChild(name);

    const role = document.createElement("td");
    const roleSelect = document.createElement("select");
    fillNames(roleSelect, access.roles || [], user.role);
    roleSelect.addEventListener("change", () =>
      updateUser(user.name, { role: roleSelect.value }, roleSelect, user.role),
    );
    role.appendChild(roleSelect);
    row.appendChild(role);

    const download = document.createElement("td");
    const downloadSelect = document.createElement("select");
    fillNames(downloadSelect, access.download_scopes || [], user.download);
    downloadSelect.addEventListener("change", () =>
      updateUser(
        user.name,
        { download: downloadSelect.value },
        downloadSelect,
        user.download,
      ),
    );
    download.appendChild(downloadSelect);
    row.appendChild(download);

    const status = document.createElement("td");
    // Three states worth telling apart: never activated, active, and active
    // with a reset outstanding.
    if (!user.activated) {
      status.textContent = user.invite_pending
        ? "invited, not yet activated"
        : "no password and no invite";
    } else {
      status.textContent = user.invite_pending ? "reset pending" : "active";
    }
    row.appendChild(status);

    const actions = document.createElement("td");
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = user.activated ? "Reset password" : "New invite link";
    reset.addEventListener("click", () => reissue(user.name));
    actions.appendChild(reset);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeUser(user.name));
    actions.appendChild(remove);
    row.appendChild(actions);

    return row;
  }

  async function updateUser(name, change, select, previous) {
    clearError();
    try {
      await send("PUT", `/api/v1/users/${encodeURIComponent(name)}`, change);
      await loadAccess();
    } catch (error) {
      showError(error.message);
      // Put the control back to what the server still believes, so the page
      // never shows a role that was refused.
      select.value = previous;
    }
  }

  function showInvite(name, result) {
    const box = byId("invite-result");
    byId("invite-who").textContent = name;
    byId("invite-days").textContent = String(result.invite_ttl_days);
    // Built from this page's own origin rather than from anything the
    // server guessed: Ridal is normally behind a reverse proxy and has no
    // reliable idea what address the browser reached it on.
    byId("invite-link").textContent =
      window.location.origin + result.invite_path;
    box.hidden = false;
  }

  async function reissue(name) {
    clearError();
    try {
      const result = await send(
        "POST",
        `/api/v1/users/${encodeURIComponent(name)}/invite`,
        {},
      );
      showInvite(name, result);
      await loadAccess();
    } catch (error) {
      showError(error.message);
    }
  }

  async function removeUser(name) {
    // Worth a confirmation, and worth saying what it does not do: the picks
    // are attributed data and stay.
    const confirmed = window.confirm(
      `Remove the account "${name}"?\n\nTheir interpretations are kept — an ` +
        `account going away does not unmake the picks. Only the account and ` +
        `their personal settings are removed.`,
    );
    if (!confirmed) return;
    clearError();
    try {
      await send("DELETE", `/api/v1/users/${encodeURIComponent(name)}`, {});
      await loadAccess();
    } catch (error) {
      showError(error.message);
    }
  }

  const addForm = byId("add-user");
  if (addForm) {
    addForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearError();
      const name = addForm.elements.name.value.trim();
      try {
        const result = await send("POST", "/api/v1/users", {
          name,
          role: addForm.elements.role.value,
          download: addForm.elements.download.value,
        });
        addForm.reset();
        showInvite(name, result);
        await loadAccess();
      } catch (error) {
        showError(error.message);
      }
    });
  }

  const accessForm = byId("access-form");
  if (accessForm) {
    accessForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearError();
      setStatus("access-status", "Saving…");
      try {
        await send("PUT", "/api/v1/access", {
          require_auth_to_read: byId("require-auth").checked,
          anonymous_download: byId("anonymous-download").value,
        });
        setStatus("access-status", "Saved");
      } catch (error) {
        showError(error.message);
        setStatus("access-status", "");
      }
    });
  }

  load();
})();
