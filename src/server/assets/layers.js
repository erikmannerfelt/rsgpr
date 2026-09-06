/* Layer definition management (the /layers page).
 *
 * First-party, embedded via assets.rs and loaded after app.js, so `RIDAL`
 * exists. Deliberately NOT under assets/vendor/ -- scripts/vendor_leaflet.sh
 * does `rm -rf` on that directory.
 *
 * The whole vocabulary is one document, so every edit is a full PUT of the
 * layer list carrying the ETag the page last read. That is why `etag` is
 * tracked here rather than being recomputed: it is what stops this page from
 * silently discarding a change made in another tab between load and save.
 */

const WRITABLE = window.RIDAL_LAYERS.writable;

const table = document.querySelector("#layers-table tbody");
const emptyNote = document.getElementById("layers-empty");
const errorBox = document.getElementById("layers-error");

let layers = [];
let etag = null;
let usage = { counts: {}, undefined: {} };

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

/** Build one row. Text goes in via textContent, never innerHTML: layer
 * names and descriptions are free-text fields a user typed. */
function renderRow(layer, index) {
  const row = document.createElement("tr");

  const swatchCell = document.createElement("td");
  if (WRITABLE) {
    const picker = document.createElement("input");
    picker.type = "color";
    picker.value = layer.color || "#888888";
    picker.setAttribute("aria-label", `Colour for ${layer.id}`);
    picker.addEventListener("change", () => {
      layers[index].color = picker.value;
      save();
    });
    swatchCell.appendChild(picker);
  } else {
    const swatch = document.createElement("span");
    swatch.className = "layer-swatch";
    swatch.style.background = layer.color || "#888888";
    swatchCell.appendChild(swatch);
  }
  row.appendChild(swatchCell);

  const idCell = document.createElement("td");
  const code = document.createElement("code");
  // Immutable by design: every stored pick refers to a layer by this
  // string, so changing it here would orphan them all silently.
  code.textContent = layer.id;
  idCell.appendChild(code);
  row.appendChild(idCell);

  row.appendChild(editableCell(layer, index, "name", "Name"));
  row.appendChild(editableCell(layer, index, "description", "Description"));

  const overhangCell = document.createElement("td");
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.checked = Boolean(layer.allow_overhangs);
  toggle.disabled = !WRITABLE;
  toggle.setAttribute("aria-label", `Allow overhangs in ${layer.id}`);
  toggle.addEventListener("change", () => {
    layers[index].allow_overhangs = toggle.checked;
    save();
  });
  overhangCell.appendChild(toggle);
  row.appendChild(overhangCell);

  const usageCell = document.createElement("td");
  const count = usage.counts[layer.id] || 0;
  usageCell.textContent = count === 1 ? "1 feature" : `${count} features`;
  row.appendChild(usageCell);

  const actionCell = document.createElement("td");
  if (WRITABLE) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => deleteLayer(index));
    actionCell.appendChild(remove);
  }
  row.appendChild(actionCell);

  return row;
}

/** A cell that becomes the field's value on blur. Name and description are
 * cosmetic, so editing them in place needs no confirmation. */
function editableCell(layer, index, field, label) {
  const cell = document.createElement("td");
  if (!WRITABLE) {
    cell.textContent = layer[field] || "";
    return cell;
  }
  const input = document.createElement("input");
  input.type = "text";
  input.value = layer[field] || "";
  input.setAttribute("aria-label", `${label} for ${layer.id}`);
  input.addEventListener("change", () => {
    const value = input.value.trim();
    layers[index][field] = value === "" ? undefined : value;
    save();
  });
  cell.appendChild(input);
  return cell;
}

function render() {
  table.replaceChildren(...layers.map(renderRow));
  emptyNote.hidden = layers.length > 0;

  const undefinedNames = Object.keys(usage.undefined);
  const section = document.getElementById("undefined-labels");
  const list = document.getElementById("undefined-list");
  section.hidden = undefinedNames.length === 0;
  list.replaceChildren(
    ...undefinedNames.map((name) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = name;
      item.appendChild(code);
      const count = usage.undefined[name];
      item.appendChild(
        document.createTextNode(
          count === 1 ? " - 1 feature" : ` - ${count} features`,
        ),
      );
      return item;
    }),
  );
}

/** Deleting a layer definition leaves its picks alone; they simply lose
 * their colour. Saying so, with the count, is the difference between an
 * informed decision and a scary one. */
function deleteLayer(index) {
  const layer = layers[index];
  const count = usage.counts[layer.id] || 0;
  const consequence =
    count === 0
      ? "No picks use it."
      : `${count} picked feature(s) use it. They will be kept, but will lose their colour until a layer with this id exists again.`;
  if (!window.confirm(`Delete the layer "${layer.id}"?\n\n${consequence}`)) {
    return;
  }
  layers.splice(index, 1);
  save();
}

async function save() {
  clearError();
  const headers = { "Content-Type": "application/json" };
  if (etag) {
    // Refuses rather than clobbers if another tab saved in between.
    headers["If-Match"] = etag;
  }
  try {
    const response = await fetch("/api/v1/layers", {
      method: "PUT",
      headers,
      body: JSON.stringify(layers),
    });
    if (response.status === 412) {
      showError(
        "These layers were changed somewhere else while this page was open. " +
          "Reload to see the current definitions, then reapply your change.",
      );
      return;
    }
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      showError(body?.error?.message || `Could not save layers (${response.status}).`);
      // Re-read so the page shows what is actually stored rather than the
      // rejected edit.
      await load();
      return;
    }
    etag = response.headers.get("ETag");
    const body = await response.json();
    layers = body.layers;
    await loadUsage();
    render();
  } catch (error) {
    showError(`Could not save layers: ${error.message}`);
  }
}

async function loadUsage() {
  try {
    usage = await RIDAL.fetchJson("/api/v1/layers/usage");
  } catch (error) {
    // Usage is advisory. Losing it must not stop the page working, so it
    // degrades to zero counts with a console note rather than an error box.
    console.warn(`Could not load layer usage: ${error.message}`);
    usage = { counts: {}, undefined: {} };
  }
}

async function load() {
  clearError();
  try {
    const response = await fetch("/api/v1/layers");
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      showError(body?.error?.message || `Could not load layers (${response.status}).`);
      return;
    }
    etag = response.headers.get("ETag") || null;
    const body = await response.json();
    layers = body.layers || [];
    await loadUsage();
    render();
  } catch (error) {
    showError(`Could not load layers: ${error.message}`);
  }
}

const form = document.getElementById("add-layer");
if (form) {
  /** Why this layer cannot be added, or null if it can.
   *
   * Specific rather than generic: "the id must be lowercase" is actionable,
   * "please match the requested format" is not, and an id is not something
   * the user can guess the rules for. */
  function rejectionReason(id, name) {
    if (id === "") {
      return [
        "A layer needs an id. It is the short name written into every pick " +
          'and exported as the "layer" column, for example "bed".',
        "id",
      ];
    }
    if (!/^[a-z0-9_-]+$/.test(id)) {
      const bad = [...id].find((c) => !/[a-z0-9_-]/.test(c));
      return [
        `The id cannot contain "${bad}". Use lowercase letters, digits, "-" ` +
          "and \"_\" only -- it ends up in exported columns and URLs. Put " +
          "capitals, spaces and punctuation in the name instead.",
        "id",
      ];
    }
    if (layers.some((layer) => layer.id === id)) {
      return [`A layer with the id "${id}" already exists.`, "id"];
    }
    if (name === "") {
      return ["A layer needs a name. This is the label shown in the viewer.", "name"];
    }
    return null;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const id = String(data.get("id") || "").trim();
    const name = String(data.get("name") || "").trim();

    const rejection = rejectionReason(id, name);
    if (rejection) {
      const [message, field] = rejection;
      showError(message);
      form.elements[field].focus();
      return;
    }

    clearError();
    const description = String(data.get("description") || "").trim();
    layers.push({
      id,
      name,
      color: String(data.get("color") || ""),
      description: description === "" ? undefined : description,
      allow_overhangs: data.get("allow_overhangs") === "on",
    });
    form.reset();
    save();
  });
}

load();
