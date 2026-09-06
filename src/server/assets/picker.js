/* Picking reflectors on the radargram (#102).
 *
 * First-party, embedded via assets.rs, loaded after viewer.js so `L`,
 * `RIDAL` and `window.RIDAL_VIEWER` all exist. Deliberately NOT under
 * assets/vendor/ -- scripts/vendor_leaflet.sh does `rm -rf` on that
 * directory.
 *
 * Built directly on Leaflet rather than on Leaflet.Draw. The interaction a
 * horizon picker needs is narrow -- extend a line, undo, finish, reassign,
 * split -- while Leaflet.Draw brings a shape palette and a modal toolbar
 * that would mostly be hidden, and has been unmaintained since before
 * Leaflet 1.9. Owning this is cheaper than owning that mismatch, and it
 * lets the overhang guardrail refuse a vertex at click time rather than at
 * save time.
 *
 * Coordinates: the map is L.CRS.Simple over the *viewer raster*, while
 * picks are stored in source trace/sample index space. The conversion is
 * redone here rather than imported from viewer.js, because both are plain
 * scripts with no module boundary between them (#120: no build step).
 */

/* Everything below is wrapped in an IIFE. This is load-bearing, not style:
 * classic scripts share one global scope, so a top-level `const CFG` here
 * collided with viewer.js's own `const CFG` and made this entire file fail
 * to parse -- silently, taking every picking control with it. Nothing in
 * here may be declared at top level; reach other scripts through explicit
 * `window.*` properties instead. `assets.rs` has a test that catches a
 * recurrence. */
(function () {
  "use strict";

  const CFG = window.RIDAL_VIEWER;

  /** Split a line's coordinates at an interior vertex.
   *
   * Returns `[head, tail]`, or `null` if the index is not interior.
   *
   * The split vertex belongs to **both** halves. A horizon split into two
   * lines should still cover every position it covered before; dropping the
   * shared vertex from one side would leave a gap exactly where the user
   * clicked. So the vertex count goes up by one -- that is the split, not a
   * duplicated line.
   *
   * Splitting at an end is refused rather than clamped, because it would
   * leave a one-vertex "line", which is not a line and cannot be exported.
   *
   * Kept as a pure function, separate from the DOM, so its behaviour can be
   * reasoned about on its own.
   */
  function splitCoordinates(coordinates, vertexIndex) {
    if (vertexIndex <= 0 || vertexIndex >= coordinates.length - 1) return null;
    return [
      coordinates.slice(0, vertexIndex + 1),
      coordinates.slice(vertexIndex),
    ];
  }

  if (CFG.writable) {
    initPicker();
  }

  function initPicker() {
    const RASTER_SCALE = CFG.viewerWidth / CFG.sourceWidth;
    const VERTICAL_RASTER_SCALE = CFG.viewerHeight / CFG.sourceHeight;
    const DEFAULT_COLOR = "#ffcc00";

    const map = window.RIDAL_MAP;
    const layerSelect = document.getElementById("pick-layer");
    const toggleButton = document.getElementById("pick-toggle");
    const undoButton = document.getElementById("pick-undo");
    const finishButton = document.getElementById("pick-finish");
    const saveButton = document.getElementById("pick-save");
    const statusEl = document.getElementById("pick-status");
    const downloadLink = document.getElementById("pick-download");
    const errorBox = document.getElementById("pick-error");
    const selectionBox = document.getElementById("pick-selection");
    const selectedLayer = document.getElementById("pick-selected-layer");
    const deleteButton = document.getElementById("pick-delete");
    const selectionHint = document.getElementById("pick-selection-hint");

    /** Stored features, as gprinterp features in index space. */
    let features = [];
    /** ETag of the document these came from, or null if none is stored yet.
     * Sent as If-Match so a save cannot silently discard another tab's edit. */
    let etag = null;
    let layers = [];
    let picking = false;
    let dirty = false;
    /** The line being drawn: array of [trace, sample], or null. */
    let draft = null;
    /** Index into `features` of the selected line, or null. */
    let selected = null;

    let drawnLines = [];
    let draftLine = null;
    let handles = [];
    let nextId = 1;

    const showError = (message) => {
      errorBox.textContent = message;
      errorBox.hidden = false;
    };
    const clearError = () => {
      errorBox.hidden = true;
      errorBox.textContent = "";
    };

    // --- Coordinate conversion ----------------------------------------------

    function toIndex(latlng) {
      const scale = window.RIDAL_XSCALE || 1;
      return [
        latlng.lng / scale / RASTER_SCALE,
        -latlng.lat / VERTICAL_RASTER_SCALE,
      ];
    }

    function toLatLng(trace, sample) {
      const scale = window.RIDAL_XSCALE || 1;
      return [-sample * VERTICAL_RASTER_SCALE, trace * RASTER_SCALE * scale];
    }

    const layerFor = (label) => layers.find((l) => l.id === label);
    const colorFor = (label) => (layerFor(label) || {}).color || DEFAULT_COLOR;
    const allowsOverhangs = (label) => Boolean((layerFor(label) || {}).allow_overhangs);

    function newFeature(coordinates, label) {
      return {
        type: "Feature",
        geometry: { type: "LineString", coordinates },
        // Unique per feature: `properties.id` is identity, and two features
        // sharing one would make the document ambiguous (SPEC 4.5).
        properties: { id: `f-${Date.now()}-${nextId++}`, label },
      };
    }

    // --- The overhang rule, applied while drawing ----------------------------

    /** Whether adding `trace` would make the draft double back.
     *
     * The same rule the server enforces at save (interp/checks.rs), run here
     * so a mis-click is refused when it happens rather than when the whole
     * document is rejected. The server stays the authority; this is only the
     * earlier, kinder half. */
    function wouldOverhang(trace) {
      if (!draft || draft.length < 1) return false;
      if (allowsOverhangs(layerSelect.value)) return false;
      const traces = draft.map((v) => v[0]);
      if (traces.length === 1) return trace === traces[0];
      const descending = traces[traces.length - 1] < traces[0];
      const last = traces[traces.length - 1];
      return descending ? trace >= last : trace <= last;
    }

    // --- Editing a stored line ----------------------------------------------

    function select(index) {
      selected = index;
      redraw();
    }

    function deselect() {
      if (selected === null) return;
      selected = null;
      redraw();
    }

    /** Replace the selected line with the two halves of a split.
     *
     * One `splice` that removes exactly one feature and inserts exactly two.
     * Written this way on purpose: a split implemented as "add both halves,
     * then remove the original" leaves the original behind whenever the
     * removal is skipped or the handler fires twice, which is how a split
     * turns into duplicate overlapping lines. */
    function splitSelectedAt(vertexIndex) {
      const feature = features[selected];
      const halves = splitCoordinates(feature.geometry.coordinates, vertexIndex);
      if (halves === null) {
        showError(
          "Pick a vertex in the middle of the line to split it -- splitting at an " +
            "end would leave a line with a single vertex.",
        );
        return;
      }
      const label = feature.properties && feature.properties.label;
      features.splice(
        selected,
        1,
        newFeature(halves[0], label),
        newFeature(halves[1], label),
      );
      // Indices have shifted, and "the selected line" no longer exists.
      selected = null;
      clearError();
      markDirty();
      redraw();
    }

    function deleteSelected() {
      features.splice(selected, 1);
      selected = null;
      markDirty();
      redraw();
    }

    // --- Rendering -----------------------------------------------------------

    function redraw() {
      drawnLines.forEach((line) => map.removeLayer(line));
      drawnLines = features.map((feature, index) => {
        const label = feature.properties && feature.properties.label;
        const isSelected = index === selected;
        const line = L.polyline(
          feature.geometry.coordinates.map(([t, s]) => toLatLng(t, s)),
          {
            color: colorFor(label),
            weight: isSelected ? 5 : 3,
            opacity: isSelected ? 1 : 0.85,
          },
        ).addTo(map);
        line.bindTooltip(
          `${label || "unlabelled"} (${feature.geometry.coordinates.length} vertices)`,
        );
        line.on("click", (event) => {
          // While picking, a click over an existing line is still a new
          // vertex -- lines must not become holes in the drawing surface.
          if (picking) return;
          L.DomEvent.stopPropagation(event);
          select(index === selected ? null : index);
        });
        return line;
      });
      redrawHandles();
      updateSelectionPanel();
      updateStatus();
    }

    /** Vertex handles: the draft's, or the selected line's. */
    function redrawHandles() {
      if (draftLine) {
        map.removeLayer(draftLine);
        draftLine = null;
      }
      handles.forEach((handle) => map.removeLayer(handle));
      handles = [];

      if (draft && draft.length) {
        const color = colorFor(layerSelect.value);
        if (draft.length > 1) {
          draftLine = L.polyline(
            draft.map(([t, s]) => toLatLng(t, s)),
            { color, weight: 3, dashArray: "6 4" },
          ).addTo(map);
        }
        handles = draft.map(([t, s], index) =>
          L.circleMarker(toLatLng(t, s), { color, radius: 4, fillOpacity: 1 })
            .addTo(map)
            .on("contextmenu", (event) => {
              L.DomEvent.stop(event);
              draft.splice(index, 1);
              redrawHandles();
              updateStatus();
            }),
        );
        return;
      }

      if (selected === null) return;
      const feature = features[selected];
      const label = feature.properties && feature.properties.label;
      handles = feature.geometry.coordinates.map(([t, s], index) => {
        const interior = index > 0 && index < feature.geometry.coordinates.length - 1;
        return L.circleMarker(toLatLng(t, s), {
          color: colorFor(label),
          fillColor: interior ? "#fff" : colorFor(label),
          radius: interior ? 5 : 4,
          fillOpacity: 1,
        })
          .addTo(map)
          .bindTooltip(interior ? "Click to split here" : "End of line")
          .on("click", (event) => {
            L.DomEvent.stopPropagation(event);
            splitSelectedAt(index);
          });
      });
    }

    function updateSelectionPanel() {
      const active = selected !== null;
      selectionBox.hidden = !active;
      if (!active) return;
      const feature = features[selected];
      const label = feature.properties && feature.properties.label;
      selectedLayer.value = label || "";
      selectionHint.textContent =
        feature.geometry.coordinates.length > 2
          ? "Click a hollow vertex to split this line."
          : "Too few vertices to split.";
    }

    function updateStatus() {
      const parts = [`${features.length} line${features.length === 1 ? "" : "s"}`];
      if (draft && draft.length) parts.push(`drawing: ${draft.length} vertices`);
      parts.push(dirty ? "unsaved" : "saved");
      statusEl.textContent = parts.join(" - ");
      statusEl.classList.toggle("dirty", dirty);

      saveButton.disabled = !dirty;
      undoButton.disabled = !draft || draft.length === 0;
      finishButton.disabled = !draft || draft.length < 2;
      downloadLink.hidden = dirty || features.length === 0;
    }

    function markDirty() {
      dirty = true;
      updateStatus();
    }

    // --- Drawing --------------------------------------------------------------

    function setPicking(on) {
      picking = on;
      toggleButton.textContent = on ? "Stop picking" : "Start picking";
      toggleButton.setAttribute("aria-pressed", String(on));
      document.getElementById("map").classList.toggle("picking", on);
      if (on) deselect();
      else finishLine();
    }

    function finishLine() {
      if (!draft || draft.length < 2) {
        draft = null;
        redrawHandles();
        updateStatus();
        return;
      }
      features.push(newFeature(draft, layerSelect.value));
      draft = null;
      markDirty();
      redraw();
    }

    map.on("click", (event) => {
      if (!picking) {
        deselect();
        return;
      }
      if (!layerSelect.value) {
        showError("Choose a layer before picking.");
        return;
      }
      const [trace, sample] = toIndex(event.latlng);
      if (trace < 0 || trace > CFG.sourceWidth || sample < 0 || sample > CFG.sourceHeight) {
        return;
      }
      if (wouldOverhang(trace)) {
        showError(
          "A line in this layer must not double back: each position along the " +
            "profile can have only one depth. Finish this line and start another, " +
            "or allow overhangs on the layer.",
        );
        return;
      }
      clearError();
      if (draft === null) draft = [];
      draft.push([trace, sample]);
      redrawHandles();
      updateStatus();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (picking) finishLine();
        else deselect();
      }
      if (event.key === "z" && (event.ctrlKey || event.metaKey) && draft && draft.length) {
        event.preventDefault();
        draft.pop();
        redrawHandles();
        updateStatus();
      }
      if (event.key === "s" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        save();
      }
    });

    window.RIDAL_REDRAW_PICKS = redraw;

    toggleButton.addEventListener("click", () => setPicking(!picking));
    undoButton.addEventListener("click", () => {
      if (draft && draft.length) {
        draft.pop();
        redrawHandles();
        updateStatus();
      }
    });
    finishButton.addEventListener("click", finishLine);
    saveButton.addEventListener("click", save);
    layerSelect.addEventListener("change", redrawHandles);
    deleteButton.addEventListener("click", deleteSelected);
    selectedLayer.addEventListener("change", () => {
      if (selected === null) return;
      features[selected].properties.label = selectedLayer.value;
      markDirty();
      redraw();
    });

    window.addEventListener("beforeunload", (event) => {
      if (dirty) event.preventDefault();
    });

    // --- Persistence ----------------------------------------------------------

    const documentUrl = `/api/v1/datasets/${CFG.radargramId}/interpretations/${CFG.user}`;

    async function save() {
      if (!dirty) return;
      finishLine();
      clearError();

      const body = {
        schema: "gprinterp",
        schema_version: "0.1",
        key: CFG.radargramId,
        date_modified: new Date().toISOString(),
        source: {
          id: CFG.radargramId,
          n_traces: CFG.sourceWidth,
          n_samples: CFG.sourceHeight,
        },
        features,
      };
      const headers = { "Content-Type": "application/json" };
      if (etag) headers["If-Match"] = etag;

      try {
        const response = await fetch(documentUrl, {
          method: "PUT",
          headers,
          body: JSON.stringify(body),
        });
        if (response.status === 412) {
          showError(
            "These picks were changed somewhere else while this page was open. " +
              "Reload to see the saved version -- nothing here is lost until you do.",
          );
          return;
        }
        if (!response.ok) {
          const failure = await response.json().catch(() => null);
          showError(failure?.error?.message || `Could not save (${response.status}).`);
          return;
        }
        etag = response.headers.get("ETag");
        dirty = false;
        updateStatus();
      } catch (error) {
        showError(`Could not save: ${error.message}`);
      }
    }

    /** Load whatever is already stored for this radargram.
     *
     * Runs on page load, unconditionally: opening a radargram shows the picks
     * that exist for it, rather than an empty canvas that would invite
     * redrawing work someone has already done. */
    async function load() {
      try {
        const response = await fetch(documentUrl);
        if (response.status === 404) {
          // Nobody has interpreted this radargram yet: a starting state, not
          // a failure.
          features = [];
          etag = null;
          redraw();
          return;
        }
        if (!response.ok) {
          const failure = await response.json().catch(() => null);
          showError(failure?.error?.message || `Could not load picks (${response.status}).`);
          return;
        }
        etag = response.headers.get("ETag");
        const body = await response.json();
        // Only lines are editable here. Anything else in the document is left
        // untouched on the server rather than silently dropped by a save --
        // which is why a document containing one is not offered for editing.
        const all = body.features || [];
        features = all.filter((f) => f.geometry && f.geometry.type === "LineString");
        if (features.length !== all.length) {
          showError(
            "This interpretation contains geometry other than lines, which this " +
              "viewer cannot edit. Saving here would drop it, so editing is " +
              "disabled for safety.",
          );
          saveButton.disabled = true;
          toggleButton.disabled = true;
          return;
        }
        redraw();
      } catch (error) {
        showError(`Could not load picks: ${error.message}`);
      }
    }

    async function loadLayers() {
      try {
        const body = await RIDAL.fetchJson("/api/v1/layers");
        layers = body.layers || [];
      } catch (error) {
        console.warn(`Could not load layers: ${error.message}`);
        layers = [];
      }
      const options = layers.length
        ? layers.map((layer) => {
            const option = document.createElement("option");
            option.value = layer.id;
            option.textContent = layer.name || layer.id;
            return option;
          })
        : [new Option("No layers defined - add one on the Layers page", "")];
      layerSelect.replaceChildren(...options);
      selectedLayer.replaceChildren(...options.map((o) => o.cloneNode(true)));
    }

    // --- Level 2 download ------------------------------------------------------

    const dialog = document.getElementById("download-dialog");
    downloadLink.addEventListener("click", (event) => {
      event.preventDefault();
      dialog.showModal();
    });
    document.getElementById("download-close").addEventListener("click", () => dialog.close());
    document.getElementById("download-go").addEventListener("click", () => {
      const spacing = document.getElementById("download-spacing").value;
      const format = document.getElementById("download-format").value;
      window.location.href =
        `${documentUrl}/level2?spacing=${encodeURIComponent(spacing)}` +
        `&format=${encodeURIComponent(format)}`;
      dialog.close();
    });

    // Layers first: colours and overhang permissions are needed before the
    // stored picks can be drawn correctly.
    loadLayers().then(load);
  }
})();
