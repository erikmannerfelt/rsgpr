/* Picking reflectors on the radargram (#102).
 *
 * First-party, embedded via assets.rs, loaded after viewer.js so `L`,
 * `RIDAL` and `window.RIDAL_VIEWER` all exist. Deliberately NOT under
 * assets/vendor/ -- scripts/vendor_leaflet.sh does `rm -rf` on that
 * directory.
 *
 * Built directly on Leaflet rather than on Leaflet.Draw. The interaction a
 * horizon picker needs is narrow -- extend a line, drag a vertex, undo,
 * finish, reassign, split -- while Leaflet.Draw brings a shape palette and
 * a modal toolbar that would mostly be hidden, and has been unmaintained
 * since before Leaflet 1.9.
 *
 * Coordinates: the map is L.CRS.Simple over the *viewer raster*, while
 * picks are stored in source trace/sample index space. The conversion is
 * redone here rather than imported from viewer.js, because both are plain
 * scripts with no module boundary between them (#120: no build step).
 *
 * Touch first: the field device is a phone. Vertex handles are `L.marker`s,
 * not `L.circleMarker`s, because only markers are draggable and only they
 * get a real touch target. Every gesture works with one finger -- there is
 * no right-click and no hover anywhere in here.
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
   * tapped. So the vertex count goes up by one -- that is the split, not a
   * duplicated line.
   *
   * Splitting at an end is refused rather than clamped, because it would
   * leave a one-vertex "line", which is not a line and cannot be exported.
   */
  function splitCoordinates(coordinates, vertexIndex) {
    if (vertexIndex <= 0 || vertexIndex >= coordinates.length - 1) return null;
    return [
      coordinates.slice(0, vertexIndex + 1),
      coordinates.slice(vertexIndex),
    ];
  }

  /** Join two lines end to end.
   *
   * `aAtStart` / `bAtStart` say which end of each line is being joined.
   * Each line is oriented so the meeting ends face each other, then `a`'s
   * own endpoint is dropped and `b`'s is kept -- the two are near each
   * other but not identical, and keeping one of them is what makes this a
   * join rather than a line with a tiny kink in it.
   *
   * The result therefore has `a.length + b.length - 1` vertices. Anything
   * else means a vertex was duplicated or lost.
   *
   * The inverse of `splitCoordinates`, and deliberately as literal about
   * it: splitting at vertex `i` and rejoining the halves returns the
   * original line.
   */
  function joinCoordinates(a, b, aAtStart, bAtStart) {
    // Orient `a` so its joining end is last, and `b` so its joining end is
    // first. Reversing a line is not a change of interpretation -- it is
    // the same reflector recorded in the other direction.
    const head = aAtStart ? a.slice().reverse() : a.slice();
    const tail = bAtStart ? b.slice() : b.slice().reverse();
    return [...head.slice(0, -1), ...tail];
  }

  /** Every vertex at which a line stops advancing in trace.
   *
   * The client-side twin of `overhang_at` in `src/interp/checks.rs`, which
   * returns only the first. This returns all of them, because they are
   * drawn: a line that doubles back three times gets three markers, so the
   * problem is visible rather than merely described.
   *
   * Direction comes from the first and last vertex, matching the server, so
   * a line drawn right-to-left is not an overhang -- it is the same
   * interpretation recorded in the opposite order.
   */
  function overhangIndices(coordinates) {
    const traces = coordinates.map((c) => c[0]);
    if (traces.length < 2) return [];
    const descending = traces[traces.length - 1] < traces[0];
    const offending = [];
    for (let i = 1; i < traces.length; i++) {
      const advances = descending
        ? traces[i] < traces[i - 1]
        : traces[i] > traces[i - 1];
      if (!advances) offending.push(i);
    }
    return offending;
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
    /** ETag of the document these came from, or null if none is stored yet. */
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
    let overhangMarkers = [];
    let nextId = 1;

    const showError = (message) => {
      errorBox.textContent = message;
      errorBox.classList.remove("toast-info");
      errorBox.hidden = false;
    };
    /** Same toast, said calmly.
     *
     * Worth having because the toast is `position: fixed` -- it costs no
     * layout, so an explanation can appear exactly when it is relevant and
     * vanish when it is not, without moving anything. */
    const showInfo = (message) => {
      errorBox.textContent = message;
      errorBox.classList.add("toast-info");
      errorBox.hidden = false;
    };
    const clearError = () => {
      errorBox.hidden = true;
      errorBox.textContent = "";
      errorBox.classList.remove("toast-info");
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
    const allowsOverhangs = (label) =>
      Boolean((layerFor(label) || {}).allow_overhangs);

    function newFeature(coordinates, label) {
      return {
        type: "Feature",
        geometry: { type: "LineString", coordinates },
        properties: { id: `f-${Date.now()}-${nextId++}`, label },
      };
    }

    // --- Handles -------------------------------------------------------------

    /** A draggable vertex handle.
     *
     * `L.marker` with a `divIcon` rather than `L.circleMarker`: circle
     * markers cannot be dragged at all, and an SVG circle is a poor touch
     * target. The icon is sized in CSS so it can grow on coarse pointers.
     */
    function makeHandle(coordinates, index, label, kind, onTap, featureIndex) {
      const [trace, sample] = coordinates[index];
      const marker = L.marker(toLatLng(trace, sample), {
        draggable: true,
        keyboard: false,
        icon: L.divIcon({
          className: `pick-handle pick-handle-${kind}`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        }),
      }).addTo(map);
      marker.setZIndexOffset(1000);

      marker.on("dragend", () => {
        const before = coordinates[index];
        const dropped = marker.getLatLng();

        // Dropped onto one of its own neighbours: the two would be on top
        // of each other, so the intent is to get rid of this one. Checked
        // before the join below because a neighbour on the same line is the
        // nearer, more local target -- joining is about a *different* line.
        if (droppedOnNeighbour(coordinates, index, dropped)) {
          removeVertex(coordinates, index);
          return;
        }

        // Dragging an end of a stored line onto the end of another is how
        // two lines are joined back together -- the inverse of tapping a
        // middle vertex to split one.
        const isEnd = index === 0 || index === coordinates.length - 1;
        if (featureIndex !== undefined && featureIndex !== null && isEnd) {
          const target = findJoinTarget(featureIndex, dropped);
          if (target) {
            joinWith(featureIndex, index === 0, target);
            return;
          }
        }

        const [newTrace, newSample] = toIndex(dropped);
        // Preserve any third element GeoJSON allows, rather than truncating
        // a position this viewer did not author.
        coordinates[index] = [newTrace, newSample, ...before.slice(2)];

        if (!allowsOverhangs(label) && overhangIndices(coordinates).length > 0) {
          coordinates[index] = before;
          showError(
            "Moving that vertex there would make the line double back, so it " +
              "would have two depths at one position. Move it somewhere the " +
              "line keeps advancing, or allow overhangs on this layer.",
          );
        } else {
          clearError();
          markDirty();
        }
        redraw();
      });

      marker.on("click", (event) => {
        L.DomEvent.stopPropagation(event);
        onTap();
      });
      return marker;
    }

    /** How near, in screen pixels, a vertex has to be dropped to a
     * neighbour to be removed, and an endpoint to another line's endpoint
     * to be joined. Screen pixels rather than trace indices: the tolerance
     * should be a fingertip regardless of zoom or horizontal stretch. */
    const SNAP_RADIUS_PX = 24;

    const COARSE_POINTER = window.matchMedia("(pointer: coarse)").matches;

    /** A segment shorter than this on screen gets no midpoint handle.
     *
     * Erik's suggestion, and it is better than the fixed vertex cap it
     * replaces: midpoints appear only where there is room to use them, so
     * zooming out thins them out on its own and they never crowd the
     * vertices they sit between. Roughly two handle widths, so a midpoint
     * and its two neighbours cannot overlap. */
    const MIN_SEGMENT_PX_FOR_MIDPOINT = COARSE_POINTER ? 72 : 44;

    function pixelsApart(latlng, [trace, sample]) {
      return map
        .latLngToContainerPoint(latlng)
        .distanceTo(map.latLngToContainerPoint(toLatLng(trace, sample)));
    }

    /** Whether `latlng` lands on the vertex before or after `index`. */
    function droppedOnNeighbour(coordinates, index, latlng) {
      return [index - 1, index + 1].some(
        (i) =>
          i >= 0 &&
          i < coordinates.length &&
          pixelsApart(latlng, coordinates[i]) <= SNAP_RADIUS_PX,
      );
    }

    /** Drop a vertex from a line.
     *
     * Refused rather than clamped when it would leave fewer than two
     * vertices: one point is not a line, cannot be exported, and there is
     * no way back from it. Deleting the whole line is a separate,
     * deliberate button.
     *
     * Says what happened, because a vertex vanishing under a finger is
     * otherwise indistinguishable from a mis-drag -- and there is no undo. */
    function removeVertex(coordinates, index) {
      if (coordinates.length <= 2) {
        showError(
          "A line needs at least two vertices, so this one cannot be removed. " +
            "Use Delete line if you meant to remove the whole line.",
        );
        redraw();
        return;
      }
      coordinates.splice(index, 1);
      markDirty();
      redraw();
      showInfo("Vertex removed -- it was dropped onto its neighbour.");
    }

    /** The small handle between two vertices that inserts a third.
     *
     * Leaflet.Draw's pattern, and the reason it works is that one gesture
     * covers both intents: a tap drops a vertex at the midpoint, while
     * pressing and dragging creates it and positions it in the same motion,
     * with no intermediate state to undo.
     *
     * The insert itself can never create an overhang -- the midpoint of two
     * points is strictly between them -- so only the drag needs validating.
     */
    function makeMidpoint(coordinates, index, label) {
      const [aTrace, aSample] = coordinates[index];
      const [bTrace, bSample] = coordinates[index + 1];
      const midpoint = [(aTrace + bTrace) / 2, (aSample + bSample) / 2];

      // The touch target is the full icon; the dot inside it is what you
      // see. They were the same element before, at 18px on a phone against
      // a vertex handle's 24px, and a finger drag whose first sample lands
      // a few pixels off then goes to the map instead of the marker --
      // which is why a midpoint could be tapped but not dragged. A tap is
      // one point and forgiving; a drag is not.
      const marker = L.marker(toLatLng(midpoint[0], midpoint[1]), {
        draggable: true,
        keyboard: false,
        icon: L.divIcon({
          className: "pick-midpoint",
          html: '<i class="pick-midpoint-dot"></i>',
          iconSize: [26, 26],
          iconAnchor: [13, 13],
        }),
      }).addTo(map);
      // Below the real vertices, so where the two overlap the vertex wins.
      marker.setZIndexOffset(900);
      // Hover-only affordance: on a touch screen the tooltip opens on the
      // same tap that adds the vertex, so it is noise at best.
      if (!COARSE_POINTER) {
        marker.bindTooltip("Tap to add a vertex here, or drag to place one");
      }

      // Inserted on `dragstart` so the drag is already moving a real
      // vertex, exactly as if it had been there all along. Deliberately no
      // redraw until the drag ends -- rebuilding the handles mid-drag would
      // destroy the marker being dragged.
      let dragging = false;
      marker.on("dragstart", () => {
        dragging = true;
        coordinates.splice(index + 1, 0, midpoint.slice());
      });

      marker.on("dragend", () => {
        const [trace, sample] = toIndex(marker.getLatLng());
        coordinates[index + 1] = [trace, sample];
        if (!allowsOverhangs(label) && overhangIndices(coordinates).length > 0) {
          coordinates.splice(index + 1, 1);
          showError(
            "A vertex there would make the line double back, so it would have " +
              "two depths at one position. Nothing was added.",
          );
        } else {
          clearError();
          markDirty();
        }
        dragging = false;
        redraw();
      });

      marker.on("click", (event) => {
        L.DomEvent.stopPropagation(event);
        // Leaflet can fire a click after a drag; the drag already did the
        // work.
        if (dragging) return;
        coordinates.splice(index + 1, 0, midpoint.slice());
        clearError();
        markDirty();
        redraw();
      });

      return marker;
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

    /** The nearest joinable endpoint to `latlng`, or null.
     *
     * Restricted to the same layer: joining a "bed" line to an "internal"
     * one would have to silently pick a label for the result, and picking
     * either is wrong. Restricted to *other* lines: dragging a line's start
     * onto its own end would close a loop, which is never a function of
     * trace. */
    function findJoinTarget(featureIndex, latlng) {
      const label = features[featureIndex].properties?.label;
      const point = map.latLngToContainerPoint(latlng);
      let best = null;
      features.forEach((feature, index) => {
        if (index === featureIndex) return;
        if ((feature.properties?.label ?? null) !== (label ?? null)) return;
        const coordinates = feature.geometry.coordinates;
        for (const atStart of [true, false]) {
          const [trace, sample] = atStart
            ? coordinates[0]
            : coordinates[coordinates.length - 1];
          const distance = point.distanceTo(
            map.latLngToContainerPoint(toLatLng(trace, sample)),
          );
          if (distance <= SNAP_RADIUS_PX && (!best || distance < best.distance)) {
            best = { index, atStart, distance };
          }
        }
      });
      return best;
    }

    /** Merge the dragged line into the one whose endpoint it was dropped on.
     *
     * Two features out, one in, via a `splice` pair that removes the higher
     * index first so the lower one is still valid -- the same discipline as
     * `splitSelectedAt`, and for the same reason: a merge written as "add
     * the joined line, then remove the two originals" leaves an original
     * behind whenever a removal is skipped. */
    function joinWith(featureIndex, draggedAtStart, target) {
      const source = features[featureIndex];
      const other = features[target.index];
      const label = source.properties?.label;

      const merged = joinCoordinates(
        source.geometry.coordinates,
        other.geometry.coordinates,
        draggedAtStart,
        target.atStart,
      );

      if (!allowsOverhangs(label) && overhangIndices(merged).length > 0) {
        showError(
          "Joining those two lines would double back, so the result would have " +
            "two depths at one position. They probably need joining at their " +
            "other ends, or they overlap along the profile.",
        );
        redraw();
        return;
      }

      const low = Math.min(featureIndex, target.index);
      const high = Math.max(featureIndex, target.index);
      features.splice(high, 1);
      features.splice(low, 1, newFeature(merged, label));

      // Select the result rather than dropping the selection: the user is
      // looking at what they just made, and its vertices are what they will
      // want to adjust next.
      selected = low;
      clearError();
      markDirty();
      redraw();
    }

    /** Replace the selected line with the two halves of a split.
     *
     * One `splice` that removes exactly one feature and inserts exactly
     * two. Written this way on purpose: a split implemented as "add both
     * halves, then remove the original" leaves the original behind whenever
     * the removal is skipped or the handler fires twice, which is how a
     * split turns into duplicate overlapping lines. */
    function splitSelectedAt(vertexIndex) {
      const feature = features[selected];
      const halves = splitCoordinates(feature.geometry.coordinates, vertexIndex);
      if (halves === null) {
        showError(
          "Tap a vertex in the middle of the line to split it -- splitting at " +
            "an end would leave a line with a single vertex.",
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
      selected = null;
      clearError();
      markDirty();
      redraw();
    }

    /** Delete the selected line.
     *
     * Does nothing without a selection, rather than deleting a line the
     * user cannot see. `features.splice(null, 1)` coerces `null` to `0` and
     * quietly removes the *first* line, which is what this did when the
     * panel was reachable with nothing selected -- once per press.
     *
     * Deleting "the last line" instead was the alternative, but a
     * destructive action needs a visible target: there is no way to show
     * which line "the last one" is, so a confirmation could not name what
     * was about to be lost. Every control in a panel headed "Selected line"
     * acts on the selection, or not at all. */
    function deleteSelected() {
      if (selected === null) return;
      features.splice(selected, 1);
      selected = null;
      markDirty();
      redraw();
    }

    // --- Rendering -----------------------------------------------------------

    function redraw() {
      drawnLines.forEach((line) => map.removeLayer(line));
      drawnLines = [];
      features.forEach((feature, index) => {
        const label = feature.properties && feature.properties.label;
        const isSelected = index === selected;
        const points = feature.geometry.coordinates.map(([t, s]) => toLatLng(t, s));

        // The wide, invisible companion goes down first so the visible line
        // draws over it, and carries all the interaction: the visible line
        // is non-interactive, so it cannot swallow a tap meant for the
        // easier target.
        const hit = RIDAL.hitLine(points).addTo(map);
        hit.bindTooltip(
          `${label || "unlabelled"} (${feature.geometry.coordinates.length} vertices)`,
        );
        hit.on("click", (event) => {
          // While picking, a tap over an existing line is still a new
          // vertex -- lines must not become holes in the drawing surface.
          if (picking) return;
          L.DomEvent.stopPropagation(event);
          select(index === selected ? null : index);
        });

        const line = L.polyline(points, {
          color: colorFor(label),
          weight: isSelected ? 5 : 3,
          opacity: isSelected ? 1 : 0.85,
          interactive: false,
        }).addTo(map);

        drawnLines.push(hit, line);
      });
      redrawHandles();
      redrawOverhangs();
      updateSelectionPanel();
      updateStatus();
    }

    /** Handles for whichever line is being edited: the draft, or the
     * selected stored line. Only one set exists at a time, so a tap on a
     * handle is never ambiguous. */
    function redrawHandles() {
      if (draftLine) {
        map.removeLayer(draftLine);
        draftLine = null;
      }
      handles.forEach((handle) => map.removeLayer(handle));
      handles = [];

      if (draft && draft.length) {
        const label = layerSelect.value;
        if (draft.length > 1) {
          draftLine = L.polyline(
            draft.map(([t, s]) => toLatLng(t, s)),
            { color: colorFor(label), weight: 3, dashArray: "6 4" },
          ).addTo(map);
        }
        handles = draft.map((_, index) =>
          makeHandle(draft, index, label, "draft", () => {
            // Tap removes. There is no right-click on a phone, and Undo
            // only ever reaches the last vertex.
            draft.splice(index, 1);
            clearError();
            redrawHandles();
            redrawOverhangs();
            updateStatus();
          }),
        );
        return;
      }

      if (selected === null) return;
      const feature = features[selected];
      const label = feature.properties && feature.properties.label;
      const coordinates = feature.geometry.coordinates;
      handles = coordinates.map((_, index) => {
        const interior = index > 0 && index < coordinates.length - 1;
        const handle = makeHandle(
          coordinates,
          index,
          label,
          interior ? "interior" : "end",
          () => {
            if (interior) splitSelectedAt(index);
          },
          selected,
        );
        handle.bindTooltip(
          interior ? "Drag to move, tap to split here" : "Drag to move",
        );
        return handle;
      });

      // A midpoint per segment, but only where one is usable: long enough
      // on screen to aim at, and actually in view. A horizon with hundreds
      // of vertices therefore costs nothing until it is zoomed into, and
      // then only for the part being looked at.
      const view = map.getBounds();
      for (let index = 0; index < coordinates.length - 1; index++) {
        const a = toLatLng(coordinates[index][0], coordinates[index][1]);
        const b = toLatLng(coordinates[index + 1][0], coordinates[index + 1][1]);
        if (!view.intersects(L.latLngBounds(a, b))) continue;
        const lengthPx = map
          .latLngToContainerPoint(a)
          .distanceTo(map.latLngToContainerPoint(b));
        if (lengthPx < MIN_SEGMENT_PX_FOR_MIDPOINT) continue;
        handles.push(makeMidpoint(coordinates, index, label));
      }
    }

    /** A marker at every vertex where a line doubles back.
     *
     * Drawn for *all* lines, including layers that allow overhangs: an
     * intentional overhang is still worth seeing, and a line saved before
     * the rule existed would otherwise look fine while quietly failing to
     * export at even spacing. */
    function redrawOverhangs() {
      overhangMarkers.forEach((marker) => map.removeLayer(marker));
      overhangMarkers = [];

      const lines = features.map((f) => [
        f.geometry.coordinates,
        (f.properties && f.properties.label) || null,
      ]);
      if (draft && draft.length) lines.push([draft, layerSelect.value]);

      for (const [coordinates, label] of lines) {
        const allowed = allowsOverhangs(label);
        for (const index of overhangIndices(coordinates)) {
          const [trace, sample] = coordinates[index];
          overhangMarkers.push(
            L.marker(toLatLng(trace, sample), {
              keyboard: false,
              icon: L.divIcon({
                className: `pick-overhang${allowed ? " pick-overhang-allowed" : ""}`,
                iconSize: [18, 18],
                iconAnchor: [9, 9],
              }),
            })
              .addTo(map)
              .bindTooltip(
                allowed
                  ? `Overhang at vertex ${index}, allowed on "${label}". This ` +
                      "layer exports as picked vertices, not evenly spaced."
                  : `Overhang at vertex ${index}: the line doubles back here, ` +
                      "so it has two depths at one position.",
              ),
          );
        }
      }
    }

    function updateSelectionPanel() {
      const active = selected !== null;
      selectionBox.hidden = !active;
      // Not only hidden: a disabled button cannot be activated even if a
      // future style rule makes the panel visible again, which is the way
      // this failed the first time.
      deleteButton.disabled = !active;
      if (!active) return;
      const feature = features[selected];
      const label = feature.properties && feature.properties.label;
      selectedLayer.value = label || "";
      const many = feature.geometry.coordinates.length > 2;
      selectionHint.textContent =
        "Drag a vertex to move it, or onto its neighbour to remove it. " +
        "Tap a small handle between two vertices to add one. " +
        (many ? "Tap a middle vertex to split. " : "") +
        "Drop an end onto another line's end in the same layer to join them.";
    }

    function countOverhangs() {
      let total = features.reduce(
        (sum, f) => sum + overhangIndices(f.geometry.coordinates).length,
        0,
      );
      if (draft) total += overhangIndices(draft).length;
      return total;
    }

    function updateStatus() {
      const parts = [
        `${features.length} line${features.length === 1 ? "" : "s"}`,
      ];
      if (draft && draft.length) parts.push(`drawing: ${draft.length}`);
      const overhangs = countOverhangs();
      if (overhangs) {
        parts.push(`${overhangs} overhang${overhangs === 1 ? "" : "s"}`);
      }
      parts.push(dirty ? "unsaved" : "saved");
      statusEl.textContent = parts.join(" · ");
      statusEl.classList.toggle("dirty", dirty);

      saveButton.disabled = !dirty;
      undoButton.disabled = !draft || draft.length === 0;
      finishButton.disabled = !draft || draft.length < 2;
      // Naming the count ties the button to the line in progress. "Finish
      // line" on its own reads as a mode switch, which is what made it
      // hard to guess what it would do.
      finishButton.textContent =
        draft && draft.length ? `Finish line (${draft.length})` : "Finish line";
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
      if (on) {
        deselect();
        // Every tap extends the *same* line until it is finished, which is
        // not guessable from a toolbar of buttons. Said once, when it
        // becomes relevant, and cleared by the first tap.
        showInfo(
          "Tap the radargram to add points to one line. Finish line ends it, " +
            "so the next tap starts a separate line.",
        );
      } else {
        finishLine();
        clearError();
      }
    }

    function finishLine() {
      if (!draft || draft.length < 2) {
        draft = null;
        redrawHandles();
        redrawOverhangs();
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
      if (
        trace < 0 ||
        trace > CFG.sourceWidth ||
        sample < 0 ||
        sample > CFG.sourceHeight
      ) {
        return;
      }

      // Test the whole candidate line, not just this vertex against the
      // previous one. Direction is a property of the line as a whole, and
      // checking pairwise let one stray vertex flip the perceived direction
      // and then reject every later point as an overhang.
      const candidate = draft
        ? draft.concat([[trace, sample]])
        : [[trace, sample]];
      if (
        !allowsOverhangs(layerSelect.value) &&
        overhangIndices(candidate).length > 0
      ) {
        showError(
          "That point would make the line double back, so it would have two " +
            "depths at one position. Carry on in the direction you started, " +
            "finish this line and begin another, or allow overhangs on this " +
            "layer.",
        );
        return;
      }
      // Clears the "how this works" hint too, on the first tap that proves
      // it was read.
      clearError();
      draft = candidate;
      redrawHandles();
      redrawOverhangs();
      updateStatus();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (picking) finishLine();
        else deselect();
      }
      if (
        event.key === "z" &&
        (event.ctrlKey || event.metaKey) &&
        draft &&
        draft.length
      ) {
        event.preventDefault();
        draft.pop();
        redrawHandles();
        redrawOverhangs();
        updateStatus();
      }
      if (event.key === "s" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        save();
      }
    });

    window.RIDAL_REDRAW_PICKS = redraw;

    // Midpoint visibility depends on zoom and pan, so the handles are
    // rebuilt when the view settles. `moveend`/`zoomend` rather than
    // `move`/`zoom`: rebuilding markers on every frame of a pan would be
    // both wasteful and visibly jumpy.
    map.on("moveend zoomend", redrawHandles);

    toggleButton.addEventListener("click", () => setPicking(!picking));
    undoButton.addEventListener("click", () => {
      if (draft && draft.length) {
        draft.pop();
        clearError();
        redrawHandles();
        redrawOverhangs();
        updateStatus();
      }
    });
    finishButton.addEventListener("click", finishLine);
    saveButton.addEventListener("click", save);
    layerSelect.addEventListener("change", () => {
      redrawHandles();
      redrawOverhangs();
      updateStatus();
    });
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
          showError(
            failure?.error?.message || `Could not save (${response.status}).`,
          );
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
     * Runs on page load, unconditionally: opening a radargram shows the
     * picks that exist for it, rather than an empty canvas that would
     * invite redoing work someone has already done. */
    async function load() {
      try {
        const response = await fetch(documentUrl);
        if (response.status === 404) {
          features = [];
          etag = null;
          redraw();
          return;
        }
        if (!response.ok) {
          const failure = await response.json().catch(() => null);
          showError(
            failure?.error?.message ||
              `Could not load picks (${response.status}).`,
          );
          return;
        }
        etag = response.headers.get("ETag");
        const body = await response.json();
        const all = body.features || [];
        features = all.filter(
          (f) => f.geometry && f.geometry.type === "LineString",
        );
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
    document
      .getElementById("download-close")
      .addEventListener("click", () => dialog.close());
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
