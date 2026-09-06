/* Viewer page behaviour (#115, #121).
 *
 * Loaded by viewer.html.jinja after leaflet.js and app.js, so `L` and
 * `RIDAL` are both defined. Server-side values arrive through the
 * `window.RIDAL_VIEWER` object the template emits immediately before this
 * file -- that inline block is the only JS the template still contains, so
 * everything here stays a plain static asset with no templating in it.
 *
 * Deliberately NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does
 * `rm -rf` on that directory.
 */

const CFG = window.RIDAL_VIEWER;

const RADARGRAM_ID = CFG.radargramId;
const GROUP = CFG.groupId;
const VIEW = "standard";
const CHUNK_SIZE = CFG.chunkSize;
const N_COLS = CFG.nCols;
const N_ROWS = CFG.nRows;
const VIEWER_WIDTH = CFG.viewerWidth;
const VIEWER_HEIGHT = CFG.viewerHeight;
const SOURCE_WIDTH = CFG.sourceWidth;
const SOURCE_HEIGHT = CFG.sourceHeight;
// viewer pixels per source trace/sample -- inverts the render raster's
// own downsampling scale, so the cursor-sync trace lookup below can map a
// viewer pixel back to a real source trace/sample index.
const RASTER_SCALE = VIEWER_WIDTH / SOURCE_WIDTH;
const VERTICAL_RASTER_SCALE = VIEWER_HEIGHT / SOURCE_HEIGHT;

let xScale = 1;

function currentProfile() {
  return document.getElementById('profile-select').value;
}

function chunkUrl(x, y, profile) {
  return `/api/v1/datasets/${RADARGRAM_ID}/views/${VIEW}/chunks/${profile}/${x}/${y}`;
}

function chunkBounds(x, y, scale) {
  // Bounds formula verified in M0 against a real headless-Chromium
  // screenshot: places chunk (0,0) upper-left with no transposition.
  // The x-scale factor stretches/squeezes only the horizontal extent --
  // a pure client-side view transform, no new render, no server
  // round-trip (the plan's explicit design for this control).
  //
  // The rightmost/bottommost chunks cover fewer than CHUNK_SIZE pixels
  // (the raster rarely divides evenly), and the server renders them at
  // exactly that size. Placing them in a full CHUNK_SIZE box instead
  // would stretch them over their neighbours' edges.
  const validWidth = Math.min(CHUNK_SIZE, VIEWER_WIDTH - x * CHUNK_SIZE);
  const validHeight = Math.min(CHUNK_SIZE, VIEWER_HEIGHT - y * CHUNK_SIZE);
  return [
    [-(y * CHUNK_SIZE + validHeight), x * CHUNK_SIZE * scale],
    [-(y * CHUNK_SIZE), (x * CHUNK_SIZE + validWidth) * scale],
  ];
}

let chunkLayers = [];
function loadChunks(map, profile, scale) {
  chunkLayers.forEach((l) => map.removeLayer(l));
  chunkLayers = [];
  for (let y = 0; y < N_ROWS; y++) {
    for (let x = 0; x < N_COLS; x++) {
      const layer = L.imageOverlay(chunkUrl(x, y, profile), chunkBounds(x, y, scale)).addTo(map);
      chunkLayers.push(layer);
    }
  }
}

const map = L.map('map', { crs: L.CRS.Simple, minZoom: -6, attributionControl: false });
// Published for picker.js, which draws onto this same map and needs the
// current horizontal stretch to convert clicks to trace indices. Plain
// globals rather than exports: these are classic scripts with no module
// boundary between them (#120: no build step).
window.RIDAL_MAP = map;
window.RIDAL_XSCALE = 1;
function fitToScale(scale) {
  map.fitBounds([[-VIEWER_HEIGHT, 0], [0, VIEWER_WIDTH * scale]]);
}
fitToScale(xScale);
loadChunks(map, currentProfile(), xScale);

document.getElementById('profile-select').addEventListener('change', () => {
  loadChunks(map, currentProfile(), xScale);
});

document.getElementById('xscale-select').addEventListener('change', (event) => {
  // Rescale from the centre trace, not the left edge: re-lay the
  // overlays at the new scale, then re-derive the map centre's
  // longitude by the same scale ratio the overlays themselves just
  // moved by. Latitude and zoom are untouched -- chunkBounds only ever
  // stretches the horizontal extent, and keeping zoom fixed is what
  // makes this a horizontal-only zoom rather than a no-op.
  const oldScale = xScale;
  const newScale = parseFloat(event.target.value);
  const center = map.getCenter();
  xScale = newScale;
  window.RIDAL_XSCALE = newScale;
  loadChunks(map, currentProfile(), xScale);
  if (window.RIDAL_REDRAW_PICKS) window.RIDAL_REDRAW_PICKS();
  map.setView(
    [center.lat, center.lng * (newScale / oldScale)],
    map.getZoom(),
    { animate: false },
  );
});

// --- Overview map: this radargram's track, plus sibling tracks in the
// same group (clickable, navigating to that radargram), plus a marker
// that follows the viewer cursor (#121's cursor-sync feature). ---
const overviewMap = RIDAL.basemap(L.map('overview-map'));

// --- Resizable split between the radargram and overview map. Only
// meaningful when the two are actually laid out side by side --
// `.layout` is flex-wrap: wrap, so on a narrow screen they stack, at
// which point a horizontal drag handle makes no sense and is hidden.
// Detected exactly (offsetTop equality), not guessed via a media-query
// breakpoint, since the wrap point depends on both panes' flex-basis. ---
(function setupSplitResizer() {
  const layout = document.getElementById('viewer-layout');
  const resizer = document.getElementById('split-resizer');
  const mapEl = document.getElementById('map');
  const overviewEl = document.getElementById('overview-map');
  const MIN_PANE_PX = 200;
  const KEYBOARD_STEP_PX = 24;

  function isSideBySide() {
    return mapEl.offsetTop === overviewEl.offsetTop;
  }

  // Leaflet does not re-lay its tiles when its container is resized by
  // something other than a window resize event it listens for itself --
  // it has to be told. Throttled to one call per frame since pointermove
  // fires far more often than the browser can usefully repaint.
  let invalidateQueued = false;
  function scheduleInvalidate() {
    if (invalidateQueued) return;
    invalidateQueued = true;
    requestAnimationFrame(() => {
      invalidateQueued = false;
      // `pan: false` is load-bearing on a phone. The default re-centres the
      // map to keep the previous centre visible, which reads as the viewer
      // jumping -- and it fires exactly when a first tap collapses the
      // browser's address bar and changes the 70vh map height. The picks
      // stay put either way; only the view was moving.
      map.invalidateSize({ pan: false });
      overviewMap.invalidateSize({ pan: false });
    });
  }

  function updateSideBySideState() {
    const sideBySide = isSideBySide();
    // `.is-hidden` only flips visibility, never `display` -- see the
    // rule in app.css for why removing the handle from flow makes the
    // layout oscillate across the wrap threshold.
    resizer.classList.toggle('is-hidden', !sideBySide);
    if (!sideBySide) {
      // Clear the override so the CSS defaults resume when the layout
      // wraps back to stacked -- otherwise a resize made while wide
      // would stick around, meaninglessly, once stacked.
      mapEl.style.flex = '';
    }
  }

  function setMapBasisPx(px) {
    const layoutWidth = layout.getBoundingClientRect().width;
    const resizerWidth = resizer.getBoundingClientRect().width;
    const maxPx = Math.max(MIN_PANE_PX, layoutWidth - resizerWidth - MIN_PANE_PX);
    const clamped = Math.min(Math.max(px, MIN_PANE_PX), maxPx);
    // `0 0 <px>` (not just flex-basis) zeroes out grow/shrink on this
    // pane specifically, so the drag result is exactly what was set --
    // the overview pane's own flex:1 absorbs whatever space is left.
    mapEl.style.flex = `0 0 ${clamped}px`;
    resizer.setAttribute(
      'aria-valuenow',
      Math.round((clamped / (layoutWidth - resizerWidth)) * 100),
    );
    scheduleInvalidate();
  }

  let dragging = false;
  resizer.addEventListener('pointerdown', (event) => {
    if (!isSideBySide()) return;
    dragging = true;
    resizer.setPointerCapture(event.pointerId);
  });
  resizer.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    setMapBasisPx(event.clientX - layout.getBoundingClientRect().left);
  });
  resizer.addEventListener('pointerup', (event) => {
    dragging = false;
    resizer.releasePointerCapture(event.pointerId);
  });

  resizer.addEventListener('keydown', (event) => {
    if (!isSideBySide()) return;
    const currentPx = mapEl.getBoundingClientRect().width;
    if (event.key === 'ArrowLeft') {
      setMapBasisPx(currentPx - KEYBOARD_STEP_PX);
      event.preventDefault();
    } else if (event.key === 'ArrowRight') {
      setMapBasisPx(currentPx + KEYBOARD_STEP_PX);
      event.preventDefault();
    }
  });

  updateSideBySideState();
  new ResizeObserver(() => {
    updateSideBySideState();
    scheduleInvalidate();
  }).observe(layout);

  // The map pane gets its own observer, deliberately not the one above:
  // that callback writes `mapEl.style.flex`, so pointing it at `mapEl`
  // would let it feed itself. This one only tells Leaflet the pane
  // resized, which is what a phone's address bar hiding does.
  new ResizeObserver(() => scheduleInvalidate()).observe(mapEl);
})();

let ownTrack = null;
const cursorMarker = L.circleMarker([0, 0], {
  color: RIDAL.cursorColor,
  radius: RIDAL.cursorRadius,
}).addTo(overviewMap);
cursorMarker.setStyle({ opacity: 0 });

const trackToLatLngs = RIDAL.trackToLatLngs;

function fitOverviewToTrack(track) {
  const lines = trackToLatLngs(track);
  const points = lines.flat();
  if (points.length > 0) {
    overviewMap.fitBounds(points);
  } else {
    overviewMap.setView([0, 0], 2);
  }
}

RIDAL.fetchJson(`/api/v1/datasets/${RADARGRAM_ID}/track`)
  .then((track) => {
    ownTrack = track;
    trackToLatLngs(track).forEach((latlngs) => {
      L.polyline(latlngs, {
        color: RIDAL.trackColor,
        weight: RIDAL.trackFocusWeight,
      }).addTo(overviewMap);
    });
    fitOverviewToTrack(track);
  })
  .catch((error) => {
    // Without the track there is no cursor sync and no map extent, so
    // this is worth saying out loud rather than leaving an empty map.
    RIDAL.reportError('overview-map', `Could not load this radargram's track: ${error.message}`);
    overviewMap.setView([0, 0], 2);
  });

if (GROUP) {
  RIDAL.fetchJson(`/api/v1/groups/${GROUP}/tracks`)
    .then((siblings) => {
      for (const [siblingId, info] of Object.entries(siblings)) {
        if (siblingId === RADARGRAM_ID) continue;
        const layers = trackToLatLngs(info.track).map((latlngs) =>
          L.polyline(latlngs, {
            color: RIDAL.siblingColor,
            weight: RIDAL.siblingWeight,
            opacity: RIDAL.siblingOpacity,
          })
            .bindPopup(RIDAL.popupContent(siblingId, info.effective_label))
            .addTo(overviewMap),
        );
        RIDAL.bindTrackHighlight(layers, null, RIDAL.siblingWeight, RIDAL.siblingFocusWeight);
      }
    })
    .catch((error) => {
      // Sibling tracks are context, not the main content: the viewer is
      // still usable without them, so this is a note rather than a
      // replacement for the map.
      RIDAL.reportError('overview-map', `Could not load sibling tracks: ${error.message}`);
    });
}

// --- Cursor sync: mousemove over the radargram viewer moves a marker on
// the overview map, using the same trace-indexed lookup as
// Track::locate_trace (Rust reference implementation in
// src/server/track.rs), so this stays exact regardless of standstills
// or uneven vertex spacing -- the whole reason track.rs stores trace
// indices instead of assuming uniform spacing. ---
function locateInVertices(vertices, traceIndex) {
  if (vertices.length === 0) return null;
  if (vertices.length === 1) return [vertices[0].lat, vertices[0].lon];
  let pos = vertices.findIndex((v) => v.trace_index >= traceIndex);
  let a, b;
  if (pos === -1) { a = vertices.length - 2; b = vertices.length - 1; }
  else if (pos === 0) { a = 0; b = 1; }
  else { a = pos - 1; b = pos; }
  const va = vertices[a], vb = vertices[b];
  const span = vb.trace_index - va.trace_index;
  const t = span > 0 ? Math.min(1, Math.max(0, (traceIndex - va.trace_index) / span)) : 0;
  return [va.lat + t * (vb.lat - va.lat), va.lon + t * (vb.lon - va.lon)];
}

function locateTrace(track, traceIndex) {
  for (const seg of track.segments) {
    if (traceIndex >= seg.trace_start - 1e-6 && traceIndex <= seg.trace_end + 1e-6) {
      return locateInVertices(seg.vertices, traceIndex);
    }
  }
  return null;
}

// --- Axes (distance/TWTT/depth) for the readout, fetched once. Each
// axis degrades independently to null server-side (`/axes`'s contract)
// for fixtures that never wrote it, so the readout below must check
// each one rather than assuming all-or-nothing. ---
let axes = null;
RIDAL.fetchJson(`/api/v1/datasets/${RADARGRAM_ID}/axes`)
  .then((a) => { axes = a; })
  .catch((error) => {
    // The readout degrades to trace-only, which is still useful, so this
    // goes to the console rather than the page.
    console.warn(`Could not load axes: ${error.message}`);
  });

function axisValue(array, index) {
  if (!array) return null;
  const i = Math.round(index);
  if (i < 0 || i >= array.length) return null;
  return array[i];
}

const readout = document.getElementById('cursor-readout');
map.on('mousemove', (event) => {
  const viewerX = event.latlng.lng / xScale;
  const viewerY = -event.latlng.lat;
  if (viewerX < 0 || viewerX > VIEWER_WIDTH || viewerY < 0 || viewerY > VIEWER_HEIGHT) {
    cursorMarker.setStyle({ opacity: 0 });
    readout.textContent = '';
    return;
  }
  const traceIndex = viewerX / RASTER_SCALE;
  const sampleIndex = viewerY / VERTICAL_RASTER_SCALE;

  let text = `trace ${Math.round(traceIndex)} / ${SOURCE_WIDTH}`;
  if (axes) {
    const distance = axisValue(axes.distance, traceIndex);
    const twtt = axisValue(axes.twtt, sampleIndex);
    const depth = axisValue(axes.depth, sampleIndex);
    if (distance !== null) text += ` · ${distance.toFixed(1)} m`;
    if (twtt !== null) text += ` · TWTT ${twtt.toFixed(1)} ns`;
    if (depth !== null) text += ` · depth ${depth.toFixed(1)} m`;
  }
  readout.textContent = text;

  if (ownTrack) {
    const pos = locateTrace(ownTrack, traceIndex);
    if (pos) {
      cursorMarker.setLatLng(pos);
      cursorMarker.setStyle({ opacity: 1 });
    }
  }
});
map.on('mouseout', () => {
  cursorMarker.setStyle({ opacity: 0 });
  readout.textContent = '';
});

// --- Metadata dialog: a button opening a <dialog> with the server's
// curated, human-readable attribute view (prettified labels, merged
// units, rounded floats -- see routes.rs::build_metadata_entries),
// plus the processing steps/log in their own <details>. ---
const dialog = document.getElementById('metadata-dialog');
document.getElementById('metadata-button').addEventListener('click', () => {
  RIDAL.fetchJson(`/api/v1/datasets/${RADARGRAM_ID}/attributes`)
    .then((data) => {
      const tbody = document.querySelector('#metadata-table tbody');
      tbody.innerHTML = '';
      data.entries.forEach(({ label, value }) => {
        const row = document.createElement('tr');
        const keyCell = document.createElement('th');
        keyCell.textContent = label;
        const valCell = document.createElement('td');
        valCell.textContent = value;
        row.append(keyCell, valCell);
        tbody.appendChild(row);
      });

      const stepsList = document.getElementById('processing-steps-list');
      stepsList.innerHTML = '';
      data.processing_steps.forEach((step) => {
        const li = document.createElement('li');
        li.textContent = step;
        stepsList.appendChild(li);
      });

      // The log is `step (duration: Xs):\tdetail\n...` -- split on
      // newlines, then turn each step's embedded tab into its own
      // indented line, so it renders as one line per step detail
      // instead of collapsing into an unreadable run-on paragraph.
      const logLines = data.processing_log
        .split('\n')
        .map((line) => line.replace(/\t/g, '\n  '));
      document.getElementById('processing-log').textContent = logLines.join('\n');

      // Its own <details>, not a metadata-table row: an acquisition
      // that merged many inputs can have an arbitrarily long list of
      // arbitrarily long paths.
      const filepathsList = document.getElementById('filepaths-list');
      filepathsList.innerHTML = '';
      data.original_filepaths.forEach((path) => {
        const li = document.createElement('li');
        li.textContent = path;
        filepathsList.appendChild(li);
      });

      dialog.showModal();
    })
    .catch((error) => {
      // The button did nothing visible on failure before; now the
      // dialog opens and says why, which is the whole point of having a
      // structured error envelope.
      const tbody = document.querySelector('#metadata-table tbody');
      tbody.innerHTML = '';
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 2;
      cell.className = 'error-text';
      cell.textContent = `Could not load metadata: ${error.message}`;
      row.appendChild(cell);
      tbody.appendChild(row);
      dialog.showModal();
    });
});
document.getElementById('metadata-close').addEventListener('click', () => dialog.close());
