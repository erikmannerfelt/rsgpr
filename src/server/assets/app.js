/* Ridal web GUI shared constants and helpers (#115, #121).
 *
 * First-party, embedded in the binary via assets.rs, loaded from
 * base.html.jinja after leaflet.js so both `L` and `RIDAL` are defined
 * before any per-page script runs. A classic script defining one frozen
 * global -- no modules, no build step (#120: production must not require a
 * separate Node dev server).
 *
 * Deliberately NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does
 * `rm -rf` on that directory.
 *
 * These colours are literal hex rather than CSS custom properties on
 * purpose: they are drawn onto satellite imagery, which looks the same in
 * either page theme, so they must not follow prefers-color-scheme.
 */

const RIDAL = Object.freeze({
  // A radargram's own track, on the index group maps and the viewer.
  trackColor: "#e63",
  trackWeight: 3,
  // The viewer draws its *own* track heavier than the index does, to
  // distinguish it from the sibling tracks beside it. Previously this was
  // an accidental 3-vs-4 discrepancy between two copy-pasted blocks; it is
  // now a named, intentional distinction.
  trackFocusWeight: 4,

  // Other radargrams in the same group, shown for context on the viewer.
  siblingColor: "#bbb",
  siblingWeight: 3,
  siblingOpacity: 0.7,
  // Weight while a sibling's track is hovered or its popup is open --
  // mirrors trackFocusWeight's role for the index page's own tracks.
  siblingFocusWeight: 5,

  // Marker tracking the cursor's trace position along the track.
  cursorColor: "#ff3b30",
  cursorRadius: 6,

  tileUrl:
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  tileAttribution: "Esri",
  tileMaxZoom: 18,

  /** Fetch JSON, turning a non-2xx response into a rejection carrying
   * the server's own message.
   *
   * Every API route answers failures with the same envelope (#120):
   * `{"error": {"code", "message"}}`. A bare `fetch(...).then(r =>
   * r.json())` throws that away -- a 500 parses as JSON perfectly well,
   * so the caller silently proceeds with an object that has no `track`
   * or `entries` field and fails later, somewhere unrelated. This
   * surfaces the message the server already took the trouble to write.
   *
   * The `code` is attached to the Error so a caller can branch on it
   * without string-matching the human-readable message. */
  async fetchJson(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (networkError) {
      // fetch() rejects only on network-level failure, where there is no
      // response and therefore no envelope to read.
      throw new Error(`network request failed (${networkError.message})`);
    }

    let body = null;
    try {
      body = await response.json();
    } catch {
      // A non-JSON body is itself the problem when the status is bad;
      // when the status is fine it means the route broke its contract.
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      throw new Error('response was not valid JSON');
    }

    if (!response.ok) {
      const envelope = body && body.error;
      const error = new Error(
        (envelope && envelope.message) || `HTTP ${response.status}`,
      );
      error.code = (envelope && envelope.code) || null;
      error.status = response.status;
      throw error;
    }
    return body;
  },

  /** Show `message` over the element with id `hostId`, as a dismissible
   * overlay.
   *
   * Failures used to be invisible: a failed `/track` left a blank map
   * that reads as "no data here" rather than "the request failed". The
   * overlay sits on the element that would otherwise be mysteriously
   * empty, so the explanation is where the user is already looking. */
  reportError(hostId, message) {
    const host = document.getElementById(hostId);
    if (!host) {
      console.error(message);
      return;
    }
    // The host is usually a Leaflet container, which is positioned;
    // guard the case where it is not so the overlay cannot escape it.
    if (getComputedStyle(host).position === 'static') {
      host.style.position = 'relative';
    }
    const box = document.createElement('div');
    box.className = 'error-overlay';
    box.setAttribute('role', 'status');

    const text = document.createElement('span');
    text.textContent = message;

    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'error-overlay-dismiss';
    dismiss.textContent = '×';
    dismiss.setAttribute('aria-label', 'Dismiss');
    dismiss.addEventListener('click', () => box.remove());

    box.append(text, dismiss);
    host.appendChild(box);
  },

  /** Add the shared basemap layer to `map` and return it. */
  basemap(map) {
    L.tileLayer(RIDAL.tileUrl, {
      maxZoom: RIDAL.tileMaxZoom,
      attribution: RIDAL.tileAttribution,
    }).addTo(map);
    return map;
  },

  /** Latitude/longitude pairs for every vertex, per track segment. */
  trackToLatLngs(track) {
    return track.segments.map((seg) => seg.vertices.map((v) => [v.lat, v.lon]));
  },

  /** One track's popup content: a link wrapping both the label and a
   * lazy-loaded overview thumbnail, so clicking the image navigates just
   * like clicking the label does -- matching PFA_website's
   * overview_map.js, but with the image inside the anchor rather than
   * beside it. The near-opaque popup background that makes this legible
   * over arbitrary basemap imagery comes from app.css's
   * .leaflet-popup-content-wrapper rule, not from anything here. */
  /** A track popup: the radargram's label, a thumbnail, and a link to it.
   *
   * `profile` applies to *both* -- the link, so arriving at the radargram
   * keeps the profile being browsed in, and the thumbnail, which is a
   * render and otherwise comes back in the default profile regardless of
   * what the rest of the page is showing.
   *
   * Callers should pass this to `bindPopup` as a function rather than a
   * string, so the profile is read when the popup opens. The viewer's
   * profile can change without a page reload, and a popup built at load
   * time would keep showing the profile that was active then. */
  popupContent(radargramId, label, profile) {
    const query = profile ? `?profile=${encodeURIComponent(profile)}` : "";
    return (
      `<a class="popup-link" href="/view/${radargramId}${query}">` +
      `${label}` +
      `<img class="popup-thumb" src="/api/v1/datasets/${radargramId}/views/standard/overview${query}" ` +
      'loading="lazy" alt="">' +
      '</a>'
    );
  },

  /** Wire up a track's hover/popup highlighting, and -- if `card` is
   * given -- two-way highlighting with its catalog card: hovering either
   * the track or the card highlights both, and the track's own popup
   * being open counts as "highlighted" too (PFA_website's
   * popupopen/popupclose pattern), so the two highlight sources agree
   * rather than fighting over the layer's weight when one ends before
   * the other. `layers` is an array because one track can be several
   * polyline segments. */
  /** How wide, in pixels, the invisible strip along a line that accepts a
   * tap or hover.
   *
   * A Leaflet polyline is only interactive within its own stroke, so a 3px
   * track has a 3px target -- unusable with a finger and fiddly with a
   * mouse. Every interactive line therefore gets a transparent companion of
   * this width. Roughly a fingertip on touch, a comfortable aim otherwise. */
  hitWidth: window.matchMedia("(pointer: coarse)").matches ? 34 : 14,

  /** A transparent, interactive companion for `latlngs`.
   *
   * Add it *before* the visible line so the visible one draws on top, and
   * put every handler on this rather than on the line it shadows -- a
   * visible line left interactive would swallow events aimed at the easier
   * target. */
  hitLine(latlngs) {
    return L.polyline(latlngs, {
      className: "hit-line",
      weight: RIDAL.hitWidth,
      opacity: 0,
      interactive: true,
    });
  },

  /** Two-way hover/popup highlighting for a set of track lines.
   *
   * Takes `{ visible, hit }` pairs: events come from the wide companion,
   * while the weight change is applied to the line that can actually be
   * seen. */
  bindTrackHighlight(pairs, card, baseWeight, focusWeight) {
    let hovered = false;
    let popupOpen = false;
    const apply = () => {
      const on = hovered || popupOpen;
      pairs.forEach(({ visible, hit }) => {
        visible.setStyle({ weight: on ? focusWeight : baseWeight });
        if (on) {
          // Order matters: the companion first, so the visible line still
          // ends up above it.
          hit.bringToFront();
          visible.bringToFront();
        }
      });
      if (card) card.classList.toggle("is-hovered", on);
    };
    pairs.forEach(({ hit }) => {
      const layer = hit;
      layer.on("mouseover", () => { hovered = true; apply(); });
      layer.on("mouseout", () => { hovered = false; apply(); });
      layer.on("popupopen", () => { popupOpen = true; apply(); });
      layer.on("popupclose", () => { popupOpen = false; apply(); });
    });
    if (card) {
      card.addEventListener("mouseenter", () => { hovered = true; apply(); });
      card.addEventListener("mouseleave", () => { hovered = false; apply(); });
    }
  },
});

/* Dismiss the header menu on Escape or a click outside it.
 *
 * `<details>` gives the disclosure, the keyboard behaviour and the open
 * state for free, but it stays open until its own summary is clicked again,
 * which is wrong for a menu: tapping the page elsewhere should close it.
 * That is the only reason this file knows the menu exists. */
(function setupSiteMenu() {
  const menu = document.getElementById("site-menu");
  if (!menu) return;

  document.addEventListener("click", (event) => {
    if (menu.open && !menu.contains(event.target)) menu.open = false;
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menu.open) {
      menu.open = false;
      // Return focus to the control that opened it, or the close is
      // invisible to a keyboard user.
      menu.querySelector("summary").focus();
    }
  });
})();
