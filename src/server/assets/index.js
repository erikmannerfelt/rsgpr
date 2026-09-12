/* Index (catalog) page behaviour (#115, #121).
 *
 * Loaded by index.html.jinja after leaflet.js and app.js. Unlike the
 * viewer, this page needs no server-side values interpolated into JS --
 * everything it needs is already in the DOM as `data-` attributes -- so
 * there is no inline block at all here.
 *
 * Deliberately NOT under assets/vendor/ -- scripts/vendor_leaflet.sh does
 * `rm -rf` on that directory.
 */

// A cataloged entry can still fail to produce a preview: AppState::build
// skips files SourceReader::open rejects, and any render failure is a
// 500. Either way the browser would show a broken-image glyph, so
// degrade to an explicit "no preview" instead (#121: rendering failures
// should be reported to the user).
document.querySelectorAll('.card-thumb img').forEach((img) => {
  img.addEventListener('error', () => img.parentElement.classList.add('is-missing'));
});

// Reload with the chosen profile as a URL query param -- shareable and
// bookmarkable, and reused as-is by every thumbnail/card link the
// template already rendered with it (see entry_card's `profile` arg).
document.getElementById('index-profile-select').addEventListener('change', (event) => {
  const params = new URLSearchParams(location.search);
  params.set('profile', event.target.value);
  location.search = params.toString();
});

// One map per group (#121), each showing every member's track. The
// catalog's target scale (~100 files, #122/#123) keeps this cheap
// enough to load eagerly rather than needing an IntersectionObserver
// lazy-init trick for per-card maps.
document.querySelectorAll('.group-map').forEach((el) => {
  const map = RIDAL.basemap(L.map(el.id));

  RIDAL.fetchJson(RIDAL.apiPath("groups", el.dataset.group, "tracks"))
    .then((members) => {
      const allPoints = [];
      for (const [radargramId, info] of Object.entries(members)) {
        const pairs = RIDAL.trackToLatLngs(info.track).map((latlngs) => {
          allPoints.push(...latlngs);
          // The wide companion goes down first and carries the popup, so a
          // track is as easy to hit as it is to see.
          const hit = RIDAL.hitLine(latlngs)
            // A function, not a string: evaluated when the popup opens, so
            // the thumbnail and the link use whatever profile is selected
            // then.
            .bindPopup(() =>
              RIDAL.popupContent(
                radargramId,
                info.effective_label,
                document.getElementById('index-profile-select').value,
              ),
            )
            .addTo(map);
          const visible = L.polyline(latlngs, {
            color: RIDAL.trackColor,
            weight: RIDAL.trackWeight,
            interactive: false,
          }).addTo(map);
          return { visible, hit };
        });
        // Two-way highlight with the matching catalog card (#121
        // planning round item 7): hovering either one highlights both.
        const card = document.getElementById(`card-${radargramId}`);
        RIDAL.bindTrackHighlight(pairs, card, RIDAL.trackWeight, RIDAL.trackFocusWeight);
      }
      if (allPoints.length > 0) {
        map.fitBounds(allPoints);
      } else {
        map.setView([0, 0], 2);
      }
    })
    .catch((error) => {
      // Previously this left a blank map with no explanation -- the
      // group's cards are still listed below it, so a silent empty map
      // reads as "this group has no tracks" rather than "the request
      // failed".
      RIDAL.reportError(el.id, `Could not load tracks for this group: ${error.message}`);
      map.setView([0, 0], 2);
    });
});

/* --- Merged downloads ----------------------------------------------------
 *
 * One handler for every scope on the page: the catalog-wide menu and one
 * per group. Each menu carries the API prefix its items hang off
 * (`data-download-base`), so a scope is a prefix and nothing else -- adding
 * a saved-selection scope later needs no change here, and a new merged
 * product needs one template line rather than one per scope.
 *
 * The points dialog is shared, with the base it was opened for remembered
 * while it is up: only one can be open at a time, so per-scope copies of
 * the markup would be dead weight.
 *
 * Dismissal (outside click, Escape) comes from app.js, which handles every
 * `.site-menu` on the page.
 */
(function setupMergedDownloads() {
  const dialog = document.getElementById('group-download-dialog');
  const menus = [...document.querySelectorAll('.download-menu[data-download-base]')];
  if (!dialog || menus.length === 0) return;

  const title = document.getElementById('group-download-title');
  const spacing = document.getElementById('group-spacing');
  const format = document.getElementById('group-format');
  let base = null;

  const go = (url) => {
    window.location.href = url;
  };

  for (const menu of menus) {
    const menuBase = menu.dataset.downloadBase;
    const label = menu.dataset.downloadLabel || '';
    for (const button of menu.querySelectorAll('button[data-download]')) {
      const product = button.dataset.download;
      button.addEventListener('click', () => {
        menu.open = false;
        // Everything except level 2 is a plain link: no options to ask for.
        if (product !== 'level2') {
          go(`${menuBase}/${product}`);
          return;
        }
        base = menuBase;
        title.textContent = label
          ? `Download layer points - ${label}`
          : 'Download layer points';
        dialog.showModal();
      });
    }
  }

  document
    .getElementById('group-download-close')
    .addEventListener('click', () => dialog.close());

  document.getElementById('group-download-go').addEventListener('click', () => {
    if (!base) return;
    // Same two-parameters-from-one-choice shape as the viewer's dialog:
    // "GeoJSON in native coordinates" is one decision to a user.
    const choice = format.value;
    const fileFormat = choice === 'csv' ? 'csv' : 'geojson';
    const crs = choice === 'geojson-native' ? '&crs=native' : '';
    dialog.close();
    go(
      `${base}/level2` +
        `?spacing=${encodeURIComponent(spacing.value)}` +
        `&format=${encodeURIComponent(fileFormat)}${crs}`,
    );
  });
})();
