const REFRESH_MS = 2000;

const $ = (sel) => document.querySelector(sel);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function shortId(gid) {
  return gid == null ? "" : "G" + String(gid).slice(-5);
}

function timeOf(iso) {
  return new Date(iso).toLocaleTimeString();
}

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function renderKpis(s) {
  const items = [
    ["Cameras online", `${s.cameras_online} / ${s.cameras}`],
    ["Active tracks", s.active_tracks],
    ["Identities", s.identities_total],
    ["Seen on 2+ cameras", s.identities_multi_camera],
    ["Events (5 min)", s.events_last_5m],
    ["Falls (1 h)", s.falls_last_hour],
  ];
  for (const [zone, n] of Object.entries(s.zone_occupancy)) items.push([`Zone ${zone}`, n]);
  $("#kpis").innerHTML = items
    .map(([label, value]) => `<div class="kpi"><div class="value">${esc(value)}</div><div class="label">${esc(label)}</div></div>`)
    .join("");
}

function renderCameras(cams) {
  $("#cameras tbody").innerHTML = cams
    .map((c) => `<tr>
      <td>${esc(c.name || c.id)}</td>
      <td><span class="badge ${c.online ? "online" : "offline"}">${c.online ? "online" : "offline"}</span></td>
      <td>${c.active_tracks}</td>
      <td>${c.events_last_5m}</td>
    </tr>`)
    .join("") || `<tr><td colspan="4" class="muted">No cameras have reported yet.</td></tr>`;
}

function renderIdentities(ids) {
  $("#identities tbody").innerHTML = ids
    .map((i) => `<tr>
      <td>${esc(shortId(i.global_id))}</td>
      <td>${esc(i.cameras.join(", "))}</td>
      <td>${esc(timeOf(i.last_seen))}</td>
      <td>${i.sightings}</td>
    </tr>`)
    .join("") || `<tr><td colspan="4" class="muted">Nobody matched across cameras yet.</td></tr>`;
}

function details(e) {
  if (e.zone) return `zone ${e.zone}`;
  if (e.event_type === "reid_match") {
    const a = e.attributes;
    return a.matched ? `matched from ${a.from_camera} (sim ${a.similarity})` : "new identity";
  }
  if (e.event_type === "track_end") return `${e.attributes.frames ?? ""} frames`;
  return `conf ${e.confidence.toFixed(2)}`;
}

function renderEvents(events) {
  $("#events tbody").innerHTML = events
    .map((e) => `<tr>
      <td>${esc(timeOf(e.ts))}</td>
      <td>${esc(e.camera_id)}</td>
      <td class="type-${esc(e.event_type)}">${esc(e.event_type.replace("_", " "))}</td>
      <td>#${e.local_track_id}</td>
      <td>${esc(shortId(e.global_id))}</td>
      <td>${esc(details(e))}</td>
    </tr>`)
    .join("") || `<tr><td colspan="6" class="muted">No events.</td></tr>`;
}

async function refresh() {
  const type = $("#type-filter").value;
  const eventsUrl = "/api/v1/events?limit=50" + (type ? `&event_type=${encodeURIComponent(type)}` : "");
  try {
    const [summary, cams, ids, events] = await Promise.all([
      getJSON("/api/v1/stats/summary"),
      getJSON("/api/v1/cameras"),
      getJSON("/api/v1/identities?min_cameras=2&limit=20"),
      getJSON(eventsUrl),
    ]);
    renderKpis(summary);
    renderCameras(cams);
    renderIdentities(ids);
    renderEvents(events);
    $("#updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    $("#updated").textContent = `API unreachable (${err.message})`;
  }
}

$("#type-filter").addEventListener("change", refresh);
refresh();
setInterval(refresh, REFRESH_MS);
