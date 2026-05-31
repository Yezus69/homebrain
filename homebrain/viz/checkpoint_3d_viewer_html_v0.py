from __future__ import annotations


def checkpoint_3d_viewer_html(scene_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HomeBrain Checkpoint 3D Visualizer V0</title>
<style>
html, body {{ margin:0; height:100%; overflow:hidden; font-family:Arial,sans-serif; color:#18202a; background:#f5f6f8; }}
#app {{ display:grid; grid-template-columns:minmax(620px,1fr) 430px; height:100vh; }}
#stage {{ position:relative; background:#e8ebef; }}
#scene {{ width:100%; height:100%; display:block; background:linear-gradient(#f8fafc,#dde4ec); cursor:grab; }}
#scene:active {{ cursor:grabbing; }}
#controls {{ position:absolute; left:12px; right:12px; bottom:12px; padding:10px; background:rgba(255,255,255,0.94); border:1px solid #bcc7d4; box-shadow:0 4px 18px rgba(20,30,40,0.12); }}
#timeline {{ width:100%; }}
#layers {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:4px 10px; font-size:12px; margin-top:8px; }}
#side {{ overflow:auto; padding:14px; border-left:1px solid #c7d0dc; background:#fff; }}
button, select {{ height:28px; }}
pre {{ white-space:pre-wrap; font-size:12px; line-height:1.35; }}
h2 {{ margin:12px 0 8px; font-size:22px; }}
h3 {{ margin:14px 0 8px; font-size:14px; }}
#rgb-frame {{ width:100%; aspect-ratio:16/10; object-fit:cover; border:1px solid #b9c3cf; background:#111827; display:block; }}
#rgb-caption {{ margin-top:6px; font-size:12px; line-height:1.35; color:#536070; }}
#rgb-link {{ display:inline-block; margin-top:4px; font-size:12px; color:#1d5e9f; }}
#status-panel {{ margin:8px 0 10px; padding:8px; background:#f2f5f8; border:1px solid #c7d0dc; font-size:12px; line-height:1.45; white-space:pre-line; }}
#warning-banner {{ display:none; margin:8px 0; padding:8px; background:#fff3cd; border:1px solid #d5a628; color:#604300; font-size:12px; line-height:1.35; }}
details {{ margin-top:12px; }}
summary {{ cursor:pointer; font-weight:bold; font-size:13px; }}
.row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.legend {{ display:flex; gap:10px; margin:8px 0 10px; font-size:12px; flex-wrap:wrap; }}
.swatch {{ display:inline-block; width:10px; height:10px; margin-right:4px; border:1px solid rgba(0,0,0,0.2); vertical-align:-1px; }}
</style>
</head>
<body>
<div id="app">
  <main id="stage">
    <canvas id="scene"></canvas>
    <div id="controls">
      <div class="row">
        <button id="play-pause" type="button">Play</button>
        <button id="reset-view" type="button">Reset View</button>
        <label>View <select id="view-mode"><option value="auto" selected>auto</option><option value="camera-frame">camera frame</option><option value="map-frame">stitched map</option></select></label>
        <label>Map mode <select id="map-mode"><option value="incremental" selected>incremental</option><option value="final">final</option></select></label>
        <span id="frame-label"></span>
      </div>
      <input id="timeline" type="range" min="0" max="0" value="0">
      <div id="layers">
        {''.join(_checkbox(item, label) for item, label in _layer_controls())}
      </div>
    </div>
  </main>
  <aside id="side">
    <h2>Checkpoint Replay</h2>
    <div class="legend">
      <span><i class="swatch" style="background:#6fb87d"></i>free</span>
      <span><i class="swatch" style="background:#c84d46"></i>obstacle</span>
      <span><i class="swatch" style="background:#9aa3ad"></i>unknown</span>
      <span><i class="swatch" style="background:#e3ad3f"></i>risk</span>
      <span><i class="swatch" style="background:#c84897"></i>hazard</span>
    </div>
    <div id="status-panel"></div>
    <div id="warning-banner"></div>
    <h3>RGB Reference Frame</h3>
    <img id="rgb-frame" alt="RGB reference frame">
    <div id="rgb-caption"></div>
    <a id="rgb-link" href="#" target="_blank" rel="noreferrer">Open RGB frame</a>
    <details id="raw-details">
      <summary>Raw Replay State</summary>
      <pre id="side-panel"></pre>
    </details>
  </aside>
</div>
<script id="scene-data" type="application/json">{scene_json}</script>
<script>
const data = JSON.parse(document.getElementById('scene-data').textContent);
const frames = data.frames || [];
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('timeline');
const mode = document.getElementById('map-mode');
const viewMode = document.getElementById('view-mode');
const play = document.getElementById('play-pause');
const resetView = document.getElementById('reset-view');
const label = document.getElementById('frame-label');
const side = document.getElementById('side-panel');
const rgbImage = document.getElementById('rgb-frame');
const rgbCaption = document.getElementById('rgb-caption');
const rgbLink = document.getElementById('rgb-link');
const statusPanel = document.getElementById('status-panel');
const warningBanner = document.getElementById('warning-banner');
let frameIndex = 0;
let timer = null;
let dragging = false;
let lastMouse = [0, 0];
const camera3d = {{ yaw: -0.72, pitch: 0.88, distance: 4.2, target: [0.8, 0.0, 0.0] }};
slider.max = Math.max(0, frames.length - 1);
const requestedFrame = Number(new URLSearchParams(window.location.search).get('frame') || 0);
frameIndex = Math.max(0, Math.min(frames.length - 1, Number.isFinite(requestedFrame) ? Math.round(requestedFrame) : 0));
slider.value = frameIndex;

function enabled(id) {{
  const el = document.getElementById(id);
  return !el || el.checked;
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function activeMap(frame) {{
  if (mode.value === 'final') return data.accumulated_scene?.final_map || frame.accumulated_map_summary || {{}};
  return frame.accumulated_map_summary || {{}};
}}

function effectiveViewMode() {{
  if (viewMode.value !== 'auto') return viewMode.value;
  return data.summary?.camera_extrinsics_assumed ? 'camera-frame' : 'map-frame';
}}

function resetCamera() {{
  if (effectiveViewMode() === 'camera-frame') {{
    camera3d.target = [0.95, -0.08, 0.04];
    camera3d.distance = 2.05;
    camera3d.yaw = -0.68;
    camera3d.pitch = 0.72;
    return;
  }}
  const finalMap = data.accumulated_scene?.final_map || activeMap(frames[0] || {{}});
  const b = finalMap.global_bounds_m || {{ min_x: -1, max_x: 2, min_y: -1, max_y: 1 }};
  camera3d.target = [(b.min_x + b.max_x) / 2, (b.min_y + b.max_y) / 2, 0.08];
  const span = Math.max(1.0, b.max_x - b.min_x, b.max_y - b.min_y);
  camera3d.distance = span * 1.2 + 0.9;
  camera3d.yaw = -0.74;
  camera3d.pitch = 0.86;
}}

function worldToScreen(point) {{
  const x = point[0] - camera3d.target[0];
  const y = point[1] - camera3d.target[1];
  const z = point[2] - camera3d.target[2];
  const cy = Math.cos(camera3d.yaw), sy = Math.sin(camera3d.yaw);
  const cp = Math.cos(camera3d.pitch), sp = Math.sin(camera3d.pitch);
  const rx = cy * x - sy * y;
  const ry = sy * x + cy * y;
  const rz = z;
  const vy = cp * ry - sp * rz;
  const vz = sp * ry + cp * rz;
  const depth = Math.max(0.2, camera3d.distance + vy);
  const focal = Math.min(canvas.clientWidth, canvas.clientHeight) * 1.18;
  const scale = focal / depth;
  return {{
    x: canvas.clientWidth / 2 + rx * scale,
    y: canvas.clientHeight * 0.54 - vz * scale,
    depth,
    scale
  }};
}}

function addFace(faces, points, color, alpha, stroke) {{
  const projected = points.map(worldToScreen);
  if (projected.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) return;
  faces.push({{
    points: projected,
    color,
    alpha,
    stroke,
    depth: projected.reduce((sum, p) => sum + p.depth, 0) / projected.length
  }});
}}

function addTile(faces, x, y, z, sx, sy, color, alpha, stroke) {{
  const hx = sx / 2, hy = sy / 2;
  addFace(faces, [[x - hx, y - hy, z], [x + hx, y - hy, z], [x + hx, y + hy, z], [x - hx, y + hy, z]], color, alpha, stroke);
}}

function shade(hex, factor) {{
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, Math.min(255, Math.round(((n >> 16) & 255) * factor)));
  const g = Math.max(0, Math.min(255, Math.round(((n >> 8) & 255) * factor)));
  const b = Math.max(0, Math.min(255, Math.round((n & 255) * factor)));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function addBox(faces, x, y, z, sx, sy, h, color, alpha) {{
  const hx = sx / 2, hy = sy / 2, z1 = z + h;
  const v = {{
    a: [x - hx, y - hy, z], b: [x + hx, y - hy, z], c: [x + hx, y + hy, z], d: [x - hx, y + hy, z],
    e: [x - hx, y - hy, z1], f: [x + hx, y - hy, z1], g: [x + hx, y + hy, z1], h: [x - hx, y + hy, z1]
  }};
  addFace(faces, [v.e, v.f, v.g, v.h], shade(color, 1.18), alpha, 'rgba(20,25,30,0.22)');
  addFace(faces, [v.a, v.b, v.f, v.e], shade(color, 0.82), alpha, 'rgba(20,25,30,0.18)');
  addFace(faces, [v.b, v.c, v.g, v.f], shade(color, 0.72), alpha, 'rgba(20,25,30,0.18)');
  addFace(faces, [v.c, v.d, v.h, v.g], shade(color, 0.62), alpha, 'rgba(20,25,30,0.18)');
  addFace(faces, [v.d, v.a, v.e, v.h], shade(color, 0.92), alpha, 'rgba(20,25,30,0.18)');
}}

function drawFace(face) {{
  ctx.beginPath();
  face.points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.closePath();
  ctx.globalAlpha = face.alpha;
  ctx.fillStyle = face.color;
  ctx.fill();
  if (face.stroke) {{
    ctx.globalAlpha = Math.min(1, face.alpha + 0.18);
    ctx.strokeStyle = face.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }}
  ctx.globalAlpha = 1;
}}

function addCells(faces, cells, kind, resolution) {{
  const size = resolution * 0.92;
  for (const c of cells || []) {{
    if (kind === 'free') addTile(faces, c.x_m, c.y_m, 0.0, size, size, '#6fb87d', 0.58, 'rgba(40,90,55,0.22)');
    if (kind === 'unknown') addBox(faces, c.x_m, c.y_m, 0.0, size, size, 0.04, '#9aa3ad', 0.34);
    if (kind === 'risky') addTile(faces, c.x_m, c.y_m, 0.025, size, size, '#e3ad3f', 0.7, 'rgba(120,75,10,0.22)');
    if (kind === 'hazard') addTile(faces, c.x_m, c.y_m, 0.055, size, size, '#c84897', 0.78, 'rgba(110,25,78,0.3)');
    if (kind === 'stale') addBox(faces, c.x_m, c.y_m, 0.0, size, size, 0.08, '#5f8fd3', 0.34);
    if (kind === 'obstacle') addBox(faces, c.x_m, c.y_m, 0.0, size, size, 0.36, '#c84d46', 0.92);
  }}
}}

function transformPoint(matrix, point) {{
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
    matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3]
  ];
}}

function localCellCorners(local, row, col, z) {{
  const channels = local.channels || {{}};
  const first = Object.values(channels).find(v => Array.isArray(v) && Array.isArray(v[0]));
  if (!first) return null;
  const rows = first.length, cols = first[0].length, r = local.resolution_m || 0.25;
  const forwardNear = (rows - row - 1) * r, forwardFar = (rows - row) * r;
  const leftA = (col - cols / 2) * r, leftB = (col + 1 - cols / 2) * r;
  const m = local.T_map_base?.matrix_4x4;
  if (!m) return null;
  return [[forwardNear, leftA, z], [forwardFar, leftA, z], [forwardFar, leftB, z], [forwardNear, leftB, z]].map(p => transformPoint(m, p));
}}

function addLocalBevCells(faces, frame) {{
  const local = frame.local_bev;
  if (!local || !local.channels) return;
  const specs = [
    ['free', '#2f8f5f', 0.28, 0.018],
    ['unknown', '#606975', 0.24, 0.07],
    ['risky', '#b7791f', 0.42, 0.09],
    ['hazard', '#a21caf', 0.46, 0.12],
    ['obstacle', '#7f1d1d', 0.5, 0.42]
  ];
  for (const [name, color, alpha, z] of specs) {{
    const channel = local.channels[name] || (name === 'obstacle' ? local.channels.occupied : null);
    if (!Array.isArray(channel)) continue;
    for (let row = 0; row < channel.length; row++) {{
      for (let col = 0; col < (channel[row] || []).length; col++) {{
        if (Number(channel[row][col]) <= 0.5) continue;
        const corners = localCellCorners(local, row, col, z);
        if (corners) addFace(faces, corners, color, alpha, 'rgba(20,25,30,0.38)');
      }}
    }}
  }}
}}

function cameraPointToViewer(point) {{
  return [point[2], point[0], -point[1]];
}}

function drawPointCloud(points, cameraFrame = false) {{
  if (!points || !points.length || !enabled('layer-depth-point-cloud')) return;
  const stride = Math.max(1, Math.ceil(points.length / 9000));
  ctx.globalAlpha = 0.76;
  for (let i = 0; i < points.length; i += stride) {{
    const raw = points[i];
    const p = worldToScreen(cameraFrame ? cameraPointToViewer(raw) : raw);
    const size = Math.max(1.2, Math.min(3.2, 10 / p.depth));
    if (raw.length >= 6) ctx.fillStyle = `rgb(${{raw[3]}},${{raw[4]}},${{raw[5]}})`;
    else ctx.fillStyle = '#334155';
    ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size);
  }}
  ctx.globalAlpha = 1;
}}

function drawLine3D(a, b, color, width, alpha = 1) {{
  const p = worldToScreen(a), q = worldToScreen(b);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(q.x, q.y);
  ctx.stroke();
  ctx.globalAlpha = 1;
}}

function drawPolyline(points, color, width, alpha = 1) {{
  if (!points || points.length < 2) return;
  for (let i = 0; i < points.length - 1; i++) drawLine3D(points[i], points[i + 1], color, width, alpha);
}}

function drawGrid(bounds) {{
  const minX = Math.floor((bounds.min_x - 0.5) * 2) / 2;
  const maxX = Math.ceil((bounds.max_x + 0.5) * 2) / 2;
  const minY = Math.floor((bounds.min_y - 0.5) * 2) / 2;
  const maxY = Math.ceil((bounds.max_y + 0.5) * 2) / 2;
  for (let x = minX; x <= maxX; x += 0.25) drawLine3D([x, minY, -0.004], [x, maxY, -0.004], '#cbd3dc', x === 0 ? 1.4 : 0.6, 0.55);
  for (let y = minY; y <= maxY; y += 0.25) drawLine3D([minX, y, -0.004], [maxX, y, -0.004], '#cbd3dc', y === 0 ? 1.4 : 0.6, 0.55);
}}

function drawCameraFrameGrid() {{
  for (let x = 0; x <= 4.5; x += 0.25) drawLine3D([x, -2.2, -0.004], [x, 2.2, -0.004], '#d6dde6', x === 0 ? 1.2 : 0.6, 0.58);
  for (let y = -2.2; y <= 2.2; y += 0.25) drawLine3D([0, y, -0.004], [4.5, y, -0.004], '#d6dde6', y === 0 ? 1.2 : 0.6, 0.58);
  drawLine3D([0, 0, 0.04], [0.45, 0, 0.04], '#111827', 3, 0.95);
  drawLine3D([0, 0, 0.04], [0, 0.35, 0.04], '#2f7ed8', 2.4, 0.9);
  drawLine3D([0, 0, 0.04], [0, 0, 0.38], '#6fb87d', 2.4, 0.9);
}}

function drawRobot(frame) {{
  const robot = frame.robot_pose;
  if (!robot) return;
  const poly = robot.footprint_polygon_map || [];
  if (poly.length > 1) drawPolyline(poly.map(p => [p[0], p[1], 0.07]), '#111827', 2.5, 0.95);
  for (const line of robot.base_axes_lines_map || []) {{
    drawLine3D(line[0], line[1], '#111827', 2.5, 0.9);
  }}
}}

function drawFrameLines(frame, acc) {{
  if (enabled('layer-robot-trajectory')) drawPolyline((acc.trajectory_points || []).map(p => [p[0], p[1], 0.08]), '#202936', 3, 0.95);
  if (enabled('layer-camera-frustum') && frame.camera_frustum) {{
    for (const seg of frame.camera_frustum.frustum_lines_map || []) drawLine3D(seg[0], seg[1], '#2f7ed8', 1.6, 0.74);
  }}
  if (enabled('layer-candidate-trajectories')) {{
    for (const c of frame.candidate_trajectories || []) {{
      if (c.selected && !enabled('layer-selected-candidate')) continue;
      const color = c.selected ? '#111827' : (c.vetoed ? '#bd2f2f' : '#687382');
      drawPolyline((c.points_map || []).map(p => [p[0], p[1], (p[2] || 0) + 0.09]), color, c.selected ? 3.6 : 2, c.vetoed ? 0.72 : 0.9);
    }}
  }}
  if (enabled('layer-safety-stops') && frame.safety_debug && (frame.safety_debug.stop_reasons || []).length > 0 && frame.robot_pose) {{
    const c = frame.robot_pose.T_map_base.matrix_4x4;
    const x = c[0][3], y = c[1][3];
    drawLine3D([x - 0.18, y - 0.18, 0.18], [x + 0.18, y + 0.18, 0.18], '#b91c1c', 4, 0.95);
    drawLine3D([x - 0.18, y + 0.18, 0.18], [x + 0.18, y - 0.18, 0.18], '#b91c1c', 4, 0.95);
  }}
  drawRobot(frame);
}}

function drawScene() {{
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const frame = frames[frameIndex] || {{}};
  const acc = activeMap(frame);
  if (effectiveViewMode() === 'camera-frame') {{
    drawCameraFrameGrid();
    drawPointCloud(frame.depth_point_cloud_camera || [], true);
    label.textContent = `Frame ${{frameIndex + 1}} / ${{frames.length}} timestamp=${{frame.timestamp_ns ?? 'null'}}`;
    renderSidePanel(frame, acc);
    return;
  }}
  const bounds = acc.global_bounds_m || {{ min_x: -1, max_x: 2, min_y: -1, max_y: 1 }};
  const resolution = acc.map_resolution_m || 0.25;
  const faces = [];
  if (enabled('layer-accumulated-free')) addCells(faces, acc.free_cells, 'free', resolution);
  if (enabled('layer-accumulated-unknown')) addCells(faces, acc.unknown_cells, 'unknown', resolution);
  if (enabled('layer-accumulated-risky')) addCells(faces, acc.risky_cells, 'risky', resolution);
  if (enabled('layer-accumulated-hazard')) addCells(faces, acc.hazard_cells, 'hazard', resolution);
  if (enabled('layer-stale-memory')) addCells(faces, acc.stale_cells, 'stale', resolution);
  if (enabled('layer-accumulated-obstacle')) addCells(faces, acc.obstacle_cells, 'obstacle', resolution);
  if (enabled('layer-local-bev')) addLocalBevCells(faces, frame);
  drawGrid(bounds);
  faces.sort((a, b) => b.depth - a.depth);
  faces.forEach(drawFace);
  drawPointCloud(acc.point_cloud || []);
  drawFrameLines(frame, acc);
  label.textContent = `Frame ${{frameIndex + 1}} / ${{frames.length}} timestamp=${{frame.timestamp_ns ?? 'null'}}`;
  renderSidePanel(frame, acc);
}}

function renderRgbFrame(frame) {{
  const rgb = frame.rgb_frame || {{}};
  const src = rgb.data_uri || rgb.relative_path || rgb.src || '';
  if (src) {{
    rgbImage.dataset.expectedSrc = src;
    rgbImage.src = src;
    rgbImage.style.display = 'block';
    rgbLink.href = src;
    rgbLink.style.display = 'inline-block';
    const availability = rgb.available === false ? 'missing source' : 'available';
    rgbCaption.textContent = `${{rgb.kind || 'rgb_frame'}} frame=${{rgb.frame_id ?? frame.frame_id}} ${{availability}}`;
  }} else {{
    rgbImage.removeAttribute('src');
    rgbImage.style.display = 'none';
    rgbLink.removeAttribute('href');
    rgbLink.style.display = 'none';
    rgbCaption.textContent = 'No RGB/reference frame was exported for this replay frame.';
  }}
}}

function renderSidePanel(frame, acc) {{
  renderRgbFrame(frame);
  renderStatusPanel(frame, acc);
  const selected = (frame.candidate_trajectories || []).find(c => c.selected);
  side.textContent = JSON.stringify({{
    route_id: data.summary?.route_id,
    frame_id: frame.frame_id,
    timestamp_ns: frame.timestamp_ns,
    pose_source: data.summary?.pose_source,
    geometry_source: data.summary?.geometry_source,
    dense_3d_claimed: data.summary?.dense_3d_claimed,
    camera_pose_rendered: data.summary?.camera_pose_rendered,
    rgb_frame_kind: frame.rgb_frame?.kind,
    rgb_frame_available: frame.rgb_frame?.available ?? Boolean(frame.rgb_frame?.data_uri || frame.rgb_frame?.src),
    model_channels_present: data.summary?.model_channels_rendered,
    model_channels_missing: data.summary?.model_channels_missing || [],
    teacher_overlay_enabled: data.summary?.teacher_overlay_enabled,
    overlay_is_runtime_truth: data.summary?.overlay_is_runtime_truth,
    map_mode: mode.value,
    selected_candidate_id: selected?.candidate_id,
    selected_candidate_score: selected?.score,
    selected_candidate_risk: selected?.risk,
    stop_reasons: frame.safety_debug?.stop_reasons || [],
    warnings: frame.warnings || [],
    safety: data.summary?.safety
  }}, null, 2);
}}

function renderStatusPanel(frame, acc) {{
  const warnings = [...(data.summary?.hard_failures || []), ...(data.summary?.warnings || []), ...(frame.warnings || [])];
  const pointCount = (acc.point_cloud || []).length;
  const obstacleCount = (acc.obstacle_cells || []).length;
  statusPanel.textContent = `route=${{data.summary?.route_id}} | frame=${{frame.frame_id}}/${{frames.length - 1}}
geometry=${{data.summary?.geometry_source}} | view=${{effectiveViewMode()}} | rgb=${{frame.rgb_frame?.available ? 'real' : 'missing'}}
points=${{effectiveViewMode() === 'camera-frame' ? (frame.depth_point_cloud_camera || []).length : pointCount}} | obstacle_cells=${{obstacleCount}} | timeline=${{mode.value}}`;
  if (warnings.length) {{
    warningBanner.style.display = 'block';
    warningBanner.textContent = `Accuracy warning: ${{Array.from(new Set(warnings)).join(', ')}}`;
  }} else {{
    warningBanner.style.display = 'none';
    warningBanner.textContent = '';
  }}
}}

function draw() {{ drawScene(); }}

window.addEventListener('resize', resize);
if (data.summary?.geometry_source === 'depth_pointcloud') {{
  if (data.summary?.camera_extrinsics_assumed) viewMode.value = 'camera-frame';
  const localLayer = document.getElementById('layer-local-bev');
  if (localLayer) localLayer.checked = false;
  const obstacleLayer = document.getElementById('layer-accumulated-obstacle');
  if (obstacleLayer) obstacleLayer.checked = false;
  const unknownLayer = document.getElementById('layer-accumulated-unknown');
  if (unknownLayer) unknownLayer.checked = false;
}}
slider.addEventListener('input', () => {{ frameIndex = Number(slider.value); draw(); }});
mode.addEventListener('change', draw);
viewMode.addEventListener('change', () => {{ resetCamera(); draw(); }});
resetView.addEventListener('click', () => {{ resetCamera(); draw(); }});
document.querySelectorAll('#layers input').forEach(input => input.addEventListener('change', draw));
play.addEventListener('click', () => {{
  if (timer) {{ clearInterval(timer); timer = null; play.textContent = 'Play'; return; }}
  play.textContent = 'Pause';
  timer = setInterval(() => {{ frameIndex = (frameIndex + 1) % Math.max(1, frames.length); slider.value = frameIndex; draw(); }}, 450);
}});
canvas.addEventListener('mousedown', event => {{ dragging = true; lastMouse = [event.clientX, event.clientY]; }});
window.addEventListener('mouseup', () => {{ dragging = false; }});
window.addEventListener('mousemove', event => {{
  if (!dragging) return;
  const dx = event.clientX - lastMouse[0];
  const dy = event.clientY - lastMouse[1];
  lastMouse = [event.clientX, event.clientY];
  camera3d.yaw += dx * 0.006;
  camera3d.pitch = Math.max(0.25, Math.min(1.35, camera3d.pitch + dy * 0.005));
  draw();
}});
canvas.addEventListener('wheel', event => {{
  event.preventDefault();
  camera3d.distance = Math.max(0.8, Math.min(20, camera3d.distance * (event.deltaY > 0 ? 1.08 : 0.92)));
  draw();
}}, {{ passive: false }});
rgbImage.addEventListener('error', () => {{
  if (rgbImage.getAttribute('src') === rgbImage.dataset.expectedSrc) {{
    rgbCaption.textContent = `${{rgbCaption.textContent}} (image failed to load; use Open RGB frame)`;
  }}
}});
resetCamera();
resize();
</script>
</body>
</html>
"""


def _checkbox(item: str, label: str) -> str:
    return f'<label><input id="{item}" type="checkbox" checked> {label}</label>'


def _layer_controls() -> list[tuple[str, str]]:
    return [
        ("layer-robot-trajectory", "robot trajectory"),
        ("layer-camera-frustum", "camera frustum"),
        ("layer-depth-point-cloud", "depth point cloud"),
        ("layer-local-bev", "local BEV"),
        ("layer-accumulated-free", "accumulated free"),
        ("layer-accumulated-obstacle", "accumulated obstacle"),
        ("layer-accumulated-unknown", "accumulated unknown"),
        ("layer-accumulated-risky", "accumulated risky"),
        ("layer-accumulated-hazard", "accumulated hazard"),
        ("layer-stale-memory", "stale memory"),
        ("layer-candidate-trajectories", "candidate trajectories"),
        ("layer-selected-candidate", "selected candidate"),
        ("layer-safety-stops", "safety stops"),
        ("layer-future-predictions", "future predictions"),
    ]
