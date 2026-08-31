const $ = (selector) => document.querySelector(selector);

const elements = {
  videoInput: $('#videoInput'), dropzone: $('#dropzone'), filePreview: $('#filePreview'),
  inputPreview: $('#inputPreview'), fileName: $('#fileName'), fileMeta: $('#fileMeta'),
  replaceFile: $('#replaceFile'), runButton: $('#runButton'), runHint: $('#runHint'),
  uploadPanel: $('#uploadPanel'), progressPanel: $('#progressPanel'), resultPanel: $('#resultPanel'),
  progressPhase: $('#progressPhase'), progressMessage: $('#progressMessage'),
  progressPercent: $('#progressPercent'), progressBar: $('#progressBar'),
  liveFrames: $('#liveFrames'), liveTracks: $('#liveTracks'), liveVehicles: $('#liveVehicles'), liveFps: $('#liveFps'),
  cancelButton: $('#cancelButton'), resultVideo: $('#resultVideo'), newAnalysis: $('#newAnalysis'),
  downloadVideo: $('#downloadVideo'), downloadTracks: $('#downloadTracks'),
  downloadDetections: $('#downloadDetections'), downloadSummary: $('#downloadSummary'),
  resultStats: $('#resultStats'), classBars: $('#classBars'), trackTable: $('#trackTable'), tableCount: $('#tableCount'),
  measurementMode: $('#measurementMode'), measurementCopy: $('#measurementCopy'),
  confidence: $('#confidence'), confidenceValue: $('#confidenceValue'), iou: $('#iou'), iouValue: $('#iouValue'),
  imageSize: $('#imageSize'), vehicleClassifier: $('#vehicleClassifier'), vehicleConfidence: $('#vehicleConfidence'),
  vehicleConfidenceValue: $('#vehicleConfidenceValue'), vehicleConfidenceRow: $('#vehicleConfidenceRow'),
  drawTrails: $('#drawTrails'), saveDetections: $('#saveDetections'), fov: $('#fov'), fovValue: $('#fovValue'),
  distanceScale: $('#distanceScale'), speedScale: $('#speedScale'), useHomography: $('#useHomography'),
  imagePoints: $('#imagePoints'), worldPoints: $('#worldPoints'), resetSettings: $('#resetSettings'),
  systemPill: $('#systemPill'), systemText: $('#systemText'), recentSection: $('#recentSection'), recentGrid: $('#recentGrid'),
  toast: $('#toast'), toastTitle: $('#toastTitle'), toastMessage: $('#toastMessage')
};

const state = { file: null, fileUrl: null, job: null, pollTimer: null, toastTimer: null };

const formatBytes = (bytes) => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${remaining}`;
};

const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[character]));

function toast(title, message) {
  elements.toastTitle.textContent = title;
  elements.toastMessage.textContent = message;
  elements.toast.classList.remove('hidden');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => elements.toast.classList.add('hidden'), 7000);
}

function bindRange(input, output, formatter = (value) => value) {
  const render = () => { output.value = formatter(input.value); output.textContent = formatter(input.value); };
  input.addEventListener('input', render);
  render();
}

bindRange(elements.confidence, elements.confidenceValue, (value) => Number(value).toFixed(2));
bindRange(elements.iou, elements.iouValue, (value) => Number(value).toFixed(2));
bindRange(elements.vehicleConfidence, elements.vehicleConfidenceValue, (value) => Number(value).toFixed(2));
bindRange(elements.fov, elements.fovValue, (value) => `${value}°`);

function validateVideo(file) {
  if (!file) return false;
  const extension = `.${file.name.split('.').pop()?.toLowerCase()}`;
  const allowed = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'];
  if (!allowed.includes(extension)) {
    toast('Unsupported file', `Choose one of: ${allowed.join(', ')}`);
    return false;
  }
  if (file.size > 2 * 1024 ** 3) {
    toast('Video is too large', 'The local upload limit is 2 GB. You can change MAX_UPLOAD_MB in .env.');
    return false;
  }
  return true;
}

function chooseFile(file) {
  if (!validateVideo(file)) return;
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  elements.inputPreview.src = state.fileUrl;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${formatBytes(file.size)} · reading metadata…`;
  elements.filePreview.classList.remove('hidden');
  elements.dropzone.classList.add('hidden');
  elements.runButton.disabled = false;
  elements.runHint.textContent = 'The first run downloads model weights once';
  elements.inputPreview.onloadedmetadata = () => {
    const width = elements.inputPreview.videoWidth;
    const height = elements.inputPreview.videoHeight;
    elements.fileMeta.textContent = `${formatBytes(file.size)} · ${width}×${height} · ${formatDuration(elements.inputPreview.duration)}`;
    elements.inputPreview.currentTime = Math.min(1, elements.inputPreview.duration / 4 || 0);
  };
}

elements.dropzone.addEventListener('click', () => elements.videoInput.click());
elements.replaceFile.addEventListener('click', () => elements.videoInput.click());
elements.videoInput.addEventListener('change', () => chooseFile(elements.videoInput.files[0]));
['dragenter', 'dragover'].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
  event.preventDefault(); elements.dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
  event.preventDefault(); elements.dropzone.classList.remove('dragging');
}));
elements.dropzone.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));

elements.vehicleClassifier.addEventListener('change', () => {
  elements.vehicleConfidenceRow.style.opacity = elements.vehicleClassifier.checked ? '1' : '.38';
});

function parsePoints(text, label) {
  let value;
  try { value = JSON.parse(text); } catch { throw new Error(`${label} must be valid JSON.`); }
  if (!Array.isArray(value) || value.length !== 4 || value.some((point) => !Array.isArray(point) || point.length !== 2)) {
    throw new Error(`${label} must contain exactly four [x, y] pairs.`);
  }
  return value.map((point) => point.map(Number));
}

function collectOptions() {
  const options = {
    confidence: Number(elements.confidence.value), iou: Number(elements.iou.value), image_size: Number(elements.imageSize.value),
    enable_vehicle_classifier: elements.vehicleClassifier.checked,
    vehicle_classifier_confidence: Number(elements.vehicleConfidence.value),
    draw_trails: elements.drawTrails.checked, save_frame_detections: elements.saveDetections.checked,
    horizontal_fov_degrees: Number(elements.fov.value), distance_scale: Number(elements.distanceScale.value),
    speed_scale: Number(elements.speedScale.value), calibration_mode: elements.useHomography.checked ? 'homography' : 'fov'
  };
  if (elements.useHomography.checked) {
    options.image_points = parsePoints(elements.imagePoints.value, 'Image points');
    options.world_points = parsePoints(elements.worldPoints.value, 'Ground points');
  }
  return options;
}

function setProgress(percent, phase, message) {
  const safe = Math.max(0, Math.min(100, Number(percent) || 0));
  elements.progressBar.style.width = `${safe}%`;
  elements.progressPercent.textContent = `${Math.round(safe)}%`;
  elements.progressPhase.textContent = phase;
  elements.progressMessage.textContent = message;
}

function updateLiveStats(stats = {}) {
  elements.liveFrames.textContent = stats.total_frames ? `${(stats.frames_processed || 0).toLocaleString()} / ${stats.total_frames.toLocaleString()}` : '—';
  elements.liveTracks.textContent = stats.unique_tracks?.toLocaleString?.() ?? '—';
  elements.liveVehicles.textContent = stats.vehicle_tracks?.toLocaleString?.() ?? '—';
  elements.liveFps.textContent = stats.processing_fps ? `${stats.processing_fps} fps` : '—';
}

function startUpload() {
  if (!state.file) return;
  let options;
  try { options = collectOptions(); } catch (error) { toast('Calibration is invalid', error.message); return; }
  elements.runButton.disabled = true;
  elements.progressPanel.classList.remove('hidden');
  elements.resultPanel.classList.add('hidden');
  setProgress(0, 'Uploading video', 'Sending the file to your local processing engine');
  elements.progressPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });

  const form = new FormData();
  form.append('video', state.file);
  form.append('options_json', JSON.stringify(options));
  const request = new XMLHttpRequest();
  request.open('POST', '/api/jobs');
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) setProgress((event.loaded / event.total) * 8, 'Uploading video', `${formatBytes(event.loaded)} of ${formatBytes(event.total)}`);
  };
  request.onerror = () => failJob('Upload failed', 'The local server could not be reached.');
  request.onload = () => {
    if (request.status < 200 || request.status >= 300) {
      let detail = 'The server rejected the video.';
      try { detail = JSON.parse(request.responseText).detail || detail; } catch {}
      failJob('Upload failed', detail);
      return;
    }
    state.job = JSON.parse(request.responseText);
    pollJob();
  };
  request.send(form);
}

elements.runButton.addEventListener('click', startUpload);

async function pollJob() {
  clearTimeout(state.pollTimer);
  if (!state.job?.id) return;
  try {
    const response = await fetch(`/api/jobs/${state.job.id}`, { cache: 'no-store' });
    if (!response.ok) throw new Error('Job status is unavailable');
    const job = await response.json();
    state.job = job;
    setProgress(job.progress, job.phase, job.message);
    updateLiveStats(job.stats);
    if (job.status === 'completed') {
      await renderResult(job);
      loadRecent();
      return;
    }
    if (job.status === 'failed') { failJob('Analysis failed', job.error || job.message); return; }
    if (job.status === 'cancelled') { failJob('Analysis cancelled', job.message); return; }
    state.pollTimer = setTimeout(pollJob, 1000);
  } catch (error) {
    setProgress(state.job.progress || 0, 'Reconnecting', 'Waiting for the local processing engine');
    state.pollTimer = setTimeout(pollJob, 2500);
  }
}

function failJob(title, message) {
  clearTimeout(state.pollTimer);
  elements.runButton.disabled = !state.file;
  elements.progressPanel.classList.add('hidden');
  toast(title, message);
}

elements.cancelButton.addEventListener('click', async () => {
  if (!state.job?.id) return;
  elements.cancelButton.disabled = true;
  try { await fetch(`/api/jobs/${state.job.id}/cancel`, { method: 'POST' }); } catch {}
});

async function renderResult(job) {
  clearTimeout(state.pollTimer);
  const artifacts = job.artifacts || {};
  elements.resultVideo.src = `${artifacts.video}?v=${Date.now()}`;
  elements.resultVideo.load();
  elements.resultVideo.play().catch(() => {});
  const safeStem = (job.filename || 'video').replace(/\.[^/.]+$/, '');
  elements.downloadVideo.href = `${artifacts.video}?download=true`;
  elements.downloadVideo.setAttribute('download', `trafficvision_${safeStem}_analyzed.mp4`);
  elements.downloadTracks.href = `${artifacts.tracks_csv}?download=true`;
  elements.downloadSummary.href = `${artifacts.summary_json}?download=true`;
  if (artifacts.detections_csv) {
    elements.downloadDetections.href = `${artifacts.detections_csv}?download=true`;
    elements.downloadDetections.classList.remove('hidden');
  } else {
    elements.downloadDetections.classList.add('hidden');
  }

  let summary = null;
  try {
    const response = await fetch(artifacts.summary_json, { cache: 'no-store' });
    if (response.ok) summary = await response.json();
  } catch {}
  const results = summary?.results || job.stats || {};
  const input = summary?.input || job.stats?.input || {};
  const measurement = summary?.measurement || job.stats?.measurement || {};
  const statDefinitions = [
    ['Unique tracks', results.unique_tracks ?? 0, 'Stable ByteTrack IDs'],
    ['Vehicles', results.vehicle_tracks ?? 0, 'Car, bus, truck, motorcycle'],
    ['Median speed', results.average_track_speed_kmh ? `~${results.average_track_speed_kmh} km/h` : '—', 'Camera-relative estimate'],
    ['Processing', results.processing_fps ? `${results.processing_fps} fps` : '—', input.frames ? `${input.frames.toLocaleString()} frames` : 'Completed']
  ];
  elements.resultStats.innerHTML = statDefinitions.map(([label, value, detail]) => `<div class="stat-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></div>`).join('');

  const classEntries = Object.entries(results.unique_by_class || {}).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...classEntries.map(([, count]) => count), 1);
  elements.classBars.innerHTML = classEntries.length ? classEntries.slice(0, 8).map(([name, count]) => `<div class="class-row"><span>${escapeHtml(name)}</span><i style="--bar-width:${Math.max(4, count / maxCount * 100)}%"></i><strong>${count}</strong></div>`).join('') : '<span class="empty-copy">No tracked objects were reported.</span>';

  const tracks = [...(summary?.tracks || [])].sort((a, b) => b.visible_duration_s - a.visible_duration_s).slice(0, 20);
  elements.tableCount.textContent = `Showing ${tracks.length} of ${results.unique_tracks || 0} tracks`;
  elements.trackTable.innerHTML = tracks.map((track) => `<tr><td>#${escapeHtml(track.track_id)}</td><td>${escapeHtml(track.class)}</td><td>${escapeHtml(track.vehicle_model || '—')}</td><td>${Number(track.visible_duration_s).toFixed(1)} s</td><td>${track.nearest_distance_m !== '' ? `${Number(track.nearest_distance_m).toFixed(1)} m` : '—'}</td><td>${track.average_speed_kmh !== '' ? `${Number(track.average_speed_kmh).toFixed(1)} km/h` : '—'}</td></tr>`).join('');

  if (measurement.mode === 'road-plane') {
    elements.measurementMode.textContent = 'Road-plane calibrated';
    elements.measurementCopy.textContent = 'Four measured image-to-ground points transform track positions into metres. Speed remains relative to the camera.';
  } else {
    elements.measurementMode.textContent = 'Monocular FOV estimate';
    elements.measurementCopy.textContent = `Distance uses a ${measurement.horizontal_fov_degrees || 70}° field of view and typical object dimensions. Speed is relative to the fixed camera.`;
  }
  elements.progressPanel.classList.add('hidden');
  elements.resultPanel.classList.remove('hidden');
  elements.runButton.disabled = false;
  elements.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  if (job.warnings?.length) toast('Analysis completed with a note', job.warnings[0]);
}

function resetWorkspace() {
  clearTimeout(state.pollTimer);
  state.job = null;
  elements.resultVideo.pause();
  elements.resultVideo.removeAttribute('src');
  elements.resultVideo.load();
  elements.resultPanel.classList.add('hidden');
  elements.progressPanel.classList.add('hidden');
  elements.uploadPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
elements.newAnalysis.addEventListener('click', resetWorkspace);

elements.resetSettings.addEventListener('click', () => {
  elements.confidence.value = '0.05'; elements.iou.value = '0.45'; elements.imageSize.value = '960';
  elements.vehicleClassifier.checked = true; elements.vehicleConfidence.value = '0.45';
  elements.drawTrails.checked = true; elements.saveDetections.checked = true; elements.fov.value = '70';
  elements.distanceScale.value = '1.00'; elements.speedScale.value = '1.00'; elements.useHomography.checked = false;
  ['input'].forEach((eventName) => [elements.confidence, elements.iou, elements.vehicleConfidence, elements.fov].forEach((element) => element.dispatchEvent(new Event(eventName))));
});

async function checkHealth() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    if (!response.ok) throw new Error();
    elements.systemPill.classList.add('online');
    elements.systemText.textContent = 'Engine ready';
  } catch {
    elements.systemPill.classList.add('offline');
    elements.systemText.textContent = 'Engine offline';
  }
}

async function loadRecent() {
  try {
    const response = await fetch('/api/jobs', { cache: 'no-store' });
    if (!response.ok) return;
    const jobs = (await response.json()).filter((job) => job.status === 'completed').slice(0, 6);
    if (!jobs.length) return;
    elements.recentGrid.innerHTML = jobs.map((job) => `<article class="recent-card"><span>${new Date(job.created_at).toLocaleString()}</span><strong>${escapeHtml(job.filename)}</strong><footer><span>${job.stats?.unique_tracks || 0} tracks</span><span>${job.stats?.vehicle_tracks || 0} vehicles</span></footer></article>`).join('');
    elements.recentSection.classList.remove('hidden');
  } catch {}
}

checkHealth();
loadRecent();
