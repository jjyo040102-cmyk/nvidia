const state = { mode: 'demo', status: null, samples: [] };
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function text(value, fallback = '—') {
  return value === null || value === undefined || value === '' ? fallback : String(value);
}

function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove('show'), 5000);
}

function selectedSample() {
  return state.samples.find((s) => s.video_id === $('sampleSelect').value);
}

function updateMode() {
  const live = state.mode === 'live';
  $('modeBadge').textContent = live ? 'LIVE' : 'DEMO';
  $('modeBadge').className = `badge ${live ? 'live' : 'demo'}`;
  $('modeToggle').textContent = live ? 'Switch to DEMO' : 'Switch to LIVE';
  $('modeCopy').textContent = live
    ? 'Nebius Token Factory runs both perception and reasoning. No scripted fallback is allowed.'
    : 'Scripted backends prove the complete agent workflow. They are not a model benchmark.';
}

function updateSample() {
  const sample = selectedSample();
  if (!sample) return;
  $('video').src = sample.media_url;
  $('cameraLabel').textContent = sample.camera_label;
  $('sampleMeta').innerHTML = [
    `${sample.duration_s.toFixed(1)}s`, sample.area_type, sample.lighting, sample.surface,
    sample.is_control ? 'CONTROL CLIP' : 'HAZARD CLIP'
  ].filter(Boolean).map((v) => `<span>${escapeHtml(v)}</span>`).join('');
}

function riskClass(score) {
  if (score >= 16) return 'critical';
  if (score >= 9) return 'high';
  return 'neutral';
}

function render(payload) {
  const report = payload.report;
  const worst = [...(report.assessments || [])].sort((a, b) => b.risk_score - a.risk_score)[0];
  $('headline').textContent = report.headline;
  $('summary').textContent = payload.summary;
  $('findings').textContent = report.assessments.length;
  $('leadTime').textContent = report.assessments.length && report.assessments.some((a) => a.lead_time_s !== null)
    ? `${Math.max(...report.assessments.filter((a) => a.lead_time_s !== null).map((a) => a.lead_time_s)).toFixed(1)} s`
    : '—';
  $('steps').textContent = report.trace?.steps?.length ?? 0;
  $('latency').textContent = report.provenance.wall_clock_ms ? `${(report.provenance.wall_clock_ms / 1000).toFixed(2)} s` : '—';
  $('riskPill').textContent = worst ? `${worst.risk_score} / 20` : 'CLEAR';
  $('riskPill').className = `risk-pill ${riskClass(worst?.risk_score ?? 0)}`;
  const immediate = (report.actions || []).find((a) => a.horizon === 'immediate') || report.actions?.[0];
  $('actionText').textContent = immediate?.action || 'No intervention crossed the reporting bar.';

  const trace = report.trace?.steps || [];
  $('traceList').innerHTML = trace.length ? trace.map((step) => `
    <li data-index="${escapeHtml(step.index)}"><strong>${escapeHtml(step.title)}</strong>${escapeHtml(text(step.detail || step.result_summary, ''))}</li>
  `).join('') : '<li class="placeholder">No trace steps were returned.</li>';

  const citations = report.citations || [];
  $('rules').innerHTML = citations.length ? citations.map((rule) => `
    <div class="rule"><strong>${escapeHtml(rule.rule_id)} · ${escapeHtml(rule.source)}</strong><p>${escapeHtml(rule.text)}</p></div>
  `).join('') : '<p class="placeholder">No rule citation was needed for this clip.</p>';

  const p = report.provenance;
  const entries = [
    ['Mode', payload.mode.toUpperCase()], ['Provider', p.inference_provider],
    ['Vision backend', p.perception_backend], ['Vision model', p.vision_model],
    ['Reasoning model', p.reasoning_model], ['Prompt tokens', p.prompt_tokens],
    ['Completion tokens', p.completion_tokens], ['Run complete', payload.incomplete ? 'NO' : 'YES']
  ];
  $('provenance').innerHTML = entries.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(text(v))}</dd>`).join('');
}

async function run() {
  const sample = selectedSample();
  if (!sample) return;
  const button = $('runButton');
  button.disabled = true;
  button.textContent = state.mode === 'live' ? 'Running Nebius investigation…' : 'Running evidence loop…';
  try {
    const response = await fetch(`/api/investigate/${encodeURIComponent(sample.video_id)}?mode=${state.mode}`, { method: 'POST' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    render(payload);
  } catch (error) {
    toast(error.message || String(error));
  } finally {
    button.disabled = false;
    button.textContent = 'Investigate the next few seconds';
  }
}

async function boot() {
  try {
    const [statusResponse, samplesResponse] = await Promise.all([fetch('/api/status'), fetch('/api/samples')]);
    state.status = await statusResponse.json();
    const samplePayload = await samplesResponse.json();
    state.samples = samplePayload.samples || [];
    state.mode = state.status.default_mode || 'demo';
    $('providerLine').textContent = state.status.has_nebius_key
      ? 'Nebius credential detected · LIVE mode is available.'
      : 'No Nebius credential detected · DEMO works without one.';
    $('sampleCount').textContent = `${state.samples.length} CASES`;
    $('sampleSelect').innerHTML = state.samples.map((sample) =>
      `<option value="${escapeHtml(sample.video_id)}">${escapeHtml(sample.video_id.replaceAll('_', ' '))}</option>`
    ).join('');
    updateMode();
    updateSample();
  } catch (error) {
    toast(`Could not initialise Vigil: ${error.message || error}`);
  }
}

$('sampleSelect').addEventListener('change', updateSample);
$('runButton').addEventListener('click', run);
$('modeToggle').addEventListener('click', () => { state.mode = state.mode === 'demo' ? 'live' : 'demo'; updateMode(); });
boot();
