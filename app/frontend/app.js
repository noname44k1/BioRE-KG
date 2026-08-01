/**
 * app.js — BioHybridKG Demo (redesigned)
 */

const API = window.location.origin;

// ─── State ───────────────────────────────────────────────
const S = {
  ragChunks:    [],
  lastResult:   null,
  examples:     [],
  chartInst:    null,
  d3Sim:        null,
};

// ─── DOM helpers ──────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initSliders();
  initCollapsibles();
  initRag();
  initTabs();

  $('analyze-btn').addEventListener('click', runPipeline);
  $('extract-btn').addEventListener('click', runExtract);

  await Promise.all([loadKgStats(), loadExamples()]);
});

// ─── Sliders ──────────────────────────────────────────────
function initSliders() {
  [
    ['source-weight', 'val-source', v => v.toFixed(2)],
    ['sim-weight',    'val-sim',    v => v.toFixed(2)],
    ['top-k',         'val-topk',   v => v],
  ].forEach(([id, valId, fmt]) => {
    const el = $(id);
    $(valId).textContent = fmt(parseFloat(el.value));
    el.addEventListener('input', () => {
      $(valId).textContent = fmt(parseFloat(el.value));
    });
  });
}

// ─── Collapsibles ─────────────────────────────────────────
function initCollapsibles() {
  // LLM config
  $('llm-toggle').addEventListener('click', () => {
    const body = $('llm-body');
    const icon = $('llm-icon');
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'flex';
    icon.classList.toggle('open', !open);
  });

  // Fused context
  $('context-toggle').addEventListener('click', () => {
    const body = $('context-body');
    const icon = $('ctx-icon');
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    icon.classList.toggle('open', !open);
  });
}

// ─── RAG ──────────────────────────────────────────────────
function initRag() {
  $('add-rag-btn').addEventListener('click', addRag);
  $('rag-text-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') addRag();
  });
}

function addRag(text, score, relation) {
  const t = text  !== undefined ? text  : $('rag-text-input').value.trim();
  const s = score !== undefined ? score : parseFloat($('rag-score-input').value) || 0.85;
  if (!t) return;
  S.ragChunks.push({ chunk: t, score: s, relation: relation || null });
  renderRagList();
  $('rag-text-input').value = '';
  $('rag-score-input').value = '0.85';
}

function removeRag(i) {
  S.ragChunks.splice(i, 1);
  renderRagList();
}

function renderRagList() {
  const list = $('rag-chunk-list');
  $('rag-count').textContent = S.ragChunks.length;

  if (S.ragChunks.length === 0) {
    list.innerHTML = '<div class="rag-empty">No RAG chunks added yet</div>';
    return;
  }

  list.innerHTML = S.ragChunks.map((c, i) => `
    <div class="rag-item">
      <span class="rag-item-badge">RAG</span>
      <span class="rag-item-text" title="${esc(c.chunk)}">${esc(c.chunk)}</span>
      <span class="rag-item-score">${c.score.toFixed(2)}</span>
      <button class="rag-item-del" onclick="removeRag(${i})">×</button>
    </div>
  `).join('');
}

// ─── Tabs (Evidence Cards / Chart) ────────────────────────
function initTabs() {
  document.querySelectorAll('.vt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.vt-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;
      $('evidence-cards').style.display = view === 'cards' ? 'flex' : 'none';
      $('evidence-chart').style.display = view === 'chart' ? 'block' : 'none';
    });
  });
}

// ─── KG Stats ─────────────────────────────────────────────
async function loadKgStats() {
  try {
    const r = await fetch(`${API}/api/kg-stats`);
    if (!r.ok) return;
    const data = await r.json();
    const m = data.datasets.find(d => d.dataset === 'MERGED');
    if (m && m.loaded) {
      $('stat-nodes').textContent = fmtNum(m.num_nodes);
      $('stat-edges').textContent = fmtNum(m.num_edges);
    }
  } catch { /* server may not be up yet */ }
}

// ─── Examples ─────────────────────────────────────────────
async function loadExamples() {
  const fallback = [
    { sentence: 'aspirin inhibits cox-2 enzyme activity',
      ground_truth: 'aspirin|INHIBITS|cox-2',
      rag_chunks: [
        { chunk: 'aspirin reduces platelet aggregation via COX inhibition', score: 0.91, relation: 'INHIBITS' },
        { chunk: 'cox-2 causes inflammation in joints and tissue', score: 0.85, relation: 'CAUSES' },
      ]},
    { sentence: 'metformin activates AMPK signaling in liver cells',
      ground_truth: 'metformin|ACTIVATES|AMPK',
      rag_chunks: [
        { chunk: 'metformin reduces glucose production through AMPK pathway', score: 0.88, relation: 'ACTIVATES' },
      ]},
    { sentence: 'warfarin inhibits vitamin K epoxide reductase',
      ground_truth: 'warfarin|INHIBITS|vitamin k epoxide reductase',
      rag_chunks: [
        { chunk: 'warfarin prevents clotting by blocking vitamin K recycling', score: 0.93, relation: 'INHIBITS' },
      ]},
    { sentence: 'fluoxetine inhibits serotonin reuptake transporter',
      ground_truth: 'fluoxetine|INHIBITS|serotonin reuptake transporter',
      rag_chunks: [
        { chunk: 'SSRIs block the serotonin transporter at synaptic cleft', score: 0.90, relation: 'INHIBITS' },
      ]},
  ];

  try {
    const r = await fetch(`${API}/api/examples`);
    S.examples = r.ok ? (await r.json()).examples : fallback;
  } catch {
    S.examples = fallback;
  }
  renderExamples();
}

function renderExamples() {
  $('example-list').innerHTML = S.examples.map((ex, i) => `
    <div class="example-item" title="${esc(ex.sentence)}" onclick="loadExample(${i})">
      ${esc(ex.sentence)}
    </div>
  `).join('');
}

function loadExample(i) {
  const ex = S.examples[i];
  $('sentence-input').value = ex.sentence;
  if (ex.ground_truth) $('ground-truth-input').value = ex.ground_truth;
  S.ragChunks = (ex.rag_chunks || []).map(c => ({ ...c }));
  renderRagList();
  toast('Example loaded', 'info');
}

// ─── Main Pipeline ────────────────────────────────────────
async function runPipeline() {
  const sentence = $('sentence-input').value.trim();
  if (!sentence) { toast('Please enter a biomedical sentence', 'error'); return; }

  setRunning(true);
  showLoading(true);
  setStatus('running', 'Running…');

  const steps = ['ls-1', 'ls-2', 'ls-3'];
  let si = 0;
  const stepTimer = setInterval(() => {
    if (si > 0) $(steps[si - 1]).className = 'ls done';
    if (si < steps.length) $(steps[si]).className = 'ls active';
    si++;
    if (si >= steps.length) clearInterval(stepTimer);
  }, 600);

  const payload = {
    sentence,
    kg_dataset:    $('kg-dataset').value,
    rag_chunks:    S.ragChunks,
    source_weight: parseFloat($('source-weight').value),
    sim_weight:    parseFloat($('sim-weight').value),
    top_k:         parseInt($('top-k').value),
    kg_src_score:  1.0,
    rag_src_score: 0.75,
  };

  try {
    const r = await fetch(`${API}/api/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    clearInterval(stepTimer);
    steps.forEach(id => $(id).className = 'ls done');

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || 'API error');
    }

    S.lastResult = await r.json();
    renderResults(S.lastResult);
    setStatus('done', `Done — ${S.lastResult.top_k_evidence.length} evidence items`);
    $('extract-btn').disabled = false;
    toast(`Pipeline complete — ${S.lastResult.top_k_evidence.length} items`, 'success');

  } catch (e) {
    clearInterval(stepTimer);
    setStatus('error', 'Error');
    toast(`Error: ${e.message}`, 'error');
  } finally {
    showLoading(false);
    setRunning(false);
  }
}

// ─── Extract Triple ───────────────────────────────────────
async function runExtract() {
  if (!S.lastResult) return;
  $('extract-btn').disabled = true;
  $('extract-btn').innerHTML = '⏳ Calling LLM…';

  $('triple-content').innerHTML = `
    <div class="triple-placeholder">
      <div style="display:flex;align-items:center;justify-content:center;gap:10px;color:var(--text-3)">
        <div style="width:18px;height:18px;border:2px solid var(--purple);border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite"></div>
        Extracting triple…
      </div>
    </div>`;

  const payload = {
    sentence:      S.lastResult.sentence,
    fused_context: S.lastResult.fused_context,
    ground_truth:  $('ground-truth-input').value.trim() || null,
    model:         $('model-input').value.trim(),
  };

  try {
    const r = await fetch(`${API}/api/extract-triple`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    renderTriple(await r.json());
    toast('Triple extracted!', 'success');
  } catch (e) {
    $('triple-content').innerHTML = `<div class="triple-placeholder" style="color:var(--red)">${esc(e.message)}</div>`;
    toast(`LLM error: ${e.message}`, 'error');
  } finally {
    $('extract-btn').disabled = false;
    $('extract-btn').innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polygon points="5,3 19,12 5,21"/>
      </svg>
      Extract via LLM`;
  }
}

// ─── Render Results ───────────────────────────────────────
function renderResults(data) {
  $('welcome-screen').style.display = 'none';
  $('results-area').style.display = 'flex';

  // Stats strip
  $('r-entities').textContent = data.entities_matched.length;
  $('r-kg').textContent       = data.kg_evidence_count;
  $('r-rag').textContent      = data.rag_evidence_count;
  $('r-topk').textContent     = data.top_k_evidence.length;
  $('r-elapsed').textContent  = data.elapsed_sec + 's';

  const chips = $('r-entity-chips');
  chips.innerHTML = '';
  data.entities_matched.forEach(e => {
    const s = document.createElement('span');
    s.className = 'e-chip';
    s.textContent = e;
    chips.appendChild(s);
  });

  renderCards(data.top_k_evidence);
  renderChart(data.top_k_evidence);
  renderGraph(data.top_k_evidence);
  $('fused-context-text').textContent = data.fused_context || '—';
}

// ─── Evidence Cards ───────────────────────────────────────
function renderCards(ev) {
  const wrap = $('evidence-cards');
  if (!ev || ev.length === 0) {
    wrap.innerHTML = `<div class="triple-placeholder">No evidence found. Try another sentence or dataset.</div>`;
    return;
  }

  wrap.innerHTML = ev.map((e, i) => {
    const isKg   = e.type === 'kg_direct';
    const cls    = isKg ? 'kg' : 'rag';
    const icon   = isKg ? '🔷' : '🟠';
    const score  = e.score;
    const sc     = score > 0.7 ? 'sc-high' : score > 0.45 ? 'sc-mid' : 'sc-low';
    const pct    = Math.min(100, Math.round(score * 100));
    const delay  = i * 0.04;

    // Triple or text
    let middle = '';
    if (isKg && e.triple && e.triple.length === 3) {
      middle = `
        <div class="ev-triple">
          <span class="ev-subj">${esc(e.triple[0])}</span>
          <span class="ev-arrow">→</span>
          <span class="ev-pred">${esc(e.triple[1])}</span>
          <span class="ev-arrow">→</span>
          <span class="ev-obj">${esc(e.triple[2])}</span>
        </div>`;
    } else {
      const txt = (e.chunk || e.text || '').replace(/^\[.*?\]\s*/, '');
      middle = `<div class="ev-text">${esc(txt)}</div>`;
    }

    // Meta
    const tags = [
      e.source && `src=${e.source}`,
      e.raw?.direction,
      e.hop && `${e.hop}-hop`,
      e.relation && `rel=${e.relation}`,
    ].filter(Boolean).map(t => `<span class="ev-tag">${esc(t)}</span>`).join('');

    // Context
    const ctx = e.raw?.context
      ? `<div class="ev-ctx">📄 ${esc(e.raw.context.slice(0, 160))}${e.raw.context.length > 160 ? '…' : ''}</div>`
      : '';

    return `
      <div class="ev-card ${cls}" style="animation-delay:${delay}s">
        <div class="ev-left">
          <div class="ev-icon ${cls}">${icon}</div>
          <span class="ev-id">${e.id || `E${i+1}`}</span>
        </div>
        <div class="ev-body">
          ${middle}
          <div class="ev-meta">${tags}</div>
          ${ctx}
        </div>
        <div class="ev-right">
          <span class="ev-score-num ${sc}">${score.toFixed(3)}</span>
          <div class="ev-bar"><div class="ev-bar-fill ${cls}" style="width:${pct}%"></div></div>
        </div>
      </div>`;
  }).join('');
}

// ─── Score Chart ──────────────────────────────────────────
function renderChart(ev) {
  if (!ev || ev.length === 0) return;
  const canvas = $('score-chart');
  if (S.chartInst) { S.chartInst.destroy(); S.chartInst = null; }

  const labels = ev.map((e, i) => e.id || `E${i+1}`);
  const scores = ev.map(e => +e.score.toFixed(3));
  const bgColors = ev.map(e =>
    e.type === 'kg_direct' ? 'rgba(37,99,235,0.55)' : 'rgba(234,88,12,0.55)');
  const borders = ev.map(e =>
    e.type === 'kg_direct' ? '#2563eb' : '#ea580c');

  S.chartInst = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets: [{ data: scores, backgroundColor: bgColors, borderColor: borders, borderWidth: 1, borderRadius: 5 }] },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255,255,255,0.97)',
          borderColor: 'rgba(0,0,0,0.1)',
          borderWidth: 1,
          titleColor: '#111827',
          bodyColor: '#4b5563',
          padding: 10,
          callbacks: {
            label: ctx => {
              const e = ev[ctx.dataIndex];
              return [`Score: ${e.score.toFixed(4)}`, e.type === 'kg_direct' ? '🔷 KG Direct' : '🟠 RAG'];
            },
            afterLabel: ctx => {
              const e = ev[ctx.dataIndex];
              return e.triple ? `${e.triple[0]} → ${e.triple[2]}` : '';
            },
          },
        },
      },
      scales: {
        x: { min: 0, max: 1.05, grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono', size: 10 } } },
        y: { grid: { display: false }, ticks: { color: '#4b5563', font: { family: 'JetBrains Mono', size: 11 } } },
      },
    },
  });
}

// ─── KG Graph ─────────────────────────────────────────────
function renderGraph(ev) {
  const svgEl = $('kg-graph-svg');
  svgEl.innerHTML = '';

  const kgEv = ev.filter(e => e.type === 'kg_direct' && e.triple);
  if (kgEv.length === 0) {
    d3.select(svgEl).append('text')
      .attr('x', '50%').attr('y', '50%')
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('fill', '#9ca3af').attr('font-size', 11).attr('font-family', 'Inter')
      .text('No KG triples to visualize');
    return;
  }

  const W = svgEl.clientWidth || 340;
  const H = 220;

  const svg = d3.select(svgEl).attr('viewBox', `0 0 ${W} ${H}`);

  // Nodes + Links
  const nodesMap = {};
  const links = [];

  kgEv.forEach(e => {
    const [s, p, o] = e.triple;
    if (!nodesMap[s]) nodesMap[s] = { id: s, src: e.source };
    if (!nodesMap[o]) nodesMap[o] = { id: o, src: e.source };
    links.push({ source: s, target: o, label: p, score: e.score });
  });

  const nodes = Object.values(nodesMap);

  // Arrow
  svg.append('defs').append('marker')
    .attr('id', 'arr').attr('viewBox', '-0 -4 8 8').attr('refX', 20).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
    .append('path').attr('d', 'M0,-3L8,0L0,3').attr('fill', '#2563eb').attr('opacity', 0.7);

  const linkG  = svg.append('g');
  const lblG   = svg.append('g');
  const nodeG  = svg.append('g');

  const linkEl = linkG.selectAll('line').data(links).join('line')
    .attr('class', 'glink')
    .attr('stroke', '#2563eb')
    .attr('stroke-width', d => Math.max(1, d.score * 2.5))
    .attr('marker-end', 'url(#arr)');

  const lblEl = lblG.selectAll('text').data(links).join('text')
    .attr('class', 'glabel').attr('fill', '#7c3aed').attr('font-size', 8)
    .attr('text-anchor', 'middle').text(d => d.label);

  const nodeEl = nodeG.selectAll('g').data(nodes).join('g').attr('class', 'gnode')
    .call(d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end',   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = d.fy = null; })
    );

  nodeEl.append('circle').attr('r', 16)
    .attr('fill', 'rgba(37,99,235,0.08)')
    .attr('stroke', '#2563eb').attr('stroke-width', 1.5);

  nodeEl.append('text')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
    .attr('fill', '#111827').attr('font-size', 8)
    .text(d => trunc(d.id, 10));

  // D3 tooltip
  const tip = d3.select('body').selectAll('#gtip').data([0]).join('div')
    .attr('id', 'gtip')
    .style('position','fixed').style('background','rgba(255,255,255,.97)')
    .style('border','1px solid rgba(37,99,235,.2)').style('border-radius','8px')
    .style('padding','8px 12px').style('font-size','11px').style('color','#111827')
    .style('pointer-events','none').style('z-index','9999').style('opacity',0).style('box-shadow','0 4px 12px rgba(0,0,0,0.1)');

  nodeEl.on('mouseover',(event, d) => {
    tip.style('opacity',1).html(`<strong>${d.id}</strong>${d.src ? `<br><small style="color:#9ca3af">src: ${d.src}</small>` : ''}`);
  }).on('mousemove', event => {
    tip.style('left',(event.clientX+12)+'px').style('top',(event.clientY-24)+'px');
  }).on('mouseout', () => tip.style('opacity',0));

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(110))
    .force('charge', d3.forceManyBody().strength(-250))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide().radius(28));

  S.d3Sim = sim;

  sim.on('tick', () => {
    linkEl
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    lblEl
      .attr('x', d => (d.source.x + d.target.x)/2)
      .attr('y', d => (d.source.y + d.target.y)/2);
    nodeEl.attr('transform', d =>
      `translate(${clamp(d.x,18,W-18)},${clamp(d.y,18,H-18)})`);
  });
}

// ─── Triple Result ────────────────────────────────────────
function renderTriple(data) {
  const h = data.head || '—', rel = data.relation || '—', t = data.tail || '—';

  let gtHtml = '';
  if (data.ground_truth) {
    gtHtml = `
      <div class="td-gt">
        <span>Ground truth:</span>
        <code style="font-family:JetBrains Mono;font-size:11px;color:var(--text-code)">${esc(data.ground_truth)}</code>
        <span class="match-pill ${data.match ? 'yes' : 'no'}">${data.match ? '✓ Match' : '✗ No match'}</span>
      </div>`;
  }

  const mockBadge = data.mock
    ? `<div class="td-gt"><span class="match-pill mock">🔵 Mock — no API key</span></div>` : '';

  $('triple-content').innerHTML = `
    <div class="triple-display">
      <div class="td-row">
        <div class="td-entity head">${esc(h)}</div>
        <div class="td-arrow">⟶</div>
        <div class="td-rel">${esc(rel)}</div>
        <div class="td-arrow">⟶</div>
        <div class="td-entity tail">${esc(t)}</div>
      </div>
      ${gtHtml}${mockBadge}
    </div>
    <div class="td-raw">${esc(data.raw_output || '')}</div>
  `;
}

// ─── UI helpers ───────────────────────────────────────────
function setRunning(on) {
  const btn = $('analyze-btn');
  btn.disabled = on;
  btn.innerHTML = on
    ? `<div style="width:16px;height:16px;border:2.5px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite"></div> Running…`
    : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5,3 19,12 5,21"/></svg> Run Pipeline`;
}

function showLoading(on) {
  $('loading-screen').style.display = on ? 'flex' : 'none';
  if (on) {
    ['ls-1','ls-2','ls-3'].forEach(id => $(id).className = 'ls');
  }
}

function setStatus(type, text) {
  $('run-status').innerHTML = `
    <span class="status-dot ${type}"></span>
    <span class="status-text">${text}</span>`;
}

function toast(msg, type = 'info') {
  const t = $('toast');
  const icons = { success: '✓', error: '⚠', info: 'ℹ' };
  t.textContent = (icons[type] || '') + ' ' + msg;
  t.className = `show ${type}`;
  clearTimeout(t._t);
  t._t = setTimeout(() => t.classList.remove('show'), 3500);
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtNum(n) { return n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n); }
function trunc(s, n) { return s && s.length > n ? s.slice(0, n)+'…' : s; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

// expose inline handlers
window.removeRag     = removeRag;
window.loadExample   = loadExample;
