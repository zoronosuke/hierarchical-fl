// 監視画面 (index.html) と再生ページ (replay.html) で共用する描画部品。外部ライブラリなし。
(function () {
  const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const seriesColor = (i) => css(SERIES[i % SERIES.length]);

  // ---- 書式 ----
  const fmt = (v, d = 3) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d));
  const fmtBytes = (v) => {
    if (v == null) return '–';
    const units = [['GB', 1e9], ['MB', 1e6], ['KB', 1e3]];
    for (const [u, s] of units) if (v >= s) return `${(v / s).toFixed(v >= 100 * s ? 0 : 1)} ${u}`;
    return `${Math.round(v)} B`;
  };
  const fmtRate = (bps) => (bps == null ? '–' : `${(bps * 8 / 1e6).toFixed(2)} Mbps`);
  const fmtDur = (s) => {
    if (s == null) return '–';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m${Math.floor(s % 60)}s`;
  };
  const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' });
  const fmtClock = (t) => new Date(t * 1000).toLocaleTimeString('ja-JP');
  const fmtDateTime = (t) => new Date(t * 1000).toLocaleString('ja-JP', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });

  // ---- 折れ線 (1軸・2px・端点マーカー・gap 秒以上の空白で線を切る) ----
  function drawLines(canvas, series, opts = {}) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    const pad = { l: 52, r: 16, t: 12, b: 28 };
    let x0 = Infinity, x1 = -Infinity, ymin = Infinity, ymax = -Infinity, count = 0;
    series.forEach(s => s.points.forEach(p => {
      if (p[1] == null || !isFinite(p[1])) return;
      count++; x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
    }));
    if (opts.xMin != null) x0 = opts.xMin;
    if (opts.xMax != null) x1 = opts.xMax;
    if (!count) {
      ctx.fillStyle = css('--muted'); ctx.font = '13px system-ui';
      ctx.fillText(opts.emptyText || 'データなし', pad.l, h / 2); canvas._hit = null; return;
    }
    if (x1 <= x0) x1 = x0 + 60;
    let y0 = opts.yMin != null ? opts.yMin : ymin, y1 = opts.yMax != null ? opts.yMax : ymax;
    if (y1 === y0) y1 = y0 + 1;
    if (opts.yMin == null) { const m = (y1 - y0) * 0.08; y0 -= m; y1 += m; if (opts.zero && ymin >= 0) y0 = 0; }
    const X = t => pad.l + (t - x0) / (x1 - x0) * (w - pad.l - pad.r);
    const Y = v => h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);
    ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1; ctx.fillStyle = css('--muted'); ctx.font = '11px system-ui';
    for (let i = 0; i <= 4; i++) {
      const v = y0 + (y1 - y0) * i / 4, y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(opts.yFmt ? opts.yFmt(v) : v.toFixed(2), pad.l - 6, y + 4);
    }
    const nx = Math.max(2, Math.floor((w - pad.l - pad.r) / 90));
    ctx.textAlign = 'center';
    for (let i = 0; i <= nx; i++) { const t = x0 + (x1 - x0) * i / nx; ctx.fillText((opts.xFmt || fmtTime)(t), X(t), h - pad.b + 16); }
    ctx.strokeStyle = css('--axis'); ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();
    (opts.spans || []).forEach(sp => {  // 停止区間の網掛け
      const a = Math.max(X(sp[0]), pad.l), b = Math.min(X(sp[1]), w - pad.r);
      if (b > a) { ctx.fillStyle = css('--grid'); ctx.globalAlpha = 0.6; ctx.fillRect(a, pad.t, b - a, h - pad.t - pad.b); ctx.globalAlpha = 1; }
    });
    const gap = opts.gap != null ? opts.gap : 300;
    series.forEach(s => {
      if (!s.points.length) return;
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath();
      let prev = null;
      s.points.forEach(p => {
        if (p[1] == null || !isFinite(p[1])) { prev = null; return; }
        const x = X(p[0]), y = Y(p[1]);
        if (!prev || p[0] - prev[0] > gap) ctx.moveTo(x, y);
        else if (opts.step) { ctx.lineTo(x, Y(prev[1])); ctx.lineTo(x, y); }
        else ctx.lineTo(x, y);
        prev = p;
      });
      ctx.stroke();
      let last = null;
      for (let i = s.points.length - 1; i >= 0; i--) { const p = s.points[i]; if (p[1] != null && isFinite(p[1])) { last = p; break; } }
      if (!last) return;
      ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 6, 0, Math.PI * 2); ctx.fillStyle = css('--surface'); ctx.fill();
      ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 4, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
    });
    canvas._hit = { x0, x1, series, pad, w, h };
  }

  function attachHover(canvas, format, tooltip) {
    canvas.addEventListener('pointermove', (e) => {
      const hit = canvas._hit; if (!hit) return;
      const rect = canvas.getBoundingClientRect(); const px = e.clientX - rect.left;
      const t = hit.x0 + (px - hit.pad.l) / (hit.w - hit.pad.l - hit.pad.r) * (hit.x1 - hit.x0);
      const rows = [];
      hit.series.forEach(s => {
        let best = null; for (const p of s.points) { if (p[1] == null) continue; if (!best || Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p; }
        if (best) rows.push(`<span class="swatch" style="background:${s.color}"></span>${s.name}: ${format(best)}`);
      });
      tooltip.innerHTML = `<div class="muted">${fmtDateTime(t)}</div>` + rows.join('<br>');
      tooltip.style.display = 'block'; tooltip.style.left = (e.clientX + 12) + 'px'; tooltip.style.top = (e.clientY + 12) + 'px';
    });
    canvas.addEventListener('pointerleave', () => { tooltip.style.display = 'none'; });
  }

  function legendTo(el, series) {
    el.innerHTML = '';
    series.forEach(s => { const sp = document.createElement('span'); sp.innerHTML = `<span class="swatch" style="background:${s.color}"></span>${s.name}`; el.appendChild(sp); });
  }

  // ---- トポロジー (5段の木) ----
  const PHASE_COLOR = { train: '--s1', aggregate: '--s2', idle: '--muted', starting: '--s4', stopped: '--muted', error: '--critical' };
  const PHASE_NAME = { train: '学習', aggregate: '集約・返送', idle: '待機', starting: '起動中', stopped: '停止', error: '異常' };

  function layoutTree(nodes) {
    const byNum = new Map(nodes.map(n => [n.num, n]));
    const children = new Map(nodes.map(n => [n.num, []]));
    let root = null;
    nodes.forEach(n => { if (n.parent != null && children.has(n.parent)) children.get(n.parent).push(n.num); else if (root == null) root = n.num; });
    const pos = new Map(); let slot = 0;
    const visit = (num, depth) => {
      const kids = children.get(num) || [];
      if (!kids.length) { pos.set(num, { x: slot++, depth }); return; }
      kids.forEach(k => visit(k, depth + 1));
      const xs = kids.map(k => pos.get(k).x);
      pos.set(num, { x: (Math.min(...xs) + Math.max(...xs)) / 2, depth });
    };
    if (root != null) visit(root, 0);
    const maxDepth = Math.max(0, ...[...pos.values()].map(p => p.depth));
    return { pos, slots: Math.max(slot, 1), depths: maxDepth + 1, byNum };
  }

  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  // nodes: [{num, id, role, parent, state, activity:{label, phase, internal}, current_round, power_w, temp_c, links}]
  // activeLinks: Set(子ノード番号)。その子と親の線を強調する
  function renderTopology(container, nodes, activeLinks) {
    const { pos, slots, depths, byNum } = layoutTree(nodes);
    const W = Math.max(container.clientWidth, 320), boxW = Math.min(190, W / slots - 12), boxH = 74, rowH = 112;
    const H = depths * rowH;
    const cx = (p) => (p.x + 0.5) * (W / slots), cy = (p) => p.depth * rowH + 12;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="トポロジー">`;
    nodes.forEach(n => {  // 線 (親→子)
      if (n.parent == null || !pos.has(n.parent)) return;
      const a = pos.get(n.parent), b = pos.get(n.num);
      const x1 = cx(a), y1 = cy(a) + boxH, x2 = cx(b), y2 = cy(b);
      const active = activeLinks && activeLinks.has(n.num);
      const link = (n.links || {}).parent;
      svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${active ? css('--s2') : css('--axis')}" stroke-width="${active ? 4 : 2}" />`;
      if (link) {
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        svg += `<text x="${mx + 6}" y="${my}" font-size="11" fill="${css('--ink-2')}">↓${esc(fmtBytes(link.recv))} ↑${esc(fmtBytes(link.sent))}</text>`;
      }
    });
    nodes.forEach(n => {  // ノード
      const p = pos.get(n.num); if (!p) return;
      const x = cx(p) - boxW / 2, y = cy(p);
      const act = n.activity || {};
      const dead = ['failed', 'error', 'unreachable', 'stopped', 'not_started', 'finished', 'no_repo'].includes(n.state);
      const phase = dead ? (['failed', 'error', 'unreachable'].includes(n.state) ? 'error' : 'stopped') : (act.phase || 'idle');
      const color = css(PHASE_COLOR[phase] || '--muted');
      const label = dead ? ({ failed: '異常終了', error: 'エラー', unreachable: '到達不能', stopped: '停止', not_started: '未起動', finished: '正常終了', no_repo: '配備なし' }[n.state]) : (act.label || '');
      const sub = [n.current_round != null ? `R${n.current_round}` : null, n.power_w != null ? `${n.power_w.toFixed(1)}W` : null, n.temp_c != null ? `${Math.round(n.temp_c)}℃` : null].filter(Boolean).join(' · ');
      svg += `<rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="8" fill="${css('--surface')}" stroke="${color}" stroke-width="${phase === 'train' || phase === 'aggregate' ? 3 : 1.5}" />`;
      svg += `<rect x="${x}" y="${y}" width="6" height="${boxH}" rx="3" fill="${color}" />`;
      svg += `<text x="${x + 14}" y="${y + 18}" font-size="13" font-weight="600" fill="${css('--ink')}">${esc(n.num)} ${esc(n.id)}</text>`;
      svg += `<text x="${x + 14}" y="${y + 36}" font-size="11.5" fill="${css('--ink')}">${esc(label).slice(0, 26)}</text>`;
      svg += `<text x="${x + 14}" y="${y + 53}" font-size="11" fill="${css('--ink-2')}">${esc(sub)}</text>`;
      if (act.internal && !dead) {
        svg += `<text x="${x + 14}" y="${y + 68}" font-size="10.5" fill="${css('--muted')}">Internal: ${esc(act.internal.label).slice(0, 20)}</text>`;
      }
    });
    svg += '</svg>';
    container.innerHTML = svg;
  }

  // ---- 直近ラウンドのタイムライン (横棒) ----
  const PROC_COLOR = { global: '--s7', edge: '--s2', internal_client: '--s3', leaf: '--s1' };
  function drawTimeline(canvas, timeline) {
    const dpr = window.devicePixelRatio || 1;
    const bars = (timeline && timeline.bars) || [];
    const rowH = 20, pad = { l: 190, r: 16, t: 8, b: 26 };
    const h = Math.max(pad.t + pad.b + bars.length * rowH, 80);
    canvas.style.height = h + 'px';
    const w = canvas.clientWidth;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    ctx.font = '11px system-ui';
    if (!bars.length) { ctx.fillStyle = css('--muted'); ctx.fillText('データ待ち', 12, 30); return; }
    const t0 = timeline.t0, t1 = Math.max(timeline.t1, t0 + 1);
    const X = t => pad.l + (t - t0) / (t1 - t0) * (w - pad.l - pad.r);
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.textAlign = 'center';
    const span = t1 - t0, step = span > 120 ? 30 : span > 40 ? 10 : 5;
    for (let s = 0; s <= span; s += step) { const x = Math.round(X(t0 + s)) + 0.5; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke(); ctx.fillText(`${s}s`, x, h - pad.b + 16); }
    bars.forEach((b, i) => {
      const y = pad.t + i * rowH;
      ctx.fillStyle = css('--ink-2'); ctx.textAlign = 'right';
      ctx.fillText(`${b.num} ${b.id} · ${b.label}`, pad.l - 8, y + 14);
      const end = b.t1 != null ? b.t1 : t1;
      ctx.fillStyle = css(PROC_COLOR[b.proc] || '--s1');
      const bx = X(b.t0), bw = Math.max(X(end) - bx, 2);
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(bx, y + 4, bw, rowH - 8, 3) : ctx.rect(bx, y + 4, bw, rowH - 8); ctx.fill();
      if (bw > 40) { ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.fillText(`${(end - b.t0).toFixed(1)}s`, bx + 4, y + 14); }
    });
  }

  window.HFLViz = { SERIES, css, seriesColor, fmt, fmtBytes, fmtRate, fmtDur, fmtTime, fmtClock, fmtDateTime, drawLines, attachHover, legendTo, renderTopology, drawTimeline, PHASE_NAME };
})();
