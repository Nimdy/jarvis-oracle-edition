window._sparkData = {};

window.pushSpark = function(key, val) {
  if (!window._sparkData[key]) window._sparkData[key] = [];
  var arr = window._sparkData[key];
  arr.push(val);
  if (arr.length > 60) arr.shift();
};

window.drawSparkline = function(canvasId, data, color, min, max) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var dpr = window.devicePixelRatio || 1;
  var w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  if (!data || data.length < 2) return;
  var lo = min != null ? min : Math.min.apply(null, data);
  var hi = max != null ? max : Math.max.apply(null, data);
  if (hi === lo) { hi = lo + 1; }
  var stepX = w / (data.length - 1);
  function yPos(v) { return h - ((v - lo) / (hi - lo)) * h; }

  ctx.beginPath();
  ctx.moveTo(0, yPos(data[0]));
  for (var i = 1; i < data.length; i++) ctx.lineTo(i * stepX, yPos(data[i]));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  var grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color.replace(')', ',0.3)').replace('rgb', 'rgba'));
  grad.addColorStop(1, 'transparent');
  ctx.lineTo((data.length - 1) * stepX, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  var lastX = (data.length - 1) * stepX, lastY = yPos(data[data.length - 1]);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
};

window.updateSparklines = function(snap) {
  var k = snap.kernel || {}, a = snap.analytics || {}, att = snap.attention || {};
  var obs = snap.observer || {}, m = snap.mutations || {};
  window.pushSpark('tick_ms', k.avg_tick_ms || 0);
  window.pushSpark('confidence', a.confidence_level || 0);
  window.pushSpark('engagement', att.engagement_level || 0);
  window.pushSpark('awareness', obs.awareness_level || 0);
  window.pushSpark('mutations', m.count || 0);

  var sparks = [
    ['tick_ms', '#0cf'], ['confidence', '#0f9'], ['engagement', '#f90'],
    ['awareness', '#c0f'], ['mutations', '#ff0']
  ];
  for (var i = 0; i < sparks.length; i++) {
    var key = sparks[i][0], col = sparks[i][1];
    var d = window._sparkData[key];
    if (d) window.drawSparkline('spark-' + key, d, col);
  }
};

function prepCanvas(canvasId) {
  var canvas = typeof canvasId === 'string' ? document.getElementById(canvasId) : canvasId;
  if (!canvas) return null;
  var dpr = window.devicePixelRatio || 1;
  var w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx: ctx, w: w, h: h };
}

window.drawMultiLine = function(canvasId, datasets, colors) {
  var c = prepCanvas(canvasId);
  if (!c) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var pad = { top: 10, right: 10, bottom: 20, left: 35 };
  var pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;

  var allVals = [].concat.apply([], datasets);
  if (!allVals.length) return;
  var lo = Math.min.apply(null, allVals), hi = Math.max.apply(null, allVals);
  if (hi === lo) hi = lo + 1;
  var maxLen = 0;
  for (var i = 0; i < datasets.length; i++) maxLen = Math.max(maxLen, datasets[i].length);

  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 0.5;
  for (var g = 0; g <= 4; g++) {
    var gy = pad.top + ph - (g / 4) * ph;
    ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + pw, gy); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText((lo + (g / 4) * (hi - lo)).toFixed(1), pad.left - 4, gy + 3);
  }

  for (var d = 0; d < datasets.length; d++) {
    var arr = datasets[d]; if (!arr.length) continue;
    var stepX = arr.length > 1 ? pw / (arr.length - 1) : 0;
    ctx.beginPath();
    for (var j = 0; j < arr.length; j++) {
      var x = pad.left + j * stepX;
      var y = pad.top + ph - ((arr[j] - lo) / (hi - lo)) * ph;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = colors[d] || '#0cf';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
};

window.drawBarChart = function(canvasId, values, color) {
  var c = prepCanvas(canvasId);
  if (!c || !values || !values.length) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var pad = { top: 10, right: 10, bottom: 30, left: 10 };
  var pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;
  var maxVal = 0;
  for (var i = 0; i < values.length; i++) maxVal = Math.max(maxVal, values[i].value);
  if (maxVal === 0) maxVal = 1;
  var barW = pw / values.length;
  var gap = Math.max(2, barW * 0.2);

  for (var i = 0; i < values.length; i++) {
    var barH = (values[i].value / maxVal) * ph;
    var x = pad.left + i * barW + gap / 2;
    var bw = barW - gap;
    ctx.fillStyle = color || '#0cf';
    ctx.fillRect(x, pad.top + ph - barH, bw, barH);
    ctx.fillStyle = '#aaa'; ctx.font = '8px monospace'; ctx.textAlign = 'center';
    var lbl = values[i].label || '';
    if (lbl.length > 6) lbl = lbl.slice(0, 6);
    ctx.fillText(lbl, x + bw / 2, h - pad.bottom + 12);
  }
};

window.drawRadar = function(canvasId, labels, values, color) {
  var c = prepCanvas(canvasId);
  if (!c || !labels || !labels.length) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var cx = w / 2, cy = h / 2, r = Math.min(cx, cy) - 30;
  var n = labels.length, step = (Math.PI * 2) / n;

  function ptAt(idx, scale) {
    var angle = -Math.PI / 2 + idx * step;
    return { x: cx + Math.cos(angle) * r * scale, y: cy + Math.sin(angle) * r * scale };
  }

  var rings = [0.25, 0.5, 0.75, 1.0];
  for (var ri = 0; ri < rings.length; ri++) {
    ctx.beginPath();
    for (var j = 0; j <= n; j++) {
      var p = ptAt(j % n, rings[ri]);
      j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  for (var j = 0; j < n; j++) {
    var p = ptAt(j, 1);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p.x, p.y);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 0.5; ctx.stroke();
    var lp = ptAt(j, 1.18);
    ctx.fillStyle = '#aaa'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText(labels[j], lp.x, lp.y + 3);
  }

  ctx.beginPath();
  for (var j = 0; j <= n; j++) {
    var v = Math.max(0, Math.min(1, values[j % n] || 0));
    var p = ptAt(j % n, v);
    j === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
  }
  ctx.strokeStyle = color || '#0cf'; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = (color || '#0cf').replace(')', ',0.2)').replace('rgb', 'rgba');
  if (ctx.fillStyle === (color || '#0cf')) ctx.fillStyle = 'rgba(0,204,255,0.2)';
  ctx.fill();
};

window.drawHeatmap = function(canvasId, rows, cols, values) {
  var c = prepCanvas(canvasId);
  if (!c || !rows || !cols || !values) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var pad = { top: 5, right: 5, bottom: 25, left: 55 };
  var pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;
  var cellW = pw / cols.length, cellH = ph / rows.length;

  function heatColor(v) {
    v = Math.max(0, Math.min(1, v));
    if (v < 0.5) {
      var t = v * 2;
      return 'rgb(' + Math.round(0 + t * 0) + ',' + Math.round(50 + t * 205) + ',' + Math.round(255 - t * 255) + ')';
    }
    var t = (v - 0.5) * 2;
    return 'rgb(' + Math.round(t * 255) + ',' + Math.round(255 - t * 255) + ',0)';
  }

  for (var r = 0; r < rows.length; r++) {
    for (var ci = 0; ci < cols.length; ci++) {
      var v = (values[r] && values[r][ci] != null) ? values[r][ci] : 0;
      ctx.fillStyle = heatColor(v);
      ctx.fillRect(pad.left + ci * cellW, pad.top + r * cellH, cellW - 1, cellH - 1);
    }
    ctx.fillStyle = '#aaa'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
    var rl = rows[r]; if (rl.length > 8) rl = rl.slice(0, 8);
    ctx.fillText(rl, pad.left - 4, pad.top + r * cellH + cellH / 2 + 3);
  }
  ctx.font = '8px monospace'; ctx.textAlign = 'center';
  for (var ci = 0; ci < cols.length; ci++) {
    var cl = cols[ci]; if (cl.length > 5) cl = cl.slice(0, 5);
    ctx.fillStyle = '#aaa';
    ctx.fillText(cl, pad.left + ci * cellW + cellW / 2, h - pad.bottom + 12);
  }
};

window.buildHistogram = function(values, numBins) {
  if (!values || !values.length || !numBins) return [];
  var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
  if (hi === lo) hi = lo + 1;
  var binW = (hi - lo) / numBins, bins = [];
  for (var i = 0; i < numBins; i++) bins.push({ min: lo + i * binW, max: lo + (i + 1) * binW, count: 0 });
  for (var j = 0; j < values.length; j++) {
    var idx = Math.min(Math.floor((values[j] - lo) / binW), numBins - 1);
    bins[idx].count++;
  }
  return bins;
};

window.drawWeightHistogram = function(canvas, bins) {
  var c = prepCanvas(canvas);
  if (!c || !bins || !bins.length) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var pad = { top: 8, right: 8, bottom: 22, left: 8 };
  var pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;
  var maxC = 0;
  for (var i = 0; i < bins.length; i++) maxC = Math.max(maxC, bins[i].count);
  if (maxC === 0) maxC = 1;
  var barW = pw / bins.length, gap = Math.max(1, barW * 0.1);

  for (var i = 0; i < bins.length; i++) {
    var barH = (bins[i].count / maxC) * ph;
    var x = pad.left + i * barW + gap / 2;
    var bw = barW - gap;
    var grad = ctx.createLinearGradient(0, pad.top + ph - barH, 0, pad.top + ph);
    grad.addColorStop(0, '#0cf'); grad.addColorStop(1, '#06a');
    ctx.fillStyle = grad;
    ctx.fillRect(x, pad.top + ph - barH, bw, barH);
    ctx.fillStyle = '#aaa'; ctx.font = '7px monospace'; ctx.textAlign = 'center';
    ctx.fillText(bins[i].min.toFixed(2), x + bw / 2, h - pad.bottom + 10);
  }
};

window.renderGauge = function(container, label, value, opts) {
  opts = opts || {};
  var max = opts.max || 1, color = opts.color || '#0cf', suffix = opts.suffix || '';
  var pct = Math.min(100, Math.max(0, (value / max) * 100));
  var el = document.createElement('div');
  el.className = 'gauge';
  el.innerHTML =
    '<div class="gauge-label">' + label + '</div>' +
    '<div class="gauge-bar"><div class="gauge-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
    '<div class="gauge-value">' + (typeof value === 'number' ? value.toFixed(2) : value) + suffix + '</div>';
  container.appendChild(el);
  return el;
};

window.renderStabilityChart = function(canvasId, data) {
  var c = prepCanvas(canvasId);
  if (!c) return;
  var ctx = c.ctx, w = c.w, h = c.h;
  var pad = { top: 15, right: 10, bottom: 25, left: 35 };
  var pw = w - pad.left - pad.right, ph = h - pad.top - pad.bottom;

  ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 0.5;
  for (var g = 0; g <= 4; g++) {
    var gy = pad.top + ph - (g / 4) * ph;
    ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + pw, gy); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText((g / 4).toFixed(2), pad.left - 4, gy + 3);
  }

  var series = [
    { key: 'contradiction_debt', color: '#f44' },
    { key: 'soul_integrity', color: '#0f9' },
    { key: 'quarantine_pressure', color: '#f90' }
  ];

  var allTs = [];
  for (var s = 0; s < series.length; s++) {
    var arr = data[series[s].key] || [];
    for (var j = 0; j < arr.length; j++) allTs.push(arr[j].ts);
  }
  if (!allTs.length) return;
  var minTs = Math.min.apply(null, allTs), maxTs = Math.max.apply(null, allTs);
  if (maxTs === minTs) maxTs = minTs + 1;

  for (var s = 0; s < series.length; s++) {
    var arr = data[series[s].key] || [];
    if (!arr.length) continue;
    ctx.beginPath();
    for (var j = 0; j < arr.length; j++) {
      var x = pad.left + ((arr[j].ts - minTs) / (maxTs - minTs)) * pw;
      var y = pad.top + ph - Math.max(0, Math.min(1, arr[j].v)) * ph;
      j === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = series[s].color; ctx.lineWidth = 1.5; ctx.stroke();
  }

  var ticks = 4;
  ctx.fillStyle = '#aaa'; ctx.font = '8px monospace'; ctx.textAlign = 'center';
  for (var t = 0; t <= ticks; t++) {
    var ts = minTs + (t / ticks) * (maxTs - minTs);
    var d = new Date(ts * 1000);
    var lbl = d.getHours() + ':' + ('0' + d.getMinutes()).slice(-2);
    ctx.fillText(lbl, pad.left + (t / ticks) * pw, h - pad.bottom + 14);
  }

  ctx.font = '9px monospace';
  for (var s = 0; s < series.length; s++) {
    var lx = pad.left + s * 90;
    ctx.fillStyle = series[s].color;
    ctx.fillRect(lx, 3, 10, 3);
    ctx.fillStyle = '#aaa'; ctx.textAlign = 'left';
    ctx.fillText(series[s].key.replace(/_/g, ' '), lx + 14, 8);
  }
};
