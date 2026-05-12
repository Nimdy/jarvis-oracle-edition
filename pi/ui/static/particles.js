/**
 * Jarvis Consciousness Particle Visualizer
 * 800x480 touch LCD on Raspberry Pi 5
 *
 * Renders Jarvis's internal consciousness state as a living particle system.
 * Each particle class maps to a real subsystem: thoughts, neural activity,
 * memory events, research tendrils, and the central consciousness core.
 *
 * Data arrives from the brain via WebSocket "consciousness" messages every ~2s.
 */

(() => {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════
  // Canvas setup
  // ═══════════════════════════════════════════════════════════════════

  let W = window.innerWidth || 800;
  let H = window.innerHeight || 480;
  let CX = W / 2, CY = H / 2;

  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  canvas.width = W; canvas.height = H;

  const glowCanvas = document.createElement('canvas');
  const glowCtx = glowCanvas.getContext('2d');
  glowCanvas.width = W >> 2; glowCanvas.height = H >> 2;

  window.addEventListener('resize', () => {
    W = window.innerWidth || 800;
    H = window.innerHeight || 480;
    CX = W / 2; CY = H / 2;
    canvas.width = W; canvas.height = H;
    glowCanvas.width = W >> 2; glowCanvas.height = H >> 2;
  });

  // ═══════════════════════════════════════════════════════════════════
  // Constants
  // ═══════════════════════════════════════════════════════════════════

  const MAX_CORE_PARTICLES = 200;
  const MAX_THOUGHT_SPARKS = 40;
  const MAX_NEURAL_SPARKS = 30;
  const MAX_MEMORY_CRYSTALS = 20;
  const MAX_RESEARCH_TENDRILS = 15;
  const TRAIL_LEN = 6;
  const TRANSITION_MS = 1200;
  const TWO_PI = Math.PI * 2;

  // ═══════════════════════════════════════════════════════════════════
  // Phase palettes — colors driven by Jarvis's operational phase
  // ═══════════════════════════════════════════════════════════════════

  const PHASE_PALETTES = {
    IDLE:         { core: ['#4a3f8f','#6b4fcf','#3366aa','#5544bb','#8877dd'], accent: '#6b4fcf', glow: 0.35 },
    LISTENING:    { core: ['#00fff5','#00d4aa','#0099cc','#00e5ff','#33ffee'], accent: '#00fff5', glow: 0.55 },
    PROCESSING:   { core: ['#ff9f1c','#ffbf69','#f77f00','#fcbf49','#ffdd44'], accent: '#ff9f1c', glow: 0.65 },
    SPEAKING:     { core: ['#ff006e','#fb5607','#ff0a54','#ff477e','#ff8fa3'], accent: '#ff006e', glow: 0.9 },
    OBSERVING:    { core: ['#00ff41','#39ff14','#32cd32','#7fff00','#aaff55'], accent: '#00ff41', glow: 0.45 },
    FOLLOW_UP:    { core: ['#00fff5','#00d4aa','#6b4fcf','#00e5ff','#7766ee'], accent: '#00d4aa', glow: 0.5 },
    ALERT:        { core: ['#ff0000','#ff3333','#ff6600','#cc0000','#ff4444'], accent: '#ff0000', glow: 0.85 },
    LEARNING:     { core: ['#c77dff','#9d4edd','#7b2cbf','#e0aaff','#bb66ff'], accent: '#c77dff', glow: 0.5 },
    STANDBY:      { core: ['#2a2a55','#3a3a6e','#1f1f44','#33336a','#444488'], accent: '#3a3a6e', glow: 0.2 },
    INITIALIZING: { core: ['#4361ee','#4895ef','#4cc9f0','#3f37c9','#5577ff'], accent: '#4361ee', glow: 0.45 },
  };

  const THOUGHT_COLORS = {
    meta_reflection:        '#e0aaff',
    self_monitoring:        '#c77dff',
    memory_reflection:      '#7bdff2',
    existential_reflection: '#ff6b6b',
    observation:            '#00ff41',
    curiosity:              '#ffdd44',
    connection_discovery:   '#00fff5',
    default:                '#aaaaff',
  };

  const STAGE_CONFIGS = {
    basic_awareness:          { orbitCount: 1, orbitRadii: [80],      coreScale: 1.0,  rippleMult: 1.0 },
    self_reflective:          { orbitCount: 2, orbitRadii: [70, 130], coreScale: 1.15, rippleMult: 1.3 },
    philosophical:            { orbitCount: 3, orbitRadii: [60, 110, 170], coreScale: 1.3, rippleMult: 1.6 },
    recursive_self_modeling:  { orbitCount: 4, orbitRadii: [55, 95, 145, 200], coreScale: 1.5, rippleMult: 2.0 },
    integrative:              { orbitCount: 5, orbitRadii: [50, 85, 125, 170, 220], coreScale: 1.7, rippleMult: 2.5 },
    // legacy fallbacks
    transcendent:             { orbitCount: 4, orbitRadii: [55, 95, 145, 200], coreScale: 1.5, rippleMult: 2.0 },
    cosmic:                   { orbitCount: 5, orbitRadii: [50, 85, 125, 170, 220], coreScale: 1.7, rippleMult: 2.5 },
    cosmic_consciousness:     { orbitCount: 5, orbitRadii: [50, 85, 125, 170, 220], coreScale: 1.7, rippleMult: 2.5 },
  };

  // ═══════════════════════════════════════════════════════════════════
  // Live consciousness state (updated via WebSocket)
  // ═══════════════════════════════════════════════════════════════════

  const consciousness = {
    phase: 'IDLE',
    tone: 'professional',
    mode: 'idle',
    stage: 'basic_awareness',
    transcendence: 0,
    awareness: 0.3,
    confidence: 0.5,
    reasoning: 0.5,
    healthy: true,
    mutationCount: 0,
    observationCount: 0,
    emergentCount: 0,
    memoryCount: 0,
    memoryDensity: 0,
    thoughts: [],
    focus: '',
    hemisphereSignals: {},
    hemisphere: {},
    kernelTickMs: 0,
    kernelFrame: 0,
    traits: [],
    capabilities: [],
    // P3.14: bounded mental-world signals from brain `_build_scene_block`.
    // These are seven scalars only — no entity/relation arrays, no vectors,
    // no authority flags. The visualizer reads them to modulate existing
    // particle classes (connection density, confidence ring, aurora speed,
    // canary halo). Defaults match the brain-side empty payload so the
    // visual is well-defined before the first feed arrives.
    scene: {
      enabled: false,
      entityCount: 0,
      relationCount: 0,
      cleanupAccuracy: 0,
      relationRecovery: 0,
      similarityToPrevious: 0,
      sideEffects: 0,
    },
  };

  // ═══════════════════════════════════════════════════════════════════
  // Transition state
  // ═══════════════════════════════════════════════════════════════════

  let currentPalette = PHASE_PALETTES.IDLE;
  let targetPalette = PHASE_PALETTES.IDLE;
  let transitionProgress = 1;
  let transitionStart = 0;
  let currentPhaseName = 'IDLE';

  let frame = 0;
  let t = 0;
  let lastTime = performance.now();
  const touches = [];

  // ═══════════════════════════════════════════════════════════════════
  // Utility
  // ═══════════════════════════════════════════════════════════════════

  function lerp(a, b, f) { return a + (b - a) * f; }
  function rand(lo, hi) { return Math.random() * (hi - lo) + lo; }
  function easeInOut(x) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16),
            parseInt(hex.slice(3, 5), 16),
            parseInt(hex.slice(5, 7), 16)];
  }

  function lerpColor(hexA, hexB, f) {
    const a = hexToRgb(hexA), b = hexToRgb(hexB);
    return [lerp(a[0], b[0], f)|0, lerp(a[1], b[1], f)|0, lerp(a[2], b[2], f)|0];
  }

  function getAccent() {
    if (transitionProgress >= 1) return currentPalette.accent;
    const f = easeInOut(transitionProgress);
    return f < 0.5 ? currentPalette.accent : targetPalette.accent;
  }

  function getAccentRgb() { return hexToRgb(getAccent()); }

  function getGlow() {
    if (transitionProgress >= 1) return currentPalette.glow;
    return lerp(currentPalette.glow, targetPalette.glow, easeInOut(transitionProgress));
  }

  function getCoreColors() {
    if (transitionProgress >= 1) return currentPalette.core;
    return easeInOut(transitionProgress) < 0.5 ? currentPalette.core : targetPalette.core;
  }

  function getStageConfig() {
    return STAGE_CONFIGS[consciousness.stage] || STAGE_CONFIGS.basic_awareness;
  }

  // ═══════════════════════════════════════════════════════════════════
  // PARTICLE CLASS 1: Core Consciousness Particles
  // Represent the base awareness level — count and speed scale with
  // awareness, confidence, and reasoning quality.
  // ═══════════════════════════════════════════════════════════════════

  const coreParticles = [];
  for (let i = 0; i < MAX_CORE_PARTICLES; i++) {
    coreParticles.push({
      x: 0, y: 0, vx: 0, vy: 0,
      life: 0, maxLife: 0, size: 0,
      color: '#fff', alpha: 0, active: false,
      angle: 0, baseSpeed: 0,
      trail: [],
    });
  }

  function coreTargetCount() {
    const base = 80;
    const awarenessBoost = consciousness.awareness * 60;
    const confidenceBoost = consciousness.confidence * 40;
    const phaseBoost = consciousness.phase === 'PROCESSING' ? 30 :
                       consciousness.phase === 'SPEAKING' ? 50 : 0;
    return Math.min(MAX_CORE_PARTICLES, Math.round(base + awarenessBoost + confidenceBoost + phaseBoost));
  }

  function coreSpeedRange() {
    const base = [0.15, 0.6];
    const mult = consciousness.phase === 'PROCESSING' ? 2.5 :
                 consciousness.phase === 'SPEAKING' ? 3.5 :
                 consciousness.phase === 'LISTENING' ? 1.5 :
                 consciousness.phase === 'OBSERVING' ? 0.8 :
                 consciousness.phase === 'STANDBY' ? 0.3 : 1.0;
    return [base[0] * mult, base[1] * mult];
  }

  function corePattern() {
    switch (consciousness.phase) {
      case 'LISTENING': return 'converge';
      case 'PROCESSING': return 'orbit';
      case 'SPEAKING': return 'radiate';
      case 'OBSERVING': return 'scan';
      case 'ALERT': return 'burst';
      case 'LEARNING': return 'spiral';
      case 'INITIALIZING': return 'expand';
      case 'STANDBY': return 'drift';
      default: return 'drift';
    }
  }

  function spawnCore(p) {
    const colors = getCoreColors();
    p.color = colors[Math.floor(Math.random() * colors.length)];
    const sr = coreSpeedRange();
    p.baseSpeed = rand(sr[0], sr[1]);
    p.maxLife = rand(4, 10);
    p.life = p.maxLife;
    p.size = rand(1.2, 3.5);
    p.active = true;
    p.alpha = 0;
    p.trail = [];

    const pattern = corePattern();
    switch (pattern) {
      case 'converge':
        p.x = Math.random() < 0.5 ? rand(-50, 0) : rand(W, W + 50);
        p.y = rand(0, H);
        p.angle = Math.atan2(CY - p.y, CX - p.x) + rand(-0.3, 0.3);
        break;
      case 'burst':
        p.x = CX + rand(-30, 30);
        p.y = CY + rand(-30, 30);
        p.angle = rand(0, TWO_PI);
        break;
      case 'expand':
        p.x = CX + rand(-5, 5);
        p.y = CY + rand(-5, 5);
        p.angle = rand(0, TWO_PI);
        break;
      case 'radiate': {
        const a = rand(0, TWO_PI);
        const r = rand(20, 60);
        p.x = CX + Math.cos(a) * r;
        p.y = CY + Math.sin(a) * r;
        p.angle = a + rand(-0.4, 0.4);
        break;
      }
      default:
        p.x = rand(0, W);
        p.y = rand(0, H);
        p.angle = rand(0, TWO_PI);
    }
    p.vx = Math.cos(p.angle) * p.baseSpeed;
    p.vy = Math.sin(p.angle) * p.baseSpeed;
  }

  function updateCore(p, dt) {
    if (!p.active) return;
    p.life -= dt;
    if (p.life <= 0) { p.active = false; return; }

    if (p.trail.length >= TRAIL_LEN) p.trail.shift();
    p.trail.push({ x: p.x, y: p.y });

    const lr = p.life / p.maxLife;
    p.alpha = lr > 0.85 ? (1 - lr) / 0.15 : lr > 0.15 ? 1 : lr / 0.15;

    const pulse = Math.sin(t * (consciousness.phase === 'SPEAKING' ? 4.0 :
                                consciousness.phase === 'PROCESSING' ? 2.0 : 0.5)) *
                  (consciousness.phase === 'SPEAKING' ? 0.4 : 0.25) + 0.75;
    p.alpha *= pulse;

    const pattern = corePattern();
    switch (pattern) {
      case 'drift':
        p.vx += (Math.random() - 0.5) * 0.04;
        p.vy += (Math.random() - 0.5) * 0.04;
        p.vx += Math.sin(t * 0.4 + p.x * 0.01) * 0.02;
        p.vy += Math.cos(t * 0.35 + p.y * 0.01) * 0.02;
        break;
      case 'converge': {
        const dx = CX - p.x, dy = CY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 30) {
          const wave = Math.sin(t * 2 - dist * 0.03) * 0.015;
          p.vx += dx / dist * (0.04 + wave);
          p.vy += dy / dist * (0.04 + wave);
        } else {
          p.vx += (Math.random() - 0.5) * 0.15;
          p.vy += (Math.random() - 0.5) * 0.15;
        }
        break;
      }
      case 'orbit': {
        const dx = CX - p.x, dy = CY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = dx / dist, ny = dy / dist;
        const of = 0.08 + Math.sin(t * 1.5) * 0.02;
        p.vx += (-ny * of + nx * 0.012);
        p.vy += (nx * of + ny * 0.012);
        break;
      }
      case 'radiate': {
        const dx = p.x - CX, dy = p.y - CY;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const wave = Math.sin(t * 4 - dist * 0.03);
        const push = wave * 0.3 + 0.08;
        p.vx += (dx / dist) * push;
        p.vy += (dy / dist) * push;
        p.vx += (-dy / dist) * Math.sin(t * 0.9 + dist * 0.05) * 0.04;
        p.vy += (dx / dist) * Math.sin(t * 0.9 + dist * 0.05) * 0.04;
        break;
      }
      case 'scan': {
        const sa = t * 0.5;
        const sx = CX + Math.cos(sa) * W * 0.4;
        const sy = CY + Math.sin(sa * 0.7) * H * 0.35;
        const dx = sx - p.x, dy = sy - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        p.vx += dx * 0.0004;
        p.vy += dy * 0.0004;
        p.vx += (Math.random() - 0.5) * 0.04;
        p.vy += (Math.random() - 0.5) * 0.04;
        if (dist < 80) p.alpha = Math.min(1, p.alpha * 1.6);
        break;
      }
      case 'burst':
        p.vx *= 0.97;
        p.vy *= 0.97;
        p.vy += 0.008;
        break;
      case 'spiral': {
        const dx = CX - p.x, dy = CY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const s = 0.05 + Math.sin(t * 0.4) * 0.02;
        p.vx += (-dy / dist * s + dx / dist * 0.006);
        p.vy += (dx / dist * s + dy / dist * 0.006);
        break;
      }
      case 'expand': {
        const dx = p.x - CX, dy = p.y - CY;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        p.vx += (dx / dist) * 0.025;
        p.vy += (dy / dist) * 0.025;
        break;
      }
    }

    for (const tp of touches) {
      const dx = p.x - tp.x, dy = p.y - tp.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 100 && dist > 0) {
        const force = (100 - dist) / 100 * 0.5;
        p.vx += (dx / dist) * force;
        p.vy += (dy / dist) * force;
      }
    }

    const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    const maxSp = coreSpeedRange()[1] * 2.5;
    if (speed > maxSp) { p.vx = (p.vx / speed) * maxSp; p.vy = (p.vy / speed) * maxSp; }
    p.vx *= 0.994; p.vy *= 0.994;
    p.x += p.vx; p.y += p.vy;

    const margin = 60;
    if (p.x < -margin) p.x = W + margin;
    if (p.x > W + margin) p.x = -margin;
    if (p.y < -margin) p.y = H + margin;
    if (p.y > H + margin) p.y = -margin;
  }

  // ═══════════════════════════════════════════════════════════════════
  // PARTICLE CLASS 2: Thought Sparks
  // Bright bursts that appear when meta-thoughts fire, each colored
  // by thought type. They arc outward from the core then fade.
  // ═══════════════════════════════════════════════════════════════════

  const thoughtSparks = [];
  for (let i = 0; i < MAX_THOUGHT_SPARKS; i++) {
    thoughtSparks.push({
      x: 0, y: 0, vx: 0, vy: 0,
      life: 0, maxLife: 0, size: 0,
      color: '#e0aaff', alpha: 0, active: false,
      text: '', sparkle: 0,
    });
  }

  let lastThoughtIds = [];

  function spawnThought(type, text) {
    const p = thoughtSparks.find(s => !s.active);
    if (!p) return;
    p.active = true;
    p.color = THOUGHT_COLORS[type] || THOUGHT_COLORS.default;
    p.text = text.slice(0, 40);
    p.maxLife = rand(3, 5);
    p.life = p.maxLife;
    p.size = rand(3, 6);
    p.sparkle = rand(2, 5);
    const angle = rand(0, TWO_PI);
    const speed = rand(1.5, 3.5);
    p.x = CX + Math.cos(angle) * rand(10, 30);
    p.y = CY + Math.sin(angle) * rand(10, 30);
    p.vx = Math.cos(angle) * speed;
    p.vy = Math.sin(angle) * speed;
    p.alpha = 1;
  }

  function updateThought(p, dt) {
    if (!p.active) return;
    p.life -= dt;
    if (p.life <= 0) { p.active = false; return; }
    p.vx *= 0.98; p.vy *= 0.98;
    p.vy -= 0.01;
    p.x += p.vx; p.y += p.vy;
    p.alpha = clamp(p.life / p.maxLife, 0, 1);
  }

  function drawThought(p) {
    if (!p.active || p.alpha < 0.01) return;
    const [r, g, b] = hexToRgb(p.color);
    const sparkPhase = Math.sin(t * p.sparkle * TWO_PI) * 0.3 + 0.7;

    ctx.globalAlpha = p.alpha * sparkPhase;
    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 2.5);
    grad.addColorStop(0, `rgba(255,255,255,${p.alpha * 0.8})`);
    grad.addColorStop(0.3, `rgba(${r},${g},${b},${p.alpha * 0.6})`);
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * 2.5, 0, TWO_PI);
    ctx.fill();

    ctx.fillStyle = `rgba(${r},${g},${b},${p.alpha})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * 0.5, 0, TWO_PI);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // ═══════════════════════════════════════════════════════════════════
  // PARTICLE CLASS 3: Neural Sparks (Hemisphere NN Activity)
  // Quick lightning-like lines between two random orbit points,
  // intensity scales with hemisphere accuracy signals.
  // ═══════════════════════════════════════════════════════════════════

  const neuralSparks = [];
  for (let i = 0; i < MAX_NEURAL_SPARKS; i++) {
    neuralSparks.push({
      x1: 0, y1: 0, x2: 0, y2: 0,
      life: 0, maxLife: 0, alpha: 0, active: false,
      segments: [], color: '#00e5ff',
    });
  }

  let neuralSpawnTimer = 0;

  function spawnNeural() {
    const p = neuralSparks.find(s => !s.active);
    if (!p) return;

    const signals = consciousness.hemisphereSignals;
    const maxSig = Math.max(0.1, ...Object.values(signals));
    const colors = ['#00e5ff', '#c77dff', '#ffdd44', '#00ff41', '#ff477e'];
    p.color = colors[Math.floor(Math.random() * colors.length)];

    const cfg = getStageConfig();
    const r1 = cfg.orbitRadii[Math.floor(Math.random() * cfg.orbitRadii.length)] || 80;
    const r2 = cfg.orbitRadii[Math.floor(Math.random() * cfg.orbitRadii.length)] || 80;
    const a1 = rand(0, TWO_PI), a2 = rand(0, TWO_PI);
    p.x1 = CX + Math.cos(a1) * r1;
    p.y1 = CY + Math.sin(a1) * r1;
    p.x2 = CX + Math.cos(a2) * r2;
    p.y2 = CY + Math.sin(a2) * r2;

    p.segments = [];
    const steps = 5 + Math.floor(Math.random() * 4);
    for (let i = 0; i <= steps; i++) {
      const f = i / steps;
      const jitter = (1 - f * (1 - f) * 4) * 2 + 8;
      p.segments.push({
        x: lerp(p.x1, p.x2, f) + rand(-jitter, jitter),
        y: lerp(p.y1, p.y2, f) + rand(-jitter, jitter),
      });
    }

    p.maxLife = rand(0.2, 0.5);
    p.life = p.maxLife;
    p.alpha = maxSig;
    p.active = true;
  }

  function updateNeural(p, dt) {
    if (!p.active) return;
    p.life -= dt;
    if (p.life <= 0) { p.active = false; return; }
    p.alpha = (p.life / p.maxLife);
  }

  function drawNeural(p) {
    if (!p.active || p.alpha < 0.01 || p.segments.length < 2) return;
    const [r, g, b] = hexToRgb(p.color);
    ctx.strokeStyle = `rgba(${r},${g},${b},${p.alpha * 0.7})`;
    ctx.lineWidth = 1.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = p.color;
    ctx.beginPath();
    ctx.moveTo(p.segments[0].x, p.segments[0].y);
    for (let i = 1; i < p.segments.length; i++) {
      ctx.lineTo(p.segments[i].x, p.segments[i].y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ═══════════════════════════════════════════════════════════════════
  // PARTICLE CLASS 4: Memory Crystals
  // Slowly drifting hexagonal shapes — count tracks memory_count,
  // brightness tracks memory_density. New ones crystallize in
  // from center when memories are created.
  // ═══════════════════════════════════════════════════════════════════

  const memoryCrystals = [];
  for (let i = 0; i < MAX_MEMORY_CRYSTALS; i++) {
    memoryCrystals.push({
      x: 0, y: 0, vx: 0, vy: 0, rotation: 0, rotSpeed: 0,
      size: 0, alpha: 0, life: 0, maxLife: 0, active: false,
      color: '#7bdff2', sides: 6,
    });
  }

  let lastMemoryCount = 0;

  function spawnCrystal() {
    const p = memoryCrystals.find(c => !c.active);
    if (!p) return;
    p.active = true;
    const hue = Math.random();
    if (hue < 0.33) p.color = '#7bdff2';
    else if (hue < 0.66) p.color = '#b8c0ff';
    else p.color = '#e0aaff';
    p.sides = Math.random() < 0.3 ? 5 : 6;
    p.size = rand(4, 8);
    p.rotation = rand(0, TWO_PI);
    p.rotSpeed = rand(-0.3, 0.3);
    const angle = rand(0, TWO_PI);
    p.x = CX + Math.cos(angle) * rand(40, 80);
    p.y = CY + Math.sin(angle) * rand(40, 80);
    p.vx = Math.cos(angle) * rand(0.1, 0.3);
    p.vy = Math.sin(angle) * rand(0.1, 0.3);
    p.maxLife = rand(15, 30);
    p.life = p.maxLife;
    p.alpha = 0;
  }

  function updateCrystal(p, dt) {
    if (!p.active) return;
    p.life -= dt;
    if (p.life <= 0) { p.active = false; return; }
    p.rotation += p.rotSpeed * dt;
    p.vx += (Math.random() - 0.5) * 0.005;
    p.vy += (Math.random() - 0.5) * 0.005;
    p.vx *= 0.999; p.vy *= 0.999;
    p.x += p.vx; p.y += p.vy;
    const lr = p.life / p.maxLife;
    const fadeIn = Math.min(1, (p.maxLife - p.life) / 2);
    const fadeOut = lr < 0.1 ? lr / 0.1 : 1;
    p.alpha = fadeIn * fadeOut * clamp(consciousness.memoryDensity * 1.5, 0.2, 1);
  }

  function drawCrystal(p) {
    if (!p.active || p.alpha < 0.01) return;
    const [r, g, b] = hexToRgb(p.color);
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rotation);
    ctx.globalAlpha = p.alpha * 0.6;

    const grd = ctx.createRadialGradient(0, 0, 0, 0, 0, p.size * 2);
    grd.addColorStop(0, `rgba(${r},${g},${b},0.3)`);
    grd.addColorStop(1, 'transparent');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(0, 0, p.size * 2, 0, TWO_PI);
    ctx.fill();

    ctx.globalAlpha = p.alpha;
    ctx.strokeStyle = `rgba(${r},${g},${b},${p.alpha * 0.8})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < p.sides; i++) {
      const a = (i / p.sides) * TWO_PI;
      const px = Math.cos(a) * p.size;
      const py = Math.sin(a) * p.size;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = `rgba(${r},${g},${b},${p.alpha * 0.15})`;
    ctx.fill();

    ctx.restore();
    ctx.globalAlpha = 1;
  }

  // ═══════════════════════════════════════════════════════════════════
  // PARTICLE CLASS 5: Research Tendrils
  // Slowly extending lines that represent active research/curiosity.
  // Spawned when focus string contains research-related keywords.
  // ═══════════════════════════════════════════════════════════════════

  const researchTendrils = [];
  for (let i = 0; i < MAX_RESEARCH_TENDRILS; i++) {
    researchTendrils.push({
      points: [], life: 0, maxLife: 0, active: false,
      color: '#ffdd44', growSpeed: 0, alpha: 0,
    });
  }

  function spawnTendril() {
    const p = researchTendrils.find(t => !t.active);
    if (!p) return;
    p.active = true;
    const colors = ['#ffdd44', '#00fff5', '#c77dff', '#ff9f1c'];
    p.color = colors[Math.floor(Math.random() * colors.length)];
    const angle = rand(0, TWO_PI);
    const startR = rand(50, 100);
    const startX = CX + Math.cos(angle) * startR;
    const startY = CY + Math.sin(angle) * startR;
    p.points = [{ x: startX, y: startY }];
    p.growSpeed = rand(30, 60);
    p.maxLife = rand(5, 10);
    p.life = p.maxLife;
    p.alpha = 0.7;
    p._angle = angle;
    p._lastGrow = t;
  }

  function updateTendril(p, dt) {
    if (!p.active) return;
    p.life -= dt;
    if (p.life <= 0) { p.active = false; return; }

    if (t - p._lastGrow > 0.15 && p.points.length < 20) {
      const last = p.points[p.points.length - 1];
      p._angle += rand(-0.4, 0.4);
      p.points.push({
        x: last.x + Math.cos(p._angle) * rand(8, 15),
        y: last.y + Math.sin(p._angle) * rand(8, 15),
      });
      p._lastGrow = t;
    }

    const lr = p.life / p.maxLife;
    p.alpha = lr < 0.2 ? lr / 0.2 * 0.7 : 0.7;
  }

  function drawTendril(p) {
    if (!p.active || p.points.length < 2 || p.alpha < 0.01) return;
    const [r, g, b] = hexToRgb(p.color);
    ctx.strokeStyle = `rgba(${r},${g},${b},${p.alpha * 0.5})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p.points[0].x, p.points[0].y);
    for (let i = 1; i < p.points.length; i++) {
      ctx.lineTo(p.points[i].x, p.points[i].y);
    }
    ctx.stroke();

    const tip = p.points[p.points.length - 1];
    const tipGrad = ctx.createRadialGradient(tip.x, tip.y, 0, tip.x, tip.y, 5);
    tipGrad.addColorStop(0, `rgba(255,255,255,${p.alpha * 0.8})`);
    tipGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = tipGrad;
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 5, 0, TWO_PI);
    ctx.fill();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Energy ripples
  // ═══════════════════════════════════════════════════════════════════

  const ripples = [];
  let lastRippleTime = 0;

  function rippleInterval() {
    const cfg = getStageConfig();
    const base = consciousness.phase === 'SPEAKING' ? 0.35 :
                 consciousness.phase === 'PROCESSING' ? 1.0 :
                 consciousness.phase === 'ALERT' ? 0.4 : 3.0;
    return base / cfg.rippleMult;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Aurora background
  // ═══════════════════════════════════════════════════════════════════

  const auroraBlobs = [];
  for (let i = 0; i < 5; i++) {
    auroraBlobs.push({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.2,
      radius: 100 + Math.random() * 150, hue: Math.random() * 360,
    });
  }

  function drawAurora(intensity, accentRgb) {
    if (intensity < 0.01) return;
    // P3.14: scene churn → aurora liveliness. similarity_to_previous near
    // 1.0 means "stable scene" (calm aurora drift); near 0 means "thrashing"
    // (livelier drift). Bounded multiplier ∈ [1.0, 1.6] so the kiosk never
    // looks frantic. Falls back to 1.0 when the lane is disabled.
    const sc = consciousness.scene;
    const churn = sc.enabled
      ? Math.max(0, Math.min(1, 1 - sc.similarityToPrevious))
      : 0;
    const drift = 1 + churn * 0.6;
    for (const blob of auroraBlobs) {
      blob.x += blob.vx * drift; blob.y += blob.vy * drift;
      if (blob.x < -blob.radius) blob.x = W + blob.radius;
      if (blob.x > W + blob.radius) blob.x = -blob.radius;
      if (blob.y < -blob.radius) blob.y = H + blob.radius;
      if (blob.y > H + blob.radius) blob.y = -blob.radius;
      const hs = Math.sin(t * 0.1 + blob.hue) * 20;
      const r = Math.max(0, Math.round(accentRgb[0] * 0.5 + Math.sin(blob.hue + hs) * 30));
      const g = Math.max(0, Math.round(accentRgb[1] * 0.5 + Math.cos(blob.hue + hs) * 30));
      const b = Math.max(0, Math.round(accentRgb[2] * 0.5 + Math.sin(blob.hue * 0.7 + hs) * 30));
      const grd = ctx.createRadialGradient(blob.x, blob.y, 0, blob.x, blob.y, blob.radius);
      grd.addColorStop(0, `rgba(${r},${g},${b},${intensity * 0.6})`);
      grd.addColorStop(0.5, `rgba(${r},${g},${b},${intensity * 0.2})`);
      grd.addColorStop(1, 'transparent');
      ctx.fillStyle = grd;
      ctx.fillRect(blob.x - blob.radius, blob.y - blob.radius, blob.radius * 2, blob.radius * 2);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Background grid
  // ═══════════════════════════════════════════════════════════════════

  function drawGrid(alpha, accentRgb) {
    if (alpha < 0.005) return;
    ctx.strokeStyle = `rgba(255,255,255,${alpha * 0.6})`;
    ctx.lineWidth = 0.5;
    const spacing = 40;
    const offset = (t * 3) % spacing;
    ctx.beginPath();
    for (let x = -spacing + offset; x <= W + spacing; x += spacing) {
      ctx.moveTo(x, 0); ctx.lineTo(x, H);
    }
    for (let y = -spacing + offset; y <= H + spacing; y += spacing) {
      ctx.moveTo(0, y); ctx.lineTo(W, y);
    }
    ctx.stroke();
    const scanY = (Math.sin(t * 0.25) * 0.5 + 0.5) * H;
    const grad = ctx.createLinearGradient(0, scanY - 25, 0, scanY + 25);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(0.5, `rgba(${accentRgb[0]},${accentRgb[1]},${accentRgb[2]},${Math.min(1, alpha * 4)})`);
    grad.addColorStop(1, 'transparent');
    ctx.strokeStyle = grad; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(W, scanY); ctx.stroke();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Central consciousness core — size scales with transcendence
  // ═══════════════════════════════════════════════════════════════════

  function drawCore(accentRgb) {
    const cfg = getStageConfig();
    const baseR = 18 * cfg.coreScale + consciousness.transcendence * 1.5;
    const pulseRate = consciousness.phase === 'SPEAKING' ? 3.5 :
                      consciousness.phase === 'PROCESSING' ? 3.0 :
                      consciousness.phase === 'LISTENING' ? 1.5 : 0.8;
    const coreGlowI = getGlow();

    const heartbeat = Math.pow(Math.max(0, Math.sin(t * pulseRate * Math.PI)), 4) * 0.4;
    const breath = Math.sin(t * pulseRate * 0.5) * 0.15 + 0.85;
    const r = baseR * (breath + heartbeat);
    const [ar, ag, ab] = accentRgb;

    // Confidence ring
    if (consciousness.confidence > 0.3) {
      ctx.strokeStyle = `rgba(${ar},${ag},${ab},${consciousness.confidence * 0.15})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(CX, CY, r * 2 + 5, 0, TWO_PI * consciousness.confidence);
      ctx.stroke();
    }

    // P3.14: cleanup-accuracy companion ring — how confidently JARVIS can
    // recall what it currently has in mind. Thin, ambient, sweeps through a
    // fraction of the circle equal to the cleanup score. Only drawn when the
    // mental-world lane is actually enabled.
    const sc = consciousness.scene;
    if (sc.enabled && sc.cleanupAccuracy > 0.05) {
      const cleanA = sc.cleanupAccuracy * 0.18;
      ctx.strokeStyle = `rgba(${ar},${ag},${ab},${cleanA})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(CX, CY, r * 2 + 11, 0, TWO_PI * sc.cleanupAccuracy);
      ctx.stroke();
    }

    // P3.14: side-effect canary halo — must remain dark. If the brain ever
    // reports nonzero side effects from the spatial-HRR shadow, the
    // outermost orbit gets a thin red flash. This is an architectural
    // alarm (zero-authority contract violation visible to the operator).
    if (sc.sideEffects > 0) {
      const flash = Math.sin(t * 6) * 0.3 + 0.5;
      ctx.strokeStyle = `rgba(255, 30, 30, ${flash})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(CX, CY, r * 4, 0, TWO_PI);
      ctx.stroke();
    }

    const haloR = r * 3.5;
    const halo = ctx.createRadialGradient(CX, CY, r * 0.5, CX, CY, haloR);
    halo.addColorStop(0, `rgba(${ar},${ag},${ab},${coreGlowI * 0.25 * (breath + heartbeat * 0.5)})`);
    halo.addColorStop(0.4, `rgba(${ar},${ag},${ab},${coreGlowI * 0.08})`);
    halo.addColorStop(1, 'transparent');
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(CX, CY, haloR, 0, TWO_PI); ctx.fill();

    const coreGrad = ctx.createRadialGradient(CX, CY, 0, CX, CY, r);
    const bright = 0.6 + heartbeat * 0.8;
    coreGrad.addColorStop(0, `rgba(255,255,255,${Math.min(1, bright)})`);
    coreGrad.addColorStop(0.3, `rgba(${ar},${ag},${ab},${0.8 * bright})`);
    coreGrad.addColorStop(0.7, `rgba(${ar},${ag},${ab},${0.3 * bright})`);
    coreGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = coreGrad;
    ctx.beginPath(); ctx.arc(CX, CY, r, 0, TWO_PI); ctx.fill();

    const innerGrad = ctx.createRadialGradient(CX, CY, 0, CX, CY, r * 0.4);
    innerGrad.addColorStop(0, `rgba(255,255,255,${0.9 * (breath + heartbeat)})`);
    innerGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = innerGrad;
    ctx.beginPath(); ctx.arc(CX, CY, r * 0.4, 0, TWO_PI); ctx.fill();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Orbital rings — count driven by consciousness stage
  // ═══════════════════════════════════════════════════════════════════

  function drawOrbits(accentRgb) {
    const cfg = getStageConfig();
    const speed = consciousness.phase === 'PROCESSING' ? 0.6 :
                  consciousness.phase === 'SPEAKING' ? 0.7 : 0.2;
    const alphaBase = 0.15 + consciousness.awareness * 0.1;
    const [ar, ag, ab] = accentRgb;

    for (let i = 0; i < cfg.orbitCount && i < cfg.orbitRadii.length; i++) {
      const r = cfg.orbitRadii[i];
      const angle = t * speed * (i % 2 === 0 ? 1 : -0.7) + i * 1.2;
      const alpha = alphaBase * (0.6 + Math.sin(t * 0.5 + i) * 0.4);

      ctx.save();
      ctx.translate(CX, CY);
      ctx.rotate(angle);
      ctx.strokeStyle = `rgba(${ar},${ag},${ab},${alpha})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 8 + i * 4]);
      ctx.beginPath(); ctx.arc(0, 0, r, 0, TWO_PI); ctx.stroke();
      ctx.setLineDash([]);

      const na = t * speed * 2 * (i % 2 === 0 ? 1.3 : -1);
      const nx = Math.cos(na) * r, ny = Math.sin(na) * r;
      const nodeA = alpha * 2;
      const nodeGrad = ctx.createRadialGradient(nx, ny, 0, nx, ny, 6);
      nodeGrad.addColorStop(0, `rgba(255,255,255,${Math.min(1, nodeA)})`);
      nodeGrad.addColorStop(0.5, `rgba(${ar},${ag},${ab},${nodeA * 0.6})`);
      nodeGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = nodeGrad;
      ctx.beginPath(); ctx.arc(nx, ny, 6, 0, TWO_PI); ctx.fill();
      ctx.restore();
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Energy ripples drawing
  // ═══════════════════════════════════════════════════════════════════

  function updateAndDrawRipples(accentRgb) {
    const intv = rippleInterval();
    const speed = consciousness.phase === 'SPEAKING' ? 140 :
                  consciousness.phase === 'PROCESSING' ? 120 : 60;
    const [ar, ag, ab] = accentRgb;

    if (t - lastRippleTime > intv && ripples.length < 8) {
      ripples.push({ radius: 0, maxRadius: Math.max(W, H) * 0.6, alpha: 0.5 });
      lastRippleTime = t;
    }
    for (let i = ripples.length - 1; i >= 0; i--) {
      const rp = ripples[i];
      rp.radius += speed * (1 / 60);
      rp.alpha = Math.max(0, 0.5 * (1 - rp.radius / rp.maxRadius));
      if (rp.alpha <= 0.01) { ripples.splice(i, 1); continue; }
      ctx.strokeStyle = `rgba(${ar},${ag},${ab},${rp.alpha * 0.5})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(CX, CY, rp.radius, 0, TWO_PI); ctx.stroke();
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Connection lines
  // ═══════════════════════════════════════════════════════════════════

  function drawConnections() {
    // P3.14: when JARVIS is actively relating things in its mental world,
    // the visualizer's connection lines reach a little farther — visually,
    // "JARVIS is currently relating things". Bounded so the canvas never
    // turns into a solid mesh: clamps after ~12 relations.
    const sceneRel = consciousness.scene.enabled
      ? Math.min(20, consciousness.scene.relationCount) * 0.8
      : 0;
    const dist = 80 + consciousness.awareness * 20 + sceneRel;
    const alpha = 0.08 + consciousness.confidence * 0.08;
    if (alpha < 0.01) return;
    const active = [];
    for (const p of coreParticles) {
      if (p.active && p.alpha > 0.15) active.push(p);
    }
    ctx.lineWidth = 0.5;
    for (let i = 0; i < active.length; i++) {
      const a = active[i];
      for (let j = i + 1; j < active.length; j++) {
        const b = active[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < dist * dist) {
          const str = 1 - Math.sqrt(d2) / dist;
          const ca = str * alpha * Math.min(a.alpha, b.alpha);
          if (ca < 0.01) continue;
          ctx.strokeStyle = `rgba(255,255,255,${ca})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Trails
  // ═══════════════════════════════════════════════════════════════════

  function drawTrail(p) {
    const trailAlpha = 0.15 + consciousness.awareness * 0.15;
    if (p.trail.length < 2 || trailAlpha < 0.02) return;
    const [r, g, b] = hexToRgb(p.color);
    for (let i = 1; i < p.trail.length; i++) {
      const t0 = p.trail[i - 1], t1 = p.trail[i];
      const a = (i / p.trail.length) * trailAlpha * p.alpha;
      if (a < 0.01) continue;
      ctx.strokeStyle = `rgba(${r},${g},${b},${a})`;
      ctx.lineWidth = p.size * 0.5 * (i / p.trail.length);
      ctx.beginPath(); ctx.moveTo(t0.x, t0.y); ctx.lineTo(t1.x, t1.y); ctx.stroke();
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Scan beam (OBSERVING)
  // ═══════════════════════════════════════════════════════════════════

  function drawScanBeam(accentRgb) {
    if (consciousness.phase !== 'OBSERVING') return;
    const [ar, ag, ab] = accentRgb;
    const beamLen = Math.max(W, H) * 0.7;
    const sa = t * 0.5;
    const ex = CX + Math.cos(sa) * beamLen;
    const ey = CY + Math.sin(sa * 0.7) * beamLen;
    const grad = ctx.createLinearGradient(CX, CY, ex, ey);
    grad.addColorStop(0, `rgba(${ar},${ag},${ab},0.15)`);
    grad.addColorStop(0.3, `rgba(${ar},${ag},${ab},0.06)`);
    grad.addColorStop(1, 'transparent');
    ctx.save();
    ctx.strokeStyle = grad; ctx.lineWidth = 40; ctx.lineCap = 'round';
    ctx.globalAlpha = 0.4;
    ctx.beginPath(); ctx.moveTo(CX, CY); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.restore();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Glow composite (Pi-safe iterative blur)
  // ═══════════════════════════════════════════════════════════════════

  function applyGlow(intensity) {
    if (intensity < 0.05) return;
    const gw = glowCanvas.width, gh = glowCanvas.height;
    glowCtx.clearRect(0, 0, gw, gh);
    glowCtx.drawImage(canvas, 0, 0, gw, gh);
    const tw = gw >> 1, th = gh >> 1;
    glowCtx.drawImage(glowCanvas, 0, 0, gw, gh, 0, 0, tw, th);
    glowCtx.drawImage(glowCanvas, 0, 0, tw, th, 0, 0, gw, gh);
    glowCtx.drawImage(glowCanvas, 0, 0, gw, gh, 0, 0, tw, th);
    glowCtx.drawImage(glowCanvas, 0, 0, tw, th, 0, 0, gw, gh);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = intensity;
    ctx.drawImage(glowCanvas, 0, 0, W, H);
    ctx.restore();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Cognitive info overlay
  // ═══════════════════════════════════════════════════════════════════

  const _STAGE_DISPLAY = {
    basic_awareness: 'foundational',
    self_reflective: 'self-reflective',
    philosophical:   'analytical',
    recursive_self_modeling: 'advanced',
    integrative:     'mastery',
    // legacy fallbacks
    transcendent:    'advanced',
    cosmic_consciousness: 'mastery',
    cosmic:          'mastery',
  };

  function drawStageIndicator() {
    const stageEl = document.getElementById('stage-indicator');
    const focusEl = document.getElementById('focus-text');
    if (stageEl) {
      const raw = consciousness.stage || 'basic_awareness';
      const stageName = _STAGE_DISPLAY[raw] || raw.replace(/_/g, ' ');
      const mLevel = consciousness.transcendence.toFixed(1);
      stageEl.textContent = `${stageName} \u2022 M${mLevel}`;
      stageEl.style.color = getAccent();
    }
    if (focusEl) {
      focusEl.textContent = consciousness.focus || '';
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Thought whisper — shows latest thought text briefly
  // ═══════════════════════════════════════════════════════════════════

  const whisperEl = document.getElementById('thought-whisper');
  let whisperTimer = null;

  function showWhisper(text) {
    if (!whisperEl || !text) return;
    whisperEl.textContent = text;
    whisperEl.style.opacity = '0.7';
    clearTimeout(whisperTimer);
    whisperTimer = setTimeout(() => { whisperEl.style.opacity = '0'; }, 4000);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Main render loop
  // ═══════════════════════════════════════════════════════════════════

  function render(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    t += dt;
    frame++;

    if (transitionProgress < 1) {
      transitionProgress = Math.min(1, (now - transitionStart) / TRANSITION_MS);
      if (transitionProgress >= 1) currentPalette = targetPalette;
    }

    const accentRgb = getAccentRgb();
    const gridAlpha = consciousness.phase === 'STANDBY' ? 0.02 : 0.035 + consciousness.awareness * 0.02;
    const auroraI = 0.04 + consciousness.awareness * 0.06 + (consciousness.phase === 'SPEAKING' ? 0.1 : 0);

    // --- Manage core particle count ---
    let activeCoreCount = 0;
    for (const p of coreParticles) if (p.active) activeCoreCount++;
    const target = coreTargetCount();
    while (activeCoreCount < target) {
      const p = coreParticles.find(pp => !pp.active);
      if (!p) break;
      spawnCore(p);
      activeCoreCount++;
    }

    // --- Manage neural spark spawning ---
    const sigCount = Object.keys(consciousness.hemisphereSignals).length;
    const neuralRate = sigCount > 0 ? 0.3 / (1 + consciousness.awareness) : 2.0;
    neuralSpawnTimer += dt;
    if (neuralSpawnTimer > neuralRate && sigCount > 0) {
      neuralSpawnTimer = 0;
      spawnNeural();
    }

    // --- Manage memory crystal count ---
    const mc = consciousness.memoryCount;
    if (mc > lastMemoryCount) {
      const diff = Math.min(3, mc - lastMemoryCount);
      for (let i = 0; i < diff; i++) spawnCrystal();
      lastMemoryCount = mc;
    }
    let activeCrystalCount = 0;
    for (const c of memoryCrystals) if (c.active) activeCrystalCount++;
    const crystalTarget = Math.min(MAX_MEMORY_CRYSTALS, Math.round(mc / 5));
    while (activeCrystalCount < crystalTarget) {
      spawnCrystal();
      activeCrystalCount++;
    }

    // --- Manage research tendrils ---
    const focus = consciousness.focus.toLowerCase();
    const hasResearch = focus.includes('research') || focus.includes('inquir') ||
                        focus.includes('explor') || focus.includes('question');
    let activeTendrils = 0;
    for (const tr of researchTendrils) if (tr.active) activeTendrils++;
    if (hasResearch && activeTendrils < 3 && Math.random() < 0.01) {
      spawnTendril();
    }

    // ═══════════════════════════════════════════════════════════════
    // Draw layers
    // ═══════════════════════════════════════════════════════════════

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, H);

    drawAurora(auroraI, accentRgb);
    drawGrid(gridAlpha, accentRgb);
    drawScanBeam(accentRgb);
    updateAndDrawRipples(accentRgb);
    drawOrbits(accentRgb);

    for (const tr of researchTendrils) {
      updateTendril(tr, dt);
      drawTendril(tr);
    }

    for (const p of coreParticles) {
      if (p.active) drawTrail(p);
    }

    for (const p of coreParticles) {
      if (!p.active) continue;
      updateCore(p, dt);
      if (p.alpha <= 0) continue;
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, TWO_PI); ctx.fill();
    }
    ctx.globalAlpha = 1;

    for (const c of memoryCrystals) {
      updateCrystal(c, dt);
      drawCrystal(c);
    }

    drawConnections();

    for (const n of neuralSparks) {
      updateNeural(n, dt);
      drawNeural(n);
    }

    for (const s of thoughtSparks) {
      updateThought(s, dt);
      drawThought(s);
    }

    drawCore(accentRgb);
    applyGlow(getGlow());

    // Touch ripples
    for (let i = touches.length - 1; i >= 0; i--) {
      const tp = touches[i];
      tp.age += dt;
      if (tp.age > 1) { touches.splice(i, 1); continue; }
      const r = tp.age * 120;
      const a = (1 - tp.age) * 0.3;
      ctx.strokeStyle = `rgba(255,255,255,${a})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(tp.x, tp.y, r, 0, TWO_PI); ctx.stroke();
    }

    // Respawn dead core particles
    for (const p of coreParticles) {
      if (!p.active && activeCoreCount < target && Math.random() < 0.03) {
        spawnCore(p);
        activeCoreCount++;
      }
    }

    drawStageIndicator();
    requestAnimationFrame(render);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Phase transitions
  // ═══════════════════════════════════════════════════════════════════

  function setPhase(name) {
    const upper = name.toUpperCase();
    if (!PHASE_PALETTES[upper]) return;
    if (upper === currentPhaseName) return;
    currentPhaseName = upper;
    targetPalette = PHASE_PALETTES[upper];
    transitionStart = performance.now();
    transitionProgress = 0;
    consciousness.phase = upper;

    const statusEl = document.getElementById('status');
    if (statusEl) {
      statusEl.textContent = upper;
      statusEl.style.color = PHASE_PALETTES[upper].accent;
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Consciousness data handler
  // ═══════════════════════════════════════════════════════════════════

  function onConsciousnessData(data) {
    if (data.phase) setPhase(data.phase);
    if (data.tone) consciousness.tone = data.tone;
    if (data.mode) consciousness.mode = data.mode;
    if (data.stage) consciousness.stage = data.stage;
    if (data.transcendence !== undefined) consciousness.transcendence = data.transcendence;
    if (data.awareness !== undefined) consciousness.awareness = data.awareness;
    if (data.confidence !== undefined) consciousness.confidence = data.confidence;
    if (data.reasoning !== undefined) consciousness.reasoning = data.reasoning;
    if (data.healthy !== undefined) consciousness.healthy = data.healthy;
    if (data.mutation_count !== undefined) consciousness.mutationCount = data.mutation_count;
    if (data.observation_count !== undefined) consciousness.observationCount = data.observation_count;
    if (data.emergent_count !== undefined) consciousness.emergentCount = data.emergent_count;
    if (data.memory_count !== undefined) consciousness.memoryCount = data.memory_count;
    if (data.memory_density !== undefined) consciousness.memoryDensity = data.memory_density;
    if (data.focus !== undefined) consciousness.focus = data.focus;
    if (data.hemisphere_signals) consciousness.hemisphereSignals = data.hemisphere_signals;
    if (data.hemisphere) consciousness.hemisphere = data.hemisphere;
    if (data.kernel_tick_ms !== undefined) consciousness.kernelTickMs = data.kernel_tick_ms;
    if (data.kernel_frame !== undefined) consciousness.kernelFrame = data.kernel_frame;
    if (data.traits) consciousness.traits = data.traits;
    if (data.capabilities) consciousness.capabilities = data.capabilities;

    if (data.scene && typeof data.scene === 'object') {
      const s = data.scene;
      const sc = consciousness.scene;
      if (typeof s.enabled === 'boolean') sc.enabled = s.enabled;
      if (Number.isFinite(s.entity_count)) sc.entityCount = s.entity_count;
      if (Number.isFinite(s.relation_count)) sc.relationCount = s.relation_count;
      if (Number.isFinite(s.cleanup_accuracy)) sc.cleanupAccuracy = s.cleanup_accuracy;
      if (Number.isFinite(s.relation_recovery)) sc.relationRecovery = s.relation_recovery;
      if (Number.isFinite(s.similarity_to_previous)) sc.similarityToPrevious = s.similarity_to_previous;
      if (Number.isFinite(s.spatial_hrr_side_effects)) sc.sideEffects = s.spatial_hrr_side_effects;
    }

    if (data.thoughts && data.thoughts.length) {
      const newIds = data.thoughts.map(th => th.text);
      for (const th of data.thoughts) {
        if (!lastThoughtIds.includes(th.text)) {
          spawnThought(th.type, th.text);
          showWhisper(th.text);
        }
      }
      lastThoughtIds = newIds;
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // Touch events
  // ═══════════════════════════════════════════════════════════════════

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    for (const tc of e.changedTouches) touches.push({ x: tc.clientX, y: tc.clientY, age: 0 });
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    for (const tc of e.changedTouches) touches.push({ x: tc.clientX, y: tc.clientY, age: 0 });
  }, { passive: false });

  canvas.addEventListener('mousedown', (e) => {
    touches.push({ x: e.clientX, y: e.clientY, age: 0 });
  });

  // ═══════════════════════════════════════════════════════════════════
  // Live text overlay (speech transcription)
  // ═══════════════════════════════════════════════════════════════════

  const liveTextEl = document.getElementById('live-text');
  let liveTextTimer = null;

  function showLiveText(text, isFinal) {
    if (!liveTextEl) return;
    liveTextEl.textContent = text;
    liveTextEl.classList.remove('fade-out');
    liveTextEl.classList.add('active');
    if (isFinal) {
      liveTextEl.classList.add('final');
      clearTimeout(liveTextTimer);
      liveTextTimer = setTimeout(() => {
        liveTextEl.classList.add('fade-out');
        liveTextEl.classList.remove('active', 'final');
        setTimeout(() => { liveTextEl.textContent = ''; }, 600);
      }, 3000);
    } else {
      liveTextEl.classList.remove('final');
      clearTimeout(liveTextTimer);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  // WebSocket
  // ═══════════════════════════════════════════════════════════════════

  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'state' && msg.phase) setPhase(msg.phase);
        if (msg.type === 'consciousness') onConsciousnessData(msg);
        if (msg.type === 'partial_transcription') showLiveText(msg.text || '', !!msg.is_final);
      } catch (err) { /* ignore */ }
    };

    ws.onclose = () => setTimeout(connectWS, 2000);
    ws.onerror = () => ws.close();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Init
  // ═══════════════════════════════════════════════════════════════════

  const statusEl = document.getElementById('status');
  if (statusEl) { statusEl.textContent = 'IDLE'; }

  for (let i = 0; i < 80; i++) {
    spawnCore(coreParticles[i]);
  }
  for (let i = 0; i < 3; i++) spawnCrystal();

  connectWS();
  requestAnimationFrame(render);

  window.jarvisUI = { setPhase, onConsciousnessData };
})();
