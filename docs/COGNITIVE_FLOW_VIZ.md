# Cognitive Flow Visualization — SCOPE & DESIGN

> Status: **DESIGN / SCOPE**. A live, spatial visualization of JARVIS's information processing —
> "watch her think." Grounded-first: **every pulse is a real emitted event; every node a real
> subsystem; every edge a real pub/sub wire.** Nothing animates that isn't actually happening.
> Built on the architecture manifest (`subsystem_registry.json`) + the existing event stream.

## The vision (operator)
*"A visualization dashboard to visually see information processing — how it's processing and where it's
going."* The inspiration: the layered, animated neural-net flow aesthetic (nodes + pulsing weighted
connections). The goal is two things at once: **legibility** (the cure for "every AI keeps getting this
architecture wrong" — make the real flow *visible*) and the **north star** ("watch the inner life happen").

## The one rule (anti-theater — the make-or-break)
This project's discipline is "verify every metric under the hood, not re-skin"; "panels move, the felt
companion doesn't." A gorgeous animation that doesn't reflect real flow is exactly the theater we fight.
So the iron rule: **if a pulse can't be traced to a real `bus.emit(...)` event, it does not render.**
Idle cognition → quiet map. Shadow/dormant subsystems → visibly dimmed (you SEE what isn't live).

## What ALREADY EXISTS (reuse — do NOT rebuild)
| Component | Location | Role in the viz |
|---|---|---|
| **EventBus** | `consciousness/events.py:521` (`emit`/`on`/`off`) | the real flow backbone — every cognitive moment passes through it |
| **~50 event constants** | `consciousness/events.py` | the signal vocabulary: `kernel:thought`, `memory:write`, `perception:*`, `conversation:*`, `consciousness:*`, `output:release_blocked`, … |
| **Live SSE stream** | `/api/events/stream` (`dashboard/app.py:1519`) + `/ws` (`:4636`) | already streams `{seq, …}` events to the frontend; built for a "Flow Map". The proven, grounded data path. |
| **Event ring** | `_EventStream._STREAM_TYPES` (`app.py:136`) | currently curates **8** types (KERNEL_THOUGHT, MEMORY_WRITE, MODE_CHANGE, CONSCIOUSNESS_ANALYSIS, CONVERSATION_RESPONSE, KERNEL_ERROR, AUTONOMY_L3_ACTIVATION_DENIED, PERCEPTION_BARGE_IN) — expand to the flow-map set |
| **Existing FlowMap** | `dashboard/static/v2/cockpit.html` ("Live Cognition Cockpit", `flow-rail`/`flow-ledger`) | a scrolling event **ledger** today — we upgrade its *visual* to a spatial topology; reuse its SSE wiring |
| **Subsystem registry** | `subsystem_registry.json` (built 2026-06-27) | the **nodes**: 49 subsystems + 15-layer integrity stack, each with status/authority/area/wired_to_experience |
| **Graph + anim libs** | sigma.js + canvas + requestAnimationFrame (already in v2) | rendering; no new heavy deps required for 2D |

**The gap is small and visual:** topology + spatial rendering + event→edge mapping on top of an event
stream that already flows. This is the natural next layer on the manifest, not a from-scratch build.

## The data model (the grounding)
**Nodes** = `subsystem_registry.json` (49 subsystems + 15 integrity layers), grouped by `area`. Visual
encoding from real fields: `status` (shipped=bright / shadow=dim / dormant=grey / gated=outlined /
signal-failure=red-x), `authority` (live/advisory/none → glow intensity), `wired_to_experience`.

**Edges** = the **real pub/sub wiring**, extracted from code, NOT hand-drawn:
- source(event) = the file(s) that `bus.emit(EVENT, …)`
- sink(event)   = the file(s) that `bus.on(EVENT, …)`
- an edge `A → B` exists iff A emits an event B listens to.
This is auto-derivable (a small extractor, like the manifest) → an `edge_map.json` → CI-auditable so it
can't drift. Live events from the SSE stream then *light up the edges that actually fired*.

**Pulse** = on each SSE event, animate a packet along its edge(s), colored by domain
(`perception:*` cyan, `memory:*` amber, `consciousness:*` violet, `output:release_blocked` red, …).

## The views (purposeful lenses — NOT one 49-node hairball)
| View | What it shows | Powered by (real events) | Priority |
|---|---|---|---|
| **A · Cognitive Pipeline** *(hero / tracer bullet)* | one conversation turn flowing input→output: `perception:transcription` → `conversation:user_message` → route → reasoning → **integrity stack L0..L12** → `conversation:response` / `output:release_blocked` | the turn's real event sequence | **build first** |
| **B · Living Topology** | all 49 subsystems + 15-stack as an area-grouped graph, pulsing with live events, status-dimmed | full SSE stream | next |
| **C · Integrity Gauntlet** | every outgoing response running the L0→L12 layers; which gate fires / passes / **blocks** | `output:*`, capability-gate, L12 intention events | deeper |
| **D · Belief Field** | live belief graph (≈258 beliefs), evidence/contradiction edges, propagation pulses | `belief_graph propagation`, contradiction events | later (bigger) |

## Tech choice
- **2D first** (sigma.js / canvas — already in the stack): fastest path to a *grounded* tracer; proves the
  data model before any cinematics. View A + B in 2D.
- **3D/WebGL (Three.js)** for the cinematic "screenshot" aesthetic on View B once the flow is proven real.
  Aesthetic is the *second* win, not the first.
- **Data**: consume the existing `/api/events/stream`; expand `_STREAM_TYPES` to the flow-map set (verify
  each is actually emitted — anti-theater); serve `subsystem_registry.json` + `edge_map.json` to the frontend.

## Build plan (grounded-first, phased)
- **Phase 0 — Topology + edge map (backend, no UI).** Auto-extract `edge_map.json` from `emit`/`on` sites;
  expose registry + edge-map via an endpoint; expand the SSE `_STREAM_TYPES` to the flow set (confirm each
  event truly fires). Deliverable: a provable node+edge graph wired to the live stream.
- **Phase 1 — View A tracer bullet.** One v2 page: load topology, subscribe to SSE, animate a *real*
  conversation turn input→stack→output. **Gate: every pulse traces to a real event.** This is the
  "is it grounded?" checkpoint before any polish.
- **Phase 2 — View B Living Topology** (2D → then WebGL cinematic), area-grouped, status-encoded.
- **Phase 3 — Views C/D** (Integrity Gauntlet, Belief Field).
- **Phase 4 — Feed the topology+flow into OSV / self-view** — the same map becomes part of JARVIS's own
  self-knowledge. The legibility loop closes (she can show you how she thinks).

## Anti-theater guarantees
- Every edge pulse ↔ a real `event_type` in the SSE stream. No decorative pulses.
- Nodes show TRUE registry status — shadow/dormant/gated are visibly *not* glowing. You see the real shape.
- Hover provenance: any node/edge → the real event(s) + `home_files` backing it (same honesty discipline).
- Idle = quiet. No synthetic filler animation when cognition is quiet.
- `edge_map.json` CI-audited against the code (emit/on sites) so the picture can't drift from reality.

## What this is NOT
- Not a re-skin of the existing stat panels (cognition.html already covers stats) — this is *flow*.
- Not a decorative particle field — every element is provenance-backed.
- Not a replacement for the cockpit ledger — it's the spatial upgrade of the same grounded stream.

## Open decisions for the operator
1. **Aesthetic order**: 2D-grounded-first then WebGL cinematic (recommended) vs. 3D from the start.
2. **Home**: new dedicated v2 page (e.g. `flow.html` / `mind.html`) vs. upgrade `cockpit.html`'s FlowMap in place.
3. **First build = View A tracer** (recommended) — agree, or start with View B topology?
