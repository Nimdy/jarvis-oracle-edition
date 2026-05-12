# Golden Words Operator Contract

> **Primary reading surface:** The dashboard `/learning` page now consolidates
> the Golden Words operator contract for open-source release. This Markdown file
> remains source/reference material and should not be deleted until a later
> archive pass.
>
Golden Words provide a deterministic operator control plane on top of normal conversation routing.

## Baseline Contract (No Verb Hacking)

Golden Commands are the **baseline contract** for known-good operator control.

- Golden coverage is deterministic and explicitly testable.
- Natural-language routing (keywords/regex/tool-router) is a **supporting lane**, not the source of truth.
- Do not add phrase-specific routing just to make a single utterance pass.
- A routing-pattern change is justified only when misses repeat across multiple phrasings/sessions and logs show stable intent.
- Natural-language misses should primarily become learning evidence for shadow evaluation and NN promotion over time.

This keeps baseline reliability high while preserving honest generalization growth.

## Trigger Grammar

Use:

`Jarvis, GOLDEN COMMAND <EXACT COMMAND>`

Prefix match rules:
- Prefix is case-insensitive.
- `Jarvis` is required by default.
- In wake-addressed/follow-up voice handling, a guarded bare prefix is also accepted: `GOLDEN COMMAND <EXACT COMMAND>`.
- Repeated whitespace is collapsed.
- Boundary punctuation is ignored around the prefix/body boundary.
- Inner punctuation separators in the command body (for example `memory, status`) are normalized to spaces for STT robustness.

Examples that normalize to the same command:
- `Jarvis, GOLDEN COMMAND STATUS`
- `jarvis golden command status`
- `Jarvis... GOLDEN COMMAND status!!!`

## Exact-Match Rules

After prefix normalization, command bodies are exact match only:
- No fuzzy matching.
- No synonym expansion.
- No typo tolerance.
- No silent fallback into natural-language execution for malformed Golden input.

If the prefix is present but body is invalid, Jarvis returns deterministic guidance with accepted commands.

## Authority Classes

Every Golden command carries a fixed authority class:
- `informational`: read/status/introspection requests
- `control`: operational control actions (pause/resume/cancel)
- `privileged`: higher-risk but non-destructive operator actions
- `destructive`: code-impact or side-effect-heavy actions

Golden authorization grants entry to a subsystem path, but does **not** bypass subsystem guards, freezes, approvals, or validation.

## Lifecycle Outcomes

Golden command state is explicit and auditable:
- `none`: no Golden prefix
- `invalid`: Golden prefix detected, body invalid
- `blocked`: command recognized but policy/mode/guard blocked execution
- `unauthorized`: command recognized but explicit authorization/confirmation missing
- `executed`: command recognized and admitted to the target path

## Supported Commands Only

Only the commands below are supported in runtime:

- `Jarvis, GOLDEN COMMAND STATUS`
- `Jarvis, GOLDEN COMMAND INTROSPECTION STATUS`
- `Jarvis, GOLDEN COMMAND MEMORY STATUS`
- `Jarvis, GOLDEN COMMAND VISION STATUS`
- `Jarvis, GOLDEN COMMAND CODEBASE STATUS`
- `Jarvis, GOLDEN COMMAND GOAL STATUS`
- `Jarvis, GOLDEN COMMAND RESEARCH WEB`
- `Jarvis, GOLDEN COMMAND RESEARCH ACADEMIC`
- `Jarvis, GOLDEN COMMAND VERIFY CODEBASE`
- `Jarvis, GOLDEN COMMAND GOAL PAUSE`
- `Jarvis, GOLDEN COMMAND GOAL RESUME`
- `Jarvis, GOLDEN COMMAND CANCEL CURRENT TASK`
- `Jarvis, GOLDEN COMMAND SELF IMPROVE DRY RUN`
- `Jarvis, GOLDEN COMMAND SELF IMPROVE EXECUTE CONFIRM`
- `Jarvis, GOLDEN COMMAND ACQUIRE <intent text>`
- `Jarvis, GOLDEN COMMAND ACQUISITION STATUS`

## Capability Acquisition

`ACQUIRE` is Golden-gated. The capability acquisition pipeline (plugin creation, skill creation, core upgrades, specialist NNs, hardware integration) is never triggered by natural language routing — only by explicit Golden commands or the dashboard API.

- `ACQUIRE <intent text>` creates a new acquisition job. The intent text after `ACQUIRE` describes what to build/learn. The classifier routes it to the appropriate outcome class and lanes.
- `ACQUISITION STATUS` returns a summary of active, completed, and failed jobs plus pending approvals.
- Non-Golden requests that sound like acquisition ("build me a tool", "create a plugin") route to general LLM chat. This is intentional — acquisition has side effects (writes code, creates plugins, modifies the brain) and must be explicitly authorized.

Example:
```
Jarvis, GOLDEN COMMAND ACQUIRE create a tool that tells me a random joke
```

## Self-Improve Safety

`SELF IMPROVE` is Golden-gated:
- Non-Golden self-improve requests are rejected.
- Use `SELF IMPROVE EXECUTE CONFIRM` for explicit destructive authorization.
- Dry-run remains available via `SELF IMPROVE DRY RUN`.

Subsystem-level freeze/approval behavior still applies.

## Observability

Golden metadata is emitted to:
- conversation response event metadata
- attribution ledger (`response_complete` data)
- flight recorder episodes
- dashboard snapshot under `golden_commands`
- dashboard Trust tab panel: **Golden Commands**

Each record includes `trace_id`, command metadata, outcome status, block reason (if any), and routed tool.
