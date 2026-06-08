#!/usr/bin/env bash
# Surgical language-training reset: strip ONLY the negative_example pollution
# (capability-gate clamps from before the gate fix) from the language corpus,
# reset the promotion governor to shadow, and drop the distilled artifacts that
# were trained on the polluted corpus. KEEPS the legitimate examples and ALL
# real companion memory (memories.json / vector_memory.db / episodes / identity).
#
# REVERSIBLE: backs up every touched file to ~/.jarvis/_lang_reset_backup_<ts>/
# first. Run with the brain STOPPED (it holds these files open); the operator
# restarts afterward. Dry-run by default — pass --apply to actually change files.
set -euo pipefail

J="$HOME/.jarvis"
CORPUS="$J/language_corpus/examples.jsonl"
QUALITY="$J/language_corpus/quality_events.jsonl"
PHASEC="$J/language_corpus/phase_c"
PROMO="$J/language_promotion.json"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

ts="$(date +%Y%m%d_%H%M%S)"
BK="$J/_lang_reset_backup_$ts"

echo "=== Language-training surgical reset (strip negatives only) ==="
echo "mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo

# ---- safety: refuse to run while the brain is up (it would re-write files) ----
if pgrep -f 'python.*main.py' >/dev/null 2>&1; then
  echo "REFUSING: a 'python main.py' (brain) process is running."
  echo "Stop the brain first, then re-run. (Files are held open by the live process.)"
  exit 1
fi

# ---- report current state ----
if [[ -f "$CORPUS" ]]; then
  total="$(wc -l < "$CORPUS")"
  negs="$(grep -c '"response_class": "negative_example"' "$CORPUS" || true)"
  # fall back to looser match if the exact key spacing differs
  [[ "$negs" == "0" ]] && negs="$(grep -c 'negative_example' "$CORPUS" || true)"
  keep=$(( total - negs ))
  echo "corpus: $CORPUS"
  echo "  total examples : $total"
  echo "  negatives      : $negs  (will be removed)"
  echo "  legitimate     : $keep  (will be KEPT)"
else
  echo "corpus not found at $CORPUS — nothing to strip"; total=0; negs=0; keep=0
fi
echo
echo "will also: reset $PROMO -> empty (governor re-learns from shadow)"
echo "           clear $QUALITY (telemetry log)"
echo "           remove $PHASEC/* (distilled artifacts trained on polluted data)"
echo "KEEPS untouched: memories.json, vector_memory.db, episodes.json, identity*.json, face_profiles.json"
echo

if [[ $APPLY -eq 0 ]]; then
  echo "DRY-RUN complete. Re-run with --apply to make changes."
  exit 0
fi

# ---- backup everything we touch ----
mkdir -p "$BK"
echo "backup -> $BK"
[[ -f "$CORPUS" ]]  && cp -a "$CORPUS"  "$BK/examples.jsonl.bak"
[[ -f "$QUALITY" ]] && cp -a "$QUALITY" "$BK/quality_events.jsonl.bak"
[[ -f "$PROMO" ]]   && cp -a "$PROMO"   "$BK/language_promotion.json.bak"
[[ -d "$PHASEC" ]]  && cp -a "$PHASEC"  "$BK/phase_c.bak"
echo "  backed up."
echo

# ---- strip negatives (atomic: filter to temp, then move) ----
if [[ -f "$CORPUS" ]]; then
  tmp="$CORPUS.stripped.$$"
  grep -v 'negative_example' "$CORPUS" > "$tmp" || true
  mv "$tmp" "$CORPUS"
  new_total="$(wc -l < "$CORPUS")"
  echo "corpus stripped: $total -> $new_total lines (removed $(( total - new_total )))"
fi

# ---- reset promotion governor (empty object => all classes start at shadow) ----
echo '{}' > "$PROMO"
echo "promotion governor reset to shadow ($PROMO)"

# ---- clear quality telemetry log ----
[[ -f "$QUALITY" ]] && : > "$QUALITY" && echo "quality log cleared ($QUALITY)"

# ---- drop distilled artifacts trained on polluted corpus (they rebuild) ----
if [[ -d "$PHASEC" ]]; then
  find "$PHASEC" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
  echo "phase_c distilled artifacts removed (will retrain from clean corpus)"
fi

echo
echo "=== DONE. Restart the brain. Backup retained at: $BK ==="
echo "To undo: stop brain, restore files from the backup dir, restart."
