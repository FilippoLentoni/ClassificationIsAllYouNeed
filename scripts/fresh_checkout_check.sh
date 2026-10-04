#!/usr/bin/env bash
# Rebuild the results and the paper from the files git would track, without the response
# cache, the ledger or the request log. Proves that metrics, thresholding, importance
# sampling, the model comparison, the escalation ensemble and the paper build need no Jev request
# and no SageMaker job.
#
# Usage: scripts/fresh_checkout_check.sh   (from the project root)
set -euo pipefail
root=$(pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# every file that is tracked or would be tracked (nothing ignored: no cache/, no .env, no results/mock)
git ls-files --cached --others --exclude-standard -z | (cd "$root" && xargs -0 cp --parents -t "$tmp")
# downloaded public datasets only, so the check does not depend on the network; never the response cache
mkdir -p "$tmp/cache"
for d in datasets openml; do [ -d "$root/cache/$d" ] && cp -r "$root/cache/$d" "$tmp/cache/$d"; done
test ! -e "$tmp/cache/responses.sqlite" && test ! -e "$tmp/cache/exchanges" && test ! -e "$tmp/.env"

export CIAYN_ROOT="$tmp" CIAYN_SKIP_BACKUP=1
unset TYPESAFE_API_KEY AWS_PROFILE || true
before=$(sha256sum "$tmp/results/live/baseline_attempts.json" | cut -d' ' -f1)
for stage in metrics thresholding importance comparison escalation; do
  "$root/.venv/bin/python" -m ciayn "$stage" --mode live 2>&1 | grep -v " INFO " | tail -2
done
"$root/.venv/bin/python" -m ciayn paper --mode live 2>&1 | grep -v " INFO " | tail -1
"$root/.venv/bin/python" -m ciayn verify-paper --mode live 2>&1 | grep -v " INFO " | tail -1

# no Jev request can have been made: there is no key, and no ledger or request log was created
test ! -e "$tmp/cache/responses.sqlite" || { echo "FAIL: a response cache or ledger was created"; exit 1; }
test ! -e "$tmp/cache/exchanges" || { echo "FAIL: a request log was created"; exit 1; }
# no SageMaker job can have been started: the record of pipeline executions is unchanged
test "$before" = "$(sha256sum "$tmp/results/live/baseline_attempts.json" | cut -d' ' -f1)" || { echo "FAIL: baseline attempts changed"; exit 1; }
# the rebuilt numbers are the ones in the working copy
cmp "$tmp/paper/generated/macros.tex" "$root/paper/generated/macros.tex" || { echo "FAIL: regenerated numbers differ"; exit 1; }
echo "fresh checkout check passed: $(grep -c newcommand "$tmp/paper/generated/macros.tex") numbers rebuilt identically, $(pdfinfo "$tmp/paper/main.pdf" | awk '/Pages/{print $2}') page PDF built, 0 Jev requests, 0 SageMaker jobs"
