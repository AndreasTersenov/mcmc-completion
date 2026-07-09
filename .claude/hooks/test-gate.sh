#!/bin/bash
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 0
compgen -G "tests/test_*.py" > /dev/null || exit 0   # no tests yet: don't block scaffolding
PY=python
[ -x "$HOME/wl-challenge-env/bin/python" ] && PY="$HOME/wl-challenge-env/bin/python"
out=$(OMP_NUM_THREADS=8 "$PY" -m pytest tests/ -q --tb=short 2>&1); code=$?
if [ $code -ne 0 ]; then
  echo "$out" | tail -30 >&2
  echo "STOP-GATE: tests failing. Fix before finishing, or mark xfail with a written justification in log/." >&2
  exit 2
fi
exit 0
