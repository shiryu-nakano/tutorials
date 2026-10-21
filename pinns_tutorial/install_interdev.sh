#!/usr/bin/env bash
# add_devinterp.sh
#
# Installs devinterp into the existing uv venv and reports its version.
# Run from repo root with the venv active (or it'll use uv's resolution):
#   bash add_devinterp.sh

set -euo pipefail

echo "== installing devinterp =="
uv pip install devinterp

echo ""
echo "== version / API probe =="
python - <<'PY'
import devinterp
print("devinterp version:", getattr(devinterp, "__version__", "unknown"))

# probe which sampler API is present (it shifted across versions)
try:
    from devinterp.slt.sampler import sample, LLCEstimator
    from devinterp.optim import SGLD
    print("sampler API: devinterp.slt.sampler.sample + LLCEstimator  (available)")
except Exception as e:
    print("sampler API import failed:", e)

try:
    from devinterp.utils import default_nbeta
    print("default_nbeta: available in devinterp.utils")
except Exception as e:
    print("default_nbeta import failed:", e)
PY