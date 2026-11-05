#!/usr/bin/env bash
# Fetch the official RobustTSF code into baselines/robusttsf.
# It is not redistributed here because the upstream repository has no license.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d baselines/robusttsf ] || git clone --depth 1 https://github.com/haochenglouis/RobustTSF baselines/robusttsf
# Point the hardcoded data paths at <repo>/data (or $NA_TSF_DATA) and define the
# traffic path, which is commented out upstream.
python - <<'PY'
import re
p = "baselines/robusttsf/dataset.py"
s = open(p).read()
if "NA_TSF_DATA" not in s:
    s = re.sub(r"^\s*#?\s*traffic_path\s*=.*$\n?", "", s, flags=re.M)
    s = re.sub(r"^\s*#?\s*ele_path\s*=.*$", (
        "DATA_DIR = os.environ.get('NA_TSF_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))\n"
        "ele_path = os.path.join(DATA_DIR, 'electricity.csv')\n"
        "traffic_path = os.path.join(DATA_DIR, 'traffic.csv')"), s, count=1, flags=re.M)
    open(p, "w").write(s)
print("RobustTSF baselines ready in baselines/robusttsf")
PY
