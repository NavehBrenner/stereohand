"""Quick sanity dump of a saved stereo calibration (baseline, rms, image size)."""

import json
import math
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "stereo_calib.json"
data = json.loads(Path(path).read_text())
translation = data["T"]
baseline_cm = math.sqrt(sum(x * x for x in translation)) * 100
print("image_size:", data["image_size"])
print("rms (px):", data.get("rms"))
print("baseline (cm):", round(baseline_cm, 2))
print("T (m):", [round(x, 4) for x in translation])
print("P2[0][3] (=-fx*baseline, px):", round(data["P2"][0][3], 2))
