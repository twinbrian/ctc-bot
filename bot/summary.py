"""Print the bot's round and season results from its logs.

    /opt/anaconda3/bin/python3 -I bot/summary.py
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for f in sorted(glob.glob(str(ROOT / "logs" / "run-*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["event"] in ("ROUND RESULT", "SEASON RESULT", "DISCONNECTED", "JOINED LATE: outage history unknown -> no in-storm trading this season; final-report trade limited to outage-free symbols"):
            print(Path(f).name, r["event"], {k: v for k, v in r.items() if k not in ("ts", "event")})
