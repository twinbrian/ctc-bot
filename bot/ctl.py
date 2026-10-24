"""Send one command to the running bot: writes bot/control.json, which the bot reads (and deletes) within 2 s.

    python3 bot/ctl.py resume GEN REBLD      after an announcement: recalibrate + trade them again ("resume all": all)
    python3 bot/ctl.py announce GEN          an announcement the bot missed: handle it as if it had arrived
    python3 bot/ctl.py fair GEN -3           optional: set GEN's valuation shift yourself (-$3/share; 0 clears)
    python3 bot/ctl.py pause GEN             freeze GEN exactly like an announcement (its ETF too; position kept)
    python3 bot/ctl.py halt                  the brake: no automatic trading at all
    python3 bot/ctl.py go                    release the brake
    python3 bot/ctl.py flatten               close everything, then stay halted
    python3 bot/ctl.py sleeve 0              formation sleeve off (sleeve 0.2: back on)

It never overwrites a command the bot has not picked up yet, and it reports whether the bot picked this one up.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CTL = Path(__file__).resolve().parent / "control.json"


def build(argv: list[str]) -> dict:
    if not argv:
        raise ValueError("no command")
    c, args = argv[0].lower(), argv[1:]
    if c in ("resume", "pause", "announce", "close"):
        if not args:
            raise ValueError(f"{c}: name at least one ticker" + (" (or: resume all)" if c == "resume" else ""))
        if c == "resume" and [a.lower() for a in args] == ["all"]:
            return {"resume": "all"}
        return {c: [a.upper() for a in args]}
    if c == "fair":
        if len(args) != 2:
            raise ValueError("fair TICKER DOLLARS, e.g. fair GEN -3")
        return {"fair": {args[0].upper(): float(args[1])}}
    if c == "halt":
        return {"halt": True}
    if c in ("go", "release", "unhalt"):
        return {"halt": False}
    if c == "flatten":
        return {"flatten": True}
    if c == "sleeve":
        return {"sleeve": float(args[0]) if args else 0.0}
    raise ValueError(f"unknown command {c!r}")


def send(cmd: dict, path: Path = CTL, wait_free: float = 5.0, wait_taken: float = 10.0) -> bool:
    t = time.time()
    while path.exists():                               # a command the bot has not read yet: never overwrite it
        if time.time() - t > wait_free:
            print(f"{path.name} is still there after {wait_free:.0f} s: is the bot running? Command NOT sent.")
            return False
        time.sleep(0.2)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cmd))
    os.replace(tmp, path)                              # atomic: the bot never reads half a file
    t = time.time()
    while path.exists():
        if time.time() - t > wait_taken:
            print(f"written: {json.dumps(cmd)} -- not picked up within {wait_taken:.0f} s: is the bot running?")
            return False
        time.sleep(0.25)
    print(f"sent: {json.dumps(cmd)} -- the bot picked it up (its log shows a CONTROL line, or 'control command error')")
    return True


if __name__ == "__main__":
    try:
        command = build(sys.argv[1:])
    except ValueError as exc:
        sys.exit(f"{exc}\n\n{__doc__}")
    sys.exit(0 if send(command) else 1)
