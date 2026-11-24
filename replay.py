"""Replay a log saved by the web test page ("Napló mentése") through the Python detector.

    python replay.py gazescroll-log-XXXX.json [--scroll-zone 0.6 --enter 0.45 ...]

Use it to tune Params offline on real recordings from your phone.
"""
import argparse
import json
from collections import Counter

from gazescroll.reading import Params, ReadingDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--scroll-zone", type=float, default=Params.scroll_zone)
    ap.add_argument("--enter", type=float, default=Params.reading_fwd_ratio)
    ap.add_argument("--stay", type=float, default=Params.reading_stay_ratio)
    ap.add_argument("--sweep-dx", type=float, default=Params.sweep_min_dx)
    a = ap.parse_args()

    d = json.load(open(a.log))
    meta, rows = d["meta"], d["frames"]
    frames = [r for r in rows if "g" in r]
    labels = [r for r in rows if "label" in r]
    valid = [r["validation"] for r in rows if "validation" in r]
    col, h = meta["col"], meta["h"]
    p = Params(line_h=meta["lineH"] / h, scroll_zone=a.scroll_zone,
               reading_fwd_ratio=a.enter, reading_stay_ratio=a.stay, sweep_min_dx=a.sweep_dx)
    det = ReadingDetector(p)
    ev, states, logged = Counter(), Counter(), Counter()
    for fr in frames:
        g = fr["g"]
        gn = None if g is None else ((g[0] - col["l"]) / col["w"], g[1] / h)
        for e in det.push(fr["t"], gn):
            ev[e.kind] += 1
        states[det.state.value] += 1
        for k in fr.get("e", []):
            logged[k] += 1
    dur = frames[-1]["t"] - frames[0]["t"] if frames else 0
    print(f"{len(frames)} frames, {dur:.0f} s")
    print("Python replay events:", dict(ev))
    print("Logged (phone) events:", dict(logged))
    print("State share:", {k: round(v / max(len(frames), 1), 3) for k, v in states.items()})
    if valid:
        v = valid[-1]
        print(f"Validation error: x {v['ex']:.0f} px, y {v['ey']:.0f} px, video {v.get('video')}")
    q = [r["q"] for r in frames if r.get("q") and r["q"].get("iris")]
    if q:
        iris = sorted(x["iris"] for x in q)
        lum = sorted(x["lum"] for x in q if x["lum"])
        print(f"Iris diameter median {iris[len(iris)//2]:.1f} px, brightness median "
              f"{lum[len(lum)//2] if lum else float('nan'):.0f}/255, video {meta.get('video')}")
    if labels:
        print("Manual corrections:", Counter(r["label"] for r in labels))
    if meta.get("profile"):
        print("Learned profile:", meta["profile"])


if __name__ == "__main__":
    main()
