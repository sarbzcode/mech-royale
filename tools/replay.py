"""Replay harness: time a bot on recorded and fuzzed observations.

    python tools/replay.py inproc [--fuzz N]          # import bot/main.py, time act() per tick
    python tools/replay.py proc [--fuzz N] -- <cmd>   # run a bot process over stdin/stdout, time answers

The recorded stream is the kit's check/stream.jsonl.gz (a hello + 300 real observations).
--fuzz N appends N mutated observations covering what the recording lacks: late zone phases,
enemies close by (charging lasers, low HP), incoming bullets and shells, offers, pickups at our
feet, supply drops, damage events, odd positions (inside buildings, outside the map).
"""
import copy
import gzip
import json
import math
import os
import random
import subprocess
import sys
import threading
import queue
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAM = os.path.join(ROOT, "kit", "mechfight-kit", "check", "stream.jsonl.gz")


def load_stream(path=STREAM):
    with gzip.open(path, "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    return msgs[0], [m for m in msgs if m.get("type") == "obs"]


def fuzz(base_obs, n, seed=7):
    """Yield n mutated observations (lazily: holding thousands in memory distorts timing)."""
    rng = random.Random(seed)
    phases = [(560, 380, 20), (380, 240, 35), (240, 140, 60), (140, 70, 100), (70, 25, 160), (25, 0, 260)]
    tick = base_obs[-1]["tick"] + 2
    for k in range(n):
        o = copy.deepcopy(rng.choice(base_obs))
        tick += 2
        o["tick"] = tick
        t = rng.uniform(0, 470)
        o["t"] = round(t, 2)
        me = o["self"]
        x, y = rng.uniform(-500, 540), rng.uniform(-320, 380)
        z = rng.choice([-5.0, 0.0, 2.0, 15.0, 30.0])
        me["pos"] = [x, y, z]
        me["hp"] = rng.choice([1400, 900, 300, 40])
        me["energy"] = rng.choice([100, 50, 3, 0])
        me["grounded"] = rng.random() < 0.8
        me["cubes"] = rng.choice([0, 1, 4, 12])
        r0, r1, dps = rng.choice(phases)
        c0 = [rng.uniform(-200, 200), rng.uniform(-150, 150)]
        a = rng.uniform(0, 6.28)
        dd = rng.uniform(0, max(0.0, r0 - r1))
        c1 = [c0[0] + dd * math.cos(a), c0[1] + dd * math.sin(a)]
        s0 = t + rng.uniform(-30, 30)
        o["zone"] = {"center": c0, "radius": r0, "next_center": c1, "next_radius": r1,
                     "shrink_start": s0, "shrink_end": s0 + rng.uniform(10, 50), "dps": dps}
        me["in_zone"] = math.hypot(x - c0[0], y - c0[1]) < r0
        kinds = ["gun", "mortar", "laser", "lightning"]
        w1 = rng.choice([None, "mortar", "laser", "lightning"])
        if w1:
            me["weapons"][1] = {"kind": w1, "cooldown": rng.choice([0, 0.5]), "charge": rng.choice([0, 0.3, 0.8]),
                                "stats": {"damage": 100, "range": 45 if w1 == "lightning" else 280, "speed": 72,
                                          "gravity": 22, "charge": 0.75, "cone_deg": 30, "energy": 3}}
        else:
            me["weapons"][1] = None
        me["weapons"][0]["ammo"] = rng.choice([0, 5, 12])
        me["weapons"][0]["reload"] = rng.choice([0, 0, 1.2])
        mechs = []
        for i in range(rng.randint(0, 15)):
            ex, ey = x + rng.uniform(-150, 150), y + rng.uniform(-150, 150)
            mechs.append({"id": i if i != me["id"] else 15, "kind": "mech", "pos": [ex, ey, rng.uniform(-5, 10)],
                          "vel": [rng.uniform(-19, 19), rng.uniform(-19, 19), rng.uniform(-5, 5)],
                          "aim": [math.atan2(y - ey, x - ex) + rng.uniform(-0.1, 0.1), 0],
                          "hp": rng.choice([1400, 600, 100]), "max_hp": 1400, "cubes": rng.choice([0, 3]),
                          "weapons": rng.sample(kinds, rng.randint(1, 2)), "charging": rng.random() < 0.3})
        for i in range(rng.randint(0, 4)):
            mechs.append({"id": 100 + i, "kind": "npc", "pos": [x + rng.uniform(-60, 60), y + rng.uniform(-60, 60), 0],
                          "vel": [1, 0, 0], "aim": [0, 0], "hp": 350, "max_hp": 350})
        o["mechs"] = mechs
        o["projectiles"] = []
        for i in range(rng.randint(0, 30)):
            px, py = x + rng.uniform(-80, 80), y + rng.uniform(-80, 80)
            kind = rng.choice(["gun", "mortar", "npc"])
            sp = 252 if kind == "gun" else (72 if kind == "mortar" else 90)
            a = math.atan2(y - py, x - px)
            o["projectiles"].append({"id": 300 + i, "kind": kind, "pos": [px, py, rng.uniform(0, 30)],
                                     "vel": [sp * math.cos(a), sp * math.sin(a), rng.uniform(-20, 20)]})
        o["pickups"] = [{"id": 8000 + i, "kind": rng.choice(kinds[1:]),
                         "pos": [x + rng.uniform(-5, 5), y + rng.uniform(-5, 5), z]} for i in range(rng.randint(0, 3))]
        o["cubes"] = [{"id": 5000 + i, "pos": [x + rng.uniform(-40, 40), y + rng.uniform(-40, 40), z]} for i in range(rng.randint(0, 3))]
        o["crates"] = [{"id": 1000 + i, "pos": [x + rng.uniform(-60, 60), y + rng.uniform(-60, 60), z], "hp": 300} for i in range(rng.randint(0, 3))]
        o["sounds"] = [{"kind": rng.choice(["gun", "mortar", "explosion", "laser", "lightning", "death", "step", "boost", "drop", "jet"]),
                        "bearing": rng.uniform(-3.14, 3.14), "dist": rng.choice(["near", "mid", "far"]), "age": rng.uniform(0, 0.1)}
                       for _ in range(rng.randint(0, 12))]
        ev = []
        for _ in range(rng.randint(0, 4)):
            ev.append(rng.choice([
                {"type": "damaged", "amount": 45, "weapon": "gun", "bearing": rng.uniform(-3, 3), "by": rng.randint(0, 15), "rear": rng.random() < 0.5},
                {"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None},
                {"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 1.0, "by": 101},
                {"type": "hit", "target": rng.randint(0, 15), "amount": 45, "weapon": "gun"},
                {"type": "kill", "target": rng.randint(0, 15)},
                {"type": "take_failed", "id": 8000, "reason": "range"},
                {"type": "pickup", "cubes": 2},
                {"type": "effect", "effect": "concuss", "by": 3},
            ]))
        o["events"] = ev
        o["drops"] = [{"id": 20000, "pos": [x + rng.uniform(-100, 100), y + rng.uniform(-100, 100), 0], "radius": 8,
                       "lands_in": rng.choice([0, 5, 20]), "capture_s": 4, "progress": 0, "holder": rng.choice([None, 3])}] if rng.random() < 0.3 else []
        o["offer"] = {"id": k, "expires": t + 5, "choices": [
            {"id": c, "name": c, "desc": c} for c in rng.sample(["plating", "armor", "gun_rate", "sensors", "jets", "laser_pierce", "nope"], 3)]} if rng.random() < 0.2 else None
        yield o


def stats(times):
    times = sorted(times)
    n = len(times)
    p = lambda q: times[min(n - 1, int(q * n))]
    return f"n {n}  p50 {p(0.5):.2f}  p90 {p(0.9):.2f}  p99 {p(0.99):.2f}  max {times[-1]:.2f} ms"


def run_inproc(hello, obs, budget=0.016):
    sys.path.insert(0, os.path.join(ROOT, "bot"))
    import main  # noqa
    hello = copy.deepcopy(hello)
    hello["map"]["dir"] = os.path.join(ROOT, "bot", "arena")
    t0 = time.perf_counter()
    bot = main.Bot(hello)
    print(f"init {time.perf_counter() - t0:.2f} s")
    times, errors = [], 0
    worst = []
    import gc
    gc.collect()
    for o in obs:
        t0 = time.perf_counter()
        try:
            a = bot.act(o, t0 + budget)
            json.dumps(a)
            if not isinstance(a.get("move", [0, 0]), list) or len(a.get("aim", [0, 0])) != 2:
                raise ValueError(f"bad action {a}")
        except Exception:
            errors += 1
            if errors <= 3:
                traceback.print_exc()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        worst.append((dt, o["tick"]))
    worst.sort(reverse=True)
    print("act() time:", stats(times), f" errors {errors}")
    print("slowest ticks:", [(round(d, 1), tk) for d, tk in worst[:5]])
    return errors


def run_proc(hello, obs, cmd):
    """Feed observations one at a time (lockstep, like a match) and time each answer."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    lines = queue.Queue()

    def rd():
        for raw in p.stdout:
            lines.put((time.perf_counter(), raw))
        lines.put((time.perf_counter(), None))
    threading.Thread(target=rd, daemon=True).start()
    t0 = time.perf_counter()
    p.stdin.write((json.dumps(hello) + "\n").encode())
    p.stdin.flush()
    at, raw = lines.get(timeout=30)
    print(f"ready in {(at - t0) * 1000:.0f} ms: {raw.strip()[:80]!r}")
    times, late, bad = [], 0, 0
    for o in obs:
        sent = time.perf_counter()
        p.stdin.write((json.dumps(o) + "\n").encode())
        p.stdin.flush()
        while True:
            try:
                at, raw = lines.get(timeout=1.0)
            except queue.Empty:
                late += 1
                break
            if raw is None:
                print("bot exited early")
                return 1
            m = json.loads(raw)
            if m.get("tick") == o["tick"]:
                dt = (at - sent) * 1000
                times.append(dt)
                if dt > 50:
                    late += 1
                break
            bad += 1
    p.stdin.write(b'{"type":"end","place":5,"kills":0}\n')
    p.stdin.flush()
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        p.kill()
        print("bot did not exit after end")
    print("answer time:", stats(times), f" over 50 ms: {late}  stale/wrong lines: {bad}")
    return 0


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "inproc"
    n = 0
    if "--fuzz" in args:
        n = int(args[args.index("--fuzz") + 1])
    hello, obs = load_stream()
    if n:
        import itertools
        obs = itertools.chain(obs, fuzz(obs, n))
    if mode == "proc":
        cmd = args[args.index("--") + 1:]
        return run_proc(hello, obs, cmd)
    return run_inproc(hello, obs)


if __name__ == "__main__":
    sys.exit(main())
