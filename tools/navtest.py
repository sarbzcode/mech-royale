"""Navigation test: drive the bot's Navigator with the simulator's movement physics.

    python tools/navtest.py [--trials N] [--seed S]

Random start (main walkable area) -> random goal (any free cell, including enclosed areas that need
a jet). Reports arrival rate, time vs straight-line time, and failures (with coordinates).
"""
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import replay  # noqa: E402
import sim  # noqa: E402
from navigation import Navigator  # noqa: E402
from worldmap import WorldMap  # noqa: E402


class _NullBot:
    def act(self, obs, deadline):
        return {}


def main():
    args = sys.argv[1:]
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else 40
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 3
    hello, _ = replay.load_stream()
    cfg = hello["config"]
    wm = WorldMap(os.path.join(ROOT, "bot", "arena"), cfg["mech"])
    s = sim.Sim(wm, hello, seed, lambda h: _NullBot())
    rng = random.Random(seed)
    big = max(range(len(wm.comp_size)), key=lambda k: wm.comp_size[k])
    starts = [c for c in range(0, wm.N, 5) if wm.comp[c] == big and not wm.near1[c]]
    goals_any = [c for c in range(0, wm.N, 5) if not wm.solid[c] and not wm.near1[c]]
    goals_other = [c for c in goals_any if wm.comp[c] != big and wm.comp_size[wm.comp[c]] > 200]
    ok = 0
    ratios = []
    fails = []
    worst_ms = 0.0
    for k in range(trials):
        sc = rng.choice(starts)
        gc = rng.choice(goals_other) if k % 5 == 4 and goals_other else rng.choice(starts)
        sx, sy = wm.center(sc)
        gx, gy = wm.center(gc)
        if math.hypot(gx - sx, gy - sy) < 60:
            continue
        m = sim.M(0, sx, sy, wm.floor_at(sx, sy), cfg, "bot")
        nav = Navigator(wm, k)
        t = 0.0
        arrived = None
        tlimit = 30.0 + math.hypot(gx - sx, gy - sy) / 11.7 * 3.0
        tick = 0
        while t < tlimit:
            if tick % 2 == 0:
                nav.set_goal(gx, gy, t)
                t0 = time.perf_counter()
                mx, my, jump, boost = nav.steer(t, m.x, m.y, m.z, m.grounded, m.energy, t0 + 0.012)
                worst_ms = max(worst_ms, (time.perf_counter() - t0) * 1000)
                m.cmd = {"move": [mx, my], "aim": [m.yaw, 0.0], "boost": boost, "jump": jump}
            s.t = t
            s.step_mech(m)
            m.cmd["jump"] = False
            t += sim.DT
            tick += 1
            if math.hypot(m.x - gx, m.y - gy) < 3.0:
                arrived = t
                break
        straight = math.hypot(gx - sx, gy - sy) / 11.7
        kind = "other-area" if wm.comp[gc] != big else "main"
        if arrived is not None:
            ok += 1
            ratios.append(arrived / straight)
        else:
            fails.append((kind, (round(sx), round(sy)), (round(gx), round(gy)), (round(m.x), round(m.y)), nav.mode))
    n = ok + len(fails)
    ratios.sort()
    print(f"trials {n}  arrived {ok} ({100.0 * ok / max(n, 1):.0f}%)  time/straight-line median "
          f"{ratios[len(ratios) // 2]:.2f} p90 {ratios[int(len(ratios) * 0.9)]:.2f}  worst steer() {worst_ms:.1f} ms")
    for f in fails[:12]:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
