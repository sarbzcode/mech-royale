"""Entry point: survival-first mechfight/1 bot."""
import gc
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mechfight as mf  # noqa: E402
from strategy import Brain  # noqa: E402
from worldmap import WorldMap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_map(hello):
    """The match map: hello.map.dir, then /arena, then the copy bundled in the image."""
    mech = ((hello.get("config") or {}).get("mech")) or {}
    dirs = []
    m = hello.get("map") or {}
    if isinstance(m.get("dir"), str):
        dirs.append(m["dir"])
    dirs += ["/arena", os.path.join(HERE, "arena")]
    for d in dirs:
        try:
            if os.path.isfile(os.path.join(d, "v2", "world.json")):
                t0 = time.perf_counter()
                wm = WorldMap(d, mech)
                mf.log(f"map {d}: {wm.nx}x{wm.ny} in {time.perf_counter() - t0:.2f}s")
                return wm
        except Exception as e:  # try the next copy
            mf.log(f"map {d} failed: {e!r}")
    return WorldMap(flat_map(), mech)


def flat_map():
    """Last resort: a flat, empty map written to /tmp so navigation still works."""
    import json
    d = os.path.join("/tmp" if os.path.isdir("/tmp") else HERE, "flatmap")
    os.makedirs(os.path.join(d, "v2"), exist_ok=True)
    nx, ny = 1070, 724
    world = {"bounds": {"minX": -512.0, "maxX": 555.0, "minY": -333.0, "maxY": 388.0},
             "terrain": {"file": "terrain.i16", "minX": -513.0, "minY": -334.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01},
             "prisms": [], "trees": {"fields": [], "data": []}}
    with open(os.path.join(d, "v2", "world.json"), "w") as f:
        json.dump(world, f)
    with open(os.path.join(d, "v2", "terrain.i16"), "wb") as f:
        f.write(bytes(2 * nx * ny))
    return d


class Bot:
    name = "survivor"

    def __init__(self, hello):
        self.brain = None
        try:
            self.brain = Brain(hello, load_map(hello))
        except Exception as e:
            mf.log(f"brain init failed, running on fallback only: {e!r}")
        gc.collect()
        if hasattr(gc, "freeze"):
            gc.freeze()   # the static map never needs scanning again

    def act(self, obs, deadline):
        if self.brain is None:
            return self.fallback(obs, {})
        return self.brain.act(obs, deadline)

    def fallback(self, obs, last):
        """Cheap safe action if act() is late or fails: head for the next circle, keep aim."""
        fb = dict(last)
        try:
            me = obs.get("self") or {}
            z = obs.get("zone") or {}
            x, y = me["pos"][0], me["pos"][1]
            cx, cy = z["next_center"]
            r = z.get("next_radius", 0)
            d = math.hypot(cx - x, cy - y)
            if not me.get("in_zone", True) or d > r * 0.9:
                fb["move"] = [(cx - x) / max(d, 1e-6), (cy - y) / max(d, 1e-6)]
                fb["boost"] = not me.get("in_zone", True)
        except Exception:
            pass
        return fb


if __name__ == "__main__":
    mf.run(Bot)
