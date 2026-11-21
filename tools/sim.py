"""Mock match simulator on the real map, to test the bot end to end before submitting.

    python tools/sim.py [--matches N] [--seed S] [--workers W] [-v]

One mech is our bot (bot/main.py, run in-process with the same act(obs, deadline) call the real
loop makes); the other 15 are house bots that share what any of them sees and hunt the weakest mech
they know about. Rules follow protocol.md approximately: movement on the 2 m grid (walls, steps,
jets), torso slew, gun / lightning / laser / mortar, vision (range, FOV, near sensors, buildings,
5 s memory), sounds, zone phases with damage, regen, NPC drones, crates, cubes and upgrades, floor
loot, supply drops, rear-hit multiplier, placement by death order (HP at the time limit).

It is an approximation of the real server and of the real house bots: use it to catch crashes,
stuck navigation, zone deaths and gross behaviour problems, and to compare versions roughly.
"""
import copy
import json
import math
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from worldmap import WorldMap, CELL  # noqa: E402

DT = 0.05
TWO_PI = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


class W:
    """Weapon state."""

    def __init__(self, kind, cfg):
        self.kind = kind
        self.st = dict(cfg["weapons"][kind])
        self.cooldown = 0.0
        self.equip = 0.0
        self.ammo = self.st.get("mag", 0)
        self.reload = 0.0
        self.charge = 0.0

    def obs(self):
        o = {"kind": self.kind, "cooldown": round(max(self.cooldown, self.equip), 3), "stats": self.st}
        if self.kind == "gun":
            o["ammo"] = int(self.ammo)
            o["reload"] = round(self.reload, 3)
        if self.kind == "laser":
            o["charge"] = round(self.charge, 3)
        return o


class M:
    """A mech."""

    def __init__(self, mid, x, y, z, cfg, ctrl):
        mech = cfg["mech"]
        self.id = mid
        self.x, self.y, self.z = x, y, z
        self.vx = self.vy = self.vz = 0.0
        self.yaw = random.uniform(-math.pi, math.pi)
        self.pitch = 0.0
        self.cmd = {"move": [0, 0], "aim": [self.yaw, 0.0], "boost": False}
        self.base_hp = mech["hp"]
        self.plating = 1.0
        self.hp = float(self.base_hp)
        self.energy = mech["energy"]
        self.max_energy = mech["energy"]
        self.cubes = 0
        self.upgrades = []
        self.armor = 0.0
        self.regen = mech["regen_per_s"]
        self.regen_delay = mech["regen_delay"]
        self.walk = mech["walk_speed"]
        self.boost_speed = mech["boost_speed"]
        self.weapons = [W("gun", cfg), None]
        self.alive = True
        self.grounded = True
        self.last_combat = -99.0
        self.last_spend = -99.0
        self.events = []
        self.sounds = []
        self.offer_queue = []
        self.offer = None
        self.offer_seq = 0
        self.place = None
        self.ctrl = ctrl
        self.seen = {}
        self.goal = None
        self.stuck_t = 0.0
        self.last_pos = (x, y, 0.0)
        self.dmg_taken = 0.0
        self.zone_dmg = 0.0
        self.death_cause = None

    @property
    def size(self):
        return 1.0 + 0.05 * self.cubes

    @property
    def max_hp(self):
        return self.base_hp * self.plating * (1.0 + 0.04 * min(self.cubes, 10))

    def dmg_mult(self):
        return 1.0 + 0.04 * min(self.cubes, 10)


class Sim:
    def __init__(self, wm, hello, seed, bot_factory, verbose=False, mode="ffa"):
        self.wm = wm
        self.mode = mode          # ffa | squad (house bots only hunt us, shared vision) | omni (squad + always know us)
        self.hello = copy.deepcopy(hello)
        self.cfg = self.hello["config"]
        self.rng = random.Random(seed)
        random.seed(seed)
        self.verbose = verbose
        self.t = 0.0
        self.tick = 0
        self.bullets = []
        self.next_id = 300000
        cfg = self.cfg
        # spawns: free cells in the main component, spread apart, inside 80% of the start circle
        big = max(range(len(wm.comp_size)), key=lambda k: wm.comp_size[k])
        zc = cfg["zone"]["start_center"]
        R0 = cfg["zone"]["start_radius"]
        self.free_cells = [c for c in range(0, wm.N, 3) if wm.comp[c] == big and not wm.near1[c]]
        pts = []
        tries = 0
        while len(pts) < 16 and tries < 20000:
            tries += 1
            c = self.rng.choice(self.free_cells)
            x, y = wm.center(c)
            if math.hypot(x - zc[0], y - zc[1]) > R0 * 0.8:
                continue
            if all(math.hypot(x - a, y - b) > 130 for a, b in pts) or tries > 15000:
                pts.append((x, y))
        self.me_slot = self.rng.randrange(16)
        self.mechs = []
        for i, (x, y) in enumerate(pts):
            self.mechs.append(M(i, x, y, wm.floor_at(x, y), cfg, "bot" if i == self.me_slot else "house"))
        self.me = self.mechs[self.me_slot]
        # npcs
        self.npcs = []
        for k in range(cfg["npc"]["count"]):
            c = self.rng.choice(self.free_cells)
            x, y = wm.center(c)
            self.npcs.append({"id": 100 + k, "x": x, "y": y, "z": wm.floor_at(x, y), "hx": x, "hy": y, "hp": float(cfg["npc"]["hp"]),
                              "alive": True, "cd": 0.0, "target": None, "agg_t": -99, "wx": x, "wy": y, "vx": 0.0, "vy": 0.0})
        self.crates = []
        for k in range(cfg["crates"]["count"]):
            c = self.rng.choice(self.free_cells)
            x, y = wm.center(c)
            self.crates.append({"id": 1000 + k, "x": x, "y": y, "z": wm.floor_at(x, y), "hp": float(cfg["crates"]["hp"]), "alive": True})
        self.cubes = []
        self.pickups = []
        kinds = ["mortar", "laser", "lightning"]
        for k in range(cfg["loot"]["floor_count"]):
            c = self.rng.choice(self.free_cells)
            x, y = wm.center(c)
            self.pickups.append({"id": 8000 + k, "kind": self.rng.choice(kinds), "x": x, "y": y, "z": wm.floor_at(x, y)})
        self.next_pickup = 8100
        self.next_cube = 5000
        # zone schedule
        self.phases = []
        c0, r0, t0 = tuple(zc), R0, 0.0
        for ph in cfg["zone"]["phases"]:
            r1 = ph["radius"]
            a = self.rng.uniform(0, TWO_PI)
            dd = (r0 - r1) * math.sqrt(self.rng.random())
            c1 = (c0[0] + dd * math.cos(a), c0[1] + dd * math.sin(a))
            # keep the circle center on the map
            c1 = (min(max(c1[0], wm.bminx + 40), wm.bmaxx - 40), min(max(c1[1], wm.bminy + 40), wm.bmaxy - 40))
            if math.hypot(c1[0] - c0[0], c1[1] - c0[1]) > r0 - r1:
                c1 = c0
            s0 = t0 + ph["hold_s"]
            s1 = s0 + ph["shrink_s"]
            self.phases.append({"c0": c0, "r0": r0, "c1": c1, "r1": r1, "s0": s0, "s1": s1, "dps": ph["dps"], "t0": t0})
            c0, r0, t0 = c1, r1, s1
        self.drops = []
        dc = cfg.get("drops") or {}
        self.drop_times = list(dc.get("times", []))
        self.drop_seq = 20000
        self.squad_known = set()
        self.squad_t = -1.0
        bot_hello = copy.deepcopy(self.hello)
        bot_hello["slot"] = self.me_slot
        bot_hello["seed"] = self.rng.randrange(1 << 31)
        bot_hello["map"]["dir"] = os.path.join(ROOT, "bot", "arena")
        self.bot = bot_factory(bot_hello)
        self.act_times = []
        self.errors = 0
        self.dead_order = []
        self.log = []

    # ---------------------------------------------------------------- zone
    def phase(self, t):
        for ph in self.phases:
            if t < ph["s1"]:
                return ph
        return self.phases[-1]

    def circle(self, t):
        ph = self.phase(t)
        if t <= ph["s0"]:
            return ph["c0"], ph["r0"], ph
        if t >= ph["s1"]:
            return ph["c1"], ph["r1"], ph
        f = (t - ph["s0"]) / (ph["s1"] - ph["s0"])
        return ((ph["c0"][0] + (ph["c1"][0] - ph["c0"][0]) * f, ph["c0"][1] + (ph["c1"][1] - ph["c0"][1]) * f),
                ph["r0"] + (ph["r1"] - ph["r0"]) * f, ph)

    def in_zone(self, x, y):
        c, r, ph = self.circle(self.t)
        return math.hypot(x - c[0], y - c[1]) < r

    # ---------------------------------------------------------------- geometry
    def surf(self, x, y):
        wm = self.wm
        c = wm.cell(x, y)
        if c < 0:
            return 1e9
        if x < wm.bminx + 2 or x > wm.bmaxx - 2 or y < wm.bminy + 2 or y > wm.bmaxy - 2:
            return 1e9
        return wm.top[c] if wm.solid[c] else wm.floor[c]

    def los(self, a, b):
        return self.wm.los(a[0], a[1], a[2], b[0], b[1], b[2])

    def sees(self, m, x, y, z, h):
        """Can mech m see a thing at (x,y,z) of height h?"""
        dx, dy = x - m.x, y - m.y
        d = math.hypot(dx, dy)
        vis = self.cfg["vision"]
        eye = (m.x, m.y, m.z + 8.0 * m.size)
        if d > vis["near_range"]:
            if d > vis["range"]:
                return False
            if abs(wrap(math.atan2(dy, dx) - m.yaw)) > math.radians(vis["fov_deg"]) / 2:
                return False
        return self.los(eye, (x, y, z + h * 0.5)) or self.los(eye, (x, y, z + h))

    # ---------------------------------------------------------------- damage
    def damage(self, tgt, amount, weapon, src, frm):
        """tgt: mech dict/obj; src: attacker mech or None; frm: (x, y) the hit came from."""
        if isinstance(tgt, M):
            if not tgt.alive:
                return
            rear = False
            if frm is not None:
                b = math.atan2(frm[1] - tgt.y, frm[0] - tgt.x)
                if abs(wrap(b - tgt.yaw)) > math.pi / 2:
                    rear = True
                    amount *= self.cfg["mech"]["rear_mult"]
            if weapon != "zone":
                amount *= (1.0 - tgt.armor)
            tgt.hp -= amount
            tgt.dmg_taken += amount
            if weapon == "zone":
                tgt.zone_dmg += amount
            tgt.last_combat = self.t
            bearing = math.atan2(frm[1] - tgt.y, frm[0] - tgt.x) if frm is not None else None
            by = src.id if isinstance(src, M) else (src["id"] if isinstance(src, dict) else None)
            ev = {"type": "damaged", "amount": round(amount, 1), "weapon": weapon, "bearing": bearing, "by": by}
            if rear:
                ev["rear"] = True
            tgt.events.append(ev)
            if isinstance(src, M):
                src.last_combat = self.t
                src.events.append({"type": "hit", "target": tgt.id, "amount": round(amount, 1), "weapon": weapon})
            if tgt.hp <= 0:
                self.kill(tgt, src, weapon)
        elif isinstance(tgt, dict):
            if not tgt.get("alive", True):
                return
            tgt["hp"] -= amount
            if isinstance(src, M):
                src.last_combat = self.t
                src.events.append({"type": "hit", "target": tgt["id"], "amount": round(amount, 1), "weapon": weapon})
            if tgt["id"] < 1000:      # npc
                tgt["target"] = src if isinstance(src, M) else tgt["target"]
                tgt["agg_t"] = self.t
                if tgt["hp"] <= 0:
                    tgt["alive"] = False
                    for _ in range(self.cfg["npc"]["cubes"]):
                        self.spawn_cube(tgt["x"] + self.rng.uniform(-2, 2), tgt["y"] + self.rng.uniform(-2, 2))
            else:                     # crate
                if tgt["hp"] <= 0:
                    tgt["alive"] = False
                    self.spawn_cube(tgt["x"], tgt["y"])
                    self.sound("crate", tgt["x"], tgt["y"])
                    if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                        self.pickups.append({"id": self.next_pickup, "kind": self.rng.choice(["mortar", "laser", "lightning"]),
                                             "x": tgt["x"] + 1, "y": tgt["y"], "z": tgt["z"]})
                        self.next_pickup += 1

    def kill(self, m, src, weapon):
        m.alive = False
        m.t_death = self.t
        m.death_cause = weapon
        self.dead_order.append((self.tick, m.id))
        for _ in range(m.cubes + self.cfg["cubes"]["death_drop_extra"]):
            self.spawn_cube(m.x + self.rng.uniform(-3, 3), m.y + self.rng.uniform(-3, 3))
        for w in m.weapons:
            if w and w.kind != "gun":
                self.pickups.append({"id": self.next_pickup, "kind": w.kind, "x": m.x, "y": m.y, "z": m.z})
                self.next_pickup += 1
        if isinstance(src, M):
            src.events.append({"type": "kill", "target": m.id})
        self.sound("death", m.x, m.y)

    def spawn_cube(self, x, y):
        self.cubes.append({"id": self.next_cube, "x": x, "y": y, "z": self.wm.floor_at(x, y)})
        self.next_cube += 1

    def give_cubes(self, m, n):
        before = m.max_hp
        old = m.cubes
        m.cubes += n
        m.hp += max(0.0, m.max_hp - before)
        th = self.cfg["cubes"]["upgrade_thresholds"]
        for k in th:
            if old < k <= m.cubes:
                m.offer_queue.append(k)

    def sound(self, kind, x, y):
        rng_ = self.cfg["sound"]["range"].get(kind, 100)
        me = self.me
        if not me.alive:
            return
        d = math.hypot(x - me.x, y - me.y)
        if d < 1.0 or d > rng_:
            return
        b = math.atan2(y - me.y, x - me.x) + self.rng.gauss(0, math.radians(self.cfg["sound"]["bearing_noise_deg"]))
        me.sounds.append({"kind": kind, "bearing": round(wrap(b), 3), "dist": "near" if d < 60 else ("mid" if d < 160 else "far"), "age": 0.05})

    # ---------------------------------------------------------------- shooting
    def muzzle(self, m):
        return m.x, m.y, m.z + 7.0 * m.size

    def fire_weapon(self, m, w):
        cfg = self.cfg
        k = w.kind
        mx, my, mz = self.muzzle(m)
        cp = math.cos(m.pitch)
        dx, dy, dz = cp * math.cos(m.yaw), cp * math.sin(m.yaw), math.sin(m.pitch)
        if k == "gun":
            sp = math.radians(w.st["spread_deg"])
            yaw = m.yaw + self.rng.gauss(0, sp)
            pit = m.pitch + self.rng.gauss(0, sp)
            cp = math.cos(pit)
            v = w.st["speed"]
            self.bullets.append({"id": self.next_id, "kind": "gun", "x": mx + 3 * dx, "y": my + 3 * dy, "z": mz + 3 * dz,
                                 "vx": v * cp * math.cos(yaw), "vy": v * cp * math.sin(yaw), "vz": v * math.sin(pit),
                                 "owner": m, "left": w.st["range"], "dmg": w.st["damage"] * m.dmg_mult(), "ox": m.x, "oy": m.y})
            self.next_id += 1
            self.sound("gun", m.x, m.y)
        elif k == "mortar":
            v = w.st["speed"]
            self.bullets.append({"id": self.next_id, "kind": "mortar", "x": mx + 3 * dx, "y": my + 3 * dy, "z": m.z + 9.5 * m.size + 3 * dz,
                                 "vx": v * dx, "vy": v * dy, "vz": v * dz, "owner": m, "left": 1e9, "ttl": w.st["ttl"],
                                 "dmg": w.st["damage"] * m.dmg_mult(), "ox": m.x, "oy": m.y})
            self.next_id += 1
            self.sound("mortar", m.x, m.y)
        elif k == "lightning":
            m.energy -= w.st["energy"]
            m.last_spend = self.t
            self.sound("lightning", m.x, m.y)
            rng_ = w.st["range"]
            cone = math.radians(w.st["cone_deg"]) / 2
            best = None
            for tgt, tx, ty, tz, h in self.targets(exclude=m):
                d = math.hypot(tx - m.x, ty - m.y)
                if d > rng_ or abs(wrap(math.atan2(ty - m.y, tx - m.x) - m.yaw)) > cone:
                    continue
                if not self.los((mx, my, mz), (tx, ty, tz + h * 0.5)):
                    continue
                if best is None or d < best[0]:
                    best = (d, tgt, tx, ty)
            if best:
                dmg = w.st["damage"] * m.dmg_mult()
                self.damage(best[1], dmg, "lightning", m, (m.x, m.y))
                hit = {id(best[1])}
                px, py = best[2], best[3]
                for _ in range(int(w.st["chain"])):
                    dmg *= w.st["chain_falloff"]
                    nxt = None
                    for tgt, tx, ty, tz, h in self.targets(exclude=m):
                        if id(tgt) in hit:
                            continue
                        d = math.hypot(tx - px, ty - py)
                        if d <= w.st["chain_range"] and (nxt is None or d < nxt[0]):
                            nxt = (d, tgt, tx, ty)
                    if not nxt:
                        break
                    self.damage(nxt[1], dmg, "lightning", m, (px, py))
                    hit.add(id(nxt[1]))
                    px, py = nxt[2], nxt[3]
        elif k == "laser":
            frac = min(1.0, w.charge / w.st["charge"])
            dmg = w.st["damage"] * (w.st["min_frac"] + (1 - w.st["min_frac"]) * frac) * m.dmg_mult()
            self.sound("laser", m.x, m.y)
            x, y, z = mx, my, mz
            for s in range(int(w.st["range"])):
                x += dx
                y += dy
                z += dz
                c = self.wm.cell(x, y)
                if c < 0 or self.wm.top[c] > z:
                    break
                hit = None
                for tgt, tx, ty, tz, h in self.targets(exclude=m, near=(x, y, 6.0)):
                    r = 2.5 * (tgt.size if isinstance(tgt, M) else 0.72)
                    if (x - tx) ** 2 + (y - ty) ** 2 < r * r and tz <= z <= tz + h:
                        hit = tgt
                        break
                if hit is not None:
                    if isinstance(hit, M) and any(ww and ww.kind == "mortar" for ww in hit.weapons):
                        dmg *= 1 + w.st["cookoff"]
                    self.damage(hit, dmg, "laser", m, (m.x, m.y))
                    break

    def targets(self, exclude=None, near=None):
        out = []
        for o in self.mechs:
            if o.alive and o is not exclude:
                if near and (o.x - near[0]) ** 2 + (o.y - near[1]) ** 2 > near[2] ** 2:
                    continue
                out.append((o, o.x, o.y, o.z, 9.0 * o.size))
        for n in self.npcs:
            if n["alive"]:
                if near and (n["x"] - near[0]) ** 2 + (n["y"] - near[1]) ** 2 > near[2] ** 2:
                    continue
                out.append((n, n["x"], n["y"], n["z"], 3.5))
        for c in self.crates:
            if c["alive"]:
                if near and (c["x"] - near[0]) ** 2 + (c["y"] - near[1]) ** 2 > near[2] ** 2:
                    continue
                out.append((c, c["x"], c["y"], c["z"], 2.4))
        return out

    def step_bullets(self):
        wm = self.wm
        keep = []
        for b in self.bullets:
            alive = True
            if b["kind"] == "mortar":
                b["vz"] -= self.cfg["weapons"]["mortar"]["gravity"] * DT
                b["ttl"] -= DT
            sub = 4
            for _ in range(sub):
                b["x"] += b["vx"] * DT / sub
                b["y"] += b["vy"] * DT / sub
                b["z"] += b["vz"] * DT / sub
                b["left"] -= math.sqrt(b["vx"] ** 2 + b["vy"] ** 2 + b["vz"] ** 2) * DT / sub
                c = wm.cell(b["x"], b["y"])
                if c < 0 or b["left"] <= 0 or b.get("ttl", 1) <= 0:
                    alive = False
                    break
                if wm.top[c] > b["z"]:
                    if b["kind"] == "mortar":
                        self.explode(b)
                    alive = False
                    break
                hit = None
                for tgt, tx, ty, tz, h in self.targets(exclude=b["owner"], near=(b["x"], b["y"], 4.0)):
                    r = (2.5 * tgt.size if isinstance(tgt, M) else (1.8 if tgt["id"] < 1000 else 1.4)) + 0.3
                    if (b["x"] - tx) ** 2 + (b["y"] - ty) ** 2 < r * r and tz - 0.3 <= b["z"] <= tz + h + 0.3:
                        hit = tgt
                        break
                if hit is not None:
                    if b["kind"] == "mortar":
                        self.explode(b, direct=hit)
                    else:
                        self.damage(hit, b["dmg"], "npc" if b["kind"] == "npc" else "gun", b["owner"], (b["ox"], b["oy"]))
                    alive = False
                    break
            if alive:
                keep.append(b)
        self.bullets = keep

    def explode(self, b, direct=None):
        st = self.cfg["weapons"]["mortar"]
        src = b["owner"]
        self.sound("explosion", b["x"], b["y"])
        if direct is not None:
            self.damage(direct, b["dmg"], "mortar", src, (b["x"], b["y"]))
        R = st["splash_radius"]
        for tgt, tx, ty, tz, h in self.targets(near=(b["x"], b["y"], R + 3)):
            if tgt is direct:
                continue
            d = math.hypot(tx - b["x"], ty - b["y"])
            if d > R:
                continue
            f = 1.0 - (1.0 - st["splash_edge"]) * d / R
            dmg = st["splash"] * f * (src.dmg_mult() if isinstance(src, M) else 1.0)
            if tgt is src:
                dmg *= self.cfg["mech"]["self_damage_mult"]
            self.damage(tgt, dmg, "mortar", src, (b["x"], b["y"]))

    # ---------------------------------------------------------------- physics
    def step_mech(self, m):
        cfg = self.cfg["mech"]
        cmd = m.cmd
        mv = cmd.get("move") or [0, 0]
        try:
            mx, my = float(mv[0]), float(mv[1])
        except Exception:
            mx = my = 0.0
        n = math.hypot(mx, my)
        if n > 1.0:
            mx, my = mx / n, my / n
            n = 1.0
        charging = any(w and w.kind == "laser" and w.charge > 0 for w in m.weapons)
        spd = m.walk
        boost = bool(cmd.get("boost")) and m.energy > 0.5
        if m.grounded and boost and n > 0.1:
            spd = m.boost_speed
            m.energy -= cfg["boost_drain"] * DT
            m.last_spend = self.t
        if charging:
            spd *= cfg["charge_move_mult"]
        tvx, tvy = mx * spd, my * spd
        acc = (cfg["ground_accel"] if m.grounded else cfg["air_accel"]) * DT
        dvx, dvy = tvx - m.vx, tvy - m.vy
        dl = math.hypot(dvx, dvy)
        if dl > acc:
            dvx, dvy = dvx / dl * acc, dvy / dl * acc
        m.vx += dvx
        m.vy += dvy
        if cmd.get("jump") and m.grounded:
            m.vz = cfg["jump_speed"]
            m.grounded = False
        if not m.grounded:
            if boost:
                m.vz = min(m.vz + cfg["jet_accel"] * DT, cfg["max_rise_speed"]) if m.vz < cfg["max_rise_speed"] else m.vz
                m.energy -= cfg["jet_drain"] * DT
                m.last_spend = self.t
            m.vz -= cfg["gravity"] * DT
        nx, ny = m.x + m.vx * DT, m.y + m.vy * DT
        nz = m.z + m.vz * DT
        h = self.surf(nx, ny)
        lim = max(m.z, nz) + cfg["step_up"]
        if h > lim:
            if self.surf(nx, m.y) <= lim:
                ny = m.y
                m.vy = 0.0
            elif self.surf(m.x, ny) <= lim:
                nx = m.x
                m.vx = 0.0
            else:
                nx, ny = m.x, m.y
                m.vx = m.vy = 0.0
        m.x, m.y = nx, ny
        g = self.surf(m.x, m.y)
        if nz <= g + 0.01:
            m.z = g
            m.vz = 0.0
            m.grounded = True
        else:
            if m.grounded and nz - g < 1.3 and m.vz <= 0:
                m.z = g          # walking down a slope
            else:
                m.z = nz
                m.grounded = False
        if m.energy < m.max_energy and self.t - m.last_spend > cfg["energy_regen_delay"]:
            m.energy = min(m.max_energy, m.energy + cfg["energy_regen"] * DT)
        m.energy = max(0.0, m.energy)
        # torso slew
        ay = float((cmd.get("aim") or [m.yaw, 0])[0])
        ap = max(cfg["min_pitch"], min(cfg["max_pitch"], float((cmd.get("aim") or [0, 0])[1])))
        tr = cfg["turn_rate"] * (1.3 ** m.upgrades.count("gyros")) * DT
        d = wrap(ay - m.yaw)
        m.yaw = wrap(m.yaw + max(-tr, min(tr, d)))
        pr = cfg["pitch_rate"] * DT
        m.pitch += max(-pr, min(pr, ap - m.pitch))

    def step_weapons(self, m, fire, reload):
        for i, w in enumerate(m.weapons):
            if w is None:
                continue
            w.cooldown = max(0.0, w.cooldown - DT)
            w.equip = max(0.0, w.equip - DT)
            if w.kind == "gun" and w.reload > 0:
                w.reload -= DT
                if w.reload <= 0:
                    w.reload = 0.0
                    w.ammo = w.st["mag"]
            trig = bool(fire[i]) if i < len(fire) else False
            ready = w.cooldown <= 0 and w.equip <= 0
            if w.kind == "laser":
                st = w.st
                if trig and ready:
                    w.charge = min(st["charge"] + st["hold_s"], w.charge + DT)
                if w.charge > 0 and ((not trig and w.charge >= st["min_charge"]) or w.charge >= st["charge"] + st["hold_s"]):
                    self.fire_weapon(m, w)
                    w.charge = 0.0
                    w.cooldown = st["cooldown"]
                elif w.charge > 0 and not trig:
                    w.charge += DT          # keeps charging up to min_charge, then fires
            elif trig and ready:
                if w.kind == "gun":
                    if w.ammo > 0 and w.reload <= 0:
                        self.fire_weapon(m, w)
                        w.ammo -= 1
                        w.cooldown = w.st["cooldown"]
                        if w.ammo <= 0:
                            w.reload = w.st["reload"]
                elif w.kind == "lightning":
                    if m.energy >= w.st["energy"]:
                        self.fire_weapon(m, w)
                        w.cooldown = w.st["cooldown"]
                elif w.kind == "mortar":
                    self.fire_weapon(m, w)
                    w.cooldown = w.st["cooldown"]
            if w.kind == "gun" and reload and w.reload <= 0 and w.ammo < w.st["mag"]:
                w.reload = w.st["reload"]

    # ---------------------------------------------------------------- npcs
    def step_npcs(self):
        cfg = self.cfg["npc"]
        for n in self.npcs:
            if not n["alive"]:
                continue
            n["cd"] = max(0.0, n["cd"] - DT)
            tgt = n["target"] if (n["target"] is not None and self.t - n["agg_t"] < 10 and n["target"].alive) else None
            if tgt is not None and math.hypot(tgt.x - n["x"], tgt.y - n["y"]) > cfg["deaggro_range"]:
                tgt = n["target"] = None
            if tgt is None and math.hypot(n["x"] - n["hx"], n["y"] - n["hy"]) < cfg["leash"]:
                best = None
                for m in self.mechs:
                    if m.alive:
                        d = math.hypot(m.x - n["x"], m.y - n["y"])
                        if d < cfg["aggro_range"] and (best is None or d < best[0]):
                            best = (d, m)
                if best and self.los((n["x"], n["y"], n["z"] + 3), (best[1].x, best[1].y, best[1].z + 4.5)):
                    tgt = best[1]
            # move
            if tgt is not None and math.hypot(n["x"] - n["hx"], n["y"] - n["hy"]) < cfg["chase_leash"]:
                gx, gy = tgt.x, tgt.y
            else:
                if self.rng.random() < 0.01 or math.hypot(n["wx"] - n["x"], n["wy"] - n["y"]) < 2:
                    a = self.rng.uniform(0, TWO_PI)
                    rr = cfg["leash"] * math.sqrt(self.rng.random())
                    n["wx"], n["wy"] = n["hx"] + rr * math.cos(a), n["hy"] + rr * math.sin(a)
                gx, gy = n["wx"], n["wy"]
            d = math.hypot(gx - n["x"], gy - n["y"])
            if d > 3:
                nx = n["x"] + (gx - n["x"]) / d * cfg["speed"] * DT
                ny = n["y"] + (gy - n["y"]) / d * cfg["speed"] * DT
                if self.surf(nx, ny) < n["z"] + 1.5:
                    n["vx"], n["vy"] = (nx - n["x"]) / DT, (ny - n["y"]) / DT
                    n["x"], n["y"] = nx, ny
                    n["z"] = self.surf(nx, ny)
            if tgt is not None and n["cd"] <= 0:
                if self.los((n["x"], n["y"], n["z"] + 3), (tgt.x, tgt.y, tgt.z + 4.5)):
                    a = math.atan2(tgt.y - n["y"], tgt.x - n["x"]) + self.rng.gauss(0, math.radians(cfg["spread_deg"]))
                    dd = math.hypot(tgt.x - n["x"], tgt.y - n["y"])
                    pz = math.atan2(tgt.z + 4.5 - n["z"] - 3, dd)
                    v = cfg["projectile_speed"]
                    self.bullets.append({"id": self.next_id, "kind": "npc", "x": n["x"], "y": n["y"], "z": n["z"] + 3,
                                         "vx": v * math.cos(a) * math.cos(pz), "vy": v * math.sin(a) * math.cos(pz), "vz": v * math.sin(pz),
                                         "owner": n, "left": 150.0, "dmg": cfg["damage"], "ox": n["x"], "oy": n["y"]})
                    self.next_id += 1
                    n["cd"] = cfg["cooldown"]
                    self.sound("npc_gun", n["x"], n["y"])

    # ---------------------------------------------------------------- house bots
    def update_squad(self):
        known = set()
        house = [m for m in self.mechs if m.alive and m.ctrl == "house"]
        for m in self.mechs:
            if not m.alive:
                continue
            for h in house:
                if h is m:
                    continue
                if self.sees(h, m.x, m.y, m.z, 9.0 * m.size):
                    known.add(m.id)
                    break
        self.squad_known = known

    def house_decide(self, m):
        if self.mode == "ffa":
            cands = [o for o in self.mechs if o.alive and o is not m and o.id in self.squad_known]
        else:
            me = self.me
            cands = [me] if me.alive and (self.mode == "omni" or me.id in self.squad_known) else []
        tgt = min(cands, key=lambda o: (o.hp, math.hypot(o.x - m.x, o.y - m.y))) if cands else None
        c, r, ph = self.circle(self.t)
        in_z = math.hypot(m.x - c[0], m.y - c[1]) < r - 5
        nc = ph["c1"]
        d_next = math.hypot(m.x - nc[0], m.y - nc[1])
        urgent = (not in_z) or (self.t > ph["s0"] - 20 and d_next > ph["r1"] * 0.8)
        if urgent or tgt is None:
            if m.goal is None or self.rng.random() < 0.02 or urgent:
                a = self.rng.uniform(0, TWO_PI)
                rr = ph["r1"] * 0.6 * math.sqrt(self.rng.random())
                m.goal = (nc[0] + rr * math.cos(a), nc[1] + rr * math.sin(a))
            gx, gy = m.goal
            want = 2.0
        else:
            gx, gy = tgt.x, tgt.y
            want = 45.0
        d = math.hypot(gx - m.x, gy - m.y)
        mx = my = 0.0
        if d > want:
            mx, my = (gx - m.x) / d, (gy - m.y) / d
            # bend around walls
            a0 = math.atan2(my, mx)
            for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4):
                a = a0 + da
                if self.wm.walk_line(m.x, m.y, m.x + 6 * math.cos(a), m.y + 6 * math.sin(a), avoid_near=False):
                    mx, my = math.cos(a), math.sin(a)
                    break
        # unstick
        lx, ly, lt = m.last_pos
        if self.t - lt > 2.0:
            if math.hypot(m.x - lx, m.y - ly) < 2.0 and d > want + 5:
                m.stuck_t = self.t + 1.0
                m.goal = None
            m.last_pos = (m.x, m.y, self.t)
        jump = False
        if self.t < m.stuck_t:
            a = self.rng.uniform(0, TWO_PI)
            mx, my = math.cos(a), math.sin(a)
            jump = True
        elif tgt is not None and d <= want + 10:
            # strafe
            s = 1 if int(self.t / 1.5 + m.id) % 2 else -1
            ux, uy = (gx - m.x) / max(d, 1), (gy - m.y) / max(d, 1)
            mx, my = -uy * s, ux * s
        fire = [False, False]
        aim = [math.atan2(my, mx) if (mx or my) else m.yaw, 0.0]
        if tgt is not None:
            mzz = m.z + 7.0 * m.size
            tz = tgt.z + 4.5 * tgt.size
            dd = math.hypot(tgt.x - m.x, tgt.y - m.y)
            tof = dd / 252.0
            px, py = tgt.x + tgt.vx * tof, tgt.y + tgt.vy * tof
            aim = [math.atan2(py - m.y, px - m.x), math.atan2(tz - mzz, max(dd, 1))]
            if dd < 200 and self.los((m.x, m.y, mzz), (tgt.x, tgt.y, tz)):
                err = abs(wrap(aim[0] - m.yaw))
                fire = [err < max(0.02, math.atan2(2.5, max(dd, 1))), False]
        m.cmd = {"move": [mx, my], "aim": aim, "boost": urgent and not in_z, "jump": jump, "fire": fire}
        if m.offer_queue:
            m.offer_queue.pop(0)
            m.plating *= 1.0  # house bots ignore upgrades

    # ---------------------------------------------------------------- our bot
    def bot_obs(self):
        m = self.me
        t = self.t
        cfg = self.cfg
        c, r, ph = self.circle(t)
        mechs = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            if self.sees(m, o.x, o.y, o.z, 9.0 * o.size):
                m.seen[o.id] = t
            if t - m.seen.get(o.id, -99) <= cfg["vision"]["memory_s"]:
                mechs.append({"id": o.id, "kind": "mech", "pos": [round(o.x, 2), round(o.y, 2), round(o.z, 2)],
                              "vel": [round(o.vx, 2), round(o.vy, 2), round(o.vz, 2)], "aim": [round(o.yaw, 3), round(o.pitch, 3)],
                              "hp": round(o.hp), "max_hp": round(o.max_hp), "cubes": o.cubes,
                              "weapons": [w.kind for w in o.weapons if w], "charging": any(w and w.kind == "laser" and w.charge > 0 for w in o.weapons)})
        for n in self.npcs:
            if not n["alive"]:
                continue
            if math.hypot(n["x"] - m.x, n["y"] - m.y) < 200 and self.sees(m, n["x"], n["y"], n["z"], 3.5):
                m.seen[n["id"]] = t
            if t - m.seen.get(n["id"], -99) <= cfg["vision"]["memory_s"]:
                mechs.append({"id": n["id"], "kind": "npc", "pos": [n["x"], n["y"], n["z"]], "vel": [n["vx"], n["vy"], 0], "aim": [0, 0],
                              "hp": n["hp"], "max_hp": cfg["npc"]["hp"]})

        def vis_items(items, h):
            out = []
            for it in items:
                if it.get("alive", True) and (it["x"] - m.x) ** 2 + (it["y"] - m.y) ** 2 < 200 ** 2 and self.sees(m, it["x"], it["y"], it["z"], h):
                    out.append(it)
            return out
        crates = [{"id": c_["id"], "pos": [c_["x"], c_["y"], c_["z"]], "hp": c_["hp"]} for c_ in vis_items(self.crates, 2.4)]
        cubes = [{"id": c_["id"], "pos": [c_["x"], c_["y"], c_["z"]]} for c_ in vis_items(self.cubes, 0.8)]
        picks = [{"id": p["id"], "kind": p["kind"], "pos": [p["x"], p["y"], p["z"]]} for p in vis_items(self.pickups, 1.0)]
        projs = []
        for b in self.bullets:
            if b["owner"] is m:
                continue
            if (b["x"] - m.x) ** 2 + (b["y"] - m.y) ** 2 < 200 ** 2:
                projs.append({"id": b["id"], "kind": b["kind"], "pos": [round(b["x"], 2), round(b["y"], 2), round(b["z"], 2)],
                              "vel": [round(b["vx"], 2), round(b["vy"], 2), round(b["vz"], 2)]})
        if m.offer is None and m.offer_queue:
            m.offer_queue.pop(0)
            m.offer_seq += 1
            pool = [u for u in cfg["upgrades"] if (not u.get("requires") or any(w and w.kind == u["requires"] for w in m.weapons))
                    and m.upgrades.count(u["id"]) < u["max"]]
            ch = self.rng.sample(pool, min(3, len(pool)))
            m.offer = {"id": m.offer_seq, "expires": t + cfg["bot"]["offer_timeout_s"],
                       "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]}
        drops = []
        for d in self.drops:
            drops.append({"id": d["id"], "pos": [d["x"], d["y"], d["z"]], "radius": d["radius"], "lands_in": round(max(0.0, d["land"] - t), 2),
                          "capture_s": d["capture_s"], "progress": round(d["progress"], 2), "holder": d["holder"]})
        st = dict(cfg["mech"])
        st["walk_speed"] = m.walk
        st["boost_speed"] = m.boost_speed
        st["armor"] = m.armor
        st["height"] = 9.0 * m.size
        st["eye_height"] = 8.0 * m.size
        st["muzzle_height"] = 7.0 * m.size
        obs = {"type": "obs", "tick": self.tick, "t": round(t, 2), "alive": sum(1 for o in self.mechs if o.alive),
               "self": {"id": m.id, "pos": [round(m.x, 2), round(m.y, 2), round(m.z, 2)], "vel": [round(m.vx, 2), round(m.vy, 2), round(m.vz, 2)],
                        "aim": [round(m.yaw, 3), round(m.pitch, 3)], "hp": round(m.hp), "max_hp": round(m.max_hp),
                        "energy": round(m.energy), "max_energy": m.max_energy, "grounded": m.grounded, "cubes": m.cubes,
                        "upgrades": list(m.upgrades), "in_zone": self.in_zone(m.x, m.y),
                        "weapons": [w.obs() if w else None for w in m.weapons], "stats": st},
               "mechs": mechs, "projectiles": projs, "crates": crates, "cubes": cubes, "pickups": picks,
               "sounds": m.sounds, "beams": [], "events": m.events, "drops": drops,
               "zone": {"center": list(ph["c0"]), "radius": ph["r0"], "next_center": list(ph["c1"]), "next_radius": ph["r1"],
                        "shrink_start": ph["s0"], "shrink_end": ph["s1"], "dps": ph["dps"]},
               "offer": m.offer}
        m.sounds = []
        m.events = []
        return obs

    def apply_bot_action(self, act):
        m = self.me
        if not isinstance(act, dict):
            self.errors += 1
            return
        cmd = dict(m.cmd)
        for k in ("move", "aim", "boost"):
            if k in act:
                cmd[k] = act[k]
        cmd["jump"] = bool(act.get("jump"))
        cmd["fire"] = act.get("fire") or [False, False]
        cmd["reload"] = bool(act.get("reload"))
        m.cmd = cmd
        if "pick" in act and m.offer is not None:
            i = int(act["pick"])
            ch = m.offer["choices"]
            if 0 <= i < len(ch):
                self.apply_upgrade(m, ch[i]["id"])
                m.offer = None
        if "take" in act:
            pid = act["take"]
            for p in self.pickups:
                if p["id"] == pid:
                    if math.hypot(p["x"] - m.x, p["y"] - m.y) > self.cfg["loot"]["pickup_radius"]:
                        m.events.append({"type": "take_failed", "id": pid, "reason": "range"})
                        break
                    if any(w and w.kind == p["kind"] for w in m.weapons):
                        m.events.append({"type": "take_failed", "id": pid, "reason": "have"})
                        break
                    slot = act.get("drop")
                    if slot is None:
                        slot = 1 if m.weapons[1] is None else (0 if m.weapons[0] is None else None)
                    if slot is None:
                        m.events.append({"type": "take_failed", "id": pid, "reason": "full"})
                        break
                    old = m.weapons[slot]
                    nw = W(p["kind"], self.cfg)
                    nw.equip = max(self.cfg["loot"]["equip_s"], nw.st.get("cooldown", 0))
                    m.weapons[slot] = nw
                    self.pickups.remove(p)
                    m.events.append({"type": "took", "id": pid, "kind": p["kind"], "slot": slot})
                    if old is not None:
                        self.pickups.append({"id": self.next_pickup, "kind": old.kind, "x": m.x, "y": m.y, "z": m.z})
                        m.events.append({"type": "dropped", "id": self.next_pickup, "kind": old.kind, "slot": slot})
                        self.next_pickup += 1
                    break
            else:
                m.events.append({"type": "take_failed", "id": pid, "reason": "gone"})

    def apply_upgrade(self, m, uid):
        m.upgrades.append(uid)
        m.events.append({"type": "upgrade", "id": uid})
        before = m.max_hp
        if uid == "plating":
            m.plating *= 1.15
            m.hp += m.max_hp - before
        elif uid == "armor":
            m.armor += 0.08
        elif uid == "nanites":
            m.regen *= 1.6
            m.regen_delay -= 1.5
        elif uid == "servos":
            m.walk *= 1.08
            m.boost_speed *= 1.08
        elif uid == "capacitor":
            m.max_energy += 30
        else:
            for u in self.cfg["upgrades"]:
                if u["id"] == uid and u.get("requires"):
                    for w in m.weapons:
                        if w and w.kind == u["requires"]:
                            for e in u["effects"]:
                                key = e["path"].split(".")[-1]
                                if key in w.st:
                                    w.st[key] = w.st[key] * e.get("mul", 1) + e.get("add", 0)

    # ---------------------------------------------------------------- main loop
    def run(self):
        cfg = self.cfg
        end_t = cfg["match"]["duration_s"]
        while True:
            alive = [m for m in self.mechs if m.alive]
            if len(alive) <= 1 or self.t >= end_t:
                break
            decide = self.tick % 2 == 0
            if decide:
                if self.t - self.squad_t >= 0.5:
                    self.update_squad()
                    self.squad_t = self.t
                for m in alive:
                    if m.ctrl == "house":
                        self.house_decide(m)
                if self.me.alive:
                    obs = json.loads(json.dumps(self.bot_obs()))
                    t0 = time.perf_counter()
                    try:
                        act = self.bot.act(obs, t0 + 0.012)
                        act = json.loads(json.dumps(act))
                    except Exception:
                        self.errors += 1
                        if self.errors <= 3:
                            traceback.print_exc()
                        act = {}
                    self.act_times.append((time.perf_counter() - t0) * 1000)
                    self.apply_bot_action(act)
                    if self.me.offer is not None and self.t > self.me.offer["expires"]:
                        self.apply_upgrade(self.me, self.me.offer["choices"][0]["id"])
                        self.me.offer = None
            for m in alive:
                self.step_mech(m)
                self.step_weapons(m, m.cmd.get("fire") or [False, False], m.cmd.get("reload"))
                if not decide:
                    pass
                m.cmd["jump"] = False
            self.step_bullets()
            self.step_npcs()
            # cubes
            for cb in list(self.cubes):
                best = None
                for m in self.mechs:
                    if m.alive:
                        d = math.hypot(m.x - cb["x"], m.y - cb["y"])
                        if d < cfg["cubes"]["pickup_radius"] and abs(m.z - cb["z"]) < 4 and (best is None or d < best[0]):
                            best = (d, m)
                if best:
                    self.cubes.remove(cb)
                    self.give_cubes(best[1], 1)
                    best[1].events.append({"type": "pickup", "cubes": best[1].cubes})
            # zone, regen
            c, r, ph = self.circle(self.t)
            for m in self.mechs:
                if not m.alive:
                    continue
                inside = math.hypot(m.x - c[0], m.y - c[1]) < r
                if not inside:
                    self.damage(m, ph["dps"] * DT, "zone", None, None)
                elif self.t - m.last_combat >= m.regen_delay and m.hp < m.max_hp:
                    m.hp = min(m.max_hp, m.hp + m.regen * DT)
            # drops
            dc = cfg.get("drops") or {}
            if self.drop_times and self.t >= self.drop_times[0]:
                self.drop_times.pop(0)
                ph_ = self.phase(self.t + dc["fall_s"])
                for _ in range(200):
                    a = self.rng.uniform(0, TWO_PI)
                    rr = ph_["r1"] * dc["spread"] * math.sqrt(self.rng.random())
                    x, y = ph_["c1"][0] + rr * math.cos(a), ph_["c1"][1] + rr * math.sin(a)
                    cc = self.wm.cell(x, y)
                    if cc >= 0 and not self.wm.solid[cc]:
                        break
                self.drops.append({"id": self.drop_seq, "x": x, "y": y, "z": self.wm.floor_at(x, y), "radius": dc["radius"],
                                   "land": self.t + dc["fall_s"], "capture_s": dc["capture_s"], "progress": 0.0, "holder": None})
                self.drop_seq += 1
            for d in list(self.drops):
                if self.t < d["land"]:
                    continue
                inside = [m for m in self.mechs if m.alive and math.hypot(m.x - d["x"], m.y - d["y"]) < d["radius"]]
                if len(inside) == 1:
                    if d["holder"] != inside[0].id:
                        d["holder"] = inside[0].id
                        d["progress"] = 0.0
                    d["progress"] += DT
                    if d["progress"] >= d["capture_s"]:
                        m = inside[0]
                        self.give_cubes(m, dc["cubes"])
                        m.hp = m.max_hp
                        m.energy = m.max_energy
                        m.events.append({"type": "drop", "id": d["id"], "cubes": m.cubes})
                        self.drops.remove(d)
                elif not inside:
                    d["holder"] = None
                    d["progress"] = 0.0
            self.t += DT
            self.tick += 1
        return self.placement()

    def placement(self):
        # death order -> place; same tick share the better place
        n = len(self.mechs)
        place = {}
        alive_before = n
        by_tick = {}
        for tk, mid in self.dead_order:
            by_tick.setdefault(tk, []).append(mid)
        for tk in sorted(by_tick):
            ids = by_tick[tk]
            p = alive_before - len(ids) + 1
            for mid in ids:
                place[mid] = p
            alive_before -= len(ids)
        surv = sorted([m for m in self.mechs if m.alive], key=lambda m: -m.hp)
        for i, m in enumerate(surv):
            place[m.id] = i + 1
        me = self.me
        return {"place": place[me.id], "t_death": None if me.alive else round(me.t_death, 1), "cause": me.death_cause,
                "dmg_taken": round(me.dmg_taken), "zone_dmg": round(me.zone_dmg), "cubes": me.cubes, "upgrades": me.upgrades,
                "weapons": [w.kind for w in me.weapons if w], "errors": self.errors,
                "act_ms_p99": sorted(self.act_times)[int(len(self.act_times) * 0.99)] if self.act_times else 0,
                "act_ms_max": max(self.act_times) if self.act_times else 0, "end_t": round(self.t, 1)}


def one_match(args):
    seed, verbose, mode = args
    import replay
    import main as botmain
    hello, _ = replay.load_stream()
    wm = _WM.get("wm")
    if wm is None:
        wm = _WM["wm"] = WorldMap(os.path.join(ROOT, "bot", "arena"), hello["config"]["mech"])
    t0 = time.perf_counter()
    sim = Sim(wm, hello, seed, botmain.Bot, verbose, mode)
    res = sim.run()
    res["seed"] = seed
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


_WM = {}


def main():
    args = sys.argv[1:]
    n = int(args[args.index("--matches") + 1]) if "--matches" in args else 4
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 1
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else max(1, min(n, (os.cpu_count() or 2) - 1))
    verbose = "-v" in args
    mode = "omni" if "--omni" in args else ("squad" if "--squad" in args else "ffa")
    jobs = [(seed + k, verbose, mode) for k in range(n)]
    results = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(one_match, jobs):
                results.append(r)
                print(json.dumps(r), flush=True)
    else:
        for j in jobs:
            r = one_match(j)
            results.append(r)
            print(json.dumps(r), flush=True)
    places = [r["place"] for r in results]
    score = sum(16 - p for p in places) / len(places)
    print(f"mode {mode}  matches {len(places)}  mean place {sum(places) / len(places):.2f}  score {score:.2f}  "
          f"zone deaths {sum(1 for r in results if r['cause'] == 'zone')}  errors {sum(r['errors'] for r in results)}  "
          f"act p99 max {max(r['act_ms_p99'] for r in results):.1f} ms")


if __name__ == "__main__":
    main()
