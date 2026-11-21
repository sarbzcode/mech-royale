"""Perception: what we know about other mechs, NPC drones, danger spots and incoming fire."""
import math

SOUND_DIST = {"near": 35.0, "mid": 110.0, "far": 240.0}
# how strongly a heard sound marks a place as dangerous (combat happening there)
SOUND_WEIGHT = {"gun": 1.0, "mortar": 1.2, "explosion": 0.8, "laser": 1.2, "laser_charge": 1.0,
                "lightning": 1.3, "death": 0.4, "npc_gun": 0.3, "crate": 0.5, "boost": 0.7,
                "jet": 0.6, "step": 0.9, "reload": 0.8}


class Track:
    __slots__ = ("id", "kind", "x", "y", "z", "vx", "vy", "vz", "hp", "max_hp", "cubes", "weapons",
                 "yaw", "pitch", "charging", "t", "t_first", "hx", "hy", "listed")

    def __init__(self, mid, kind):
        self.id = mid
        self.kind = kind
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.hp = self.max_hp = 1400.0
        self.cubes = 0
        self.weapons = ()
        self.yaw = self.pitch = 0.0
        self.charging = False
        self.t = self.t_first = -99.0
        self.hx = self.hy = 0.0      # NPC home estimate (first sighting)
        self.listed = False          # present in the latest observation

    def pos_at(self, t, max_dt=2.0):
        dt = min(max(t - self.t, 0.0), max_dt)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def dist2(self, x, y):
        return (self.x - x) ** 2 + (self.y - y) ** 2


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Perception:
    def __init__(self, cfg, me_id):
        self.cfg = cfg
        self.me_id = me_id
        self.enemies = {}        # id -> Track (mechs)
        self.npcs = {}           # id -> Track (drones)
        self.dangers = []        # [x, y, t, weight]
        self.attackers = {}      # mech id -> last time it damaged us
        self.hurt = []           # (t, amount, by)
        self.last_hurt_t = -99.0
        self.last_hurt_mech_t = -99.0
        self.last_hurt_npc_t = -99.0
        self.last_zone_hurt_t = -99.0
        self.last_deal_t = -99.0
        self.last_rear_t = -99.0
        self.hurt_bearing = None
        self.projectiles = []
        self.beams = []
        self.pickup_fail = {}    # pickup id -> time of a failed take
        self.kills = 0
        self.t = 0.0

    def update(self, obs):
        t = _f(obs.get("t"), self.t)
        self.t = t
        me = obs.get("self") or {}
        pos = me.get("pos") or [0.0, 0.0, 0.0]
        mx, my = _f(pos[0]), _f(pos[1])
        for tr in self.enemies.values():
            tr.listed = False
        for tr in self.npcs.values():
            tr.listed = False
        for m in obs.get("mechs") or []:
            try:
                mid = int(m.get("id"))
                kind = m.get("kind", "mech")
                if kind == "mech" and mid == self.me_id:
                    continue
                book = self.enemies if kind == "mech" else self.npcs
                tr = book.get(mid)
                if tr is None:
                    tr = book[mid] = Track(mid, kind)
                    tr.t_first = t
                p = m.get("pos") or [0, 0, 0]
                v = m.get("vel") or [0, 0, 0]
                tr.x, tr.y, tr.z = _f(p[0]), _f(p[1]), _f(p[2])
                tr.vx, tr.vy, tr.vz = _f(v[0]), _f(v[1]), _f(v[2])
                if kind != "mech" and tr.t_first == t:
                    tr.hx, tr.hy = tr.x, tr.y
                if "hp" in m:
                    tr.hp = _f(m.get("hp"), tr.hp)
                if "max_hp" in m:
                    tr.max_hp = max(1.0, _f(m.get("max_hp"), tr.max_hp))
                tr.cubes = int(_f(m.get("cubes"), tr.cubes))
                tr.weapons = tuple(m.get("weapons") or ())
                a = m.get("aim") or [0, 0]
                tr.yaw, tr.pitch = _f(a[0]), _f(a[1])
                tr.charging = bool(m.get("charging"))
                tr.t = t
                tr.listed = True
            except Exception:
                continue
        # forget old tracks (dead or long gone)
        for book, ttl in ((self.enemies, 45.0), (self.npcs, 120.0)):
            for k in [k for k, tr in book.items() if t - tr.t > ttl]:
                del book[k]

        for ev in obs.get("events") or []:
            try:
                typ = ev.get("type")
                if typ == "damaged":
                    amt = _f(ev.get("amount"))
                    by = ev.get("by")
                    w = ev.get("weapon")
                    self.last_hurt_t = t
                    self.hurt.append((t, amt, by))
                    if w == "zone" or by is None:
                        self.last_zone_hurt_t = t
                    elif w == "npc" or (isinstance(by, (int, float)) and by >= 100):
                        self.last_hurt_npc_t = t
                    else:
                        self.last_hurt_mech_t = t
                        self.attackers[int(by)] = t
                        b = ev.get("bearing")
                        if b is not None:
                            self.hurt_bearing = (_f(b), t)
                            # the attacker is somewhere along that bearing: mark it
                            tr = self.enemies.get(int(by))
                            if tr is None or t - tr.t > 1.0:
                                d = 80.0
                                self.dangers.append([mx + d * math.cos(_f(b)), my + d * math.sin(_f(b)), t, 2.0])
                    if ev.get("rear"):
                        self.last_rear_t = t
                elif typ == "hit":
                    tgt = ev.get("target")
                    if isinstance(tgt, (int, float)) and tgt < 1000:
                        self.last_deal_t = t
                elif typ == "kill":
                    self.kills += 1
                    self.enemies.pop(int(_f(ev.get("target"), -1)), None)
                elif typ == "take_failed":
                    self.pickup_fail[int(_f(ev.get("id"), -1))] = t
            except Exception:
                continue
        self.hurt = [h for h in self.hurt if t - h[0] < 10.0]

        for s in obs.get("sounds") or []:
            try:
                kind = s.get("kind")
                w = SOUND_WEIGHT.get(kind)
                if not w:
                    continue
                d = SOUND_DIST.get(s.get("dist"), 150.0)
                b = _f(s.get("bearing"))
                self.dangers.append([mx + d * math.cos(b), my + d * math.sin(b), t - _f(s.get("age")), w])
            except Exception:
                continue
        for tr in self.enemies.values():
            if tr.listed:
                self.dangers.append([tr.x, tr.y, t, 0.5])
        if len(self.dangers) > 150:
            self.dangers = self.dangers[-150:]
        self.dangers = [d for d in self.dangers if t - d[2] < 25.0]
        self.projectiles = obs.get("projectiles") or []
        self.beams = obs.get("beams") or []

    # ------------------------------------------------------------------ queries
    def visible(self, max_age=0.25):
        t = self.t
        return [tr for tr in self.enemies.values() if t - tr.t <= max_age]

    def known(self, max_age=12.0):
        t = self.t
        return [tr for tr in self.enemies.values() if t - tr.t <= max_age]

    def recent_damage(self, window=3.0):
        t = self.t
        return sum(a for (tt, a, _) in self.hurt if t - tt <= window)

    def danger_at(self, x, y, radius=110.0):
        """Weighted, time-decayed count of combat signs near (x, y)."""
        t = self.t
        r2 = radius * radius
        s = 0.0
        for dx_, dy_, tt, w in self.dangers:
            dx = dx_ - x
            if dx > radius or dx < -radius:
                continue
            dy = dy_ - y
            d2 = dx * dx + dy * dy
            if d2 < r2:
                s += w * (1.0 - math.sqrt(d2) / radius) * max(0.0, 1.0 - (t - tt) / 25.0)
        return s

    def danger_points(self, max_n=4):
        """A few strongest recent danger spots for path costs: (x, y, radius, weight)."""
        t = self.t
        pts = []
        for tr in self.enemies.values():
            age = t - tr.t
            if age < 15.0:
                px, py = tr.pos_at(t, 3.0)
                pts.append((px, py, 70.0, 6.0 * (1.0 - age / 15.0) + 1.0))
        for tr in self.npcs.values():
            if t - tr.t < 90.0:
                pts.append((tr.hx * 0.5 + tr.x * 0.5, tr.hy * 0.5 + tr.y * 0.5, 55.0, 4.0))
        pts.sort(key=lambda p: -p[3])
        return pts[:max_n]
