"""Combat: aiming solutions (lead, ballistics) and per-weapon trigger logic."""
import math

TWO_PI = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class Combat:
    def __init__(self, cfg, wm):
        self.cfg = cfg
        self.wm = wm
        w = cfg.get("weapons") or {}
        self.base = {k: dict(v) for k, v in w.items() if isinstance(v, dict)}
        self.laser_holding = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def slots(me):
        out = {}
        for i, w in enumerate(me.get("weapons") or []):
            if w and w.get("kind"):
                out[w["kind"]] = (i, w)
        return out

    def stat(self, w, kind, key, d):
        st = (w or {}).get("stats") or {}
        if key in st:
            return _f(st[key], d)
        return _f(self.base.get(kind, {}).get(key, d), d)

    @staticmethod
    def muzzle(me):
        p = me.get("pos") or [0, 0, 0]
        st = me.get("stats") or {}
        return _f(p[0]), _f(p[1]), _f(p[2]) + _f(st.get("muzzle_height"), 7.0)

    @staticmethod
    def body_center(tr):
        size = 1.0 + 0.05 * max(0, tr.cubes)
        return tr.z + 4.5 * size

    def lead(self, me, tr, speed, t_now, extra_dt=0.0):
        """Aim (yaw, pitch) to hit tr's body center with a straight projectile; returns (yaw, pitch, dist, tof)."""
        mx, my, mz = self.muzzle(me)
        age = max(0.0, min(1.0, t_now - tr.t))
        bx = tr.x + tr.vx * age
        by = tr.y + tr.vy * age
        bz = self.body_center(tr)
        vz = tr.vz * 0.3 if abs(tr.vz) < 4.0 else 0.0
        tof = 0.0
        px, py, pz = bx - mx, by - my, bz - mz
        for _ in range(3):
            dt = tof + extra_dt
            px = bx + tr.vx * dt - mx
            py = by + tr.vy * dt - my
            pz = bz + vz * dt - mz
            d = math.sqrt(px * px + py * py + pz * pz)
            tof = d / max(speed, 1.0)
        return math.atan2(py, px), math.atan2(pz, math.hypot(px, py)), math.hypot(px, py), tof

    def mortar_solution(self, me, tr, w, t_now, high=False):
        """Ballistic aim for the mortar at tr's feet; None if out of reach."""
        v = self.stat(w, "mortar", "speed", 72.0)
        g = self.stat(w, "mortar", "gravity", 22.0)
        p = me.get("pos") or [0, 0, 0]
        st = me.get("stats") or {}
        tube = _f(p[2]) + _f(st.get("height"), 9.0) + 0.5
        age = max(0.0, min(1.0, t_now - tr.t))
        tof = 0.0
        sol = None
        for _ in range(3):
            tx = tr.x + tr.vx * (age + tof)
            ty = tr.y + tr.vy * (age + tof)
            dx, dy = tx - _f(p[0]), ty - _f(p[1])
            x = math.hypot(dx, dy) - 3.0
            if x < 5.0:
                return None
            yv = tr.z + 0.5 - tube
            k = v ** 4 - g * (g * x * x + 2.0 * yv * v * v)
            if k < 0:
                return None
            r = math.sqrt(k)
            th = math.atan((v * v + r) / (g * x)) if high else math.atan((v * v - r) / (g * x))
            tof = x / max(0.1, v * math.cos(th))
            sol = (math.atan2(dy, dx), th, x + 3.0, tof)
        return sol

    # ------------------------------------------------------------------ main entry
    def engage(self, t, me, tr, allow=("gun", "lightning", "laser", "mortar"), aggressive=False):
        """Aim and trigger decisions against track tr. Returns (aim, fire[2], reload, info)."""
        wm = self.wm
        slots = self.slots(me)
        fire = [False, False]
        reload = False
        cur = me.get("aim") or [0.0, 0.0]
        cy, cp = _f(cur[0]), _f(cur[1])
        mx, my, mz = self.muzzle(me)
        size = 1.0 + 0.05 * max(0, tr.cubes)
        bz = self.body_center(tr)
        d = math.hypot(tr.x - mx, tr.y - my)
        los_c = wm.los(mx, my, mz, tr.x, tr.y, bz)
        los = los_c or wm.los(mx, my, mz, tr.x, tr.y, tr.z + 8.6 * size)
        energy = _f(me.get("energy"))
        tol = max(0.006, math.atan2(1.9 * size, max(d, 1.0)))

        # pick the weapon that sets the aim this tick
        aim = None
        primary = None
        if "lightning" in slots and "lightning" in allow:
            i, w = slots["lightning"]
            rng = self.stat(w, "lightning", "range", 45.0)
            if d <= rng * 0.92 and los:
                primary = "lightning"
        if primary is None and "laser" in slots and "laser" in allow:
            i, w = slots["laser"]
            rng = self.stat(w, "laser", "range", 280.0)
            if (los and d <= rng * 0.95 and _f(w.get("cooldown")) <= 0.3) or _f(w.get("charge")) > 0:
                primary = "laser"
        if primary is None and "gun" in slots and "gun" in allow:
            i, w = slots["gun"]
            if los and d <= self.stat(w, "gun", "range", 230.0) * 0.92:
                primary = "gun"
        if primary is None and "mortar" in slots and "mortar" in allow:
            primary = "mortar"
        if primary is None and "gun" in slots:
            primary = "gun"

        if primary == "mortar":
            i, w = slots["mortar"]
            sol = self.mortar_solution(me, tr, w, t, high=not los)
            if sol:
                aim = [sol[0], sol[1]]
        elif primary == "laser":
            ya, pa, _, _ = self.lead(me, tr, 1e6, t, extra_dt=0.05)
            aim = [ya, pa]
        elif primary == "gun":
            i, w = slots["gun"]
            ya, pa, _, _ = self.lead(me, tr, self.stat(w, "gun", "speed", 252.0), t, extra_dt=0.05)
            aim = [ya, pa]
        if aim is None:
            ya, pa, _, _ = self.lead(me, tr, 1e6, t)
            aim = [ya, pa]

        # triggers: each weapon fires if its own solution matches the actual torso aim
        for kind, (i, w) in slots.items():
            if kind not in allow or i > 1:
                continue
            cd = _f(w.get("cooldown"))
            if kind == "gun":
                ammo = _f(w.get("ammo"), 1)
                rel = _f(w.get("reload"))
                if ammo <= 0 and rel <= 0:
                    reload = True
                if not los or ammo <= 0 or rel > 0 or d > self.stat(w, "gun", "range", 230.0) * 0.95:
                    continue
                ya, pa, _, _ = self.lead(me, tr, self.stat(w, "gun", "speed", 252.0), t, extra_dt=0.03)
                if abs(wrap(ya - cy)) < tol * 1.2 and abs(pa - cp) < tol * 1.5 + 0.01:
                    fire[i] = True
            elif kind == "lightning":
                rng = self.stat(w, "lightning", "range", 45.0)
                cost = self.stat(w, "lightning", "energy", 3.0)
                ya = math.atan2(tr.y - my, tr.x - mx)
                cone = math.radians(self.stat(w, "lightning", "cone_deg", 30.0)) * 0.5
                if los and d <= rng * 0.95 and abs(wrap(ya - cy)) < cone * 0.75 and energy >= cost + (2 if aggressive else 8) and cd <= 0.05:
                    fire[i] = True
            elif kind == "laser":
                charge = _f(w.get("charge"))
                full = self.stat(w, "laser", "charge", 0.75)
                ya, pa, _, _ = self.lead(me, tr, 1e6, t, extra_dt=0.05)
                aligned = abs(wrap(ya - cy)) < tol and abs(pa - cp) < tol + 0.01
                if charge > 0:
                    # keep holding until full and aligned, then let go
                    fire[i] = not (charge >= full * 0.97 and aligned and los)
                elif cd <= 0.0 and los and d <= self.stat(w, "laser", "range", 280.0) * 0.95 and abs(wrap(ya - cy)) < 0.5:
                    fire[i] = True
            elif kind == "mortar":
                if cd > 0 or d < 22.0:
                    continue
                sol = self.mortar_solution(me, tr, w, t, high=not los)
                if sol and abs(wrap(sol[0] - cy)) < 0.03 and abs(sol[1] - cp) < 0.025:
                    fire[i] = True
        return aim, fire, reload, {"primary": primary, "los": los, "dist": d}

    def shoot_point(self, me, x, y, z, use=("gun", "lightning")):
        """Aim at a static point (crate) and fire weapons that would hit it."""
        slots = self.slots(me)
        mx, my, mz = self.muzzle(me)
        d = math.hypot(x - mx, y - my)
        ya = math.atan2(y - my, x - mx)
        pa = math.atan2(z - mz, max(d, 0.1))
        cur = me.get("aim") or [0.0, 0.0]
        cy, cp = _f(cur[0]), _f(cur[1])
        fire = [False, False]
        reload = False
        tol = max(0.006, math.atan2(1.0, max(d, 1.0)))
        if not self.wm.los(mx, my, mz, x, y, z):
            return [ya, pa], fire, reload
        for kind, (i, w) in slots.items():
            if kind not in use or i > 1:
                continue
            if kind == "gun":
                ammo = _f(w.get("ammo"), 1)
                if ammo <= 0 and _f(w.get("reload")) <= 0:
                    reload = True
                if ammo > 0 and _f(w.get("reload")) <= 0 and abs(wrap(ya - cy)) < tol and abs(pa - cp) < tol + 0.01:
                    fire[i] = True
            elif kind == "lightning":
                if d < self.stat(w, "lightning", "range", 45.0) * 0.9 and abs(wrap(ya - cy)) < 0.15 \
                        and _f(me.get("energy")) > 30:
                    fire[i] = True
        return [ya, pa], fire, reload
