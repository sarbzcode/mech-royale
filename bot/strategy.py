"""Strategy: survival first. Decide where to be, when to fight, what to pick up.

Modes (one per tick):
  zone   - must move to stay ahead of the shrinking circle
  flee   - hurt / outnumbered / weakest in sight: break line of sight, regen
  fight  - attacked or enemy very close: face it, strafe, shoot back
  hunt   - a weak, isolated enemy in reach and we are healthy: finish it
  drop   - safe supply drop nearby: capture it
  loot   - safe weapon / cube / crate nearby
  hide   - sit at a covered spot inside the next circle, scan, stay quiet
"""
import math
import random
from time import perf_counter

from combat import Combat, wrap
from navigation import Navigator
from perception import Perception
from worldmap import CELL

UPGRADE_PRIORITY = [
    "plating", "armor", "nanites", "servos", "sensors", "gyros",
    "lightning_efficiency", "gun_damage", "gun_rate", "laser_charge", "laser_damage",
    "lightning_range", "lightning_chain", "jets", "capacitor", "reactor",
    "gun_mag", "gun_velocity", "mortar_reload", "mortar_velocity", "mortar_splash", "laser_pierce",
]
WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 1, "gun": 0}


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class ZoneState:
    def __init__(self, z, t):
        z = z or {}
        c = z.get("center") or [0.0, 0.0]
        nc = z.get("next_center") or c
        self.c0 = (_f(c[0]), _f(c[1]))
        self.r0 = _f(z.get("radius"), 600.0)
        self.c1 = (_f(nc[0]), _f(nc[1]))
        self.r1 = _f(z.get("next_radius"), self.r0)
        self.s0 = _f(z.get("shrink_start"), 0.0)
        self.s1 = _f(z.get("shrink_end"), 0.0)
        self.dps = _f(z.get("dps"), 0.0)
        self.t = t
        self.key = (round(self.c1[0], 1), round(self.c1[1], 1), round(self.r1, 1))

    def circle(self, t):
        if t <= self.s0 or self.s1 <= self.s0:
            return (self.c0, self.r0) if t <= self.s0 else (self.c1, self.r1)
        if t >= self.s1:
            return self.c1, self.r1
        f = (t - self.s0) / (self.s1 - self.s0)
        return ((self.c0[0] + (self.c1[0] - self.c0[0]) * f, self.c0[1] + (self.c1[1] - self.c0[1]) * f),
                self.r0 + (self.r1 - self.r0) * f)

    def inside(self, x, y, t=None, margin=0.0):
        (cx, cy), r = self.circle(self.t if t is None else t)
        return math.hypot(x - cx, y - cy) < r - margin

    def time_safe(self, x, y, margin=3.0):
        """Seconds until (x, y) leaves the shrinking circle (1e9 if safe all phase)."""
        t = self.t
        (c0x, c0y), r0 = self.c0, self.r0
        (c1x, c1y), r1 = self.c1, self.r1
        if (x - c1x) ** 2 + (y - c1y) ** 2 <= (r1 - margin) ** 2 and r1 > margin:
            return 1e9
        s0, s1 = self.s0, self.s1
        span = s1 - s0
        tt = t if t > s0 else s0
        if t > s0 and not self.inside(x, y, t, margin):
            return 0.0
        if t <= s0 and (x - c0x) ** 2 + (y - c0y) ** 2 >= (r0 - margin) ** 2:
            return 0.0
        if span <= 0:
            return max(0.0, s1 - t)
        while tt < s1:
            tt += 1.0
            f = (tt - s0) / span
            if f > 1.0:
                f = 1.0
            cx = c0x + (c1x - c0x) * f
            cy = c0y + (c1y - c0y) * f
            r = r0 + (r1 - r0) * f - margin
            if r <= 0 or (x - cx) ** 2 + (y - cy) ** 2 >= r * r:
                return max(0.0, tt - 1.0 - t)
        return max(0.0, s1 - t)


class Brain:
    def __init__(self, hello, wm):
        cfg = hello.get("config") or {}
        self.cfg = cfg
        self.wm = wm
        self.me_id = int(_f(hello.get("slot"), -1))
        seed = int(_f(hello.get("seed"), 0)) & 0xFFFFFFFF
        self.rng = random.Random(seed)
        self.nav = Navigator(wm, seed)
        self.per = Perception(cfg, self.me_id)
        self.cb = Combat(cfg, wm)
        mech = cfg.get("mech") or {}
        self.walk = _f(mech.get("walk_speed"), 11.7)
        self.boost_speed = _f(mech.get("boost_speed"), 19.5)
        self.regen_delay = _f(mech.get("regen_delay"), 5.0)
        loot = cfg.get("loot") or {}
        self.pick_r = _f(loot.get("pickup_radius"), 4.0)
        self.spot = None
        self.spot_score = -1e9
        self.spot_t = -99.0
        self.spot_key = None
        self.search = None
        self.open_cache = {}
        self.mode = "start"
        self.mode_t = 0.0
        self.strafe_sign = 1.0
        self.strafe_flip_t = 0.0
        self.scan_off = self.rng.uniform(0, 2 * math.pi)
        self.picked_offers = set()
        self.loot = None           # ("pickup"|"cube"|"crate", id, x, y, z)
        self.loot_t = -99.0
        self.drop_id = None
        self.target_id = None
        self.target_t = -99.0
        self.last_aim = [0.0, 0.0]
        self.zone_alarm = False
        self.ticks = 0
        self.last_me = {}
        self.loot_stand = None
        self.drop_goal = None
        self.drop_rush = False

    # ================================================================== main
    def act(self, obs, deadline):
        self.ticks += 1
        t = _f(obs.get("t"))
        me = obs.get("self") or {}
        per, wm, nav = self.per, self.wm, self.nav
        per.update(obs)
        zs = ZoneState(obs.get("zone"), t)
        pos = me.get("pos") or [0.0, 0.0, 0.0]
        x, y, z = _f(pos[0]), _f(pos[1]), _f(pos[2])
        hp, mhp = _f(me.get("hp"), 1.0), max(1.0, _f(me.get("max_hp"), 1400.0))
        hpf = hp / mhp
        energy = _f(me.get("energy"))
        grounded = bool(me.get("grounded", True))
        act = {}

        # ---------------------------------------------------------- situation
        vis = per.visible(0.25)
        known = per.known(10.0)
        dists = {tr.id: math.hypot(tr.x - x, tr.y - y) for tr in known}
        near_vis = [tr for tr in vis if dists[tr.id] < 200.0]
        nearest = min((dists[tr.id] for tr in vis), default=1e9)
        engaged = t - per.last_hurt_mech_t < 4.0
        recent_dmg = per.recent_damage(3.0)
        attackers = [per.enemies[i] for i, tt in per.attackers.items() if t - tt < 6.0 and i in per.enemies]
        weakest = bool(near_vis) and all(tr.hp > hp for tr in near_vis)
        crowd = sum(1 for tr in known if dists[tr.id] < 120.0)
        alive = int(_f(obs.get("alive"), 16))
        final = zs.r1 <= 1.0 and t > zs.s1 - 20.0
        late = zs.r1 <= 70.0 or alive <= 4

        # zone urgency
        safe_t = zs.time_safe(x, y, margin=4.0)
        dnext = math.hypot(x - zs.c1[0], y - zs.c1[1])
        in_next = dnext < max(0.0, zs.r1 - 3.0)
        travel = max(0.0, dnext - max(0.0, zs.r1 - 6.0)) * 1.35 / self.walk
        self.last_me = me
        zone_urgent = (not zs.inside(x, y, margin=1.0)) or (not in_next and safe_t < travel * 1.4 + 8.0)

        # ---------------------------------------------------------- combat target
        target = self._choose_target(t, x, y, hp, hpf, vis, dists, attackers, engaged, late, final, crowd)

        # ---------------------------------------------------------- mode
        flee = (not final) and (
            (hpf < 0.45 and (engaged or nearest < 130.0))
            or (weakest and engaged and len(near_vis) >= 1 and hpf < 0.8)
            or (crowd >= 3 and engaged)
            or (engaged and recent_dmg > 0.18 * mhp))
        if flee and target is not None and target.hp < 0.2 * target.max_hp and dists.get(target.id, 1e9) < 60.0 and hpf > 0.3:
            flee = False   # it's nearly dead and close: finish it
        if zone_urgent:
            mode = "zone"
        elif flee:
            mode = "flee"
        elif target is not None and (engaged or dists.get(target.id, 1e9) < 50.0 or final):
            mode = "fight"
        elif target is not None:
            mode = "hunt"
        else:
            mode = None
            if not late or alive > 2:
                if self._drop_choice(obs, t, x, y, zs, known, hpf):
                    mode = "drop"
                elif self._loot_choice(obs, me, t, x, y, zs, known):
                    mode = "loot"
            if mode is None:
                mode = "hide"
        if mode != self.mode:
            self.mode_t = t
            if mode in ("flee", "zone") and self.mode not in ("flee", "zone"):
                self.spot_t = -99.0       # re-pick a spot with the new priorities
        self.mode = mode

        # ---------------------------------------------------------- where to go
        nav_deadline = deadline - 0.003
        threat_pts = per.danger_points()
        goal = None
        speed = 1.0
        boost = False
        if mode == "zone":
            self._ensure_spot(t, x, y, zs, threat_pts, nav_deadline, flee=False, urgent=True)
            goal = self.spot or zs.c1
            boost = (not zs.inside(x, y, margin=1.0)) or safe_t < travel * 1.2 + 4.0
        elif mode == "flee":
            self._ensure_spot(t, x, y, zs, threat_pts, nav_deadline, flee=True)
            goal = self.spot or self._away_point(x, y, vis or known, zs)
            boost = energy > 25.0
        elif mode == "fight":
            goal = None
        elif mode == "hunt":
            goal = self._hunt_point(x, y, target, me)
        elif mode == "drop":
            goal = self.drop_goal
            boost = self.drop_rush and energy > 40.0
        elif mode == "loot":
            goal = (self.loot[2], self.loot[3]) if self.loot[0] != "crate" else self.loot_stand
        else:
            self._ensure_spot(t, x, y, zs, threat_pts, nav_deadline, flee=False)
            goal = self.spot or zs.c1
        if final and mode != "flee":
            (cx, cy), r = zs.circle(t + 2.0)
            goal = (cx, cy) if mode != "fight" else goal

        mx = my = 0.0
        jump = False
        if goal is not None:
            nav.set_goal(goal[0], goal[1], t, threat_pts if mode in ("hide", "flee", "zone", "loot") else ())
            mx, my, jump, nboost = nav.steer(t, x, y, z, grounded, energy, nav_deadline, speed)
            boost = boost or nboost
        else:
            nav.hist.append((t, x, y))

        # fight movement: strafe around the target, drift toward cover
        if mode == "fight" and target is not None:
            mx, my, boost = self._fight_move(t, x, y, z, grounded, energy, target, zs, threat_pts, nav_deadline)

        # dodge incoming fire (all modes)
        ex, ey = self._evade(x, y, z)
        if ex or ey:
            mx, my = mx + ex, my + ey
            n = math.hypot(mx, my)
            if n > 1.0:
                mx, my = mx / n, my / n
        # stay out of NPC drone reach unless already moving through
        mx, my = self._npc_push(x, y, mx, my)

        # ---------------------------------------------------------- aim + fire
        aim = None
        fire = [False, False]
        reload = False
        if target is not None and mode in ("fight", "hunt", "zone", "flee", "drop", "loot", "hide"):
            allow = ("gun", "lightning", "laser", "mortar")
            if mode == "flee":
                allow = ("gun", "lightning")
            shoot = mode in ("fight", "hunt") or final or (target.id in per.attackers and t - per.attackers[target.id] < 6.0)
            aim, f2, reload, info = self.cb.engage(t, me, target, allow=allow, aggressive=final)
            if shoot:
                fire = f2
        elif mode == "loot" and self.loot and self.loot[0] == "crate":
            l = self.loot
            aim, fire, reload = self.cb.shoot_point(me, l[2], l[3], l[4] + 1.2)
        if aim is None:
            aim = self._scan_aim(t, x, y, z, mx, my, known, zs)
            reload = self._idle_reload(me)
        act["move"] = [round(mx, 4), round(my, 4)]
        act["boost"] = bool(boost)
        if jump:
            act["jump"] = True
        act["aim"] = [round(_f(aim[0]), 4), round(max(-0.69, min(1.29, _f(aim[1]))), 4)]
        if fire[0] or fire[1]:
            act["fire"] = [bool(fire[0]), bool(fire[1])]
        if reload:
            act["reload"] = True
        self._pickups(obs, me, x, y, z, act)
        self._offer(obs, act)
        self.last_aim = act["aim"]
        return act

    # ================================================================== targets
    def _choose_target(self, t, x, y, hp, hpf, vis, dists, attackers, engaged, late, final, crowd):
        best, bs = None, -1e9
        att_ids = {a.id for a in attackers}
        for tr in vis:
            d = dists.get(tr.id, 1e9)
            frac = tr.hp / max(1.0, tr.max_hp)
            s = None
            if tr.id in att_ids and d < 230.0:
                s = 100.0 - d * 0.2 + (1.0 - frac) * 40.0
            elif d < 45.0:
                s = 80.0 - d * 0.5 + (1.0 - frac) * 30.0
            elif final and d < 150.0:
                s = 70.0 - d * 0.3
            elif (frac < 0.35 or tr.hp < 450.0) and d < 160.0 and hpf > 0.6:
                # weak target: only when isolated (nobody else close to it or us)
                others = sum(1 for o in vis if o.id != tr.id and o.dist2(tr.x, tr.y) < 90.0 ** 2)
                if others == 0 and crowd <= 2:
                    s = 50.0 - d * 0.2 + (1.0 - frac) * 30.0
            elif late and frac < 0.5 and d < 120.0 and hpf > 0.7 and crowd <= 1:
                s = 40.0 - d * 0.2
            if s is None:
                continue
            if tr.id == self.target_id:
                s += 15.0
            if s > bs:
                best, bs = tr, s
        if best is not None:
            self.target_id = best.id
            self.target_t = t
        else:
            self.target_id = None
        return best

    # ================================================================== zone / spots
    def _region(self, zs):
        r = zs.r1
        if r <= 1.0:
            return zs.c1[0], zs.c1[1], 0.0
        margin = min(30.0, max(4.0, 0.18 * r + 3.0))
        return zs.c1[0], zs.c1[1], max(0.0, r - margin)

    def _ensure_spot(self, t, x, y, zs, threats, deadline, flee=False, urgent=False):
        """Keep a good hiding spot inside the next circle; search incrementally."""
        cx, cy, R = self._region(zs)
        need = (self.spot is None or self.spot_key != zs.key
                or math.hypot(self.spot[0] - cx, self.spot[1] - cy) > R + 0.5
                or t - self.spot_t > (6.0 if flee else 14.0))
        if self.spot is not None and not flee:
            # is someone sitting near our spot?
            for tr in self.per.known(6.0):
                if tr.dist2(self.spot[0], self.spot[1]) < 60.0 ** 2:
                    need = need or t - self.spot_t > 3.0
                    break
        if need and self.search is None:
            self.search = {"stage": 0, "flee": flee, "key": zs.key, "t": t}
        if self.search is not None:
            self._search_step(t, x, y, zs, deadline)
        if self.spot is None and R <= 0.5:
            self.spot = zs.c1

    def _search_step(self, t, x, y, zs, deadline):
        """Incremental spot search: 0 sample, 1 cheap score (all), 2 danger + openness (top few)."""
        s = self.search
        wm = self.wm
        cx, cy, R = self._region(zs)
        flee = s["flee"]
        if s["stage"] == 0:
            if R < 3.0:
                c = wm.nearest_free(wm.cell(cx, cy), 10)
                self.spot = wm.center(c) if c >= 0 else (cx, cy)
                self.spot_key, self.spot_t, self.search = zs.key, t, None
                return
            mycell = wm.cell(x, y)
            fc = wm.nearest_free(mycell, 4) if mycell >= 0 else -1
            mycomp = wm.comp[fc] if fc >= 0 else -1
            cands = []
            n = 100
            reach = 160.0 if not flee else 120.0
            near_ok = math.hypot(x - cx, y - cy) < R + reach
            rng = self.rng
            for k in range(n):
                if k < n // 2 and near_ok:
                    a = rng.uniform(0, 2 * math.pi)
                    rr = reach * math.sqrt(rng.random())
                    px, py = x + rr * math.cos(a), y + rr * math.sin(a)
                    if (px - cx) ** 2 + (py - cy) ** 2 > R * R:
                        continue
                else:
                    a = rng.uniform(0, 2 * math.pi)
                    rr = R * math.sqrt(rng.random())
                    px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
                c = wm.cell(px, py)
                if c < 0 or wm.solid[c] or wm.near1[c]:
                    continue
                if mycomp >= 0 and wm.comp[c] != mycomp:
                    continue
                cands.append(c)
            if self.spot is not None:
                c = wm.cell(self.spot[0], self.spot[1])
                if c >= 0 and not wm.solid[c] and math.hypot(self.spot[0] - cx, self.spot[1] - cy) <= R + 0.5:
                    cands.append(c)
            per = self.per
            ens = []
            for tr in per.enemies.values():
                age = t - tr.t
                if age <= 20.0:
                    ex, ey = tr.pos_at(t, 3.0)
                    ens.append((ex, ey, 1.0 - age / 25.0, math.hypot(ex - x, ey - y)))
            s["ens"] = ens
            s["npcs"] = [(tr.hx, tr.hy) for tr in per.npcs.values()]
            s["cands"] = cands
            s["scored"] = []
            s["i"] = 0
            s["stage"] = 1
            return
        if s["stage"] == 1:
            cands = s["cands"]
            k = 0
            while s["i"] < len(cands):
                if k and not (k & 7) and perf_counter() > deadline:
                    return
                c = cands[s["i"]]
                s["i"] += 1
                k += 1
                s["scored"].append((self._cheap_score(c, x, y, zs, flee, s["ens"], s["npcs"]), c))
            s["scored"].sort(reverse=True)
            s["top"] = s["scored"][:14]
            s["i"] = 0
            s["best"] = None
            s["stage"] = 2
            return
        top = s["top"]
        spot_cell = wm.cell(self.spot[0], self.spot[1]) if self.spot is not None else -2
        k = 0
        while s["i"] < len(top):
            if k and perf_counter() > deadline:
                return
            k += 1
            sc, c = top[s["i"]]
            s["i"] += 1
            op = self.open_cache.get(c)
            if op is None:
                op = self.open_cache[c] = wm.openness(c)
                if len(self.open_cache) > 20000:
                    self.open_cache.clear()
            px, py = wm.center(c)
            total = sc - op / 10.0 - 1.5 * self.per.danger_at(px, py, 110.0)
            if c == spot_cell:
                total += 3.0
            if s["best"] is None or total > s["best"][0]:
                s["best"] = (total, c)
        if s["best"] is not None:
            self.spot = wm.center(s["best"][1])
            self.spot_score = s["best"][0]
        elif self.spot is None:
            self.spot = (cx, cy)
        self.spot_key = s["key"]
        self.spot_t = t
        self.search = None

    def _cheap_score(self, c, x, y, zs, flee, ens, npcs):
        wm = self.wm
        px, py = wm.center(c)
        sc = 1.6 * wm.wall_dirs(c, 4)
        if wm.canopy[c]:
            sc += 2.0
        if wm.near2[c]:
            sc += 0.5
        d = math.hypot(px - x, py - y)
        sc -= d / (45.0 if flee else 30.0)
        cx, cy, R = self._region(zs)
        if R > 1.0:
            rel = math.hypot(px - cx, py - cy) / R
            sc -= 6.0 * max(0.0, rel - 0.8)
            if zs.r1 >= 140.0:
                sc -= 2.0 * max(0.0, 0.3 - rel) / 0.3     # center is where everyone converges
        tw = 3.0 if flee else 1.0
        for ex, ey, fresh, de in ens:
            dx, dy = px - ex, py - ey
            d2 = dx * dx + dy * dy
            if d2 < 28900.0:
                sc -= tw * (170.0 - math.sqrt(d2)) / 14.0 * fresh
                # don't run past an enemy to get there
                if flee and de < d and d2 < d * d:
                    sc -= 4.0
        for hx, hy in npcs:
            d2 = (px - hx) ** 2 + (py - hy) ** 2
            if d2 < 5625.0:
                sc -= (75.0 - math.sqrt(d2)) / 5.0
        return sc

    def _away_point(self, x, y, threats, zs):
        if not threats:
            return zs.c1
        ax = ay = 0.0
        for tr in threats:
            dx, dy = x - tr.x, y - tr.y
            d = max(5.0, math.hypot(dx, dy))
            ax += dx / d / d
            ay += dy / d / d
        n = math.hypot(ax, ay)
        if n < 1e-9:
            return zs.c1
        px, py = x + ax / n * 60.0, y + ay / n * 60.0
        (cx, cy), r = zs.circle(self.per.t + 5.0)
        dd = math.hypot(px - cx, py - cy)
        if dd > r * 0.85 and dd > 1e-6:
            k = r * 0.85 / dd
            px, py = cx + (px - cx) * k, cy + (py - cy) * k
        return px, py

    # ================================================================== fight movement
    def _fight_move(self, t, x, y, z, grounded, energy, tr, zs, threats, deadline):
        dx, dy = tr.x - x, tr.y - y
        d = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / d, dy / d
        if t > self.strafe_flip_t:
            self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
            self.strafe_flip_t = t + self.rng.uniform(0.7, 1.6)
        sx, sy = -uy * self.strafe_sign, ux * self.strafe_sign
        # preferred range by weapon: lightning wants close, else keep distance
        slots = self.cb.slots(self.last_me)
        want = 60.0
        if "lightning" in slots:
            want = 30.0
        radial = 0.0
        if d > want + 15.0:
            radial = 0.5
        elif d < want - 15.0:
            radial = -0.6
        mx, my = sx + ux * radial, sy + uy * radial
        # blend toward cover spot if we have one nearby
        if self.spot is not None:
            cdx, cdy = self.spot[0] - x, self.spot[1] - y
            cd = math.hypot(cdx, cdy)
            if 3.0 < cd < 60.0:
                mx += cdx / cd * 0.5
                my += cdy / cd * 0.5
        # keep inside the zone
        (cx, cy), r = zs.circle(t + 3.0)
        dc = math.hypot(x - cx, y - cy)
        if dc > r * 0.8 and dc > 1e-6:
            mx += (cx - x) / dc * 1.5
            my += (cy - y) / dc * 1.5
        # don't walk into walls
        n = math.hypot(mx, my)
        if n > 1e-6:
            mx, my = mx / n, my / n
            if not self.wm.walk_line(x, y, x + mx * 5.0, y + my * 5.0, avoid_near=False):
                mx, my = self.nav._local_dir(x, y, mx, my)
        self.nav.hist.append((t, x, y))
        boost = energy > 60.0 and d < 40.0 and "lightning" not in slots
        return mx, my, boost

    def _hunt_point(self, x, y, tr, me):
        slots = self.cb.slots(me)
        want = 35.0 if "lightning" in slots else 90.0
        dx, dy = x - tr.x, y - tr.y
        d = max(1.0, math.hypot(dx, dy))
        if d <= want:
            return (x, y)
        return tr.x + dx / d * want, tr.y + dy / d * want

    # ================================================================== evasion
    def _evade(self, x, y, z):
        ex = ey = 0.0
        cz = z + 4.5
        shells = 0
        for p in self.per.projectiles[:60]:
            try:
                kind = p.get("kind")
                px, py, pz = (float(v) for v in p.get("pos"))
                vx, vy, vz = (float(v) for v in p.get("vel"))
            except Exception:
                continue
            if kind == "mortar":
                if shells >= 6 or (px - x) ** 2 + (py - y) ** 2 > 250.0 ** 2:
                    continue
                shells += 1
                g = 22.0
                # where will it land?
                lx, ly, lz, vzz = px, py, pz, vz
                hit = None
                for _ in range(70):
                    lx += vx * 0.1
                    ly += vy * 0.1
                    lz += vzz * 0.1
                    vzz -= g * 0.1
                    if vzz < 0 and lz < self.wm.floor_at(lx, ly) + 0.5:
                        hit = (lx, ly)
                        break
                if hit is not None:
                    dd = math.hypot(x - hit[0], y - hit[1])
                    if dd < 13.0:
                        k = (13.0 - dd) / 13.0 * 2.0
                        if dd < 0.5:
                            ex += k
                        else:
                            ex += (x - hit[0]) / dd * k
                            ey += (y - hit[1]) / dd * k
                continue
            rx, ry, rz = x - px, y - py, cz - pz
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1.0:
                continue
            tca = (rx * vx + ry * vy + rz * vz) / v2
            if tca <= 0.0 or tca > 0.7:
                continue
            qx, qy = px + vx * tca - x, py + vy * tca - y
            miss = math.hypot(qx, qy)
            if miss < 3.5:
                vh = math.hypot(vx, vy)
                if vh < 1e-6:
                    continue
                nx_, ny_ = -vy / vh, vx / vh
                side = -1.0 if (qx * nx_ + qy * ny_) > 0 else 1.0
                ex += nx_ * side * 1.2
                ey += ny_ * side * 1.2
        # laser being charged at us
        for tr in self.per.visible(0.25):
            if tr.charging:
                b = math.atan2(y - tr.y, x - tr.x)
                if abs(wrap(tr.yaw - b)) < 0.12:
                    d = max(1.0, math.hypot(x - tr.x, y - tr.y))
                    ux, uy = (x - tr.x) / d, (y - tr.y) / d
                    s = 1.0 if (self.ticks // 12) % 2 else -1.0
                    ex += -uy * s * 1.3
                    ey += ux * s * 1.3
        return ex, ey

    def _npc_push(self, x, y, mx, my):
        px = py = 0.0
        t = self.per.t
        for tr in self.per.npcs.values():
            if t - tr.t > 8.0:
                continue
            dx, dy = x - tr.x, y - tr.y
            d = math.hypot(dx, dy)
            if 1e-6 < d < 55.0:
                k = (55.0 - d) / 55.0
                px += dx / d * k
                py += dy / d * k
        if px or py:
            mx, my = mx + px, my + py
            n = math.hypot(mx, my)
            if n > 1.0:
                mx, my = mx / n, my / n
        return mx, my

    # ================================================================== aim when idle
    def _scan_aim(self, t, x, y, z, mx, my, known, zs):
        per = self.per
        base = None
        best = None
        for tr in known:
            d2 = tr.dist2(x, y)
            if best is None or d2 < best[0]:
                best = (d2, tr)
        if best is not None:
            tr = best[1]
            ex, ey = tr.pos_at(t, 2.0)
            d = math.hypot(ex - x, ey - y)
            return [math.atan2(ey - y, ex - x), math.atan2(tr.z + 4.5 - (z + 7.0), max(d, 1.0))]
        if per.hurt_bearing is not None and t - per.hurt_bearing[1] < 6.0:
            return [per.hurt_bearing[0], 0.0]
        strongest = None
        for dx_, dy_, tt, w in per.dangers:
            if t - tt < 8.0:
                s = w * (1.0 - (t - tt) / 8.0)
                if strongest is None or s > strongest[0]:
                    strongest = (s, dx_, dy_)
        if strongest is not None:
            base = math.atan2(strongest[2] - y, strongest[1] - x)
            return [base + 0.35 * math.sin(t * 1.3 + self.scan_off), 0.0]
        if math.hypot(mx, my) > 0.3:
            base = math.atan2(my, mx)
            return [base + 0.6 * math.sin(t * 0.9 + self.scan_off), 0.0]
        # standing: sweep across the open side (toward the zone center works as a default)
        base = math.atan2(zs.c1[1] - y, zs.c1[0] - x) if math.hypot(zs.c1[0] - x, zs.c1[1] - y) > 5 else self.scan_off
        return [base + 1.4 * math.sin(t * 0.55 + self.scan_off), 0.0]

    def _idle_reload(self, me):
        for w in me.get("weapons") or []:
            if w and w.get("kind") == "gun":
                st = w.get("stats") or {}
                mag = _f(st.get("mag"), 12.0)
                if _f(w.get("reload")) <= 0 and _f(w.get("ammo"), mag) < mag * 0.6:
                    return True
        return False

    # ================================================================== loot / drops
    def _enemy_near(self, px, py, r, max_age=12.0):
        t = self.per.t
        for tr in self.per.enemies.values():
            if t - tr.t <= max_age and tr.dist2(px, py) < r * r:
                return True
        return False

    def _npc_near(self, px, py, r):
        for tr in self.per.npcs.values():
            if (tr.hx - px) ** 2 + (tr.hy - py) ** 2 < r * r or tr.dist2(px, py) < r * r:
                return True
        return False

    def _loot_choice(self, obs, me, t, x, y, zs, known):
        wm = self.wm
        slots = self.cb.slots(me)
        w1 = (me.get("weapons") or [None, None])
        s1 = w1[1].get("kind") if len(w1) > 1 and w1[1] else None
        best = None
        if self._enemy_near(x, y, 110.0, 10.0):
            self.loot = None
            return False
        for p in obs.get("pickups") or []:
            try:
                kind = p.get("kind")
                pid = int(p.get("id"))
                px, py, pz = (float(v) for v in p.get("pos"))
            except Exception:
                continue
            if kind in slots or kind not in WEAPON_RANK:
                continue
            if WEAPON_RANK[kind] <= WEAPON_RANK.get(s1, -1):
                continue
            if t - self.per.pickup_fail.get(pid, -99.0) < 6.0:
                continue
            if pz > wm.floor_at(px, py) + 2.5:
                continue
            d = math.hypot(px - x, py - y)
            if d > 110.0 or self._enemy_near(px, py, 100.0) or self._npc_near(px, py, 50.0):
                continue
            if zs.time_safe(px, py, 6.0) < d / self.walk + 10.0:
                continue
            sc = WEAPON_RANK[kind] * 40.0 - d
            if best is None or sc > best[0]:
                best = (sc, ("pickup", pid, px, py, pz))
        for cb in obs.get("cubes") or []:
            try:
                cid = int(cb.get("id"))
                px, py, pz = (float(v) for v in cb.get("pos"))
            except Exception:
                continue
            if pz > wm.floor_at(px, py) + 2.5:
                continue
            d = math.hypot(px - x, py - y)
            if d > 90.0 or self._enemy_near(px, py, 90.0) or self._npc_near(px, py, 50.0):
                continue
            if zs.time_safe(px, py, 6.0) < d / self.walk + 10.0:
                continue
            sc = 45.0 - d
            if best is None or sc > best[0]:
                best = (sc, ("cube", cid, px, py, pz))
        if best is None and t < 260.0 and self._enemy_near(x, y, 160.0, 15.0) is False:
            mx_, my_, mz_ = self.cb.muzzle(me)
            for cr in obs.get("crates") or []:
                try:
                    cid = int(cr.get("id"))
                    px, py, pz = (float(v) for v in cr.get("pos"))
                except Exception:
                    continue
                if pz > wm.floor_at(px, py) + 2.0:
                    continue
                d = math.hypot(px - x, py - y)
                if d > 70.0 or self._npc_near(px, py, 50.0):
                    continue
                if zs.time_safe(px, py, 8.0) < 30.0:
                    continue
                if not wm.los(mx_, my_, mz_, px, py, pz + 1.2):
                    continue
                sc = 20.0 - d
                if best is None or sc > best[0]:
                    best = (sc, ("crate", cid, px, py, pz))
        if best is None:
            keep = self.loot
            if keep is not None and keep[0] != "crate" and t - self.loot_t < 10.0                     and math.hypot(keep[2] - x, keep[3] - y) > 3.0 and not self._enemy_near(keep[2], keep[3], 100.0):
                return True
            self.loot = None
            return False
        self.loot = best[1]
        if self.loot[0] == "crate":
            # shoot from here if close enough, else walk a bit closer
            d = math.hypot(self.loot[2] - x, self.loot[3] - y)
            self.loot_stand = (x, y) if d < 45.0 else (x + (self.loot[2] - x) * 0.5, y + (self.loot[3] - y) * 0.5)
        self.loot_t = t
        return True

    def _drop_choice(self, obs, t, x, y, zs, known, hpf):
        best = None
        for dr in obs.get("drops") or []:
            try:
                did = dr.get("id")
                px, py = float(dr["pos"][0]), float(dr["pos"][1])
                lands = _f(dr.get("lands_in"))
                holder = dr.get("holder")
            except Exception:
                continue
            if holder is not None and holder != self.me_id:
                continue
            d = math.hypot(px - x, py - y)
            if d > 230.0:
                continue
            eta = d / self.walk
            if lands > eta + 12.0:
                continue
            if self._enemy_near(px, py, 120.0, 12.0) or self._npc_near(px, py, 55.0):
                continue
            if zs.time_safe(px, py, 10.0) < eta + 15.0:
                continue
            sc = -d
            if best is None or sc > best[0]:
                best = (sc, did, px, py, lands, eta)
        if best is None:
            self.drop_id = None
            return False
        _, did, px, py, lands, eta = best
        self.drop_id = did
        self.drop_goal = (px, py)
        self.drop_rush = lands < eta
        return True

    def _pickups(self, obs, me, x, y, z, act):
        """Take a better weapon if one is within reach right now."""
        self.last_me = me
        slots = self.cb.slots(me)
        ws = me.get("weapons") or [None, None]
        s1 = ws[1].get("kind") if len(ws) > 1 and ws[1] else None
        if s1 == "laser" and _f(ws[1].get("charge")) > 0:
            return
        best = None
        for p in obs.get("pickups") or []:
            try:
                kind = p.get("kind")
                pid = int(p.get("id"))
                px, py, pz = (float(v) for v in p.get("pos"))
            except Exception:
                continue
            if kind in slots or kind not in WEAPON_RANK or kind == "gun":
                continue
            if math.hypot(px - x, py - y) > self.pick_r - 0.4 or abs(pz - z) > 3.8:
                continue
            if WEAPON_RANK[kind] <= WEAPON_RANK.get(s1, -1):
                continue
            if t_fail_recent(self.per, pid):
                continue
            if best is None or WEAPON_RANK[kind] > WEAPON_RANK[best[1]]:
                best = (pid, kind)
        if best is not None:
            act["take"] = best[0]
            if s1 is not None:
                act["drop"] = 1
            if self.loot and self.loot[0] == "pickup" and self.loot[1] == best[0]:
                self.loot = None

    def _offer(self, obs, act):
        off = obs.get("offer")
        if not off:
            return
        oid = off.get("id")
        choices = off.get("choices") or []
        if not choices or oid in self.picked_offers:
            return
        best_i, best_r = 0, 1e9
        for i, ch in enumerate(choices[:3]):
            cid = ch.get("id") if isinstance(ch, dict) else None
            r = UPGRADE_PRIORITY.index(cid) if cid in UPGRADE_PRIORITY else 500 + i
            if r < best_r:
                best_i, best_r = i, r
        act["pick"] = best_i
        self.picked_offers.add(oid)


def t_fail_recent(per, pid):
    return per.t - per.pickup_fail.get(pid, -99.0) < 3.0
