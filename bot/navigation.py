"""Navigation: time-sliced A* on the WorldMap grid, path following, stuck recovery.

All work is bounded by a deadline (perf_counter seconds); an unfinished search resumes next tick.
"""
import array
import heapq
import math
import random
from collections import deque
from time import perf_counter

from worldmap import CELL

SQ2 = math.sqrt(2.0)
NB = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))


class Planner:
    """A* from a start cell to a goal cell; step() runs until a deadline."""

    def __init__(self, wm):
        self.wm = wm
        N = wm.N
        self.g = array.array("f", bytes(4 * N))
        self.par = array.array("i", bytes(4 * N))
        self.seen = array.array("I", bytes(4 * N))
        self.closed = array.array("I", bytes(4 * N))
        self.sid = 0
        self.status = "idle"
        self.heap = []
        self.path = None

    def start(self, s, goal, dangers=(), weight=1.6, max_expand=40000):
        self.sid += 1
        if self.sid > 0xFFFFFF00:
            N = self.wm.N
            self.seen = array.array("I", bytes(4 * N))
            self.closed = array.array("I", bytes(4 * N))
            self.sid = 1
        nx = self.wm.nx
        self.s, self.goal = s, goal
        self.gi, self.gj = goal % nx, goal // nx
        self.dangers = [(float(x), float(y), float(r) * r, float(r), float(w)) for (x, y, r, w) in dangers]
        self.hw = weight * CELL
        self.expanded = 0
        self.max_expand = max_expand
        self.g[s] = 0.0
        self.seen[s] = self.sid
        self.par[s] = -1
        self.heap = [(0.0, 0.0, s)]
        self.status = "running"
        self.path = None

    def step(self, deadline):
        if self.status != "running":
            return self.status
        wm = self.wm
        nx, ny = wm.nx, wm.ny
        solid, floor, plat, near1, near2 = wm.solid, wm.floor, wm.plat, wm.near1, wm.near2
        su = wm.step_up
        slim = wm.max_slope * CELL
        g, par, seen, closed = self.g, self.par, self.seen, self.closed
        sid, goal, gi, gj, hw = self.sid, self.goal, self.gi, self.gj, self.hw
        dangers = self.dangers
        ox, oy = wm.ox, wm.oy
        heap = self.heap
        pop, push = heapq.heappop, heapq.heappush
        k = 0
        while heap:
            k += 1
            if not (k & 15) and perf_counter() > deadline:
                return "running"
            _, gc, c = pop(heap)
            if closed[c] == sid:
                continue
            closed[c] = sid
            if c == goal:
                self._finish()
                return "found"
            self.expanded += 1
            if self.expanded > self.max_expand:
                self.status = "fail"
                return "fail"
            ci = c % nx
            cj = c // nx
            fc = floor[c]
            pc = plat[c]
            for di, dj, w in NB:
                bi = ci + di
                bj = cj + dj
                if bi < 0 or bj < 0 or bi >= nx or bj >= ny:
                    continue
                b = bj * nx + bi
                if solid[b] or closed[b] == sid:
                    continue
                if di and dj and (solid[c + di] or solid[c + dj * nx]):
                    continue
                rise = floor[b] - fc
                if rise > su and (pc or plat[b] or rise > slim * w):
                    continue
                cost = w * CELL * (3.5 if near1[b] else (1.4 if near2[b] else 1.0))
                if dangers:
                    bx = ox + (bi + 0.5) * CELL
                    by = oy + (bj + 0.5) * CELL
                    for (x, y, r2, r, wt) in dangers:
                        ddx = bx - x
                        ddy = by - y
                        d2 = ddx * ddx + ddy * ddy
                        if d2 < r2:
                            cost += wt * w * CELL * (1.0 - math.sqrt(d2) / r)
                ng = gc + cost
                if seen[b] != sid or ng < g[b]:
                    seen[b] = sid
                    g[b] = ng
                    par[b] = c
                    dx = bi - gi
                    dy = bj - gj
                    if dx < 0:
                        dx = -dx
                    if dy < 0:
                        dy = -dy
                    h = hw * (dx + dy - 0.5857864 * (dx if dx < dy else dy))
                    push(heap, (ng + h, ng, b))
        self.status = "fail"
        return "fail"

    def _finish(self):
        out = []
        c = self.goal
        guard = 0
        while c != -1 and guard < 200000:
            out.append(c)
            c = self.par[c] if c != self.s else -1
            guard += 1
        out.reverse()
        self.path = out
        self.status = "found"
        self.heap = []


class Navigator:
    """Goal -> move vector. Plans with Planner, follows the path with line-of-walk lookahead."""

    def __init__(self, wm, seed=0):
        self.wm = wm
        self.planner = Planner(wm)
        self.rng = random.Random(seed)
        self.goal = None          # (x, y)
        self.goal_cell = -1
        self.path = []            # [(x, y)]
        self.idx = 0
        self.planning = False
        self.plan_fail_t = -99.0
        self.hist = deque(maxlen=16)
        self.stuck_count = 0
        self.unstick_until = -1.0
        self.unstick_dir = (0.0, 0.0)
        self.jet_until = -1.0
        self.last_plan_t = -99.0
        self.dangers = ()
        self.climb = False
        self.climb_t = -99.0
        self.jetting = False
        self._plan_pending = False
        self.mode = "idle"

    # ------------------------------------------------------------------ goals
    def set_goal(self, x, y, t, dangers=(), force=False):
        wm = self.wm
        if not force and self.goal is not None and math.hypot(x - self.goal[0], y - self.goal[1]) < 3.0:
            return
        gc = wm.cell(x, y)
        gc = wm.nearest_free(gc, 8) if gc >= 0 else -1
        if gc < 0:
            x = min(max(x, wm.bminx + 6), wm.bmaxx - 6)
            y = min(max(y, wm.bminy + 6), wm.bmaxy - 6)
            gc = wm.nearest_free(wm.cell(x, y), 12)
        self.goal = (x, y)
        self.goal_cell = gc
        self.dangers = tuple(dangers)
        self.path = []
        self.idx = 0
        self.planning = False
        self._plan_pending = True

    def clear(self):
        self.goal = None
        self.path = []
        self.planning = False

    def replan(self, dangers=None):
        if dangers is not None:
            self.dangers = tuple(dangers)
        self._plan_pending = True

    def has_path(self):
        return bool(self.path)

    def remaining(self, x, y):
        """Approximate remaining path length (m) from (x, y) to the goal."""
        if self.goal is None:
            return 0.0
        if not self.path:
            return math.hypot(self.goal[0] - x, self.goal[1] - y) * 1.3
        p = self.path
        i = min(self.idx, len(p) - 1)
        d = math.hypot(p[i][0] - x, p[i][1] - y)
        return d + (len(p) - 1 - i) * CELL * 1.1

    # ------------------------------------------------------------------ per tick
    def steer(self, t, x, y, z, grounded, energy, deadline, speed_hint=1.0):
        """Returns (mx, my, jump, boost_needed)."""
        wm = self.wm
        self.hist.append((t, x, y))
        if self.goal is None:
            self.mode = "idle"
            return 0.0, 0.0, False, False
        gx, gy = self.goal
        dist_goal = math.hypot(gx - x, gy - y)
        c = wm.cell(x, y)
        on_top = c >= 0 and wm.solid[c] and z > wm.floor[c] + 2.0
        sc = wm.nearest_free(c, 4) if c >= 0 else -1

        # climbing: back to normal planning once we stand in the goal's area (or after a while)
        if (self.climb and grounded and not on_top and sc >= 0 and self.goal_cell >= 0
                and t - self.climb_t > 2.0
                and (wm.comp[sc] == wm.comp[self.goal_cell] or t - self.climb_t > 15.0)):
            self.climb = False
            self._plan_pending = True

        # start a plan if needed
        if getattr(self, "_plan_pending", False) and sc >= 0 and self.goal_cell >= 0:
            self._plan_pending = False
            self.path = []
            if wm.comp[sc] == wm.comp[self.goal_cell]:
                self.climb = False
                self.planner.start(sc, self.goal_cell, self.dangers)
                self.planning = True
                self.last_plan_t = t
            else:
                # goal in another walkable area: walk to our area's closest cell, then jet across
                proxy = -1
                if wm.comp_size[wm.comp[sc]] >= 30000:
                    proxy = wm.nearest_free(self.goal_cell, 30, comp=wm.comp[sc])
                if proxy >= 0 and proxy != sc:
                    self.climb = False
                    self.planner.start(sc, proxy, self.dangers)
                    self.planning = True
                    self.last_plan_t = t
                else:
                    self.climb = True   # enclosed area (courtyard, bowl, roof): go straight, jet over
                    self.climb_t = t
                    self.planning = False
        if self.planning:
            st = self.planner.step(deadline)
            if st == "found":
                self.planning = False
                self.path = [wm.center(cc) for cc in self.planner.path]
                self.idx = 0
            elif st == "fail":
                self.planning = False
                self.plan_fail_t = t
                self.path = []
                self.climb = True
                self.climb_t = t + 4.0   # retry a normal plan a few seconds later

        if dist_goal < 1.6:
            self.mode = "arrived"
            return 0.0, 0.0, False, False

        jump = False
        mx, my = gx - x, gy - y
        if self.path and not on_top:
            p = self.path
            n = len(p)
            # advance past reached waypoints
            while self.idx < n - 1 and math.hypot(p[self.idx][0] - x, p[self.idx][1] - y) < 2.8:
                self.idx += 1
            # end of a path that stops short of the goal (other area): jet the rest of the way
            if self.idx >= n - 1 and math.hypot(p[-1][0] - x, p[-1][1] - y) < 3.0 and dist_goal > 4.0:
                self.path = []
                self.climb = True
                self.climb_t = t
            # off the path? find the closest upcoming waypoint, replan if far
            best_i, best_d = self.idx, 1e9
            for i in range(self.idx, min(n, self.idx + 12)):
                d = math.hypot(p[i][0] - x, p[i][1] - y)
                if d < best_d:
                    best_d, best_i = d, i
            self.idx = best_i
            if best_d > 9.0 and t - self.last_plan_t > 1.0:
                self._plan_pending = True
            tgt = p[self.idx]
            for k in range(min(n - 1, self.idx + 9), self.idx, -1):
                if wm.walk_line(x, y, p[k][0], p[k][1]):
                    tgt = p[k]
                    break
            mx, my = tgt[0] - x, tgt[1] - y
            self.mode = "path"
        else:
            # no path (yet): head toward the goal, bending around walls
            self.mode = "climb" if self.climb else ("plan" if self.planning else "direct")
            if not self.climb and not on_top:
                mx, my = self._local_dir(x, y, mx, my)

        # stuck detection: wanted to move but hardly moved over ~1.4 s
        if len(self.hist) >= 14:
            t0, x0, y0 = self.hist[0]
            if t - t0 > 1.2 and math.hypot(x - x0, y - y0) < 1.5 and dist_goal > 3.0:
                self.stuck_count += 1
                self.hist.clear()
                a = self.rng.uniform(-math.pi, math.pi)
                self.unstick_dir = (math.cos(a), math.sin(a))
                self.unstick_until = t + 0.8
                self.jet_until = t + 1.2
                if self.stuck_count % 2 == 0:
                    self._plan_pending = True
        if t < self.unstick_until and self.stuck_count % 3 == 2:
            mx, my = self.unstick_dir
        boost = False
        if self.climb or t < self.jet_until:
            # a full jet climbs ~60 m: start only with enough energy, then burn it all in one go
            if grounded:
                if energy >= (80.0 if self.climb else 25.0):
                    jump = True
                    self.jetting = True
            elif self.jetting and energy > 3.0:
                boost = True
            else:
                self.jetting = False
        n = math.hypot(mx, my)
        if n < 1e-6:
            return 0.0, 0.0, jump, boost
        s = min(1.0, speed_hint, dist_goal / 3.0 + 0.25)
        return mx / n * s, my / n * s, jump, boost

    def _local_dir(self, x, y, mx, my):
        """Pick the walkable direction closest to (mx, my), probing 6 m ahead."""
        wm = self.wm
        a0 = math.atan2(my, mx)
        for da in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.2, -2.2):
            a = a0 + da
            ex, ey = x + 6.0 * math.cos(a), y + 6.0 * math.sin(a)
            if wm.walk_line(x, y, ex, ey, avoid_near=False):
                return math.cos(a), math.sin(a)
        return mx, my
