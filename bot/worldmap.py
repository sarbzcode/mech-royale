"""Static map model: terrain heights, a 2 m navigation grid, line of sight, cover.

Built once at startup from v2/world.json + v2/terrain.i16 (protocol.md section 11).
Pure standard library; every per-tick query is O(1) or a short ray march.
"""
import array
import json
import math
import os
import sys
from collections import deque

CELL = 2.0          # nav grid resolution (m)
SIGHT_WALL = 9.5    # a wall this much above our floor hides a whole mech (center + top)


class WorldMap:
    def __init__(self, map_dir, mech_cfg=None):
        mech_cfg = mech_cfg or {}
        self.step_up = float(mech_cfg.get("step_up", 1.2))
        self.max_slope = math.tan(math.radians(min(float(mech_cfg.get("max_slope_deg", 50)), 80.0) - 4.0))
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            wj = json.load(f)
        t = wj["terrain"]
        self.tx0, self.ty0 = float(t["minX"]), float(t["minY"])
        self.tstep = float(t.get("step", 1.0))
        self.tnx, self.tny = int(t["nx"]), int(t["ny"])
        self.tscale = float(t.get("scale", 0.01))
        terr = array.array("h")
        with open(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), "rb") as f:
            terr.frombytes(f.read())
        if sys.byteorder == "big":
            terr.byteswap()
        if len(terr) < self.tnx * self.tny:
            raise ValueError("terrain file too short")
        self.terr = terr
        b = wj.get("bounds") or {}
        self.bminx = float(b.get("minX", self.tx0))
        self.bmaxx = float(b.get("maxX", self.tx0 + (self.tnx - 1) * self.tstep))
        self.bminy = float(b.get("minY", self.ty0))
        self.bmaxy = float(b.get("maxY", self.ty0 + (self.tny - 1) * self.tstep))

        # nav grid
        self.ox, self.oy = self.tx0, self.ty0
        self.nx = int((self.tnx - 1) * self.tstep / CELL)
        self.ny = int((self.tny - 1) * self.tstep / CELL)
        nx, ny = self.nx, self.ny
        N = nx * ny
        self.N = N
        floor = array.array("f", bytes(4 * N))
        fast = abs(self.tstep - 1.0) < 1e-9 and abs(CELL - 2.0) < 1e-9
        sc = self.tscale
        for j in range(ny):
            base = j * nx
            if fast:
                k = (2 * j + 1) * self.tnx + 1
                floor[base:base + nx] = array.array("f", [v * sc for v in terr[k:k + 2 * nx:2]])
            else:
                y = self.oy + (j + 0.5) * CELL
                for i in range(nx):
                    floor[base + i] = self.ground(self.ox + (i + 0.5) * CELL, y)
        terrain_floor = array.array("f", floor)
        solid = bytearray(N)      # 1 = cannot stand here (wall, steep, map edge)
        plat = bytearray(N)       # 1 = floor is a prism top (bridge deck / low platform)
        top = array.array("f", floor)   # highest solid surface (for line of sight)
        maxtop = {}                       # cell -> highest building/stadium top over it

        self._steep(floor, solid)
        for p in wj.get("prisms", []):
            try:
                self._add_prism(p, floor, terrain_floor, solid, plat, top, maxtop)
            except Exception:
                continue
        su = self.step_up
        for c, ht in maxtop.items():
            if ht > floor[c] + su:
                solid[c] = 1
                if ht > top[c]:
                    top[c] = ht
            elif ht > floor[c]:
                floor[c] = ht
                if ht > top[c]:
                    top[c] = ht
                plat[c] = 1
        # map edge is a wall: keep a radius of margin
        m = 3.0
        for j in range(ny):
            y = self.oy + (j + 0.5) * CELL
            rowedge = y < self.bminy + m or y > self.bmaxy - m
            base = j * nx
            for i in range(nx):
                if rowedge:
                    solid[base + i] = 1
                else:
                    x = self.ox + (i + 0.5) * CELL
                    if x < self.bminx + m or x > self.bmaxx - m:
                        solid[base + i] = 1
        self.floor, self.solid, self.plat, self.top = floor, solid, plat, top

        # wall proximity rings (Chebyshev distance in cells) via big-int row dilation
        self.near1 = self._dilate(solid, 1)
        self.near2 = self._dilate(solid, 2)
        # tall walls (hide a mech) for cover scoring
        tall = bytearray(N)
        for c in range(N):
            if solid[c] and top[c] > floor[c] + 6.0:
                tall[c] = 1
        self.tall = tall
        self.canopy = self._canopy(wj.get("trees") or {})
        self.comp, self.comp_size = self._components()

    # ------------------------------------------------------------------ terrain
    def ground(self, x, y):
        """Bilinear terrain height (m) at world (x, y)."""
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        nx, ny = self.tnx, self.tny
        if fx < 0:
            fx = 0.0
        elif fx > nx - 1.001:
            fx = nx - 1.001
        if fy < 0:
            fy = 0.0
        elif fy > ny - 1.001:
            fy = ny - 1.001
        i = int(fx)
        j = int(fy)
        ax = fx - i
        ay = fy - j
        t = self.terr
        k = j * nx + i
        h00 = t[k]
        h10 = t[k + 1]
        h01 = t[k + nx]
        h11 = t[k + nx + 1]
        return ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay) * self.tscale

    def _steep(self, floor, solid):
        """Mark cells on terrain steeper than the walkable slope (2 m and 1 m differences)."""
        nx, ny = self.nx, self.ny
        lim = self.max_slope * CELL
        for j in range(ny):
            base = j * nx
            row = floor[base:base + nx]
            bad = [i + 1 for i, (a, b) in enumerate(zip(row, row[1:])) if a - b > lim or b - a > lim]
            for i in bad:
                solid[base + i] = 1
                solid[base + i - 1] = 1
            if j + 1 < ny:
                up = floor[base + nx:base + 2 * nx]
                for i, (a, b) in enumerate(zip(row, up)):
                    if a - b > lim or b - a > lim:
                        solid[base + i] = 1
                        solid[base + nx + i] = 1
        if abs(self.tstep - 1.0) > 1e-9 or abs(CELL - 2.0) > 1e-9:
            return
        t, tnx, sc = self.terr, self.tnx, self.tscale
        lim1 = self.max_slope * self.tstep / sc
        for j in range(ny):
            cj = 2 * j + 1
            if cj + 1 >= self.tny:
                continue
            k = cj * tnx
            c = t[k + 1:k + 1 + 2 * nx:2]
            lft = t[k:k + 2 * nx:2]
            rgt = t[k + 2:k + 2 + 2 * nx:2]
            up = t[k + tnx + 1:k + tnx + 1 + 2 * nx:2]
            dn = t[k - tnx + 1:k - tnx + 1 + 2 * nx:2]
            base = j * nx
            for i, (h, a, b, u, d) in enumerate(zip(c, lft, rgt, up, dn)):
                if max(a, b, u, d) - h > lim1 or h - min(a, b, u, d) > lim1:
                    solid[base + i] = 1

    # ------------------------------------------------------------------ prisms
    def _cells_in(self, ring, holes):
        """Cell indices whose centers are inside ring and outside holes (even-odd scanline)."""
        edges = []
        polys = [ring] + [h for h in (holes or []) if h]
        for poly in polys:
            n = len(poly) // 2
            for k in range(n):
                x1, y1 = poly[2 * k], poly[2 * k + 1]
                k2 = (k + 1) % n
                x2, y2 = poly[2 * k2], poly[2 * k2 + 1]
                if y1 != y2:
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    edges.append((y1, y2, x1, (x2 - x1) / (y2 - y1)))
        if not edges:
            return []
        ys = ring[1::2]
        ox, oy, nx, ny = self.ox, self.oy, self.nx, self.ny
        j0 = max(0, int(math.floor((min(ys) - oy) / CELL - 0.5)))
        j1 = min(ny - 1, int(math.ceil((max(ys) - oy) / CELL - 0.5)))
        out = []
        for j in range(j0, j1 + 1):
            y = oy + (j + 0.5) * CELL
            xs = [x1 + (y - y1) * s for (y1, y2, x1, s) in edges if y1 <= y < y2]
            if len(xs) < 2:
                continue
            xs.sort()
            base = j * nx
            for k in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[k] - ox) / CELL - 0.5)))
                i1 = min(nx - 1, int(math.floor((xs[k + 1] - ox) / CELL - 0.5)))
                if i1 >= i0:
                    out.extend(range(base + i0, base + i1 + 1))
        return out

    def _add_prism(self, p, floor, terrain_floor, solid, plat, top, maxtop):
        ring = p.get("ring") or []
        if len(ring) < 6:
            return
        kind = p.get("kind", "building")
        pb, pt = float(p.get("base", 0.0)), float(p.get("top", 0.0))
        cells = self._cells_in(ring, p.get("holes") or [])
        if not cells:
            return
        if kind == "bridge":
            # a deck if its top is near the ground at its ends, else an elevated walkway
            ends = [self.ground(ring[2 * k], ring[2 * k + 1]) for k in range(len(ring) // 2)]
            if pt <= max(ends) + 1.6:
                for c in cells:
                    if pt > floor[c] - 0.5:
                        floor[c] = max(floor[c], pt)
                        plat[c] = 1
                        if pt > top[c]:
                            top[c] = pt
                        solid[c] = 0   # deck overrides steep river banks beneath it
            else:
                for c in cells:
                    if pb - terrain_floor[c] < 9.8:
                        solid[c] = 1
                    if pt > top[c] and pb - terrain_floor[c] < 9.8:
                        top[c] = pt
            return
        for c in cells:
            if pt > maxtop.get(c, -1e9):
                maxtop[c] = pt

    # ------------------------------------------------------------------ derived layers
    def _dilate(self, mask, r):
        """Chebyshev dilation by r cells using one big int per row (8 bits per cell)."""
        nx, ny = self.nx, self.ny
        rows = [int.from_bytes(bytes(mask[j * nx:(j + 1) * nx]), "big") for j in range(ny)]
        full = (1 << (8 * nx)) - 1
        h = []
        for v in rows:
            acc = v
            for s in range(1, r + 1):
                acc |= (v << (8 * s)) | (v >> (8 * s))
            h.append(acc & full)
        out = bytearray(self.N)
        for j in range(ny):
            acc = 0
            for dj in range(-r, r + 1):
                jj = j + dj
                if 0 <= jj < ny:
                    acc |= h[jj]
            out[j * nx:(j + 1) * nx] = acc.to_bytes(nx, "big")
        return out

    def _canopy(self, trees):
        can = bytearray(self.N)
        fields = trees.get("fields") or []
        data = trees.get("data") or []
        n = len(fields)
        if not n:
            return can
        try:
            ix = fields.index("crownX") if "crownX" in fields else fields.index("x")
            iy = fields.index("crownY") if "crownY" in fields else fields.index("y")
            iz, ih = fields.index("z"), fields.index("height")
            ir, ib = fields.index("crownR"), fields.index("crownBase")
        except ValueError:
            return can
        nx, ny = self.nx, self.ny
        for k in range(len(data) // n):
            row = data[k * n:(k + 1) * n]
            try:
                x, y, z = float(row[ix]), float(row[iy]), float(row[iz])
                h, r, cb = float(row[ih]), float(row[ir]), float(row[ib])
            except (TypeError, ValueError):
                continue
            # crown must cover a mech's body (center 4.5 m .. top 9 m) to hide it
            if r < 1.5 or cb > 4.5 or h < 8.0:
                continue
            ci, cj = int((x - self.ox) / CELL), int((y - self.oy) / CELL)
            rc = int(r / CELL) + 1
            for j in range(max(0, cj - rc), min(ny, cj + rc + 1)):
                for i in range(max(0, ci - rc), min(nx, ci + rc + 1)):
                    dx = self.ox + (i + 0.5) * CELL - x
                    dy = self.oy + (j + 0.5) * CELL - y
                    if dx * dx + dy * dy <= r * r:
                        c = j * nx + i
                        if can[c] < 250:
                            can[c] += 1
        return can

    def edge_ok(self, a, b, diag):
        """Can a mech walk from cell a to adjacent cell b?"""
        if self.solid[b]:
            return False
        rise = self.floor[b] - self.floor[a]
        if rise <= self.step_up:
            return True
        if self.plat[a] or self.plat[b]:
            return False
        return rise <= self.max_slope * CELL * (1.414 if diag else 1.0)

    def _components(self):
        nx, ny, N = self.nx, self.ny, self.N
        comp = array.array("i", [-1]) * N
        sizes = []
        solid, floor, plat = self.solid, self.floor, self.plat
        su = self.step_up
        slim = self.max_slope * CELL
        cid = 0
        for s in range(N):
            if solid[s] or comp[s] >= 0:
                continue
            comp[s] = cid
            q = deque([s])
            size = 0
            while q:
                c = q.popleft()
                size += 1
                i = c % nx
                fc = floor[c]
                pc = plat[c]
                for d in (1, -1, nx, -nx):
                    b = c + d
                    if b < 0 or b >= N or solid[b] or comp[b] >= 0:
                        continue
                    if d == 1 and i == nx - 1 or d == -1 and i == 0:
                        continue
                    dz = abs(floor[b] - fc)
                    if dz > su and (pc or plat[b] or dz > slim):
                        continue
                    comp[b] = cid
                    q.append(b)
            sizes.append(size)
            cid += 1
        return comp, sizes

    # ------------------------------------------------------------------ queries
    def cell(self, x, y):
        i = int((x - self.ox) * 0.5)
        j = int((y - self.oy) * 0.5)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return -1

    def center(self, c):
        return (self.ox + (c % self.nx + 0.5) * CELL, self.oy + (c // self.nx + 0.5) * CELL)

    def floor_at(self, x, y):
        c = self.cell(x, y)
        return self.floor[c] if c >= 0 else self.ground(x, y)

    def free(self, c):
        return 0 <= c < self.N and not self.solid[c]

    def nearest_free(self, c, maxr=6, comp=None):
        """Closest non-solid cell to c within maxr rings (optionally in component comp)."""
        if c < 0:
            return -1
        nx, ny = self.nx, self.ny
        ci, cj = c % nx, c // nx
        if not self.solid[c] and (comp is None or self.comp[c] == comp):
            return c
        best, bd = -1, 1e9
        for r in range(1, maxr + 1):
            for j in range(cj - r, cj + r + 1):
                if j < 0 or j >= ny:
                    continue
                for i in (range(ci - r, ci + r + 1) if j in (cj - r, cj + r) else (ci - r, ci + r)):
                    if 0 <= i < nx:
                        b = j * nx + i
                        if not self.solid[b] and (comp is None or self.comp[b] == comp):
                            d = (i - ci) ** 2 + (j - cj) ** 2
                            if d < bd:
                                bd, best = d, b
            if best >= 0:
                return best
        return -1

    def los(self, x0, y0, z0, x1, y1, z1, step=1.5):
        """True if the straight segment clears buildings and terrain (2 m grid heights)."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        dist = math.hypot(dx, dy)
        n = int(dist / step) + 1
        if n <= 1:
            return True
        top, nx, ny, ox, oy = self.top, self.nx, self.ny, self.ox, self.oy
        inv = 1.0 / n
        for k in range(1, n):
            f = k * inv
            i = int((x0 + dx * f - ox) * 0.5)
            j = int((y0 + dy * f - oy) * 0.5)
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            if top[j * nx + i] > z0 + dz * f:
                return False
        return True

    def walk_line(self, x0, y0, x1, y1, avoid_near=True):
        """True if a mech can walk straight from (x0,y0) to (x1,y1) on the grid."""
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        n = int(dist / 1.0) + 1
        prev = self.cell(x0, y0)
        if prev < 0:
            return False
        solid, near = self.solid, self.near1
        for k in range(1, n + 1):
            f = k / n
            c = self.cell(x0 + dx * f, y0 + dy * f)
            if c < 0 or solid[c]:
                return False
            if c != prev:
                if avoid_near and near[c] and not near[prev]:
                    return False
                diag = (c - prev) not in (1, -1, self.nx, -self.nx)
                if not self.edge_ok(prev, c, diag):
                    return False
                prev = c
        return True

    def openness(self, c, rays=16, maxd=120.0):
        """Mean open sight distance (m) from cell c at mech height: lower = better hidden."""
        if c < 0:
            return maxd
        x0, y0 = self.center(c)
        f0 = self.floor[c] + SIGHT_WALL
        top, nx, ny, ox, oy = self.top, self.nx, self.ny, self.ox, self.oy
        total = 0.0
        for k in range(rays):
            a = 2 * math.pi * k / rays
            ca, sa = math.cos(a) * CELL, math.sin(a) * CELL
            d = maxd
            x, y = x0, y0
            for s in range(1, int(maxd / CELL) + 1):
                x += ca
                y += sa
                i = int((x - ox) * 0.5)
                j = int((y - oy) * 0.5)
                if i < 0 or j < 0 or i >= nx or j >= ny or top[j * nx + i] > f0:
                    d = s * CELL
                    break
            total += d
        return total / rays

    def wall_dirs(self, c, reach=4):
        """How many of 8 directions have a tall wall within `reach` cells (cover sides)."""
        if c < 0:
            return 0
        nx, ny = self.nx, self.ny
        ci, cj = c % nx, c // nx
        tall = self.tall
        n = 0
        for di, dj in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            for s in range(1, reach + 1):
                i, j = ci + di * s, cj + dj * s
                if i < 0 or j < 0 or i >= nx or j >= ny:
                    break
                if tall[j * nx + i]:
                    n += 1
                    break
        return n
