"""Render the nav grid to a PNG for inspection: python tools/render_map.py <arena_dir> <out.png>"""
import os, struct, sys, time, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bot"))
from worldmap import WorldMap

def png(path, w, h, rows):
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

def main():
    t0 = time.perf_counter()
    wm = WorldMap(sys.argv[1], {})
    print(f"built in {time.perf_counter()-t0:.2f}s  grid {wm.nx}x{wm.ny}  solid {sum(wm.solid)}  plat {sum(wm.plat)}  comps {len(wm.comp_size)} largest {sorted(wm.comp_size)[-5:]}")
    big = max(range(len(wm.comp_size)), key=lambda k: wm.comp_size[k])
    rows = []
    for j in range(wm.ny - 1, -1, -1):
        r = bytearray()
        for i in range(wm.nx):
            c = j * wm.nx + i
            if wm.solid[c]:
                col = (60, 60, 60) if wm.tall[c] else (170, 60, 60)
            elif wm.plat[c]:
                col = (60, 90, 230)
            elif wm.comp[c] != big:
                col = (230, 200, 40)
            else:
                f = wm.floor[c]
                g = max(0, min(255, int(200 + f * 6)))
                col = (g, g, g)
                if wm.near1[c]: col = (g - 40, g - 40, g - 20)
                if wm.canopy[c]: col = (90, 170, 90)
            r += bytes(col)
        rows.append(r)
    png(sys.argv[2], wm.nx, wm.ny, rows)

main()
