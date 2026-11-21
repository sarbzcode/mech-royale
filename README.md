# mech-royale

A survival-first bot for the Builders Buildathon **Mech Royale** track (`mechfight/1` protocol).

Placement is what scores (16 − place), so the bot plays to outlast: it keeps inside the shrinking
zone, waits at covered spots, avoids crowds and NPC drones, and fights only when attacked or when a
weak, isolated enemy is in reach.

## Layout

| Path | What it is |
| --- | --- |
| `Dockerfile` | The image the scorer builds: `python:3.13-slim` (pinned by digest), standard library only. |
| `bot/main.py` | Entry point: loads the map, builds the brain, provides a safe fallback action. |
| `bot/mechfight.py` | The kit's wrapper, extended with a per-tick deadline and a watchdog that always answers in time. |
| `bot/worldmap.py` | Map model: terrain, 2 m nav grid (walls, steps, slopes, bridge decks), line of sight, cover. |
| `bot/navigation.py` | Time-sliced A* and path following with stuck recovery and jet-climbing. |
| `bot/perception.py` | Tracks of mechs and drones, sound-based danger memory, damage events. |
| `bot/combat.py` | Lead and ballistic aiming; gun, lightning, laser and mortar trigger logic. |
| `bot/strategy.py` | The decisions: zone timing, hiding spots, flee/fight/hunt, loot, supply drops, upgrades. |
| `bot/arena/` | Copy of the map, used only if `/arena` is missing. |
| `tools/replay.py` | Times the bot on the recorded stream plus fuzzed observations (in-process or over stdin/stdout). |
| `tools/sim.py` | Approximate local match simulator on the real map (`--squad`/`--omni` for harder house bots). |
| `tools/render_map.py` | Renders the nav grid to a PNG. |

## Commands

```sh
docker build -t mech-bot .
python3 kit/mechfight-kit/check/check.py -- python3 -u bot/main.py
python3 kit/mechfight-kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 \
  --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/kit/mechfight-kit/arena:/arena:ro" mech-bot
python3 tools/replay.py inproc --fuzz 3000
python3 tools/sim.py --matches 8 --squad
```

`kit/` is the extracted `mechfight-kit.tar.gz` (not committed).
