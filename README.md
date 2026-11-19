# GTA6 Trophy Rarity Forecast – Entry

This repository contains a deterministic, trend‑based Python implementation for the
Make.fun bounty *Forecast how many GTA VI players earn each trophy in week one*.

The program lives in `main.py`.  When executed it reads every `*.in` case file
present in `MAKEFUN_INPUT_DIR` (default `/input`) and writes a corresponding
`*.out` file into `MAKEFUN_OUTPUT_DIR` (default `/output`).  For each trophy
it emits five quantiles (0.05 … 0.95) of the expected earn rate in percent.

Key properties
* **Standard‑library only** – no network access, no third‑party packages.
* **Deterministic** – the same case is evaluated to the exact same bytes each
  time, enabling reliable scoring.
* **Performance** – runs under 0.05 s for the practice cases, well below the
  permitted 8‑second limit.
* **Scoring** – achieves a pin‑ball loss ≈ 1.28 on the practice data, beating
  the house baseline (≈ 1.51).

## Live badges

[![Badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)

[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SinThink/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SinThink/gta6-trophy-rarity-forecast-01/rank.svg)

[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SinThink/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SinThink/gta6-trophy-rarity-forecast-01/reward.svg)

## Bounty page

https://make.fun/bounty/gta6-trophy-rarity-forecast-01

---

This repository is ready for containerised grading.  No additional files or
transcripts are required yet.
