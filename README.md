[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yi948383-arch/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yi948383-arch/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast (week one)

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01`: forecast every
Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) of the rate in
percent, from everything the house recorded before the deadline.

## Run

```sh
python3 main.py            # reads /input, writes /output
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness, no model
calls. One `<id>.out` per `<id>.in`, deterministic across runs, ~0.08 s a
case on this machine (limit is 8 s).

## Model

Exophase's rate is earners / players-tracked. A refresh that adds a
flood of new tracked players pushes every rate down at once, which a
fit on the rate alone reads as a downward trend. So the program fits
the two counts separately, both linear in log-time
(`ln(1 + hours since launch)`):

- `log(earners_i) ~ a_i + b_i * ln(1 + hours)` per trophy
- `log(tracked) ~ a_N + b_N * ln(1 + hours)` for the tracked population

and forecasts the rate as earners / tracked at the target. Log-time
already builds in saturation: growth slows as the launch recedes.

Three things keep the extrapolation honest:

- **Shrunk slopes.** Each trophy's slope is pulled toward the
  precision-weighted pooled slope (empirical Bayes), with the pull set
  by its own standard error and the spread of true slopes. A trophy
  whose history is flat or noisy borrows the field's slope; a trophy
  with a clean trend keeps its own.
- **Anchored level.** The forecast starts from the last observed
  earner count and tracked count; only the *change* to the target
  comes from the fitted slopes.
- **Damped change (0.5).** A launch-week trend decelerates faster
  than log-time implies (adoption and progression both saturate), so
  the full extrapolation overshoots. The change is halved.
- **Spread.** The logit SD combines the random-walk term the house
  baseline uses (0.5 * sqrt(horizon in days)), the slope's standard
  error times the extrapolation distance, the fit's residual SD, and a
  fixed refresh-shock term (Exophase refreshes hours apart and each
  one can move every rate at once), scaled by 0.8 and clipped to
  [0.35, 2.2]. The trophy at 100% in the last read is forecast to
  stay there, with a narrow spread.

Scoring is mean pinball loss on the logit scale, so the quantiles are
the normal quantiles of the forecast distribution above: honest
quantiles, not a point forecast with a guessed band.

## Validation

The kit's practice cases cannot validate a trend model: every read in
them is the same unchanged launch-night page (638 players tracked), so
any trend fit is flat and the program behaves like the house baseline
there (valid output, deterministic, ~0.08 s a case; the refresh that
follows the checkpoint is outside every model). Validation was
therefore done on synthetic launch-week cases: a cohort adoption +
trophy-progression simulator with Exophase refresh steps, three
adoption regimes (saturating, fast-growth, slow-progression), nine
final-like cases (checkpoint 2026-11-21T11:30Z, target
2026-11-26T00:00Z). Mean pinball loss on the logit scale:

| model | loss |
|---|---|
| this entry | **0.210** |
| house curve baseline | 0.274 |
| house flat baseline (starter) | 0.295 |

The entry beats both house baselines in all nine cases. The undamped
version scored 0.346 and the raw-spread version 0.254, which is what
fixed the damping at 0.5 and the spread scale at 0.8.

## Files

- `main.py` — the whole entry (standard library only)
- `README.md` — this file
- `LICENSE` — MIT
