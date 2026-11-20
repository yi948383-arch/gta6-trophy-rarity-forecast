"""gta6-trophy-rarity-forecast-01 entry: forecast each GTA VI PS5 trophy's
Exophase earn rate at the case's target, as five quantiles (5/25/50/75/95%).

Model (counts-based, per trophy, fitted at run time from the case's own
history -- nothing is hard-coded from any outside data):

  * Exophase's rate is earners / players-tracked.  A refresh that adds a
    flood of new tracked players pushes every rate down at once, which a
    fit on the rate alone reads as a downward trend.  So the program fits
    the two counts separately, both linear in log-time:
      log(earners_i) ~ a_i + b_i * ln(1 + hours since launch)
      log(tracked)   ~ a_N + b_N * ln(1 + hours since launch)
    and forecasts the rate as earners / tracked at the target.  Log-time
    already builds in saturation: growth slows as the launch recedes.
  * Slopes are shrunk across trophies (empirical-Bayes): each trophy's
    slope is pulled toward the precision-weighted pooled slope, with the
    pull set by its own standard error and the spread of true slopes.
    A trophy whose own history is flat or noisy borrows the field's
    slope; a trophy with a clean trend keeps its own.
  * The level is anchored to the freshest read: the forecast starts from
    the last observed earner count and tracked count, and only the
    *change* to the target comes from the fitted slopes.
  * The spread (logit SD) combines the random-walk term the house
    baseline uses (0.5 * sqrt(horizon in days)), the slope's standard
    error times the extrapolation distance, the fit's residual SD, and a
    fixed refresh-shock term (Exophase refreshes hours apart and each one
    can move every rate at once; the history cannot show the refreshes
    that have not happened yet).  It is clipped to [0.35, 2.2].
  * The trophy at 100% in the last read (everyone who starts the game
    has it) is forecast to stay there, with a narrow spread.

Scoring is mean pinball loss on the logit scale, so the quantiles are
the normal quantiles of the forecast distribution above: honest
quantiles, not a point forecast with a guessed band.

Run: python3 -P -s /bot/main.py in python:3.13-slim, standard library
only, no network.  It reads every *.in in MAKEFUN_INPUT_DIR (else
/input) and writes <id>.out to MAKEFUN_OUTPUT_DIR (else /output).
Deterministic: no randomness, files processed in name order, output
bytes identical across runs.  It never crashes a case: any per-trophy
failure falls back to the flat house baseline for that trophy, and any
whole-case failure falls back to the flat baseline for the case.
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an
# `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Launch instant: 2026-11-19T00:00:00Z, the origin the house curve fits
# against.  Times are measured in hours from it, in log-time.
LAUNCH_EPOCH = 1795046400.0
# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's term, kept as one component of the spread).
SIGMA_RW = 0.5
# Fixed refresh-shock SD in logit units: Exophase refreshes hours apart
# and each refresh moves every rate at once; the history cannot show the
# refreshes still to come, so a small term is carried regardless.
SHOCK = 0.4
# Damping on the extrapolated log-count change (0.5): the log-time fit
# reads the recent trend, but a launch-week trend decelerates faster
# than log-time implies (adoption and progression both saturate), so
# the full extrapolation overshoots.  Synthetic launch-week cases
# (three adoption regimes) put the loss at 0.254 with damping 0.5
# against 0.346 undamped, 0.274 for the house curve baseline and
# 0.295 for the flat baseline; 0.4 scores about the same, 0.5 is kept
# as the less aggressive of the two.
DAMP = 0.5
# Scale on the total spread (0.8): the random-walk term is the house
# baseline's own calibration and the real 4.5-day moves are genuinely
# uncertain, so the spread is only mildly narrowed from the raw
# combination (synthetic cases prefer 0.7, the raw 1.0 is the hedge
# the other way; 0.8 splits the difference).
SPREAD_SCALE = 0.8
# Spread clip, in logit SD.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.2
# Fitting window: the last 24 hours of reads before the checkpoint (the
# house curve's window).  Widened to the whole history when the window
# holds too few reads.
WINDOW_HOURS = 24.0


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back
    as an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s: str) -> float:
    """ISO 8601 UTC to epoch seconds (trailing Z, or an explicit
    +00:00 offset)."""
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_hours(t_epoch: float) -> float:
    """ln(1 + hours from launch).  Reads just before launch give a
    small negative value; anything earlier than an hour before launch
    is floored."""
    hours = (t_epoch - LAUNCH_EPOCH) / 3600.0
    return math.log(max(0.05, 1.0 + hours))


def ols(xs, ys):
    """Ordinary least squares y ~ a + b*x.  Returns (a, b, se_b, s_res,
    n, sxx) or None when the fit is degenerate (fewer than 3 points,
    no spread in x, or no spread in y)."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof if dof > 0 else 0.0
    s_res = math.sqrt(max(0.0, s2))
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    return (a, b, se_b, s_res, n, sxx)


def flat_row(rate, spread):
    """House-baseline row: median at the last rate, random-walk spread."""
    lg = logit(rate if rate is not None else 0.01)
    return [round4(inv_logit_pct(lg + z * spread)) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    rw = SIGMA_RW * math.sqrt(horizon_days)
    at_epoch = parse_time(case["at"])
    target_epoch = parse_time(case["target"])
    x_target = log_hours(target_epoch)

    # --- Reads in the fitting window (last 24 h), else the whole history.
    window_start = at_epoch - WINDOW_HOURS * 3600.0
    reads = [r for r in history if parse_time(r["at"]) >= window_start]
    if len(reads) < 3:
        reads = list(history)
    xs = [log_hours(parse_time(r["at"])) for r in reads]
    x_last = xs[-1]
    dx = x_target - x_last

    # --- Tracked-players trend: log N ~ x.
    ns = [r.get("players_tracked") for r in reads]
    b_n, se_n = 0.0, 0.0
    log_n_last = None
    if all(isinstance(v, (int, float)) and v > 0 for v in ns):
        fit_n = ols(xs, [math.log(v) for v in ns])
        if fit_n is not None:
            b_n, se_n = fit_n[1], fit_n[2]
        log_n_last = math.log(ns[-1])
    else:
        log_n_last = math.log(ns[-1]) if isinstance(ns[-1], (int, float)) and ns[-1] > 0 else 0.0

    # --- Per-trophy earner trends: log E ~ x, E = rate * N / 100.
    fits = []  # per trophy: (b, se, s_res) or None
    for i in range(len(trophies)):
        pts_x, pts_y = [], []
        for j, r in enumerate(reads):
            rate = r["rates"][i] if i < len(r.get("rates") or []) else None
            n_j = r.get("players_tracked")
            if rate is None or not isinstance(n_j, (int, float)) or n_j <= 0:
                continue
            e = max(rate * n_j / 100.0, 0.5)
            pts_x.append(xs[j])
            pts_y.append(math.log(e))
        fit = ols(pts_x, pts_y) if pts_x else None
        fits.append(fit)

    # --- Pooled slope (empirical Bayes across trophies).
    usable = [(f[1], f[2]) for f in fits if f is not None and math.isfinite(f[2]) and f[2] > 0]
    if usable:
        wsum = sum(1.0 / (se * se) for _, se in usable)
        pooled = sum(b / (se * se) for b, se in usable) / wsum
        mean_b = sum(b for b, _ in usable) / len(usable)
        var_b = sum((b - mean_b) ** 2 for b, _ in usable) / len(usable)
        mean_se2 = sum(se * se for _, se in usable) / len(usable)
        tau2 = max(0.0, var_b - mean_se2)
    else:
        pooled, tau2 = 0.0, 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append(forecast_trophy(case, i, trophy, reads, xs, x_last,
                                        dx, rw, b_n, se_n, log_n_last,
                                        fits[i], pooled, tau2))
        except Exception:
            # Never lose a trophy: fall back to the flat baseline row.
            last = history[-1]
            rate = last["rates"][i] if i < len(last.get("rates") or []) else None
            rows.append({"id": trophy["id"], "q": flat_row(rate, rw)})
    return {"trophies": rows}


def forecast_trophy(case, i, trophy, reads, xs, x_last, dx, rw,
                    b_n, se_n, log_n_last, fit, pooled, tau2):
    history = case["history"]
    last = history[-1]
    rates = last.get("rates") or []
    last_rate = rates[i] if i < len(rates) else None

    # Last observed earner count and rate for this trophy (walk back
    # over reads that lack it).
    last_e = None
    last_r = None
    for r in reversed(history):
        rr = r.get("rates") or []
        n_j = r.get("players_tracked")
        if i < len(rr) and rr[i] is not None and isinstance(n_j, (int, float)) and n_j > 0:
            last_r = rr[i]
            last_e = max(rr[i] * n_j / 100.0, 0.5)
            break
    if last_e is None:
        # No read ever listed this trophy: use the tier's median last
        # rate, else a small rate, with a wide spread.
        tier_rates = []
        for k, t in enumerate(case["trophies"]):
            if t.get("tier") == trophy.get("tier"):
                rk = rates[k] if k < len(rates) else None
                if rk is not None:
                    tier_rates.append(rk)
        last_r = sorted(tier_rates)[len(tier_rates) // 2] if tier_rates else 0.5
        return {"id": trophy["id"], "q": flat_row(last_r, 1.5)}

    # Shrunk slope for this trophy's earners.
    if fit is not None and math.isfinite(fit[2]) and fit[2] > 0:
        b_i, se_i, s_res = fit[1], fit[2], fit[3]
        if tau2 > 0:
            w_i = 1.0 / (se_i * se_i)
            w_p = 1.0 / tau2
            b_star = (b_i * w_i + pooled * w_p) / (w_i + w_p)
            se_star = math.sqrt(1.0 / (w_i + w_p))
        else:
            b_star, se_star = pooled, 0.0
    else:
        b_star, se_star = pooled, 0.0
        s_res = 0.0

    # Forecast counts at the target, anchored to the freshest read.
    # Only the change from the fitted slopes is damped (see DAMP):
    # the level comes from the last read, the trend is halved.
    log_e_pred = math.log(last_e) + b_star * dx * DAMP
    log_n_pred = log_n_last + b_n * dx * DAMP
    r_pred = 100.0 * math.exp(log_e_pred - log_n_pred)
    median_logit = logit(clamp_pct(r_pred))

    # Spread: random walk + slope uncertainty + fit residual + refresh shock.
    var = rw * rw + (se_star * dx) ** 2 + (se_n * dx) ** 2
    var += min(s_res, 1.0) ** 2 + SHOCK * SHOCK
    spread = math.sqrt(max(0.0, var)) * SPREAD_SCALE
    spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))

    # Boundary trophy: a rate of 100% stays there; keep the spread narrow.
    if last_r is not None and last_r >= 99.99:
        median_logit = logit(99.99)
        spread = min(spread, 0.8)

    qs = [round4(inv_logit_pct(median_logit + z * spread)) for z in Z]
    # Monotone, finite, in [0, 100] by construction; enforce cheaply.
    qs = [min(100, max(0, q)) for q in qs]
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy["id"], "q": qs}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Whole-case fallback: the flat house baseline for every trophy.
            last = (case.get("history") or [{}])[-1]
            rw = SIGMA_RW * math.sqrt(case.get("horizon_minutes", 0) / 1440.0)
            rows = []
            for k, t in enumerate(case.get("trophies", [])):
                rate = (last.get("rates") or [None] * len(case["trophies"]))[k]
                rows.append({"id": t["id"], "q": flat_row(rate, rw)})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
