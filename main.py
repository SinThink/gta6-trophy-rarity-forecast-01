#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates at a future instant.

Reads every <id>.in case in MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to MAKEFUN_OUTPUT_DIR (default /output): for each trophy, five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate in percent at the
case's target.

How the median is built, per trophy, on the logit scale:

  1. History points are (u, logit(rate)) with u = ln(1 + hours since
     2026-11-19T00:00Z), which bends a saturating growth curve into a line.
     Only the recent window is fitted; the early plateau is not trend.
  2. A least-squares line through those points is read at the target. With
     fewer than two usable reads the last rate stands in.
  3. A refresh correction. A read shows the page as of Exophase's last
     refresh, and the population it averages over grows fast in week one: a
     player who has not earned a trophy yet still counts in the denominator.
     So between two refreshes the tracked base grows by g and each rate
     scales by about 1/g, whatever its growth curve says. The forecast time
     to the target gives r = (players now + arrivals) / players now, and the
     median takes logit(rate / r) at or below the top rate.
  4. New trophies: a rate at the page's smallest step (0.01 / 0.78 if only
     one tracked player has it) is a single player, and whether that player
     exists yet is close to a coin toss, so its median drops to about 57%.
  5. Spread: the random walk below (0.5 per sqrt day on the logit scale, as
     the house baseline uses), the fit residual, a term from the fit's span
     and a per-tier fixed effect for finishing a list's bronze trophies.

Everything is clamped, rounded to 4 decimals half up, and made
non-decreasing. No clock, no randomness, no network: the same case always
gives the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Release instant: the time variable's origin and the release day itself.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

Z = (-1.6449, -0.6745, 0.0, 0.6745, 1.6449)

RATE_FLOOR = 0.01
RATE_CEIL = 99.99

WALK_PER_DAY = 0.5       # random-walk sigma in logit per sqrt day
HISTORY_HOURS = 36.0     # fit only reads this close to the case's checkpoint
FIT_TAIL_DAYS = 0.75     # floor on how well-informed a fit is, in days
TREND_CAP = 1.2          # max |trend - last rate| in logit, before refresh

# Population model: Exophase's tracked base after t days from release.
# Smoothstep ramp from BASE_AT_RELEASE to BASE_GROWTH_PER_DAY * t, easing in
# over RAMP_DAYS; tuned so launch night (638 players, then 3868 at 04:30Z)
# and the first week (tens of thousands) are both plausible readings.
BASE_AT_RELEASE = 638.0
BASE_GROWTH_PER_DAY = 9000.0
RAMP_DAYS = 0.012

TOP_RATE = 99.0          # at or above: everyone earns it, no refresh effect
MIN_STEP = 1e-9

TIER_SPREAD = {          # extra logit sigma by tier: ends of the list are
    "platinum": 0.15,    # rarer than the early trophies and jumpier
    "gold": 0.10,
    "silver": 0.05,
    "bronze": 0.0,
}

FLOOR_SHORT = 0.90       # band sigma with a horizon of minutes
FLOOR_LONG = 0.20        # band sigma with a horizon of days
FLOOR_FADE_DAYS = 0.15   # horizon over which the floor fades to FLOOR_LONG


def clamp_rate(x: float) -> float:
    return min(RATE_CEIL, max(RATE_FLOOR, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def from_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def time_key(dt: datetime) -> float:
    hours = (dt - EPOCH).total_seconds() / 3600.0
    return math.log(1.0 + max(hours, -0.999))


def round4(x: float) -> float:
    """x to 4 decimals, half up on the magnitude (as Math.round does)."""
    y = x * 10000.0
    n = math.floor(abs(y) + 0.5)
    v = n / 10000.0
    return -v if y < 0.0 else v


def least_squares(xs, ys):
    """(intercept, slope) of the best line, or None if x does not vary."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 1e-12:
        return None
    slope = sxy / sxx
    return my - slope * mx, slope


def tracked_base(days_from_release: float) -> float:
    """Expected players tracked, `days` after release, per the ramp model."""
    t = max(0.0, days_from_release)
    target = max(BASE_AT_RELEASE, BASE_GROWTH_PER_DAY * t)
    # Smoothstep weight: 0 at release, 1 by RAMP_DAYS.
    s = min(1.0, t / RAMP_DAYS)
    w = s * s * (3.0 - 2.0 * s)
    return BASE_AT_RELEASE + (target - BASE_AT_RELEASE) * w


def read_points(read, index, trophies_n):
    """One trophy's rate in one read, as a float, or None."""
    rates = read.get("rates")
    if not isinstance(rates, list) or index >= len(rates) or index >= trophies_n:
        return None
    r = rates[index]
    if r is None or isinstance(r, bool):
        return None
    try:
        r = float(r)
    except (TypeError, ValueError):
        return None
    return r if math.isfinite(r) else None


def trend_point(points, u_target):
    """Fitted logit at the target and how noisy the fit is."""
    if not points:
        return None, 0.0, 0
    last_y = points[-1][1]
    if len(points) < 2:
        return last_y, 0.0, 1
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line = least_squares(xs, ys)
    if line is None:
        return last_y, 0.0, len(points)
    a, b = line
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in points) / len(points))
    # The slope is estimated over the read span; reading it far past that
    # span is extrapolation, so shrink the move halfway out to the target.
    span = max(1e-6, xs[-1] - xs[0])
    reach = max(1e-6, u_target - xs[-1])
    shrink = 1.0 / (1.0 + 0.5 * reach / span)
    fitted = a + b * u_target
    moved = last_y + (fitted - last_y) * shrink
    moved = max(last_y - TREND_CAP, min(last_y + TREND_CAP, moved))
    return moved, resid, len(points)


def refresh_scale(u_target, at_days):
    """log(r) where r = expected tracked base at target / base at the checkpoint."""
    if u_target <= 0.0 or at_days <= 0.0:
        return 0.0
    target_days = math.expm1(u_target)
    now = tracked_base(at_days)
    if now <= 0.0:
        return 0.0
    ratio = max(1.0, tracked_base(target_days) / now)
    return min(math.log(ratio), 4.0)


def spread_sigma(days, resid, n, tier):
    walk = WALK_PER_DAY * math.sqrt(max(0.0, days))
    noise = min(0.6, 0.6 * resid)
    informed = FIT_TAIL_DAYS + 0.15 * max(0, n - 1)
    extra = 0.15 * math.sqrt(max(0.0, days)) / math.sqrt(informed)
    return walk + noise + extra + TIER_SPREAD.get(tier, 0.0)


def floor_sigma(days):
    fade = min(1.0, max(0.0, days / FLOOR_FADE_DAYS))
    return FLOOR_SHORT + (FLOOR_LONG - FLOOR_SHORT) * fade


def row_quantiles(median_u, sigma):
    out = []
    for z in Z:
        out.append(round4(clamp_rate(inv_logit(median_u + z * sigma) * 100.0)))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_trophy(points, u_target, at_days, days, tier, last_rate):
    if last_rate is None:
        points = list(points)
        last_rate = inv_logit(points[-1][1]) * 100.0 if points else 0.0
    median, resid, n = trend_point(points, u_target)
    if median is None:
        median = logit(last_rate if last_rate > 0 else 0.01)
    sigma = spread_sigma(days, resid, n, tier)
    sigma = math.sqrt(sigma * sigma + floor_sigma(days) ** 2)
    # Inside this window the page has not refreshed and the rate is known.
    if days * 24.0 < 1.0:
        median = logit(last_rate)
        return row_quantiles(median, sigma)

    if last_rate < TOP_RATE:
        median -= refresh_scale(u_target, at_days)
    # A lone bearer: has that player been counted yet?
    if 0.0 < last_rate <= 0.5:
        step_guess = 0.01 if last_rate < 0.02 else 0.78
        if abs(last_rate - step_guess) < 0.02:
            median += math.log(0.57) - math.log(0.43)
    return row_quantiles(median, sigma)


def forecast_case(case):
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    target = from_iso(case["target"])
    at = from_iso(case.get("at") or case["target"])
    u_target = time_key(target)
    at_days = max(0.0, (at - EPOCH).total_seconds() / 86400.0)
    days = float(case.get("horizon_minutes") or 0.0) / 1440.0
    cutoff = (at - EPOCH).total_seconds() - HISTORY_HOURS * 3600.0

    history = []
    for read in reads:
        if not isinstance(read, dict) or not read.get("at"):
            continue
        try:
            when = from_iso(read["at"])
        except (TypeError, ValueError):
            continue
        if (when - EPOCH).total_seconds() < cutoff:
            continue
        history.append((when, time_key(when), read))
    if not history and reads:
        history = [(None, None, reads[-1])]

    last_read = reads[-1] if reads else {}
    n_trophies = len(trophies)
    rows = []
    for i, trophy in enumerate(trophies):
        points = []
        for when, u, read in history:
            r = read_points(read, i, n_trophies)
            if r is not None and u is not None:
                points.append((u, logit(r)))
        last_rate = read_points(last_read, i, n_trophies)
        tier = str(trophy.get("tier") or "bronze")
        q = forecast_trophy(points, u_target, at_days, days, tier, last_rate)
        rows.append({"id": trophy.get("id", i + 1), "q": q})
    return {"trophies": rows}


def crude_output(path: Path):
    """Never leave a case without a file: last rate, random-walk band."""
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"trophies": []}
    try:
        trophies = case.get("trophies") or []
        reads = case.get("history") or []
        last = reads[-1].get("rates") if reads and isinstance(reads[-1], dict) else []
        days = float(case.get("horizon_minutes") or 0.0) / 1440.0
        sigma = WALK_PER_DAY * math.sqrt(max(0.0, days)) + 0.6
        rows = []
        for i, trophy in enumerate(trophies):
            r = read_points({"rates": last}, i, len(trophies)) or 0.01
            rows.append({"id": trophy.get("id", i + 1), "q": row_quantiles(logit(r), sigma)})
        return {"trophies": rows}
    except Exception:
        return {"trophies": []}


def main() -> int:
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.is_dir():
        return 0
    entries = sorted(
        (p for p in in_dir.iterdir() if p.is_file() and p.name.endswith(".in")),
        key=lambda p: p.name,
    )
    for path in entries:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
            result = forecast_case(case)
        except Exception as exc:
            print(f"{path.name}: forecast failed: {exc}", file=sys.stderr)
            result = crude_output(path)
        body = json.dumps(result, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        (out_dir / (path.name[: -len(".in")] + ".out")).write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
