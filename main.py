"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (gta6-twitch-ccu-forecast-01).

Model note (the whole model, in one place). GTA VI unlocks at midnight local time on
19 November 2026, rolling country by country: New Zealand first (11:00 UTC 18 Nov),
central Europe 23:00 UTC, the UK 00:00 UTC, US East 05:00 UTC, US West 08:00 UTC.
So the UTC release day is not one spike but a staircase: hour 1 (00:00-01:00 UTC) only
holds the UK/Europe midnight crowd, the US wakes up at hours 6 and 9, and the peak
lands in the first evening when US afternoon overlaps the European evening.

That shape is what past launches show. Benchmarks from the public record: Hogwarts
Legacy peaked at ~1.27M Twitch viewers, Cyberpunk 2077 at ~1.14M, Elden Ring at
~910k, God of War Ragnarok at ~480k. But raw peaks do not transfer across launches;
ratios to the first hour do. For every comparable past launch in the case file we take
its release-day hourly curve, divide each hour by that launch's hour-1 print, and
across launches take the median log ratio and a robust log spread per hour. Only
console launches with a midnight-local rolling unlock count as comparables: a
simultaneous global PC unlock (or a paid early-access window) puts the surge in hour 1
itself, so those launches' ratios to hour 1 are a different animal and would flatten
the curve and blow up the spread. Early-access-contaminated and weak analogues are
left out the same way.

What the fitted curve says, from the comparables in the practice cases: hour 1's
audience dips to about 0.75x by hour 4 (the small hours, before the US unlocks),
climbs through the morning to ~1.1x by hour 7, then rises steeply into the evening
peak of ~2.4x at hour 19 (19:00-20:00 UTC: US afternoon, European evening), holds
~2.3x through hour 22 and decays to ~1.9x by hour 24. The log spread is tight early
(~0.12 at hour 2) and widens to ~0.55 by the evening, capped there: with only a
dozen clean analogues, the raw cross-launch spread overstates how much launches
really disagree once the unlock pattern is matched.

The forecast for hour h is q_tau(h) = p1 * r_h * exp(z_tau * s_h), where p1 is the
case's hour-1 print, r_h and s_h are the median ratio and log spread for hour h
computed at runtime from the case's own comparables (falling back to constants fitted
offline from the same procedure when a case has too few usable comparables), and
z_tau is the standard normal quantile at each scored level. Deterministic, standard
library only, no network, no randomness.

Runtime: python3 main.py, reads every *.in in $MAKEFUN_INPUT_DIR (else /input),
writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Cap on the per-hour log spread: the raw cross-launch dispersion overstates the
# disagreement among matched midnight-local console launches.
SPREAD_CAP = 0.55
# A comparable needs at least this many release-day hours to count.
MIN_DAY_HOURS = 12
# An hour's curve needs at least this many launches behind it, else the fallback.
MIN_LAUNCHES = 3

# Fallback (r_h, s_h) per hour 2..24, fitted offline by the same procedure on the
# practice cases' midnight-local console comparables. Used only when a case has too
# few usable comparables for that hour.
FALLBACK = {
    2: (0.898, 0.121), 3: (0.803, 0.228), 4: (0.750, 0.377),
    5: (0.821, 0.309), 6: (0.989, 0.446), 7: (1.046, 0.546),
    8: (1.060, 0.550), 9: (1.026, 0.550), 10: (1.080, 0.550),
    11: (1.085, 0.550), 12: (1.078, 0.550), 13: (1.200, 0.550),
    14: (1.313, 0.550), 15: (1.518, 0.550), 16: (1.820, 0.550),
    17: (1.951, 0.550), 18: (2.065, 0.549), 19: (2.402, 0.527),
    20: (2.372, 0.514), 21: (2.330, 0.506), 22: (2.298, 0.550),
    23: (2.182, 0.550), 24: (1.902, 0.550),
}


def round2(x):
    """x rounded to 2 decimals, half up; a whole number comes back as an int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _day_curve(comp):
    """Release-day UTC hour -> viewers (hours 1..24) for one comparable launch."""
    try:
        day_start = _parse_iso(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for pt in comp.get("hourly") or []:
        try:
            h = int((_parse_iso(pt["from"]) - day_start).total_seconds() // 3600) + 1
            v = pt.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= h <= 24 and isinstance(v, (int, float)) and v > 0:
            out[h] = v
    return out


def _is_clean_comparable(comp):
    """Midnight-local rolling console launch, not early-access contaminated or weak.

    The unlock note carries a tier: only "comparable" counts, and only when the
    unlock is a midnight-local roll. Simultaneous global unlocks and early-access
    windows put the surge in hour 1 itself, which breaks the ratio shape.
    """
    unlock = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in unlock or "tier: weak" in unlock:
        return False
    if "tier: target-history" in unlock:
        return False
    if "tier: comparable" not in unlock and "midnight local" not in unlock:
        return False
    return "midnight local" in unlock


def _robust_sd(xs):
    """Interquartile-range based log spread; robust to one launch's odd shape."""
    if len(xs) < 4:
        return statistics.pstdev(xs) if len(xs) > 1 else 0.3
    q = statistics.quantiles(xs, n=4)
    return (q[2] - q[0]) / 1.349


def fit_curve(comparables):
    """Per hour 2..24: (median ratio r_h, log spread s_h) from the comparables."""
    logs = {h: [] for h in range(2, 25)}
    for comp in comparables or []:
        if not _is_clean_comparable(comp):
            continue
        curve = _day_curve(comp)
        if 1 not in curve or len(curve) < MIN_DAY_HOURS:
            continue
        p1 = curve[1]
        for h in range(2, 25):
            if h in curve:
                logs[h].append(math.log(curve[h] / p1))
    fitted = {}
    for h in range(2, 25):
        xs = logs[h]
        if len(xs) >= MIN_LAUNCHES:
            fitted[h] = (math.exp(statistics.median(xs)),
                         min(_robust_sd(xs), SPREAD_CAP))
        else:
            fitted[h] = FALLBACK[h]
    return fitted


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curve = fit_curve(case.get("comparables"))
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        r, s = curve[h]
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * r * math.exp(z * s))
            q = min(max(q, 0.01), 1e9)
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
