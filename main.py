#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: a statistical forecaster for
Polymarket Yes prices, one case at a time, fully offline.

Model, per market, in logit space (so prices stay inside (0, 1) and a
one-tick move near 0.03 counts for more than the same move near 0.5):

  * The 1-minute logit returns are split into a diffusion part and a
    jump part.  Variance is winsorized, returns beyond four robust
    sigmas are counted as jumps, and their rate (per minute) and mean
    squared size are estimated separately.  Statistics blend the full
    history with the last 12 hours, because volatility clusters.
  * Over a horizon of h minutes the no-jump / one-jump / two-or-more
    jump mixture is a three-component normal mixture in logit space:
    jump counts are Poisson with mean lambda*h.  Quantiles come from
    inverting that mixture CDF by bisection, which keeps every row
    non-decreasing in the quantile levels by construction.
  * The centre starts at the last logit's random walk, plus a drift
    shrunk hard toward zero by its own t-statistic, plus a gentle
    pull toward the median logit of the last 12 hours.
  * A floor tied to the market's own tick size keeps dead-quiet
    markets from getting a zero-width forecast.

The program is deterministic: fixed constants, no randomness, no
clock, no network.  It reads every *.in file of $MAKEFUN_INPUT_DIR
(or /input) in name order and writes one <id>.out per file into
$MAKEFUN_OUTPUT_DIR (or /output).
"""

import json
import math
import os
from statistics import median

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

DEFAULT_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)

# --- model constants -------------------------------------------------------
PRICE_EPS = 0.0005        # clamp prices into [EPS, 1-EPS] before the logit
HEAD_SKIP = 30            # opening minutes of a series ignored for statistics
RECENT_WINDOW = 720       # minutes in the "recent" volatility window
RECENT_WEIGHT = 0.55      # weight of the recent window in blended variances
WINSOR_K = 6.0            # winsorize returns at this many robust sigmas
JUMP_K = 4.0              # |return| beyond this many sigmas counts as a jump
DIFFUSION_FLOOR = 0.25    # diffusion keeps at least this share of total var
MIN_LOGIT_SIGMA = 0.006   # absolute floor on per-minute total logit sigma
TICK_SIGMA_SHARE = 0.25   # tick-based floor: this share of one tick (logit)
MAX_LOGIT_SIGMA = 0.35    # cap on per-minute total logit sigma
BASELINE_SIGMA_PRICE = 0.0025  # the house baseline's per-minute price sigma
BASELINE_BLEND = 0.3      # log-space shrinkage of our sigma toward baseline
JUMP_SCALE = 0.15         # damp the jump second moment (news is priced in)
HORIZON_GAMMA = 0.85      # variance grows like h**GAMMA (mean reversion)
DRIFT_T_PRIOR = 3.0       # t-stat shrinkage prior for the drift
DRIFT_SIGMA_CAP = 0.30    # drift per minute capped at this share of sigma
ANCHOR_PULL = 0.12        # strength of the pull toward the recent median
ANCHOR_TAU = 600.0        # minutes over which the anchor pull phases in
OUT_LO = 0.001            # final forecasts are clamped into [OUT_LO, OUT_HI]
OUT_HI = 0.999
SQRT2 = math.sqrt(2.0)


def logit(p):
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _mean(xs):
    return sum(xs) / len(xs)


def vol_components(returns):
    """Split 1-minute logit returns into (diffusion var, jump rate,
    jump mean-square).  Returns None for an empty series."""
    n = len(returns)
    if n == 0:
        return None
    m2 = _mean([r * r for r in returns])
    s0 = math.sqrt(m2)
    if s0 == 0.0:
        return (0.0, 0.0, 0.0)
    cap = WINSOR_K * s0
    cap2 = cap * cap
    m2w = _mean([r * r if r * r < cap2 else cap2 for r in returns])
    s = math.sqrt(m2w)
    thresh = JUMP_K * s
    jumps = [r for r in returns if abs(r) > thresh]
    lam = len(jumps) / n
    jump_m2 = _mean([j * j for j in jumps]) if jumps else 0.0
    diff_var = m2w - lam * jump_m2
    floor_var = DIFFUSION_FLOOR * m2w
    if diff_var < floor_var:
        diff_var = floor_var
    return (diff_var, lam, jump_m2)


def blend_vol(full, recent):
    """Blend full-history and recent-window variance components."""
    if recent is None:
        return full
    if full is None:
        return recent
    w = RECENT_WEIGHT
    diff_var = (1.0 - w) * full[0] + w * recent[0]
    lam = (1.0 - w) * full[1] + w * recent[1]
    if full[2] > 0.0 and recent[2] > 0.0:
        jump_m2 = 0.5 * (full[2] + recent[2])
    else:
        jump_m2 = max(full[2], recent[2])
    return (diff_var, lam, jump_m2)


def mixture_cdf(x, mu, comps):
    total = 0.0
    for weight, sd in comps:
        total += weight * 0.5 * (1.0 + math.erf((x - mu) / (sd * SQRT2)))
    return total


def mixture_quantile(mu, comps, tau):
    """Invert the mixture CDF by bisection; monotone in tau."""
    var = sum(w * sd * sd for w, sd in comps)
    sd_tot = math.sqrt(var) if var > 0.0 else 1e-9
    lo = mu - 10.0 * sd_tot
    hi = mu + 10.0 * sd_tot
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if mixture_cdf(mid, mu, comps) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def horizon_components(diff_var, lam, jump_m2, h):
    """The three normal components (weight, sd) of the h-minute move."""
    growth = h ** HORIZON_GAMMA
    var_d = diff_var * growth
    lam_h = lam * growth
    if lam_h <= 1e-12 or jump_m2 <= 0.0:
        return [(1.0, math.sqrt(var_d + lam_h * jump_m2))]
    w0 = math.exp(-lam_h)
    w1 = lam_h * w0
    w2 = 1.0 - w0 - w1
    comps = [(w0, math.sqrt(var_d)), (w1, math.sqrt(var_d + jump_m2))]
    if w2 > 1e-12:
        k2 = (lam_h - w1) / w2  # expected jump count given at least two
        comps.append((w2, math.sqrt(var_d + k2 * jump_m2)))
    else:
        comps[0] = (w0 + w2, comps[0][1])
    return comps


def forecast_market(prices, horizon, taus):
    """horizon rows, one per minute after the checkpoint; each row one
    forecast per level in taus, non-decreasing, in [OUT_LO, OUT_HI]."""
    if not prices:
        centre = 0.5
        return [[centre] * len(taus) for _ in range(horizon)]
    clamped = [min(1.0 - PRICE_EPS, max(PRICE_EPS, float(p))) for p in prices]
    last_price = clamped[-1]
    last_logit = logit(last_price)
    logits = [logit(p) for p in clamped]
    returns = [logits[i + 1] - logits[i] for i in range(len(logits) - 1)]

    stats_returns = returns[HEAD_SKIP:] if len(returns) > HEAD_SKIP + 300 else returns
    full = vol_components(stats_returns)
    recent = None
    if len(stats_returns) > RECENT_WINDOW + 60:
        recent = vol_components(stats_returns[-RECENT_WINDOW:])
    if full is None:
        full = (0.0, 0.0, 0.0)
    diff_var, lam, jump_m2 = blend_vol(full, recent)

    # Damp the jumps, then shrink the total per-minute sigma toward the
    # house baseline's sigma in log space: a market whose own history
    # screams chaos still gets pulled toward the typical scale, because
    # most of that chaos is already behind the checkpoint.
    jump_m2 *= JUMP_SCALE
    raw_var = diff_var + lam * jump_m2
    if raw_var > 0.0:
        sigma_price = math.sqrt(raw_var) * last_price * (1.0 - last_price)
        target_price = math.exp(
            (1.0 - BASELINE_BLEND) * math.log(sigma_price)
            + BASELINE_BLEND * math.log(BASELINE_SIGMA_PRICE)
        )
        scale = (target_price / sigma_price) ** 2
        diff_var *= scale
        jump_m2 *= scale

    # Floors and caps on the total per-minute variance, in logit space.
    steps = [abs(clamped[i + 1] - clamped[i]) for i in range(len(clamped) - 1)]
    ticks = [s for s in steps if s > 0.0]
    tick = min(ticks) if ticks else 0.001
    tick_logit = tick / max(last_price * (1.0 - last_price), 1e-4)
    floor_sigma = max(MIN_LOGIT_SIGMA, TICK_SIGMA_SHARE * tick_logit)
    total_var = diff_var + lam * jump_m2
    if total_var < floor_sigma * floor_sigma:
        diff_var += floor_sigma * floor_sigma - total_var
        total_var = floor_sigma * floor_sigma
    if total_var > MAX_LOGIT_SIGMA * MAX_LOGIT_SIGMA:
        scale = (MAX_LOGIT_SIGMA * MAX_LOGIT_SIGMA) / total_var
        diff_var *= scale
        jump_m2 *= scale
        total_var = MAX_LOGIT_SIGMA * MAX_LOGIT_SIGMA
    sigma = math.sqrt(total_var)

    # Centre path: shrunk drift plus a gentle pull to the recent median.
    drift = 0.0
    if stats_returns:
        mean_r = _mean(stats_returns)
        se = math.sqrt(max(total_var, 1e-12) / len(stats_returns))
        if se > 0.0:
            t_stat = mean_r / se
            shrink = abs(t_stat) / (abs(t_stat) + DRIFT_T_PRIOR)
            drift = mean_r * shrink
        cap = DRIFT_SIGMA_CAP * sigma
        drift = max(-cap, min(cap, drift))
    anchor = median(logits[-RECENT_WINDOW:]) if len(logits) > 60 else last_logit
    anchor_gap = anchor - last_logit

    rows = []
    for h in range(1, horizon + 1):
        mu = last_logit + drift * h
        mu += ANCHOR_PULL * anchor_gap * (1.0 - math.exp(-h / ANCHOR_TAU))
        comps = horizon_components(diff_var, lam, jump_m2, h)
        row = []
        for tau in taus:
            if tau == 0.5:
                q_logit = mu
            else:
                q_logit = mixture_quantile(mu, comps, tau)
            row.append(logistic(q_logit))
        # Round to 6 decimals, clamp, and enforce non-decreasing.
        fixed = []
        prev = OUT_LO
        for value in row:
            v = round(value, 6)
            if v < OUT_LO:
                v = OUT_LO
            elif v > OUT_HI:
                v = OUT_HI
            if v < prev:
                v = prev
            fixed.append(v)
            prev = v
        rows.append(fixed)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        taus = case.get("quantiles") or list(DEFAULT_TAUS)
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for market in case["markets"]:
            q = forecast_market(market["prices"], horizon, taus)
            forecasts.append({"id": market["id"], "q": q})
        payload = json.dumps({"forecasts": forecasts}, separators=(",", ":"))
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")


if __name__ == "__main__":
    main()
