# Roll rules compared — every weekly start since 1997, held to today

Generated 24 September 2026 by `scripts/roll_rule_comparison.py`. 1,513 weekly starts, 12 Sep 1997 → 04 Sep 2026, each held to 14 Sep 2026; the same loan portfolio (*Keep the loan*) on every row, the rotation under four roll rules. Default setup otherwise (1 bn SPX, 250 m Lombard loan at SOFR + 75 bp capitalised, lending values 75 % SPX / 0 % calls / 90 % T-bills, 5-year ATM calls on SPXFP, 52 weekly steps, 15 % withholding). Settled starts: on or before 22 Sep 2020, so every call has expired at least once; unsettled: later starts, whose NAV today is partly a model mark. The annualised blocks use the starts held at least a year. USD m unless stated.

**The rules.** *Keep's exposure* (default): at every expiry the replacement closes the gap between Keep's SPX value and what the rotation's holdings were bought to carry (its SPX plus each surviving call's slot, grown with the index), so each replacement restores its own slot; paid from the payoff and the T-bills, then by selling SPX; every quarter-end the live exposure (SPX + the calls' dollar delta) is brought back inside ±10 % of Keep's — calls sold, most in the money first, at the model mark less 1 vol point (or at the mark: *no haircut*), calls bought from the T-bills. *Same dollar delta*: an in-the-money call is replaced by an ATM call with the same dollar delta (about twice the units; the option units double at every in-the-money expiry). *Same index units*: replaced on the same index units. Under the last two a worthless call is replaced on the same units, SPX is sold delta-for-delta and no band applies.

## Settled starts (every call expired at least once): value today (1,202 starts held at least a year)

| Final NAV (m) | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| lowest | 2,011 | 1,973 | 1,973 | 1,991 | 1,971 |
| 5th pct | 2,548 | 2,580 | 2,581 | 2,704 | 2,560 |
| 25th pct | 4,186 | 4,226 | 4,226 | 4,902 | 4,251 |
| median | 7,312 | 7,239 | 7,250 | 11,104 | 6,815 |
| 75th pct | 8,818 | 8,891 | 8,909 | 15,455 | 8,244 |
| 95th pct | 11,585 | 11,685 | 11,691 | 27,655 | 11,275 |
| highest | 14,240 | 14,833 | 14,837 | 34,942 | 14,736 |

| Annualised return | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| lowest | 8.8% | 8.5% | 8.5% | 10.0% | 7.7% |
| 5th pct | 9.2% | 9.1% | 9.1% | 10.7% | 8.2% |
| 25th pct | 11.9% | 11.8% | 11.8% | 14.2% | 11.5% |
| median | 15.3% | 15.4% | 15.4% | 17.7% | 15.5% |
| 75th pct | 16.5% | 16.8% | 16.8% | 19.0% | 16.8% |
| 95th pct | 18.3% | 18.6% | 18.6% | 20.2% | 18.5% |
| highest | 25.4% | 25.3% | 25.3% | 25.6% | 25.2% |
| rotation ahead on (same start) |  | 72% | 74% | 99% | 53% |
| median Δ annualised, Rotate − Keep |  | +0.1 pp | +0.1 pp | +2.4 pp | +0.1 pp |
| worst Δ final NAV (m), same start |  | -696 | -672 | -23 | -2,176 |
| best Δ final NAV (m), same start |  | +593 | +597 | +21,672 | +498 |

| Today | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| dry powder today, median (m) | 5,375 | 4,479 | 4,493 | 4,405 | 4,551 |
| calls / NAV today, median |  | 18% | 18% | 43% | 13% |
| exposure ÷ Keep's today, median | 1.00 | 1.10 | 1.10 | 2.40 | 0.98 |
| starts with a cut replacement |  | 0 | 0 | 0 | 0 |
| margin calls (Keep) | 0 |  |  |  |  |

## Unsettled starts (calls still open on the end day): value today (260 starts held at least a year)

| Final NAV (m) | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| lowest | 906 | 894 | 894 | 894 | 894 |
| 5th pct | 998 | 990 | 990 | 990 | 990 |
| 25th pct | 1,186 | 1,181 | 1,181 | 1,181 | 1,181 |
| median | 1,492 | 1,480 | 1,480 | 1,490 | 1,491 |
| 75th pct | 1,644 | 1,653 | 1,653 | 1,664 | 1,665 |
| 95th pct | 1,884 | 1,868 | 1,868 | 1,889 | 1,881 |
| highest | 2,175 | 2,138 | 2,138 | 2,148 | 2,136 |

| Annualised return | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| lowest | 13.7% | 13.2% | 13.2% | 13.6% | 13.6% |
| 5th pct | 14.5% | 14.1% | 14.1% | 14.4% | 14.4% |
| 25th pct | 17.2% | 16.9% | 16.9% | 17.0% | 17.0% |
| median | 21.7% | 21.2% | 21.2% | 21.2% | 21.2% |
| 75th pct | 24.6% | 24.6% | 24.6% | 24.8% | 24.8% |
| 95th pct | 28.7% | 27.8% | 27.8% | 27.8% | 27.8% |
| highest | 42.8% | 41.9% | 41.9% | 41.9% | 41.9% |
| rotation ahead on (same start) |  | 30% | 30% | 40% | 40% |
| median Δ annualised, Rotate − Keep |  | -0.3 pp | -0.3 pp | -0.2 pp | -0.2 pp |
| worst Δ final NAV (m), same start |  | -42 | -42 | -36 | -42 |
| best Δ final NAV (m), same start |  | +15 | +15 | +33 | +33 |

| Today | Keep the loan | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) | Same dollar delta | Same index units |
|---|---|---|---|---|---|
| dry powder today, median (m) | 1,042 | 895 | 895 | 861 | 862 |
| calls / NAV today, median |  | 17% | 17% | 18% | 18% |
| exposure ÷ Keep's today, median | 1.00 | 1.10 | 1.10 | 1.15 | 1.12 |
| starts with a cut replacement |  | 0 | 0 | 0 | 0 |
| margin calls (Keep) | 0 |  |  |  |  |

## At each market bottom (every start running that day)

Lowest and median NAV of each portfolio on the bottom day across the starts running that day (the same starts on every column), the share of those starts where the rotation has more dry powder than Keep's room before a margin call, and where its NAV is higher. Starts after the settled cut run only into the 2022 bottom.

| Bottom | starts | Keep: lowest / median NAV | Keep's exposure (1 vol pt): lowest / median NAV | Keep's exposure (no haircut): lowest / median NAV | Same dollar delta: lowest / median NAV | Same index units: lowest / median NAV | Keep's exposure (1 vol pt): more dry powder / higher NAV | Keep's exposure (no haircut): more dry powder / higher NAV | Same dollar delta: more dry powder / higher NAV | Same index units: more dry powder / higher NAV |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000–2002 (09 Oct 2002) | 265 | 241 / 390 | 306 / 436 | 306 / 436 | 306 / 436 | 306 / 436 | 94% / 100% | 94% / 100% | 94% / 100% | 94% / 100% |
| 2007–2009 (09 Mar 2009) | 600 | 142 / 293 | 211 / 384 | 211 / 384 | 225 / 384 | 234 / 393 | 96% / 100% | 96% / 100% | 96% / 100% | 96% / 100% |
| 2020 (23 Mar 2020) | 1,176 | 413 / 1,772 | 416 / 1,834 | 416 / 1,835 | 416 / 1,821 | 416 / 1,806 | 93% / 86% | 93% / 86% | 50% / 73% | 79% / 68% |
| 2022 (12 Oct 2022) | 1,309 | 503 / 3,003 | 508 / 3,089 | 508 / 3,097 | 508 / 3,666 | 508 / 2,986 | 42% / 90% | 43% / 90% | 26% / 94% | 38% / 70% |

## The target rule's mechanics

| Per start (median, or total) | Keep's exposure (1 vol pt) | Keep's exposure (no haircut) |
|---|---|---|
| rolls (expiries with a decision) | 63 | 63 |
| rolls skipped (gap ≤ 0) | 0 | 0 |
| rebalances up (calls sold) | 28 | 28 |
| rebalances down (calls bought) | 0 | 0 |
| partial rebalances (T-bills short) | 0 | 0 |
| calls sold at rebalances (m) | 497 | 500 |
| calls bought at rebalances (m) | 0 | 0 |
| unwind cost (m), median start | 1.9 | 0.0 |
| unwind cost (m), all starts | 7,521 | 0 |
| cut replacements | 0 | 0 |
| exposure ÷ Keep's today: 5th / median / 95th pct | 1.02 / 1.10 / 1.10 | 1.02 / 1.10 / 1.10 |

Each replacement restores its own slot, so the weekly ladder survives the expiries; the band sales remove whole tranches over time. Between quarterly checks the live exposure drifts with the delta; on the check days it is inside the band unless the T-bills were short (partial).

Run time 524 s.
