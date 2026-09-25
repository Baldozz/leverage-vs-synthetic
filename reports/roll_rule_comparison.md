# Roll rules compared — every weekly start since 1997, held to today

Generated 25 September 2026 by `scripts/roll_rule_comparison.py`. 1,513 weekly starts, 12 Sep 1997 → 04 Sep 2026, each held to 14 Sep 2026; the same loan portfolio (*Keep the loan*) on every row, the rotation under five roll rules. Default setup otherwise (1 bn SPX, 250 m Lombard loan at SOFR + 75 bp capitalised, lending values 75 % SPX / 0 % calls / 90 % T-bills, the bank calling at 90 % of the lending value, 5-year ATM calls on SPXFP, 52 weekly steps, 15 % withholding). Settled starts: on or before 22 Sep 2020, so every call has expired at least once; unsettled: later starts, whose NAV today is partly a model mark. The annualised blocks use the starts held at least a year. USD m unless stated.

**The rules.** *Keep's exposure* (default): at every expiry the replacement closes the gap between Keep's SPX value and what the rotation's holdings were bought to carry (its SPX plus each surviving call's slot, grown with the index), so each replacement restores its own slot; paid from the payoff and the T-bills, then by selling SPX; every month-end the live exposure (SPX + the calls' dollar delta) is brought back inside ±10 % of Keep's — calls sold, most in the money first, at the model mark less 1 vol point (or at the mark: *no haircut*), calls bought from the T-bills and, once they are gone, from SPX sold for them, s = rest ÷ (δ/c − 1) (*quarterly, T-bills only*: the check every quarter and nothing sold to fund it — the default before 25 September). *Same dollar delta*: an in-the-money call is replaced by an ATM call with the same dollar delta (about twice the units; the option units double at every in-the-money expiry). *Same index units*: replaced on the same index units. Under the last two a worthless call is replaced on the same units, SPX is sold delta-for-delta and no band applies.

## Settled starts (every call expired at least once): value today (1,202 starts held at least a year)

| Final NAV (m) | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| lowest | 2,011 | 1,972 | 1,972 | 1,973 | 1,991 | 1,971 |
| 5th pct | 2,548 | 2,570 | 2,571 | 2,580 | 2,704 | 2,560 |
| 25th pct | 4,186 | 4,212 | 4,213 | 4,226 | 4,902 | 4,251 |
| median | 7,312 | 7,225 | 7,231 | 7,239 | 11,104 | 6,815 |
| 75th pct | 8,818 | 8,884 | 8,890 | 8,891 | 15,455 | 8,244 |
| 95th pct | 11,585 | 11,664 | 11,673 | 11,685 | 27,655 | 11,275 |
| highest | 14,240 | 14,796 | 14,800 | 14,833 | 34,942 | 14,736 |

| Annualised return | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| lowest | 8.8% | 8.6% | 8.6% | 8.5% | 10.0% | 7.7% |
| 5th pct | 9.2% | 9.0% | 9.0% | 9.1% | 10.7% | 8.2% |
| 25th pct | 11.9% | 11.9% | 12.0% | 11.8% | 14.2% | 11.5% |
| median | 15.3% | 15.4% | 15.4% | 15.4% | 17.7% | 15.5% |
| 75th pct | 16.5% | 16.7% | 16.7% | 16.8% | 19.0% | 16.8% |
| 95th pct | 18.3% | 18.6% | 18.6% | 18.6% | 20.2% | 18.5% |
| highest | 25.4% | 25.3% | 25.3% | 25.3% | 25.6% | 25.2% |
| rotation ahead on (same start) |  | 76% | 77% | 72% | 99% | 53% |
| median Δ annualised, Rotate − Keep |  | +0.1 pp | +0.1 pp | +0.1 pp | +2.4 pp | +0.1 pp |
| worst Δ final NAV (m), same start |  | -566 | -545 | -696 | -23 | -2,176 |
| best Δ final NAV (m), same start |  | +556 | +560 | +593 | +21,672 | +498 |

| Today | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| dry powder today, median (m) | 4,795 | 4,046 | 4,056 | 4,031 | 3,965 | 4,096 |
| calls / NAV today, median |  | 18% | 18% | 18% | 43% | 13% |
| exposure ÷ Keep's today, median | 1.00 | 1.09 | 1.09 | 1.10 | 2.40 | 0.98 |
| starts with a cut replacement |  | 0 | 0 | 0 | 0 | 0 |
| margin calls (Keep) | 30 |  |  |  |  |  |

## Unsettled starts (calls still open on the end day): value today (260 starts held at least a year)

| Final NAV (m) | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| lowest | 906 | 894 | 894 | 894 | 894 | 894 |
| 5th pct | 998 | 990 | 990 | 990 | 990 | 990 |
| 25th pct | 1,186 | 1,181 | 1,181 | 1,181 | 1,181 | 1,181 |
| median | 1,492 | 1,479 | 1,479 | 1,480 | 1,490 | 1,491 |
| 75th pct | 1,644 | 1,651 | 1,651 | 1,653 | 1,664 | 1,665 |
| 95th pct | 1,884 | 1,865 | 1,866 | 1,868 | 1,889 | 1,881 |
| highest | 2,175 | 2,138 | 2,138 | 2,138 | 2,148 | 2,136 |

| Annualised return | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| lowest | 13.7% | 13.2% | 13.2% | 13.2% | 13.6% | 13.6% |
| 5th pct | 14.5% | 14.1% | 14.1% | 14.1% | 14.4% | 14.4% |
| 25th pct | 17.2% | 16.9% | 16.9% | 16.9% | 17.0% | 17.0% |
| median | 21.7% | 21.2% | 21.2% | 21.2% | 21.2% | 21.2% |
| 75th pct | 24.6% | 24.6% | 24.6% | 24.6% | 24.8% | 24.8% |
| 95th pct | 28.7% | 27.7% | 27.8% | 27.8% | 27.8% | 27.8% |
| highest | 42.8% | 41.9% | 41.9% | 41.9% | 41.9% | 41.9% |
| rotation ahead on (same start) |  | 27% | 28% | 30% | 40% | 40% |
| median Δ annualised, Rotate − Keep |  | -0.3 pp | -0.3 pp | -0.3 pp | -0.2 pp | -0.2 pp |
| worst Δ final NAV (m), same start |  | -41 | -41 | -42 | -36 | -42 |
| best Δ final NAV (m), same start |  | +13 | +14 | +15 | +33 | +33 |

| Today | Keep the loan | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) | Same dollar delta | Same index units |
|---|---|---|---|---|---|---|
| dry powder today, median (m) | 908 | 808 | 809 | 806 | 775 | 775 |
| calls / NAV today, median |  | 17% | 17% | 17% | 18% | 18% |
| exposure ÷ Keep's today, median | 1.00 | 1.09 | 1.09 | 1.10 | 1.15 | 1.12 |
| starts with a cut replacement |  | 0 | 0 | 0 | 0 | 0 |
| margin calls (Keep) | 0 |  |  |  |  |  |

## At each market bottom (every start running that day)

Lowest and median NAV of each portfolio on the bottom day across the starts running that day (the same starts on every column), the share of those starts where the rotation has more dry powder than Keep's room before a margin call, and where its NAV is higher. Starts after the settled cut run only into the 2022 bottom.

| Bottom | starts | Keep: lowest / median NAV | Keep's exposure (default): lowest / median NAV | Keep's exposure (no haircut): lowest / median NAV | Keep's exposure (quarterly, T-bills only below the band): lowest / median NAV | Same dollar delta: lowest / median NAV | Same index units: lowest / median NAV | Keep's exposure (default): more dry powder / higher NAV | Keep's exposure (no haircut): more dry powder / higher NAV | Keep's exposure (quarterly, T-bills only below the band): more dry powder / higher NAV | Same dollar delta: more dry powder / higher NAV | Same index units: more dry powder / higher NAV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2000–2002 (09 Oct 2002) | 265 | 241 / 390 | 290 / 419 | 290 / 419 | 306 / 436 | 306 / 436 | 306 / 436 | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% |
| 2007–2009 (09 Mar 2009) | 600 | 142 / 293 | 198 / 367 | 199 / 367 | 211 / 384 | 225 / 384 | 234 / 393 | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% |
| 2020 (23 Mar 2020) | 1,176 | 413 / 1,772 | 416 / 1,882 | 416 / 1,888 | 416 / 1,834 | 416 / 1,821 | 416 / 1,806 | 100% / 95% | 100% / 95% | 96% / 86% | 59% / 73% | 85% / 68% |
| 2022 (12 Oct 2022) | 1,309 | 503 / 3,003 | 508 / 3,141 | 508 / 3,145 | 508 / 3,089 | 508 / 3,666 | 508 / 2,986 | 71% / 96% | 72% / 96% | 70% / 90% | 31% / 94% | 46% / 70% |

## The target rule's mechanics

| Per start (median, or total) | Keep's exposure (default) | Keep's exposure (no haircut) | Keep's exposure (quarterly, T-bills only below the band) |
|---|---|---|---|
| rolls (expiries with a decision) | 53 | 53 | 63 |
| rolls skipped (gap ≤ 0) | 0 | 0 | 0 |
| rebalances up (calls sold) | 65 | 65 | 28 |
| rebalances down (calls bought) | 0 | 0 | 0 |
| partial rebalances (T-bills short) | 0 | 0 | 0 |
| calls sold at rebalances (m) | 565 | 567 | 497 |
| calls bought at rebalances (m) | 0 | 0 | 0 |
| SPX sold at rebalances (m) | 0 | 0 | 0 |
| unwind cost (m), median start | 2.2 | 0.0 | 1.9 |
| unwind cost (m), all starts | 7,425 | 0 | 7,521 |
| cut replacements | 0 | 0 | 0 |
| exposure ÷ Keep's today: 5th / median / 95th pct | 1.02 / 1.09 / 1.10 | 1.02 / 1.09 / 1.10 | 1.02 / 1.10 / 1.10 |

Each replacement restores its own slot; the band's purchases in a long fall take over the slots of the calls that then expire (skipped rolls), and its sales remove whole tranches over time, so the weekly ladder thins to a handful of calls. Between monthly checks the live exposure drifts with the delta; on the check days it is inside the band unless nothing was left to fund it (partial).

Run time 769 s.
