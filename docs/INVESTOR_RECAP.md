# Keep the Lombard loan, or rotate into long-dated calls?

*Project recap for review, 25 September 2026. All figures are in USD and come from historical market data, September 1997 to
14 September 2026, on the weekly start grid. The results are for illustration only. They are not a recommendation. This version
replaces the recap of 24 September: the rotation now keeps the same market exposure as the loan portfolio (section 1), the bank is
assumed to call margin at 90 % of the lending value, and every number below comes from these rules.*

---

## 1. The question

The portfolio holds **USD 1 bn of S&P 500 (SPX)** with a **USD 250 m Lombard loan** against it. The bank's lending value is
75 %, the rate is SOFR + 75 bp, and interest is capitalised. The net asset value (NAV) is USD 750 m. There are two options:

| | **Keep the loan** | **Rotate into calls** |
|---|---|---|
| What happens | Nothing changes. The loan stays and interest keeps capitalising. | Every week for 52 weeks, sell some SPX, buy 5-year at-the-money calls on the S&P 500 futures index (SPXFP), and use the rest of the proceeds to repay part of the loan. After one year the loan is gone. |
| Market exposure | USD 1 bn of SPX, which moves with the market | The same as the loan portfolio's, by construction: each call is sized so that its delta × notional replaces the SPX sold that week, and the rules below keep the exposure matched afterwards. |
| Leverage | Explicit: a loan, and a margin call when the loan exceeds **90 % of the lending value** (an LTV of 90 %; an assumption to confirm with the bank) | Built into the calls: no loan and no margin call. The most a call can lose is its premium. |
| At expiry | — | The expiring call is settled and replaced by a new 5-year call sized to restore the exposure it was bought to carry, so that the exposure the holdings were bought to carry equals the loan portfolio's again. The replacement is paid from the payoff, then from the T-bills, then by selling SPX. If the exposure is already there (rare), nothing is bought and the payoff stays in T-bills. |
| Every quarter-end | — | The **live** exposure (SPX plus the calls' delta of the day) is checked against the loan portfolio's. Above +10 %, calls are sold — the most in-the-money first — at the model price less 1 vol point, and the proceeds go to T-bills. Below −10 %, ATM calls are bought from the T-bills, as far as the T-bills go. Nothing is sold to fund it (an alternative that does sell SPX is in the tool and in section 3.5). |

We looked at three things:
1. **Return**: which portfolio is worth more today?
2. **Risk in a crash**: in a fall of 20 % or more, did the loan get a margin call, and how close did it come?
3. **Dry powder**: what could still be borrowed against the holdings before a margin call — 90 % of the lending value (75 % of the
   SPX, 0 % of the calls, 90 % of the T-bills), less the loan?

**Why the exposure is matched.** The earlier rule (a new call carrying the same dollar delta as the expiring one) doubled the option
units at every profitable expiry and never reduced them: after a few good cycles the rotation carried two to five times the loan
portfolio's exposure, and its outperformance was leverage, not a better structure. The rules above make the two portfolios
comparable: same exposure, different financing — a loan against the stock, or premium paid for calls. That is the question.

## 2. How the test works

- **Historical data only, no simulation.** Both portfolios are started on **every week from 12 September 1997 to 4 September
  2026** — 1,513 start dates, up to one week before the end day (the app also runs every trading day, ≈ 7,300 starts). Each one is
  held to today with the actual daily prices, rates and dividends. The 50 starts of the last year were still rotating on the end
  day and are shown as they stand; the 311 starts after 22 September 2020 have calls that have not yet expired, so their value
  today is a model mark rather than a settled outcome. The 1,202 starts up to September 2020 are *settled*: every call has
  expired at least once. Annualised figures use the 1,462 starts held at least a year.
- **Market data** comes from Bloomberg: the S&P 500 with dividends reinvested (net of 15 % withholding tax), the S&P 500 futures
  index the calls are written on, 3-month LIBOR and then Term SOFR for the loan, T-bills for cash, the 5-year Treasury, and
  implied volatility.
- **Call prices** use Black–Scholes at the 5-year implied volatility and the 5-year Treasury yield of each day. Over the period
  the 5-year at-the-money premium averaged **about 16 % of notional** (13.9 % on 4 September 2026). The calls are revalued
  every day; a call sold early at a quarter-end check is sold 1 vol point below the model price.
- **Checks.** Every day and for both portfolios, the model confirms that the change in NAV equals market gains minus interest
  paid and the cost of early sales, to the cent. The option pricer is checked against textbook values and an independent
  reference. The two earlier roll rules still reproduce their previous results to the cent. Every table in the app is recomputed
  by the test suite from direct engine runs; all tests pass.

## 3. Results

### 3.1 Return: value today (starts held at least a year, 1,462 of 1,513)

| Final NAV (USD m, start 750 m) | Lowest | 5th pct | 25th pct | Median | 75th pct | 95th pct | Highest |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep the loan | 906 | 1,237 | 2,739 | 6,588 | 8,482 | 11,328 | 14,240 |
| Rotate into calls | 894 | 1,233 | 2,826 | 6,630 | 8,448 | 11,612 | 14,833 |

| Annualised return to today | Lowest | 5th pct | 25th pct | Median | 75th pct | 95th pct |
|---|---:|---:|---:|---:|---:|---:|
| Keep the loan | 8.8 % | 9.2 % | 12.0 % | 15.6 % | 17.1 % | 24.2 % |
| Rotate into calls | 8.5 % | 9.1 % | 12.0 % | 15.8 % | 17.3 % | 24.4 % |

- **The two portfolios earn the same.** Same start, same end, the rotation ends ahead on 65 % of all start dates and on **72 % of
  the settled ones**; its final NAV relative to keeping the loan is +1.1 % at the median of the settled starts (5th–95th
  percentile −3.7 % to +3.9 %; worst −9.9 % for a 1 September 2000 start, best +5.3 % for a 2 January 2009 start). The median
  annualised return is 15.8 % against 15.6 %, +0.1 points a year. On the 260 unsettled starts of 2020–25 the rotation is
  slightly behind (−0.6 % at the median, ahead on 30 %): they have paid premium and not yet collected.
- **Financing cost, in plain terms.** Keeping the loan costs the interest, which capitalises. Rotating costs the premium, which is
  partly recovered by the payoffs: from a 1997 start the rotation has paid USD 2,556 m of premiums and received 2,105 m of payoffs
  over 29 years, against 419 m of interest on the loan. On this history the two came out about even.
- **What the rotation holds today.** At the median start, 68 % of its NAV is SPX, 18 % is call value (5th–95th percentile 12 % to
  33 %) and 17 % is T-bills — the payoffs not needed for replacements. From a 1997 start it is worth USD 12.3 bn against 12.1 bn
  for keeping the loan: 6.5 bn in SPX, 3.0 bn in calls, 2.8 bn in T-bills, 18 calls alive.
- **No call was ever lost and no replacement had to be cut** for lack of SPX on any of the 1,513 starts.

### 3.2 Risk in a crash: margin calls

There were four falls of 20 % or more in the SPX price index since 1997:

| Correction | Peak | Bottom | SPX fall | Back at the peak |
|---|---|---|---:|---|
| 2000–2002 | 24 Mar 2000 | 9 Oct 2002 | −49 % | 30 May 2007 |
| 2007–2009 | 9 Oct 2007 | 9 Mar 2009 | −57 % | 28 Mar 2013 |
| 2020 | 19 Feb 2020 | 23 Mar 2020 | −34 % | 18 Aug 2020 |
| 2022 | 3 Jan 2022 | 12 Oct 2022 | −25 % | 19 Jan 2024 |

- **With the bank calling at 90 % of the lending value, 30 of the 1,513 start dates hit a margin call on the loan** — starts
  between 2 July 1999 and 8 September 2000, the top of the dot-com bubble, all of them on 9 March 2009 at the bottom of the
  financial crisis, ten years and two bear markets later. The worst is the start of **24 March 2000**: on 9 March 2009 its LTV
  reached **96 %**, the loan (366 m after nine years of capitalised interest) exceeded 90 % of the lending value by USD 23 m, and
  its NAV had fallen from 750 m to 142 m. Had the bank waited for the full lending value (LTV 100 %), no start would have been
  called; the same start would have had 15 m of room left.
- On that same start and day the rotation stood at **224 m** with 146 m of dry powder and no loan: 216 m of SPX, 50 calls of
  2005–06 deep out of the money, worth 7 m.
- **The rotation cannot get a margin call** because it has no loan once the build is over. Its lowest NAV at any bottom was
  USD 211 m, for a start on 1 September 2000.

### 3.3 At each market bottom

The table covers every start running on the bottom day (starts still rotating included). Figures are in USD m, lowest / median
over those starts; dry powder = 90 % of the lending value of what is held − loan.

| Bottom | Starts | NAV, lowest / median: **Keep** | **Rotate** | Rotation's NAV higher on | Highest LTV, Keep (margin calls) | Dry powder, lowest / median: **Keep** | **Rotate** | Rotation has more dry powder on |
|---|---:|---|---|---:|---|---|---|---:|
| Oct 2002 | 265 | 241 / 390 | 306 / 436 | 100 % | 72 % (0) | 71 / 170 | 194 / 262 | 99 % |
| Mar 2009 | 600 | 142 / 293 | 211 / 384 | 100 % | 96 % (30) | −23 / 96 | 126 / 248 | 99 % |
| Mar 2020 | 1,176 | 413 / 1,772 | 416 / 1,834 | 86 % | 50 % (0) | 197 / 1,088 | 204 / 1,201 | 96 % |
| Oct 2022 | 1,309 | 503 / 3,003 | 508 / 3,089 | 90 % | 45 % (0) | 257 / 1,910 | 283 / 1,875 | 70 % |

- **In the two deep bear markets (2000–02 and 2007–09) the rotation was ahead on every start date in NAV** and had more dry
  powder on 99 % of them. At the 2009 low its median dry powder was USD 248 m against 96 m for keeping the loan; in the worst
  case 126 m against a margin call. The rotation's losses stop at the premium, while the loan keeps growing as the equity falls.
- **In the short falls of 2020 and 2022 the NAVs were close** (the rotation higher on 86 % and 90 % of starts) and so was the dry
  powder.
- **Today, after the long rally, the rotation has less dry powder on every start** (median 3.6 bn against 4.3 bn). Its wealth is
  partly in calls, which carry **no lending value** in this setup, and partly in T-bills, which carry 90 %; it has sold SPX — the
  lendable asset — to pay the premiums. If the bank lent against in-the-money long-dated calls (the sidebar lets you set that
  lending value; it is 0 % by default), the ordering would change. This is a term to obtain from the bank, not a result of the
  market history.

### 3.4 How well the exposure is matched

The two portfolios have the same exposure on every purchase day and after every roll. Between quarter-end checks the calls'
delta moves with the market — up in a rally, down in a fall — so the rotation's live exposure drifts: on the 1997 start it ran
between 0.69× and 1.26× the loan portfolio's, and today it sits at the top of the band on most starts (1.10×) after the rally.
The drift is one-sided in a long bear market: below the band the rule buys calls only with T-bills, and a start that walks into
2001–02 or 2008 before its first expiry has none, so it stays under-exposed until its first payoff (the 1997 start sat at 0.7×
for a year around the 2002 low). That under-exposure is part of why its NAV holds up better at the bottoms in 3.3. A typical
start made 29 quarter-end sales and no purchase over its life; the 1 vol point given up on those sales cost 2 m at the median.

### 3.5 The other roll rules, side by side

The tool keeps the earlier rules, and the comparison in `reports/roll_rule_comparison.md` runs all of them on the same starts
(settled starts held at least a year, 1,202):

| | Keep the loan | Match the exposure (default) | Same, selling SPX for calls below the band | Same dollar delta (the earlier default) | Same index units |
|---|---:|---:|---:|---:|---:|
| Median annualised return | 15.3 % | 15.4 % | 15.4 % | 17.7 % | 15.5 % |
| Rotation ahead of Keep on | | 72 % | 82 % | 99 % | 53 % |
| Worst / best same-start difference in final NAV | | −696 / +593 m | −262 / +593 m | −23 / +21,672 m | −2,176 / +498 m |
| Lowest NAV at the 2009 bottom | 142 m | 211 m | 232 m | 225 m | 234 m |
| Exposure ÷ Keep's today, median | 1.00 | 1.10 | 1.10 | 2.40 | 0.98 |
| Calls as a share of NAV today, median | | 18 % | 18 % | 43 % | 13 % |

The dollar-delta rule's numbers are those of a portfolio carrying 2.4× the exposure: a bigger bet, not a better structure. The
variant that sells SPX for calls when the T-bills run out keeps the exposure matched through the bear markets and comes out
slightly ahead of the default on most lines (except the 2002 low), at the cost of turning stock into options at the low; it is an
option in the tool, not the default.

### 3.6 In short

| | Keep the loan | Rotate into calls |
|---|---|---|
| Typical long-run return | ≈ 15.6 % p.a. | ≈ 15.8 % p.a. — the same |
| Bad outcomes | A deep crash with an entry near the peak: 30 starts of 1999–2000 margin-called in March 2009 (LTV up to 96 %) | Young starts that have paid premium and not yet collected (2020–25); a long flat market, where the premium is paid for nothing |
| Margin call | Possible, and it happened in the history at the 90 % level | Impossible once the build is over (no loan) |
| Dry powder in a deep bear market | Low, and in the worst case gone | Clearly higher |
| Dry powder after a long rally / today | Higher | Lower, because the calls carry no lending value |

**In the history, rotating into calls with the exposure matched has earned the same as keeping the loan, removed the margin-call
risk, and left far more dry powder at the bottom of a severe crash.** Its price is a portfolio with less borrowing capacity after
long rallies unless the bank lends against the calls, and a premium that is paid in every market and recovered only in rising
ones.

## 4. What the tool contains

A browser app with two pages and one sidebar. Every input can be changed: size, loan, spread, lending values of the SPX, the calls
and the T-bills, the LTV at which the bank calls, call tenor, build pace, withholding tax, what an expiring call is replaced on
(the exposure match, the same dollar delta, the same index units), how often and how tightly the exposure is checked, the
haircut on early sales, what is bought below the band, whether a worthless call is replaced, what a payoff left after a roll buys
under the earlier rules, and daily or weekly start dates. Nothing runs until *Launch simulation* is pressed.

1. **Historical simulation.** The path of every start date since 1997, drawn as a fan and coloured by start year, held to today or
   to any of the four market bottoms; the NAV and the dry powder of both portfolios on the end day by start date, with the net and
   the loan's LTV; the rotation's exposure relative to the loan portfolio's over time; the distribution of the annualised return;
   the corrections table; and for each market bottom the worst start's full balance sheet (SPX, T-bills, calls, loan, lending
   value, dry powder, LTV, interest). Results can be downloaded as CSV.
2. **Call premium history.** For every day since 1997: buy a 5-year at-the-money call for a fixed amount, or put the same amount
   into the index, and compare the two at maturity. The chart is built the same way as the J.P. Morgan / Bloomberg chart on the
   same topic.

A full daily record of every start date (NAV, dry powder, exposure, balance sheet) and the log of every roll and quarter-end trade
can be exported for independent analysis.

## 5. Limits and open points

1. **The long-dated volatility is an estimate, not a dealer quote.** Bloomberg's implied volatility stops at 2 years. The
   5-year figure is extended from it with a fitted relationship. Around the 2009 low, the dealer premiums implied by the
   J.P. Morgan chart were about 25 % higher than ours (roughly 30 % of notional against our 24 %). Our model therefore makes
   calls bought near crash lows look cheaper than they were. **Before any decision, check current 5-year premiums with dealers.
   Their historical 5-, 7- and 10-year marks would replace the estimate.**
2. **Trading costs are approximated.** Calls sold early are priced 1 vol point below the model mark; bid/ask on the purchases,
   which can be wide in stress, is not deducted. Counterparty risk and collateral terms (CSA) are not modelled either.
3. **Lombard terms are assumed.** The 75 % lending value and the margin call at 90 % of it are placeholders to confirm with the
   bank; a margin call is flagged but not acted on (no forced sale), and in practice a bank can cut the lending value or the
   facility in a crisis. Whether the bank lends against in-the-money calls, and at what rate, decides the dry-powder comparison
   after rallies (3.3).
4. **The exposure is matched within ±10 %, quarterly.** Between checks it drifts with the calls' delta, and in a long fall with
   no T-bills it stays below the band until the next expiry (3.4). Differences smaller than the band should not be over-read.
5. **Overlapping periods.** 1,513 weekly start dates over 29 years give only about six independent 5-year windows. The
   percentiles show the range of past outcomes. They are not probabilities.
6. **Not modelled:** taxes beyond dividend withholding (the SPX sold to fund replacements would realise gains), currency risk
   (everything is in USD), and any use of the loan proceeds outside the two portfolios.
7. **Data to request from the bank:** a history of 5-year at-the-money premiums or vols (and 7-, 10-year), live indications in
   size with bid/ask, the lending values it would apply to SPX, long-dated calls and T-bills, the loan's rate basis and the LTV
   at which it calls margin, and the collateral terms for the options.
