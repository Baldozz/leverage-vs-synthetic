# Keep the Lombard loan, or rotate into long-dated calls?

*Project recap for review, 24 September 2026. All figures are in USD and come from historical market data, September 1997 to
14 September 2026, on the weekly start grid. The results are for illustration only. They are not a recommendation. This version
replaces the recap of 18 September: the rules for replacing a call at expiry changed on 24 September (section 1) and every
number below comes from the new rules.*

---

## 1. The question

The portfolio holds **USD 1 bn of S&P 500 (SPX)** with a **USD 250 m Lombard loan** against it. The bank's lending value is
75 %, the rate is SOFR + 75 bp, and interest is capitalised. The net asset value (NAV) is USD 750 m. There are two options:

| | **Keep the loan** | **Rotate into calls** |
|---|---|---|
| What happens | Nothing changes. The loan stays and interest keeps capitalising. | Every week for 52 weeks, sell some SPX, buy 5-year at-the-money calls on the S&P 500 futures index (SPXFP), and use the rest of the proceeds to repay part of the loan. After one year the loan is gone. |
| Market exposure | USD 1 bn of SPX, which moves with the market | Still USD 1 bn at the end of the build. Each call is sized so that its delta × notional replaces the SPX sold that week. |
| Leverage | Explicit: a loan with a margin call if LTV (loan ÷ lending value) goes above 100 % | Built into the calls: no loan and no margin call. The most a call can lose is its premium. |
| At expiry (rules since 24 Sep) | — | A call that expires **in the money** is replaced the same day by a new 5-year call carrying the **same dollar delta**: the expiring call has a delta of 1, so the new one is sized at units × index ÷ its delta, about twice the units. A call that expires **worthless is replaced** on the same units. Replacements are paid from the payoff, then from the T-bills, then by **selling SPX delta-for-delta** (the exposure sold equals the exposure the calls bring; the excess over the premium goes to T-bills). A replacement that not even all the SPX could fund would be cut to what can be funded — this never happened on any start of this run. |

We looked at three things:
1. **Return**: which portfolio is worth more today?
2. **Risk in a crash**: in a fall of 20 % or more, how close did the loan come to a margin call?
3. **Dry powder**: what could still be borrowed against the holdings — 75 % of the SPX, 0 % of the calls, 90 % of the T-bills, less the loan?

**One consequence of the new roll rule to keep in view.** Doubling the units at every in-the-money expiry, and never reducing
them, makes the rotation a re-leveraging programme: after a few good five-year cycles most of its value is option value. The
earlier rule (same index units) halved the exposure at each good expiry and banked the difference — a de-risking programme. Both
rules are in the tool; the results below are for the new default.

## 2. How the test works

- **Historical data only, no simulation.** Both portfolios are started on **every week from 12 September 1997 to 4 September
  2026** — 1,513 start dates, up to one week before the end day (the app also runs every trading day, ≈ 7,300 starts). Each one is
  held to today with the actual daily prices, rates and dividends. The 50 starts of the last year were still rotating on the end
  day and are shown as they stand; the 311 starts after 22 September 2020 have calls that have not yet expired, so their value
  today is a model mark rather than a settled outcome. Annualised figures use the 1,462 starts held at least a year.
- **Market data** comes from Bloomberg: the S&P 500 with dividends reinvested (net of 15 % withholding tax), the S&P 500 futures
  index the calls are written on, 3-month LIBOR and then Term SOFR for the loan, T-bills for cash, the 5-year Treasury, and
  implied volatility.
- **Call prices** use Black–Scholes at the 5-year implied volatility and the 5-year Treasury yield of each day. Over the period
  the 5-year at-the-money premium averaged **about 16 % of notional** (13.7 % on 18 September 2026). The calls are revalued
  every day.
- **Checks.** Every day and for both portfolios, the model confirms that the change in NAV equals market gains minus interest
  paid, to the cent. The option pricer is checked against textbook values and an independent reference. Every table in the app is
  recomputed by the test suite from direct engine runs; all tests pass.

## 3. Results

### 3.1 Return: value today (starts held at least a year, 1,462 of 1,513)

| Final NAV (USD m, start 750 m) | Lowest | 5th pct | 25th pct | Median | 75th pct | 95th pct | Highest |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep the loan | 906 | 1,237 | 2,739 | 6,588 | 8,482 | 11,328 | 14,240 |
| Rotate into calls | 894 | 1,233 | 3,103 | 9,906 | 14,004 | 26,368 | 34,942 |

| Annualised return to today | Lowest | 5th pct | 25th pct | Median | 75th pct | 95th pct |
|---|---:|---:|---:|---:|---:|---:|
| Keep the loan | 8.8 % | 9.2 % | 12.0 % | 15.6 % | 17.1 % | 24.2 % |
| Rotate into calls | 10.0 % | 10.8 % | 14.3 % | 18.1 % | 19.3 % | 24.5 % |

- **Same start, same end, the rotation ends ahead on 88 % of the start dates**; its final NAV relative to keeping the loan is
  +41 % at the median (5th–95th percentile −1 % to +157 %; worst −2 % for a 16 April 2021 start, best +199 % for a 17 May 2002
  start). The median annualised return is 18.1 % against 15.6 %, +2.1 points a year.
- Every start from 1997 to 2019 ends ahead. The starts behind are the young ones — 2020–21 and 2024–25 starts whose calls have
  not yet had a five-year cycle: they have paid the premium and not yet collected.
- **No call is ever lost under the new rules** (0 of 1,513 starts let a call lapse, against 18 % under the old rule), and no
  replacement had to be cut for lack of SPX.
- **What the rotation holds today.** At the median start, 41 % of its NAV is call value (5th–95th percentile 17 % to 73 %) and 9 %
  is T-bills; the rest is SPX. From a 1997 start it is worth USD 28.6 bn against 12.1 bn for keeping the loan — 21.3 bn of it in
  calls, 5.2 bn in SPX and 2.0 bn in T-bills. That composition is the ratchet of section 1 at work, and it drives the dry-powder
  result in 3.3.

### 3.2 Risk in a crash: margin calls

There were four falls of 20 % or more in the SPX price index since 1997:

| Correction | Peak | Bottom | SPX fall | Back at the peak |
|---|---|---|---:|---|
| 2000–2002 | 24 Mar 2000 | 9 Oct 2002 | −49 % | 30 May 2007 |
| 2007–2009 | 9 Oct 2007 | 9 Mar 2009 | −57 % | 28 Mar 2013 |
| 2020 | 19 Feb 2020 | 23 Mar 2020 | −34 % | 18 Aug 2020 |
| 2022 | 3 Jan 2022 | 12 Oct 2022 | −25 % | 19 Jan 2024 |

- **None of the 1,513 start dates hit a margin call on the loan.** The closest was a start on **24 March 2000**, the top of the
  dot-com bubble. On 9 March 2009 its LTV reached **96 %**: the SPX could have fallen only about 4 % more, or USD 15 m of lending
  value, before the bank called for margin. Its NAV had fallen from 750 m to 142 m.
- On that same start and day the rotation stood at **245 m** with 182 m of dry powder: its 52 calls of 2000–01 had expired
  worthless in 2005–06 and been replaced by selling SPX, so it went into the crash with 52 live calls and no loan.
- **The rotation cannot get a margin call** because it has no loan once the build is over. Its lowest NAV at any bottom was
  USD 225 m, for a start on 1 September 2000.

### 3.3 At each market bottom

The table covers every start running on the bottom day (starts still rotating included). Figures are in USD m, lowest / median
over those starts; dry powder = lending value of what is held − loan.

| Bottom | Starts | NAV, lowest / median: **Keep** | **Rotate** | Highest LTV, Keep | Dry powder, lowest / median: **Keep** | **Rotate** | Rotation has more dry powder on |
|---|---:|---|---|---:|---|---|---:|
| Oct 2002 | 265 | 241 / 390 | 306 / 436 | 72 % | 110 / 222 | 215 / 297 | 94 % of starts |
| Mar 2009 | 600 | 142 / 293 | 225 / 384 | 96 % | 15 / 140 | 144 / 278 | 96 % |
| Mar 2020 | 1,176 | 413 / 1,772 | 416 / 1,821 | 50 % | 247 / 1,244 | 251 / 1,127 | 50 % |
| Oct 2022 | 1,309 | 503 / 3,003 | 508 / 3,666 | 45 % | 313 / 2,166 | 321 / 1,817 | 26 % |

- **In the two deep bear markets (2000–02 and 2007–09) the rotation was ahead on every start date in NAV,** and had more dry
  powder on 94–96 % of them. At the 2009 low its median dry powder was USD 278 m against 140 m for keeping the loan; in the
  worst case 144 m against 15 m. The rotation's losses stop at the premium, while the loan keeps growing as the equity falls.
- **In the short falls of 2020 and 2022 the NAVs were close** (the rotation higher on 73 % and 94 % of starts), but the rotation
  had *less* dry powder on most starts in 2022, and less again today (median 3.7 bn against 4.8 bn; more on only 6 % of starts).
  The reason is the composition of 3.1: after good cycles the rotation's wealth is in calls, which carry **no lending value** in
  this setup, and it has sold SPX — the lendable asset — to pay for them. If the bank lent against in-the-money long-dated calls
  (the sidebar lets you set that lending value; it is 0 % by default), the ordering would reverse. This is a term to obtain
  from the bank, not a result of the market history.

### 3.4 In short

| | Keep the loan | Rotate into calls |
|---|---|---|
| Typical long-run return | ≈ 15.6 % p.a. | ≈ 18.1 % p.a. |
| Bad outcomes | Deep crash with an entry near the peak (LTV 96 % in 2009) | Young starts that have paid premium and not yet collected (2020–21, 2024–25); a long flat market would add the premium burn of a doubled notional |
| Margin call | None in the history, but a close call in 2009 | Impossible once the build is over (no loan) |
| Dry powder in a deep bear market | Low, and in the worst case almost nil | Clearly higher |
| Dry powder after a long rally / today | Higher | Lower, because the calls carry no lending value |

**In the history, rotating into calls under the new rules has beaten keeping the loan on 88 % of start dates, removes the
margin-call risk, and leaves far more dry powder at the bottom of a severe crash.** Its price is a portfolio whose value is
increasingly in options — a re-leveraging programme with a growing premium burn — and less borrowing capacity after long rallies
unless the bank lends against the calls. The roll rule (same dollar delta vs same units) is the main policy decision left to review;
the tool runs both.

## 4. What the tool contains

A browser app with two pages and one sidebar. Every input can be changed: size, loan, spread, lending values of the SPX, the calls
and the T-bills, call tenor, build pace, withholding tax, what an in-the-money call is replaced on (same dollar delta or same
units), whether a worthless call is replaced, what a payoff left after a roll buys (T-bills, SPX, more calls), and daily or weekly
start dates. Nothing runs until *Launch simulation* is pressed.

1. **Historical simulation.** The path of every start date since 1997, drawn as a fan and coloured by start year, held to today or
   to any of the four market bottoms; the NAV and the dry powder of both portfolios on the end day by start date, with the net;
   the distribution of the annualised return; the corrections table; and for each market bottom the worst start's full balance
   sheet (SPX, T-bills, calls, loan, lending value, dry powder, LTV, interest). Results can be downloaded as CSV.
2. **Call premium history.** For every day since 1997: buy a 5-year at-the-money call for a fixed amount, or put the same amount
   into the index, and compare the two at maturity. The chart is built the same way as the J.P. Morgan / Bloomberg chart on the
   same topic.

A full daily record of every start date (NAV, dry powder, balance sheet) can be exported for independent analysis.

## 5. Limits and open points

1. **The long-dated volatility is an estimate, not a dealer quote.** Bloomberg's implied volatility stops at 2 years. The
   5-year figure is extended from it with a fitted relationship. Around the 2009 low, the dealer premiums implied by the
   J.P. Morgan chart were about 25 % higher than ours (roughly 30 % of notional against our 24 %). Our model therefore makes
   calls bought near crash lows look cheaper than they were — and under the new rule every good expiry buys about twice the
   notional, so the premium level matters more than before. **Before any decision, check current 5-year premiums with dealers.
   Their historical 5-, 7- and 10-year marks would replace the estimate.**
2. **No trading costs.** Bid/ask on long-dated OTC options, which can be wide in stress, is not deducted. Counterparty risk
   and collateral terms (CSA) are not modelled either.
3. **Lombard terms are fixed, and calls carry no lending value.** In practice a bank can cut the lending value or the facility
   in a crisis; a margin call is flagged but not acted on (no forced sale). Whether the bank lends against in-the-money calls, and
   at what rate, decides the dry-powder comparison after rallies (3.3).
4. **The roll rule is a leverage choice.** Rolling on the same dollar delta doubles the option units at every in-the-money
   expiry; the value of the rotation is then dominated by calls and its premium burn grows with each cycle. A flat market lasting
   a full cycle would cost more than the history shows. The same-units rule is the conservative alternative.
5. **Exposure is matched at purchase and at each roll.** Between expiries the calls' delta moves with the market, so the rotated
   exposure is not held at exactly the loan portfolio's.
6. **Overlapping periods.** 1,513 weekly start dates over 29 years give only about six independent 5-year windows. The
   percentiles show the range of past outcomes. They are not probabilities.
7. **Not modelled:** taxes beyond dividend withholding (the SPX sold to fund replacements would realise gains), currency risk
   (everything is in USD), and any use of the loan proceeds outside the two portfolios.
8. **Data to request from the bank:** a history of 5-year at-the-money premiums or vols (and 7-, 10-year), live indications in
   size with bid/ask, the lending values it would apply to SPX, long-dated calls and T-bills, the loan's rate basis and margin
   mechanics, and the collateral terms for the options.
