# Keep the Lombard loan, or rotate into long-dated calls?

*Project recap for review, 18 September 2026. All figures are in USD and come from historical market data, September 1997 to
14 September 2026. The results are for illustration only. They are not a recommendation.*

---

## 1. The question

The portfolio holds **USD 1 bn of S&P 500 (SPX)** with a **USD 250 m Lombard loan** against it. The bank's lending value is
75 %, the rate is SOFR + 75 bp, and interest is capitalised. The net asset value (NAV) is USD 750 m. There are two options:

| | **Keep the loan** | **Rotate into calls** |
|---|---|---|
| What happens | Nothing changes. The loan stays and interest keeps capitalising. | Every week for 52 weeks, sell some SPX, buy 5-year at-the-money calls on the S&P 500, and use the rest of the proceeds to repay part of the loan. After one year the loan is gone. |
| Market exposure | USD 1 bn of SPX, which moves with the market | Still USD 1 bn at the end of the build. Each call is sized so that its delta × notional replaces the SPX sold that week. |
| Leverage | Explicit: a loan with a margin call if LTV (loan ÷ lending value) goes above 100 % | Built into the calls: no loan and no margin call. The most the calls can lose is their premium. |
| At expiry | — | A call that expires **in the money** is replaced the same day by a new 5-year call on the same number of index units, paid from the payoff. A call that expires **worthless is not replaced**, so that part of the exposure is lost. |

We looked at three things:
1. **Return**: which portfolio is worth more today?
2. **Risk in a crash**: in a fall of 20 % or more, how close did the loan come to a margin call?
3. **Dry powder**: at the market bottom, how much could be borrowed to buy more (lending value − loan + cash)?

## 2. How the test works

- **Historical data only, no simulation.** Both portfolios are started on **every week from September 1997 to September
  2020**, 1,202 start dates in all. Each one is held to today with the actual daily prices, rates and dividends. The last
  start is the latest one whose calls have all expired inside the data.
- **Market data** comes from Bloomberg. It covers the S&P 500 with dividends reinvested (net of 15 % withholding tax), the
  S&P 500 futures index the calls are written on, 3-month LIBOR and then Term SOFR for the loan, T-bills for cash, the 5-year
  Treasury, and implied volatility.
- **Call prices** use Black–Scholes at the 5-year implied volatility and the 5-year Treasury yield of each day. Over the
  period the 5-year at-the-money premium averaged **about 16 % of notional** (range 12–22 % at the start dates tested). The
  calls are revalued every day.
- **Checks.** Every day and for both portfolios, the model confirms that the change in NAV equals market gains minus
  interest paid, to the cent. The option pricer is checked against textbook values and an independent reference. All 151
  automated tests pass.

## 3. Results

### 3.1 Return: value today

| Final NAV (USD m, start 750 m) | Lowest | 5th pct | 25th pct | Median | 75th pct | 95th pct | Highest |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep the loan | 2,011 | 2,548 | 4,186 | 7,312 | 8,818 | 11,585 | 14,240 |
| Rotate into calls | 1,971 | 2,560 | 4,242 | 5,944 | 7,226 | 10,778 | 14,736 |

| Annualised return to today | 5th pct | 25th pct | Median | 75th pct | 95th pct |
|---|---:|---:|---:|---:|---:|
| Keep the loan | 9.2 % | 11.9 % | 15.3 % | 16.5 % | 18.3 % |
| Rotate into calls | 7.5 % | 10.2 % | 15.5 % | 16.8 % | 18.5 % |

- **On a typical start date the two are about the same.** The median annualised return is 15.3 % for Keep the loan and
  15.5 % for Rotate into calls. The rotation ends ahead on 52 % of start dates.
  (The medians of the final NAV differ more than the returns do because the start dates are held for different lengths of time.)
- **The rotation's weak results all come from one cause: calls that expired worthless and were not replaced.** On 18 % of
  start dates every call expired worthless, and the portfolio had no option exposure after that:
  - starts from September 1997 to mid-2000, whose calls expired in the 2002–03 bear market;
  - starts from late 2003 to early 2005, whose calls expired around the 2008–09 low.

  From a 1997 start, for example, the rotation is worth about USD 8.3 bn today against 12.0 bn for keeping the loan. For
  every start from mid-2005 onward, at least some calls expired in the money and were rolled.

### 3.2 Risk in a crash: margin calls

There were four falls of 20 % or more in the SPX price index since 1997:

| Correction | Peak | Bottom | SPX fall | Back at the peak |
|---|---|---|---:|---|
| 2000–2002 | 24 Mar 2000 | 9 Oct 2002 | −49 % | 30 May 2007 |
| 2007–2009 | 9 Oct 2007 | 9 Mar 2009 | −57 % | 28 Mar 2013 |
| 2020 | 19 Feb 2020 | 23 Mar 2020 | −34 % | 18 Aug 2020 |
| 2022 | 3 Jan 2022 | 12 Oct 2022 | −25 % | 19 Jan 2024 |

- **None of the 1,202 start dates hit a margin call on the loan.** The closest was a start on **24 March 2000**, the top
  of the dot-com bubble. On 9 March 2009 its LTV reached **96 %**. The SPX could have fallen only about 4 % more, or
  USD 15 m of lending value, before the bank called for margin. Its NAV had fallen from 750 m to 142 m.
- **The rotation cannot get a margin call** because it has no loan. Its lowest NAV at any bottom was USD 269 m, for a start
  on 12 October 2007, the market peak.

### 3.3 At each market bottom

The table covers starts whose rotation was finished at least a month before the bottom. Figures are in USD m, with the
worst case first and then the typical case.

| Bottom | Starts | NAV, lowest / median: **Keep** | **Rotate** | Highest LTV, Keep | Dry powder, lowest / median: **Keep** | **Rotate** |
|---|---:|---|---|---:|---|---|
| Oct 2002 | 210 | 241 / 348 | 306 / 409 | 72 % | 110 / 188 | 215 / 291 |
| Mar 2009 | 545 | 142 / 293 | 269 / 410 | 96 % | 15 / 138 | 179 / 307 |
| Mar 2020 | 1,121 | 521 / 1,820 | 559 / 1,635 | 45 % | 325 / 1,273 | 371 / 1,223 |
| Oct 2022 | 1,202 | 788 / 3,126 | 780 / 2,681 | 33 % | 527 / 2,251 | 502 / 1,955 |

- **In the two deep bear markets (2000–02 and 2007–09) the rotation was ahead for every start date,** both in NAV and in
  dry powder. At the 2009 low its median dry powder was USD 307 m against 138 m for keeping the loan. In the worst case it
  was 179 m against 15 m. The rotation's losses stop at the premium, while the loan keeps growing as the equity falls.
- **In the short falls of 2020 and 2022 the two were about even.** The rotation was slightly behind in 2022: its dry powder
  was higher on only 29 % of start dates. When the fall is shallow the loan is never under pressure, and premiums are a cost
  that brings nothing back.
- **Today, keeping the loan leaves more borrowing capacity** (median 5.4 bn against 4.3 bn). That portfolio holds more
  SPX, and in this setup the calls carry no lending value.

### 3.4 In short

| | Keep the loan | Rotate into calls |
|---|---|---|
| Typical long-run return | ≈ 15.3 % p.a. | ≈ 15.5 % p.a. |
| Bad outcomes | Deep crash with an entry near the peak (LTV 96 % in 2009) | Calls expire worthless and lapse (18 % of starts; weaker return afterwards) |
| Margin call | None in the history, but a close call in 2009 | Impossible (no loan) |
| Dry powder in a deep bear market | Low, and in the worst case almost nil | Clearly higher |
| Dry powder in a shallow correction or today | Slightly higher | Slightly lower |

**In the history, rotating into calls has not cost return on a typical start date, removes the margin-call risk, and
leaves far more dry powder at the bottom of a severe crash.** Its specific weakness is a call that expires worthless after
five poor years: under the current rule that exposure is lost and not rebuilt. The tool lets you replace worthless calls
anyway. This is the main policy decision left to review.

## 4. What the tool contains

A browser app with two pages. Every input can be changed: size, loan, spread, lending values, call tenor, build pace,
withholding tax, what happens to worthless calls and to payoffs left after a roll, and daily or weekly start dates.

1. **Any start date since 1997.** The path of every start date, drawn as a fan and coloured by start year. It can be held to
   today or to any of the four market bottoms. The page also shows the spread of final values, the corrections table, and
   a table for each market bottom with the worst start's full balance sheet (SPX, calls, cash, loan, LTV, dry powder).
   Results can be downloaded as CSV.
2. **Call premium history.** For every day since 1997: buy a 5-year at-the-money call for a fixed amount, or put the same
   amount into the index, and compare the two at maturity. The chart is built the same way as the J.P. Morgan / Bloomberg
   chart on the same topic.

A full daily record of every start date (NAV, dry powder, balance sheet) can be exported for independent analysis.

## 5. Limits and open points

1. **The long-dated volatility is an estimate, not a dealer quote.** Bloomberg's implied volatility stops at 2 years. The
   5-year figure is extended from it with a fitted relationship. Around the 2009 low, the dealer premiums implied by the
   J.P. Morgan chart were about 25 % higher than ours (roughly 30 % of notional against our 24 %). Our model therefore makes
   calls bought near crash lows look cheaper than they were. **Before any decision, check current 5-year premiums with
   dealers. Their historical 5-, 7- and 10-year marks would replace the estimate.**
2. **No trading costs.** Bid/ask on long-dated OTC options, which can be wide in stress, is not deducted. Counterparty risk
   and collateral terms (CSA) are not modelled either.
3. **Lombard terms are fixed.** In practice a bank can cut the lending value or the facility in a crisis. A margin call is
   flagged but not acted on: no forced sale is modelled.
4. **Exposure is matched once.** The calls are sized to replace the SPX sold on the day they are bought. After that their
   delta moves with the market, so the rotated exposure is not held at exactly USD 1 bn.
5. **Overlapping periods.** 1,202 start dates over 23 years give only about five independent 5-year windows. The
   percentiles show the range of past outcomes. They are not probabilities.
6. **Not modelled:** taxes beyond dividend withholding, currency risk (everything is in USD), and any use of the
   loan proceeds outside the two portfolios.
7. **Data still pending:** dealer vol marks (see point 1). The 7-year Treasury is observed from 2009 and interpolated between the 5- and 10-year yields before that.
