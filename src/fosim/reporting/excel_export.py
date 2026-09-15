"""Single-path audit export to Excel (SPEC §5.6): ledger, step-by-step balance sheet, tranche table,
P&L components, so an analyst can re-check the calculation by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fosim.engine.state import COMPONENTS, StrategyResult
from fosim.market.paths import MarketPaths


def balance_sheet_frame(res: StrategyResult, paths: MarketPaths, path: int) -> pd.DataFrame:
    g = paths.grid
    cols = {
        "step": np.arange(g.n_steps + 1), "t_years": g.t, "month": g.month_index, "is_month_end": g.is_month_end,
        "index_level": paths.S[path, :, 0], "held_level": paths.held[path], "iv_short": paths.iv_short[path], "r_short": paths.r_short[path],
    }
    for s in ("cash", "loan", "held_mv", "spot_mv", "option_mv", "illiquid_mv", "illiquid_true", "swap_mtm", "nav", "nav_true", "nav_bid",
              "utilisation", "lending_value", "loan_rate", "cash_rate", "option_rate_5y", "dollar_delta", "dollar_delta_smile", "dollar_gamma",
              "vega_1pt", "theta_year", "rho_100bp", "dq_100bp", "n_tranches", "futures_notional", "effective_leverage", "equity_exposure"):
        cols[s] = res.series(s)[path]
    df = pd.DataFrame(cols)
    il = paths.illiquids
    for j, nm in enumerate(il.names):
        df[f"ill_{nm}_true"] = il.nav_true[path, :, j]
        df[f"ill_{nm}_reported"] = il.nav_reported[path, :, j]
        df[f"ill_{nm}_contributions"] = il.contributions[path, :, j]
        df[f"ill_{nm}_distributions"] = il.distributions[path, :, j]
        df[f"ill_{nm}_unfunded"] = il.unfunded[path, :, j]
    comp = pd.DataFrame(res.recorder.components[path], columns=[f"pnl_{c}" for c in COMPONENTS])
    df = pd.concat([df, comp], axis=1)
    df["pnl_sum"] = comp.sum(axis=1)
    df["nav_change"] = df["nav"].diff()
    df["identity_residual"] = df["nav_change"] - df["pnl_sum"]
    df.loc[0, "identity_residual"] = np.nan
    return df


def export_path_audit(results: dict[str, StrategyResult], paths: MarketPaths, path: int, out_file: str | Path) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine="openpyxl") as xw:
        readme = pd.DataFrame(
            {
                "item": ["path", "n_steps", "dt", "seed", "note"],
                "value": [path, paths.n_steps, paths.grid.freq, paths.seed, "All amounts USD. Ledger rows are cash flows (account 'cash'/'loan'); 'memo:*' rows are non-cash cost memos; 'event' rows record margin/ruin events."],
            }
        )
        readme.to_excel(xw, sheet_name="README", index=False)
        for name, res in results.items():
            balance_sheet_frame(res, paths, path).to_excel(xw, sheet_name=f"{name}_balance_sheet", index=False)
            led = res.ledger.to_frame()
            led = led[led.path == path] if path in res.ledger.ledger_paths else led.iloc[0:0]
            led.to_excel(xw, sheet_name=f"{name}_ledger", index=False)
            tr = res.ledger.tranche_frame()
            if len(tr):
                tr[tr.path == path].to_excel(xw, sheet_name=f"{name}_tranches", index=False)
            ev = res.events
            pd.DataFrame(
                {
                    "metric": ["margin_warnings", "margin_calls", "forced_liquidations", "cure_sales_volume", "forced_sale_volume", "slippage_loss", "liquidity_shortfalls", "ruin", "ruin_step", "min_headroom", "dry_powder_deployments", "dry_powder_deployed_usd", "cash_constrained_purchases", "option_purchases", "option_sales", "roll_cost_usd", "futures_margin_calls", "counterparty_losses"],
                    "value": [ev.margin_warnings[path], ev.margin_calls[path], ev.forced_liquidations[path], ev.cure_sales_volume[path], ev.forced_sale_volume[path], ev.slippage_loss[path], ev.liquidity_shortfalls[path], bool(ev.ruin[path]), int(ev.ruin_step[path]), ev.min_headroom[path], ev.dry_powder_deployments[path], ev.dry_powder_deployed_usd[path], ev.cash_constrained_purchases[path], ev.option_purchases[path], ev.option_sales[path], ev.roll_cost_usd[path], ev.futures_margin_calls[path], ev.counterparty_losses[path]],
                }
            ).to_excel(xw, sheet_name=f"{name}_events", index=False)
    return out_file
