"""Pydantic v2 configuration schema (SPEC §9).

Fail-fast validation: unknown keys are rejected (`extra="forbid"`), ranges are enforced, weights
must sum to one, the correlation matrix must be symmetric with unit diagonal (PSD repair happens
in ``fosim.market.correlation`` with a loud warning), tenor grids must be consistent, and the
USD-only guard (SPEC §3, test 33) rejects any non-USD currency field, FX block or quanto option.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fosim.engine.conventions import DayCount, ReturnType, StepFrequency

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
NonNeg = Annotated[float, Field(ge=0.0)]
Pos = Annotated[float, Field(gt=0.0)]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------- run / portfolio


class RunConfig(_Base):
    n_paths: Annotated[int, Field(ge=1, le=2_000_000)] = 10_000
    horizon_years: Annotated[float, Field(gt=0, le=50)] = 10.0
    dt: StepFrequency = "monthly"
    seed: Annotated[int, Field(ge=0)] = 20260915
    currency: Literal["USD"] = "USD"
    label: str = ""


class PortfolioWeights(_Base):
    equities: Prob
    illiquids: Prob

    @model_validator(mode="after")
    def _sum(self) -> PortfolioWeights:
        if not math.isclose(self.equities + self.illiquids, 1.0, abs_tol=1e-9):
            raise ValueError("portfolio.weights must sum to 1.0")
        return self


class PortfolioConfig(_Base):
    nav0: Pos = 1_000_000_000.0
    weights: PortfolioWeights


# ----------------------------------------------------------------------------- leverage / loan


class LtvBase(_Base):
    equity_index: Prob = 0.5
    illiquid: Prob = 0.0
    cash: Prob = 0.9


class LtvStressRule(_Base):
    iv_short_above: Pos | None = None
    index_drawdown_below: Annotated[float, Field(ge=-1.0, le=0.0)] | None = None
    multiplier: Prob = 0.8

    @model_validator(mode="after")
    def _one_trigger(self) -> LtvStressRule:
        if self.iv_short_above is None and self.index_drawdown_below is None:
            raise ValueError("ltv_stress_schedule rule needs iv_short_above or index_drawdown_below")
        return self


class MarginThresholds(_Base):
    warn: Pos = 0.9
    call: Pos = 1.0
    closeout: Pos = 1.1
    target_after_cure: Pos = 0.8

    @model_validator(mode="after")
    def _order(self) -> MarginThresholds:
        if not (self.target_after_cure < self.warn <= self.call < self.closeout):
            raise ValueError(
                "thresholds must satisfy target_after_cure < warn <= call < closeout"
            )
        return self


class StrategyDConfig(_Base):
    enabled: bool = False
    cash_buffer_pct_nav: Prob = 0.05


class LeverageConfig(_Base):
    ratio_of_nav: Annotated[float, Field(ge=0.0, le=2.0)] = 0.25
    allocation_of_borrowed_funds: Literal["pro_rata", "equity_only"] = "pro_rata"
    day_count: DayCount = "ACT/360"
    interest: Literal["pay_cash", "capitalise"] = "pay_cash"
    capitalisation_frequency: Literal["monthly", "step"] = "monthly"  # capitalise: add accrued interest to the loan at month-ends (monthly roll) or every step
    ltv_base: LtvBase = LtvBase()
    ltv_stress_schedule: list[LtvStressRule] = []
    thresholds: MarginThresholds = MarginThresholds()
    cure_grace_days: Annotated[int, Field(ge=0)] = 2
    rebalancing: Literal["buy_and_hold", "maintain_target_leverage"] = "buy_and_hold"
    slippage_bps: NonNeg = 25.0
    slippage_stress_multiplier: Annotated[float, Field(ge=1.0)] = 4.0
    strategy_d: StrategyDConfig = StrategyDConfig()


class SpreadTier(_Base):
    utilisation_below: Pos
    spread: Annotated[float, Field(ge=-0.05, le=0.2)]


class LoanTermsConfig(_Base):
    base_index: Literal["SOFR"] = "SOFR"
    base_convention: Literal["overnight_compounded", "term"] = "overnight_compounded"
    reset_months: Annotated[int, Field(ge=1, le=12)] = 1
    base_floor: Annotated[float, Field(ge=-0.05, le=0.2)] | None = 0.0
    spread_tiers: list[SpreadTier]
    tier_basis: Literal["facility_utilisation", "loan_size"] = "facility_utilisation"
    commitment_fee_bps: NonNeg = 0.0
    arrangement_fee_bps: NonNeg = 0.0
    facility_limit: Pos = 400_000_000.0
    rate_type: Literal["floating", "fixed", "floating_swapped"] = "floating"
    fixed_years: Pos | None = None
    fixed_rate: float | None = None
    renewal_years: Pos | None = None
    renewal_spread_stress: NonNeg = 0.0

    @field_validator("spread_tiers")
    @classmethod
    def _tiers(cls, v: list[SpreadTier]) -> list[SpreadTier]:
        if not v:
            raise ValueError("loan_terms.spread_tiers must have at least one tier")
        ub = [t.utilisation_below for t in v]
        if ub != sorted(ub) or len(set(ub)) != len(ub):
            raise ValueError("spread_tiers must be strictly increasing in utilisation_below")
        return v

    @model_validator(mode="after")
    def _fixed(self) -> LoanTermsConfig:
        if self.rate_type in ("fixed", "floating_swapped") and (
            self.fixed_years is None or self.fixed_rate is None
        ):
            raise ValueError("rate_type fixed/floating_swapped requires fixed_years and fixed_rate")
        return self


# ----------------------------------------------------------------------------- equities


class DividendCurve(_Base):
    tenors: list[Pos]
    q: list[float]

    @model_validator(mode="after")
    def _grid(self) -> DividendCurve:
        _check_grid(self.tenors, {"q": self.q}, "pricing_dividend_curve")
        return self


class EquityIndexConfig(_Base):
    name: str = Field(min_length=1)
    underlying_type: Literal["price_return", "total_return", "net_total_return", "excess_return"] = "price_return"
    option_book_weight: Prob = 1.0
    weight_in_equity_sleeve: Prob = 1.0
    pricing_dividend_curve: DividendCurve
    dividend_stress_cut: Prob = 0.0
    dividend_stress_trigger: Annotated[float, Field(ge=-1.0, le=0.0)] = -0.25
    spot: Pos = 100.0
    expected_return: Annotated[float, Field(gt=-0.9, lt=1.0)]
    return_type: ReturnType = "arithmetic"
    dividend_yield: Annotated[float, Field(ge=0.0, le=0.2)] = 0.0
    realized_vol: Annotated[float, Field(ge=0.0, le=2.0)]
    dividend_wht: Prob = 0.0
    jump_mean: float = -0.15
    jump_vol: NonNeg = 0.10

    @model_validator(mode="after")
    def _consistency(self) -> EquityIndexConfig:
        if self.underlying_type == "excess_return" and self.dividend_yield > 0:
            raise ValueError(f"index {self.name}: excess_return index cannot pay separate dividends (dividend_yield must be 0)")
        if self.underlying_type == "total_return":
            if any(abs(q) > 0 for q in self.pricing_dividend_curve.q):
                raise ValueError(
                    f"index {self.name}: underlying_type=total_return requires a zero "
                    "pricing_dividend_curve (q = 0 in pricing) — flags are inconsistent"
                )
        return self


class HeldPortfolioConfig(_Base):
    betas: dict[str, float]  # composition of the held book: Δln H = Σ β_i Δln S_i (+ alpha, tracking error)
    tracking_error_vol: Annotated[float, Field(ge=0.0, le=1.0)] = 0.03
    expected_alpha: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0
    beta_to_option_book: float | None = None  # exposure translation held → option-book index; None → Σ β_i w_i


class EquityModelConfig(_Base):
    type: Literal["gbm", "merton", "coupled_stochvol", "bootstrap", "historical_replay"] = "merton"
    jump_lambda: NonNeg = 0.10
    jumps_enabled: bool = True
    vrp_ratio: Annotated[float, Field(ge=-1.0, lt=1.0)] = 0.15


class HistoricalReplayConfig(_Base):
    file: str | None = None
    start: str | None = None  # YYYY-MM-DD; the window runs for run.horizon_years from here
    columns: dict[str, str] = {}  # role -> column: index names, optional "IV", "RATES"
    held_column: str | None = None  # optional column for the held portfolio level (else from betas)
    iv_scale: Pos = 1.0  # multiplier applied to the IV column (anchors a long-dated vol series to today's quote)


class BootstrapConfig(_Base):
    file: str | None = None
    frequency: Literal["monthly", "daily"] = "monthly"
    columns: dict[str, str] = {}  # role -> column: index names, "IV", "RATES"
    series_type: Literal["price", "simple_return", "log_return"] = "price"
    expected_block_length: Pos = 12.0
    recentre_means: bool = True


# ----------------------------------------------------------------------------- implied vol


class ShortIVConfig(_Base):
    theta: Pos = 0.18
    kappa: Pos = 4.0
    eta: NonNeg = 1.0
    rho_equity: Annotated[float, Field(ge=-1.0, le=1.0)] = -0.7
    jump_add: float = 0.5
    floor: Pos = 0.08
    cap: Pos = 1.5
    iv0: Pos | None = None

    @model_validator(mode="after")
    def _bounds(self) -> ShortIVConfig:
        if not (self.floor <= self.theta <= self.cap):
            raise ValueError("implied_vol.short requires floor <= theta <= cap")
        return self


class TermStructureConfig(_Base):
    tenors: list[Pos]
    theta: list[Pos]
    beta: list[Prob]

    @model_validator(mode="after")
    def _grid(self) -> TermStructureConfig:
        _check_grid(self.tenors, {"theta": self.theta, "beta": self.beta}, "term_structure")
        return self


class ImpliedVolConfig(_Base):
    short: ShortIVConfig = ShortIVConfig()
    term_structure: TermStructureConfig
    skew_1y: Annotated[float, Field(ge=-2.0, le=2.0)] = -0.10
    skew_mode: Literal["sticky_moneyness", "sticky_strike"] = "sticky_moneyness"
    vol_floor: Pos = 0.05
    bid_ask_widening_per_iv: NonNeg = 0.0


# ----------------------------------------------------------------------------- rates / cash


class VasicekParams(_Base):
    r0: float = 0.04
    a: Pos = 0.3
    b: float = 0.035
    sigma: NonNeg = 0.01
    rho_equity: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.2
    term_premium: float = 0.0


class FlatCurve(_Base):
    tenors: list[Pos]
    zero: list[float]

    @model_validator(mode="after")
    def _grid(self) -> FlatCurve:
        _check_grid(self.tenors, {"zero": self.zero}, "flat_curve")
        return self


class ScenarioShift(_Base):
    type: Literal["none", "parallel", "steepener", "flattener", "buckets"] = "none"
    bp: float = 0.0
    buckets: dict[float, float] | None = None


class CashConfig(_Base):
    instrument: Literal["tbills", "mmf", "deposit"] = "tbills"
    spread: Annotated[float, Field(ge=-0.05, le=0.05)] = 0.001
    concentration_limit_pct: Prob | None = None


class RatesConfig(_Base):
    model: Literal["flat", "vasicek", "curve"] = "flat"
    usd: VasicekParams = VasicekParams()
    flat_curve: FlatCurve | None = None
    curves_file: str | None = None
    curve_type: Literal["zero", "par"] = "zero"
    scenario_shift: ScenarioShift = ScenarioShift()
    cash: CashConfig = CashConfig()

    @model_validator(mode="after")
    def _curve(self) -> RatesConfig:
        if self.model == "curve" and self.curves_file is None:
            raise ValueError("rates.model=curve requires rates.curves_file")
        return self


# ----------------------------------------------------------------------------- pricing


class DealerQuote(_Base):
    index: str
    tenor: Pos
    strike_pct_spot: Pos = 1.0
    bid_pct: NonNeg | None = None
    ask_pct: NonNeg | None = None
    bid_iv: Pos | None = None
    ask_iv: Pos | None = None
    quote_date: str | None = None
    spot: Pos | None = None
    rate: float | None = None
    dividend: float | None = None

    @model_validator(mode="after")
    def _has_quote(self) -> DealerQuote:
        if self.bid_pct is None and self.ask_pct is None and self.bid_iv is None and self.ask_iv is None:
            raise ValueError("dealer quote needs a premium % (bid/ask_pct) or an implied vol (bid/ask_iv)")
        return self


class ScenarioOverride(_Base):
    index: str | None = None
    strike_pct_spot: Pos = 1.0  # strike used to convert premium_pct into an implied vol
    month_from: Annotated[int, Field(ge=0)] = 0
    month_to: Annotated[int, Field(ge=0)] | None = None
    tenor: Pos | None = None
    iv: Pos | None = None
    premium_pct: Pos | None = None

    @model_validator(mode="after")
    def _one(self) -> ScenarioOverride:
        if (self.iv is None) == (self.premium_pct is None):
            raise ValueError("scenario override needs exactly one of iv or premium_pct")
        return self


class PricingConfig(_Base):
    source: Literal[
        "dealer_quotes", "vol_surface_file", "parametric", "scenario_overrides", "historical_iv_file"
    ] = "parametric"
    dealer_quotes: list[DealerQuote] = []
    vol_surface_file: str | None = None
    historical_iv_file: str | None = None
    scenario_overrides: list[ScenarioOverride] = []
    option_funding_spread: Annotated[float, Field(ge=-0.05, le=0.05)] = 0.0
    bid_ask_vol_pts_new: NonNeg = 0.005
    bid_ask_vol_pts_unwind: NonNeg = 0.0075
    stress_bid_ask_multiplier: Annotated[float, Field(ge=1.0)] = 3.0
    stress_iv_short_above: Pos = 0.35
    commission_bps_notional: NonNeg = 0.0
    arbitrage_check: Literal["enforce", "warn"] = "enforce"

    @model_validator(mode="after")
    def _source(self) -> PricingConfig:
        if self.source == "vol_surface_file" and self.vol_surface_file is None:
            raise ValueError("pricing.source=vol_surface_file requires pricing.vol_surface_file")
        if self.source == "dealer_quotes" and not self.dealer_quotes:
            raise ValueError("pricing.source=dealer_quotes requires at least one entry in pricing.dealer_quotes")
        if self.source == "historical_iv_file" and self.historical_iv_file is None:
            raise ValueError("pricing.source=historical_iv_file requires pricing.historical_iv_file")
        if self.source == "scenario_overrides" and not self.scenario_overrides:
            raise ValueError("pricing.source=scenario_overrides requires pricing.scenario_overrides")
        return self


# ----------------------------------------------------------------------------- options / exposure


class CounterpartyConfig(_Base):
    enabled: bool = False
    default_intensity: NonNeg = 0.005
    recovery: Prob = 0.4


SizingMode = Literal[
    "notional_match",
    "delta_match",
    "beta_adjusted_delta_match",
    "premium_budget",
    "cash_reserve_target",
    "delta_fraction_of_A",
]


class OptionsConfig(_Base):
    tenor_years: Pos = 5.0
    strike_mode: Literal["atm_spot", "atm_forward", "pct_spot", "pct_forward", "delta_target"] = "atm_spot"
    strike_param: float | None = None
    delta_definition: Literal["bsm", "smile"] = "bsm"
    option_type: Literal["vanilla"] = "vanilla"
    exercise: Literal["european"] = "european"
    sizing_mode: SizingMode = "notional_match"
    sizing_param: float | None = None
    sizing_basis: Literal["usd", "pct_nav"] = "usd"
    ladder_mode: Literal["bullet", "monthly_buildup_hold_to_expiry", "rolling_ladder", "fixed_notional_schedule"] = "rolling_ladder"
    purchase_frequency: Literal["monthly", "weekly"] = "monthly"  # fixed_notional_schedule: how often a tranche is bought
    notional_per_purchase: Pos | None = None  # fixed_notional_schedule: USD notional of each purchase (or use notional_pct_nav)
    notional_pct_nav: Pos | None = None  # fixed_notional_schedule: each purchase = this fraction of current NAV (overrides the USD amount)
    target_total_notional: Pos | None = None  # fixed_notional_schedule: cap on the active book; None = keep adding without limit
    target_basis: Literal["purchase_notional", "current_notional", "delta"] = "purchase_notional"  # what the target is measured in
    target_pct_nav: Pos | None = None  # target = this fraction of B's current NAV (overrides target_total_notional)
    buildup_months: Annotated[int, Field(ge=1, le=120)] = 12
    hold_months_before_roll: Annotated[int, Field(ge=1, le=600)] = 12
    bullet_roll_residual_years: NonNeg = 1.0
    buildup_holding: Literal["cash", "spot_selldown"] = "cash"
    resize_on_roll: Literal["keep_notional", "target_pct_nav", "target_dollar_delta"] = "target_pct_nav"
    iv_pause_threshold: Pos | None = None
    extra_slots: Annotated[int, Field(ge=0, le=600)] = 24
    counterparty: CounterpartyConfig = CounterpartyConfig()

    @model_validator(mode="after")
    def _params(self) -> OptionsConfig:
        if self.strike_mode in ("pct_spot", "pct_forward"):
            if self.strike_param is None or self.strike_param <= 0:
                raise ValueError(f"strike_mode={self.strike_mode} requires a positive strike_param (e.g. 1.10)")
        if self.strike_mode == "delta_target":
            if self.strike_param is None or not (0.0 < self.strike_param < 1.0):
                raise ValueError("strike_mode=delta_target requires strike_param in (0, 1) (e.g. 0.50)")
        if self.sizing_mode in ("premium_budget", "cash_reserve_target", "delta_fraction_of_A"):
            if self.sizing_param is None or self.sizing_param < 0:
                raise ValueError(f"sizing_mode={self.sizing_mode} requires a non-negative sizing_param")
        if self.hold_months_before_roll > self.tenor_years * 12:
            raise ValueError("hold_months_before_roll cannot exceed the option tenor")
        if self.ladder_mode == "fixed_notional_schedule" and self.notional_per_purchase is None and self.notional_pct_nav is None:
            raise ValueError("ladder_mode=fixed_notional_schedule requires notional_per_purchase (USD) or notional_pct_nav")
        return self


class ExposureTarget(_Base):
    type: Literal["pct_of_A_exposure", "usd", "pct_nav"] = "pct_of_A_exposure"
    value: NonNeg = 1.0


class FuturesOverlayConfig(_Base):
    enabled: bool = False
    initial_margin_pct: Prob = 0.10


class ExposureConfig(_Base):
    policy: Literal["static", "rebalance_bands", "restrike_on_move", "spot_topup", "futures_overlay"] = "static"
    target: ExposureTarget = ExposureTarget()
    band_pp: Prob = 0.10
    moneyness_band: tuple[Pos, Pos] = (0.75, 1.40)
    futures_overlay: FuturesOverlayConfig = FuturesOverlayConfig()
    comparison_lenses: list[Literal["equal_capital", "equal_delta", "equal_vol", "equal_return"]] = [
        "equal_capital",
        "equal_delta",
        "equal_vol",
    ]

    @model_validator(mode="after")
    def _band(self) -> ExposureConfig:
        if self.moneyness_band[0] >= self.moneyness_band[1]:
            raise ValueError("moneyness_band must be [low, high] with low < high")
        if self.policy == "futures_overlay" and not self.futures_overlay.enabled:
            raise ValueError("exposure.policy=futures_overlay requires exposure.futures_overlay.enabled=true")
        return self


# ----------------------------------------------------------------------------- dry powder


class DryPowderTier(_Base):
    index_drawdown: Annotated[float, Field(ge=-1.0, le=0.0)] | None = None
    iv_short_above: Pos | None = None
    deploy_pct_of_deployable: Prob

    @model_validator(mode="after")
    def _trigger(self) -> DryPowderTier:
        if self.index_drawdown is None and self.iv_short_above is None:
            raise ValueError("dry_powder tier needs index_drawdown and/or iv_short_above")
        return self


class LiquidityReserveConfig(_Base):
    floor_pct_nav: Prob = 0.03
    capital_call_months: Annotated[int, Field(ge=0)] = 12
    capital_call_stress_multiplier: Annotated[float, Field(ge=1.0)] = 1.5


class ExitRuleConfig(_Base):
    enabled: bool = False
    convert_over_months: Annotated[int, Field(ge=1)] = 12


class DryPowderConfig(_Base):
    enabled: bool = True
    trigger_index: str | None = None  # index whose drawdown fires the tiers; None → SPX if configured, else the option-book index
    tiers: list[DryPowderTier] = []
    instrument: Literal["spot_index", "call_tranches"] = "spot_index"
    liquidity_reserve: LiquidityReserveConfig = LiquidityReserveConfig()
    exit_rule: ExitRuleConfig = ExitRuleConfig()

    @field_validator("tiers")
    @classmethod
    def _sorted(cls, v: list[DryPowderTier]) -> list[DryPowderTier]:
        dd = [t.index_drawdown for t in v if t.index_drawdown is not None]
        if dd != sorted(dd, reverse=True):
            raise ValueError("dry_powder tiers must be ordered from the mildest to the deepest drawdown")
        return v


# ----------------------------------------------------------------------------- illiquids


class HFLiquidity(_Base):
    frequency_months: Annotated[int, Field(ge=1)] = 3
    notice_days: Annotated[int, Field(ge=0)] = 90
    gate: Prob = 0.25
    lockup_months: Annotated[int, Field(ge=0)] = 0


class TACrisis(_Base):
    drawdown_trigger: Annotated[float, Field(ge=-1.0, le=0.0)] = -0.25
    contribution_multiplier: NonNeg = 1.5
    distribution_multiplier: NonNeg = 0.5


class TAModel(_Base):
    rc: Prob = 0.25
    b: Pos = 2.5
    life_years: Pos = 10.0
    age_years: NonNeg = 0.0
    crisis: TACrisis | None = None


class IlliquidConfig(_Base):
    name: str = Field(min_length=1)
    weight: Prob
    type: Literal["open_ended", "closed_end"]
    expected_return: Annotated[float, Field(gt=-0.9, lt=1.0)]
    return_type: ReturnType = "arithmetic"
    vol: Annotated[float, Field(ge=0.0, le=2.0)]
    rho_equity: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.5
    jump_beta: float = 0.0
    smoothing_phi: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    reporting_months: Annotated[int, Field(ge=1)] = 3
    liquidity: HFLiquidity | None = None
    unfunded_commitments: NonNeg = 0.0
    ta_model: TAModel | None = None
    new_commitment_pacing_pa: NonNeg = 0.0

    @model_validator(mode="after")
    def _type(self) -> IlliquidConfig:
        if self.type == "closed_end" and self.ta_model is None:
            raise ValueError(f"illiquid {self.name}: closed_end requires ta_model")
        if self.type == "open_ended" and self.unfunded_commitments > 0:
            raise ValueError(f"illiquid {self.name}: open_ended cannot have unfunded_commitments")
        return self


class IlliquidsMarkingConfig(_Base):
    margin_uses: Literal["reported", "true"] = "reported"
    risk_metrics_use: Literal["true_marks", "reported"] = "true_marks"


# ----------------------------------------------------------------------------- correlation / misc


class CorrelationConfig(_Base):
    names: list[str]
    matrix: list[list[float]]

    @model_validator(mode="after")
    def _valid(self) -> CorrelationConfig:
        n = len(self.names)
        if len(set(self.names)) != n:
            raise ValueError("correlation.names must be unique")
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (n, n):
            raise ValueError(f"correlation.matrix must be {n}x{n} to match names")
        if not np.allclose(m, m.T, atol=1e-12):
            raise ValueError("correlation.matrix must be symmetric")
        if not np.allclose(np.diag(m), 1.0, atol=1e-12):
            raise ValueError("correlation.matrix must have a unit diagonal")
        if np.any(np.abs(m) > 1.0 + 1e-12):
            raise ValueError("correlation.matrix entries must lie in [-1, 1]")
        return self


class StrategyBalance(_Base):
    """Inception balance sheet entered directly (USD). NAV = equities + cash + illiquids − loan."""

    equities: NonNeg = 0.0  # held SPX book at t0 (A, C; B may keep some)
    loan: NonNeg = 0.0
    cash: NonNeg = 0.0


class CustomInceptionConfig(_Base):
    """Optional override of the portfolio/leverage-derived balance sheets (fully customisable strategies)."""

    enabled: bool = False
    illiquids: NonNeg = 0.0  # identical in every strategy (cannot be traded)
    A: StrategyBalance = StrategyBalance()
    B: StrategyBalance = StrategyBalance()
    C: StrategyBalance = StrategyBalance()
    reference_equity_exposure: Pos | None = None  # "A's equity exposure" used by B's sizing modes; None → A.equities


class CostsConfig(_Base):
    equity_bps: NonNeg = 5.0
    equity_bps_stressed: NonNeg = 40.0
    stamp_duty_bps: NonNeg = 0.0


class SpendingConfig(_Base):
    pct_nav_pa: Prob = 0.0
    fixed_usd_pa: NonNeg = 0.0


class AnalyticsConfig(_Base):
    crra_gamma: NonNeg = 3.0
    var_levels: list[Prob] = [0.95, 0.99]
    percentiles: list[Annotated[float, Field(ge=0, le=100)]] = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# ----------------------------------------------------------------------------- root


def _usd_guard(obj: Any, path: str = "") -> None:
    """Recursively reject any currency field ≠ USD, FX block or quanto reference (SPEC §3)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            kl = key.lower()
            p = f"{path}.{key}" if path else key
            if "currency" in kl and (not isinstance(v, str) or v.upper() != "USD"):
                raise ValueError(f"{p}: only USD is supported (got {v!r}); FX / non-USD views are out of scope")
            if kl == "fx" or kl.startswith("fx_") or kl.endswith("_fx") or "quanto" in kl:
                raise ValueError(f"{p}: FX / quanto inputs are out of scope — the model is USD only")
            if isinstance(v, str) and v.strip().lower() == "quanto":
                raise ValueError(f"{p}: quanto options are out of scope — the model is USD only")
            _usd_guard(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _usd_guard(v, f"{path}[{i}]")


class SimConfig(_Base):
    run: RunConfig = RunConfig()
    portfolio: PortfolioConfig
    leverage: LeverageConfig
    held_equity_portfolio: HeldPortfolioConfig
    equity_indices: list[EquityIndexConfig]
    equity_model: EquityModelConfig = EquityModelConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    historical: HistoricalReplayConfig = HistoricalReplayConfig()
    implied_vol: ImpliedVolConfig
    rates: RatesConfig = RatesConfig()
    loan_terms: LoanTermsConfig
    pricing: PricingConfig = PricingConfig()
    options: OptionsConfig = OptionsConfig()
    exposure: ExposureConfig = ExposureConfig()
    dry_powder: DryPowderConfig = DryPowderConfig()
    illiquids: list[IlliquidConfig]
    illiquids_marking: IlliquidsMarkingConfig = IlliquidsMarkingConfig()
    custom_inception: CustomInceptionConfig = CustomInceptionConfig()
    correlation: CorrelationConfig | None = None
    costs: CostsConfig = CostsConfig()
    spending: SpendingConfig = SpendingConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()

    @model_validator(mode="before")
    @classmethod
    def _guard(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _usd_guard(data)
        return data

    @model_validator(mode="after")
    def _cross(self) -> SimConfig:
        if not self.equity_indices:
            raise ValueError("at least one equity index is required")
        names = [i.name for i in self.equity_indices]
        if len(set(names)) != len(names):
            raise ValueError("equity index names must be unique")
        if not math.isclose(sum(i.option_book_weight for i in self.equity_indices), 1.0, abs_tol=1e-9):
            raise ValueError("equity_indices[].option_book_weight must sum to 1.0")
        if not math.isclose(sum(i.weight_in_equity_sleeve for i in self.equity_indices), 1.0, abs_tol=1e-9):
            raise ValueError("equity_indices[].weight_in_equity_sleeve must sum to 1.0")
        if self.illiquids and not math.isclose(sum(i.weight for i in self.illiquids), 1.0, abs_tol=1e-9):
            raise ValueError("illiquids[].weight must sum to 1.0")
        ill_names = [i.name for i in self.illiquids]
        if len(set(ill_names)) != len(ill_names) or set(ill_names) & set(names):
            raise ValueError("illiquid names must be unique and distinct from index names")
        for k in self.held_equity_portfolio.betas:
            if k not in names:
                raise ValueError(f"held_equity_portfolio.betas references unknown index {k!r}")
        for idx in names:
            if idx not in self.held_equity_portfolio.betas:
                raise ValueError(f"held_equity_portfolio.betas missing index {idx!r}")
        expected = [*names, *ill_names, "IV", "RATES"]
        # the held-portfolio idiosyncratic shock is independent by construction (not in the matrix)
        if self.correlation is not None:
            missing = [n for n in expected if n not in self.correlation.names]
            extra = [n for n in self.correlation.names if n not in expected]
            if missing or extra:
                raise ValueError(
                    f"correlation.names must be exactly {expected} (missing={missing}, unknown={extra})"
                )
        for q in self.pricing.dealer_quotes:
            if q.index not in names:
                raise ValueError(f"dealer quote references unknown index {q.index!r}")
        for o in self.pricing.scenario_overrides:
            if o.index is not None and o.index not in names:
                raise ValueError(f"scenario override references unknown index {o.index!r}")
        if self.dry_powder.trigger_index is not None and self.dry_powder.trigger_index not in names:
            raise ValueError(f"dry_powder.trigger_index {self.dry_powder.trigger_index!r} is not a configured index")
        max_t = max(self.implied_vol.term_structure.tenors)
        if self.options.tenor_years > max_t + 1e-12:
            raise ValueError(
                f"options.tenor_years={self.options.tenor_years} exceeds the implied-vol term-structure grid (max {max_t})"
            )
        if self.leverage.ratio_of_nav * self.portfolio.nav0 > self.loan_terms.facility_limit + 1e-6:
            raise ValueError("initial loan exceeds loan_terms.facility_limit")
        if self.equity_model.type == "bootstrap" and self.bootstrap.file is None:
            raise ValueError("equity_model.type=bootstrap requires bootstrap.file")
        if self.equity_model.type == "historical_replay" and (self.historical.file is None or self.historical.start is None):
            raise ValueError("equity_model.type=historical_replay requires historical.file and historical.start")
        if self.equity_model.type == "historical_replay" and self.run.n_paths != 1:
            raise ValueError("historical_replay is a single deterministic path: set run.n_paths: 1")
        if self.options.ladder_mode == "fixed_notional_schedule" and self.options.purchase_frequency == "weekly" and self.run.dt == "monthly":
            raise ValueError("weekly option purchases need run.dt weekly or daily")
        if self.custom_inception.enabled:
            ci = self.custom_inception
            if ci.A.loan > self.loan_terms.facility_limit + 1e-6:
                raise ValueError("custom_inception.A.loan exceeds loan_terms.facility_limit")
            for nm, b in (("A", ci.A), ("B", ci.B), ("C", ci.C)):
                if b.equities + b.cash + ci.illiquids - b.loan <= 0:
                    raise ValueError(f"custom_inception.{nm}: NAV must be positive")
        if self.exposure.policy == "futures_overlay" and self.exposure.futures_overlay.enabled:
            pass  # allowed but flagged in the UI (SPEC §5.7.8)
        return self

    # convenience -----------------------------------------------------------------------------
    @property
    def index_names(self) -> list[str]:
        return [i.name for i in self.equity_indices]

    @property
    def illiquid_names(self) -> list[str]:
        return [i.name for i in self.illiquids]


def _check_grid(tenors: list[float], cols: dict[str, list[float]], label: str) -> None:
    if any(len(c) != len(tenors) for c in cols.values()):
        raise ValueError(f"{label}: tenors and value lists must have the same length")
    if any(t2 <= t1 for t1, t2 in itertools.pairwise(tenors)):
        raise ValueError(f"{label}: tenors must be strictly increasing")


def load_config(path: str | Path) -> SimConfig:
    """Load and validate a YAML configuration file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return SimConfig.model_validate(raw)
